# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import clip
import os
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F_t
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, \
    UniPCMultistepScheduler, StableDiffusionXLControlNetPipeline, AutoencoderKL, \
    FluxControlNetModel, FluxMultiControlNetModel, FluxControlNetPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, \
    FlowMatchEulerDiscreteScheduler, StableUnCLIPImg2ImgPipeline
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from huggingface_hub import login

from gloss.model.loss import canonicalize_training_objective, FLOW_MATCHING_OBJECTIVE
from gloss.utils.paths import SD21_UNCLIP_MODEL_ID, get_base_model_dir

@dataclass
class PretrainedModelConfig:
    name: str = "madebyollin/sdxl-vae-fp16-fix"
    dtype: str = "float32"

@dataclass
class VaeConfig(PretrainedModelConfig):
    name: str = "madebyollin/sdxl-vae-fp16-fix"
    dtype: str = "float32"

@dataclass
class ClipConfig(PretrainedModelConfig):
    name: str = "ViT-L/14"
    dtype: str = "float16"


def parse_dtype(input_str):
    supported = {
        'float32': torch.float32,
        'float16': torch.float16}
    if input_str in supported:
        return supported[input_str]
    raise ValueError(f'Unsupported dtype string: {input_str}')

class SimpleScaler(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def __call__(self, x):
        return x * self.scale

def diffusers_model_from_config(cls, config: PretrainedModelConfig):
    model = cls.from_pretrained(config.name, torch_dtype=parse_dtype(config.dtype))
    return model


def clip_model_from_config(config: ClipConfig):
    return clip.load(config.name).to(parse_dtype(config.dtype))


def load_pretrained_diffusion_controlnet_pipeline(model_name:str, controls:list[str], **kwargs):
    """loading pretrained diffusion model with controlnet,
       if using sdxl, need to add vae;
       Use the faster UniPCMultistepScheduler and 
       enable model offloading to speed up inference and reduce memory usage.

    Args:
        model_name (str): _description_
        controls (list[str]): _description_

    Returns:
        _type_: _description_
    """
    if 'xl' in model_name:
        controlnets = []
        for control in controls:
            assert 'xl' in control
            controlnets.append(ControlNetModel.from_pretrained(
                    control, torch_dtype=torch.float16, use_safetensors=True, cache_dir=".cache"))
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_name,
            controlnet=controlnets,
            vae=vae,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
            cache_dir=".cache")
    else:
        controlnets = []
        for control in controls:
            controlnets.append(ControlNetModel.from_pretrained(
                    control, torch_dtype=torch.float16, cache_dir=".cache"))
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            model_name, 
            controlnet=controlnets,
            torch_dtype=torch.float16, 
            use_safetensors=True,
            cache_dir=".cache")
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()
    return pipeline


SD_21_BASE = "stabilityai/stable-diffusion-2-1"
SD_21_DEPTH_CONTROL = "thibaud/controlnet-sd21-depth-diffusers"
SD_21_CANNY_CONTROL = "thibaud/controlnet-sd21-canny-diffusers"
SD_21_NORMAL_CONTROL = "thibaud/controlnet-sd21-normalbae-diffusers"
FLUX_BASE = "black-forest-labs/FLUX.1-dev"
FLUX_CONTROL = 'InstantX/FLUX.1-dev-Controlnet-Union'
DEFAULT_SD21_UNCLIP_DIR = SD21_UNCLIP_MODEL_ID  # override with GLOSS_MODEL_DIR


def load_sd21_depth_canny_control(config):
    SD_CONTROLS = []
    for control in config.controls:
        if control  == 'depth':
            SD_CONTROLS .append(SD_21_DEPTH_CONTROL)
        if control  == 'midas':
            SD_CONTROLS .append(SD_21_DEPTH_CONTROL)
        if control == 'canny':
            SD_CONTROLS .append(SD_21_CANNY_CONTROL)
        if control == 'normal':
            SD_CONTROLS .append(SD_21_NORMAL_CONTROL)
    pipe = load_pretrained_diffusion_controlnet_pipeline(SD_21_BASE, SD_CONTROLS)
    pipeline_kwargs = {'controlnet_conditioning_scale': config.strengths, 'num_inference_steps': 50, 'guidance_scale': 3.5}
    return pipe, pipeline_kwargs


def load_flux_control(config):
    controlnet_union = FluxControlNetModel.from_pretrained(FLUX_CONTROL, torch_dtype=torch.bfloat16)
    controlnet = FluxMultiControlNetModel([controlnet_union])
    pipe = FluxControlNetPipeline.from_pretrained(FLUX_BASE, controlnet=controlnet, torch_dtype=torch.bfloat16).to("cuda")
    pipeline_kwargs = {'controlnet_conditioning_scale': config.strengths, 'num_inference_steps': 50, 'guidance_scale': 3.5}
    return pipe, pipeline_kwargs


def load_pipeline_components(in_channels:int, model_card:str="stabilityai/stable-diffusion-2-1-unclip",
                             train_vae=False, from_scratch=True, vae_in_channel=10,
                             vae_latent_channels=4, objective="diffusion", flow_shift=1.0)->dict:
    # """load all pipeline component to set up the pipeline, all component frozen except the unet
    # The pipeline includes vae that encodes conditional images to latent space, and decodes the output
    # noise scheduler that does v prediction
    # feature extractor to transform input image to canonical imagenet mean std and size
    # image encoder that encodes the image prompt
    # image noise scheduler that add noise to the image prompt (not used for first experiment)

    # Args:
    #     model_card (str, optional): _description_. Defaults to "stabilityai/stable-diffusion-2-1-unclip".

    # Returns:
    #     dict: _description_
    # """
    # GLOSS_MODEL_DIR wins, else the release-hosted Hub copy (the argument is kept for API compatibility).
    model_card = get_base_model_dir()
    if train_vae:
        vae_config = AutoencoderKL.load_config(model_card, subfolder="vae")
        vae_config["in_channels"] = vae_in_channel
        vae_config["latent_channels"] = vae_latent_channels
        train_vae = AutoencoderKL.from_config(vae_config)
    vae = AutoencoderKL.from_pretrained(model_card, subfolder="vae", use_safetensors=True)
    text_model = CLIPTextModel.from_pretrained(model_card, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(model_card, subfolder="tokenizer")
    if from_scratch:
        unet_config = UNet2DConditionModel.load_config(model_card, subfolder="unet")
        unet_config["in_channels"] = in_channels
        unet = UNet2DConditionModel.from_config(unet_config)
    else:
        unet = UNet2DConditionModel.from_pretrained(model_card, subfolder="unet")
        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        new_conv_in = nn.Conv2d(
            in_channels, 320, kernel_size=conv_in_kernel, padding=conv_in_padding
        )
        with torch.no_grad():
            new_conv_in.weight.zero_()  # zero init
            new_conv_in.weight[:, :4, :, :] = unet.conv_in.weight  # copy pretrained
            new_conv_in.bias[:] = unet.conv_in.bias
        unet.conv_in = new_conv_in

    objective = canonicalize_training_objective(objective)
    base_scheduler = DDPMScheduler.from_pretrained(model_card, subfolder="scheduler")
    if objective == FLOW_MATCHING_OBJECTIVE:
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=base_scheduler.config.num_train_timesteps,
            shift=flow_shift,
        )
    else:
        scheduler = base_scheduler
    feature_extractor = CLIPImageProcessor.from_pretrained(model_card, subfolder="feature_extractor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_card, subfolder="image_encoder")
    image_normalizer = StableUnCLIPImageNormalizer(embedding_dim=1024) #TODO: get to from the hub json file 
    image_noising_scheduler = DDPMScheduler.from_pretrained(model_card, subfolder="image_noising_scheduler")
    pipe_components = {"vae":vae, "text_encoder": text_model, "tokenizer": tokenizer, "unet":unet, "scheduler":scheduler, 
                       "image_encoder": image_encoder, "image_noising_scheduler": image_noising_scheduler, "image_normalizer": image_normalizer,
                       "feature_extractor": feature_extractor}
    if train_vae:
        pipe_components["train_vae"] = train_vae
    return pipe_components


def load_pipe_from_components(pipe_components):
    pipe = StableUnCLIPImg2ImgPipeline(**pipe_components)
    return pipe


def pipe_generate(pipe:StableUnCLIPImg2ImgPipeline, scheduler, prompt_embeds, image_embeds, conds, uncond=None, num_inference_steps=20, cfg_scale=1.0):
    with torch.no_grad():
        conds = conds
        scheduler.set_timesteps(num_inference_steps, device=conds.device)
        timesteps = scheduler.timesteps
        latents = torch.randn((conds[:, :4, :, :].shape), device=conds.device).to(conds.dtype) # TODO: set channel num somewhere
        for i, t in enumerate(pipe.progress_bar(timesteps)):
            latent_model_input = torch.cat([latents, conds], dim=1)
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                class_labels=image_embeds,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]
            if cfg_scale > 1.0:
                assert uncond is not None
                latent_model_input = torch.cat([latents, uncond], dim=1)
                uncond_noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    class_labels=image_embeds,
                    cross_attention_kwargs=None,
                    return_dict=False,
                )[0]
                noise_pred = uncond_noise_pred + cfg_scale * (noise_pred - uncond_noise_pred)
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor , return_dict=False)[0]
        images = pipe.image_processor.postprocess(images, 'pt') #"pil"
        return images


def get_prompt_embedding(pipe, prompts, device):
    with torch.no_grad():
        prompt_embeds = pipe.encode_prompt(prompts, device, 1, False)[0]
    return prompt_embeds


def get_view_embedding(pipe, view, device):
    with torch.no_grad():
        # expects image range [0, 1]
        _image_prompt = F.interpolate(view, size=(224, 224), mode='bicubic', align_corners=True, antialias=True)
        _image_prompt = F_t.normalize(_image_prompt, pipe.feature_extractor.image_mean, pipe.feature_extractor.image_std)
        view_embeds = pipe._encode_image(_image_prompt, device, 1, 1, False, 0, None, None)
    return view_embeds


def _parse_global_step(value):
    if value is None:
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _looks_like_safetensors(ckpt_fp):
    try:
        with open(ckpt_fp, "rb") as f:
            head = f.read(9)
    except OSError:
        return False
    if len(head) < 9:
        return False
    header_len = int.from_bytes(head[:8], "little", signed=False)
    file_size = os.path.getsize(ckpt_fp)
    return 2 <= header_len <= file_size - 8 and head[8:9] == b"{"


def load_checkpoint_state_dict(ckpt_fp, *, map_location="cpu", prefer_ema=True):
    if ckpt_fp.lower().endswith(".safetensors") or _looks_like_safetensors(ckpt_fp):
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError(
                f"Loading safetensors checkpoint {ckpt_fp} requires the safetensors package"
            ) from exc

        device = map_location
        if isinstance(device, torch.device):
            device = device.type
        with safe_open(ckpt_fp, framework="pt", device=device) as f:
            metadata = f.metadata() or {}
            state_dict = {key: f.get_tensor(key) for key in f.keys()}
        return state_dict, _parse_global_step(metadata.get("global_step"))

    loaded = torch.load(ckpt_fp, map_location=map_location)
    if isinstance(loaded, dict):
        if prefer_ema and "model_ema" in loaded:
            return loaded["model_ema"]["avg"], _parse_global_step(loaded.get("global_step"))
        if "model" in loaded:
            return loaded["model"], _parse_global_step(loaded.get("global_step"))
        if "model_ema" in loaded:
            return loaded["model_ema"]["avg"], _parse_global_step(loaded.get("global_step"))
    return loaded, -1


def load_ckpt(pipe_components, ckpt_fp):
    loaded_state_dict, _ = load_checkpoint_state_dict(ckpt_fp, map_location="cpu")

    if not isinstance(loaded_state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format in {ckpt_fp}: expected a state-dict-like mapping")

    target_keys = set(pipe_components["unet"].state_dict().keys())
    candidate_state_dicts = [loaded_state_dict]

    stripped_state_dict = {}
    for k, v in loaded_state_dict.items():
        if "." in k:
            stripped_state_dict[".".join(k.split(".")[1:])] = v
        else:
            stripped_state_dict[k] = v
    if stripped_state_dict != loaded_state_dict:
        candidate_state_dicts.append(stripped_state_dict)

    def _score(sd):
        return sum(1 for k in sd.keys() if k in target_keys)

    state_dict = max(candidate_state_dicts, key=_score)
    if _score(state_dict) == 0:
        sample_keys = list(loaded_state_dict.keys())[:5]
        raise RuntimeError(
            f"Could not match checkpoint keys in {ckpt_fp} to the UNet. Sample keys: {sample_keys}"
        )

    pipe_components['unet'].load_state_dict(state_dict)
    pipe = load_pipe_from_components(pipe_components)
    return pipe
