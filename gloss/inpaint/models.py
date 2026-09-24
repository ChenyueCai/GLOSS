# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import torch, torchvision
from diffusers import UNet2DModel, DDIMScheduler, DDIMPipeline

import kaolin
import gloss
from gloss.model.attention import replace_attention_processors, SamplewiseAttnProcessor2_0
from gloss.model.standard import load_pipeline_components, load_ckpt, pipe_generate, get_prompt_embedding, get_view_embedding
from gloss.model.diffusion_single_step import SingleStepLDM
from gloss.model import image_diffusion
from gloss.model import controlnet
from gloss.inpaint.color_correction import reinhard_color_transfer
from gloss.logging.logging import log_tensor, default_log_setup


class TextureInpaintBaseModel(object):
    def __init__(self, num_in_channels, in_channels, ckpt_path, use_fp16=False):
        self.device = torch.device("cuda")
        self.use_fp16 = use_fp16
        self.ckpt_path = ckpt_path
        self.num_in_channels = num_in_channels
        self.in_channels = in_channels
        self.prompt_embedding = None
        self.view_embedding = None

    def set_embedding(self, prompt, view):
        pass

    def set_attention_proc(self, attnproc, **kwargs):
        pass

    def set_input(self, data):
        # assume data is a list of dictionary
        # composite albedo with mask
        input = {}
        for d in data:
            # masks should be 0-1
            d["inpaint_mask"] = d["inpaint_mask"] / 2 + 0.5 if d["inpaint_mask"].min() < 0 else d["inpaint_mask"]
            d["albedo_alpha"] = d["albedo_alpha"] / 2 + 0.5 if d["albedo_alpha"].min() < 0 else d["albedo_alpha"]
            d["inpaint_mask"] = (d["inpaint_mask"] > 0.99).float()
            d["albedo_alpha"] = (d["albedo_alpha"] > 0.99).float()
            background = 1.0 - (d["background_alpha"] / 2 + 0.5)
            d["albedo"] = (
                d["albedo"] * d["inpaint_mask"] * d["albedo_alpha"] + torch.ones_like(d["albedo"]) * background * -1.0
            )
            d["inpaint_mask"] = d["inpaint_mask"] * d["albedo_alpha"] + background
            for ch in self.in_channels:
                if ch not in input:
                    input[ch] = []
                input[ch].append(d[ch])
        for k, v in input.items():
            input[k] = torch.cat(v, dim=0)
        num_samples = len(data)
        if self.prompt_embedding is not None:
            input["prompt_embs"] = self.prompt_embedding.repeat(num_samples, 1, 1)  # TODO
        if self.view_embedding is not None:
            input["view_embs"] = self.view_embedding.repeat(num_samples, 1)
        return input

    def inpaint(self, data, log=False, log_dir=None, masked=False, color_correct=False):
        pass


class TextureInpaintStandardModel(TextureInpaintBaseModel):
    def __init__(self, num_in_channels, in_channels, ckpt_path, use_fp16=False):
        super().__init__(num_in_channels, in_channels, ckpt_path, use_fp16)
        self.pipe = None
        self.pipeline_component = load_pipeline_components(num_in_channels)
        self.noise_scheduler = self.pipeline_component["scheduler"]
        self.num_inference_step = 20

    def set_embedding(self, prompt, view):
        null_prompt = self.pipe.encode_prompt("", self.device, 1, False)[0]
        self.prompt_embedding = null_prompt
        # p_emb = get_prompt_embedding(self.pipe, prompt, self.device)
        v_emb = get_view_embedding(self.pipe, view, self.device)
        # self.prompt_embedding = p_emb
        self.view_embedding = v_emb

    def set_pipe(self):
        self.pipe = load_ckpt(self.pipeline_component, self.ckpt_path).to(self.device)
        if self.use_fp16:
            self.pipe = self.pipe.to(torch.float16)

    def set_attention_proc(self, attnproc, **kwargs):
        replace_attention_processors(
            self.pipeline_component["unet"], attnproc, store_attention_weights=False, **kwargs
        )  # todo: pass down kwargs
        self.set_pipe()

    def encode_input(self, input):
        encoded_inputs = []
        # latent_mask = -torch.nn.functional.max_pool2d(-input["inpaint_mask"].permute(0, 3, 1, 2), 8)
        for k, v in input.items():
            if k in self.in_channels:
                v = v.permute(0, 3, 1, 2)
                if v.shape[1] == 3:
                    with torch.no_grad():
                        _encoded = (
                                self.pipe.vae.encode(v, return_dict=False)[0].sample()
                                * self.pipe.vae.config.scaling_factor
                        )
                else:
                    _encoded = -torch.nn.functional.max_pool2d(-v, 8)
                # if k == "albedo":
                #     _encoded *= latent_mask
                encoded_inputs.append(_encoded)
        return torch.cat(encoded_inputs, dim=1)

    @torch.no_grad()
    def background_latent(self, res):
        color_image = torch.ones((1, 3, res, res), device=self.device) * -1

        if self.use_fp16:
            color_image = color_image.to(torch.float16)
        latents = self.pipe.vae.encode(color_image, return_dict=False)[0].sample() \
                  * self.pipe.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def process_input(self, data):
        raw_input = self.set_input(data)
        if self.use_fp16:
            raw_input = {k: v.to(torch.float16) for k, v in raw_input.items()}
        cond = self.encode_input(raw_input)
        return cond

    @torch.no_grad()
    def step(self, cond, latents, t):
        num_samples = cond.shape[0]

        prompt_embed = self.prompt_embedding.repeat(num_samples, 1, 1)  # TODO
        image_embed = self.view_embedding.repeat(num_samples, 1)

        if self.use_fp16:
            cond = cond.to(torch.float16)
            prompt_embed = prompt_embed.to(torch.float16)
            image_embed = image_embed.to(torch.float16)
            latents = latents.to(torch.float16)

        latent_model_input = torch.cat([latents, cond], dim=1)
        noise_pred = self.pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embed,
            class_labels=image_embed,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]
        return noise_pred

    def inpaint(self, data, log=False, log_dir=None, masked=False, color_correct=False):
        self.raw_input = self.set_input(data)
        if self.use_fp16:
            self.raw_input = {k: v.to(torch.float16) for k, v in self.raw_input.items()}
        if log:
            for ch in self.in_channels:
                torchvision.utils.save_image(
                    (self.raw_input[ch].permute(0, 3, 1, 2) / 2) + 0.5,
                    os.path.join(log_dir, f"log-{ch}.png"),
                )
        cond = self.encode_input(self.raw_input)
        # prompt embed, image embed
        prompt_embed, image_embed = self.raw_input["prompt_embs"], self.raw_input["view_embs"]
        pipe_rendering = pipe_generate(
            self.pipe,
            self.noise_scheduler,
            prompt_embed,
            image_embed,
            cond,
            self.num_inference_step,
        )
        if color_correct:
            if log:
                torchvision.utils.save_image(pipe_rendering, os.path.join(log_dir, f"log-output-raw.png"))

            alpha = (self.raw_input["inpaint_mask"] == 1).permute(0, 3, 1, 2)
            input_rgb = (self.raw_input["albedo"].permute(0, 3, 1, 2) / 2 + 0.5)
            for i in range(pipe_rendering.shape[0]):
                bg_alpha = (data[i]["background_alpha"] == -1)
                alpha[i, bg_alpha[..., 0]] = 0
                if alpha[i].sum() > 0:
                    result = reinhard_color_transfer(input_rgb[i], pipe_rendering[i], alpha[i])
                    pipe_rendering[i] = result
        if masked:
            alpha = (self.raw_input["inpaint_mask"] == 1).permute(0, 3, 1, 2)
            pipe_rendering = (self.raw_input["albedo"].permute(0, 3, 1, 2) / 2 + 0.5) * alpha + pipe_rendering * ~alpha
        if log:
            torchvision.utils.save_image(pipe_rendering, os.path.join(log_dir, f"log-output.png"))
        return pipe_rendering.float()


class TextureInpaintSingleStepModel(TextureInpaintBaseModel):
    def __init__(self, num_in_channels, in_channels, ckpt_path, use_fp16=False):
        super().__init__(num_in_channels, in_channels, ckpt_path, use_fp16)
        self.model = SingleStepLDM(vae_in_channels=num_in_channels).to(self.device)
        weights = torch.load(self.ckpt_path)
        self.model.unet.load_state_dict(weights['model_unet_ema']['avg'])
        self.model.vae.load_state_dict(weights['model_vae_ema']['avg'])
        self.prompt_embedding = self.model.encode_prompt("")

    def set_attention_proc(self, attnproc, **kwargs):
        replace_attention_processors(self.model.unet, SamplewiseAttnProcessor2_0)

    def inpaint(self, data, log=False, log_dir=None, masked=False):
        self.raw_input = self.set_input(data)
        self.raw_input['albedo_alpha'] *= self.raw_input['inpaint_mask']
        self.raw_input['inpaint_mask'] = 1 - self.raw_input['inpaint_mask']
        if log:
            for ch in self.in_channels:
                torchvision.utils.save_image(
                    (self.raw_input[ch].permute(0, 3, 1, 2) / 2) + 0.5,
                    os.path.join(log_dir, f"log-{ch}.png"),
                )
        model_inputs = torch.cat([self.raw_input[ch] for ch in self.in_channels], dim=-1).permute(0, 3, 1, 2)
        prompt_embeds = self.raw_input["prompt_embs"]
        with torch.no_grad():
            output = self.model(model_inputs, prompt_embeds) / 2 + 0.5
        self.raw_input['inpaint_mask'] = 1 - self.raw_input['inpaint_mask']
        return output.detach()


class TextureInpaintImageDiffusionModel(TextureInpaintBaseModel):
    def __init__(self, num_in_channels, in_channels, ckpt_path, use_fp16=False):
        super().__init__(num_in_channels, in_channels, ckpt_path, use_fp16)
        self.model = UNet2DModel(sample_size=256, in_channels=num_in_channels,
                                 block_out_channels=(32, 32, 32, 32), add_attention=False).to(self.device)
        self.scheduler = DDIMScheduler(
            prediction_type="sample",
            num_train_timesteps=1000
        )
        self.num_inference_steps = 10
        weights = torch.load(self.ckpt_path)
        self.model.load_state_dict(weights['model_ema']['avg'])
        self.pipe = DDIMPipeline(unet=self.model, scheduler=self.scheduler)

    def set_attention_proc(self, attnproc, **kwargs):
        pass
        #image_diffusion.set_attn_processor(self.model, attnproc)

    def inpaint(self, data, log=False, log_dir=None, masked=False):
        self.raw_input = self.set_input(data)
        self.raw_input['albedo_alpha'] *= self.raw_input['inpaint_mask']
        if log:
            for ch in self.in_channels:
                torchvision.utils.save_image(
                    (self.raw_input[ch].permute(0, 3, 1, 2) / 2) + 0.5,
                    os.path.join(log_dir, f"log-{ch}.png"),
                )
        model_inputs = [self.raw_input[ch] for ch in self.in_channels if ch != "inpaint_mask"]
        model_inputs = torch.cat(model_inputs, dim=-1).permute(0, 3, 1, 2)
        output = image_diffusion.pipe_generate(self.pipe, self.scheduler, model_inputs,
                                               num_inference_steps=self.num_inference_steps) / 2 + 0.5
        return output


class TextureInpaintControlNetModel(TextureInpaintBaseModel):
    def __init__(self, num_in_channels, in_channels, ckpt_path, use_fp16=False):
        super().__init__(num_in_channels, in_channels, ckpt_path, use_fp16)
        self.pipe = None
        self.pipeline_component = controlnet.load_controlnet_pipeline_components()
        weights = torch.load(self.ckpt_path)
        self.pipeline_component["controlnet"].load_state_dict(weights['model_ema']['avg'])
        if "model_unet_ema" in weights:
            self.pipeline_component["unet"].load_state_dict(weights['model_unet_ema']['avg'])
        self.num_inference_steps = 20

    def set_pipe(self):
        self.pipe = controlnet.load_pipe_from_components(self.pipeline_component).to(self.device)

    def set_attention_proc(self, attnproc, **kwargs):
        replace_attention_processors(
            self.pipeline_component["unet"], attnproc, **kwargs
        )  # todo: pass down kwargs
        replace_attention_processors(
            self.pipeline_component["controlnet"], attnproc, **kwargs
        )  # todo: pass down kwargs
        self.set_pipe()

    def inpaint(self, data, log=False, log_dir=None, masked=False):
        self.raw_input = self.set_input(data)
        self.raw_input['albedo_alpha'] *= self.raw_input['inpaint_mask']
        self.raw_input['albedo_alpha'] = 1 - self.raw_input['albedo_alpha']
        if log:
            for ch in self.in_channels:
                torchvision.utils.save_image(
                    (self.raw_input[ch].permute(0, 3, 1, 2) / 2) + 0.5,
                    os.path.join(log_dir, f"log-{ch}.png"),
                )
        input_image = self.raw_input["albedo"].permute(0, 3, 1, 2)
        mask_image = self.raw_input["albedo_alpha"].permute(0, 3, 1, 2)
        control_image = self.raw_input["geo_camera_normals"].permute(0, 3, 1, 2)
        if self.prompt_embedding is None:
            null_prompt = self.pipe.encode_prompt("", self.device, 1, False)[0]
            self.prompt_embedding = null_prompt.repeat(input_image.shape[0], 1, 1)
        output = self.pipe(
            image=input_image, mask_image=mask_image, control_image=control_image,
            height=256, width=256, prompt_embeds=self.prompt_embedding, guidance_scale=1.0,
            num_inference_steps=self.num_inference_steps, output_type="pt", return_dict=False
        )[0]
        if log:
            torchvision.utils.save_image(output, os.path.join(log_dir, f"log-output.png"))
        return output
