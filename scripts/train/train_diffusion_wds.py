# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

_WANDB_SMOKE_TEST = os.environ.get("WANDB_SMOKE_TEST", "").lower() in {"1", "true", "yes"}
if _WANDB_SMOKE_TEST and __name__ == "__main__":
    from scripts.train.wandb_smoke import main as _wandb_smoke_main

    _wandb_smoke_main()
    raise SystemExit(0)

import torch

import time
import copy
import json

import logging
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
import torch.functional as F
import torchvision
import torchmetrics
from torchvision.transforms import v2

from torch.utils.data.sampler import Sampler
import lpips
from accelerate import Accelerator

from diffusers import  StableUnCLIPImg2ImgPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

import kaolin
import gloss
from gloss.data.render_dataloader import MultiViewConfig, ViewConfig
from gloss.data.render_webdataset import configure_multi_view_webdataset, discover_available_webdataset_shards
from gloss.model.ema import EMA
from gloss.model.attention import replace_attention_processors, SamplewiseAttnProcessor2_0
from gloss.model.loss import canonicalize_training_objective, compute_min_snr_v_weights, \
    masked_mse_loss, masked_weighted_mse_loss, reconstruct_x0_from_prediction, sample_training_target
from gloss.model.standard import load_checkpoint_state_dict, load_pipe_from_components, load_pipeline_components, pipe_generate, get_view_embedding, get_prompt_embedding
# from gloss.utils.single_view import random_polygon_mask
from gloss.utils.mask_generator import batched_mask_generate
from gloss.config.experiment import ExperimentConfig, ExperimentHelper
from gloss.config.specification import TrainerConfig
from gloss.data.utils import load_split_indices
from gloss.logging import log_tensor, log_tensor_dict
from gloss.utils import reclaim_cuda_memory, get_string_hash
from gloss.utils.parser import ParserHelper
from gloss.utils.style_loss import style_loss


logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_gpu_memory():
    if torch.cuda.is_available():
        gpu_memory_reserved = torch.cuda.memory_reserved(0)
        gpu_memory_allocated =  torch.cuda.memory_allocated(0)
        print(f"Reserved GPU memory: {gpu_memory_reserved / 1e6:.2f} MB, Allocated GPU memory: {gpu_memory_allocated / 1e6:.2f} MB")
    else:
        print("CUDA is not available.")
        
def get_sigma(tensor, device, mean=-2.0, std=0.5):
    assert tensor.dim() == 4
    bsz = tensor.shape[0]
    num_channel = tensor.shape[1]
    rnd_normal = torch.randn([bsz, num_channel, 1, 1], device=device)
    sigma = (rnd_normal * std + mean).exp()
    return sigma

def is_geom_cond(channel):
    geom_conds = ['geo_camera_normals', 'relative_positions', 'positions', 'camera_normals']
    if channel in geom_conds:
        return True
    else:
        return False

def generate_viz(inputs, ground_truth, outputs=None, verbose=False, visualize_vae=None, max_num=10,
                 inputs_key="Inputs", ground_truth_key="GroundTruth"):
    def _logif(fun, *args, **kwargs):
        if verbose:
            fun(*args, **kwargs)

    batchsize = min(inputs[0][1].shape[0], max_num)
    width = inputs[0][1].shape[3]

    if visualize_vae is not None:
        visualize_vae.eval()

    to_cat = []
    for title, values in zip([inputs_key, "Outputs", ground_truth_key], [inputs, outputs, ground_truth]):
        if values is None:
            continue
        to_cat.append(gloss.viz.text.torch_image_with_text(title, 50, width * batchsize))

        for item in values:
            channel = item[0]
            v = item[1][:batchsize, ...]

            if v.shape[1] == 1:
                to_cat.append(torchvision.utils.make_grid(v.detach().cpu().repeat(1, 3, 1, 1), nrow=batchsize, padding=0))
            else:
                to_cat.append(torchvision.utils.make_grid(v.detach().cpu(), nrow=batchsize, padding=0))
                _logif(log_tensor, to_cat[-1], f'input viz {title}[{channel}]', logger, print_stats=True)
                if visualize_vae is not None and title != "Outputs":
                    with torch.no_grad():
                        v_dec = visualize_vae(v)["sample"].detach().cpu()
                        _logif(log_tensor, v, f'dec input {title}[{channel}]', logger, print_stats=True)
                        _logif(log_tensor, v_dec, f'dec output {title}[{channel}]', logger, print_stats=True)
                        to_cat.append(torchvision.utils.make_grid(v_dec, nrow=batchsize, padding=0))
                        del v_dec
                gloss.utils.reclaim_cuda_memory()

    to_cat = torch.cat(to_cat, dim=1)
    _logif(log_tensor, to_cat, 'to_cat', logger, print_stats=True)
    to_cat = to_cat / 2 + 0.5
    to_cat = (to_cat.clip(0, 1) * 255).to(torch.uint8)
    return to_cat


def generate_autoencoder_viz(vae, inputs, ground_truth, verbose):
    return generate_viz(inputs, ground_truth, verbose=verbose, visualize_vae=vae)
        

def filter_mask(mask, kernel_size=5, sigma=1):
    """Given a binary mask of shape B C H W, gaussian filter apply to the values, threshold 

    Args:
        mask (_type_): _description_
    """
    gaussianBlur = torchvision.transforms.GaussianBlur(kernel_size, sigma)
    mask_blur = gaussianBlur(mask)
    filter_indices = torch.argwhere(torch.abs(mask_blur -  1.0) > 1e-6)
    mask[filter_indices[:, 0], filter_indices[:, 1], filter_indices[:, 2], filter_indices[:, 3]] = 0.0
    return mask
    

def train_loop(accelerator, model, optimizer, noise_scheduler, train_dataloader, eval_dataloader,\
    lr_scheduler, loss_fn, pipe:StableUnCLIPImg2ImgPipeline, config, data_config, **kwargs):
    """ Main train loop

    Args:
        prompt (_type_): image prompt
        single_view (_type_): image prompt
        accelerator (_type_): accelerator
        model (_type_): unet
        optimizer (_type_): optimizer
        train_dataloader (_type_):dataloader
        lr_scheduler (_type_): learning rate scheduler
        loss_fn (_type_): loss function
        pipe (StableUnCLIPImg2ImgPipeline):

    Returns:
        _type_: _description_
    """
    ########## LOAD MESH ##########
    if not kwargs.get("skip_mesh_load", False):
        mesh_fname = os.path.join(kwargs["global_root_dir"], data_config.mesh)
        mesh = kaolin.io.mesh.import_mesh(mesh_fname).to(accelerator.device)
        mesh.vertices = kaolin.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)
        accelerator.print(f"sucessfully load and process mesh from {mesh_fname}")
    
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler, noise_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler, noise_scheduler
    )
    accelerator.print(count_parameters(model))
    unwrapped_model = accelerator.unwrap_model(model)

    def normalize_state_dict_keys(state_dict):
        if not isinstance(state_dict, dict):
            return state_dict
        normalized = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            normalized[key] = value
        return normalized
    
    ########## LOAD CHECKPOINT ########## 
    global_step = 0
    latest_checkpoint = os.path.join(exp_helper.checkpt_dir, f'chkpt_latest.ckpt')

    def save_weights_only_checkpoint(checkpoint_fp, step_to_save):
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError(
                "Saving model-only checkpoints requires the safetensors package"
            ) from exc

        target_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
        }[kwargs["persistent_weights_dtype"]]
        source_state_dict = ema.state_dict()["avg"] if config.ema_mu is not None else unwrapped_model.state_dict()
        weights_only_state_dict = {}
        for key, value in source_state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            tensor = value.detach()
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=target_dtype)
            weights_only_state_dict[key] = tensor.cpu().contiguous()
        save_file(weights_only_state_dict, checkpoint_fp, metadata={"global_step": str(step_to_save)})

    def save_checkpoint(step_to_save):
        checkpoint_fp = os.path.join(exp_helper.checkpt_dir, f"chkpt_{step_to_save}.ckpt")
        weights_checkpoint_fp = os.path.join(exp_helper.checkpt_dir, f"chkpt_{step_to_save}.safetensors")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if config.ema_mu is not None:
                full_state_dict = {
                        "model_ema": ema.state_dict(),
                        "opt_state": optimizer.state_dict(),
                        "global_step": step_to_save,
                    }
            else:
                full_state_dict = {
                        "model": unwrapped_model.state_dict(),
                        "opt_state": optimizer.state_dict(),
                        "global_step": step_to_save,
                    }
            if exp_helper.config.save_latest_checkpoint:
                accelerator.save(full_state_dict, latest_checkpoint)
                accelerator.print(f"Updating checkpoint to {latest_checkpoint}")
            if exp_helper.should_persist_checkpoint(step_to_save):
                if kwargs["persistent_weights_only"]:
                    save_weights_only_checkpoint(weights_checkpoint_fp, step_to_save)
                    accelerator.print(f"Wrote weights-only checkpoint to {weights_checkpoint_fp}")
                else:
                    accelerator.print(f"Wrote checkpoint to {checkpoint_fp}")
                    accelerator.save(full_state_dict, checkpoint_fp)
            del full_state_dict
            reclaim_cuda_memory()

    checkpoint_path = kwargs.get("init_checkpoint")
    if exp_helper.config.save_latest_checkpoint and os.path.isfile(latest_checkpoint):
        loaded = torch.load(latest_checkpoint, map_location="cpu")
        if config.ema_mu is not None:
            unwrapped_model.load_state_dict(normalize_state_dict_keys(loaded['model_ema']['avg']))
        else:
            unwrapped_model.load_state_dict(normalize_state_dict_keys(loaded['model']))
        optimizer.load_state_dict(loaded['opt_state'])
        global_step = loaded['global_step']
        exp_helper.start_training_at(global_step)
        for _ in range(global_step):
            lr_scheduler.step()
        accelerator.print(f"Loaded model from {latest_checkpoint} at global iteration {global_step}; "
                          f"fast-forwarded lr_scheduler to match num_training_steps={config.total_iterations}")
    elif checkpoint_path:
        loaded_state_dict, init_step = load_checkpoint_state_dict(checkpoint_path, map_location="cpu")
        unwrapped_model.load_state_dict(normalize_state_dict_keys(loaded_state_dict))
        if init_step >= 0:
            accelerator.print(f"Initialized model from {checkpoint_path} at global iteration {init_step}")
        else:
            accelerator.print(f"Initialized model from {checkpoint_path}")

    if accelerator.is_local_main_process:
        if config.ema_mu is not None:
            ema = EMA(config.ema_mu)
            ema.reset()
            if exp_helper.config.save_latest_checkpoint and os.path.isfile(latest_checkpoint):
                ema.load_state_dict(loaded['model_ema'], "cpu")
                del loaded
    
    accelerator.print("Preparing model")
    vae = pipe.vae
    if accelerator.is_main_process:
        vae = vae.to(accelerator.device)

    if kwargs["image_loss"]:
        net_lpips = lpips.LPIPS(net='vgg').to(accelerator.device)
        net_lpips.requires_grad_(False)

        if kwargs["style_loss"]:
            net_vgg = torchvision.models.vgg16(pretrained=True).features
            for param in net_vgg.parameters():
                param.requires_grad_(False)
            net_vgg = accelerator.prepare(net_vgg)
    
    def encode_inputs(raw_inputs):
        def _encode_input(val):
            if val.shape[1] == 3:
                res = pipe.vae.encode(val, return_dict=False)[0].sample() * pipe.vae.config.scaling_factor
            else:  # we assume others are not first
                res = -torch.nn.functional.max_pool2d(-val, 8)
            reclaim_cuda_memory()
            return res
        with torch.no_grad():
            encoded_inputs = [_encode_input(x[1]) for x in raw_inputs]
        return torch.cat(encoded_inputs, dim=1)
    
    ########## LOAD MASKING ########## 
    
    def get_loss_mask(_batch):
        _background_mask = _batch['background_alpha'] / 2 + 0.5
        _loss_mask = _batch['albedo_alpha'] / 2 + 0.5
        _loss_mask = (_loss_mask > 0.5).float()
        return _loss_mask

    def mask_input_channel(_channel, _input_batch):
        background =  1.0 - (_input_batch["background_alpha"] / 2 + 0.5)
        albedo_alpha = _input_batch["albedo_alpha"] / 2 + 0.5
        albedo_alpha = (albedo_alpha > 0.5).float()
        if _channel in data_config.output_channels:
            _val = _input_batch[_channel] * _input_batch['inpaint_polygon'] 
            _val = _val * albedo_alpha + torch.ones_like(_val) * background * -1.0
            return _val
        else:
            return _input_batch[_channel] 

    def sample_mask_generation_config():
        full_ratio = 0.1
        min_ref, max_ref = 1, 4
        mask_prob = torch.rand(1)
        if mask_prob <= 0.25:
            full_ratio = 1.0
            min_ref = 4
        elif mask_prob <= 0.5:
            full_ratio = 0.5
            min_ref = 2
        return full_ratio, min_ref, max_ref

    def compute_mask_ratio(inpaint_polygon, albedo_mask):
        masked_area = (1.0 - inpaint_polygon) * albedo_mask
        valid_area = albedo_mask.sum().clamp_min(1.0)
        return masked_area.sum() / valid_area

    def apply_sampled_inpaint_mask(_batch, _albedo_mask, _background_mask):
        full_ratio, min_ref, max_ref = sample_mask_generation_config()
        _batch['inpaint_polygon'] = batched_mask_generate(
            _batch["albedo"], None, full_ratio, min_ref, max_ref
        ).permute(0, 3, 1, 2)
        _batch['inpaint_mask'] = _batch['inpaint_polygon'] * _albedo_mask + (1.0 - _background_mask)
        return compute_mask_ratio(_batch['inpaint_polygon'], _albedo_mask)

    def apply_fixed_eval_inpaint_mask(_batch, _albedo_mask, _background_mask):
        eval_batch_size = _batch["albedo"].shape[0]
        num_visible = min(2, eval_batch_size)
        inpaint_polygon = torch.zeros_like(_batch["albedo_alpha"])
        visible_indices = torch.randperm(eval_batch_size, device=_batch["albedo_alpha"].device)[:num_visible]
        inpaint_polygon[visible_indices] = 1.0
        _batch['inpaint_polygon'] = inpaint_polygon
        _batch['inpaint_mask'] = _batch['inpaint_polygon'] * _albedo_mask + (1.0 - _background_mask)
        return compute_mask_ratio(_batch['inpaint_polygon'], _albedo_mask)

    ########## LOAD VGG ########## 
    if accelerator.is_main_process and config.log_lpips:
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(accelerator.device)
        
    ########## LOAD COLOR JITTERING ########## 
    jitter = v2.ColorJitter(saturation=0.3, contrast=0.3, hue=0.2)
    
    ########## SET NULL PROMPT ########## 
    null_prompt = get_prompt_embedding(pipe, [""], accelerator.device)
    
    ########## LOAD AUGMENTATION ##########
    do_augmentation = False
    if config.augment_every != -1:
        do_augmentation = True
    training_objective = canonicalize_training_objective(config.objective)
    

    target_steps = config.total_iterations
    epoch = 0
    last_log_time = time.time()
    last_log_completed_steps = 0

    warmup_grad_accum_steps = max(1, int(getattr(config, "warmup_grad_accum_steps", 1)))
    warmup_grad_accum_iters = max(0, int(getattr(config, "warmup_grad_accum_iters", 0)))
    grad_accum_counter = 0
    if warmup_grad_accum_steps > 1 and warmup_grad_accum_iters > 0:
        accelerator.print(
            f"Warmup gradient accumulation enabled: averaging over "
            f"{warmup_grad_accum_steps} micro-steps for the first "
            f"{warmup_grad_accum_iters} iterations."
        )
    
    ########## TRAINING ##########
    while global_step < target_steps:
        accelerator.print(f"epoch: {epoch}")
        epoch += 1
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for idx, batch in tqdm(enumerate(train_dataloader)):
            verbose = (global_step == 0)
            def _logif(fun, *args, **kwargs):
                if verbose:
                    fun(*args, **kwargs)
            
            if do_augmentation and idx % config.augment_every == 0:
                n = batch[data_config.input_channels[0]].shape[0] // 2
                ########## set random rotate angles ########## 
                angles = torch.rand(n) * 360.0
                for k in batch.keys():
                    if isinstance(batch[k], torch.Tensor):
                        augs = []
                        for i in range(n):
                            # rotate, pad the rest with -1
                            augs.append(torchvision.transforms.functional.rotate(batch[k][i].unsqueeze(0), angle=angles[i].item(), fill=-1.0))
                        augs = torch.cat(augs, dim=0)
                        batch[k] = torch.cat([batch[k][:n], augs], dim=0)
                        batch[k] = batch[k].to(accelerator.device, non_blocking=True)
            else:
                for k in batch.keys():
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(accelerator.device, non_blocking=True)
            
            batch_size = batch[data_config.input_channels[0]].shape[0] 
            prompt_embeds = null_prompt.repeat(batch_size, 1, 1)
            
            with torch.no_grad():
                seed = torch.randint(1234567890, (1,))
                torch.manual_seed(seed)
                _albedo = batch["albedo"] / 2 + 0.5
                batch["albedo"] = jitter(_albedo) * 2 - 1 
                torch.manual_seed(seed)
                _view = jitter(batch["view"])
                batch['view_embed'] = get_view_embedding(pipe, _view, accelerator.device)     
            sing_view_embed = batch['view_embed']   
                  
            # 0..1 masks (0 == unknown)
            loss_mask = get_loss_mask(batch)
            background_mask = batch['background_alpha'] / 2 + 0.5
            albedo_mask = batch['albedo_alpha'] / 2 + 0.5
            albedo_mask = (albedo_mask > 0.5).float()
            
            if do_augmentation and idx % config.augment_every == 0:
                batch['inpaint_polygon'] = torch.cat([torch.zeros_like(batch['albedo_alpha'][:batch_size//2]), torch.ones_like(batch['albedo_alpha'][:batch_size//2])])
                batch['inpaint_mask'] = batch['inpaint_polygon'] * albedo_mask + (1.0 - background_mask)
                # random shuffle the batch
                perm = torch.randperm(batch_size)
                for k in batch.keys():
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k][perm] 
                train_mask_ratio = compute_mask_ratio(batch['inpaint_polygon'], albedo_mask)
            else:
                train_mask_ratio = apply_sampled_inpaint_mask(batch, albedo_mask, background_mask)
            
            raw_inputs = []
            for channel in data_config.input_channels:
                if is_geom_cond(channel) and torch.rand(1) < train_config.p_cfg:
                    raw_inputs.append((channel, torch.zeros_like(batch[channel])))
                else:
                    raw_inputs.append((channel, mask_input_channel(channel, batch)))

            ground_truth = [(channel, batch[channel]) for channel in data_config.output_channels]
            
            #loss_mask_latent = resize(loss_mask)
            loss_mask_latent = -torch.nn.functional.max_pool2d(-loss_mask, 8)
            loss_mask_latent = filter_mask(loss_mask_latent)

            encoded_ground_truth = encode_inputs(ground_truth)
            x = encoded_ground_truth
            encoded_inputs = encode_inputs(raw_inputs)
            del batch
            
            # Run any initial visualizations
            if global_step == 0 and accelerator.is_main_process:
                # log_tensor_dict(batch, 'raw batch', logger, print_stats=True)
                log_tensor_dict(dict(raw_inputs), 'raw_inputs', logger, print_stats=True)
                log_tensor_dict(dict(ground_truth), 'raw_ground_truth', logger, print_stats=True)

                viz_fname = os.path.join(exp_helper.viz_dir, 'encoder_test.jpg')
                torchvision.io.write_jpeg(generate_autoencoder_viz(vae, raw_inputs, ground_truth, verbose=verbose),
                                          viz_fname, quality=90)
                accelerator.print(f'Wrote encoder sanity to {viz_fname}')

            if exp_helper.should_checkpoint(global_step):
                save_checkpoint(global_step)
            
                
            if exp_helper.should_eval(global_step):
                # get output from the training dataset 
                if accelerator.is_main_process:
                    log_tensor_dict(dict(raw_inputs), 'raw_inputs', logger, print_stats=True)
                    log_tensor_dict(dict(ground_truth), 'raw_ground_truth', logger, print_stats=True) 
                    accelerator.log({"train_eval_mask_ratio": train_mask_ratio.item()}, step=global_step)
                    exp_helper.log("train_eval_mask_ratio", train_mask_ratio.item(), global_step)
                    output_images = (pipe_generate(pipe, noise_scheduler, prompt_embeds, sing_view_embed, encoded_inputs) - 0.5) * 2
                    outputs = [('result', output_images)]
                    viz_fname = os.path.join(exp_helper.viz_dir, 'train', f'step_{global_step}.jpg')
                    torchvision.io.write_jpeg(generate_viz(raw_inputs, ground_truth, outputs, verbose=verbose,),
                                              viz_fname, quality=90)
                    accelerator.print(f'Wrote encoder sanity to {viz_fname}')
                    del output_images
                    del outputs
                    reclaim_cuda_memory()

                # Evaluate on every rank so non-main ranks do not race ahead into the next
                # gradient collective while rank 0 is still rendering evaluation images.
                eval_lpips_losses = []
                eval_mask_ratios = []
                with torch.no_grad():
                    for idx, batch in tqdm(enumerate(eval_dataloader), disable=not accelerator.is_local_main_process):
                        if idx > 9:
                            break

                        for k in batch.keys():
                            if isinstance(batch[k], torch.Tensor):
                                batch[k] = batch[k].to(accelerator.device)

                        eval_loss_mask = get_loss_mask(batch)
                        eval_batch_size = eval_loss_mask.shape[0]
                        eval_prompt_embeds = null_prompt.repeat(eval_batch_size, 1, 1)
                        eval_view = batch["view"]
                        batch['view_embed'] = get_view_embedding(pipe, eval_view, accelerator.device)   
                        eval_sing_view_embeds = batch["view_embed"]
                        
                        eval_background_mask = batch['background_alpha'] / 2 + 0.5
                        eval_albedo_mask = batch['albedo_alpha'] / 2 + 0.5
                        eval_albedo_mask = (eval_albedo_mask > 0.5).float()
                        eval_mask_ratio = apply_fixed_eval_inpaint_mask(batch, eval_albedo_mask, eval_background_mask)
                        eval_mask_ratios.append(eval_mask_ratio.item())
                        eval_raw_inputs = [(channel, mask_input_channel(channel, batch))
                            for channel in data_config.input_channels]
                        
                        eval_ground_truth = [(channel, batch[channel]) for channel in data_config.output_channels]
                        eval_encoded_inputs = encode_inputs(eval_raw_inputs)
                        eval_output_images = (pipe_generate(pipe, noise_scheduler, eval_prompt_embeds, eval_sing_view_embeds, eval_encoded_inputs) - 0.5) * 2
                            
                        if config.log_lpips and accelerator.is_main_process:
                            eval_lpips_loss = torch.mean(loss_fn_vgg(eval_output_images*eval_loss_mask, eval_ground_truth[0][1]*eval_loss_mask))
                            eval_lpips_losses.append(eval_lpips_loss.item())
                            del eval_lpips_loss
                        
                        if accelerator.is_main_process:
                            eval_outputs = [('result', eval_output_images)]
                            viz_fname = os.path.join(exp_helper.viz_dir, 'eval', f'step_{global_step}_batch_{idx}.jpg')
                            torchvision.io.write_jpeg(generate_viz(eval_raw_inputs, eval_ground_truth, eval_outputs, verbose=verbose,),
                                                      viz_fname, quality=90)
                            accelerator.print(f'Wrote evaluation results to {viz_fname}')
                            del eval_outputs
                        del eval_output_images
                        del eval_raw_inputs
                        del eval_encoded_inputs
                        del eval_ground_truth
                        reclaim_cuda_memory()  
                accelerator.wait_for_everyone()
                if config.log_lpips:
                    local_lpips_sum = torch.tensor(sum(eval_lpips_losses), device=accelerator.device, dtype=torch.float32)
                    local_lpips_count = torch.tensor(len(eval_lpips_losses), device=accelerator.device, dtype=torch.float32)
                    total_lpips_sum = accelerator.reduce(local_lpips_sum, reduction="sum")
                    total_lpips_count = accelerator.reduce(local_lpips_count, reduction="sum")
                    if accelerator.is_main_process and total_lpips_count.item() > 0:
                        mean_eval_lpips = (total_lpips_sum / total_lpips_count).item()
                        accelerator.log({"eval_lpips_loss": mean_eval_lpips}, step=global_step)
                        exp_helper.log("eval_lpips_loss", mean_eval_lpips, global_step)
                local_mask_sum = torch.tensor(sum(eval_mask_ratios), device=accelerator.device, dtype=torch.float32)
                local_mask_count = torch.tensor(len(eval_mask_ratios), device=accelerator.device, dtype=torch.float32)
                total_mask_sum = accelerator.reduce(local_mask_sum, reduction="sum")
                total_mask_count = accelerator.reduce(local_mask_count, reduction="sum")
                if accelerator.is_main_process and total_mask_count.item() > 0:
                    mean_eval_mask_ratio = (total_mask_sum / total_mask_count).item()
                    accelerator.log({"eval_mask_ratio": mean_eval_mask_ratio}, step=global_step)
                    exp_helper.log("eval_mask_ratio", mean_eval_mask_ratio, global_step)
            
            if accelerator.is_main_process and exp_helper.should_log_metric(global_step):
                output_images = (pipe_generate(pipe, noise_scheduler, prompt_embeds, sing_view_embed, encoded_inputs) - 0.5) * 2
                if config.log_lpips:
                    lpips_loss = torch.mean(loss_fn_vgg(output_images*loss_mask, ground_truth[0][1]*loss_mask))
                    accelerator.log({"lpips_loss": lpips_loss.item()}, step=global_step)
                    exp_helper.log("lpips_loss",  lpips_loss.item(), global_step)
                del output_images
                reclaim_cuda_memory()
                
            reclaim_cuda_memory()

            with torch.no_grad():
                sigma = get_sigma(encoded_inputs, accelerator.device)
                encoded_inputs = torch.randn_like(encoded_inputs) * sigma + encoded_inputs

            do_image_losses = (global_step % 10 == 0) & kwargs["image_loss"]
            max_timestep_fraction = 0.2 if do_image_losses else 1.0
            timesteps, x_noisy, train_target, flow_sigma = sample_training_target(
                x,
                noise_scheduler,
                training_objective,
                config.num_training_timesteps,
                max_timestep_fraction=max_timestep_fraction,
            )
            model.train()

            with accelerator.accumulate(model):
                encoded_inputs = torch.cat([x_noisy, encoded_inputs], dim=1)
                model_pred = model(encoded_inputs,
                                   timesteps,
                                   encoder_hidden_states=prompt_embeds,
                                   class_labels=sing_view_embed,
                                   return_dict=False)[0]

                image_losses = 0
                if do_image_losses:
                    pred_x0 = reconstruct_x0_from_prediction(
                        x_noisy,
                        model_pred,
                        noise_scheduler,
                        timesteps,
                        training_objective,
                        sigma=flow_sigma,
                    )
                    pred_images = pipe.vae.decode(pred_x0 / pipe.vae.config.scaling_factor, return_dict=False)[0]
                    # outputs = [('result', pred_images)]
                    # image_loss_fname = os.path.join(exp_helper.viz_dir, 'train', f'step_image_loss_{global_step}.jpg')
                    # torchvision.io.write_jpeg(generate_viz(raw_inputs, ground_truth, outputs, verbose=verbose, ),
                    #                          image_loss_fname, quality=90)
                    # del raw_inputs
                    gt_images = ground_truth[0][1]

                    masked_pred = pred_images * loss_mask
                    masked_gt = gt_images * loss_mask
                    loss_lpips = torch.mean(net_lpips(masked_pred, masked_gt))

                    if kwargs["style_loss"]:
                        t_vgg_renorm = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        x_tgt_pred_renorm = t_vgg_renorm(masked_pred * 0.5 + 0.5)
                        x_tgt_renorm = t_vgg_renorm(masked_gt * 0.5 + 0.5)
                        loss_gram = style_loss(x_tgt_pred_renorm, x_tgt_renorm, net_vgg)
                        image_losses = loss_lpips + 0.01 * loss_gram
                    else:
                        image_losses = loss_lpips

                if config.min_snr_gamma is not None:
                    snr_weights = compute_min_snr_v_weights(
                        noise_scheduler,
                        timesteps,
                        training_objective,
                        config.min_snr_gamma,
                        sigma=flow_sigma,
                    )
                    mse_loss_val = masked_weighted_mse_loss(
                        model_pred, train_target, loss_mask_latent, snr_weights
                    )
                else:
                    mse_loss_val = loss_fn(model_pred, train_target, loss_mask_latent)
                loss = mse_loss_val + 0.1 * image_losses
                if torch.isnan(loss):           # loss is a tensor
                    print(f"[STOP] NaN detected on epoch {epoch}. Aborting training.")
                    exit(0)

                in_warmup = (
                    warmup_grad_accum_steps > 1
                    and global_step < warmup_grad_accum_iters
                )
                effective_k = warmup_grad_accum_steps if in_warmup else 1
                loss_to_backward = loss / effective_k if effective_k > 1 else loss
                accelerator.backward(loss_to_backward)

                grad_accum_counter += 1
                do_optim_step = (grad_accum_counter >= effective_k)
                if do_optim_step:
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    grad_accum_counter = 0
                lr_scheduler.step()

                reclaim_cuda_memory()
                if exp_helper.should_log(global_step):
                    accelerator.log({"train_loss": loss.item()}, step=global_step)
                    accelerator.print({"train_loss": loss.item()})
                    completed_steps = global_step + 1
                    now = time.time()
                    elapsed = max(now - last_log_time, 1e-6)
                    delta_steps = max(completed_steps - last_log_completed_steps, 1)
                    global_batch_size = train_config.batch_size * accelerator.num_processes
                    steps_per_sec = delta_steps / elapsed
                    samples_per_sec = (delta_steps * global_batch_size) / elapsed
                    accelerator.log(
                        {
                            "benchmark/steps_per_sec": steps_per_sec,
                            "benchmark/samples_per_sec": samples_per_sec,
                            "benchmark/global_batch_size": global_batch_size,
                        },
                        step=global_step,
                    )
                    accelerator.print(
                        {
                            "benchmark_steps_per_sec": round(steps_per_sec, 4),
                            "benchmark_samples_per_sec": round(samples_per_sec, 2),
                            "benchmark_global_batch_size": global_batch_size,
                        }
                    )
                    last_log_time = now
                    last_log_completed_steps = completed_steps

                if accelerator.is_main_process and do_optim_step:
                    if config.ema_mu is not None:
                        ema.update(model)

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1

            if global_step >= target_steps:
                if exp_helper.should_checkpoint(global_step):
                    save_checkpoint(global_step)
                progress_bar.close()
                break


if __name__ == '__main__':
    parser_helper = ParserHelper('Inpainter from scratch')
    parser_helper.add_dataclass_flags(ExperimentConfig, 'exp')
    parser_helper.add_dataclass_flags(TrainerConfig, 'train')
    parser_helper.add_dataclass_flags(MultiViewConfig, 'data')
    parser_helper.parser.add_argument('--multi_attention', default=False, action='store_true', help='use multipatch attention or not')
    parser_helper.parser.add_argument('--image_loss', default=False, action='store_true', help='whether to use image loss or not')
    parser_helper.parser.add_argument('--style_loss', default=False, action='store_true', help='whether to use style loss or not')
    parser_helper.parser.add_argument('--fine_tune', default=False, action='store_true', help='whether to fine tune model or not')
    parser_helper.parser.add_argument('--init_checkpoint', type=str, default=None,
                                      help='Optional checkpoint used to initialize the model before training')
    parser_helper.parser.add_argument('--train_num_workers', type=int, default=1,
                                      help='Number of dataloader workers for train WDS loading')
    parser_helper.parser.add_argument('--eval_num_workers', type=int, default=1,
                                      help='Number of dataloader workers for eval WDS loading')
    parser_helper.parser.add_argument('--persistent_weights_only', default=False, action='store_true',
                                      help='Save persistent checkpoints as model-only safetensors instead of full optimizer-backed .ckpt files')
    parser_helper.parser.add_argument('--persistent_weights_dtype', type=str, default='float16', choices=('float16', 'float32'),
                                      help='Floating-point dtype to use when writing model-only persistent checkpoints')
    args = parser_helper.parse_args()
    
    exp_config = args.exp
    train_config = args.train
    data_config = args.data

    if args.image_loss:
        args.fine_tune = True
    
    data_root_dir = os.path.join(args.global_root_dir, data_config.data_base_dir)
    available_wds_shards = discover_available_webdataset_shards(data_root_dir)
    if available_wds_shards:
        data_config.num_views = len(available_wds_shards)

    # DO NOT USE MANUAL SEED

    exp_helper = ExperimentHelper(exp_config, args)
    config_path = os.path.join(exp_helper.config_dir, 'config.yml')
    if not os.path.exists(config_path):
        parser_helper.write_config_yml(os.path.join(exp_helper.config_dir, 'config.yml'))
    with open(os.path.join(exp_helper.config_dir, 'config.yml'), "r") as file:
        config_string = file.read()

    accelerator = Accelerator(
        mixed_precision="fp16",
        log_with="wandb",
        project_dir= exp_helper.log_dir,
    )
    
    run_id = get_string_hash(exp_config.name)
    os.environ["WANDB_RUN_ID"] = run_id
    os.environ['WANDB_DEBUG'] = 'true'

    all_configs = {**exp_config.__dict__, **train_config.__dict__, **data_config.__dict__}

    # Accelerate's WandB tracker calls wandb.init(project=project_name, **init_kwargs).
    accelerator.init_trackers(project_name='gloss', config=all_configs, 
                              init_kwargs={"wandb":{"group": exp_config.group, "name": exp_config.name, "resume": "allow"}},)
    wandb_run = accelerator.get_tracker("wandb", unwrap=True)
    if accelerator.is_main_process:
        wandb_mode = getattr(getattr(wandb_run, "settings", None), "mode", None)
        wandb_url = getattr(wandb_run, "url", None)
        requested_mode = os.environ.get("WANDB_MODE", "online").lower()
        if requested_mode == "online" and wandb_mode != "online":
            # Catch a silent fallback to offline when online logging was asked for.
            raise RuntimeError(
                f"W&B did not start in online mode (mode={wandb_mode!r}). "
                "Run `wandb login` first, or set WANDB_MODE=offline to log locally."
            )
        accelerator.print(f"W&B run initialized (mode={wandb_mode}): {wandb_url or 'local only'}")
        if os.environ.get("WANDB_SMOKE_TEST", "").lower() in {"1", "true", "yes"}:
            accelerator.log({"wandb_smoke_test": 1.0}, step=0)
            accelerator.print("W&B smoke test metric logged; exiting before training setup.")
            accelerator.end_training()
            raise SystemExit(0)
    
    if data_config.normal_cond == "None":
        data_config.num_in_channels -= 4
    if data_config.position_cond == "None":
        data_config.num_in_channels -= 4
    
    pretrained_unet_files = (
        os.path.join(os.environ.get("GLOSS_MODEL_DIR", ""), "unet", "diffusion_pytorch_model.safetensors"),
        os.path.join(os.environ.get("GLOSS_MODEL_DIR", ""), "unet", "diffusion_pytorch_model.bin"),
    )
    use_pretrained_unet = args.fine_tune and args.init_checkpoint is None and any(
        os.path.isfile(path) for path in pretrained_unet_files
    )
    pipeline_components = load_pipeline_components(
        data_config.num_in_channels,
        from_scratch=not use_pretrained_unet,
        objective=train_config.objective,
        flow_shift=train_config.flow_shift,
    )
    pipeline_components["vae"].requires_grad_(False)
    pipeline_components["text_encoder"].requires_grad_(False)
    pipeline_components["image_encoder"].requires_grad_(False)
    if args.multi_attention:
        replace_attention_processors(pipeline_components['unet'], SamplewiseAttnProcessor2_0)
    model = pipeline_components['unet']
    pipe = load_pipe_from_components(pipeline_components).to(accelerator.device,)
    noise_scheduler = pipeline_components['scheduler']
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.lr, eps=1e-4)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=train_config.lr_warmup_steps,
        num_training_steps=train_config.total_iterations
    )
    def loss_fn(prediction, target, _loss_mask):
        return masked_mse_loss(prediction, target, _loss_mask)

    accelerator.print("Setting up Data Loader")
    data_base_dir = os.path.join(args.global_root_dir, exp_config.base_dir)

    indices_fp = os.path.join(exp_helper.config_dir, "indices.json")
    data_meta_fp = os.path.join(args.global_root_dir, data_config.data_base_dir, "data_meta.json")
    if os.path.exists(data_meta_fp):
        with open(data_meta_fp, 'r', encoding='utf-8') as f:
            data_meta = json.load(f)
        filtered_indices = data_meta["low_quality_sample_indices"]
    else:
        filtered_indices = []
    dataset_mask = [i for i in range(data_config.num_views) if i not in filtered_indices]
    num_filtered_views = len(dataset_mask)
    
    _normal_cond = copy.deepcopy(data_config.normal_cond)
    if "sea_urchin_shell" in exp_config.name:
        data_config.normal_cond = "geonormal"

    train_dataset, eval_dataset = configure_multi_view_webdataset(data_config, args.global_root_dir, indices_fp,
                                                                  train_config.batch_size, data_config.eval_views,
                                                                  random_seed=data_config.seed, mask=dataset_mask,
                                                                  eval_shuffle=False)
    eval_dataset.batch_size = 8
    
    view_config = ViewConfig(data_config.mesh, data_config.num_local_views, fov_min=data_config.fov_min,
                             fov_max=data_config.fov_max, camera_dist=data_config.camera_dist)
    
    data_config.normal_cond = _normal_cond
    _normal, _position = 'camera_normals', 'relative_positions'
    if data_config.normal_cond == "geonormal":
        _normal = 'geo_camera_normals'
    if data_config.position_cond == "global":
        _position = 'positions'
    _in_channels = ['albedo', 'inpaint_mask']    
    if not data_config.position_cond == "None":
        _in_channels.insert(0, _position)
    if not data_config.normal_cond == "None":
        _in_channels.insert(0, _normal)

    view_config.input_channels=_in_channels
    
    train_loader = train_dataset.get_dataloader(
        train_config.batch_size * accelerator.num_processes,
        num_workers=args.train_num_workers,
    )
    eval_loader = eval_dataset.get_dataloader(
        8,
        num_workers=args.eval_num_workers,
    )
    train_loop(accelerator, model, optimizer, noise_scheduler, train_loader, eval_loader, lr_scheduler, loss_fn, \
        pipe, train_config, view_config, global_root_dir=args.global_root_dir, image_loss=args.image_loss, style_loss=args.style_loss,
        init_checkpoint=args.init_checkpoint, persistent_weights_only=args.persistent_weights_only,
        persistent_weights_dtype=args.persistent_weights_dtype)
