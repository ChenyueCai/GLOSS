# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
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
from gloss.data.render_dataloader import MultiViewConfig, ViewConfig, configure_multi_view_dataset, cat_collate_dicts, \
    train_eval_split
from gloss.model.ema import EMA
from gloss.model.attention import replace_attention_processors, SamplewiseAttnProcessor2_0
from gloss.model.loss import canonicalize_training_objective, masked_mse_loss, sample_training_target
from gloss.model.standard import load_pipe_from_components, load_pipeline_components, pipe_generate, get_view_embedding, get_prompt_embedding
from gloss.utils.single_view import random_polygon_mask
from gloss.config.experiment import ExperimentConfig, ExperimentHelper
from gloss.config.specification import TrainerConfig
from gloss.data.utils import load_split_indices
from gloss.logging import log_tensor, log_tensor_dict
from gloss.utils import reclaim_cuda_memory, get_string_hash
from gloss.utils.parser import ParserHelper


logging.basicConfig(
    level=logging.INFO,          # Anything below INFO (i.e., DEBUG) is discarded
    format="%(levelname)s | %(name)s | %(message)s",
    force=True                   # Overwrites any existing root-logger setup (3.8+)
)
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
    geom_conds = ['geo_camera_normals', 'relative_positions', 'camera_normals']
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
    mesh_fname = os.path.join(kwargs["global_root_dir"], data_config.mesh)
    mesh = kaolin.io.mesh.import_mesh(mesh_fname).to(accelerator.device) 
    mesh.vertices = kaolin.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)
    accelerator.print(f"sucessfully load and process mesh from {mesh_fname}")
    
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler, noise_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler, noise_scheduler
    )
    accelerator.print(count_parameters(model))
    
    ########## LOAD CHECKPOINT ########## 
    global_step = 0
    latest_checkpoint = os.path.join(exp_helper.checkpt_dir, f'chkpt_latest.ckpt')
    if os.path.isfile(latest_checkpoint):
        loaded = torch.load(latest_checkpoint)
        if config.ema_mu is not None:
            model.load_state_dict(loaded['model_ema']['avg'])
        else:
            model.load_state_dict(loaded['model'])
        optimizer.load_state_dict(loaded['opt_state'])
        global_step = loaded['global_step']
        exp_helper.start_training_at(global_step)
        accelerator.print(f"Loaded model from {latest_checkpoint} at global iteration {global_step}")

    if accelerator.is_local_main_process:
        if config.ema_mu is not None:
            ema = EMA(config.ema_mu)
            ema.reset()
            if os.path.isfile(latest_checkpoint):
                ema.load_state_dict(loaded['model_ema'], "cpu")
                del loaded
    
    accelerator.print("Preparing model")
    vae = pipe.vae
    if accelerator.is_main_process:
        vae = vae.to(accelerator.device)
    vae_encoder_size = 32 # TODO
    resize = torchvision.transforms.Resize((vae_encoder_size, vae_encoder_size), torchvision.transforms.InterpolationMode.NEAREST)
    
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
        _loss_mask = (_loss_mask > 0.99).float()
        return _loss_mask

    def mask_input_channel(_channel, _input_batch):
        background =  1.0 - (_input_batch["background_alpha"] / 2 + 0.5)
        albedo_alpha = batch["albedo_alpha"] / 2 + 0.5
        if _channel in data_config.output_channels:
            _val = _input_batch[_channel] * _input_batch['inpaint_polygon'] 
            _val = _val * albedo_alpha + torch.ones_like(_val) * background * -1.0
            return _val
        else:
            return _input_batch[_channel] 

    ########## LOAD VGG ########## 
    if accelerator.is_main_process and config.log_lpips:
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(accelerator.device)
        
    ########## LOAD COLOR JITTERING ########## 
    jitter = v2.ColorJitter(saturation=0.3, contrast=0.3, hue=0.2)
    
    ########## SET NULL PROMPT ########## 
    null_prompt = get_prompt_embedding(pipe, [""], accelerator.device)
    training_objective = canonicalize_training_objective(config.objective)

    target_steps = config.total_iterations - global_step
    epoch = 0
    
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
            batch['inpaint_polygon'] = random_polygon_mask(
                data_config.resolution, batch_size, device=accelerator.device).permute(0, 3, 1, 2)
            batch['inpaint_mask'] = batch['inpaint_polygon'] * albedo_mask + (1.0 - background_mask)
            
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
                checkpoint_fp = os.path.join(exp_helper.checkpt_dir, f"chkpt_{global_step}.ckpt")
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    opt_unwrapped = accelerator.unwrap_model(optimizer)
                    if config.ema_mu is not None:
                        full_state_dict = {
                                "model_ema": ema.state_dict(),
                                "opt_state": opt_unwrapped.state_dict(),
                                "global_step": global_step,
                            }
                    else:
                        full_state_dict = {
                                "model": model.state_dict(),
                                "opt_state": opt_unwrapped.state_dict(),
                                "global_step": global_step,
                            }
                    accelerator.save(full_state_dict, latest_checkpoint)
                    #accelerator.print(f"Wrote checkpoint to {checkpoint_fp}")
                    #accelerator.save(full_state_dict, checkpoint_fp)
                    accelerator.print(f"Updating checkpoint to {latest_checkpoint}")
                    if exp_helper.should_persist_checkpoint(global_step):
                        accelerator.print(f"Wrote checkpoint to {checkpoint_fp}")
                        accelerator.save(full_state_dict, checkpoint_fp)
                    del full_state_dict
                    reclaim_cuda_memory()  

            if accelerator.is_main_process and exp_helper.should_eval(global_step):
                # get output from the training dataset 
                log_tensor_dict(dict(raw_inputs), 'raw_inputs', logger, print_stats=True)
                log_tensor_dict(dict(ground_truth), 'raw_ground_truth', logger, print_stats=True) 
                output_images = (pipe_generate(pipe, noise_scheduler, prompt_embeds, sing_view_embed, encoded_inputs) - 0.5) * 2
                outputs = [('result', output_images)]
                viz_fname = os.path.join(exp_helper.viz_dir, 'train', f'step_{global_step}.jpg')
                torchvision.io.write_jpeg(generate_viz(raw_inputs, ground_truth, outputs, verbose=verbose,),
                                          viz_fname, quality=90)
                accelerator.print(f'Wrote encoder sanity to {viz_fname}')
                del output_images
                del outputs
                del raw_inputs
                reclaim_cuda_memory()
                
                # rand select 10 batch
                eval_lpips_losses = []
                with torch.no_grad():
                    for idx, batch in tqdm(enumerate(eval_dataloader)):
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
                        
                        eval_albedo_mask = batch['albedo_alpha'] / 2 + 0.5
                        batch['inpaint_polygon'] = random_polygon_mask(
                            data_config.resolution, batch_size, device=accelerator.device).permute(0, 3, 1, 2)
                        batch['inpaint_mask'] = batch['inpaint_polygon'] * eval_albedo_mask 
                        eval_raw_inputs = [(channel, mask_input_channel(channel, batch))
                            for channel in data_config.input_channels]
                        
                        eval_ground_truth = [(channel, batch[channel]) for channel in data_config.output_channels]
                        eval_encoded_inputs = encode_inputs(eval_raw_inputs)
                        eval_output_images = (pipe_generate(pipe, noise_scheduler, eval_prompt_embeds, eval_sing_view_embeds, eval_encoded_inputs) - 0.5) * 2
                            
                        if config.log_lpips:
                            eval_lpips_loss = torch.mean(loss_fn_vgg(eval_output_images*eval_loss_mask, eval_ground_truth[0][1]*eval_loss_mask))
                            accelerator.log({"eval_lpips_loss": eval_lpips_loss.item()}, step=global_step)
                            eval_lpips_losses.append(eval_lpips_loss.item())
                            del eval_lpips_loss
                        
                        eval_outputs = [('result', eval_output_images)]
                        viz_fname = os.path.join(exp_helper.viz_dir, 'eval', f'step_{global_step}_batch_{idx}.jpg')
                        torchvision.io.write_jpeg(generate_viz(eval_raw_inputs, eval_ground_truth, eval_outputs, verbose=verbose,),
                                                  viz_fname, quality=90)
                        accelerator.print(f'Wrote evaluation results to {viz_fname}')
                        del eval_output_images
                        del eval_outputs
                        del eval_raw_inputs
                        del eval_encoded_inputs
                        del eval_ground_truth
                        reclaim_cuda_memory()  
                exp_helper.log("eval_lpips_loss",  sum(eval_lpips_losses) / len(eval_lpips_losses), global_step)
            
            if accelerator.is_main_process and exp_helper.should_log_metric(global_step):
                output_images = (pipe_generate(pipe, noise_scheduler, prompt_embeds, sing_view_embed, encoded_inputs) - 0.5) * 2
                if config.log_lpips:
                    lpips_loss = torch.mean(loss_fn_vgg(output_images*loss_mask, ground_truth[0][1]*loss_mask))
                    accelerator.log({"lpips_loss": lpips_loss.item()}, step=global_step)
                    exp_helper.log("lpips_loss",  lpips_loss.item(), global_step)
                del output_images
                del ground_truth
                reclaim_cuda_memory()
                
            reclaim_cuda_memory()

            with torch.no_grad():
                sigma = get_sigma(encoded_inputs, accelerator.device)
                encoded_inputs = torch.randn_like(encoded_inputs) * sigma + encoded_inputs
            
            timesteps, x_noisy, train_target, _flow_sigma = sample_training_target(
                x,
                noise_scheduler,
                training_objective,
                config.num_training_timesteps,
            )
            model.train()

            with accelerator.accumulate(model):
                encoded_inputs = torch.cat([x_noisy, encoded_inputs], dim=1)
                model_pred = model(encoded_inputs,
                                   timesteps,
                                   encoder_hidden_states=prompt_embeds,
                                   class_labels=sing_view_embed,
                                   return_dict=False)[0]

                loss = loss_fn(model_pred, train_target, loss_mask_latent) # TODO
                if torch.isnan(loss):           # loss is a tensor
                    print(f"[STOP] NaN detected on epoch {epoch}. Aborting training.")
                    exit(0)    
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                reclaim_cuda_memory()
                if exp_helper.should_log(global_step):
                    accelerator.log({"train_loss": loss.item()}, step=global_step)
                    accelerator.print({"train_loss": loss.item()})

                if accelerator.is_main_process:
                    if config.ema_mu is not None:
                        ema.update(model)

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            global_step += 1           


if __name__ == '__main__':
    parser_helper = ParserHelper('Inpainter from scratch')
    parser_helper.add_dataclass_flags(ExperimentConfig, 'exp')
    parser_helper.add_dataclass_flags(TrainerConfig, 'train')
    parser_helper.add_dataclass_flags(MultiViewConfig, 'data')
    parser_helper.parser.add_argument('--multi_attention', default=False, action='store_true', help='use multipatch attention or not')
    parser_helper.parser.add_argument('--fine_tune', default=False, action='store_true', help='whether to fine tune model or not')
    args = parser_helper.parse_args()
    
    exp_config = args.exp
    train_config = args.train
    data_config = args.data
    
    data_root_dir = os.path.join(args.global_root_dir, data_config.data_base_dir)
    data_config.num_views = sum(1 for name in os.listdir(data_root_dir) 
                                if os.path.isdir(os.path.join(data_root_dir, name)))
    
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

    accelerator.init_trackers(project_name='gloss-multiview', config=all_configs, 
                              init_kwargs={"wandb":{"group": exp_config.group, "name": exp_config.name, "resume": "allow"}},)
    if args.fine_tune:
        pipeline_components = load_pipeline_components(
            data_config.num_in_channels,
            from_scratch=False,
            objective=train_config.objective,
            flow_shift=train_config.flow_shift,
        )
    else:
        pipeline_components = load_pipeline_components(
            data_config.num_in_channels,
            from_scratch=True,
            objective=train_config.objective,
            flow_shift=train_config.flow_shift,
        )
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
    dataset = configure_multi_view_dataset(data_config, accelerator.device, args.global_root_dir)
    
    if data_config.eval_views < train_config.batch_size:
        data_config.eval_views = train_config.batch_size
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
    print(f"loading {num_filtered_views} dataset items")
    train_dataset, eval_dataset = train_eval_split(
        dataset,
        indices_fp,
        data_config.eval_ratio,
        data_config.eval_views,
        random_seed=data_config.seed,
        mask=dataset_mask,
        global_root_dir=args.global_root_dir,
        mesh_path=data_config.mesh,
    )
    
    view_config = ViewConfig(data_config.mesh, data_config.num_local_views, fov_min=data_config.fov_min,
                             fov_max=data_config.fov_max, camera_dist=data_config.camera_dist)
    if data_config.normal_cond == "geonormal":
        _in_channels = ['geo_camera_normals', 'relative_positions', 'albedo', 'inpaint_mask']
    if data_config.normal_cond == "normal":
        _in_channels = ['camera_normals', 'relative_positions', 'albedo', 'inpaint_mask']
    view_config.input_channels=_in_channels
    eval_config = copy.deepcopy(train_config)
    eval_config.batch_size = train_config.batch_size #TODO
    
    if args.multi_attention:
        print("Using multi attention")
        class MultiAttentionSampler(Sampler):
            def __init__(self, num_local_views, num_views, batch_size):
                self.batch_size = batch_size
                self.num_local_view = num_local_views
                self.num_views = num_views
            def __iter__(self):
                views = torch.tensor(list(range(self.num_views)))
                views = views[torch.randperm(self.num_views)]
                for i in views:
                    start, end = i * self.num_local_view, (i+1) * self.num_local_view
                    indices = torch.tensor(list(range(start, end)))
                    indices = indices[torch.randperm(end - start)]
                    for j in range(0, self.num_local_view, self.batch_size):
                        batch = indices[j:j + self.batch_size]
                        if len(batch) == self.batch_size:
                            yield batch
            def __len__(self):
                return (self.num_local_view // self.batch_size) * self.num_views
        split_dict = load_split_indices(indices_fp)    
        num_train_views = len(split_dict['train_indices'])
        num_eval_views = len(split_dict['eval_indices'])        
        train_sampler = MultiAttentionSampler(data_config.num_local_views, num_train_views, train_config.batch_size)
        eval_sampler = MultiAttentionSampler(data_config.eval_views, num_eval_views, eval_config.batch_size)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=train_sampler, collate_fn=cat_collate_dicts, num_workers=0, pin_memory=False)
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_sampler=eval_sampler, collate_fn=cat_collate_dicts, num_workers=0)
        
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_config.batch_size, shuffle=True, 
            collate_fn=cat_collate_dicts, num_workers=0, pin_memory=False)
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=eval_config.batch_size, shuffle=False, 
            collate_fn=cat_collate_dicts, num_workers=0)

    train_loop(accelerator, model, optimizer, noise_scheduler, train_loader, eval_loader, lr_scheduler, loss_fn, \
        pipe, train_config, view_config, global_root_dir=args.global_root_dir)
    
