# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# set_attn_processor is adapted from diffusers UNet2DConditionModel class
# Source file: https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/models/unets/unet_2d_condition.py#L722
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from diffusers import UNet2DModel, DDIMScheduler, DDIMPipeline


def pipe_generate(pipe:DDIMPipeline, scheduler, conds, num_inference_steps=20):
    with torch.no_grad():
        conds = conds
        scheduler.set_timesteps(num_inference_steps, device=conds.device)
        timesteps = scheduler.timesteps
        latents = torch.randn((conds[:, :3, :, :].shape), device=conds.device) # TODO: set channel num somewhere
        #TODO: add classifier free guidance
        for i, t in enumerate(pipe.progress_bar(timesteps)):
            latent_model_input = torch.cat([conds, latents], dim=1)
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                return_dict=False,
            )[0]
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        images = latents
        return images


def set_attn_processor(model, processor):
    # only add to mid_block otherwise model will be very slow
    def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
        if hasattr(module, "set_processor"):
            if "mid_block" in name:
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

    for name, module in model.named_children():
        fn_recursive_attn_processor(name, module, processor())
