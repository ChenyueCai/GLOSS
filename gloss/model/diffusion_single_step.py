# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Portions adapted from img2img-turbo
# Source files: https://github.com/GaParmar/img2img-turbo/blob/main/src/pix2pix_turbo.py, https://github.com/GaParmar/img2img-turbo/blob/main/src/model.py
# Source licensed under MIT License
#
# Copyright (c) 2024 img-to-img-turbo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler


def make_1step_sched(model_card):
    noise_scheduler_1step = DDPMScheduler.from_pretrained(model_card, subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


class SingleStepLDM(torch.nn.Module):
    def __init__(self, vae_in_channels, latent_channels=4, skip_connection=False,
                 model_card="stabilityai/stable-diffusion-2-1"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_card, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_card, subfolder="text_encoder")

        vae_config = AutoencoderKL.load_config(model_card, subfolder="vae", cache_dir=".cache")
        vae_config["in_channels"] = vae_in_channels
        vae_config["latent_channels"] = latent_channels
        self.vae = AutoencoderKL.from_config(vae_config)
        self.skip_connection = skip_connection
        if skip_connection:
            self.vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
            self.vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
            # add the skip connection convs
            self.vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.vae.decoder.ignore_skip = False

        unet_config = UNet2DConditionModel.load_config(model_card, subfolder="unet", cache_dir=".cache")
        unet_config["in_channels"] = latent_channels
        self.unet = UNet2DConditionModel.from_config(unet_config)

        self.timesteps = torch.tensor([1], device="cuda").long()
        self.sched = make_1step_sched(model_card)

        self.vae.decoder.gamma = 1

        self.text_encoder.requires_grad_(False)

    def encode_prompt(self, prompt):
        caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                                        padding="max_length", truncation=True, return_tensors="pt"
                                        ).input_ids.to(self.text_encoder.device)
        caption_enc = self.text_encoder(caption_tokens)[0]
        return caption_enc

    def forward(self, x, prompt_embeds):
        # try one encoder for everything first
        z = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
        unet_input = z

        model_pred = self.unet(unet_input,
                               timestep=self.timesteps,
                               encoder_hidden_states=prompt_embeds,
                               ).sample

        z_denoised = self.sched.step(model_pred, self.timesteps[0], unet_input, return_dict=True).prev_sample
        z_denoised = z_denoised.to(model_pred.dtype)
        if self.skip_connection:
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        output_image = (self.vae.decode(z_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image

    def get_trainable_params(self):
        trainable_params = []
        trainable_params += list(self.unet.parameters())
        trainable_params += list(self.vae.parameters())
        return trainable_params

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.unet.requires_grad_(True)
        self.vae.requires_grad_(True)
