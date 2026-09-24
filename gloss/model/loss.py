# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import torch
import torch.nn.functional as F


DIFFUSION_OBJECTIVE = "diffusion"
FLOW_MATCHING_OBJECTIVE = "flow_matching"


def canonicalize_training_objective(objective):
    if objective is None:
        return DIFFUSION_OBJECTIVE

    normalized = objective.strip().lower().replace("-", "_")
    aliases = {
        "diffusion": DIFFUSION_OBJECTIVE,
        "ddpm": DIFFUSION_OBJECTIVE,
        "v_prediction": DIFFUSION_OBJECTIVE,
        "flow_matching": FLOW_MATCHING_OBJECTIVE,
        "flowmatch": FLOW_MATCHING_OBJECTIVE,
        "flow": FLOW_MATCHING_OBJECTIVE,
        "fm": FLOW_MATCHING_OBJECTIVE,
    }
    if normalized not in aliases:
        raise ValueError(
            f"Unsupported training objective '{objective}'. "
            f"Expected one of {sorted(aliases)}."
        )
    return aliases[normalized]


def masked_mse_loss(prediction, target, loss_mask):
    return F.mse_loss(prediction * loss_mask, target * loss_mask)


def masked_weighted_mse_loss(prediction, target, loss_mask, sample_weights):
    """MSE loss with per-sample weights. Scale matches masked_mse_loss when weights are ones."""
    masked_diff = (prediction - target) * loss_mask
    per_sample = masked_diff.pow(2).mean(dim=list(range(1, masked_diff.dim())))
    weights = sample_weights.to(per_sample.dtype).reshape(per_sample.shape)
    return (per_sample * weights).mean()


def compute_snr(noise_scheduler, timesteps):
    """Signal-to-noise ratio alpha_t^2 / (1 - alpha_t^2) for a DDPM-style scheduler."""
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
    idx = timesteps.long().clamp(min=0, max=alphas_cumprod.numel() - 1)
    alpha_bar = alphas_cumprod[idx]
    snr = alpha_bar / (1.0 - alpha_bar).clamp(min=1e-8)
    return snr


def compute_min_snr_v_weights(noise_scheduler, timesteps, objective, gamma, *, sigma=None):
    """Per-sample min-SNR-gamma weights for a v-prediction / flow-matching target.

    For DDPM v-prediction: w = min(snr, gamma) / (snr + 1).
    For flow matching with x_t = (1 - sigma) x_0 + sigma * noise, we treat
    snr = ((1 - sigma) / sigma)^2 and apply the same formula.
    """
    objective = canonicalize_training_objective(objective)
    if objective == DIFFUSION_OBJECTIVE:
        snr = compute_snr(noise_scheduler, timesteps)
    else:
        if sigma is None:
            raise ValueError("sigma is required to compute min-SNR weights for flow matching")
        flat_sigma = sigma.reshape(sigma.shape[0]).to(torch.float32).clamp(min=1e-5, max=1.0 - 1e-5)
        snr = ((1.0 - flat_sigma) / flat_sigma).pow(2)
    return torch.clamp(snr, max=float(gamma)) / (snr + 1.0)


def sample_training_target(clean_latents, scheduler, objective, num_training_timesteps, *,
                           max_timestep_fraction=1.0):
    objective = canonicalize_training_objective(objective)
    if max_timestep_fraction <= 0 or max_timestep_fraction > 1.0:
        raise ValueError(f"max_timestep_fraction must be in (0, 1], got {max_timestep_fraction}")

    batch_size = clean_latents.shape[0]
    device = clean_latents.device
    noise = torch.randn_like(clean_latents)

    if objective == DIFFUSION_OBJECTIVE:
        max_timestep = max(1, int(num_training_timesteps * max_timestep_fraction))
        timesteps = torch.randint(0, max_timestep, (batch_size,), device=device, dtype=torch.int64)
        noisy_latents = scheduler.add_noise(clean_latents, noise, timesteps)
        target = scheduler.get_velocity(clean_latents, noise, timesteps)
        return timesteps, noisy_latents, target, None

    sigma_max = min(max_timestep_fraction, 1.0)
    sigma = torch.rand((batch_size,), device=device, dtype=clean_latents.dtype) * sigma_max
    sigma = sigma.clamp(min=1e-5, max=min(1.0 - 1e-5, sigma_max))
    sigma_view = sigma.view(batch_size, *([1] * (clean_latents.dim() - 1)))
    noisy_latents = (1.0 - sigma_view) * clean_latents + sigma_view * noise
    target = noise - clean_latents
    timesteps = sigma * float(num_training_timesteps)
    return timesteps, noisy_latents, target, sigma


def reconstruct_x0_from_prediction(noisy_latents, prediction, scheduler, timesteps, objective, *,
                                   sigma=None):
    objective = canonicalize_training_objective(objective)
    if objective == DIFFUSION_OBJECTIVE:
        pred_x0 = torch.zeros_like(noisy_latents)
        for idx in range(noisy_latents.shape[0]):
            pred_x0[idx] = scheduler.step(
                prediction[idx],
                timesteps[idx],
                noisy_latents[idx],
                return_dict=False,
            )[1]
        return pred_x0

    if sigma is None:
        raise ValueError("sigma is required to reconstruct x0 for flow matching")

    sigma_view = sigma.view(sigma.shape[0], *([1] * (noisy_latents.dim() - 1)))
    return noisy_latents - sigma_view * prediction


def encode_cond(vae, raw_conds, vae_scale_factor):
    conds = []
    for cond in raw_conds:
        conds.append(cond[1])
    conds = torch.cat(conds, dim=1)
    encoding = vae.encode(conds, return_dict=False)[0].sample() / vae_scale_factor
    return encoding


def vae_reg_loss(vae, batch, vae_scale_factor, data_config):
    l1_loss = torch.nn.L1Loss()
    raw_inputs = []
    for channel in data_config.input_channels:
        if channel not in data_config.output_channels:
            raw_inputs.append((channel, batch[channel]))
    raw_reg_inputs = copy.copy(raw_inputs)
    albedo_alpha = batch["albedo_alpha"] / 2 + 0.5
    albedo = batch["albedo"]
    B, C, _, _ = albedo.shape
    c1 = torch.rand(B, C, 1, 1, dtype=albedo.dtype, device=albedo.device) * 2.0 - 1.0
    c2 = torch.rand(B, C, 1, 1, dtype=albedo.dtype, device=albedo.device) * 2.0 - 1.0
    raw_inputs.append(("albedo", albedo * albedo_alpha + c1 * (1 - albedo_alpha)))
    raw_reg_inputs.append(("albedo", albedo * albedo_alpha + c2 * (1 - albedo_alpha)))
    encoding = encode_cond(vae, raw_inputs, vae_scale_factor)
    encoding_1 = encode_cond(vae, raw_reg_inputs, vae_scale_factor)
    return l1_loss(encoding, encoding_1)
