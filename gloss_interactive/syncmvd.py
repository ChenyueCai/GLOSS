# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
SyncMVD-style multi-view diffusion inference for the interactive brush.

The default `ReferenceBrush._model_inference` path runs the inpaint model's
self-contained `inpaint(...)` method, which denoises each view of the batch
independently (with sample-wise cross-attention but no UV-space sync). For
strokes that span multiple cameras, that produces per-view results that don't
agree on shared surface regions, leading to seams.

`run_syncmvd_inference` replaces that single call with the same multi-view
fusion loop used in `scripts/completion/complete_stage_1.py`: a shared latent
UV texture is initialized; for each diffusion timestep where multi-view sync
is enabled, the latent UV is rendered through every target camera, the views
are denoised one step, and the per-view predicted-x0 latents are backprojected
into the shared UV with cosine-weight averaging. The latent UV is then DDPM-
stepped from t to t-1. After all timesteps, the per-view final latents are
decoded through the VAE.

Reference patches in the batch participate only in the model's cross-attention
(they're not views of the current mesh, so no UV fusion). Their slots in the
latent batch are filled with the pretrained background latent and never
contribute backprojections.
"""
from typing import Optional, List, Dict
import math

import torch
import kaolin
from diffusers import DDPMScheduler

from gloss.utils.render_fast import custom_mesh_batched_render
from gloss.utils import reclaim_cuda_memory
from gloss.inpaint.syncmvd_utils import composite_rendered_view
from gloss.inpaint.completion_utils import backproject_view_to_texture, CameraConfig


@torch.no_grad()
def run_syncmvd_inference(
    inpaint_model,
    mesh,
    target_cameras: List[kaolin.render.camera.Camera],
    inpaint_data: List[Dict[str, torch.Tensor]],
    num_refs: int,
    camera_config: Optional[CameraConfig] = None,
    num_inference_steps: int = 20,
    multiview_diffusion_end: float = 0.4,
    latent_resolution: int = 512,
    exp_start: float = 0.0,
    exp_end: float = 15.0,
) -> torch.Tensor:
    """Run SyncMVD multi-view diffusion fusion across a batch of target cameras.

    Args:
        inpaint_model: a ``TextureInpaintStandardModel`` (provides ``noise_scheduler``,
            ``process_input``, ``step``, ``background_latent``, ``pipe.vae``).
        mesh: kaolin SurfaceMesh whose UV plane is being painted.
        target_cameras: list of N target cameras to denoise jointly.
        inpaint_data: list of (num_refs + N) channel dicts already in model
            input format. Refs come first, targets last — matching the order
            ``ReferenceBrush.apply_stroke_to_views`` builds them.
        num_refs: how many leading entries of ``inpaint_data`` are reference
            patches (no associated camera; not fused into the latent UV).
        camera_config: optional ``CameraConfig`` for per-pixel backprojection
            angle gating. Defaults to ``CameraConfig()`` (90deg).
        num_inference_steps: DDIM step count (default 20, same as completion).
        multiview_diffusion_end: do multi-view fusion only while
            ``t > (1 - end) * num_train_timesteps``. Default 0.4 = top 60% of
            timesteps (early, high-noise) are synced; later steps are per-view.
        latent_resolution: UV latent texture resolution. Default 512.
        exp_start, exp_end: cosine-weight exponent schedule for backprojection
            averaging. Higher values weight head-on views more strongly.

    Returns:
        Tensor of shape (num_refs + N, 3, view_h, view_w), in [0, 1] — the
        VAE-decoded views for the entire batch. Callers typically slice
        ``[-N:]`` to keep just the target outputs (this matches the
        contract of ``ReferenceBrush._model_inference``).
    """
    if camera_config is None:
        camera_config = CameraConfig()
    device = torch.device("cuda")
    num_targets = len(target_cameras)
    actual_batch_size = num_refs + num_targets

    target_cameras = [c.to(device) for c in target_cameras]
    cond_input = inpaint_model.process_input(inpaint_data).to(device)
    if cond_input.shape[0] != actual_batch_size:
        raise ValueError(
            f"inpaint_data has {cond_input.shape[0]} entries but expected num_refs + len(target_cameras) "
            f"= {num_refs} + {num_targets} = {actual_batch_size}"
        )

    # Init latent UV texture and noise schedulers.
    latent_texture = torch.normal(
        0, 1, (4, latent_resolution, latent_resolution), device=device
    ).permute(1, 2, 0)
    inpaint_model.noise_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = inpaint_model.noise_scheduler.timesteps
    num_train_timesteps = inpaint_model.noise_scheduler.config.num_train_timesteps

    bg_latent = inpaint_model.background_latent(256).float()
    scheduler_config = dict(inpaint_model.noise_scheduler.config)
    scheduler_config["prediction_type"] = "sample"
    latent_noise_scheduler = DDPMScheduler.from_config(scheduler_config)
    latent_noise_scheduler.set_timesteps(num_inference_steps, device=device)

    # Latent-resolution camera batch for the target cameras only.
    # Refs don't render meaningfully into the current mesh's UV so they're
    # held with bg latents and excluded from fusion.
    target_latent_batch = kaolin.render.camera.Camera.cat([c for c in target_cameras])
    # latent feature-map size = camera image size / 8 by SD convention; we use
    # latent_resolution // 16 here = 32 for 512 (matches completion_stage_1).
    latent_view_res = latent_resolution // 16
    target_latent_batch.width = latent_view_res
    target_latent_batch.height = latent_view_res

    latent: Optional[torch.Tensor] = None
    for j, t in enumerate(timesteps):
        do_multi_view_update = t > (1 - multiview_diffusion_end) * num_train_timesteps
        current_exp = ((exp_end - exp_start) * j / num_inference_steps) + exp_start

        if do_multi_view_update:
            # Render the shared latent UV through each target camera.
            r = custom_mesh_batched_render(
                target_latent_batch, mesh, latent_texture,
                requires_positions=False, process_as_albedo=False, backend="cuda",
            )
            target_fg_latents = r["textured"].permute(0, 3, 1, 2)              # (N, 4, lr, lr)
            target_fg_masks = r["mask"].permute(0, 3, 1, 2) / 2 + 0.5            # (N, 1, lr, lr)
            del r

            # Refs occupy the first num_refs slots of the batch; their fg
            # contribution is bg_latent with zero mask (no UV info).
            ref_fg_latents = bg_latent.repeat(num_refs, 1, 1, 1) if num_refs > 0 else None
            ref_fg_masks = (
                torch.zeros((num_refs, 1, latent_view_res, latent_view_res), device=device)
                if num_refs > 0 else None
            )
            if num_refs > 0:
                fg_latents = torch.cat([ref_fg_latents, target_fg_latents], dim=0)
                fg_masks = torch.cat([ref_fg_masks, target_fg_masks], dim=0)
            else:
                fg_latents = target_fg_latents
                fg_masks = target_fg_masks

            if j == 0:
                init_latents = torch.randn((actual_batch_size, 4, latent_view_res, latent_view_res), device=device)
                latent = composite_rendered_view(
                    inpaint_model.noise_scheduler, init_latents, fg_latents, fg_masks, t + 1,
                )
            else:
                prev_t = timesteps[j - 1]
                batch_bg_latents = bg_latent.repeat(actual_batch_size, 1, 1, 1)
                latent = composite_rendered_view(
                    inpaint_model.noise_scheduler, batch_bg_latents, fg_latents, fg_masks, prev_t,
                )
        elif j == 0:
            # First step but multi-view fusion already off — pure noise.
            latent = torch.randn((actual_batch_size, 4, latent_view_res, latent_view_res), device=device)

        # Single denoising step.
        noise_pred = inpaint_model.step(cond_input, latent, t)
        latent, latent_image = inpaint_model.noise_scheduler.step(
            noise_pred, t, latent, return_dict=False,
        )

        # Backproject target predicted-x0 latents into the shared UV with
        # cosine-weight averaging, then DDPM-step the latent UV t -> t-1.
        if do_multi_view_update:
            combined_texture = None
            total_weights = None
            for i, cam in enumerate(target_cameras):
                # backproject_view_to_texture expects view as BHWC.
                view_lat = latent_image[num_refs + i:num_refs + i + 1].permute(0, 2, 3, 1)
                lat_cam = target_latent_batch[i]
                orig_tex, tex_mask, cos_weights = backproject_view_to_texture(
                    lat_cam, view_lat, mesh, latent_texture,
                    camera_config, device, latent=True,
                )
                if cos_weights is None:
                    continue
                if current_exp > 0:
                    min_value = 1e-6 ** (1 / current_exp)
                    cos_weights[cos_weights <= min_value] = min_value
                weights = tex_mask * cos_weights ** current_exp
                if combined_texture is None:
                    total_weights = weights
                    combined_texture = orig_tex[:, :4] * weights
                else:
                    total_weights += weights
                    combined_texture += orig_tex[:, :4] * weights
                del orig_tex, weights, tex_mask, cos_weights
                reclaim_cuda_memory()

            if combined_texture is not None:
                combined_texture /= total_weights + 1e-8
                sample = latent_texture.unsqueeze(0).permute(0, 3, 1, 2)
                prev_tex = latent_noise_scheduler.step(
                    combined_texture, t, sample, return_dict=False,
                )[0]
                latent_texture = prev_tex[0].permute(1, 2, 0)
                del combined_texture, total_weights, prev_tex
                reclaim_cuda_memory()

    # Decode all per-view latents through the VAE.
    final_latents = latent
    if inpaint_model.use_fp16:
        final_latents = final_latents.to(torch.float16)
    view_decoded = inpaint_model.pipe.vae.decode(
        final_latents / inpaint_model.pipe.vae.config.scaling_factor, return_dict=False,
    )[0]
    view_decoded = inpaint_model.pipe.image_processor.postprocess(view_decoded, "pt")
    return view_decoded
