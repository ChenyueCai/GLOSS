# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import kaolin
import logging
import lpips
import random
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn

from gloss.logging import log_tensor
from gloss.utils.render_fast import fast_batched_render
from gloss.utils.kaolin_utils import clone_camera_with_new_resolution

logger = logging.getLogger(__name__)


def diff_project_render(mesh, cameras, rendered_views, initial_textures,
                        texture_gradient_mask=None,
                        augmented_resolutions=[], batch_size=20, super_batch_size=1,
                        simple_loss_fn=torch.nn.MSELoss(),
                        lpips_weight=0,
                        iterations=100, lr=5e-2,
                        scheduler_step_size=20, scheduler_gamma=0.5, quiet=False):
    """
    Back projects single camera view to the object.

    Args:
        mesh:
        camera:
        render_res:
        channels: channels in render_res that should be back-projected
        texture_height:
        texture_width:
        min_pixel_count: minimum pixel count for a face to be included
        max_angle_deviation: maximum angle deviation from perpendicular to camera plane for a face to be included
            (both back and forward normals are accepted, the view just can't be too perpendicular to the
            camera plane)
    Returns:
        dictinoary with results of backprojecting each selected channel
    """

    class IndexDataset(torch.utils.data.Dataset):
        def __init__(self, num):
            self.num = num

        def __len__(self):
            return self.num

        def __getitem__(self, idx):
            return torch.LongTensor([idx])

    resolutions = [rendered_views.shape[-2]] + augmented_resolutions

    # Set up batch randomization
    dataset = IndexDataset(len(cameras))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set up optimization variables
    custom_textures = torch.clone(initial_textures).contiguous()
    custom_textures.requires_grad = True

    # Set optimization parameters
    optim = torch.optim.Adam(params=[custom_textures], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # Set up losses
    if lpips_weight > 0:
        lpips_loss_fn = lpips.LPIPS(net="vgg").to(rendered_views.device)
    else:
        lpips_loss_fn = lambda x, y: 0

    def loss_str(prefix, sl, ll, tl):
        if lpips_weight > 0:
            return "%s: SIMPLE %0.5f, LPIPS %0.5f = TOTAL %0.5f" % (prefix, sl.item(), ll.item(), tl.item())
        else:
            return "%s: %0.5f" % (prefix, tl.item())

    # Optimize
    data_iterator = iter(dataloader)
    for it in range(iterations):
        optim.zero_grad()
        simple_loss = 0
        lpips_loss = 0
        for i in range(super_batch_size):
            try:
                data_idx = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)  # reset iterator
                data_idx = next(data_iterator)

            # Get random batch info
            batch_cameras = [cameras[i] for i in data_idx]
            batch_gt = rendered_views[data_idx, ...].squeeze(1)

            # Pick random resolution
            r = random.randint(0, len(resolutions) - 1)
            res = resolutions[r]

            # Resize camera and image if needed
            if r != 0:
                batch_cameras = [clone_camera_with_new_resolution(c, res) for c in batch_cameras]
                resize = torchvision.transforms.Resize((batch_cameras[0].height, batch_cameras[0].width))
                batch_gt = resize(batch_gt)
            batch_cameras = kaolin.render.camera.Camera.cat(batch_cameras)

            # Let's render with current textures
            if texture_gradient_mask is None:
                input_textures = custom_textures
            else:
                input_textures = (texture_gradient_mask * initial_textures + (
                        1 - texture_gradient_mask) * custom_textures).contiguous()
            batch_res = fast_batched_render(batch_cameras, mesh, input_textures, backend="nvdiffrast")[
                'textured'].permute(0, 3, 1, 2)

            # Let's make sure ranges are sane
            if not quiet and it == 0:
                log_tensor(batch_gt, 'batch_gt', logger, print_stats=True)
                log_tensor(batch_res, 'batch_res', logger, print_stats=True)

            # Get the losses
            simple_loss = simple_loss + simple_loss_fn(batch_gt, batch_res).mean()
            if lpips_weight > 0:
                lpips_loss = lpips_loss + lpips_loss_fn(batch_gt, batch_res).mean()

        # Aggregated loss for the super-batch
        loss = simple_loss + lpips_loss * lpips_weight
        if not quiet and it < 5:
            logger.debug(loss_str(f"It {it}", simple_loss, lpips_loss, loss))

        # Do optimization step
        loss.backward()
        optim.step()
        scheduler.step()
        if not quiet and it % (iterations // 10) == 0 or it == iterations - 1:
            logger.info(loss_str(f"It {it}", simple_loss, lpips_loss, loss))
    return custom_textures

    #
    #     for idx, data_idx in enumerate(dataloader):
    #         optim.zero_grad()
    #
    #         # Get random batch info
    #         batch_cameras = [cameras[i] for i in data_idx]
    #         batch_gt = rendered_views[data_idx, ...].squeeze(1)
    #
    #         # Pick random resolution
    #         r = random.randint(0, len(resolutions) - 1)
    #         res = resolutions[r]
    #
    #         # Resize camera and image if needed
    #         if r != 0:
    #             batch_cameras = [clone_camera_with_new_resolution(c, res) for c in batch_cameras]
    #             resize = torchvision.transforms.Resize((batch_cameras[0].height, batch_cameras[0].width))
    #             batch_gt = resize(batch_gt)
    #         batch_cameras = kaolin.render.camera.Camera.cat(batch_cameras)
    #
    #         # Let's render with current textures
    #         if texture_gradient_mask is None:
    #             input_textures = custom_textures
    #         else:
    #             input_textures = (texture_gradient_mask * initial_textures + (
    #                         1 - texture_gradient_mask) * custom_textures).contiguous()
    #         batch_res = fast_batched_render(batch_cameras, mesh, input_textures, backend="nvdiffrast")[
    #             'textured'].permute(0, 3, 1, 2)
    #
    #         # Let's make sure ranges are sane
    #         if epoch == 0 and idx == 0:
    #             log_tensor(batch_gt, 'batch_gt', logger, print_stats=True)
    #             log_tensor(batch_res, 'batch_res', logger, print_stats=True)
    #
    #         # Get the loss
    #         simple_loss = simple_loss_fn(batch_gt, batch_res)
    #         lpips_loss = lpips_loss_fn(batch_gt, batch_res)
    #         loss = simple_loss + lpips_loss * lpips_weight
    #
    #         if epoch == 0 and idx < 10:
    #             logger.debug(loss_str(f"Loss at the start {idx}", simple_loss, lpips_loss, loss))
    #
    #         # Do optimization step
    #         loss.backward()
    #         optim.step()
    #
    #     scheduler.step()
    #     if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
    #         logger.info(loss_str(f"Epoch {epoch}", simple_loss, lpips_loss, loss))
    # return custom_textures
