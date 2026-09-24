# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import torch
import warp as wp

from torchvision.transforms.v2.functional._misc import _get_gaussian_kernel2d

from gloss.logging import log_tensor
import kaolin

logger = logging.getLogger(__name__)


def sobel_filter(dtype=torch.float32, device="cuda"):
    Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], device=device, dtype=dtype)
    Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], device=device, dtype=dtype)
    return torch.stack([Gx, Gy]).unsqueeze(1)


def sum_filter(kernel_size, dtype=torch.float32, device="cuda"):
    res = torch.ones((kernel_size, kernel_size), dtype=dtype, device=device)
    return res.unsqueeze(0).unsqueeze(0)


def mean_filter(kernel_size, dtype=torch.float32, device="cuda"):
    return sum_filter(kernel_size, dtype, device) / (kernel_size * kernel_size)


def get_imgradient(image):
    """ Computes the norm of the Sobel image gradient for each pixel.

    Args:
        image:

    Returns:

    """
    conv = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)
    conv.weight = torch.nn.Parameter(sobel_filter(image.dtype, image.device), requires_grad=False)
    res = conv(image)
    res = torch.linalg.norm(res, dim=1)
    return res.unsqueeze(1)

def get_patch_mean(image, patch_size):
    """ For each pixel `p` in I, consider a patch around it of size patch_size
        (assume odd patch_size, so the pixel is in the middle). This function computes the mean of this
        patch for every pixel.

        Args:
            image: float image tensor of shape B x 1 x H x W
            patch_size: odd integer for the patch size around each pixel

        Returns:
            tensor of same shape as image, but with 2 channels, holding min and max values for the patch
            around each pixel.
    """
    half_patch_size = int(patch_size // 2)
    conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=patch_size, stride=1,
                           padding=half_patch_size, bias=False, padding_mode='reflect')
    conv.weight = torch.nn.Parameter(mean_filter(patch_size, image.dtype, image.device), requires_grad=False)
    return conv(image)


@wp.kernel
def _tex_bilateral_filter_wp_kernel(image: wp.array4d(dtype=float),
                                    guidance: wp.array4d(dtype=float),
                                    filter: wp.array2d(dtype=float),
                                    output: wp.array4d(dtype=float),
                                    sigma_r_sq: float,
                                    half_patch_size: int):
    nchannels = image.shape[1]
    width = image.shape[3]
    height = image.shape[2]

    # get thread index
    b, i, j = wp.tid()

    # get patch
    i_min = max(0, i - half_patch_size)
    j_min = max(0, j - half_patch_size)
    i_max = min(i + half_patch_size + 1, width)
    j_max = min(j + half_patch_size + 1, height)

    for ch in range(0, nchannels):
        # utility variables
        G_pix = guidance[b, ch, j, i]
        sum_fg = wp.float32(0.0)
        accum_val = wp.float32(0.0)

        # manually compute convolution, given custom kernel for this patch's pixel
        for row in range(j_min, j_max):
            for col in range(i_min, i_max):
                # Gaussian range kernel for this particular patch pixel
                g = wp.exp(-wp.pow(guidance[b, ch, row, col] - G_pix, 2.0) / (2.0 * sigma_r_sq))
                f_val = filter[row - j + half_patch_size, col - i + half_patch_size]
                fg = g * f_val

                # Bilateral texture filter response, accumulated one pixel at a time
                I_val = image[b, ch, row, col]
                accum_val = accum_val + I_val * fg
                sum_fg = sum_fg + fg
        output[b, ch, j, i] = accum_val / sum_fg


def _compute_bilateral_filter_internal(image, guidance_image,
                                       gaussian_filter,
                                       sigma_r,
                                       patch_size):
    assert image.ndim == 4, f'{image.shape}'

    b = image.shape[0]
    height = image.shape[2]
    width = image.shape[3]

    result = torch.zeros_like(image)

    half_patch_size = int(patch_size // 2)
    sigma_r_sq = wp.float32(sigma_r * sigma_r)

    # this fails if warp is not initialized
    try:
        wp_device = wp.device_from_torch(image.device)
    except Exception as e:
        wp.init()
        wp_device = wp.device_from_torch(image.device)

    wp.launch(kernel=_tex_bilateral_filter_wp_kernel,
              dim=(b, width, height),
              inputs=[image, guidance_image, gaussian_filter, result, sigma_r_sq, half_patch_size],
              device=wp_device)
    return result


def bilateral_texture_filter(image, k, iterations):
    """ Preforms an approximate version of bilateral texture filtering, where guidance image G'_p
    in Eq. 5 is replaced by B_p, skipping patch shift computation.

    Args:
        image: B x channels x H x W float32 image
        k: patch_size
        iterations: number of iterations to run filter for

    Returns:
        tensor of the same shape as image
    """
    assert image.ndim == 4, f'{image.shape}'
    b, nchannels, height, width = image.shape

    # Defaults from paper, except sigma_r is set at half the size
    filter_patch_size = 2 * k - 1
    sigma_s = k - 1
    sigma_r = 0.025 * math.sqrt(nchannels)

    f = _get_gaussian_kernel2d([filter_patch_size, filter_patch_size], [sigma_s, sigma_s], dtype=image.dtype, device=image.device)

    for it in range(iterations):
        B = torch.cat([get_patch_mean(image[:, ch:ch + 1, ...], k) for ch in range(nchannels)], dim=1)
        # Note, we skip computing guidance using patch shift, as this doesn't make a huge difference for
        # conditional image applications
        G_prime = B
        image = _compute_bilateral_filter_internal(image, G_prime, f, sigma_r, filter_patch_size)

    return image