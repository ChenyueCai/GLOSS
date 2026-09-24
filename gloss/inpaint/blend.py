# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import numpy as np
import cv2

##############################################################################
# Helper functions for format conversions
##############################################################################

def numpy_to_torch(img_np: np.ndarray, device='cpu') -> torch.Tensor:
    """
    Convert a NumPy image of shape (H,W,3) in [0..255] to a PyTorch float tensor
    of shape (1,3,H,W) in [0..1].
    """
    # Ensure float32
    img_float = img_np.astype(np.float32) / 255.0
    # Reorder HWC -> CHW
    img_chw = np.transpose(img_float, (2, 0, 1))
    # Add batch dimension -> (1,3,H,W)
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)
    return img_tensor

def torch_to_numpy(img_t: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch float tensor of shape (1,3,H,W) in [0..1] to NumPy
    image of shape (H,W,3) in [0..255].
    """
    # Clamp/normalize
    img_t = img_t.clamp(0.0, 1.0)
    # Remove batch dim => (3,H,W)
    img_chw = img_t.squeeze(0).cpu().detach().numpy()
    # Transpose => (H,W,3)
    img_hwc = np.transpose(img_chw, (1, 2, 0))
    img_np = (img_hwc * 255.0).round().astype(np.uint8)
    return img_np

def mask_numpy_to_torch(mask_np: np.ndarray, device='cpu') -> torch.Tensor:
    """
    Convert a NumPy mask (H,W) of 0/1 to a PyTorch float tensor (1,1,H,W).
    """
    mask = mask_np.astype(np.float32)
    # Add batch dim and channel dim
    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
    return mask_t

##############################################################################
# 1) ALPHA (LINEAR) BLENDING IN PYTORCH
##############################################################################

def alpha_blend(
    img1_t: torch.Tensor, mask1_t: torch.Tensor,
    img2_t: torch.Tensor, mask2_t: torch.Tensor,
    alpha: float = 0.5
) -> torch.Tensor:
    """
    Perform a simple alpha (linear) blending in the overlap region:
        out = alpha*img1 + (1-alpha)*img2
    where both mask1 and mask2 are 1.

    Inputs:
    -------
    img1_t, img2_t : (1, 3, H, W) float in [0..1]
    mask1_t, mask2_t : (1, 1, H, W) float in {0,1}
    alpha : float in [0..1]

    Returns:
    --------
    merged : (1, 3, H, W) float in [0..1]
    """
    # We will construct a final output that merges according to the masks
    merged = torch.zeros_like(img1_t)

    # region1 = mask1 & not mask2
    region1 = (mask1_t > 0.5) & (mask2_t < 0.5)
    # region2 = mask2 & not mask1
    region2 = (mask2_t > 0.5) & (mask1_t < 0.5)
    # overlap = mask1 & mask2
    overlap = (mask1_t > 0.5) & (mask2_t > 0.5)

    merged[region1.expand_as(img1_t)] = img1_t[region1.expand_as(img1_t)]
    merged[region2.expand_as(img2_t)] = img2_t[region2.expand_as(img2_t)]
    # Overlap => alpha blend
    merged[overlap.expand_as(img1_t)] = (
        alpha * img1_t[overlap.expand_as(img1_t)]
        + (1 - alpha) * img2_t[overlap.expand_as(img1_t)]
    )
    return merged, torch.logical_or(mask1_t, mask2_t)

##############################################################################
# 2) LAPLACIAN (MULTI-BAND) PYRAMID BLENDING IN PYTORCH
##############################################################################

def pyramid_downsample(x: torch.Tensor) -> torch.Tensor:
    """
    Naive "Gaussian" downsample using bilinear interpolation.
    Input: (1, C, H, W)
    Output: (1, C, H/2, W/2)
    """
    return F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

def pyramid_upsample(x: torch.Tensor, sizeHW: tuple) -> torch.Tensor:
    """
    Upsample to a specified (H,W) using bilinear interpolation.
    """
    return F.interpolate(x, size=sizeHW, mode='bilinear', align_corners=False)

def build_gaussian_pyramid(x: torch.Tensor, levels=5):
    """
    Build a Gaussian pyramid (list) of length 'levels+1'.
    x: (1, C, H, W)
    """
    pyramid = [x]
    current = x
    for _ in range(levels):
        current = pyramid_downsample(current)
        pyramid.append(current)
    return pyramid

def build_laplacian_pyramid(x: torch.Tensor, levels=5):
    """
    Build a Laplacian pyramid from x.
    x: (1, C, H, W)
    Return: list of length 'levels+1'
    """
    g_pyr = build_gaussian_pyramid(x, levels=levels)
    l_pyr = []
    for i in range(levels):
        # shape of g_pyr[i]: (1, C, H_i, W_i)
        # shape of g_pyr[i+1]: (1, C, H_{i+1}, W_{i+1})
        up = pyramid_upsample(g_pyr[i+1], (g_pyr[i].shape[2], g_pyr[i].shape[3]))
        lap = g_pyr[i] - up
        l_pyr.append(lap)
    # The topmost level is just the last Gaussian
    l_pyr.append(g_pyr[-1])
    return l_pyr

def collapse_laplacian_pyramid(l_pyr):
    """
    Reconstruct a (1, C, H, W) image from a laplacian pyramid l_pyr.
    """
    levels = len(l_pyr) - 1
    current = l_pyr[-1]
    for i in range(levels - 1, -1, -1):
        up = pyramid_upsample(current, (l_pyr[i].shape[2], l_pyr[i].shape[3]))
        current = l_pyr[i] + up
    return current

def laplacian_blend(
    img1_t: torch.Tensor, mask1_t: torch.Tensor,
    img2_t: torch.Tensor, mask2_t: torch.Tensor,
    levels: int = 5
) -> torch.Tensor:
    """
    Multi-band (Laplacian) pyramid blending in PyTorch.

    Steps:
     1) Build Laplacian pyramids for img1 and img2.
     2) Build Gaussian pyramids for 'weight' = mask1 / (mask1+mask2).
     3) Blend laplacian levels using L = w * L1 + (1-w) * L2 (elementwise).
     4) Collapse blended pyramid.

    Inputs:
    -------
    img1_t, img2_t : (1,3,H,W) float in [0..1]
    mask1_t, mask2_t : (1,1,H,W) float in {0,1}
    levels : how many pyramid levels

    Returns:
    --------
    blended : (1,3,H,W)
    """
    # Compute float masks
    eps = 1e-7
    m1 = (mask1_t > 0.5).float()
    m2 = (mask2_t > 0.5).float()

    # weight1 = m1 / (m1 + m2 + eps)
    # But we need shape matching to (1,3,H,W). Let's just expand along channel dim.
    denom = m1 + m2 + eps
    w1 = m1 / denom  # shape (1,1,H,W)
    w2 = m2 / denom  # shape (1,1,H,W)

    # Expand to 3 channels
    w1_3c = w1.repeat(1, 4, 1, 1)  # (1,3,H,W)
    w2_3c = w2.repeat(1, 4, 1, 1)

    # Build Laplacian pyramids
    lap1 = build_laplacian_pyramid(img1_t, levels=levels)
    lap2 = build_laplacian_pyramid(img2_t, levels=levels)

    # Build Gaussian pyramids for w1, w2
    gauss_w1 = build_gaussian_pyramid(w1_3c, levels=levels)
    gauss_w2 = build_gaussian_pyramid(w2_3c, levels=levels)

    blended_pyr = []
    for i in range(levels+1):
        L1 = lap1[i]
        L2 = lap2[i]
        W1 = gauss_w1[i]
        W2 = gauss_w2[i]
        # Blend Laplacian levels
        blended_level = L1 * W1 + L2 * W2
        blended_pyr.append(blended_level)

    # Collapse pyramid
    blended = collapse_laplacian_pyramid(blended_pyr)
    blended = torch.clamp(blended, 0.0, 1.0)
    return blended, torch.logical_or(mask1_t, mask2_t)


##############################################################################
# 3) POISSON (GRADIENT-DOMAIN) BLENDING IN PYTORCH
##############################################################################
def poisson_blend(
    img1_t: torch.Tensor, mask1_t: torch.Tensor,
    img2_t: torch.Tensor, mask2_t: torch.Tensor,
    num_iterations: int = 1000,
    lr: float = 0.1
) -> torch.Tensor:
    """
    A *simplified* gradient-domain blending approach in PyTorch. It attempts
    to minimize the difference in gradients between the "source region" and
    the "target region" for overlapping pixels, while preserving the known
    outside region as constraints.

    For real usage on large images, consider a dedicated Poisson solver (OpenCV's
    seamlessClone or a specialized PDE method). This is just a demonstration of
    how one *could* do it in PyTorch by formulating it as an optimization problem.

    We'll treat 'img1' as the "source" for overlap, and 'img2' as the "target."
    The goal: In the overlap region, we want the final output's gradient to match
    img1's gradient, but outside that region, it should match img2 exactly.

    Algorithm:
      1) Start with output = img2.clone()  (the "target" base)
      2) For pixels in overlap = mask1 & mask2, we define a loss that tries to
         match the x-gradient & y-gradient to those of img1.
      3) Outside overlap, we fix the output to match img2 exactly (so no update
         there).
      4) We do a simple gradient descent in PyTorch.

    Inputs:
    -------
    img1_t, img2_t : (1,3,H,W) in [0..1]
    mask1_t, mask2_t : (1,1,H,W) in {0,1}
    num_iterations : number of optimization steps
    lr : learning rate for the gradient descent

    Returns:
    --------
    output_t : (1,3,H,W) in [0..1] blended result
    """
    # Convert overlap mask to float
    overlap_mask = ((mask1_t > 0.5) & (mask2_t > 0.5)).float()  # (1,1,H,W)
    # We'll expand to 3 channels
    overlap_mask_3c = overlap_mask.repeat(1, 4, 1, 1)  # (1,3,H,W)

    # We create a clone of img2 as our *variable* to optimize
    # We'll do requires_grad=True so we can do gradient-based updates.
    output = img2_t.clone().detach()
    output.requires_grad = True

    # Precompute gradient of img1 in x,y directions
    # We'll do a finite difference. For simplicity, we skip boundaries or wrap them.
    # shape: (1,3,H,W)
    def gradient_x(img):
        return img[:, :, :, 1:] - img[:, :, :, :-1]  # (1,3,H,W-1)

    def gradient_y(img):
        return img[:, :, 1:, :] - img[:, :, :-1, :]  # (1,3,H-1,W)

    grad1_x = gradient_x(img1_t)
    grad1_y = gradient_y(img1_t)

    optimizer = torch.optim.SGD([output], lr=lr, momentum=0.0)

    for _ in range(num_iterations):
        optimizer.zero_grad()

        # Compute gradient of current output in overlap
        out_grad_x = gradient_x(output)
        out_grad_y = gradient_y(output)

        # We only care about overlapping region => slice them accordingly
        # Overlap in the gradient domain => slightly smaller if we think about edges, but
        # for simplicity we'll just multiply by the overlap mask except at boundary differences.
        # Let's define "mask_x" for overlap except rightmost column, "mask_y" except bottom row:
        mask_x = overlap_mask_3c[:, :, :, 1:]  # shape (1,3,H,W-1)
        mask_y = overlap_mask_3c[:, :, 1:, :]  # shape (1,3,H-1,W)

        # L2 loss on gradient difference
        loss_x = ((out_grad_x - grad1_x) * mask_x).pow(2).mean()
        loss_y = ((out_grad_y - grad1_y) * mask_y).pow(2).mean()

        # We also add a "hard constraint" for pixels outside mask1 or inside mask2=0
        # Actually, if mask2=1 but mask1=0 => we want output = img2. We can do a penalty:
        # penalty = (output - img2_t)^2 * region
        # But in pure Poisson you'd fix boundary condition exactly. 
        # For demonstration, let's do a big penalty that strongly enforces output=img2
        no_overlap_mask_3c = (mask2_t < 0.5).repeat(1, 4, 1, 1)
        constraint_loss = ((output - img2_t) * no_overlap_mask_3c).pow(2).mean()

        total_loss = loss_x + loss_y + 1000.0 * constraint_loss
        total_loss.backward()

        optimizer.step()

    output_t = torch.clamp(output, 0.0, 1.0)
    return output_t, torch.logical_or(mask1_t, mask2_t)


##############################################################################
# DEMO / USAGE
##############################################################################

if __name__ == "__main__":
    # Example usage: Suppose we have two NumPy images, same size (H,W,3),
    # and two masks (H,W) of 0/1. We'll just generate synthetic data here.

    H, W = 200, 300
    # Synthetic image1 = white, image2 = black
    img1_np = np.ones((H, W, 3), dtype=np.uint8) * 255
    img2_np = np.ones((H, W, 3), dtype=np.uint8) * 155
    # Synthetic masks: left half for img1, right half for img2
    mask1_np = np.zeros((H, W), dtype=np.uint8)
    mask1_np[:, :W//2] = 1
    mask2_np = np.zeros((H, W), dtype=np.uint8)
    mask2_np[:, W//2:] = 1

    device = 'cpu'
    # Convert to PyTorch
    t1 = numpy_to_torch(img1_np, device=device)
    t2 = numpy_to_torch(img2_np, device=device)
    m1 = mask_numpy_to_torch(mask1_np, device=device)
    m2 = mask_numpy_to_torch(mask2_np, device=device)

    # 1) Alpha blend
    out_alpha = alpha_blend(t1, m1, t2, m2, alpha=0.5)
    out_alpha_np = torch_to_numpy(out_alpha)
    # Save or visualize as needed, e.g. cv2.imwrite("alpha_result.png", out_alpha_np)

    # 2) Laplacian pyramid blend
    out_lap = laplacian_blend(t1, m1, t2, m2, levels=4)
    out_lap_np = torch_to_numpy(out_lap)
    # e.g. cv2.imwrite("laplacian_result.png", out_lap_np)

    # 3) Poisson (gradient-domain) blend (demonstration)
    out_poisson = poisson_blend(t1, m1, t2, m2, num_iterations=500, lr=0.2)
    out_poisson_np = torch_to_numpy(out_poisson)
    # e.g. 
    print(out_poisson_np.max())
    cv2.imwrite("poisson_result.png", out_poisson_np)

    print("Done blending demos in PyTorch.")
