# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import cv2
from PIL import Image


def tensor_to_image(tensor):
    """Convert a PyTorch tensor (C, H, W) in [0, 1] to uint8 RGB image (H, W, C)."""
    image = tensor.detach().cpu().numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))  # (H, W, C)
    return image


def image_to_tensor(image):
    """Convert RGB uint8 image (H, W, C) to PyTorch tensor (C, H, W) in [0, 1]."""
    image = np.transpose(image, (2, 0, 1))  # (C, H, W)
    tensor = torch.from_numpy(image).float() / 255.0
    return tensor


def reinhard_color_transfer(original_tensor, inpainted_tensor, mask_tensor):
    original_img = tensor_to_image(original_tensor)
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(bool)

    lab = cv2.cvtColor(original_img, cv2.COLOR_RGB2LAB)
    correction_target = lab[mask_np]

    inpainted_img = tensor_to_image(inpainted_tensor)

    lab = cv2.cvtColor(inpainted_img, cv2.COLOR_RGB2LAB)
    inpaint_source = lab[mask_np]

    mean_orig, std_orig = correction_target.mean(0), correction_target.std(0)
    mean_out, std_out = inpaint_source.mean(0), inpaint_source.std(0)

    eps = 1e-5
    mean_diff = np.linalg.norm(mean_out - mean_orig)
    std_diff = np.linalg.norm(std_out - std_orig)
    if mean_diff < 20 and std_diff < 20 and std_out.mean() > 5.0 and std_orig.mean() > 5.0:
        corrected_lab = ((lab - mean_out) / (std_out + eps)) * std_orig + mean_orig
    else:
        corrected_lab = lab
    corrected_lab = corrected_lab.astype(np.uint8)

    corrected_rgb = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB)
    corrected_rgb = np.clip(corrected_rgb, 0, 255).astype(np.uint8)

    return image_to_tensor(corrected_rgb)


def polynomial_color_transfer(original_tensor, inpainted_tensor, mask_tensor, degree=2):
    """
    Computes and applies a polynomial color transformation from source to target RGB colors.
    based on A. Ilie and G. Welch. Ensuring color consistency across multiple cameras. In ICCV, pages 1268–1275, 2005. 2, 5, 7
    """
    original_img = tensor_to_image(original_tensor)
    inpainted_img = tensor_to_image(inpainted_tensor)
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(bool)

    source_colors = inpainted_img[mask_np]
    target_colors = original_img[mask_np]
    h, w, c = inpainted_img.shape
    apply_to = inpainted_img.reshape(-1, c)

    # Normalize RGBs to [0, 1] for stability (optional)
    scale = 255.0
    source_colors = source_colors.astype(np.float32) / scale
    target_colors = target_colors.astype(np.float32) / scale
    apply_to_scaled = apply_to.astype(np.float32) / scale

    def build_polynomial_features(rgb, degree):
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]  # Shape: (N, 1)
        powers = np.arange(1, degree + 1).reshape(1, -1)  # Shape: (1, D)
        r_feats = r ** powers  # (N, D)
        g_feats = g ** powers  # (N, D)
        b_feats = b ** powers  # (N, D)
        features = np.concatenate([r_feats, g_feats, b_feats, np.ones_like(r)], axis=1)  # (N, 3D+1)
        return features

    # Build feature matrix
    X = build_polynomial_features(source_colors, degree)     # (N, D)
    Y = target_colors                                         # (N, 3)

    # Fit least-squares transform: Y ≈ X @ M0.T
    M0_T, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    M0 = M0_T.T                                               # (3, D)

    # Apply to new data
    X_new = build_polynomial_features(apply_to_scaled, degree)  # (M, D)
    Y_new = (M0 @ X_new.T).T                                     # (M, 3)

    Y_new = np.clip(Y_new * scale, 0, scale).astype(np.uint8)
    Y_new = Y_new.reshape(h, w, c)
    return image_to_tensor(Y_new)


def compute_color_histogram(image, mask, num_bins=25):
    # image shape: [H, W, 3], mask shape: [H, W, 1]
    hist = []
    for i in range(3):  # for each channel
        channel = image[..., i]
        masked_channel = channel[mask[..., 0] > 0]
        if len(masked_channel) == 0:
            hist.append(torch.zeros(num_bins).to(image.device))
            continue
        hist_channel = torch.histc(masked_channel, bins=num_bins, min=-1.0, max=1.0)
        hist_channel = hist_channel / (hist_channel.sum() + 1e-6)  # normalize
        hist.append(hist_channel)
    hist = torch.cat(hist)
    return hist