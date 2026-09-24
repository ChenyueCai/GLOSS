# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import numpy as np
import skimage.segmentation
import skimage.util
import skimage.color
import torch
import torchvision

logger = logging.getLogger(__name__)


def posterization_adaptive_default_args(torch_img=None, resolution=None):
    assert (torch_img is None) != (resolution is None), f'must provide exactly one of torch_img, resolution'

    if torch_img is not None:
        assert isinstance(torch_img, torch.Tensor)
        assert torch_img.ndim == 3
        assert torch_img.shape[-1] == 3

        max_dim = max(torch_img.shape[:2])
    elif resolution is not None:
        max_dim = int(resolution)

    blur_kernel = max_dim // 50
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    return {"blur_kernel": blur_kernel,
            "blur_sigma": max_dim / 100,
            "n_superpixels": 400 if max_dim > 256 else 100}


def posterize_image(torch_img, blur_kernel=9, blur_sigma=6.0, n_superpixels=400, max_lab_distance=10.0, debug_output=False):
    """Posterizes the input image using a combination of superpixels, farthest point sampling in LAB space and
    clustering. Rather naive, but relatively fast and better than off-the-shelf variant available in Pillow.

    Args:
        torch_img: float image array W x H x 3, 0..1
        blur_kernel: set to None to skip gaussian blur of the input; else set to kernel size
        blur_sigma: used for gaussian blur of input if blur_kernel is not None
        n_superpixels: number of superpixels to use when preprocessing image
        max_lab_distance: max distance in CIELAB color space to use when clustering (~10 is usually a good value)
        debug_output: if extra debug output should be created for visualization

    Returns:
        (posterized_image, optional extra data dict)

    """
    assert isinstance(torch_img, torch.Tensor)
    assert torch_img.ndim == 3
    assert torch_img.shape[-1] == 3
    assert torch_img.min() >= 0
    assert torch_img.max() <= 1
    device = torch_img.device

    # Blur (optional)
    if blur_kernel is not None:
        torch_img = torchvision.transforms.GaussianBlur(blur_kernel, sigma=blur_sigma)(torch_img.permute(2, 0, 1)).permute(1, 2, 0)

    # Create superpixels
    img = skimage.util.img_as_float(torch_img.cpu().numpy())
    segments_slic = skimage.segmentation.slic(img, n_segments=n_superpixels, compactness=10, sigma=1, start_label=0)
    segment_labels = np.unique(segments_slic)
    segments_torch = torch.from_numpy(segments_slic).to(device)

    if debug_output:
        segments_viz = torch.from_numpy(skimage.segmentation.mark_boundaries(img, segments_slic))
        segments_viz_mean = torch.zeros_like(torch_img)   # mean colors visualized

    # For each superpixel, compute mean color, stdev
    seg_colors = torch.zeros((len(segment_labels), 3), dtype=torch.float32, device=device)
    seg_std = torch.zeros((len(segment_labels), 1), dtype=torch.float32, device=device)
    for seg in segment_labels:
        mean_color = torch_img[segments_torch == seg, :].mean(dim=0)
        seg_colors[seg, :] = mean_color
        seg_std[seg, :] = torch_img[segments_torch == seg, :].std(dim=0).max()
        if debug_output:
            segments_viz_mean[segments_torch == seg, :] = mean_color.unsqueeze(0)

    # Sort segments from least to most standard deviation
    sorted_segs = [x[1] for x in sorted([(seg_std[s], s) for s in segment_labels])]

    # Convert segment color to Lab space
    lab_seg_colors = torch.from_numpy(skimage.color.rgb2lab(seg_colors.cpu().numpy())).to(device)

    # Run clustering (furthest point sampling with simple clustering)
    lab_selected_colors = None  # cluster colors
    rgb_selected_colors = None
    seg_selected_color_idx = torch.zeros((len(sorted_segs),), dtype=torch.long, device=device) # cluster color index per segment
    processed = torch.zeros((len(segment_labels),), dtype=torch.bool, device=device)  # which segments got processed
    segments_viz_post = torch.zeros_like(torch_img)  # final posterization result
    min_distances_to_cluster = None  # for each non-processed segment, what is the distance to closest cluster
    seg_to_process = sorted_segs[0] # segment to start processing from

    # We cap number of iterations to avoid accidental infinite loop (in practice will take way fewer iterations)
    for idx in range(len(segment_labels)):
        if processed.sum() >= len(segment_labels):
            break  # Done

        seg = seg_to_process
        assert not processed[seg], f'Bug: seg_to_process {seg} should be guaranteed to not have been processed yet'
        color = lab_seg_colors[seg].unsqueeze(0)
        rgb_color = seg_colors[seg].unsqueeze(0)

        if lab_selected_colors is None:  # first cluster
            new_color_idx = 0
            seg_selected_color_idx[seg] = new_color_idx
            lab_selected_colors = color
            rgb_selected_colors = rgb_color
        else:
            new_color_idx = int(lab_selected_colors.shape[0])
            seg_selected_color_idx[seg] = new_color_idx
            lab_selected_colors = torch.cat([lab_selected_colors, color], dim=0)
            rgb_selected_colors = torch.cat([rgb_selected_colors, rgb_color], dim=0)

        # Now we'll process all segments within delta of this one (compute distance in LAB, more perceptually uniform)
        distances = torch.linalg.norm(lab_seg_colors - color, dim=1)

        # Add all segments close enough in color to the current color cluster
        close_enough = torch.logical_and(distances < max_lab_distance, torch.logical_not(processed))
        seg_selected_color_idx[close_enough] = new_color_idx

        # Update cluster color to be the average of these selected segments
        mean_color = lab_seg_colors[close_enough].mean(dim=0)
        mean_rgb_color = seg_colors[close_enough].mean(dim=0)
        lab_selected_colors[-1, :] = mean_color
        rgb_selected_colors[-1, :] = mean_rgb_color

        # Mark all these segments as done
        processed[close_enough] = True

        # Update viz
        newly_processed = torch.where(close_enough)[0]
        for s in newly_processed:
            segments_viz_post[segments_torch == s, :] = rgb_selected_colors[seg_selected_color_idx[s], :].unsqueeze(0)

        # For all (not yet processed) segments, update min distance to existing cluster
        distances = torch.linalg.norm(lab_seg_colors - mean_color, dim=1)
        if min_distances_to_cluster is None:
            min_distances_to_cluster = distances
        else:
            min_distances_to_cluster = torch.min(min_distances_to_cluster, distances)

        min_distances_to_cluster[processed] = 0  # ignore processed segments

        # Select furthest color to use next
        max_dist, seg_to_process = min_distances_to_cluster.max(dim=0)

        if debug_output:
            logger.debug(f'Processed segment {idx} ({seg}), total processed {processed.sum().item()}/{processed.numel()}'
                         f'--> next seg {seg_to_process} (Lab delta {max_dist})')

    extra_data = {}
    if debug_output:
        extra_data = {'segments_viz': segments_viz,
                      'mean_colors_viz': segments_viz_mean,
                      'blurred_image': torch_img}

    return segments_viz_post, extra_data
