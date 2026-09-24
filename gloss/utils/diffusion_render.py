# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch, torchvision
import numpy as np
import cv2
            

def get_depth(raw_depth:torch.Tensor):
    """_summary_

    Args:
        raw_depth (torch.Tensor): NDC, range between 0-1, background is 0

    Returns:
        _type_: relative depth of the raw depth value
    """
    zero_mask = torch.argwhere(torch.all(raw_depth == 0.0, dim=3))
    raw_depth[:, zero_mask[:,1], zero_mask[:,2], :] = 1.0
    depth_map = 1 / raw_depth
    depth_map = depth_map.clip(min=0)
    depth_map = depth_map.permute(0, 3, 1, 2)
    depth_min = torch.amin(depth_map, keepdim=True)
    depth_max = torch.amax(depth_map, keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    depth = torch.cat([depth_map] * 3, dim=1)
    return depth

# def get_depth(im):
#     eps = 1e-8
#     im = im.clamp(min=eps)
#     depth_map = 1 / im
#     depth_map = depth_map.clip(min=0)
#     depth_map = depth_map.permute(0, 3, 1, 2)
#     depth_min = torch.amin(depth_map, keepdim=True)
#     depth_max = torch.amax(depth_map, keepdim=True)
#     depth_map = (depth_map - depth_min) / (depth_max - depth_min)
#     depth = torch.cat([depth_map] * 3, dim=1)
#     return depth


def get_canny(im:torch.Tensor, low=180, high=200):
    """_summary_

    Args:
        im (torch.Tensor): assume we have a tensor image of range [0,1], BHWC
        low (int, optional): _description_. Defaults to 100.
        high (int, optional): _description_. Defaults to 200.

    Returns:
        _type_: B C HW
    """
    im = im.clip(0, 1).detach().cpu().squeeze(0).numpy() * 255
    im = im.clip(0, 255).astype(np.uint8)
    canny  = np.tile(cv2.Canny(im, low, high)[:, :, None], (1, 1, 3))
    canny = torch.from_numpy(canny).unsqueeze(0).float().permute(0, 3, 1, 2).cuda() / 255.0
    return canny
