import cv2
import torch
import numpy as np

import time


def expand_mask_soft(mask, max_distance=20):
    """
    Expands a binary mask outward with a soft gradient.
    
    Args:
        mask (torch.Tensor): binary mask (0 or 1), in the shape of B C H W
        max_distance (int): how far the soft gradient extends
    
    Returns:
        torch.Tensor: non binary soft mask with values in [0,1], , in the shape of B C H W
    """
    # convert mask to numpy arr
    mask_arr = mask.squeeze(0).squeeze(0).cpu().numpy()
    mask_arr = mask_arr.astype(np.uint8)

    # Compute distance transform from the background
    dist = cv2.distanceTransform(1 - mask_arr, cv2.DIST_L2, 5)

    # Clip distances to max_distance
    dist = np.clip(dist, 0, max_distance)

    # Normalize to [0,1] and invert (inside=1, outside decreases)
    soft_mask = np.exp(-dist / max_distance * 3)  # exponential falloff
    soft_mask = np.clip(soft_mask, 0.0, 1.0)
    
    soft_mask = torch.tensor(soft_mask).unsqueeze(0).unsqueeze(0).to(mask.device)

    return soft_mask


def composite_inpaint(texture_existing, texture_inpaint,
                      mask_existing, mask_inpaint, mask_fill, 
                      soft=True, soft_margin=20):
    """ 
    Composite the existing texture map with inpaint texture over regions that mask_fill covers, 
    Expand the mask_fill with soft margin if needed
    
    Args:
        texture_existing (_type_): existing texture map in B 4 H W, range(0,1)
        texture_inpaint (_type_): inpaint texture map in B 4 H W, range(0,1)
        mask_existing (_type_): binary mask of the existing texture (0 or 1), in the shape of B C H W
        mask_inpaint (_type_): binary mask of the inpaint texture (0 or 1), in the shape of B C H W
        mask_fill (_type_): binary mask to be filled (0 or 1), in the shape of B C H W
        soft (bool, optional): _description_. Defaults to True.
        soft_margin (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: updated texture map, range(0,1) for all channels
    """
    
    if soft:
        mask_fill_soft = expand_mask_soft(mask_fill, max_distance=soft_margin)
        w_inpaint = (mask_fill_soft * mask_existing * mask_inpaint + mask_fill)
    else:
        w_inpaint = mask_fill
    composite = texture_inpaint * w_inpaint + texture_existing * (1.0 - w_inpaint)
    alpha = mask_existing + mask_fill
    composite[:, 3, ...] = alpha
    return composite


import torch
import torch.nn.functional as F

def gaussian_kernel(size: int, sigma: float, channels: int):
    """Create 2D Gaussian kernel for depthwise conv"""
    coords = torch.arange(size) - size // 2
    grid = coords.repeat(size, 1)
    kernel = torch.exp(-(grid**2 + grid.t()**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    kernel = kernel.view(1, 1, size, size).repeat(channels, 1, 1, 1)
    return kernel

def gaussian_blur(img: torch.Tensor, size: int = 21, sigma: float = None):
    """
    Apply Gaussian blur to tensor image (B,C,H,W).
    If sigma is None, choose automatically based on kernel size.
    """
    if sigma is None:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8  # OpenCV-style default
    
    C = img.shape[1]
    kernel = gaussian_kernel(size, sigma, C).to(img.device)
    return F.conv2d(img, kernel, padding=size//2, groups=C)


def dilate_nonblack_pool(texture: torch.Tensor, kernel_size=3, iterations=None, threshold=1e-5, inpaint_mask=None, mode='avg'):
    """
    Dilate non-black regions in texture using max_pool2d.

    Args:
        texture (torch.Tensor): [B, C, H, W] in [0, 1]
        kernel_size (int): pooling kernel (e.g. 3, 5, 7)
        iterations (int): number of dilation passes
        threshold (float): threshold to consider non-black
    """
    dilated = texture.clone()
    
    
    while True:
        # mask of where there is any color
        mask = (dilated.abs().sum(dim=1, keepdim=True) > threshold).float()
        expanded_mask = F.max_pool2d(mask, kernel_size, stride=1, padding=kernel_size // 2)
        
        if mode == 'max':
            # expand the mask by max pooling
            # expand color by max pooling each channel independently
            color_expanded = F.max_pool2d(dilated, kernel_size, stride=1, padding=kernel_size // 2)
            
        # ---- we can also use average color to fill 
        # instead of max pool, get the sum of neighbors
        # create kernel for conv
        if mode == 'avg':
            kernel = torch.ones((dilated.shape[1], 1, kernel_size, kernel_size)).to(dilated.device)
            # instead of max pool, get the sum of neighbors
            neighbor_sum = F.conv2d(dilated, kernel, padding=kernel_size // 2, groups=dilated.shape[1])
            neighbor_count = F.conv2d(mask, kernel, padding=kernel_size // 2)
            # if neighbor count is too little e.g. 1 or 2 then we don't use it
            valid_neighbor = (neighbor_count > 3).float() #TODO: min neighbor threshold
            neighbor_sum = neighbor_sum * valid_neighbor
            expanded_mask = expanded_mask * valid_neighbor
            avg_neighbors = neighbor_sum / (neighbor_count + 1e-6)
            color_expanded = avg_neighbors
        # keep existing color where mask existed, else fill from pooled color
        dilated = torch.where(mask.bool(), dilated, color_expanded)
        
        if iterations is not None:
            iterations -= 1
            if iterations == 0:
                return dilated

        if inpaint_mask is not None:
            dilated = texture + dilated * inpaint_mask
            incomplete_mask = torch.clip(inpaint_mask - expanded_mask, 0, 1)
            if incomplete_mask.sum() == 0.0:
                return dilated


def erode(mask, kernel_size=3, iterations=1):
    padding = kernel_size // 2
    mask = (mask > 0.5).int().clip(0.0, 1.0)
    for _ in range(iterations):
        mask = 1 - F.max_pool2d(
            1 - mask,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
    return mask.clip(0.0, 1.0)