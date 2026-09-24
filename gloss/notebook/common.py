# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import kaolin
import logging
import math
import os

import matplotlib.pyplot as plt
import torch
import torchvision

from gloss.logging import default_log_setup
from gloss.viz.text import image_with_header


logger = logging.getLogger(__name__)


def to_numpy_uint(imgs):
    """
    :param imgs: B x C x H x W 0...1 torch tensor float32
    :return: B x H x W x C 0...255 numpy array
    """
    return (imgs.detach() * 255).to(torch.uint8).permute(0, 2, 3, 1).clip(0, 255).cpu().numpy()


def quick_viz(imgs, save_file=None, save_prefix=None, nrow=None, inches=15, save_suffix=".png"):
    """
    Visualize a batch of images as a single row.
    :param imgs: B x C x H x W torch tensor in range 0..1
    :param save_file: if provided, will save entire image there
    :param save_prefix: if provided, will save every image in the batch individually with this prefix
    :param nrow: number of images per row, will be set to B if not provided
    :param inches: controls the size of the figure
    """
    if nrow is None:
        nrow = imgs.shape[0]
    res = torchvision.utils.make_grid(imgs.detach(), nrow=nrow, padding=0).permute(1, 2, 0)
    res = (res * 255).clip(0, 255).to(torch.uint8).detach().cpu().numpy()
    fig = plt.figure()
    fig.set_size_inches(nrow * inches, int(math.ceil(imgs.shape[0] / nrow)) * inches, forward=True)
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(res)
    rm_axmarks(ax)

    if save_file is not None:
        res = torch.from_numpy(res).permute(2, 0, 1).to(torch.uint8)
        torchvision.io.write_png(res, save_file)  # TODO: test, switched to torchvision
    if save_prefix is not None:
        res_imgs = to_numpy_uint(imgs)
        for i in range(res_imgs.shape[0]):
            torchvision.io.write_png(res_imgs[i, ...], save_prefix + ("img%02d%s" % (i, save_suffix)))
    return ax


def rm_axmarks(in_ax):
    """
    Removes matplotlib axes to better visualize an image.
    """
    in_ax.set_xticklabels([])
    in_ax.set_yticklabels([])
    in_ax.set_xticks([])
    in_ax.set_yticks([])
    in_ax.axis("off")


def configure_output(log_level=logging.DEBUG):
    """
    Utility to configure logger for a notebook with desirable logging level.
    :return: Logger
    """
    default_log_setup(log_level)
    logger = logging.getLogger("notebook")
    torch.set_printoptions(linewidth=120)
    logging.getLogger("matplotlib.font_manager").setLevel(20)
    logging.getLogger("matplotlib.axes._base").setLevel(20)
    logging.getLogger("matplotlib.pyplot").setLevel(20)
    logging.getLogger("PIL.Image").setLevel(20)
    logging.getLogger("PIL").setLevel(20)
    return logger


def visualize_dictionary_batched(r, rescale=True, max_num=10, inches=5):
    """Visualizes a dictionary of batched images, with each line containing all data for one batch, labeled."""
    keys = [ x for x in r.keys()]
    to_return = None
    for i in range(min(r[keys[0]].shape[0], max_num)):
        tmp = {}
        for k, v in r.items():
            v = v[i:i+1, ...]
            if len(v.shape) == 3:
                v = v.unsqueeze(-1)
            if v.shape[-1] == 2:
                v = torch.cat([v, torch.zeros_like(v[..., :1])], dim=-1)
            elif v.shape[-1] == 1:
                v = v.repeat(1, 1, 1, 3)
            if kaolin.utils.testing.check_tensor(v, shape=(1, None, None, 3), throw=False):
                tmp[k] = v
            else:
                logger.warning(f'Could not visualize {k}')
        if rescale:
            den = 2.0
            bias = 0.5
        else:
            den = 1.0
            bias = 0.0
        res = quick_viz(torch.cat([image_with_header(v / den + bias, k) for k, v in tmp.items() ]).permute(0, 3, 1, 2),
              inches=inches) # inches affect notebook size on disk
        if i == 0:
            to_return = res
    return to_return
