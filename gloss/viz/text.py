# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import logging
import numpy as np
import torch

import torch.utils.data
from PIL import Image, ImageDraw, ImageFont

from gloss import ROOT_DIR

logger = logging.getLogger(__name__)

DEFAULT_MARGIN = 0

BUNDLED_FONTS_PATH = os.path.realpath(os.path.join(os.path.dirname(ROOT_DIR), 'resources', 'fonts'))


def torch_image_with_text(text, rows, cols=None, font_size=None):
    """
    @param text:
    @return: 3 x W x W float32 [-1..1] torch image
    """
    if cols is None:
        cols = rows
    if font_size is None:
        font_size = max(15, int(min(rows, cols) / 256 * 45))

    return torch.from_numpy(
        write_text_on_image(np.ones((rows, cols, 3), dtype=np.uint8) * 255, text, font_size=font_size)) \
               .to(torch.float32).permute(2, 0, 1) / 255 * 2 - 1


def write_text_on_image(np_uint8_image, text, font_size=35, color='rgb(0, 0, 0)'):
    """
    Takes and returns a numpy uint8 array.
    """
    pilim = Image.fromarray(np_uint8_image)
    font = ImageFont.truetype(os.path.join(BUNDLED_FONTS_PATH, 'OpenSans-Regular.ttf'), size=font_size)
    draw = ImageDraw.Draw(pilim)
    _, _, tw, th = draw.textbbox((0, 0), text=text, font=font)
    # tw, th = draw.textsize(text, font=font)
    iw = np_uint8_image.shape[1]
    ih = np_uint8_image.shape[0]

    start_x = 0
    start_y = 0
    if tw > iw:
        logger.warning('Text width is wider than image')
    else:
        start_x = (iw - tw) // 2

    if th > ih:
        logger.warning('Text height is larger than image height')
    else:
        start_y = (ih - th) // 2

    draw.text((start_x, start_y), text, font=font, fill=color)
    return np.array(pilim)


def image_with_header(img, text, font_size=None):
    """
    Expands the rows of the input image batch up and writes given text on that new image strip.

    Args:
        img (torch.Tensor): 0..1 float tensor of shapes: H x W x {3, 4}, or {3, 4} x H x W, or batched with
            batch dimension first.
        text (str or list of str): single text string (if passing image of batch 1), or a list of strings per
            each image in the batch.
        font_size (number or None): if not set, will figure out automatically

    Returns:
        Expanded img, same device, channel layout and number of dimensions
    """
    squeeze = False
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
        squeeze = True
    if type(text) is str:
        text = [text]

    hwc = False
    headers = []
    for b in range(img.shape[0]):
        hwc = img.shape[-1] < img.shape[-2]
        width = img.shape[2 if hwc else 3]
        height = width // 8
        header = torch_image_with_text(text[b], height, width, font_size=font_size)  # 3 x height x width
        nchannels = img.shape[-1] if hwc else img.shape[1]
        if nchannels == 4:
            header = torch.cat([header, torch.ones_like(header[:1, ...])], dim=0)
        elif nchannels != 3:
            logger.error(f'Sorry, cannot work with non-3 or 4 channel images')
            return img

        if hwc:
            header = header.permute(1, 2, 0)
        headers.append(header)

    if len(headers) == 0:
        return img

    headers = torch.stack(headers).to(img.device)
    res = torch.cat([headers, img], dim=1 if hwc else 2)
    if squeeze:
        res = res.squeeze(0)
    return res