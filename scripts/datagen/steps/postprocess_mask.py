# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os, sys, shutil, yaml
import torch, torchvision
import tqdm
import logging
import gloss.logging
from gloss.utils.parser import ParserHelper
import gloss.utils.single_view
from gloss.notebook.common import *
from pathlib import Path

logger = logging.getLogger(__name__)

def image_is_valid(path: str) -> bool:
    p = Path(path)
    if not p.is_file():                    # 1️⃣  does the file exist?
        return False
    try:
        # 2️⃣  verify header (cheap)
        with Image.open(p) as img:
            img.verify()                   # raises if header is corrupt

        # 3️⃣  fully load pixels (catches truncated data)
        with Image.open(p) as img:
            img.load()                     # raises if pixel data is corrupt
    except Exception:                      # (IOError, SyntaxError, PIL.UnidentifiedImageError …)
        return False
    return True

if __name__ == '__main__':
    parser_helper = ParserHelper('Sample script to get sample output of LocalRendersDataset.')
    parser_helper.parser.add_argument('--output_dir', type=str, required=True, help='Output directory to write data to.')
    
    args = parser_helper.parse_args()
    device = "cuda"
    
    view_dir = os.path.join(args.output_dir, 'gen_view')
    mask_dir = os.path.join(args.output_dir, 'condition_output', 'mask')
    masked_view_dir = os.path.join(args.output_dir, 'gen_view_masked')
    os.makedirs(masked_view_dir, exist_ok=True)
    
    img_fps = [fp for fp in os.listdir(view_dir) if fp.endswith('.png')]
    num_images = len(img_fps)
    for idx in tqdm.tqdm(range(num_images)):
        masked_view_fp = os.path.join(masked_view_dir, 'view%04d.png'% idx)
        if image_is_valid(masked_view_fp):
            print(f"image {idx} already masked. pass.")
            pass
        else:
            view_fp = os.path.join(view_dir, 'view%04d.png'% (idx))
            mask_fp = os.path.join(mask_dir, 'mask%04d.png'% (idx))
            view = torchvision.io.read_image(view_fp) / 255.0   # [C, H, W]
            mask = torchvision.io.read_image(mask_fp) / 255.0   # [C, H, W]
            # Resize mask to match view resolution (e.g. 512→1024 for SDXL views)
            if mask.shape[-2:] != view.shape[-2:]:
                import torch.nn.functional as F
                mask = F.interpolate(
                    mask.unsqueeze(0), size=view.shape[-2:], mode="nearest"
                ).squeeze(0)
            view_masked = view * mask
            masked_view_fp = os.path.join(masked_view_dir, 'view%04d.png'% idx)
            torchvision.utils.save_image(view_masked, masked_view_fp)
