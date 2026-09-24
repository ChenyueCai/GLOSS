# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os, sys, shutil, yaml
import shutil, errno, os
import json
import torch, torchvision
import logging
from gloss.utils.diffusion_render import get_canny, get_depth
import gloss.model.standard
from omegaconf import OmegaConf
from gloss.data.render_dataloader import ViewConfig, configure_single_view_dataset, cat_collate_dicts
import gloss.model.standard
import gloss.logging
from gloss.logging import log_tensor, log_tensor_dict
from gloss.utils.kaolin_utils import camera_to_meta
from gloss.utils.parser import ParserHelper
import gloss.utils.single_view
from gloss.notebook.common import *

def parse_prompts(fname):
    #sys.setdefaultencoding('utf-8')
    with open(fname) as f:
        lines = [x for x in [x.strip() for x in f.readlines()] if len(x) > 0]
    return lines

if __name__ == '__main__':
    parser_helper = ParserHelper('Sample script to get sample output of LocalRendersDataset.')
    parser_helper.parser.add_argument('--mesh', type=str, required=True, help='Mesh filename in obj, usd or gltf format.')
    parser_helper.parser.add_argument('--overwrite', action="store_true")
    parser_helper.parser.add_argument('--resolution', type=int, default=512)
    parser_helper.parser.add_argument('--fov_min', type=float, default=1.0 )
    parser_helper.parser.add_argument('--fov_max', type=float, default=1.1)
    parser_helper.parser.add_argument('--azi_min', type=float, default=0.0 )
    parser_helper.parser.add_argument('--azi_max', type=float, default=3.14)
    parser_helper.parser.add_argument('--elev_min', type=float, default=-1.0 )
    parser_helper.parser.add_argument('--elev_max', type=float, default=1.7)
    parser_helper.parser.add_argument('--viewdist_min', type=float, default=0.95 )
    parser_helper.parser.add_argument('--viewdist_max', type=float, default=1.05)
    parser_helper.parser.add_argument('--prompts', type=str, required=True, help='File with one prompt per line.')
    parser_helper.parser.add_argument('--negative_prompt', type=str, default="low quality, bad quality, deformed, low-poly")
    parser_helper.parser.add_argument('--output_dir', type=str, required=True, help='Output directory to write data to.')
    parser_helper.parser.add_argument('--controls', type=list, default=['depth', 'canny', 'normal'])
    parser_helper.parser.add_argument('--strengths', type=list, default=[2.0, 2.0, 2.0])
    
    args = parser_helper.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    # render single view dataset to check parameters 

    # Parse configurations from flags
    # prompts = parse_prompts(args.prompts)
    # print(f'Parsed {len(prompts)} prompts from {args.prompts}')
    
    debug_dir = os.path.join(args.output_dir, 'debug')
    view_dir = os.path.join(args.output_dir, 'debug_meta')

    try:
        shutil.rmtree(debug_dir)
        shutil.rmtree(view_dir)
    except FileNotFoundError:
        pass                 # already gone, nothing to do
    except OSError as e:     # handle other removal problems
        print(f"Delete failed: {e.strerror}", file=sys.stderr)
    
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(view_dir, exist_ok=True)
    
    
    channels = ['camera_normals']
    view_config = ViewConfig(mesh=args.mesh, num_training_views=10, cache_data_dir=view_dir,
                                fov_min=args.fov_min, fov_max=args.fov_max, resolution=args.resolution,
                                input_channels=channels, output_channels=channels)
    view_dataset = configure_single_view_dataset(view_config, device, args.output_dir, debug=True)
    

    for idx, data in enumerate(view_dataset):
        if idx == 32:
            exit()
        debug_albedo = data['camera_normals']
        torchvision.utils.save_image(debug_albedo.permute(0, 3, 1, 2) / 2 + 0.5, os.path.join(debug_dir, 'view%04d.png' % idx))
        