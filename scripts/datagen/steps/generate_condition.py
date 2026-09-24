# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os, sys, shutil
from pathlib import Path
import yaml   # pip install pyyaml
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

logger = logging.getLogger(__name__)


def parse_prompts(fname):
    #sys.setdefaultencoding('utf-8')
    with open(fname) as f:
        lines = [x for x in [x.strip() for x in f.readlines()] if len(x) > 0]
    return lines


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    

if __name__ == '__main__':
    parser_helper = ParserHelper('Sample script to get sample output of LocalRendersDataset.')
    p = parser_helper.parser 
    
    p.add_argument('--mesh', type=str, required=True,
                   help='Mesh filename in obj, usd or gltf format.')
    def _str2bool(v):
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ('true', '1', 'yes', 'y'):
            return True
        if s in ('false', '0', 'no', 'n', ''):
            return False
        raise ValueError(f'expected boolean, got {v!r}')
    p.add_argument('--overwrite', type=_str2bool, default=True,
                   help='If true (default), wipe meta/, gen_view/, and '
                        'condition_output/ before rendering.')
    p.add_argument('--resolution', type=int, default=512)
    p.add_argument('--prompts', type=str, default=None,
                   help='Optional file with one prompt per line. When omitted, '
                        'meta/view*.yml is written without a prompt field; '
                        'num_views is taken from --num_views. Prompts are '
                        'attached to each view downstream by '
                        'generate_view_prompts.py.')
    p.add_argument('--num_views', type=int, default=500,
                   help='Number of views to render when --prompts is not set.')
    p.add_argument('--mesh_name', type=str, default=None,
                   help='Mesh key into --configs. Required when --prompts '
                        'is not supplied (otherwise inferred from the prompts '
                        'file stem).')
    p.add_argument('--configs', type=str, required=True,
                   help='File with view configs.')
    p.add_argument('--output_dir', type=str, required=True,
                   help='Output directory to write data to.')
    p.add_argument('--mode', type=str, default="generate")
    p.add_argument("--fov_min", type=float, nargs="+",
                    help="one or more azi_min values", default=[1.0])
    p.add_argument("--fov_max", type=float, nargs="+",
                    help="one or more azi_min values", default=[1.1])
    p.add_argument("--azi_min", type=float, nargs="+",
                    help="one or more azi_min values", default=[0.0])
    p.add_argument("--azi_max", type=float, nargs="+",
                    help="one or more azi_min values", default=[6.28])
    p.add_argument("--elev_min", type=float, nargs="+",
                    help="one or more azi_min values", default=[-1.0])
    p.add_argument("--elev_max", type=float, nargs="+",
                    help="one or more azi_min values", default=[1.7])
    p.add_argument("--viewdist_min", type=float, nargs="+",
                    help="one or more azi_min values", default=[0.95])
    p.add_argument("--viewdist_max", type=float, nargs="+",
                    help="one or more azi_min values", default=[1.05])
    p.add_argument("--canny_normal_low", type=int, default=180,
                   help="Canny low threshold applied to per-pixel camera-normals (canny-normal output).")
    p.add_argument("--canny_normal_high", type=int, default=200,
                   help="Canny high threshold applied to per-pixel camera-normals (canny-normal output).")

    args = parser_helper.parse_args()
    device = "cuda"
    torch.manual_seed(0)

    if args.prompts:
        mesh_name = os.path.splitext(os.path.basename(args.prompts))[0]
    elif args.mesh_name:
        mesh_name = args.mesh_name
    else:
        raise ValueError("Either --prompts or --mesh_name must be provided "
                         "to look up view configs.")
    view_cfgs = load_yaml(Path(args.configs))
    view_args = None
    for entry in view_cfgs:
        if entry["mesh_name"] == mesh_name:
            view_args = entry["more_args"]
    if view_args is None:
        raise ValueError(f"No view config entry for mesh_name={mesh_name!r} "
                         f"in {args.configs}")
    
    for k in ("fov_min", "fov_max", "azi_min", "azi_max", "elev_min", 
              "elev_max", "viewdist_min", "viewdist_max"):
        setattr(args, k, view_args[k]) 
    
    os.makedirs(args.output_dir, exist_ok=True)
    assert os.path.isdir(args.output_dir), f'Output dir DNE: {args.output_dir}'
    # for experimentation purpose record the camera
    cfg = vars(args).copy()
    cfg_fp = os.path.join(args.output_dir, "meta.json")
    with open(cfg_fp , "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved arguments to {cfg_fp}")
    
    # Make subdirectory
    view_dir = os.path.join(args.output_dir, 'meta')
    img_dir = os.path.join(args.output_dir, 'gen_view')
    superres_dir = os.path.join(args.output_dir, 'gen_view_super')
    decomposite_dir = os.path.join(args.output_dir, 'gen_view_decomposite')
    condition_dir = os.path.join(args.output_dir, 'condition_output')
    if args.overwrite:
        if os.path.exists(view_dir):
            shutil.rmtree(view_dir)
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        if os.path.exists(superres_dir):
            shutil.rmtree(superres_dir)
        if os.path.exists(decomposite_dir):
            shutil.rmtree(decomposite_dir)
        if os.path.exists(condition_dir):
            shutil.rmtree(condition_dir)        
    os.makedirs(view_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(superres_dir, exist_ok=True)
    os.makedirs(decomposite_dir, exist_ok=True)
    os.makedirs(condition_dir, exist_ok=True)
    
    # prep for different channel dir
    os.makedirs(os.path.join(condition_dir, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(condition_dir, 'geonormal'), exist_ok=True)
    os.makedirs(os.path.join(condition_dir, 'canny-normal'), exist_ok=True)
    os.makedirs(os.path.join(condition_dir, 'canny-geonormal'), exist_ok=True)
    os.makedirs(os.path.join(condition_dir, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(condition_dir, 'mask'), exist_ok=True)
    
    # Parse prompts (optional). When missing, meta/view*.yml is written without
    # a prompt field; the new generate_view_prompts.py step fills them in.
    if args.prompts:
        prompts = parse_prompts(args.prompts)
        print(f'Parsed {len(prompts)} prompts from {args.prompts}')
        num_views_full = len(prompts)
    else:
        prompts = None
        num_views_full = args.num_views
        print(f'No --prompts supplied; rendering {num_views_full} views with '
              f'no prompt field in meta/view*.yml.')

    num_views = 10 if args.mode == "debug" else num_views_full

    # Parse configurations from flags
    channels = ['normals', 'raw_depth', 'render', 'mask', 'camera_normals', 'geo_camera_normals']
    view_config = ViewConfig(mesh=args.mesh, num_training_views=num_views, cache_data_dir=view_dir,
                                   fov_min=args.fov_min, fov_max=args.fov_max, 
                                   azi_min=args.azi_min, azi_max=args.azi_max, 
                                   elev_min=args.elev_min, elev_max=args.elev_max,
                                   viewdist_min=args.viewdist_min, viewdist_max=args.viewdist_max,
                                   resolution=args.resolution,
                                   input_channels=channels, output_channels=channels)
    view_dataset = configure_single_view_dataset(view_config, device, args.output_dir)

    for idx, data in enumerate(view_dataset):
        existed = False
        view_path = os.path.join(view_dir, 'view%04d.yml' % idx)
        img_path = os.path.join(img_dir, 'view%04d.png' % idx) 
        if os.path.exists(img_path) and os.path.exists(view_path):
            try:
                with open(view_path, 'r') as f:
                    yaml.safe_load(f)
                existed = True
            except yaml.YAMLError as e:
                print(f"Invalid YAML file: {view_path}\nError: {e}")
                existed = False
        if existed:
            print(f"{view_path} alread exited")
        else:   
            #log_tensor_dict(data, 'val', logger, print_stats=True)
            _normal, _raw_depth, _cam_normal, _geo_cam_normal, _mask = data['normals'], data['raw_depth'], data['camera_normals'], data['geo_camera_normals'], data["mask"]
            _canny_geonormal = get_canny(_geo_cam_normal)
            # _cam_normal is in [-1, 1]; remap to [0, 1] (matching the saved
            # normal*.png) so canny thresholds operate on the visible image,
            # not on a hard-zero-clipped version of it.
            _canny_normal = get_canny(_cam_normal / 2 + 0.5, low=args.canny_normal_low, high=args.canny_normal_high)
            _depth = get_depth(_raw_depth)
            torchvision.utils.save_image(_cam_normal.permute(0, 3, 1, 2)/2 + 0.5, os.path.join(condition_dir, 'normal', 'normal%04d.png'% idx))
            torchvision.utils.save_image(_geo_cam_normal.permute(0, 3, 1, 2)/2 + 0.5, os.path.join(condition_dir, 'geonormal', 'geonormal%04d.png'% idx))
            torchvision.utils.save_image(_depth, os.path.join(condition_dir, 'depth', 'depth%04d.png'% idx))
            torchvision.utils.save_image(_canny_normal, os.path.join(condition_dir, 'canny-normal', 'canny%04d.png'% idx))
            torchvision.utils.save_image(_canny_geonormal, os.path.join(condition_dir, 'canny-geonormal', 'canny%04d.png'% idx))
            torchvision.utils.save_image(_mask.permute(0, 3, 1, 2), os.path.join(condition_dir, 'mask', 'mask%04d.png'% idx))
            camera = view_dataset.get_camera(idx)
            meta = {'camera': camera_to_meta(camera)}
            if prompts is not None:
                meta['prompt'] = prompts[idx % len(prompts)]
            OmegaConf.save(config=OmegaConf.create(meta), f=view_path)
        if idx == num_views - 1:
            exit()
