# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os, shutil
from tqdm.auto import tqdm
import math

import re
import tarfile
from itertools import chain
from collections import defaultdict

import torch

import kaolin
from gloss.utils.parser import ParserHelper
from gloss.data.render_dataloader import MultiViewConfig, ViewConfig, configure_local_view_dataset
from gloss.utils import reclaim_cuda_memory

from pathlib import Path


def remove_pngs(dir_path):
    """Delete all *.png files in dir_path (non‑recursive)."""
    for fp in Path(dir_path).glob("*.png"):
        try:
            fp.unlink()           # permanently removes the file
            # print(f"Deleted {fp}")   # uncomment for logging
        except Exception as e:
            print(f"Could not delete {fp}: {e}")


def _assert_alpha_nondegenerate(data_cache_dir, view, sample_n=8):
    """Fail loudly before WDS-packing if every sampled albedo_alpha PNG is
    all-zero. Catches mesh/material edge cases (e.g. glTF without an explicit
    metallicFactor) that would otherwise silently produce a WDS shard whose
    loss mask is empty everywhere — train_diffusion_wds then trains with
    train_loss=0 for the entire run.
    """
    import imageio.v2 as imageio
    alpha_files = sorted(p for p in Path(data_cache_dir).glob("*_albedo_alpha.png"))
    if not alpha_files:
        return  # nothing rendered yet; let downstream code handle the empty case
    sampled = alpha_files if len(alpha_files) <= sample_n else \
              alpha_files[::max(1, len(alpha_files) // sample_n)][:sample_n]
    maxes = [int(imageio.imread(p).max()) for p in sampled]
    if max(maxes) == 0:
        raise RuntimeError(
            f"All {len(sampled)} sampled albedo_alpha PNGs in {data_cache_dir} "
            f"are zero (view={view}). This is the silent-failure mode triggered "
            f"by glTF materials with implicit metallicFactor=1 (or other PBR "
            f"settings that zero the alpha-extraction render). Patch the mesh's "
            f"scene.gltf with pbrMetallicRoughness.metallicFactor=0.0 and "
            f"re-render before packing into WDS."
        )
           
            
def process_view(view, base_dir, target_dir):
    _data_cache_dir = os.path.join(base_dir, view, 'datacache')
    print(_data_cache_dir)
    assert os.path.exists(_data_cache_dir)
    _assert_alpha_nondegenerate(_data_cache_dir, view)
    image_files = [f for f in os.listdir(_data_cache_dir) if f.endswith(".png")]
    prefix_pattern = re.compile(
        r"^(e-?\d+\.\d+_-?\d+\.\d+_-?\d+\.\d+"
        r"a-?\d+\.\d+_-?\d+\.\d+_-?\d+\.\d+"
        r"u-?\d+\.\d+_-?\d+\.\d+_-?\d+\.\d+"
        r"_f-?\d+\.\d+)"
    )
    grouped = defaultdict(list)

    for filename in image_files:
        match = prefix_pattern.match(filename)
        if match:
            prefix = match.group(1)
            grouped[prefix].append(filename)

    expected_num_channels = max([len(v) for v in grouped.values()])
    valid_samples = [v for v in grouped.values() if len(v) == expected_num_channels]
    chunks = chunk_list(valid_samples, 100)
    num_tar_files = math.ceil(len(valid_samples) / 100)
    for i in range(num_tar_files):
        filenames = list(chain.from_iterable(chunks[i]))
        create_shard(os.path.join(target_dir, f"{view}-{i}.tar"), _data_cache_dir, view, filenames)


def create_shard(shard_file, base_dir, view, filenames):
    with tarfile.open(shard_file, "w") as tar:
        for img_file in filenames:
            img_path = os.path.join(base_dir, img_file)
            tar.add(img_path, arcname=f"{view}/{img_file}")


def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def move_if_exists(src, dst):
    """Move a cache artifact if it still exists from this generation pass."""
    if not os.path.exists(src):
        print(f"Skipping missing cache artifact: {src}")
        return
    shutil.move(src, dst)


def parse_available_view_indices(texture_dir):
    """Return sorted single-view ids that exist under texture_dir."""
    pattern = re.compile(r"^view(\d+)\.png$")
    indices = []
    for filename in os.listdir(texture_dir):
        match = pattern.match(filename)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def load_excluded_view_indices(meta_json):
    if meta_json is None:
        return set()
    with open(meta_json, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return set(meta.get("exclude_from_training_indices", []))


def generate_views(mesh, view_config, args, device):
    _dataset, _ = configure_local_view_dataset(view_config, device, args.global_root_dir, mesh)
    print("thresholding", args.x_margin, args.y_margin, args.z_threshold)
    _dataset.turn_on_sample_quality_check(args.x_margin, args.y_margin, args.z_threshold)
    good_samples = []
    for i in tqdm(range(view_config.num_training_views)):
        data = _dataset[i]
        if not _dataset.check_bad_sample(data):
            good_samples.append(i)
            
    # create a dir called bad samples
    bad_sample_dir = os.path.join(_dataset.cache_dir, 'bad samples')
    if os.path.exists(bad_sample_dir) and os.path.isdir(bad_sample_dir):
        shutil.rmtree(bad_sample_dir)
    os.makedirs(bad_sample_dir, exist_ok=True)
    channels = view_config.input_channels[:-1]
    channels.append("positions")
    channels.append("background_alpha")
    channels.append("albedo_alpha")
    for i in range(view_config.num_training_views):
        if i not in good_samples:
            extrinsics_hash =  _dataset.extrinsics_sampler.get_hash(i)
            intrinsics_hash =  _dataset.intrinsics_sampler.get_hash(i)
            item_hash = f"{extrinsics_hash}_{intrinsics_hash}"
            file_names = [f"{item_hash}_{channel}.png" for channel in channels]
            for fn in file_names:
                move_if_exists(os.path.join(_dataset.cache_dir, fn), bad_sample_dir)
            
    # update extrinsics and intrinsics.pt
    archive_dir = os.path.join(_dataset.cache_dir, 'archive')
    extrinsics_fp = os.path.join(_dataset.cache_dir, 'extrinsics.pt')
    intrinsics_fp = os.path.join(_dataset.cache_dir, 'intrinsics.pt')
    os.makedirs(archive_dir, exist_ok=True)
    move_if_exists(extrinsics_fp, os.path.join(archive_dir, 'extrinsics.pt'))
    move_if_exists(intrinsics_fp, os.path.join(archive_dir, 'intrinsics.pt'))
    torch.save(_dataset.extrinsics_sampler.samples[good_samples], extrinsics_fp)
    torch.save(_dataset.intrinsics_sampler.samples[good_samples], intrinsics_fp)
    del _dataset
    reclaim_cuda_memory()
        

if __name__ == '__main__':
    parser_helper = ParserHelper('Inpainter from scratch')
    parser_helper.add_dataclass_flags(MultiViewConfig, 'data')
    parser_helper.parser.add_argument('--start_view', type=int, default=0, help="Number of threads for parallel processing")
    parser_helper.parser.add_argument('--end_view', type=int, default=-1, help="Number of threads for parallel processing")
    parser_helper.parser.add_argument('--num_workers', type=int, default=16, help="Number of threads for parallel processing")
    parser_helper.parser.add_argument('--x_margin', type=float, default=0.75, help="normal x filter")
    parser_helper.parser.add_argument('--y_margin', type=float, default=0.75, help="normal y filter")
    parser_helper.parser.add_argument('--z_threshold', type=float, default=0.1, help="normal z threshold")
    parser_helper.parser.add_argument(
        '--meta_json',
        type=str,
        default=None,
        help="Optional meta.json with exclude_from_training_indices for single-view filtering",
    )
    parser_helper.parser.add_argument(
        '--max_single_views',
        type=int,
        default=-1,
        help="Maximum number of kept single views to process after exclusions; -1 = all",
    )
    args = parser_helper.parse_args()
    device = torch.device('cuda:0')
    
    def _get_full_fp(fp):
        return os.path.join(args.global_root_dir, fp)
    
    data_config = args.data
    data_config.jitter = False
    torch.manual_seed(data_config.seed)
    
    mesh_fname =_get_full_fp(data_config.mesh)
    mesh = kaolin.io.mesh.import_mesh(mesh_fname).to(device)  # SurfaceMesh object
    # Local patches are sampled in mesh-local space, and the single-view
    # texture is in UV space (rotation-invariant), so the local-rendering
    # frame is independent of orientation.json. We deliberately skip the
    # orientation rotation here because apply_orientation_to_mesh only
    # rotates vertices, not vertex_normals/tangents — applying it would
    # leave camera_normals stale relative to the rotated geometry.
    mesh.vertices = kaolin.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)
    data_base =  _get_full_fp(data_config.data_base_dir)
    os.makedirs(data_base, exist_ok=True)
    tar_data_base = data_base + "-wds"
    os.makedirs(tar_data_base, exist_ok=True)
    texture_dir, sampling_dir=  _get_full_fp(data_config.single_view_texture_dir), _get_full_fp(data_config.sampling_dir)
    
    available_view_indices = parse_available_view_indices(texture_dir)
    excluded_view_indices = load_excluded_view_indices(args.meta_json)
    selected_view_indices = [
        idx for idx in available_view_indices if idx not in excluded_view_indices
    ]
    if args.max_single_views >= 0:
        selected_view_indices = selected_view_indices[:args.max_single_views]
    end_view = None if args.end_view == -1 else args.end_view
    selected_view_indices = selected_view_indices[args.start_view:end_view]

    print(
        "selected single views:",
        len(selected_view_indices),
        "available:",
        len(available_view_indices),
        "excluded:",
        len(excluded_view_indices),
    )

    for idx in tqdm(selected_view_indices):
        _data_base_dir = os.path.join(data_base, "view%04d" % idx)
        _data_cache_dir = os.path.join(_data_base_dir, 'datacache')
        _single_view_texture = os.path.join(texture_dir, "view%04d.png" % idx)
        _face_sampling_weights = os.path.join(sampling_dir, "view%04d.pt" % idx)
        _in_channels = ['camera_normals', 'geo_camera_normals', 'relative_positions', 'albedo', 'inpaint_mask']
        _config = ViewConfig(mesh=data_config.mesh, num_training_views=data_config.num_local_views,
                            num_reference_views=0, cache_data_dir=_data_cache_dir,
                            zip_cache=False, single_view_texture=_single_view_texture,
                            camera_dist=data_config.camera_dist, fov_min=data_config.fov_min, fov_max=data_config.fov_max,
                            face_sampling_weights=_face_sampling_weights, on_device_cache=True,
                            input_channels=_in_channels)
        # check if already finish genererating 
        extrinsics_fp, intrinsics_fp = os.path.join(_data_cache_dir, "extrinsics.pt"), os.path.join(_data_cache_dir, "intrinsics.pt")
        if os.path.exists(extrinsics_fp) and os.path.exists(intrinsics_fp):
            print("already generated")
        else:
            generate_views(mesh, _config, args, device)
        num_png = sum([1 for fn in os.listdir(_data_cache_dir) if fn.endswith('.png')])
        if num_png > 0:
            process_view("view%04d" % idx, data_base, tar_data_base)
            remove_pngs(_data_cache_dir)

