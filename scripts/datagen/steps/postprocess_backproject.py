# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import math
import re
from PIL import Image

import torch, torchvision
import kaolin

from omegaconf import OmegaConf
import gloss
from gloss.utils.parser import ParserHelper
from gloss.utils.kaolin_utils import apply_orientation_to_mesh, camera_from_meta
import gloss.utils.single_view
import gloss.utils.render


VIEW_ID_RE = re.compile(r"view(\d{4})")


def load_excluded_view_indices(meta_json):
    """Read exclude_from_training_indices from a single_view/meta.json.

    Returns an empty set when meta_json is None / missing / lacks the key.
    """
    if meta_json is None:
        return set()
    if not os.path.isfile(meta_json):
        return set()
    with open(meta_json, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return set(int(i) for i in meta.get("exclude_from_training_indices") or [])


def load_meta(fp):
    return OmegaConf.load(fp)


def load_render(fp):
    # scale to -1 to 1
    # B H W C
    render = (torchvision.io.read_image(fp).unsqueeze(0).permute(0, 2, 3, 1)/255 - 0.5) * 2
    return render

def get_avg_color(img, mask):
    return torch.sum(img * mask, dim=(0, 1, 2)) / torch.sum(mask)


def extract_view_id(fp: str) -> int:
    match = VIEW_ID_RE.search(os.path.basename(fp))
    if match is None:
        raise ValueError(f"Could not extract view id from {fp}")
    return int(match.group(1))


if __name__ == '__main__':
    parser_helper = ParserHelper('Sample script to get sample output of LocalRendersDataset.')
    parser_helper.parser.add_argument('--mesh', type=str, required=True, help='Mesh Path')
    parser_helper.parser.add_argument('--texture_height', type=int, default=1024)
    parser_helper.parser.add_argument('--texture_width', type=int, default=1024)
    parser_helper.parser.add_argument('--output_dir', type=str, required=True, help='Output directory to write data to.')
    parser_helper.parser.add_argument(
        '--meta_json',
        type=str,
        default=None,
        help='Optional single_view/meta.json with exclude_from_training_indices; '
             'matching view ids are skipped.',
    )
    parser_helper.parser.add_argument(
        '--orientation_json',
        type=str,
        default=None,
        help='Optional single_view/orientation.json. When present and any of '
             'yaw/pitch/roll is non-zero, the mesh is rotated to match the '
             'frame the camera meta yamls were sampled in. Without this, '
             'rotated meshes backproject to the wrong UV faces.',
    )
    parser_helper.parser.add_argument(
        '--start_view', type=int, default=0,
        help='First index (after exclusion filter) to process; mirrors step 8.',
    )
    parser_helper.parser.add_argument(
        '--end_view', type=int, default=-1,
        help='End index (exclusive, after exclusion filter); -1 = all. Mirrors step 8.',
    )
    parser_helper.parser.add_argument(
        '--max_angle_deviation', type=float, default=math.pi * 0.3,
        help='Max angle (radians) between face geo-normal and camera ray for the '
             'face to be back-projected. Lower = stricter (only near-frontal '
             'faces). Default math.pi*0.3 (~54 deg).',
    )
    args = parser_helper.parse_args()
    
    device = "cuda"
    torch.manual_seed(0)
    
    assert os.path.isdir(args.output_dir), f'Output dir DNE: {args.output_dir}'
    view_dir = os.path.join(args.output_dir, 'meta')
    img_dir = os.path.join(args.output_dir, 'gen_view_super')
    debug_dir = os.path.join(args.output_dir, 'debug_output')
    
    # make texture and sampling directory
    texture_dir = os.path.join(args.output_dir, 'textures_sr')
    os.makedirs(texture_dir, exist_ok=True)
    sampling_dir = os.path.join(args.output_dir, 'sampling')
    os.makedirs(sampling_dir, exist_ok=True)
    texture_debug_dir = os.path.join(args.output_dir, 'texture_debug')
    os.makedirs(texture_debug_dir, exist_ok=True)
    
    # set up mesh
    mesh = kaolin.io.mesh.import_mesh(args.mesh).to(device)  # SurfaceMesh object
    orient = apply_orientation_to_mesh(mesh, args.orientation_json)
    if any(abs(orient[k]) > 1e-7 for k in ("yaw", "pitch", "roll")):
        print(f"Applied orientation yaw={orient['yaw']:.4f} pitch={orient['pitch']:.4f} "
              f"roll={orient['roll']:.4f} from {args.orientation_json}")
    mesh.vertices = kaolin.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)
    print(mesh)  # view attributes
    
    #optional lighting
    lighting = kaolin.render.easy_render.default_lighting().to(device)
    
    meta_files = sorted(
        os.path.join(view_dir, fp)
        for fp in os.listdir(view_dir)
        if fp.endswith(".yml")
    )
    render_files = sorted(
        os.path.join(img_dir, fp)
        for fp in os.listdir(img_dir)
        if fp.endswith(".basecolor.png")
    )

    meta_by_id = {extract_view_id(fp): fp for fp in meta_files}
    render_by_id = {extract_view_id(fp): fp for fp in render_files}
    shared_ids = sorted(meta_by_id.keys() & render_by_id.keys())

    excluded_view_indices = load_excluded_view_indices(args.meta_json)
    if excluded_view_indices:
        before = len(shared_ids)
        shared_ids = [i for i in shared_ids if i not in excluded_view_indices]
        print(f"Excluding {before - len(shared_ids)} view(s) per "
              f"{args.meta_json} (auto-filtered by score_views.py)")

    end = None if args.end_view == -1 else args.end_view
    if args.start_view or end is not None:
        before = len(shared_ids)
        shared_ids = shared_ids[args.start_view:end]
        print(f"Slicing to [{args.start_view}:{args.end_view}] -> "
              f"{len(shared_ids)} of {before} view(s)")

    missing_render_ids = sorted(meta_by_id.keys() - render_by_id.keys())
    missing_meta_ids = sorted(render_by_id.keys() - meta_by_id.keys())
    if missing_render_ids:
        print(f"Skipping {len(missing_render_ids)} meta view(s) with no superresolved basecolor: {missing_render_ids[:10]}")
    if missing_meta_ids:
        print(f"Skipping {len(missing_meta_ids)} render view(s) with no metadata: {missing_meta_ids[:10]}")
    if not shared_ids:
        raise RuntimeError(
            f"No matching view ids between {view_dir} and {img_dir}. "
            "Expected files like view0000.yml and view0000.basecolor.png."
        )

    for idx in shared_ids:
        texture_path = os.path.join(texture_dir, 'view%04d.png' % idx)
        texture_debug_path = os.path.join(texture_debug_dir, 'view%04d.png' % idx)
        sampling_path = os.path.join(sampling_dir, 'view%04d.pt' % idx)
        view_path = meta_by_id[idx]
        render_path = render_by_id[idx]
        
        exist = False
        if os.path.exists(texture_path):
            try:
                Image.open(texture_path).verify()
                exist = True
                print('Already created. Pass')
            except Exception:
                exist = False
        if not exist:
            meta = load_meta(view_path)
            camera = camera_from_meta(meta.camera).to(device)
            
            render = load_render(render_path).to(device)
            render_res = gloss.utils.render.render_all_features(camera, mesh, lighting)
            render_res['albedo'] = render
            channels = ['albedo', 'render']
            backprojection, mask = gloss.utils.single_view.backproject_render(mesh, camera, render_res, channels, \
                args.texture_height, args.texture_width, min_pixel_count=2, max_angle_deviation=args.max_angle_deviation, upscale=True)
            albedo_debug = (backprojection['render'] + 1) / 2
            albedo_debug = albedo_debug * mask +  get_avg_color(albedo_debug, mask) * torch.ones_like(albedo_debug) * (1 - mask)
            albedo_debug[..., -1] = mask.squeeze(-1)
            albedo = (backprojection['albedo'] + 1) / 2 
            albedo = albedo * mask +  get_avg_color(albedo, mask) * torch.ones_like(albedo) * (1 - mask)
            albedo[..., -1] = mask.squeeze(-1)
            target_hw = render.shape[1:3]
            rendered_face_idx = render_res[kaolin.render.easy_render.RenderPass.face_idx]
            if rendered_face_idx.shape[1:] != target_hw:
                rendered_face_idx = torch.nn.functional.interpolate(
                    rendered_face_idx.unsqueeze(0).to(torch.float),
                    size=target_hw,
                    mode='nearest',
                )[0]
            geo_camera_normals = render_res['geo_camera_normals']
            if geo_camera_normals.shape[1:3] != target_hw:
                geo_camera_normals = torch.nn.functional.interpolate(
                    geo_camera_normals.permute(0, 3, 1, 2),
                    size=target_hw,
                    mode='bilinear',
                    align_corners=False,
                ).permute(0, 2, 3, 1)
            valid_faces, _, face_pixel_counts = gloss.utils.single_view.get_valid_faces(
                rendered_face_idx,
                geo_camera_normals,
                mesh.faces.shape[0],
                min_pixel_count=5,
                max_angle_deviation=args.max_angle_deviation,
            )
            sampling_weights = gloss.utils.single_view.get_triangle_sampling_weights(mesh.vertices, mesh.faces, face_pixel_counts)
            torchvision.utils.save_image(albedo.permute(0, 3, 1, 2), texture_path)
            torchvision.utils.save_image(albedo_debug.permute(0, 3, 1, 2), texture_debug_path)
            torch.save(sampling_weights, sampling_path)
