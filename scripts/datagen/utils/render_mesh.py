# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import math
import os
import re
import sys
import torch
import torchvision


import kaolin
import kaolin.render.easy_render as easy_render

import gloss.logging

logger = logging.getLogger(__name__)


def _parse_color(in_str):
    try:
        color = [float(int(x)) / 255.0 for x in in_str.strip().split(',')]
        return (color[0], color[1], color[2])
    except Exception as e:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample script for rendering a mesh from a fixed viewpoint.')
    parser.add_argument('--mesh_filename', type=str, required=True,
                        help='Mesh filename in obj, usd or gltf format.')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Image resolution to render at.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory to write checkpoints to; must exist.')
    parser.add_argument('--base_name', type=str, default=None,
                        help='Base name to use; will determine automatically if not set.')
    parser.add_argument('--backend', type=str, default=None,
                        help='Backend to use for differentiable rendering, choose "nvdiffrast" or "cuda".')
    parser.add_argument('--use_default_material', type=str, default=None,
                        help='Set to 3 comma-deliminted integers of the diffuse color to use or blank to use default')
    gloss.logging.add_log_level_flag(parser)
    args = parser.parse_args()

    gloss.logging.default_log_setup(args.log_level)

    if not os.path.isdir(args.output_dir):
        raise RuntimeError(f'Output directory does not exist: --output_dir={args.output_dir}')

    # Read the mesh
    mesh = kaolin.io.import_mesh(args.mesh_filename, triangulate=True)
    mesh = mesh.cuda()
    mesh.vertices = kaolin.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)

    if args.use_default_material is not None:
        mesh.materials = [easy_render.default_material(_parse_color(args.use_default_material)).cuda()]
        mesh.material_assignments[...] = 0

    print(mesh)

    # Create a pinhole camera
    camera = easy_render.default_camera(args.resolution).cuda()

    # Create lighting
    lighting = easy_render.default_lighting().cuda()

    # Render the mesh
    res = easy_render.render_mesh(camera, mesh, lighting=lighting, backend=args.backend)
    logger.info(kaolin.utils.testing.tensor_info(res["render"], name='rendering', print_stats=True))

    # Write out the rendering
    bname = args.base_name
    if bname is None:
        bname = re.sub(r'[^a-zA-Z0-9_]+', '_', os.path.split(args.mesh_filename)[-1]) + \
                ('' if args.backend is None else f'_{args.backend}') + \
                ('' if args.use_default_material is None else '_defmat')
    for k, v in res.items():
        logger.info(kaolin.utils.testing.tensor_info(v, name=f'output of pass {k}', print_stats=True))
        fname = os.path.join(args.output_dir, bname + f'_{k}.png')
        if k == easy_render.RenderPass.render.name:
            torchvision.io.write_png((v.squeeze(0).clamp(0, 1) * 255).to(torch.uint8).cpu().permute(2, 0, 1), fname)

    print(f'Wrote to {args.output_dir}')

