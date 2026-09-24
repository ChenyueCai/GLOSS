# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import torch
import warp as wp

from gloss.logging import log_tensor
from gloss.utils.geometry import get_nullspace

logger = logging.getLogger(__name__)


def _sample_directions(normals, dirs_per_vec):
    # sample xy plane
    samples = (torch.randn((normals.shape[0], dirs_per_vec, 2), dtype=normals.dtype, device=normals.device) * 0.3).clip(
        min=-0.8, max=0.8)

    nullspace = get_nullspace(normals)

    # we'll use the samples to find coordinates on the nullspace plane
    res = torch.sum(samples.unsqueeze(-1) * nullspace.unsqueeze(1), dim=2) + normals.unsqueeze(1)
    res = res / torch.linalg.norm(res, dim=-1).unsqueeze(-1)
    return res


@wp.kernel
def _multi_raycast_wp_kernel(mesh: wp.uint64,
                              ray_origin: wp.array(dtype=wp.vec3),
                              ray_dir: wp.array2d(dtype=wp.vec3),
                              ray_hit: wp.array2d(dtype=wp.float32)):
    tid_o, tid_r = wp.tid()  # multiple rays per origin

    t = float(-1.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index

    # ray cast against the mesh
    if wp.mesh_query_ray(mesh, ray_origin[tid_o], ray_dir[tid_o, tid_r], 1.e+6, t, u, v, sign, n, f):
        # if we got a hit then set color to the face normal
        pass

    ray_hit[tid_o, tid_r] = t


def compute_shape_diameter(vertices, faces, vertex_normals, rays_per_vertex=40, remove_outliers=False, normalize=False):
    # create warp mesh
    wp_vertices = wp.from_torch(vertices, dtype=wp.vec3)
    wp_indices = wp.from_torch(faces.to(torch.int32).reshape(-1), dtype=wp.int32)
    wp_mesh = wp.Mesh(wp_vertices, wp_indices)

    directions = _sample_directions(vertex_normals * -1.0, rays_per_vertex)
    directions = wp.from_torch(directions, dtype=wp.vec3)

    # generate 20 rays per vertex pointing away from normal
    origins = vertices - 0.0001 * vertex_normals
    origins = wp.from_torch(origins, dtype=wp.vec3)
    ray_hits = wp.zeros(shape=(vertices.shape[0], rays_per_vertex), dtype=float, device=wp_vertices.device)

    # use warp to ray cast
    wp_device = wp_vertices.device

    wp.launch(kernel=_multi_raycast_wp_kernel,
              dim=(vertices.shape[0], rays_per_vertex),
              inputs=[wp_mesh.id, origins, directions, ray_hits],
              device=wp_device)

    # aggregate distances
    ray_hits = wp.to_torch(ray_hits)
    counts = torch.sum(ray_hits > 0, dim=-1)

    valid_hits = ray_hits.clone().clip(min=0.0)
    diameter = valid_hits.sum(dim=-1) / counts.clip(min=1.0)

    # where we did not get hits, we'll just fill in average diameter
    ave_diameter = diameter[counts > 0].mean()
    diameter[counts == 0] = ave_diameter

    if remove_outliers:
        highest = torch.quantile(diameter, 0.9)
        diameter = (diameter / highest).clip(max=1.0)
        if not normalize:
            diameter = diameter * highest
    elif normalize:
        highest = diameter.max()
        diameter = diameter / highest

    return diameter