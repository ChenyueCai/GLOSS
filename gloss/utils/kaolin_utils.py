# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
import copy
import numpy as np
import kaolin
import torch


def apply_orientation_to_mesh(mesh, orientation_json_path) -> dict:
    """Rotate mesh.vertices in-place by the yaw/pitch/roll stored in
    ``orientation.json`` (Three.js XYZ Euler order, R = Rx(pitch) @ Ry(yaw) @ Rz(roll)).

    The single-view condition step (``run_condition_orientation.py`` /
    ``single_view_gen.load_mesh_for_gen``) applies this rotation BEFORE
    centering+normalizing the mesh. Any downstream stage that re-loads the
    raw .gltf must apply the same rotation, otherwise camera extrinsics in
    meta yamls (sampled against the rotated mesh) project onto the wrong
    faces — visible as scrambled UV textures after backprojection.

    Returns the parsed orientation dict ({"yaw","pitch","roll"} in radians).
    Missing file or all-zero angles is a no-op.
    """
    out = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    if orientation_json_path is None or not os.path.isfile(str(orientation_json_path)):
        return out
    with open(orientation_json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    for k in ("yaw", "pitch", "roll"):
        if k in data:
            out[k] = float(data[k])
    if all(abs(out[k]) < 1e-7 for k in ("yaw", "pitch", "roll")):
        return out
    dtype, device = mesh.vertices.dtype, mesh.vertices.device
    cx, sx = math.cos(out["pitch"]), math.sin(out["pitch"])
    cy, sy = math.cos(out["yaw"]),   math.sin(out["yaw"])
    cz, sz = math.cos(out["roll"]),  math.sin(out["roll"])
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=dtype, device=device)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=dtype, device=device)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=dtype, device=device)
    R = Rx @ Ry @ Rz
    mesh.vertices = mesh.vertices @ R.T
    return out


def load_mesh(mesh_path, device=torch.device('cuda')):
    mesh = kaolin.io.mesh.import_mesh(mesh_path)
    mesh = mesh.to(device)  # Let's move the mesh to cuda
    #ogger.info('original   ' + mesh.describe_attribute('vertices', print_stats=True))
    mesh.vertices = kaolin.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)
    return mesh


def compute_dimension_scales(points):
    """
    Computes scale of the points bounding box along all theree axes separately.

    Args:
        points: N x 3

    Returns:
       (3,) tensor
    """
    assert len(points.shape) == 2, f'Points have unexpected shape {points.shape}'

    vmin = points.min(dim=0, keepdim=True)[0]
    vmax = points.max(dim=0, keepdim=True)[0]
    scales = (vmax - vmin)
    return scales


def compute_normalization(points, eps=1e-6):
    """
    Computes normalization necessary to center and normalize a set of points (e.g. accourding to default
    kaolin function kaolin.ops.pointcloud.center_points, so that same normalization can be applied to different
    point sets.

    Args:
        points:
        eps:

    Returns:

    """
    assert len(points.shape) == 3, f'Points have unexpected shape {points.shape}'

    vmin = points.min(dim=1, keepdim=True)[0]
    vmax = points.max(dim=1, keepdim=True)[0]
    vmid = (vmin + vmax) / 2
    den = (vmax - vmin).max(dim=-1, keepdim=True)[0].clip(min=eps)
    return vmid, den


def normalize_points(points, vmid, den):
    """
    Same as kaolin.ops.pointcloud.center_points, but the normalization params can be computed separately
    using compute_normalization.

    Args:
        points:
        vmid:
        den:

    Returns:

    """
    res = points - vmid
    res = res / den
    return res


def all_vertices_in_faces(vertices, faces):
    idx = torch.unique(faces.reshape(-1))
    return vertices[idx, ...]


def split_mesh_into_part_meshes(mesh, face_part_assignments, renormalize=True):
    """
    Splits single mesh into many meshes based on face assignments. Not super efficient, as we don't split up
    the vertices, but keep all the redundant vertices and vertex attributes per mesh.

    Args:
        mesh: unbatched SurfaceMesh
        face_part_assignments: Nfaces x Nparts int tensor
        renormalize: if vertices of each part should be recentered and renormalized

    Returns:
        list of SurfaceMesh
    """
    # TODO: should add option to hide faces to easy_render instead
    meshes = []
    unique_ids = sorted([x.item() for x in torch.unique(face_part_assignments)])
    #print(unique_ids)

    for uid in unique_ids:
        new_faces = mesh.faces[face_part_assignments == uid]
        new_face_uvs_idx = mesh.face_uvs_idx[face_part_assignments == uid, ...]
        new_material_assignments = mesh.material_assignments[face_part_assignments == uid]

        new_vertices = mesh.vertices
        if renormalize:
            normalization = compute_normalization(all_vertices_in_faces(mesh.vertices, new_faces).unsqueeze(
                0))  # normalization based just on vertices in part faces
            new_vertices = normalize_points(mesh.vertices, *normalization).squeeze(0)

        # TODO: might need to handle more attributes; not fully thorough
        meshes.append(
            kaolin.rep.SurfaceMesh(vertices=new_vertices, faces=new_faces, uvs=mesh.uvs, face_uvs_idx=new_face_uvs_idx,
                                   vertex_normals=mesh.vertex_normals, vertex_tangents=mesh.vertex_tangents,
                                   material_assignments=new_material_assignments, materials=mesh.materials))
    return unique_ids, meshes


def create_lower_res_render_func(render_func, downscale_factor=8):
    def _new_closure(in_cam):
        cam = copy.deepcopy(in_cam)
        cam.width = in_cam.width // downscale_factor
        cam.height = in_cam.height // downscale_factor
        return render_func(cam)
    return _new_closure


def compute_cam_fov(intrinsics):
    # compute FOV from focal
    aspectScale = intrinsics.width / 2.0
    tanHalfAngle = aspectScale / intrinsics.focal_x.item()
    fov = np.arctan(tanHalfAngle) * 2
    return fov


def clone_camera_with_new_resolution(cam, new_width, new_height=None):
    if new_height is None:
        new_height = int(cam.height / cam.width * new_width)
    cam = copy.deepcopy(cam)
    cam.width = new_width
    cam.height = new_height
    return cam


def intrinsics_to_meta(intrinsics):
    return {
        'width': intrinsics.width,
        'height': intrinsics.height,
        'focal_x': intrinsics.focal_x.item(),
        'focal_y': intrinsics.focal_y.item(),
        'x0': intrinsics.x0.item(),
        'y0': intrinsics.y0.item(),
        'near': intrinsics.near,
        'far': intrinsics.far
    }

def intrinsics_from_meta(meta):
    """
    Expects parsed file like the following:
    {
        "width": 512,
        "height": 512,
        "focal_x": number,
        "focal_y": number,
        "x0": 0.0,
        "y0": 0.0,
        "near": 0.10000000149011612,
        "far": 100.0
    }

    Returns:
        (kaolin.render.camera.PinholeIntrinsics)
    """
    return kaolin.render.camera.PinholeIntrinsics.from_focal(**meta)

def extrinsics_to_meta(extrinsics):
    return {'view_matrix': extrinsics.view_matrix().detach().cpu().numpy().tolist()}


def extrinsics_from_meta(meta):
    """
    Expects parsed json like the following:
    "view_matrix": [
                [
                    0.9561348557472229,
                    0.29292652010917664,
                    -1.1175870007207322e-08,
                    -1.8727691173553467
                ],
                [
                    -0.021320868283510208,
                    0.06959300488233566,
                    0.9973475337028503,
                    -1.9461642503738403
                ],
                [
                    0.2921495735645294,
                    -0.9535987973213196,
                    0.07278575748205185,
                    -15.293581008911133
                ],
                [
                    -0.0,
                    0.0,
                    -0.0,
                    1.0
                ]
            ],
    Returns:
        (kaolin.render.camera.CameraExtrinsics)
    """
    view_matrix = torch.from_numpy(np.array(meta['view_matrix']))
    return kaolin.render.camera.CameraExtrinsics.from_view_matrix(view_matrix)


def camera_to_meta(cam):
    return {'intrinsics': intrinsics_to_meta(cam.intrinsics),
            'extrinsics': extrinsics_to_meta(cam.extrinsics)}


def camera_from_meta(meta):
    return kaolin.render.camera.Camera(extrinsics_from_meta(meta['extrinsics']), intrinsics_from_meta(meta['intrinsics']))


def hack_import_albedo_gltf_materials_raw(fname):
    from pygltflib import GLTF2, ImageFormat, BufferFormat

    gltf = GLTF2.load(fname)
    gltf.convert_buffers(BufferFormat.BINARYBLOB)
    gltf.convert_images(ImageFormat.BUFFERVIEW)

    materials = []
    for mat in gltf.materials:
        diffuse_texture = None
        if 'KHR_materials_pbrSpecularGlossiness' in mat.extensions:
            mat_inner = mat.extensions['KHR_materials_pbrSpecularGlossiness']
            if 'diffuseTexture' in mat_inner:
                diffuse_texture = kaolin.io.gltf._load_img(gltf, mat_inner['diffuseTexture']['index'], False).to(torch.float32) / 255.0
        elif mat.pbrMetallicRoughness is not None:
            mat_inner = mat.pbrMetallicRoughness
            if mat_inner.baseColorTexture is not None:
                diffuse_texture = kaolin.io.gltf._load_img(gltf, mat_inner.baseColorTexture.index, False).to(torch.float32) / 255.0
        materials.append(kaolin.render.materials.PBRMaterial(diffuse_texture=diffuse_texture, specular_color=(0.5, 0.5, 0.5),
                       is_specular_workflow=True,
                       material_name=mat.name))
    return materials
