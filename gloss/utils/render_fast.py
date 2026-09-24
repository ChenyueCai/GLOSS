# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import torch
import kaolin as kal
from kaolin.render.easy_render.mesh import mesh_rasterize_interpolate_nvdiffrast
from kaolin.render.mesh.nvdiffrast_context import nvdiffrast_is_available, default_nvdiffrast_context

if nvdiffrast_is_available():
    import nvdiffrast.torch

from gloss.logging import log_tensor
from gloss.utils.render import world_normals_to_camera_normals_batched

logger = logging.getLogger(__name__)


def custom_mesh_batched_render(cameras, mesh, custom_texture, normals_texture=None, requires_positions=True,
                               process_as_albedo=True, backend=None):
    """ Renders all the channels we need for training/inference.

    Args:
        cameras: kaolin Camera containing a batch of cameras
        mesh: kaolin SurfaceMesh
        custom_texture: H x W x C
        normals_texture: H x W x 3 normals texture or None
        requires_positions: if positions should be rendered
        process_as_albedo: assumes that custom_texture is albedo with/without alpha and generates those channels,
           otherwise processing will be left to the caller of the function

    Returns:
       dictionary of tensors, including mask, camera_normals, face_idx, mask, normalized -1...1
       if process_as_albedo, will include 'albedo' and 'albedo_alpha', else will include 'textured' containing
       all the channels.
    """
    if requires_positions:
        mesh.face_features = mesh.face_vertices * 2
        mesh.vertex_features = None  # reset just in case

    if custom_texture.shape[-1] == 3 and process_as_albedo:
        custom_texture = torch.cat([custom_texture, torch.ones_like(custom_texture[..., :1])], dim=-1)

    res = fast_batched_render(cameras, mesh, custom_texture, normals_texture=normals_texture, backend=backend)
    final_res = {}
    final_res['normals'] = res['normals']
    final_res['mask'] = (res['face_idx'] >= 0).unsqueeze(-1).float()  # 0..1
    final_res['camera_normals'] = res['camera_normals']
    final_res['face_idx'] = res['face_idx'].unsqueeze(-1)
    final_res['uvs'] = res['uvs']

    if requires_positions:
        # Same as gloss.data.render_dataloader.RelativeToCenterValuePorocessor, but batched
        val = res['features']
        #log_tensor(val, 'features', logger)
        row, col = val.shape[1] // 2, val.shape[2] // 2
        center_val = val[:, row, col, :].unsqueeze(1).unsqueeze(1)
        #log_tensor(center_val, 'center_val', logger)
        #log_tensor(final_res['mask'], 'mask', logger)
        final_res['relative_positions'] = val - center_val * final_res['mask']
    if process_as_albedo:
        final_res['albedo'] = res['textured'][..., :-1] * final_res['mask'] * 2 - 1
        final_res['albedo_alpha'] = res['textured'][..., -1:] * final_res['mask'] * 2 - 1
    else:
        final_res['textured'] = res['textured']  # no processing
    final_res['mask'] = final_res['mask'] * 2 - 1  # consistent normalization
    return final_res


def fast_batched_render(cameras, mesh, custom_texture, normals_texture=None, backend=None):
    """ Renders custom texture, camera normals and positions ONLY. Does not work with multiple
    materials per mesh.

    Args:
        cameras: kaolin Camera containing a batch of cameras
        mesh: kaolin SurfaceMesh
        custom_n_channel_texture: H x W x C
        normals_texture: H x W x 3 normals texture or None

    Returns: dict with
            'textured': n_cameras x cam_height x cam_width x C
            'normals': n_cameras x cam_height x cam_width x 3
            'camera_normals': n_cameras x cam_height x cam_width x 3
            if mesh.face_features is set, will also include:
            'features': n_cameras x cam_height x cam_width x n_features
    """
    if backend is None:
        backend = "nvdiffrast" if nvdiffrast_is_available() else "cuda"

    if backend == "nvdiffrast":
        nvdiffrast_context = default_nvdiffrast_context(device=cameras.device, raise_error=True)

        face_idx, im_base_normals, im_tangents, uv_map, im_features = mesh_rasterize_interpolate_nvdiffrast(
            mesh, cameras, nvdiffrast_context,
            normals_required=True, uvs_required=True, tangents_required=True, features_required=True)
    elif backend == "cuda":
        face_idx, im_base_normals, im_tangents, uv_map, im_features = mesh_rasterize_interpolate_cuda(
            mesh, cameras, normals_required=True, uvs_required=True, tangents_required=True, features_required=True)
    else:
        raise ValueError(f'Unsupported backend {backend}, "nvdiffrast" and "cuda" are supported.')
    # log_tensor(face_idx, 'face_idx', logger, print_stats=True)
    # log_tensor(im_base_normals, 'normals', logger, print_stats=True)
    # log_tensor(uv_map, 'uv_map', logger, print_stats=True)

    # Now let's texture map
    texcoords = uv_map.reshape(uv_map.shape[0], 1, -1, 2).contiguous()
    if normals_texture is not None:
        stack_texture = torch.cat([normals_texture, custom_texture], dim=-1)
    else:
        stack_texture = custom_texture

    if backend == "nvdiffrast":
        textured = nvdiffrast.torch.texture(
                stack_texture.unsqueeze(0), texcoords, filter_mode='linear')  # [0, 0]  # first 2 dims are 1, 1
        # log_tensor(textured, 'textured', logger, print_stats=True)
        textured = textured.squeeze(1).reshape(face_idx.shape[0], face_idx.shape[1], face_idx.shape[2],
                                               stack_texture.shape[-1])
        # log_tensor(textured, 'textured, reshaped', logger, print_stats=True)
    elif backend == "cuda":
        texcoords[..., 1] = 1 - texcoords[..., 1]
        stack_texture = stack_texture.permute(2, 0, 1).unsqueeze(0).repeat(texcoords.shape[0], 1, 1, 1)
        textured = kal.render.mesh.texture_mapping(texcoords, stack_texture, mode='bilinear')
        textured = textured.squeeze(1).reshape(face_idx.shape[0], face_idx.shape[1], face_idx.shape[2], stack_texture.shape[1])

    perturbation_normal = None
    if normals_texture is not None:
        perturbation_normal = textured[..., :3]
        textured = textured[..., 3:]

    # Employ normals magic
    # TODO: need to do the sign thing from kaolin?
    im_bitangents = None
    if im_tangents is not None and im_base_normals is not None:
        im_bitangents = torch.nn.functional.normalize(torch.cross(im_tangents, im_base_normals), dim=-1)

    if perturbation_normal is not None and im_tangents is not None and im_bitangents is not None:
        #perturbation_normal = perturbation_normal.unsqueeze(0).unsqueeze(0)
        shading_normals = torch.nn.functional.normalize(
            im_tangents * perturbation_normal[..., :1]
            - im_bitangents * perturbation_normal[..., 1:2]
            + im_base_normals * perturbation_normal[..., 2:3],
            dim=-1
        )
        im_base_normals = shading_normals

    result = {'textured': textured,
              'normals': im_base_normals,
              'camera_normals': world_normals_to_camera_normals_batched(im_base_normals, cameras),
              'face_idx': face_idx,
              'uvs': uv_map}
    if im_features is not None:
        result['features'] = im_features

    return result


def mesh_rasterize_interpolate_cuda(
        mesh, camera, normals_required=True, uvs_required=True, tangents_required=True, features_required=True):
    """ Performs rasterization and interpolation using bundled Kaolin CUDA kernel. Returns image-space values, given
    camera resolution, for attributes that are required and available, and `None` for others.

    Args:
        mesh (SurfaceMesh): unbatched surface mesh
        camera (Camera): batched camera
        normals_required (bool): if True, will compute interpolated mesh normals, else return None
        uvs_required (bool): if True, and present in mesh, will compute interpolated mesh uvs, else return None
        tangents_required (bool): if True, will compute interpolated mesh tangents, else return None
        features_required (bool): if True, and present in mesh, will compute interpolated mesh features, else return None

    Returns:
        (tuple of): face_idx, im_normals, im_tangents, im_uvs, im_features
    """
    vertices_camera = camera.extrinsics.transform(mesh.vertices)
    vertices_image = camera.intrinsics.transform(vertices_camera)

    face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(vertices_camera, mesh.faces)
    face_vertices_image = kal.ops.mesh.index_vertices_by_faces(vertices_image, mesh.faces)[..., :2]

    in_face_features = []
    idx_normals = idx_uvs = idx_tangents = idx_features = -1
    current_idx = 0
    if normals_required:
        in_face_features.append(mesh.face_normals)
        idx_normals = current_idx
        current_idx += in_face_features[-1].shape[-1]
    if uvs_required and mesh.has_or_can_compute_attribute('face_uvs'):
        in_face_features.append(mesh.face_uvs)
        idx_uvs = current_idx
        current_idx += in_face_features[-1].shape[-1]
    if tangents_required and mesh.has_or_can_compute_attribute('face_tangents'):
        in_face_features.append(mesh.face_tangents)
        idx_tangents = current_idx
        current_idx += in_face_features[-1].shape[-1]
    if features_required and mesh.has_or_can_compute_attribute('face_features'):
        in_face_features.append(mesh.face_features)
        idx_features = current_idx
        current_idx += in_face_features[-1].shape[-1]

    if len(in_face_features) == 0:
        in_face_features = (torch.zeros(tuple(list(mesh.faces.shape) + [1]), dtype=camera.dtype, device=camera.device),)

    in_face_features = torch.cat(in_face_features, dim=-1).float()
    # added to support batched camera
    in_face_features = in_face_features[None].repeat(face_vertices_camera.shape[0], 1, 1, 1)
    face_features, face_idx = kal.render.mesh.rasterize(
        int(camera.height), int(camera.width),
        face_features=in_face_features,
        face_vertices_z=face_vertices_camera[..., -1],  # can be face_vertices_image[..., -1] instead?
        face_vertices_image=face_vertices_image)

    im_normals = im_uvs = im_tangents = im_features = None
    if idx_normals >= 0:
        im_normals = face_features[..., idx_normals:idx_normals+3]
    if idx_uvs >= 0:
        im_uvs = face_features[..., idx_uvs:idx_uvs+2] % 1.
    if idx_tangents >= 0:
        im_tangents = face_features[..., idx_tangents:idx_tangents+3]
    if idx_features >= 0:
        im_features = face_features[..., idx_features:]

    return face_idx, im_normals, im_tangents, im_uvs, im_features
