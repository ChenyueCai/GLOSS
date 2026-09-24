# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import kaolin
import logging
import math
import torch

from gloss.utils.geometry import curvature, merge_duplicate, merge_mapping, normalize

from gloss.viz.text import image_with_header
from gloss.logging import log_tensor

logger = logging.getLogger(__name__)

def world_normals_to_camera_normals(image_normals, camera):
    """ Converts world space normals rendered into an image given camera to normals relative to
    the camera.

    Args:
        image_normals:
        camera:

    Returns:

    """
    normals = image_normals.reshape(1, -1, 3)
    normals_camera = camera.extrinsics.transform(normals)
    normals_camera = normals_camera - camera.t.reshape(1, 1, 3)
    normals_camera = normals_camera.reshape(image_normals.shape)
    return normals_camera

def world_normals_to_camera_normals_batched(image_normals, cameras):
    """

    Args:
        im_normals:
        cameras:

    Returns:

    """
    normals = image_normals.reshape(image_normals.shape[0], -1, 3)
    normals_camera = cameras.extrinsics.transform(normals)
    normals_camera = normals_camera - cameras.t.reshape(len(cameras), 1, 3)
    normals_camera = normals_camera.reshape(image_normals.shape)
    return normals_camera


def render_all_features(cam, mesh, lighting,
                        required_passes=None,
                        **render_kwargs):
    """ Renders all features into a dictionary of features, appropriately normalized.
    All channels rescaled to -1..1 range, including masks and images.

    Can also render "depth", "geo_normals" (geometric normals, without using the normals texture) and
    "geo_camera_normals" (same, but relative to camera), "camera_normals" (normals that do use normals texture if provided,
    but relative to the camera), "positions" which are xyz positions, assuming mesh was pre-normalized.
    Also provides "albedo_alpha", i.e. the alpha map when rendering transparent texture, as well as optional
    raymaps "ray_directions", "ray_origins".

    Note: this function is not at all optimized!! Do provide required passes to avoid extra slowness.

    Args:
        required_passes: list of strings of required passes (else will render all)
        cam: kaolin camera to use
        mesh: kaolin SurfaceMesh to render
        lighting: kaolin SgLighting parameters
        render_kwargs: extra args to pass to kaolin.render.easy_render.render_mesh

    Returns:
        dictionary of all rendered passes
    """
    def _pass_required(name):
        return required_passes is None or name in required_passes

    def _get_materials():
        if 'custom_materials' in render_kwargs:
            return render_kwargs['custom_materials']
        else:
            return mesh.materials

    def _has_clear_material():
        materials = _get_materials()
        if materials is None:
            return False
        for m in materials:
            if type(m) == kaolin.render.materials.PBRMaterial:
                texture = getattr(m.hwc(), 'diffuse_texture', None)
                if texture is not None:
                    if texture.shape[-1] > 3:
                        return True
        return False

    def _set_texture_to_alpha(materials):
        for m in materials:
            if type(m) == kaolin.render.materials.PBRMaterial:
                texture = getattr(m.hwc(), 'diffuse_texture', None)
                if texture is not None:
                    if texture.shape[-1] > 3:
                        m.diffuse_texture = texture[..., 3:4].repeat(1, 1, 3)
                    else:
                        m.diffuse_texture = torch.ones_like(texture)
                # The alpha-extraction render hijacks diffuse_texture as a
                # geometry mask, so neutralize PBR factors that would otherwise
                # zero the rendered "albedo" — most importantly metallicFactor,
                # which kaolin defaults to 1.0 when the glTF omits it and which
                # would multiply the alpha output by (1 - metallic) = 0.
                if getattr(m, 'metallic_value', None) is not None:
                    m.metallic_value = torch.zeros_like(m.metallic_value)
                if getattr(m, 'metallic_roughness_texture', None) is not None:
                    m.metallic_roughness_texture = torch.zeros_like(m.metallic_roughness_texture)

    # We are going to add extra features to the mesh for all the extra channels we want
    # These variables help us keep track which feature channels correspond to which feature name
    name_idx_nchannels = {}
    current_idx = 0
    rendered_features = []

    # Render depth
    if _pass_required('depth'):
        # TODO: find a better way to render depth and add it to Kaolin!
        # Note: this normalization seems to look the best, but has not been thoroughly verified.
        vertices_camera = cam.extrinsics.transform(mesh.vertices)
        vertices_clip = cam.intrinsics.transform(vertices_camera)
        vertices_clip = vertices_clip[..., 2:3]
        vertices_clip = vertices_clip - vertices_clip.min()
        max_val = vertices_clip.max()
        eps = 1e-8
        if max_val < eps:
            max_val = eps
        vertices_clip /= max_val  # now 0...1
        vertices_clip = 1 - vertices_clip
        face_vertices_clip = kaolin.ops.mesh.index_vertices_by_faces(vertices_clip, mesh.faces).squeeze(0)
        rendered_features.append(face_vertices_clip)
        name_idx_nchannels['depth'] = (current_idx, face_vertices_clip.shape[-1])
        current_idx += face_vertices_clip.shape[-1]
    
    if _pass_required('raw_depth'):
        # Note: this is an unnormalized version of depth
        vertices_camera = cam.extrinsics.transform(mesh.vertices)
        vertices_clip = cam.intrinsics.transform(vertices_camera)
        vertices_clip = vertices_clip[..., 2:3]  # should be -1..1, but in fact is 0...1
        face_vertices_clip = kaolin.ops.mesh.index_vertices_by_faces(vertices_clip, mesh.faces).squeeze(0)
        rendered_features.append(face_vertices_clip)
        name_idx_nchannels['raw_depth'] = (current_idx, face_vertices_clip.shape[-1])
        current_idx += face_vertices_clip.shape[-1]
    
    if _pass_required('geo_normals') or _pass_required('geo_camera_normals'):
        rendered_features.append(mesh.face_normals)
        name_idx_nchannels['geo_normals'] = (current_idx, mesh.face_normals.shape[-1])
        current_idx += mesh.face_normals.shape[-1]

    if _pass_required('positions'):
        rendered_features.append(mesh.face_vertices * 2)
        name_idx_nchannels['positions'] = (current_idx, mesh.face_vertices.shape[-1])
        current_idx += mesh.face_vertices.shape[-1]
    # if _pass_required('gaussian_curvatures'):
    #     v, mp, _ = merge_mapping(mesh.vertices)
    #     f = merge_duplicate(mesh.faces, mp)
    #     gc = curvature(v, f).g[mp].unsqueeze(-1)
    #     ngc = normalize(gc)
    #     ngc = ngc[mesh.faces].to(mesh.faces.device)
    #     rendered_features.append(ngc)
    #     name_idx_nchannels['gaussian_curvatures'] = (current_idx, ngc.shape[-1])
    #     current_idx += ngc.shape[-1]

    if len(rendered_features) == 0:
        rendered_features = None
    elif len(rendered_features) == 1:
        rendered_features = rendered_features[0]
    else:
        rendered_features = torch.cat(rendered_features, dim=-1)

    # Let's set all the extra features we want to render
    mesh.face_features = rendered_features
    mesh.vertex_features = None  # reset just in case

    # Build a render-kwargs copy where the diffuse texture is RGB-only and
    # metallic is forced to 0. Without this, kaolin's texture_sample_materials
    # does `albedo *= (1 - metallic)` (mesh.py:330), which drives the rendered
    # albedo toward black wherever the metallic mask is high — and the
    # metallic mask can coincide with the basecolor's filled-vs-real region,
    # leaving "holes" in the rendered albedo that mirror albedo_alpha.
    # The original 4-channel texture and metallic_texture are preserved on
    # render_kwargs so the alpha-only render below still works correctly.
    main_render_kwargs = render_kwargs
    needs_override = _has_clear_material() or any(
        type(m) == kaolin.render.materials.PBRMaterial
        and getattr(m, 'metallic_texture', None) is not None
        for m in (_get_materials() or [])
    )
    if needs_override:
        main_render_kwargs = copy.deepcopy(render_kwargs)
        if 'custom_materials' not in main_render_kwargs:
            main_render_kwargs['custom_materials'] = copy.deepcopy(_get_materials())
        for m in main_render_kwargs['custom_materials']:
            if type(m) == kaolin.render.materials.PBRMaterial:
                tex = getattr(m.hwc(), 'diffuse_texture', None)
                if tex is not None and tex.shape[-1] > 3:
                    m.diffuse_texture = tex[..., :3].contiguous()
                m.metallic_texture = None
                m.metallic_value = 0.0
    render_res = kaolin.render.easy_render.render_mesh(cam, mesh, lighting=lighting, **main_render_kwargs)
    if _pass_required('camera_normals'):
        render_res['camera_normals'] = world_normals_to_camera_normals(
            render_res[kaolin.render.easy_render.RenderPass.normals], cam)
    render_res['normals'] = render_res['normals']

    if rendered_features is not None:
        rendered_features = render_res[kaolin.render.easy_render.RenderPass.features]

    # Now we'll unwrap all the rendered features into individual semantically named channels
    for name, val in name_idx_nchannels.items():
        render_res[name] = rendered_features[..., val[0]:val[0] + val[1]]

    if _pass_required('geo_camera_normals'):
        render_res['geo_camera_normals'] = world_normals_to_camera_normals(render_res['geo_normals'], cam)

    render_res['mask'] = (render_res['face_idx'] >= 0).unsqueeze(-1).float()

    if _pass_required('ray_directions') or _pass_required('ray_origins'):
        pixel_grid = kaolin.render.camera.raygen.generate_centered_custom_resolution_pixel_coords(
            img_width=cam.width, img_height=cam.height, res_x=cam.width, res_y=cam.height, device=cam.device
        )
        rays_o, rays_d = kaolin.render.camera.raygen.generate_pinhole_rays(cam, pixel_grid)
        rays_d = rays_d.reshape(1, cam.height, cam.width, 3)
        rays_o = rays_o.reshape(1, cam.height, cam.width, 3)
        render_res['ray_directions'] = rays_d * render_res['mask'].float()
        render_res['ray_origins'] = rays_o * render_res['mask'].float()

    # Remove features, no longer needed
    if kaolin.render.easy_render.RenderPass.features in render_res:
        del render_res[kaolin.render.easy_render.RenderPass.features]

    if _pass_required('albedo_alpha'):
        # TODO: this is *extremely* inefficient; should just fix TODO in kaolin/render/easy_render/mesh.py L317
        if _has_clear_material():
            materials = copy.deepcopy(_get_materials())
            _set_texture_to_alpha(materials)
            new_kwargs = copy.deepcopy(render_kwargs)
            new_kwargs['custom_materials'] = materials
            tmp_render_res = kaolin.render.easy_render.render_mesh(cam, mesh, lighting=lighting, **new_kwargs)
            # Binarize: the backprojected texture's alpha is 0/1 at texel
            # resolution but bilinear sampling makes it continuous at face
            # boundaries. Mark any non-zero coverage as 1 so albedo_alpha
            # captures every pixel touched by the backprojected texture.
            alpha = torch.clip(tmp_render_res['albedo'][..., :1], 0, 1)
            render_res['albedo_alpha'] = (alpha > 0.0).to(alpha.dtype) * render_res['mask']
        else:
            render_res['albedo_alpha'] = torch.ones_like(render_res['albedo'][..., :1]) * render_res['mask']

    render_res = {k: v for k, v in render_res.items() if _pass_required(k)}

    # Do consistent normalization of all the channels
    for name in [kaolin.render.easy_render.RenderPass.albedo,
                 kaolin.render.easy_render.RenderPass.render,
                 kaolin.render.easy_render.RenderPass.diffuse,
                 kaolin.render.easy_render.RenderPass.specular,
                 kaolin.render.easy_render.RenderPass.uvs,
                 "depth", "mask", "albedo_alpha"]:
        if name in render_res:
            render_res[name] = render_res[name] * 2 - 1

    return render_res


def make_camera_from_extr_intr(extrinsics, intrinsics, resolution=512, device=None):
    eye, lookat, up = extrinsics[:3], extrinsics[3:6], extrinsics[6:]
    return kaolin.render.camera.Camera(
        kaolin.render.camera.CameraExtrinsics.from_lookat(
                eye=eye,
                at=lookat,
                up=up,
                dtype=torch.float32,
                device=device),
        kaolin.render.camera.PinholeIntrinsics.from_fov(
                width=resolution,
                height=resolution,
                fov=intrinsics,
                device=device,
                dtype=torch.float32))
    

def make_local_camera(mesh_position, normal, prev_position, resolution=256, fov=1.0, debug_output=False):
    """Create camera pointing at a given position on the mesh.

    Args:
        mesh_position (torch.Tensor): 3D position on the mesh in world coordinates.
        normal (torch.Tensor): 3D normal of the mesh in world coordinates.
        prev_position (torch.Tensor): previous 3D position on the mesh in world coordinates.
        resolution (int): Resolution of the camera.
        fov (float): Field of view of the camera.
        debug_output (bool): if to print extra debug info

    Returns: kaolin camera object
    """
    up = prev_position - mesh_position  # vector pointing up from the camera
    eye = mesh_position + normal  # location of camera center

    if debug_output:
        print(f'Mesh position: {mesh_position}')
        print(f'Normal: {normal}')
        print(f'Up: {up}')
        print(f'Eye: {eye}')
        print(f'Fov {fov}')

    device = mesh_position.device

    return kaolin.render.camera.Camera(
        kaolin.render.camera.CameraExtrinsics.from_lookat(
                eye=eye,
                at=mesh_position,
                up=up,
                dtype=torch.float32,
                device=torch.device('cuda')),
        kaolin.render.camera.PinholeIntrinsics.from_fov(
                width=resolution,
                height=resolution,
                fov=fov,
                device=torch.device('cuda'),
                dtype=torch.float32))


def to_viz_channel(img, name='', normalize=True):
    if img.ndim == 3:
        img = img.unsqueeze(-1)
    if img.shape[-1] == 1:
        img = img.repeat(1,1,1,3)
    elif img.shape[-1] == 2:
        img = torch.cat([img, torch.zeros_like(img[..., -1:])], dim=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    elif img.shape[-1] == 3:
        img = img[..., :3]
    else:
        print(f'Unrenderable shape {img.shape}, skipping ({name})')
        return None

    if normalize:
        img = img.float()
        mn = img.min()
        if mn < 0:
            if mn >= -1:
                img = img + 1  # assume -1..1 range
            else:
                img = img - mn

        mx = img.max()
        if img.max() > 1:
            if mx <= 2:
                img = img / 2
            else:
                img = img / mx
    return img

def composite_rendering_result(render_res, dim=2):
    to_render = []
    for k, v in render_res.items():
        renderable = to_viz_channel(v, k)
        if renderable is not None:
            to_render.append(image_with_header(renderable, k))
    if len(to_render) == 1:
        res = to_render[0]
    else:
        res = torch.cat(to_render, dim=dim)
    if dim != 0:
        res = res.squeeze(0)
    return res


