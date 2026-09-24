# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import kaolin
import logging
import math
import os
import torch
import random

from gloss.logging import log_tensor, log_tensor_dict
import gloss.utils.render

logger = logging.getLogger(__name__)


def get_valid_faces(rendered_face_idx, rendered_geo_normals, num_faces,
                    min_pixel_count=1,
                    max_angle_deviation=math.pi * 0.4):
    """
    Heuristic for getting which faces contribute meaningfully to output rendering.

    Args:
        num_faces:
        min_pixel_count:
        max_angle_deviation:
        rendered_face_idx:
        rendered_geo_normals: in the non-image range; -1..1, all normalized

    Returns:
        tuple (valid_faces, alpha, face_pixel_counts) where
            valid_faces: num_faces bool array with True for all valid faces
            alpha: 1 x H x W float array with 1 for all valid pixels
            face_pixel_counts: num_faces long array with num pixels where each face was rasterized to

    """
    assert rendered_face_idx.shape[0] == 1, f'Only batch of size one supported'

    # Did a face get rasterized to this pixel?
    alpha = rendered_face_idx != -1

    # Did the rasterized face have a normal that's not too perpendicular to the camera?
    norm_alpha = torch.abs(rendered_geo_normals[0, :, :, -1]) > math.cos(max_angle_deviation)
    alpha = torch.logical_and(alpha, norm_alpha)

    valid_faces = torch.zeros((num_faces,), dtype=torch.bool, device=rendered_face_idx.device)
    visible_faces, counts = torch.unique(rendered_face_idx[0, ...][alpha[0, ...]], return_counts=True)
    face_pixel_counts = torch.zeros((num_faces,), dtype=counts.dtype, device=rendered_face_idx.device)
    visible_faces = visible_faces.to(torch.int64)
    face_pixel_counts[visible_faces] = counts
    visible_faces = visible_faces[counts >= min_pixel_count]
    valid_faces[visible_faces] = True

    alpha = alpha.to(torch.float32)
    return valid_faces, alpha, face_pixel_counts


def get_face_pixel_counts(mesh, camera):
    render_res = gloss.utils.render.render_all_features(
        camera, mesh, lighting=None,
        required_passes=[kaolin.render.easy_render.RenderPass.face_idx, 'geo_camera_normals'])
    _, _, face_pixel_counts = get_valid_faces(
        render_res[kaolin.render.easy_render.RenderPass.face_idx],
        render_res["geo_camera_normals"],
        mesh.faces.shape[0],
        min_pixel_count=5,
        max_angle_deviation=math.pi * 0.3,
    )
    return face_pixel_counts


def _compute_discrete_face_weights_ref(face_areas, face_pixel_counts, quantile_max=0.8, quantile_min=0.6):
    ref_face_weights = torch.zeros_like(face_areas)
    threshold_1 = torch.quantile(face_pixel_counts[face_pixel_counts > 0].float(), quantile_min)
    ref_face_weights[face_pixel_counts >= threshold_1] = 1
    threshold_2 = torch.quantile(face_pixel_counts[face_pixel_counts > 0].float(), quantile_max)
    ref_face_weights[face_pixel_counts >= threshold_2] = 2
    sampling_weights = face_areas * ref_face_weights
    return sampling_weights


def _compute_discrete_face_weights(face_areas, face_pixel_counts, quantile_max, quantile_min, min_face_pixel_count=3):
    face_weights = torch.zeros_like(face_areas)
    q = quantile_max
    thresh = torch.quantile(face_pixel_counts[face_pixel_counts > 0].float(), q)
    face_weights[face_pixel_counts >= thresh] = 2
    face_weights[face_pixel_counts < thresh] = 1
    q = quantile_min
    thresh = torch.quantile(face_pixel_counts[face_pixel_counts > 0].float(), q)
    face_weights[face_pixel_counts < thresh] = 0.1
    face_weights[face_pixel_counts <= min_face_pixel_count] = 0

    return face_areas * face_weights


def get_ref_sampling_weights(vertices, faces, face_pixel_counts):
    face_areas = (
            kaolin.ops.mesh.face_areas(vertices.unsqueeze(0), faces).squeeze(0)
            * 1000
    )
    return _compute_discrete_face_weights_ref(face_areas, face_pixel_counts)


def get_eval_sampling_weights(vertices, faces, face_pixel_counts):
    """
    Heuristic to come up with face sampling weights that prioritize both larger area faces
    of the mesh, and also faces that occur in a single rendered view.

    Assumes mesh positions to be normalized to a reasonable volume (~unit).

    Args:
        vertices: (num_verts, 3) float array of vertex positions
        faces: (num_faces, 3) int array of face vertex indices
        face_pixel_counts: (num_faces,) int array of number of pixels hit per face (more is better)

    Returns:
        (num_faces,) float array of sampling weights
    """
    face_areas = kaolin.ops.mesh.face_areas(vertices.unsqueeze(0), faces).squeeze(0) * 1000
    return _compute_discrete_face_weights(face_areas, face_pixel_counts, quantile_max=0.99, quantile_min=0.98)


def get_triangle_sampling_weights(vertices, faces, face_pixel_counts):
    """
    Heuristic to come up with face sampling weights that prioritize both larger area faces
    of the mesh, and also faces that occur in a single rendered view.

    Assumes mesh positions to be normalized to a reasonable volume (~unit).

    Args:
        vertices: (num_verts, 3) float array of vertex positions
        faces: (num_faces, 3) int array of face vertex indices
        face_pixel_counts: (num_faces,) int array of number of pixels hit per face (more is better)

    Returns:
        (num_faces,) float array of sampling weights
    """
    face_areas = kaolin.ops.mesh.face_areas(vertices.unsqueeze(0), faces).squeeze(0) * 1000
    return _compute_discrete_face_weights(face_areas, face_pixel_counts, quantile_max=0.7, quantile_min=0.5)


def get_tex_face_idx(mesh, h, w):
    face_vertices = kaolin.ops.mesh.index_vertices_by_faces(mesh.vertices.unsqueeze(0), mesh.faces)
    face_uvs = mesh.face_uvs.tile((1, 1, 1, 1))
    face_uvs[..., 1] = 1 - face_uvs[..., 1]
    face_vertices = kaolin.ops.mesh.index_vertices_by_faces(mesh.vertices.unsqueeze(0), mesh.faces)
    face_uvs = mesh.face_uvs.tile((1, 1, 1, 1))
    face_uvs[..., 1] = 1 - face_uvs[..., 1]
    face_uvs = face_uvs * 2 - 1
    _, tex_face_idx = kaolin.render.mesh.rasterize(
        h,
        w,
        face_features=face_vertices[..., :2] / 2 + 0.5,
        face_vertices_z=torch.zeros_like(face_vertices[..., -1]),
        face_vertices_image=face_uvs,
    )
    return tex_face_idx
    
def get_valid_faces_from_texture(mesh, texture, all_filled=True):
    tex_face_idx = get_tex_face_idx(mesh, texture.shape[2], texture.shape[3])
    all_face_ids, counts = torch.unique(tex_face_idx[tex_face_idx != -1], return_counts=True)
    all_counts = torch.zeros(mesh.faces.shape[0]).long().to(mesh.faces.device)
    all_counts[all_face_ids] = counts
    cond_texture_mask = texture[:, 3] == 1
    masked_tex_face_idx = tex_face_idx[cond_texture_mask]
    init_face_ids, init_counts = torch.unique(masked_tex_face_idx[masked_tex_face_idx != -1], return_counts=True)
    if not all_filled:
        return init_face_ids
    filled_faces = all_counts[init_face_ids] == init_counts
    valid_face_ids = init_face_ids[filled_faces]
    return valid_face_ids


def backproject_render(mesh, camera, render_res, channels, texture_height, texture_width,
                       min_pixel_count=2, max_angle_deviation=math.pi * 0.3,
                       upscale=False, return_face_idx=False, sample_mode='nearest'):
    """
    Back projects single camera view to the object.

    Args:
        mesh:
        camera:
        render_res:
        channels: channels in render_res that should be back-projected
        texture_height:
        texture_width:
        min_pixel_count: minimum pixel count for a face to be included
        max_angle_deviation: maximum angle deviation from perpendicular to camera plane for a face to be included
            (both back and forward normals are accepted, the view just can't be too perpendicular to the
            camera plane)
        sample_mode: 'nearest' (default) or 'bilinear' filtering when each texel pulls its color from the
            view image. Nearest turns every view pixel into a solid block of texels (16x16 at 4K from a
            256 px view) which shows as sawtooth edges; bilinear removes that at the cost of softness.
    Returns:
        dictinoary with results of backprojecting each selected channel
    """
    # Figure out which faces are valid
    rendered_face_idx = render_res[kaolin.render.easy_render.RenderPass.face_idx]
    target_hw = None
    if upscale:
        target_h = max(render_res[channel].shape[1] for channel in channels)
        target_w = max(render_res[channel].shape[2] for channel in channels)
        target_hw = (target_h, target_w)
        if rendered_face_idx.shape[1:] != target_hw:
            rendered_face_idx = torch.nn.functional.interpolate(
                rendered_face_idx.unsqueeze(0).to(torch.float),
                size=target_hw,
                mode='nearest',
            )[0]

    for channel in channels:
        if upscale:
            in_render = render_res[channel].permute(0, 3, 1, 2)
            if in_render.shape[-2:] != target_hw:
                in_render = torch.nn.functional.interpolate(
                    in_render,
                    size=target_hw,
                    mode='bilinear',
                    align_corners=False
                )
            render_res[channel] = in_render.permute(0, 2, 3, 1)

    if upscale and render_res['geo_camera_normals'].shape[1:3] != target_hw:
        render_res['geo_camera_normals'] = torch.nn.functional.interpolate(
            render_res['geo_camera_normals'].permute(0, 3, 1, 2),
            size=target_hw,
            mode='bilinear',
            align_corners=False
        )
        render_res['geo_camera_normals'] = render_res['geo_camera_normals'].permute(0, 2, 3, 1)
    
    valid_faces, alpha, _ = get_valid_faces(rendered_face_idx,
        render_res['geo_camera_normals'],
        mesh.faces.shape[0],
        min_pixel_count=min_pixel_count,
        max_angle_deviation=max_angle_deviation
    )
    alpha = (rendered_face_idx != -1).to(torch.float32)  # no need to apply aggressive alpha to source image

    if valid_faces.sum() == 0:
        if return_face_idx:
            return torch.ones((1, 4, texture_height, texture_width), dtype=torch.float32, device=mesh.faces.device) * -1.0, \
            torch.zeros((1, 1, texture_height, texture_width), dtype=torch.float32, device=mesh.faces.device), None
        return torch.ones((1, 4, texture_height, texture_width), dtype=torch.float32, device=mesh.faces.device) * -1.0, \
            torch.zeros((1, 1, texture_height, texture_width), dtype=torch.float32, device=mesh.faces.device)

    vertices_image = camera.transform(mesh.vertices)
    face_vertices_image = kaolin.ops.mesh.index_vertices_by_faces(vertices_image.unsqueeze(0), mesh.faces)
    face_uvs = mesh.face_uvs.tile((1, 1, 1, 1))
    face_uvs[..., 1] = 1 - face_uvs[..., 1]
    face_uvs = face_uvs * 2 - 1
    tex_image_features, tex_face_idx = kaolin.render.mesh.rasterize(
        texture_height,
        texture_width,
        face_features=face_vertices_image[..., :2] / 2 + 0.5,
        face_vertices_z=torch.zeros_like(face_vertices_image[..., -1]),
        face_vertices_image=face_uvs,
        valid_faces=valid_faces.unsqueeze(0)
    )

    mask = (tex_image_features[..., 0] != 0) | (tex_image_features[..., 1] != 0) # regions that are not backprojected
    mask = mask.int().unsqueeze(-1)
    
    res = {}
    for channel in channels:
        in_render = render_res[channel].permute(0, 3, 1, 2)  # -1 .. 1

        if in_render.shape[1] == 3:
            rgb = in_render
            new_alpha = alpha.unsqueeze(1) # 0..1
        else:
            rgb = in_render[:, :-1, ...]
            new_alpha = (in_render[:, -1:, ...] / 2.0 + 0.5) * alpha.unsqueeze(1)  # rescale to 0..1 for multiplication to work
        in_render = torch.cat([rgb, new_alpha * 2 - 1.0], dim=1)
        if sample_mode == 'bilinear':
            # Colours interpolate, but the write mask (last channel, in {-1, +1}) must
            # stay hard: a fractional mask lets texels whose sample straddles the
            # silhouette or the update boundary through with a near-zero alpha and a
            # colour blended with background, which shows up as dark speckles.
            rgb_s = kaolin.render.mesh.texture_mapping(tex_image_features, in_render[:, :-1], mode='bilinear')
            a_s = kaolin.render.mesh.texture_mapping(tex_image_features, in_render[:, -1:], mode='nearest')
            tmp_texture = torch.cat([rgb_s, a_s], dim=-1)
        else:
            tmp_texture = kaolin.render.mesh.texture_mapping(tex_image_features, in_render, mode=sample_mode)
        res[channel] = tmp_texture

    if return_face_idx:
        return res, mask, tex_face_idx
        
    return res, mask

def random_polygon_mask(resolution, batchsize, device="cuda", mean_radius=0.5, std_radius=0.2, full_ratio=0.1):
    full_mask_indices = (torch.rand(batchsize) < full_ratio).nonzero(as_tuple=True)[0]
    targs = {"device": device}
    center = torch.rand((batchsize, 2), **targs)
    nverts = random.randint(4, 7)
    radii = torch.clip(torch.randn((batchsize, nverts), **targs) * std_radius + mean_radius,
                       min=max(0.01, mean_radius - std_radius),
                       max=min(0.8, mean_radius + std_radius))
    angles = torch.rand((batchsize, nverts + 1), **targs)
    angles = torch.sort(angles, dim=-1).values
    angles = torch.softmax(angles, dim=-1)
    angles = torch.cumsum(angles, dim=-1)
    angles = angles * torch.pi * 2
    angles = angles[:, :-1]
    x = torch.cos(angles) * radii
    y = torch.sin(angles) * radii
    # do a fan around center
    nfaces = nverts
    faces = torch.zeros((nfaces, 3), dtype=torch.long, **targs)
    faces[:, 1] = torch.arange(1, nverts + 1, **targs)
    faces[:, 2] = torch.arange(2, nverts + 2, **targs)
    faces[-1, 2] = 1

    center = center.unsqueeze(1)
    vertices = torch.cat([center, torch.stack([x, y], dim=-1) + center], dim=1) * 2 - 1

    # Let's rasterize it now
    vertices = torch.cat([vertices, torch.zeros_like(vertices[..., :1])], dim=-1)
    mesh = kaolin.rep.SurfaceMesh(vertices=vertices, faces=faces)
    mesh.to_batched()
    image_features, tex_face_idx = kaolin.render.mesh.rasterize(
        resolution,
        resolution,
        face_features=torch.ones_like(mesh.face_vertices),
        face_vertices_z=torch.zeros_like(mesh.face_vertices[..., -1]),
        face_vertices_image=mesh.face_vertices[..., :2]
    )
    mask = (tex_face_idx < 0).float().unsqueeze(-1)
    mask[full_mask_indices, ...] = torch.zeros_like(mask[full_mask_indices, ...])
    return mask


def setup_mesh_single_view(camera, mesh, texture_resolution, custom_view=None, channels=None,
                           output_dir=None, output_prefix=''):
    """
    Given single view rendering of the mesh, performs necessary steps to back-project the view,
    and to be able to sample the mesh locally around the faces where it is appearance was rendered.

    Args:
        camera: camera from which the mesh was rendred (kal.render.camera.Camera)
        mesh: kal.rep.SurfaceMesh object, unbatched; must have UVs.
        texture_resolution: texture resolution to use for back-projected textures
        custom_view: 1 x 3 x H x W float tensor in range -1..1
        channels: if backprojecting standard rendering; channels that should be back-projected (if none,
            will default to albedo channel)
        output_dir: if set, will save textures and sampling weights there (will overwrite files, dir must exist)
        output_prefix: if saving, will prefix saved files with this

    Returns:
        dict of backprojected textures, sampling weights per face of the mesh
    """
    if channels is None:
        channels = []

    if custom_view is not None:
        assert custom_view.ndim == 4
        assert custom_view.shape[0] == 1
        assert custom_view.shape[1] <= 4

        highres_camera = copy.deepcopy(camera)
        highres_camera.height = custom_view.shape[2]
        highres_camera.width = custom_view.shape[3]
        channels.append('custom')
    else:
        highres_camera = camera
        if len(channels) == 0:
            channels = ['albedo']

    render_res = gloss.utils.render.render_all_features(highres_camera, mesh, lighting=None)
    if custom_view is not None:
        render_res['custom'] = custom_view

    # Backproject texture
    back_proj_res, _ = backproject_render(mesh, highres_camera, render_res, channels,
                                       texture_resolution, texture_resolution,
                                       min_pixel_count=2,
                                       max_angle_deviation=math.pi * 0.3)
    log_tensor_dict(back_proj_res, 'backproject_res', logger, print_stats=True)

    # For sampling camera views, we are more conservative and discard faces that hit less than 5 pixels
    valid_faces, _, face_pixel_counts = get_valid_faces(
        render_res[kaolin.render.easy_render.RenderPass.face_idx],
        render_res['geo_camera_normals'],
        mesh.faces.shape[0],
        min_pixel_count=5,
        max_angle_deviation=math.pi * 0.3
        )
    sampling_weights = get_triangle_sampling_weights(mesh.vertices, mesh.faces, face_pixel_counts)
    log_tensor(sampling_weights, 'sampling_weights', logger, print_stats=True)

    if output_dir is not None:
        for k, v in back_proj_res.items():
            img_path = os.path.join(output_dir, f'{output_prefix}{k}.png')
            if os.path.exists(img_path):
                logger.warning(f'Overwriting path: {img_path}')
            kaolin.io.utils.write_image(v / 2.0 + 0.5, img_path)
            logger.info(f'Saved {img_path}')

        weights_path = os.path.join(output_dir, 'face_weights.pt')
        torch.save(sampling_weights, weights_path)
        logger.info(f'Saved {weights_path}')

    return back_proj_res, sampling_weights


def make_single_view_cam(azi_range:list, elev_range:list, view_dist_range:list, fov_range:list, resolution:int, device, at=None):
    """_summary_

    Args:
        azi_range (list):
        elev_range (list): _description_
        view_dist_range (list): _description_
        fov_range (list): _description_
        resolution (int): _description_
        device (_type_): _description_
        at: optional 3-element look-at target. Defaults to origin. Used so
            callers can point the camera at an off-origin anchor while keeping
            the orbital azi/elev parametrization around that anchor.

    Returns:
        _type_: _description_
    """
    azimuth = torch.rand(1) * (azi_range[1]-azi_range[0]) +azi_range[0]
    elevation = torch.rand(1) * (elev_range[1]-elev_range[0]) +elev_range[0]
    r = torch.rand(1) * (view_dist_range[1]-view_dist_range[0]) +view_dist_range[0]
    fov = torch.rand(1) * (fov_range[1]-fov_range[0]) +fov_range[0]
    direction = kaolin.render.lighting.sg_direction_from_azimuth_elevation(azimuth, elevation)
    world_up = torch.tensor([0.0, 1.0, 0.0])
    up = world_up - direction[:, 1][:, None] * direction
    if at is None:
        at_t = torch.tensor([0.0, 0.0, 0.0])
    else:
        at_t = torch.as_tensor(at, dtype=torch.float32).reshape(3)
    eye = direction * r + at_t

    camera = kaolin.render.camera.Camera(
        kaolin.render.camera.CameraExtrinsics.from_lookat(
                eye=eye,
                at=at_t,
                up=up,
                dtype=torch.float32,
                device=device),
        kaolin.render.camera.PinholeIntrinsics.from_fov(
                width=resolution,
                height=resolution,
                fov=fov,
                device=device,
                dtype=torch.float32))
    return camera


class SingleViewCameraExtrinsicsSampler:
    def __init__(self, azim_range, elev_range, view_dist_range, debug=False):
        self.azim_range = azim_range # [[min val], [max val]]
        self.elev_range = elev_range
        self.view_dist_range = view_dist_range
        self.debug = debug
        
    def _sample_from_range(self, ranges, n):
        if isinstance(ranges[0], float):
            ranges = [[ranges[0]], [ranges[1]]]
        assert isinstance(ranges[0], list) and isinstance(ranges[1], list)
        mins, maxs = [torch.as_tensor(v) for v in ranges]

        if mins.shape != maxs.shape:
            raise ValueError("Min and max lists must be the same length.")
        if torch.any(maxs < mins):
            raise ValueError("Each max must be ≥ its corresponding min.")
        mins  = mins.unsqueeze(-1)               # (k, 1)
        maxs  = maxs.unsqueeze(-1)
        u = torch.rand(len(mins), n)  # (k, n)
        sample = mins + (maxs - mins) * u       # element‑wise linear interpolation
        sample = torch.flatten(sample)
        return sample[torch.randperm(sample.numel())][:n]
        
        
    def generate(self, num_samples):
        if self.debug:
            azimuths = torch.tensor([0, torch.pi / 2, torch.pi, torch.pi * 3 / 2, 0, 0, 0, 0])
            elevations = torch.tensor([0, 0, 0, 0, torch.pi/2, torch.pi/4, -torch.pi/4, -torch.pi/2])
            view_dists = [0.85, 1.0, 1.15, 1.25]
            azimuths = azimuths.repeat(4)
            elevations = elevations.repeat(4)
            view_dists = torch.cat([torch.ones(8) * i for i in view_dists])
        else:
            self.num_samples = num_samples
            azimuths = self._sample_from_range(self.azim_range, num_samples)
            elevations = self._sample_from_range(self.elev_range, num_samples)
            view_dists = self._sample_from_range(self.view_dist_range, num_samples)
        eye = kaolin.render.lighting.sg_direction_from_azimuth_elevation(azimuths, elevations)
        world_up = torch.tensor([0.0, 1.0, 0.0]) 
        up = world_up - eye[:, 1][:, None] * eye  
        eye = eye * view_dists[:, None]
        at = torch.zeros_like(eye)
        return torch.cat([eye, at, up], axis=1)
    
    def set_samples(self, samples):
        self.samples = samples
        self.num_samples = samples.shape[0]
    
    def get(self, idx):
        return self.samples[idx % self.num_samples]
    
    def get_hash(self, idx):
        sample = self.get(idx)
        sample_str = ['%.4f' % s.item() for s in sample]
        e_hash = '_'.join(sample_str[:3])
        a_hash = '_'.join(sample_str[3:6])
        u_hash = '_'.join(sample_str[6:])
        hash = f'e{e_hash}a{a_hash}u{u_hash}'
        return hash


def cartesian2spherical(coordinates:torch.Tensor):
    return kaolin.ops.coords.cartesian2spherical(coordinates[2], coordinates[0], coordinates[1])
