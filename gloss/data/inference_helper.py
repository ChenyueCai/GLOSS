# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import os.path

import kaolin
import logging
import math
from omegaconf import OmegaConf
import torch
import torchvision
from tqdm import tqdm

import gloss.utils.kaolin_utils
import gloss.utils.render
from gloss.utils.render_fast import custom_mesh_batched_render
import gloss.utils.single_view
from gloss.logging import log_tensor, log_tensor_dict


logger = logging.getLogger(__name__)


def load_view_and_meta(meta_yml, base_color_path='auto', roughness_path='auto', metallic_path='auto'):
    """
    Loads single view and camera meta information. For example, given folder structure like this:
    koi_fish/single_view/civitai2.0/meta/view0001.yml
    koi_fish/single_view/civitai2.0/meta/view0002.yml
    koi_fish/single_view/civitai2.0/meta/view0003.yml
    koi_fish/single_view/civitai2.0/gen_view_decomposite/view0001.basecolor.png
    koi_fish/single_view/civitai2.0/gen_view_decomposite/view0001.metallic.png
    koi_fish/single_view/civitai2.0/gen_view_decomposite/view0001.roughness.png
    koi_fish/single_view/civitai2.0/gen_view_decomposite/view0002.basecolor.png
    koi_fish/single_view/civitai2.0/gen_view_decomposite/...

    will load camera, basecolor, roughness, metallic as follows:

    camera, channels, meta = load_view_and_meta("koi_fish/single_view/civitai2.0/meta/view0001.yml")

    # TODO: should also load super-resolution if available and upscale other material maps by default

    Args:
        meta_yml:
        base_color_path:
        roughness_path:
        metallic_path:

    Returns:

    """
    meta = OmegaConf.load(meta_yml)
    camera = gloss.utils.kaolin_utils.camera_from_meta(meta['camera'])

    view_base_dir = os.path.relpath(os.path.join(os.path.dirname(meta_yml), os.pardir, 'gen_view_decomposite'))
    basename = os.path.basename(meta_yml).split('.')[0]
    if base_color_path == 'auto':
        base_color_path = os.path.join(view_base_dir, f'%s.basecolor.png' % basename)
    if roughness_path == 'auto':
        roughness_path = os.path.join(view_base_dir, f'%s.roughness.png' % basename)
    if metallic_path == 'auto':
        metallic_path = os.path.join(view_base_dir, f'%s.metallic.png' % basename)

    channels = {}
    for k, p in zip(['albedo', 'roughness', 'metallic'], [base_color_path, roughness_path, metallic_path]):
        if p is not None:
            if k == 'albedo':
                channels[k] = kaolin.io.utils.read_image(p).unsqueeze(0)
            else:
                channels[k] = kaolin.io.utils.read_image(p)[..., :1].unsqueeze(0)
            channels[k] = channels[k] * 2.0 - 1.0  # keeping with convention that renders are -1...1

    return camera, channels, meta


# TODO(Masha): the interface to this could be cleaner
class SingleViewReferencesHelper:
    def __init__(self, mesh, camera_view, normal_map, known_rendered_channels,
                 view_resolution=256, textures_size=2048):
        """
        Assists with generating conditional views for texture/material completion based on rendered channels
        (albedo, roughness, etc.) for a single camera view.

        # TODO (Anita, Chenyue): you can maybe reuse large part of this for the FID computation; you'd just
        need to be able to initialize this not from single view, but complete texture

        Example:
            see notebooks/experimental/nearest_neighbor.ipynb
            need to initialize, then set_camera_samples, then precompute_references

        Args:
            mesh: kaolin SurfaceMesh
            camera_view: kaolin Camera
            known_rendered_channels: dict from string to 1 x cam_height x cam_width x C image -1...1 range
            normal_map: height x width x 3, -1...1 range, will be rescaled to textures_size
            view_resolution: how large reference view renders should be
            textures_size: size of the backprojected texture for single view material
        """
        self.mesh = mesh

        # Single known view
        self.camera = copy.deepcopy(camera_view)
        self._known_rendered_channels = known_rendered_channels  # renderings
        self._texture_size = textures_size
        self.normal_map = normal_map
        self.partial_material_maps = None  # texture maps from known_rendered_channels
        self.fused_partial_material_map = None
        self.name_to_channels = {}  # helps convert fused channels back to named channels
        self.generate_backprojected_materials()  # generates partial_material_map and name_to_channels

        # Generate cameras
        self._extrinsics_sampler = None
        self._intrinsics_sampler = None
        self._resolution = view_resolution
        self._references = None  # Can keep in memory; should be only about 2-3GB for 1000 256x256 samples
        self._num_views = None

    def set_camera_samplers(self, extrinsics_sampler, intrinsics_sampler, num_views):
        self._extrinsics_sampler = extrinsics_sampler
        self._intrinsics_sampler = intrinsics_sampler
        self._num_views = num_views
        self._extrinsics_sampler.set_samples(self._extrinsics_sampler.generate(num_views))  # TODO: awkward, fix API
        self._intrinsics_sampler.set_samples(self._intrinsics_sampler.generate(num_views))

    def get_camera(self, idx):
        extrinsics = self._extrinsics_sampler.get(idx)
        intrinsics = self._intrinsics_sampler.get(idx)
        return gloss.utils.render.make_camera_from_extr_intr(extrinsics, intrinsics, resolution=self._resolution, device=self.mesh.faces.device)

    def get_camera_batch(self, batch_id, batch_size):
        start_idx = batch_id * batch_size
        end_idx = min(start_idx + batch_size, self._num_views)
        return kaolin.render.camera.Camera.cat([self.get_camera(idx) for idx in range(start_idx, end_idx)])

    def _render_batch(self, cameras):
        r = custom_mesh_batched_render(cameras, self.mesh, self.fused_partial_material_map, self.normal_map,
                                       requires_positions=True, process_as_albedo=False)
        #log_tensor_dict(r, 'rendered_batch------------------', logger, print_stats=True)
        result = {}
        result['camera_normals'] = r['camera_normals']
        result['relative_positions'] = r['relative_positions']
        # result['face_idx'] = r['face_idx']  # TODO: add if needed
        for k, v in self.name_to_channels.items():
            result[k] = r['textured'][...,v[0]:v[0] + v[1]] * 2 - 1.0
        #log_tensor_dict(result, 'rendered processed batch -------------------', logger, print_stats=True)
        return result

    def precompute_references(self, batch_size=64):
        self.mesh.face_features = None
        self.mesh.vertex_features = None  # reset just in case

        num_batches = int(math.ceil(self._num_views / batch_size))
        print(f'num batches:   -------- {num_batches}')
        references = {}
        for batch in tqdm(range(num_batches)):
            cameras = self.get_camera_batch(batch, batch_size)
            r = self._render_batch(cameras)
            if len(references) == 0:
                references = {k: [v] for k, v in r.items()}
            else:
                for k, v in r.items():
                    references[k].append(v)
        self._references = {k: torch.cat(v, dim=0) for k, v in references.items()}

    def get_standard_reference_face_sampling_weights(self):
        # TODO: we can have a better sampler using the rendering and raycast
        face_pixel_counts = gloss.utils.single_view.get_face_pixel_counts(
            self.mesh, self.camera)
        weights = gloss.utils.single_view.get_ref_sampling_weights(
            self.mesh.vertices, self.mesh.faces, face_pixel_counts)
        #log_tensor(weights, 'sampling_weights', logger, print_stats=True)
        return weights

    def get_alternative_reference_face_sampling_weights(self):
        pass


    def _check_known_channel_dimensions(self):
        width = None
        height = None
        for ch, val in self._known_rendered_channels.items():
            kaolin.utils.testing.check_tensor(val, shape=(1, None, None, None), throw=True)
            assert val.shape[-1] < val.shape[1], f'Is your channel BCHW not BHWC? Shape: {val.shape}'
            if width is None:
                height = val.shape[1]
                width = val.shape[2]
            kaolin.utils.testing.check_tensor(val, shape=(1, height, width, None), throw=True)
        return width, height

    def _check_and_fix_normal_map(self):
        if self.normal_map is None:
            return
        kaolin.utils.testing.check_tensor(self.normal_map, shape=(None, None, 3), throw=True)
        expected_shape = [self.fused_partial_material_map.shape[0], self.fused_partial_material_map.shape[1]]
        if not kaolin.utils.testing.check_tensor(self.normal_map, shape=expected_shape + [3], throw=False):
            prev_shape = self.normal_map.shape
            self.normal_map = torchvision.transforms.Resize(expected_shape)(self.normal_map.permute(2, 0, 1)).permute(1, 2, 0)
            logger.info(f'Provided normal map rescaled to texture resolution: {prev_shape} --> {self.normal_map.shape}')

    def generate_backprojected_materials(self):
        width, height = self._check_known_channel_dimensions()
        self.camera.width = width
        self.camera.height = height

        render_res = gloss.utils.render.render_all_features(self.camera, self.mesh, lighting=None, required_passes=[kaolin.render.easy_render.RenderPass.face_idx,'geo_camera_normals'])
        render_res.update(self._known_rendered_channels)
        #log_tensor_dict(render_res, 'render_res', logger, print_stats=True)
        back_proj_res, mask = gloss.utils.single_view.backproject_render(
            self.mesh, self.camera, render_res,
            self._known_rendered_channels.keys(),
            texture_height=self._texture_size, texture_width=self._texture_size)
        # log_tensor_dict(back_proj_res, 'backproject_res (raw) ------- ', logger, print_stats=True)
        # log_tensor(mask, 'mask', logger, print_stats=True)
        # we'll use 4th channel of albedo as the alpha
        if 'albedo' in back_proj_res:
            back_proj_res['albedo_alpha'] = back_proj_res['albedo'][..., -1:]
            back_proj_res['albedo'] = back_proj_res['albedo'][...,:-1]
        else:
            back_proj_res['albedo_alpha'] = mask * 2.0 - 1.0
        # TODO: should multiply by mask?
        #back_proj_res['mask'] = mask * 2.0 - 1.0
        #log_tensor_dict(back_proj_res, 'backproject_res (processed) ------- ', logger, print_stats=True)

        start_idx = 0
        self.name_to_channels = {}
        self.partial_material_maps = {}
        fused = []
        for k, v in back_proj_res.items():
            self.name_to_channels[k] = (start_idx, v.shape[-1])
            start_idx += v.shape[-1]
            value = v / 2.0 + 0.5
            fused.append(value)  # Back to 0..1
            self.partial_material_maps[k] = value

        self.fused_partial_material_map = torch.cat(fused, dim=-1).squeeze(0)
        #log_tensor(self.fused_partial_material_map, 'fused material map', logger, print_stats=True)
        self._check_and_fix_normal_map()



