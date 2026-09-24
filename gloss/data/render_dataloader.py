# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os, io, copy
import imageio
import zipfile
from dataclasses import dataclass, field
import logging
import random
import math
from tqdm import tqdm
import time

import numpy as np

import torch
import torchvision.io.image
from torch.utils.data import Dataset

import kaolin

import gloss.utils.color
import gloss.utils.render
from gloss.logging import log_tensor, log_tensor_dict
from gloss.data.utils import (
    apply_excluded_train_view_ids,
    build_split_dict_for_view_ids,
    generate_split_indices,
    load_excluded_train_view_ids,
    load_split_indices,
    repair_degenerate_albedo_alpha,
    save_split_indices,
)
from gloss.utils.single_view import get_valid_faces, SingleViewCameraExtrinsicsSampler, cartesian2spherical, backproject_render
from gloss.utils.kaolin_utils import hack_import_albedo_gltf_materials_raw

class MyTimer:
    def __init__(self, message):
        self.message = message
    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.cuda.synchronize()
        print(self.message, ":", time.time() - self.start_time)

logger = logging.getLogger(__name__)


class FovSampler:
    def __init__(self, min_fov, max_fov):
        self.max_fov = max_fov
        self.min_fov = min_fov
        self.samples = None

    def generate(self, num_samples):
        self.num_samples = num_samples
        samples = torch.rand((num_samples)) * (self.max_fov - self.min_fov) + self.min_fov
        return samples

    def set_samples(self, samples):
        self.num_samples = samples.shape[0]
        self.samples = samples

    def get(self, idx):
        return self.samples[idx % self.num_samples]

    def get_hash(self, idx):
        sample = self.get(idx)
        sample_str = '%0.1f' % sample.item()
        return f"f{sample_str}"



class FixedLightSampler:
    def __call__(self, *args, **kwargs):
        return kaolin.render.easy_render.default_lighting()


def get_lighting_from_cam(camera):
    light_sampler = NormalLightSampler(0, 0)
    cam_normals = copy.deepcopy(camera.t) #camera.extrinsics.cam_forward())
    lighting = light_sampler(cam_normals.flatten()).cuda()
    return lighting


class NormalLightSampler:
    """
    Samples Spherical Gaussian lighting parameters based on a normal direction of the pixel the
    camera is pointing at. Randomizes direction without causing this part of the mesh to be backlit
    (e.g. if the direction is on the opposite side).
    """
    def __init__(self, min_angle_from_normal=torch.pi * 0.1, max_angle_from_normal=torch.pi * 0.5):
        # TODO: add amplitude, sharpness and also color
        self.angle_deviation = (min_angle_from_normal, max_angle_from_normal)

    def sample_light_direction(self, normal):
        # Get perpendicular direction, dot product guaranteed zero:
        # n[idx[0]] * n[idx[1]] - n[idx[0]] * n[idx[1]] + 0
        largest_dims = torch.topk(normal, 3)[1]
        perp_direction = torch.zeros_like(normal)
        perp_direction[largest_dims[1]] = -normal[largest_dims[0]]
        perp_direction[largest_dims[0]] = normal[largest_dims[1]]
        perp_direction[largest_dims[2]] = 0

        # Randomly rotate perpendicular vector around normal to get a random perpendicular vector
        vec_args = {'device': normal.device, 'dtype': normal.dtype}
        random_axis = kaolin.math.quat.quat_rotate(
            kaolin.math.quat.quat_from_angle_axis(
                torch.rand((1, 1), **vec_args) * torch.pi * 2, normal.unsqueeze(0)),
            perp_direction.unsqueeze(0))

        # Now let's rotate the normal around the random axis
        random_angle = torch.rand((1, 1), **vec_args) * (self.angle_deviation[1] - self.angle_deviation[0]) + \
                       self.angle_deviation[0]
        return kaolin.math.quat.quat_rotate(
            kaolin.math.quat.quat_from_angle_axis(random_angle, random_axis),
            normal.unsqueeze(0))

    def __call__(self, normal):
        direction = self.sample_light_direction(normal)
        return kaolin.render.lighting.SgLightingParameters(amplitude=3., direction=direction, sharpness=5.)


class RelativeToCenterValueProcessor:
    """
    Processes the output of the renderer (render_all_features function) and applies normalization
    to a specific channel (e.g. positions) that makes all positions relative to the
    center pixel position.
    """
    def __init__(self, input_channel='positions', output_channel='relative_positions',
                 use_mask=True):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.use_mask = use_mask

    def __call__(self, render_res):
        val = render_res[self.input_channel]

        row, col = val.shape[1] // 2, val.shape[2] // 2
        center_val = val[:, row, col, :]
        mask = 1.0
        if self.use_mask and 'mask' in render_res:
            mask = render_res['mask'].float() / 2 + 0.5
        render_res[self.output_channel] = val - center_val * mask
        return render_res


class PosterizationProcessor:
    """
    Processes the output of the renderer (render_all_features function) and adds a posterization result over the
    configured input channel, adding a value with the configured channel to the output.

    Provide either resolution, or low-level parameters to gloss.utils.color.posterize_image.
    """
    def __init__(self, input_channel='albedo', output_channel='post', max_lab_distance=10.0,
                 resolution=None, blur_kernel=None, blur_sigma=None, n_superpixels=None):
        self.input_channel = input_channel
        self.output_channel = output_channel

        self.posterization_args = {'max_lab_distance': max_lab_distance}
        if resolution is not None:
            self.posterization_args = gloss.utils.color.posterization_adaptive_default_args(resolution=resolution)
        else:
            assert blur_kernel is not None
            assert blur_sigma is not None
            assert n_superpixels is not None
        if blur_kernel is not None:
            self.posterization_args['blur_kernel'] = blur_kernel
        if blur_sigma is not None:
            self.posterization_args['blur_sigma'] = blur_sigma
        if n_superpixels is not None:
            self.posterization_args['n_superpixels'] = n_superpixels

    def __call__(self, render_res):
        torch_img = render_res[self.input_channel][0, ...]
        posterized, post_data = gloss.utils.color.posterize_image(
            torch_img / 2 + 0.5, debug_output=False, **self.posterization_args)
        render_res[self.output_channel] = posterized.unsqueeze(0) * 2 - 1.0
        return render_res

def cat_collate_dicts(samples):
    res = {}
    for k in samples[0].keys():
        if k == 'prompt_embed' or k =='view_embed':
            res[k] = torch.cat([x[k] for x in samples], dim=0)
        elif k == "dataset_id":
            res[k] = [x[k] for x in samples]
        else:
            res[k] = torch.cat([x[k] for x in samples], dim=0).permute(0, 3, 1, 2)
    return res


@dataclass
class MultiViewConfig:
    """Configures MultiviewDataset"""
    mesh:str
    num_views:int
    num_local_views:int
    train_view_limit: int = 0
    train_view_selection: str = "random"
    num_reference_views:int = 0
    data_base_dir:str = None
    zip_cache: bool = False
    prompt_fp:str = None
    single_view_dir:str = None
    prompt_emb_dir:str = None
    view_emb_dir:str = ""
    single_view_texture_dir:str = None
    sampling_dir:str = None
    num_in_channels:int = 17
    camera_dist:float = 1.0
    fov_min:float = 0.2
    fov_max:float = 0.8
    resolution:int = 256
    """number of albedo to cache on gpu"""
    max_num_cache:int = 0
    seed:int=0
    eval_ratio:float = 0.2
    eval_views:int = 50
    jitter:bool = True
    normal_cond:str = "normal"
    position_cond:str="relative"

from dataclasses import dataclass, field
from typing import Union, List, Sequence, Iterable
FloatOrList = Union[float, List[float]]

@dataclass
class ViewConfig:
    """Configures LocalMeshRenderDataset for one specific mesh view."""
    mesh: str
    """Mesh filename"""
    num_training_views: int
    """Number of training views to sample"""
    num_reference_views:int = 0
    """Number of reference views to sample"""
    cache_data_dir: str = None
    """Default cache directory for rendered data"""
    zip_cache: bool = False
    """If cache is a zip file instead of a folder"""
    resolution: int = 256
    """Resolution to use"""
    camera_dist: float = 1.0
    """Camera distance from surface"""
    fov_min:  float = 0.2
    fov_max:  float = 1.0
    azi_min:  FloatOrList = 0.0
    azi_max:  FloatOrList = 3.14
    elev_min: FloatOrList = -1.0
    elev_max: FloatOrList = 1.7
    viewdist_min: FloatOrList = 0.85
    viewdist_max: FloatOrList = 1.15
    viewdist_min: float = 0.85
    viewdist_max: float = 1.15
    on_device_cache:bool = False
    single_view_texture: str = None
    """Texture RGBA image file from backprojecting single view rendering onto mesh uvs."""
    face_sampling_weights: str = None
    """Filename of .pt sampling weights per face."""
    data_base_dir: str = ''
    """Base dir for all the data"""
    input_channels: list[str] = field(
        default_factory=lambda: ['geo_camera_normals', 'camera_normals', 'relative_positions', 'albedo', 'inpaint_mask'])
    """The input channels to the transformer."""
    num_in_channels: int = 17
    """The number of input channels to the transformer."""
    output_channels: list[str] = field(
        default_factory=lambda: ['albedo'])
    """The output channels to the transformer."""

_special_channels = ['inpaint_mask', 'rough_color']
def input_channel_to_required(ch):
    """ This is a hack assuming certain names for channels and their requirements.
    Given these preset names, returns standard output channels of render_all_features
    that are required for computation.

    Args:
        ch: str required channel

    Returns: list of required channels to compute ch

    """
    if ch == 'inpaint_mask':
        return ['albedo_alpha']
    elif ch == 'rough_color':
        return ['albedo']
    elif ch == 'relative_positions':
        return ['positions']
    else:
        return [ch]

def configure_single_view_dataset(config: ViewConfig, device, global_root_dir, **kwargs):
    """Create Mesh Render Dataset, Use SingleViewCameraExtrinsics instead of LocalExtrinsics

    Args:
        config (ViewConfig): _description_
        device (_type_): _description_
        global_root_dir (_type_): _description_
    """
    def _to_data_path(fpath):
        if fpath is None:
            return None
        return os.path.join(global_root_dir, config.data_base_dir, fpath)

    mesh_fname = _to_data_path(config.mesh)
    mesh = kaolin.io.mesh.import_mesh(mesh_fname).to(device)  # SurfaceMesh object
    mesh.vertices = kaolin.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)
    print(mesh)  # view attributes

    render_kwargs = {}
    if config.single_view_texture is not None:
        texture_path = _to_data_path(config.single_view_texture)
        albedo = kaolin.io.utils.read_image(texture_path).to(device).contiguous()
        custom_materials = copy.copy(mesh.materials)
        custom_materials[0].diffuse_texture = albedo
        render_kwargs = {'custom_materials': custom_materials}

    channels = set()
    for ch in config.input_channels + config.output_channels:
        channels.update(input_channel_to_required(ch))

    azim_range, elev_range, view_dist_range = [config.azi_min, config.azi_max], \
        [config.elev_min, config.elev_max], [config.viewdist_min, config.viewdist_max]

    if "debug" in kwargs:
        if kwargs["debug"]:
            dataset = LocalMeshRendersDataset(
            mesh, config.num_training_views,
            resolution=config.resolution,
            extrinsics_sampler=SingleViewCameraExtrinsicsSampler(azim_range, elev_range, view_dist_range, debug=True), #LocalCameraExtrinsicsSampler(mesh, face_sampling_weights=sampling_weights),
            intrinsics_sampler=FovSampler(config.fov_min, config.fov_max),
            channels=channels, cache=False,
            render_kwargs=render_kwargs)
    else:
        dataset = LocalMeshRendersDataset(
            mesh, config.num_training_views,
            resolution=config.resolution,
            extrinsics_sampler=SingleViewCameraExtrinsicsSampler(azim_range, elev_range, view_dist_range), #LocalCameraExtrinsicsSampler(mesh, face_sampling_weights=sampling_weights),
            intrinsics_sampler=FovSampler(config.fov_min, config.fov_max),
            channels=channels, cache=False,
            render_kwargs=render_kwargs)

    if 'rough_color' in config.input_channels:
        dataset.add_processor(
            PosterizationProcessor(input_channel='albedo', output_channel='rough_color',
                                   resolution=config.resolution), add_channels=['rough_color'])

    if 'relative_positions' in config.input_channels:
        dataset.add_processor(
            RelativeToCenterValueProcessor(input_channel='positions', output_channel='relative_positions'),
            add_channels=['relative_positions'])

    dataset.set_cache_dir(_to_data_path(config.cache_data_dir))
    dataset.set_sampler()
    return dataset


def configure_test_view_datasets(config: ViewConfig, device, global_root_dir, camera, ref=False):
    def _to_data_path(fpath):
        if fpath is None:
            return None
        return os.path.join(global_root_dir, config.data_base_dir, fpath)

    mesh_fname = _to_data_path(config.mesh)
    mesh = kaolin.io.mesh.import_mesh(mesh_fname).to(device)  # SurfaceMesh object
    mesh.vertices = kaolin.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)

    channels = set()
    for ch in config.input_channels + config.output_channels:
        channels.update(input_channel_to_required(ch))
    render_res = gloss.utils.render.render_all_features(camera, mesh, lighting=None)
    _, _, face_pixel_counts = get_valid_faces(
        render_res[kaolin.render.easy_render.RenderPass.face_idx],
        render_res["geo_camera_normals"],
        mesh.faces.shape[0],
        min_pixel_count=5,
        max_angle_deviation=math.pi * 0.3,
    )
    face_areas = (
        kaolin.ops.mesh.face_areas(mesh.vertices.unsqueeze(0), mesh.faces).squeeze(0)
        * 1000
    )
    face_weights = torch.zeros_like(face_areas)
    face_weights[face_pixel_counts >= 1] = 0
    face_weights[face_pixel_counts == 0] = 1
    sampling_weights = face_areas * face_weights
    ExtrinsicsSampler = LocalCameraExtrinsicsSampler(mesh=mesh, camera_dist=config.camera_dist)
    ExtrinsicsSampler.set_sampling_weights(sampling_weights)

    _mesh = copy.deepcopy(mesh)
    custom_materials = [m.to(mesh.vertices.device) for m in hack_import_albedo_gltf_materials_raw(mesh_fname)]
    render_kwargs = {"custom_materials": custom_materials}
    gt_channels = set(config.output_channels)
    gt_channels.update(["geo_camera_normals"])
    gt_dataset = LocalMeshRendersDataset(
        _mesh, config.num_training_views,
        resolution=config.resolution,
        extrinsics_sampler=ExtrinsicsSampler,
        intrinsics_sampler=FovSampler(config.fov_min, config.fov_max),
        channels=gt_channels,
        render_kwargs=render_kwargs
        )

    gt_dataset.set_cache_dir(_to_data_path(config.cache_data_dir +"_gt"))
    gt_dataset.set_sampler()

    single_render_res = gloss.utils.render.render_all_features(camera, mesh, lighting=None, **render_kwargs)
    uv_dict, mask = backproject_render(mesh, camera, single_render_res, ['albedo'], 4096, 4096)
    # save the uv
    uv = ((uv_dict['albedo'].squeeze(0) + 1.0) / 2.0).contiguous()
    custom_materials = [m.to(mesh.vertices.device) for m in hack_import_albedo_gltf_materials_raw(mesh_fname)]
    custom_materials[0].diffuse_texture = uv
    render_kwargs = {"custom_materials": custom_materials}

    dataset = LocalMeshRendersDataset(
        mesh, config.num_training_views,
        resolution=config.resolution,
        extrinsics_sampler=ExtrinsicsSampler,
        intrinsics_sampler=FovSampler(config.fov_min, config.fov_max),
        channels=channels,
        render_kwargs=render_kwargs
        )

    if 'rough_color' in config.input_channels:
        dataset.add_processor(
            PosterizationProcessor(input_channel='albedo', output_channel='rough_color',
                                   resolution=config.resolution), add_channels=['rough_color'])

    if 'relative_positions' in config.input_channels:
        dataset.add_processor(
            RelativeToCenterValueProcessor(input_channel='positions', output_channel='relative_positions'),
            add_channels=['relative_positions'])
    dataset.set_cache_dir(_to_data_path(config.cache_data_dir+"_gt"))
    dataset.set_sampler()
    dataset.set_cache_dir(_to_data_path(config.cache_data_dir))

    # create a reference dataset
    if ref:
        ref_face_weights = torch.zeros_like(face_areas)
        threshold_1 = torch.quantile(face_pixel_counts[face_pixel_counts > 0].float(), 0.6)
        ref_face_weights[face_pixel_counts >= threshold_1] = 1
        threshold_2 = torch.quantile(face_pixel_counts[face_pixel_counts > 0].float(), 0.8)
        ref_face_weights[face_pixel_counts >= threshold_2] = 2
        ref_sampling_weights = face_areas * ref_face_weights
        RefExtrinsicsSampler = LocalCameraExtrinsicsSampler(mesh=mesh, camera_dist=config.camera_dist)
        RefExtrinsicsSampler.set_sampling_weights(ref_sampling_weights)
        ref_dataset = LocalReferenceRendersDataset(
                mesh, config.num_reference_views,
                resolution=config.resolution,
                extrinsics_sampler=RefExtrinsicsSampler,
                intrinsics_sampler=FovSampler(config.fov_min, config.fov_max),
                channels=channels,
                render_kwargs=render_kwargs
            )
        ref_dataset.processors = dataset.processors
        ref_dataset.channels = dataset.channels

        ref_dataset.set_cache_dir(_to_data_path(config.cache_data_dir+"_ref"))
        ref_dataset.set_sampler()
        return dataset, ref_dataset, gt_dataset
    else:
        return dataset, gt_dataset


def configure_local_view_dataset(config: ViewConfig, device, global_root_dir, verbose=True, **kwargs):
    """Create Mesh Render Dataset, UseLocalExtrinsics instead of SingleViewCameraExtrinsics

    Args:
        config (ViewConfig): _description_
        device (_type_): _description_
        global_root_dir (_type_): _description_
        verbose (bool, optional): _description_. Defaults to True.
    """
    def _to_data_path(fpath):
        if fpath is None:
            return None
        return os.path.join(global_root_dir, config.data_base_dir, fpath)

    if "mesh" in kwargs:
        mesh = kwargs["mesh"]
    else:
        mesh_fname = _to_data_path(config.mesh)
        mesh = kaolin.io.mesh.import_mesh(mesh_fname).to(device)  # SurfaceMesh object
        mesh.vertices = kaolin.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)

    if config.single_view_texture is not None:
        texture_path = _to_data_path(config.single_view_texture)
        if config.on_device_cache:
            albedo = kaolin.io.utils.read_image(texture_path).to(device).contiguous()
            log_tensor(albedo, 'custom albedo', logger, print_stats=True)
            custom_materials = copy.copy(mesh.materials)
            custom_materials[0].diffuse_texture = albedo
            render_kwargs = {'custom_materials': custom_materials}
        else:
            render_kwargs = {'texture_path': texture_path}
    else:
        render_kwargs = {}

    channels = set()
    for ch in config.input_channels + config.output_channels:
        channels.update(input_channel_to_required(ch))

    sampling_weights = torch.load(config.face_sampling_weights, map_location='cpu')
    ExtrinsicsSampler = LocalCameraExtrinsicsSampler(mesh=mesh, camera_dist=config.camera_dist)
    ExtrinsicsSampler.set_sampling_weights(sampling_weights)
    dataset = LocalMeshRendersDataset(
        mesh, config.num_training_views,
        resolution=config.resolution,
        extrinsics_sampler=ExtrinsicsSampler,
        intrinsics_sampler=FovSampler(config.fov_min, config.fov_max),
        channels=channels,
        render_kwargs=render_kwargs
        )

    if 'rough_color' in config.input_channels:
        dataset.add_processor(
            PosterizationProcessor(input_channel='albedo', output_channel='rough_color',
                                   resolution=config.resolution), add_channels=['rough_color'])

    if 'relative_positions' in config.input_channels:
        dataset.add_processor(
            RelativeToCenterValueProcessor(input_channel='positions', output_channel='relative_positions'),
            add_channels=['relative_positions'])

    if config.zip_cache:
        dataset.set_cache_dir(_to_data_path(config.cache_data_dir+".zip"))
    else:
        dataset.set_cache_dir(_to_data_path(config.cache_data_dir))
    dataset.set_sampler()

    ref_dataset = None
    if config.num_reference_views > 0:
        ref_dataset = LocalReferenceRendersDataset(
            mesh, config.num_reference_views,
            resolution=config.resolution,
            extrinsics_sampler=copy.copy(ExtrinsicsSampler),
            intrinsics_sampler=FovSampler(config.fov_min, config.fov_max),
            channels=channels,
            render_kwargs=render_kwargs
        )
        ref_dataset.processors = dataset.processors
        ref_dataset.channels = dataset.channels
        if config.zip_cache:
            ref_dataset.set_cache_dir(_to_data_path(config.cache_data_dir + "_ref.zip"))
        else:
            os.makedirs(_to_data_path(config.cache_data_dir+"_ref"), exist_ok=True)
            ref_dataset.set_cache_dir(_to_data_path(config.cache_data_dir+"_ref"))
        ref_dataset.set_sampler()

    del sampling_weights
    del render_kwargs

    return dataset, ref_dataset


def configure_multi_view_dataset(config: MultiViewConfig, device, global_root_dir):
    def _get_full_fp(fp):
        return os.path.join(global_root_dir, fp)
    mesh_fname =_get_full_fp(config.mesh)
    _mesh = kaolin.io.mesh.import_mesh(mesh_fname).to(device)  # SurfaceMesh object
    _mesh.vertices = kaolin.ops.pointcloud.center_points(_mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)
    data_base =  _get_full_fp(config.data_base_dir)
    os.makedirs(data_base, exist_ok=True)
    texture_dir, sampling_dir=  _get_full_fp(config.single_view_texture_dir), _get_full_fp(config.sampling_dir)
    subdatasets, ref_subdatasets = [], []
    for idx in range(config.num_views):
        if idx < config.max_num_cache:
            _on_device_cache = True
        else:
            _on_device_cache = False
        assert os.path.exists(data_base)
        _data_base_dir = os.path.join(data_base, "view%04d" % idx)
        _data_cache_dir = os.path.join(_data_base_dir, 'datacache')
        if not config.zip_cache:
            os.makedirs(_data_base_dir, exist_ok=True)
            os.makedirs(_data_cache_dir, exist_ok=True)
        else:
            assert os.path.exists(_data_cache_dir+".zip")
            if config.num_reference_views > 0:
                assert os.path.exists(_data_cache_dir + "_ref.zip")
        _single_view_texture = os.path.join(texture_dir, "view%04d.png" % idx)
        _face_sampling_weights = os.path.join(sampling_dir, "view%04d.pt" % idx)
        if config.normal_cond == "geonormal":
            _in_channels = ['geo_camera_normals', 'relative_positions', 'albedo', 'inpaint_mask']
        if config.normal_cond == "normal":
            _in_channels = ['camera_normals', 'relative_positions', 'albedo', 'inpaint_mask']
        _config = ViewConfig(mesh=config.mesh, num_training_views=config.num_local_views,
                             num_reference_views=config.num_reference_views, cache_data_dir=_data_cache_dir,
                             zip_cache=config.zip_cache, single_view_texture=_single_view_texture,
                             camera_dist=config.camera_dist, fov_min=config.fov_min, fov_max=config.fov_max,
                             face_sampling_weights=_face_sampling_weights, on_device_cache=_on_device_cache,
                             input_channels=_in_channels)
        _subdataset, _ref_subdataset = configure_local_view_dataset(_config, device, global_root_dir, mesh=_mesh)
        subdatasets.append(_subdataset)
        ref_subdatasets.append(_ref_subdataset)
    if config.num_reference_views == 0:
        ref_subdatasets = None
    dataset = MultiviewLocalMeshRendersDataset(
        _get_full_fp(config.single_view_dir),
        subdatasets,
        device,
        ref_subdatasets=ref_subdatasets)
    return dataset


class EvalCameraExtrinsicsSampler:
    def __init__(self, view_camera:kaolin.render.camera.Camera, **kwargs):
        self.view_camera = view_camera
        self.view_cam_pos = self.view_camera.extrinsics.cam_pos()[0].T
        self.view_cam_up = self.view_camera.extrinsics.cam_up()[0].T
        self.view_cam_forward = self.view_camera.extrinsics.cam_forward()[0].T
        self.view_cam_pos_sp = cartesian2spherical(self.view_cam_pos.squeeze(0))
        self.view_azimuth = self.view_cam_pos_sp[0].unsqueeze(0)
        self.view_elevation = self.view_cam_pos_sp[1].unsqueeze(0)

        self.samples = None
        self.hash_samples = None

        if "azi_jitter_range" in kwargs:
            self.azi_jitter_range = kwargs["azi_jitter_range"]
        else:
            self.azi_jitter_range = 1.0
        if "azi_jitter_deviation" in kwargs:
            self.azi_jitter_deviation = kwargs["azi_jitter_deviation"]
        else:
            self.azi_jitter_deviation = 0.0
        if "elev_jitter_range" in kwargs:
            self.elev_jitter_range = kwargs["elev_jitter_range"]
        else:
            self.elev_jitter_range = 1.0
        if "elev_jitter_deviation" in kwargs:
            self.elev_jitter_deviation = kwargs["elev_jitter_deviation"]
        else:
            self.elev_jitter_deviation = 0.0
        if "dist_jitter_range" in kwargs:
            self.dist_jitter_range = kwargs["dist_jitter_range"]
        else:
            self.dist_jitter_range = 0.1
        if "dist_jitter_deviation" in kwargs:
            self.dist_jitter_deviation = kwargs["dist_jitter_deviation"]
        else:
            self.dist_jitter_deviation = 0.0

    def generate(self, num_samples:int):
        azimuths = self.view_azimuth + (torch.rand(num_samples) - 0.5) * self.azi_jitter_range + self.azi_jitter_deviation
        elevations = self.view_elevation + (torch.rand(num_samples) - 0.5) * self.elev_jitter_range + self.elev_jitter_deviation
        dists = torch.ones(num_samples) + (torch.rand(num_samples) - 0.5) * self.dist_jitter_range + self.dist_jitter_deviation
        self.hash_samples = torch.cat([azimuths.unsqueeze(-1), elevations.unsqueeze(-1), dists.unsqueeze(-1)], dim=1)
        eye = kaolin.render.lighting.sg_direction_from_azimuth_elevation(azimuths, elevations)
        world_up = torch.tensor([0.0, 1.0, 0.0])
        up = world_up - eye[:, 1][:, None] * eye
        eye = eye * dists[:, None]
        at = torch.zeros_like(eye)
        return torch.cat([eye, at, up], dim=1)#samples

    def set_samples(self, samples):
        self.num_samples = samples.shape[0]
        self.samples = samples

    def get(self, idx):
        return self.samples[idx % self.num_samples]

    def get_hash(self, idx):
        sample = self.hash_samples[idx % self.num_samples]
        sample_str = ['%.4f' % s.item() for s in sample]
        a_hash = sample_str[0]
        e_hash = sample_str[1]
        d_hash = sample_str[2]
        hash = f'a{a_hash}e{e_hash}d{d_hash}'
        return hash


class LocalCameraExtrinsicsSampler:
    def __init__(self, mesh=None, camera_dist=1.0):
        self.mesh = mesh
        self.camera_dist = camera_dist
        self.samples = None
        self.face_sampling_weights = None

    def set_sampling_weights(self, sampling_weights):
        self.face_sampling_weights = sampling_weights

    def generate(self, num_samples):
        assert self.face_sampling_weights is not None
        device = self.mesh.faces.device
        self.face_sampling_weights = self.face_sampling_weights.to(device)
        faces_to_sample = torch.where(self.face_sampling_weights > 0)[0]
        self.num_samples = num_samples
        face_normals = self.mesh.face_normals[faces_to_sample, ...]
        # weights to compute barycentric coordinates per face sample
        face_colors = torch.zeros_like(face_normals)
        face_colors[..., 0, 0] = 1
        face_colors[..., 1, 1] = 1
        face_colors[..., 2, 2] = 1

        sampling_weights = self.face_sampling_weights[faces_to_sample].unsqueeze(0)

        # sample points over the mesh, using only a subset of faces
        res = kaolin.ops.mesh.sample_points(
            self.mesh.vertices.unsqueeze(0),
            self.mesh.faces[faces_to_sample],
            num_samples=num_samples,
            areas=sampling_weights,
            face_features=torch.cat([face_normals, face_colors], dim=-1).unsqueeze(0))

        cam_face = faces_to_sample[res[1][0, ...]]
        cam_pos = res[0][0, ...]
        rand_vertex = torch.randint(0, 3, (1,)).to(device)
        cam_vidx = self.mesh.faces[cam_face, rand_vertex]
        cam_normals = res[2][0, :, :3]

        up = self.mesh.vertices[cam_vidx] - cam_pos
        eye = cam_pos + torch.nn.functional.normalize(cam_normals) * self.camera_dist
        lookat = cam_pos
        return torch.cat([eye, lookat, up], dim=1)

    def set_samples(self, samples):
        self.num_samples = samples.shape[0]
        self.samples = samples

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


class MultiviewLocalMeshRendersDataset(Dataset):
    def __init__(self, view_dir, subdatasets, device, sub_indices=None, ref_subdatasets=None):
        self.view_dir = view_dir
        self.num_sub_datasets = len(subdatasets)
        if sub_indices is None:
            self.sub_indices = list(range(self.num_sub_datasets))
        else:
            self.sub_indices = sub_indices
        self.datasets = subdatasets
        self.view_per_dataset = subdatasets[0].num_renders
        self.ref_datasets = ref_subdatasets
        self.ref_view_per_dataset = 0
        if ref_subdatasets is not None:
            self.ref_view_per_dataset = ref_subdatasets[0].num_renders
        self.device = device

    def __len__(self):
        return len(self.sub_indices) * self.view_per_dataset

    def _get_dataset_id(self, idx):
        item_id, dataset_id = idx % self.view_per_dataset,  self.sub_indices[idx // self.view_per_dataset]
        return dataset_id, item_id

    def _get_view(self, idx):
        _view_path = os.path.join(self.view_dir, 'view%04d.basecolor.png' %idx)
        return torchvision.io.read_image(_view_path).unsqueeze(0).permute(0, 2, 3, 1) / 255

    def _get_random_reference(self, dataset_id):
        # this is used during training to get a random reference
        item_id = random.choice(range(self.ref_view_per_dataset))
        _local_renders = self.ref_datasets[dataset_id][item_id]
        return _local_renders

    def _get_reference(self, idx):
        # this is used to iterate through the references and cache them
        item_id, dataset_id = idx % self.ref_view_per_dataset, idx // self.ref_view_per_dataset
        _local_renders = self.ref_datasets[dataset_id][item_id]
        return _local_renders

    def __getitem__(self, idx):
        dataset_id, item_id = self._get_dataset_id(idx)
        _local_renders = self.datasets[dataset_id][item_id]
        _local_renders.update({"view": self._get_view(dataset_id)})
        _local_renders.update({"dataset_id": dataset_id})
        return _local_renders


class LocalMeshRendersDataset(Dataset):
    def __init__(self, mesh, num_renders,
                 resolution,
                 intrinsics_sampler=FovSampler(0.2, 1.0),
                 extrinsics_sampler=None,
                 light_sampler=NormalLightSampler(),
                 channels=None,
                 cache=True,
                 render_kwargs={}
                 ):
        """
        Dataset that constructs local render viewpoints by sampling over a single mesh.

        First, positions on the provided subset of mesh faces is sampled.
        Second, for every point sampled, we construct a camera that is pointing at the mesh at that location
        and has a random up direction.
        Intrinsics and light sampling is configured by the inputs provided.

        Call `set_cache_dir` to enable hashing of rendered views (quantizes per-triangle samples for hash hits).

        Example:
            # For example, create custom material from back-projected texture of one view
            custom_materials = copy.deepcopy(mesh.materials)
            custom_materials[0].diffuse_texture = back_proj_res['albedo'].squeeze(0).contiguous()

            # Set sampling weights for example based on the back projection
            sampling_weights = gloss.utils.single_view.get_triangle_sampling_weights(mesh.vertices, mesh.faces, face_pixel_counts)

            # Create dataset and configure it
            dataset = LocalMeshRendersDataset(mesh, 10, face_sampling_weights=sampling_weights,
                                              resolution=resolution,
                                              render_kwargs={'custom_materials': custom_materials})

            # Configure custom processor that adds rough color channel based on rendered albedo
            dataset.add_processor(
                PosterizationProcessor(input_channel='albedo', output_channel='rough_color',
                           resolution=resolution), add_channels=['rough_color'])

            # Configure a cache directory; only first call to dataset[i] will render the output
            dataset.set_cache_dir('/tmp/datacache_weighted')

        Args:
            mesh:
            num_renders:
            faces_to_sample: Long tensor of face_ids that should be included in sampling
            face_sampling_weights: float (num_faces,) tensor containing sampling weights for each face
                (if not provided, will instead use face areas)
            resolution:
            intrinsics_sampler:
            light_sampler:
            channels:
        """

        if channels is None:
            channels = ['render', 'albedo', 'geo_camera_normals', 'camera_normals', 'normals', 'geo_normals', 'positions', 'mask', 'albedo_alpha']

        self.mesh = mesh
        self.num_renders = num_renders
        self.resolution = resolution
        self.intrinsics_sampler = intrinsics_sampler
        self.extrinsics_sampler = extrinsics_sampler
        self.light_sampler = light_sampler
        self.channels = set(channels)
        self.render_kwargs = render_kwargs

        self.device = self.mesh.faces.device

        self.cache_dir = None
        self.processors = []
        self.cache = cache
        self.zip_file = None
        self.valid_indices = list(range(num_renders))
        self.check_sample_quality = False

    def turn_on_sample_quality_check(self, x_margin=0.25, y_margin=0.25, z_threshold=0.0,
                                     min_coverage=0.05, min_normal_std=0.03,
                                     uniform_normal_coverage=0.95):
        self.check_sample_quality = True
        self.x_margin, self.y_margin, self.z_threshold = x_margin, y_margin, z_threshold
        # Extra normal-quality knobs (added to catch flat / inside-mesh / off-target patches
        # that the x/y/z-margin fraction filter misses):
        #   min_coverage          minimum fraction of the patch occupied by foreground
        #   min_normal_std        minimum stdev of geo_camera_normals over foreground
        #   uniform_normal_coverage  only enforce min_normal_std when coverage exceeds this
        #                            (avoids penalizing thin silhouettes whose normals are
        #                            naturally near-uniform along the silhouette edge)
        self.min_coverage = min_coverage
        self.min_normal_std = min_normal_std
        self.uniform_normal_coverage = uniform_normal_coverage

    def set_cache_dir(self, cache_dir=None):
        self.cache_dir = cache_dir
        if cache_dir is not None:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)

    def set_sampler(self):
        assert self.cache_dir is not None
        if self.cache_dir.endswith(".zip"):
            with zipfile.ZipFile(self.cache_dir, 'r') as zipf:
                with zipf.open('extrinsics.pt') as f:
                    buffer = io.BytesIO(f.read())
                    extrinsics_samples = torch.load(buffer, map_location='cpu')
                    self.extrinsics_sampler.set_samples(extrinsics_samples)
                with zipf.open('intrinsics.pt') as f:
                    buffer = io.BytesIO(f.read())
                    intrinsics_samples = torch.load(buffer, map_location='cpu')
                    self.intrinsics_sampler.set_samples(intrinsics_samples)
        else:
            extrinsics_sample_fp = os.path.join(self.cache_dir, 'extrinsics.pt')
            intrinsics_sample_fp = os.path.join(self.cache_dir, 'intrinsics.pt')
            if os.path.exists(extrinsics_sample_fp) and os.path.exists(intrinsics_sample_fp):
                print(f'loading precomputed extrinsics and intrinsics from {extrinsics_sample_fp}')
                extrinsics_samples, intrinsics_samples = torch.load(extrinsics_sample_fp, map_location='cpu'), \
                    torch.load(intrinsics_sample_fp, map_location='cpu')
                self.intrinsics_sampler.set_samples(intrinsics_samples)
                self.extrinsics_sampler.set_samples(extrinsics_samples)
            else:
                print(f'saving extrinsics to {extrinsics_sample_fp}')
                print(f'saving intrinsics to {intrinsics_sample_fp}')
                self.intrinsics_sampler.set_samples(self.intrinsics_sampler.generate(self.num_renders))
                self.extrinsics_sampler.set_samples(self.extrinsics_sampler.generate(self.num_renders))
                torch.save(self.extrinsics_sampler.samples, extrinsics_sample_fp)
                torch.save(self.intrinsics_sampler.samples, intrinsics_sample_fp)


    def add_processor(self, processor, add_channels=None):
        """ Adds a processor to the output. The output is a dictionary produced by render_all_features;
        processors can use and modify that dictionary in any way.
        """
        self.processors.append(processor)
        if add_channels is not None:
            self.channels.update(add_channels)

    def __len__(self):
        return self.num_renders

    def get_camera_name(self, idx):
        return self.__make_cam_hash(idx)

    def __make_cam_hash(self, idx):
        extrinsics_hash = self.extrinsics_sampler.get_hash(idx)
        intrinsics_hash = self.intrinsics_sampler.get_hash(idx)
        return f"{extrinsics_hash}_{intrinsics_hash}"

    def get_camera(self, idx):
        extrinsics = self.extrinsics_sampler.get(idx)
        intrinsics = self.intrinsics_sampler.get(idx)
        return gloss.utils.render.make_camera_from_extr_intr(extrinsics, intrinsics, resolution=self.resolution, device=self.mesh.faces.device)

    def get_lighting(self, idx):
        cam_normal = self.extrinsics_sampler.get(idx)[:3] - self.extrinsics_sampler.get(idx)[3:6]
        return self.light_sampler(cam_normal).to(self.mesh.faces.device)

    def __make_cached_filename(self, item_hash, channel):
        if self.cache_dir.endswith(".zip"):
            return f'{item_hash}_{channel}.png'
        return os.path.join(self.cache_dir, f'{item_hash}_{channel}.png')

    def __write_cached_file(self, image, fname):
        # all channels are -1..1
        torchvision.io.write_png(((image / 2 + 0.5).squeeze(0).clamp(0, 1) * 255).to(torch.uint8).cpu().permute(2, 0, 1), fname)

    def __read_cached_file(self, fname):
        # all channels are -1..1
        return torchvision.io.image.read_image(fname).float().permute(1, 2, 0).unsqueeze(0) / 255.0 * 2 - 1.0

    def __read_cached_from_zip(self, item_hash):
        try:
            data = {}
            for channel in self.channels:
                fname = f'{item_hash}_{channel}.png'
                with zipfile.ZipFile(self.cache_dir, 'r') as f:
                    buffer = io.BytesIO(f.open(fname).read())
                    image = torch.from_numpy(imageio.imread(buffer))
                    if image.ndim < 3:
                        image = image[None]
                    else:
                        image = image.permute(2, 0, 1)
                    image = image.float().permute(1, 2, 0).unsqueeze(0) / 255.0 * 2 - 1.0
                    data[channel] = image
            return data
        except Exception as e:
            print(f"Error: {e} from {self.cache_dir}")
            return None

    def __get_hashed_data(self, item_hash):
        if self.cache_dir is None:
            return None

        if not self.cache_dir.endswith(".zip"):
            filenames = {channel: self.__make_cached_filename(item_hash, channel) for channel in self.channels}
            for filename in filenames.values():
                if not os.path.isfile(filename):
                    return None
            return {channel: self.__read_cached_file(filename) for channel, filename in filenames.items()}
        else:
            return self.__read_cached_from_zip(item_hash)

    def __store_hashed_data(self, item_hash, data):
        if self.cache_dir is None:
            return

        for channel, val in data.items():
            fname = self.__make_cached_filename(item_hash, channel)
            self.__write_cached_file(val, fname)

    def __get_albedo(self):
        albedo = kaolin.io.utils.read_image(self.albedo_path).to(torch.device('cuda')).contiguous() # todo: add a device flag
        return albedo

    def __render_view(self, camera, lighting, required_passes):
        # also load mesh dynamically if not cached on cuda
        render_kwargs = {}
        if 'texture_path' in self.render_kwargs:
            albedo = kaolin.io.utils.read_image(self.render_kwargs['texture_path']).to(self.device).contiguous()
            custom_materials = copy.copy(self.mesh.materials)
            custom_materials[0].diffuse_texture = albedo
            render_kwargs = {'custom_materials': custom_materials}
        if 'custom_materials' in self.render_kwargs:
            render_kwargs = self.render_kwargs

        # Render the view for pre-sampled camera and lighting and cache if configured
        render_res = gloss.utils.render.render_all_features(
            camera, self.mesh, lighting,
            required_passes=required_passes, **render_kwargs)
        return render_res

    def __set_results_background(self, results):
        normal_key = [k for k in results.keys() if 'normal' in k][0]
        normal = results[normal_key]
        mask = (torch.abs(normal) < 0.01).all(dim=-1, keepdim=True).float()
        for k in results.keys():
            results[k] = results[k] * (1.0 - mask) + (torch.ones_like(results[k]) * -1.0) * mask
        results["background_alpha"] = (1.0 - mask) * 2.0 - 1.0
        return results

    def check_bad_sample(self, result):
        background_alpha = torch.clip(result['background_alpha'], 0.0, 1.0)
        bg_sum = torch.sum(background_alpha)
        total_pixels = float(background_alpha.numel())

        # 1) coverage: patch must contain enough foreground
        coverage = float(bg_sum.item()) / total_pixels if total_pixels else 0.0
        if coverage < getattr(self, 'min_coverage', 0.0):
            print(f"bad sample (low coverage={coverage:.3f} < {self.min_coverage})")
            return True

        if bg_sum.item() <= 0:
            # nothing in the foreground at all — already caught by coverage, but be defensive
            print("bad sample (empty foreground)")
            return True

        background = torch.zeros_like(result['geo_camera_normals'])
        background[..., -1] = (1.0 - background_alpha).squeeze(-1)
        geo_normal = result['geo_camera_normals'] * background_alpha + background

        # 2) original x/y/z margin filter — fraction of foreground pixels with bad normals
        t = torch.sum(geo_normal[..., 0] <= -self.x_margin) + torch.sum(geo_normal[..., 0] >= self.x_margin) +\
            torch.sum(geo_normal[..., 1] <= -self.y_margin) + torch.sum(geo_normal[..., 1] >= self.y_margin) + torch.sum(geo_normal[..., -1] <= self.z_threshold)
        is_bad = (t / bg_sum) > 0.3
        if is_bad:
            print("bad sample")
            print(torch.sum(geo_normal[..., 0] <= -self.x_margin) + torch.sum(geo_normal[..., 0] >= self.x_margin))
            print(torch.sum(geo_normal[..., 1] <= -self.y_margin) + torch.sum(geo_normal[..., 1] >= self.y_margin))
            print(torch.sum(geo_normal[..., -1] <= self.z_threshold))
            print(bg_sum)
            return True

        # 3) normal-uniformity filter — patches looking head-on at a single flat tri have
        # near-zero stdev of geo_camera_normals over the foreground. Only enforce when
        # the patch is mostly foreground so we don't reject legitimate close-ups whose
        # silhouette is naturally narrow.
        min_std = getattr(self, 'min_normal_std', 0.0)
        if min_std > 0 and coverage > getattr(self, 'uniform_normal_coverage', 0.95):
            mask_b = background_alpha.squeeze(-1) > 0.5
            n_fg = result['geo_camera_normals'][mask_b]
            if n_fg.numel() > 0:
                std_per_axis = n_fg.std(dim=0)  # (3,)
                normal_std = float(std_per_axis.mean().item())
                if normal_std < min_std:
                    print(f"bad sample (uniform normals std={normal_std:.4f} < {min_std} at coverage={coverage:.3f})")
                    return True
        return False

    # TODO: might want to write custom sampler later
    def __getitem__(self, idx):
        # If render was cached, retrieve
        item_hash = self.__make_cam_hash(idx)
        result = self.__get_hashed_data(item_hash)
        if self.cache_dir is not None:
            if self.cache_dir.endswith(".zip") and result is None:
                while result is None:
                    if idx in self.valid_indices:
                        self.valid_indices.remove(idx)
                    idx = random.choice(self.valid_indices)
                    item_hash = self.__make_cam_hash(idx)
                    result = self.__get_hashed_data(item_hash)
        if result is not None:
            result = self.__set_results_background(result)
            return result
        render_res = self.__render_view(self.get_camera(idx), self.get_lighting(idx), self.channels)
        for p in self.processors:
            p(render_res)
        result = {channel: render_res[channel] for channel in self.channels}
        result = self.__set_results_background(result)
        if self.cache:
            self.__store_hashed_data(item_hash, result)
        return result


class LocalReferenceRendersDataset(LocalMeshRendersDataset):
    def set_sampler(self):
        assert self.cache_dir is not None
        if self.cache_dir.endswith(".zip"):
            with zipfile.ZipFile(self.cache_dir, 'r') as zipf:
                with zipf.open('extrinsics.pt') as f:
                    buffer = io.BytesIO(f.read())
                    extrinsics_samples = torch.load(buffer, map_location='cpu')
                    self.extrinsics_sampler.set_samples(extrinsics_samples)
                with zipf.open('intrinsics.pt') as f:
                    buffer = io.BytesIO(f.read())
                    intrinsics_samples = torch.load(buffer, map_location='cpu')
                    self.intrinsics_sampler.set_samples(intrinsics_samples)
        else:
            extrinsics_sample_fp = os.path.join(self.cache_dir, 'extrinsics.pt')
            intrinsics_sample_fp = os.path.join(self.cache_dir, 'intrinsics.pt')
            if os.path.exists(extrinsics_sample_fp) and os.path.exists(intrinsics_sample_fp):
                print(f'loading precomputed extrinsics and intrinsics from {extrinsics_sample_fp}')
                extrinsics_samples, intrinsics_samples = torch.load(extrinsics_sample_fp, map_location='cpu'), \
                    torch.load(intrinsics_sample_fp, map_location='cpu')
                self.intrinsics_sampler.set_samples(intrinsics_samples)
                self.extrinsics_sampler.set_samples(extrinsics_samples)
            else:
                idx = 0
                intrinsics_samples, extrinsics_samples = [], []
                progress_bar = tqdm(total=self.num_renders)
                progress_bar.set_description(f"Set reference cameras")
                while idx < self.num_renders:
                    intrinsics = self.intrinsics_sampler.generate(1)[0]
                    extrinsics = self.extrinsics_sampler.generate(1)[0]
                    camera = gloss.utils.render.make_camera_from_extr_intr(extrinsics, intrinsics,
                                                                           resolution=self.resolution,
                                                                           device=self.mesh.faces.device)
                    cam_normal = extrinsics[:3] - extrinsics[3:6]
                    lighting = self.light_sampler(cam_normal).to(self.mesh.faces.device)
                    render_res = self._LocalMeshRendersDataset__render_view(camera, lighting, ["albedo_alpha", "raw_depth"])
                    alpha = render_res["albedo_alpha"]
                    bg_pixels = (render_res["raw_depth"] == 0).sum()
                    mask_percentage = (alpha == 1).sum() / (self.resolution * self.resolution - bg_pixels)
                    if mask_percentage >= 0.9:
                        intrinsics_samples.append(intrinsics)
                        extrinsics_samples.append(extrinsics)
                        idx += 1
                        progress_bar.update(1)
                self.intrinsics_sampler.set_samples(torch.stack(intrinsics_samples, 0))
                self.extrinsics_sampler.set_samples(torch.stack(extrinsics_samples, 0))
                torch.save(self.extrinsics_sampler.samples, extrinsics_sample_fp)
                torch.save(self.intrinsics_sampler.samples, intrinsics_sample_fp)

    def __getitem__(self, idx):
        # If render was cached, retrieve
        item_hash = self._LocalMeshRendersDataset__make_cam_hash(idx)
        result = self._LocalMeshRendersDataset__get_hashed_data(item_hash)
        if result is not None:
            result = self._LocalMeshRendersDataset__set_results_background(result)
            return result
        render_res = self._LocalMeshRendersDataset__render_view(self.get_camera(idx), self.get_lighting(idx), self.channels)
        for p in self.processors:
            p(render_res)
        result = {channel: render_res[channel] for channel in self.channels}
        if self.cache:
            self._LocalMeshRendersDataset__store_hashed_data(item_hash, result)
        result = self._LocalMeshRendersDataset__set_results_background(result)
        return result


def train_eval_split(
    dataset:MultiviewLocalMeshRendersDataset,
    indices_fp,
    eval_ratio=0.2,
    eval_views=10,
    random_seed=0,
    mask=None,
    global_root_dir=None,
    mesh_path=None,
):
    """_summary_

    Args:
        dataset (MultiviewLocalMeshRendersDataset): render dataset
        indices_fp (_type_): file path to indices
        eval_ratio (float, optional): _description_. Defaults to 0.2.
        eval_views (int, optional): _description_. Defaults to 10.
        random_seed (int, optional): _description_. Defaults to 0.
        mask (_type_, optional): a list of local mesh render dataset

    Returns:
        train and eval datasets(MultiviewLocalMeshRendersDataset)
    """

    available_view_ids = list(range(dataset.num_sub_datasets))
    if mask is not None:
        available_view_ids = list(mask)
    excluded_train_view_ids = load_excluded_train_view_ids(dataset.view_dir)

    def regenerate_split():
        return build_split_dict_for_view_ids(
            available_view_ids,
            num_eval=eval_ratio,
            random_seed=random_seed,
            global_root_dir=global_root_dir,
            mesh_path=mesh_path,
        )

    if not os.path.exists(indices_fp):
        split_dict, _ = regenerate_split()
        save_split_indices(split_dict, indices_fp)
    else:
        split_dict = load_split_indices(indices_fp)
        expected_split_dict, used_predefined_eval = regenerate_split()
        available_view_set = set(available_view_ids)
        train_indices = [idx for idx in split_dict.get("train_indices", []) if idx in available_view_set]
        eval_indices = [idx for idx in split_dict.get("eval_indices", []) if idx in available_view_set]
        if used_predefined_eval:
            split_dict = expected_split_dict
            save_split_indices(split_dict, indices_fp)
        elif not train_indices or not eval_indices:
            split_dict, _ = regenerate_split()
            save_split_indices(split_dict, indices_fp)
        else:
            split_dict = {
                "train_indices": train_indices,
                "eval_indices": eval_indices,
            }
    split_dict = apply_excluded_train_view_ids(split_dict, excluded_train_view_ids)
    if not split_dict["train_indices"]:
        raise ValueError(
            "No training views remain after applying excluded training ids from "
            f"{os.path.join(os.path.dirname(dataset.view_dir), 'meta.json')}"
        )
    save_split_indices(split_dict, indices_fp)
    train_indices, eval_indices = split_dict['train_indices'], split_dict['eval_indices']
    train_dataset = MultiviewLocalMeshRendersDataset(dataset.view_dir, dataset.datasets, dataset.device, train_indices)
    eval_dataset = MultiviewLocalMeshRendersDataset(dataset.view_dir, dataset.datasets, dataset.device, eval_indices)
    eval_dataset.view_per_dataset = eval_views
    return train_dataset, eval_dataset


def get_lighting_from_cam(camera):
    light_sampler = NormalLightSampler()
    cam_normals = copy.deepcopy(camera.extrinsics.cam_forward())
    lighting = light_sampler(cam_normals.flatten()).cuda()
    return lighting
