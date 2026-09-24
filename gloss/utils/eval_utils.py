# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os, copy
import re
import math
import csv
from typing import List, Tuple
from pathlib import Path

import gloss.utils
import torch, torchvision
from torcheval.metrics import FrechetInceptionDistance
import kaolin
import lpips

import gloss
from gloss.data.render_dataloader import (
    LocalCameraExtrinsicsSampler,
    LocalMeshRendersDataset,
    FovSampler,
    RelativeToCenterValueProcessor,
)
from gloss.utils.single_view import get_valid_faces
from gloss.utils.misc import reclaim_cuda_memory
from gloss.utils.single_view import backproject_render


Tensor = torch.Tensor
TaggedTensor = List[Tuple[str, Tensor]]


def generate_eval_data(
    mesh, single_view_camera, num_src, num_tar, channels, fov, cam_dist, **kwargs
):
    """generate src and tar dataset with channel + inpainting mask
    target view has full inpainting masks
    """
    seed = kwargs.get("seed")
    if seed is not None:
        torch.manual_seed(seed)
    
    _mesh = copy.deepcopy(mesh)
    resolution = 256
    intrinsics_sampler = FovSampler(fov[0], fov[1])
    intr_samples = intrinsics_sampler.generate(num_src + num_tar)
    intrinsics_sampler.set_samples(intr_samples)

    source_camera_samplers = LocalCameraExtrinsicsSampler(mesh, cam_dist)
    source_sampling_weights = _get_src_sampling_weights(mesh, single_view_camera)
    source_camera_samplers.set_sampling_weights(source_sampling_weights)

    ### mesh custom material
    render_kwargs = {}
    texture_path = kwargs["custom_materials"]
    if texture_path is not None:
        albedo = (
            kaolin.io.utils.read_image(texture_path)
            .to(mesh.faces.device)
            .contiguous()
        )
        custom_materials = copy.copy(mesh.materials)
        custom_materials[0].diffuse_texture = albedo
        custom_render_kwargs = {"custom_materials": custom_materials}
    else:
        single_render_res = gloss.utils.render.render_all_features(single_view_camera, mesh, lighting=None, **render_kwargs)
        uv_dict, mask = backproject_render(mesh, single_view_camera, single_render_res, ['albedo'], 2048, 2048)
        # save the uv 
        uv = ((uv_dict['albedo'].squeeze(0) + 1.0) / 2.0).contiguous()
        
        custom_materials = copy.copy(mesh.materials)
        custom_materials[0].diffuse_texture = uv
        render_kwargs = {"custom_materials": custom_materials}
    
    if "source_camera_samples" in kwargs:
        src_samples = kwargs["source_camera_samples"]
        source_camera_samplers.set_samples(src_samples)
    else:
        src_samples = source_camera_samplers.generate(num_src)
        source_camera_samplers.set_samples(src_samples)

    target_camera_samplers = LocalCameraExtrinsicsSampler(mesh, cam_dist)
    target_sampling_weights = _get_tar_sampling_weights(mesh, single_view_camera)
    target_camera_samplers.set_sampling_weights(target_sampling_weights)

    if "target_camera_samples" in kwargs:
        tar_samples = kwargs["target_camera_samples"]
        target_camera_samplers.set_samples(tar_samples)
    else:
        tar_samples = target_camera_samplers.generate(num_tar)
        target_camera_samplers.set_samples(tar_samples)

    _channels = copy.deepcopy(channels)
    _channels.append("albedo_alpha")
    if "relative_positions" in _channels:
        _channels.append("positions")
    if "inpaint_mask" in _channels:
        _channels.remove("inpaint_mask")
        
    tar_gt_render_kwargs = {}
    if "custom_materials" in kwargs:
        if texture_path is not None:
            tar_gt_render_kwargs = custom_render_kwargs
    
    tar_gt_data = LocalMeshRendersDataset(
        _mesh,
        num_tar,
        resolution,
        intrinsics_sampler,
        target_camera_samplers,
        channels=_channels,
        cache=False,
        render_kwargs=tar_gt_render_kwargs,
    )
    if "relative_positions" in _channels:
        tar_gt_data.add_processor(
            RelativeToCenterValueProcessor(
                input_channel="positions", output_channel="relative_positions"
            ),
            add_channels=["relative_positions"],
        )

    src_data = LocalMeshRendersDataset(
        mesh,
        num_src,
        resolution,
        intrinsics_sampler,
        source_camera_samplers,
        channels=_channels,
        cache=False,
        render_kwargs=render_kwargs,
    )
    if "relative_positions" in _channels:
        src_data.add_processor(
            RelativeToCenterValueProcessor(
                input_channel="positions", output_channel="relative_positions"
            ),
            add_channels=["relative_positions"],
        )
    tar_data = LocalMeshRendersDataset(
        mesh,
        num_tar,
        resolution,
        intrinsics_sampler,
        target_camera_samplers,
        channels=_channels,
        cache=False,
        render_kwargs=render_kwargs,
    )
    if "relative_positions" in _channels:
        tar_data.add_processor(
            RelativeToCenterValueProcessor(
                input_channel="positions", output_channel="relative_positions"
            ),
            add_channels=["relative_positions"],
        )
    
    eval_data = {
        "mesh": _mesh,
        "src": src_data,
        "tar": tar_data,
        "tar_gt": tar_gt_data,
        "num_src": num_src,
        "num_tar": num_tar,
        "source_camera_samples": src_samples,
        "target_camera_samples": tar_samples,
    }
    return eval_data


def eval_data_to_samples(eval_data):
    """
    a list of the sample inputs
    """
    _, h, w, _ = eval_data["src"][0]["albedo"].shape
    device = eval_data["src"][0]["albedo"].device

    input = []
    for i in range(eval_data["num_src"]):
        src_sample = eval_data["src"][i]
        src_sample["albedo_alpha"] = (
            1.0 - 2 * torch.all((src_sample["albedo"] + 1.0) < 1e-6, dim=-1).int() # FIXME: Hacky mask
        ).unsqueeze(-1)
        src_sample["albedo"] = src_sample["albedo"] * (src_sample["albedo_alpha"].clip(0.0, 1.0))
        src_sample["inpaint_mask"] = src_sample["albedo_alpha"].clip(0.0, 1.0)
        input.append(src_sample)
    for i in range(eval_data["num_tar"]):
        tar_sample = eval_data["tar"][i]
        tar_sample["albedo_alpha"] = (
            1.0 - 2 * torch.all((tar_sample["albedo"] + -1.0) < 1e-6, dim=-1).int()
        ).unsqueeze(-1)
        tar_sample["albedo"] = tar_sample["albedo"] * (tar_sample["albedo_alpha"].clip(0.0, 1.0))
        tar_sample["inpaint_mask"] = tar_sample["albedo_alpha"].clip(0.0, 1.0) #torch.zeros((1, h, w, 1)).to(device)
        input.append(tar_sample)
    return input


def _get_src_sampling_weights(mesh, camera):
    # consider faces with relatively high pixel counts
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
    threshold_1 = torch.quantile(face_pixel_counts[face_pixel_counts > 0].float(), 0.6)
    face_weights[face_pixel_counts >= threshold_1] = 1
    threshold_2 = torch.quantile(face_pixel_counts[face_pixel_counts > 0].float(), 0.8)
    face_weights[face_pixel_counts >= threshold_2] = 2
    return face_areas * face_weights


def _get_tar_sampling_weights(mesh, camera):
    # only consider faces that does not have any face pixel counts
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
    return face_areas * face_weights


def _eval_data_to_samples(eval_data):
    """
    a list of the sample inputs
    """
    _, h, w, _ = eval_data["src"][0]["albedo"].shape
    device = eval_data["src"][0]["albedo"].device

    input = []
    for i in range(eval_data["num_src"]):
        src_sample = eval_data["src"][i]
        src_sample["albedo_alpha"] = (
            1.0 - 2 * torch.all(src_sample["albedo"] <= 0.0, dim=-1).int() # FIXME: Hacky mask
        ).unsqueeze(-1)
        src_sample["albedo"] = src_sample["albedo"] * src_sample["albedo_alpha"]
        src_sample["inpaint_mask"] = torch.ones((1, h, w, 1)).to(device) * torch.clip(
            src_sample["albedo_alpha"], 0.0, 1.0
        )
        input.append(src_sample)
    for i in range(eval_data["num_tar"]):
        tar_sample = eval_data["tar"][i]
        tar_sample["albedo_alpha"] = (
            1.0 - 2 * torch.all(tar_sample["albedo"] <= 0.0, dim=-1).int()
        ).unsqueeze(-1)
        tar_sample["inpaint_mask"] = tar_sample["albedo_alpha"].clip(0, 1) #torch.zeros((1, h, w, 1)).to(device)
        input.append(tar_sample)
    return input


def eval_data_to_gt(eval_data):
    gt, gt_mask = [], []
    for i in range(eval_data["num_tar"]):
        gt.append((eval_data["tar_gt"][i]["albedo"] + 1.0) / 2)  # 1 h w c [examine dimensions]
        gt_mask.append(1.0 - 2 * torch.all(eval_data["tar_gt"][i]["albedo"] == -1.0, dim=-1).int())
    gt = torch.cat(gt, dim=0)
    gt_mask = torch.cat(gt_mask, dim=0).unsqueeze(-1)
    return gt, gt_mask


def calculate_metrics(
    predictions: torch.Tensor, ground_truths: torch.Tensor, device=torch.device("cuda")
):
    lpip = lpips.LPIPS(net="vgg").to(device)
    fid = FrechetInceptionDistance().to(device)
    fid_metric = _calculate_metrics(predictions, ground_truths, "FID", fid)
    lpips_metric = _calculate_metrics(predictions, ground_truths, "LPIPS", lpip)
    del lpip
    del fid
    reclaim_cuda_memory()
    return {"FID": "{:.2f}".format(fid_metric), "lpips": "{:.2f}".format(lpips_metric)}


def _calculate_metrics(imgs1, imgs2, fn_name, fn):
    if fn_name == "FID":
        fn.update(imgs1.clip(0.0, 1.0), is_real=False)
        fn.update(imgs2.clip(0.0, 1.0), is_real=True)

        metrics = fn.compute()
        return metrics
    if fn_name == "LPIPS":
        metrics = torch.mean(fn(imgs1, imgs2))  # TODO: memory control
        return metrics


def save_dict_to_csv(data, fp, overwrite):
    f_exist = os.path.exists(fp) and os.path.getsize(fp) > 0
    append_true = not overwrite and f_exist
    with open(fp, mode="a" if append_true else "w", newline="") as file:
        fieldnames = data.keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not f_exist:
            writer.writeheader()
        writer.writerow(data)
    print("data successfully written to file path")
    
    
def _to_dict(tagged: TaggedTensor) -> dict[str, Tensor]:
    """("albedo", T) → {"albedo": T}"""
    return {name: t.cpu() for name, t in tagged}


def save_eval_tensors(
    out_fp,
    test_ground_truths: TaggedTensor,
    test_ground_truths_mask: TaggedTensor,
    test_raw_inputs: TaggedTensor,
):
    payload = {
        "ground_truths":      _to_dict(test_ground_truths),
        "ground_truths_mask": _to_dict(test_ground_truths_mask),
        "raw_inputs":         _to_dict(test_raw_inputs),
    }
    Path(out_fp).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_fp)          # binary .pt file (zip-based by default)


# ─── load_eval_tensors.py ──────────────────────────────────────────────────────
def _to_list(tagged_dict: dict[str, Tensor]) -> TaggedTensor:
    """{"albedo": T} → [("albedo", T)]  (order doesn’t matter for tensors)."""
    return list(tagged_dict.items())


def load_eval_tensors(fp, device="cpu"):
    device = torch.device(device)
    data = torch.load(fp, map_location=device)
    test_ground_truths      = _to_list(data["ground_truths"])
    test_ground_truths_mask = _to_list(data["ground_truths_mask"])
    test_raw_inputs         = _to_list(data["raw_inputs"])
    return test_ground_truths, test_ground_truths_mask, test_raw_inputs


def save_dict_to_csv(data, fp, overwrite):
    f_exist = os.path.exists(fp) and os.path.getsize(fp) > 0
    append_true = not overwrite and f_exist
    with open(fp, mode="a" if append_true else "w", newline="") as file:
        fieldnames = data.keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not f_exist:
            writer.writeheader()
        writer.writerow(data)
    print("data successfully written to file path")


def extract_view_numbers(folder):
    """
    Extract numbers from files named like view%04d.png in a given folder.

    Args:
        folder (str): Path to the folder containing PNG files.

    Returns:
        list[int]: Sorted list of extracted numbers.
    """
    pattern = re.compile(r"view(\d{4})\.png$")
    numbers = []

    for fname in os.listdir(folder):
        match = pattern.match(fname)
        if match:
            numbers.append(int(match.group(1)))

    return sorted(numbers)