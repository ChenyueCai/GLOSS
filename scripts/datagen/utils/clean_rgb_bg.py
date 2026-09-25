# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os, yaml
import torch
import kaolin
from gloss.utils.kaolin_utils import camera_from_meta, load_mesh
from gloss.utils.render_fast import custom_mesh_batched_render
from gloss.utils.paths import get_data_dir
import torchvision


objects = ["brick", "cabbage", "croissant", "dirty_tire", "fire_hydrant", "gourd", "koi_fish", "rusty_barrel_metal", "sea_urchin_shell", "turtle"]
data_dir = str(get_data_dir())

for o in objects:
    mesh = load_mesh(f"{data_dir}/mesh/{o}/scene.gltf")
    camera_dir = f"{data_dir}/test_metas/{o}"
    view_dir = f"{data_dir}/rgb_views/{o}"
    save_dir = f"{data_dir}/rgb_views_cleaned/{o}"
    os.makedirs(save_dir, exist_ok=True)
    for i in range(50):
        view = torchvision.io.read_image(f"{view_dir}/view{i:04d}.png").float() / 255.0
        with open(f"{camera_dir}/view{i:04d}.yml", 'r') as f:
            # read yml file 
            camera_meta = yaml.safe_load(f)
        print(camera_meta)
        camera = camera_from_meta(camera_meta["camera"]).to(mesh.vertices.device)
        batch_camera = kaolin.render.camera.Camera.cat([camera])
        render_res = custom_mesh_batched_render(batch_camera, mesh, mesh.materials[0].diffuse_texture, mesh.materials[0].hwc().normals_texture,
                                                requires_positions=True, process_as_albedo=False, backend="cuda")
        background_mask = (torch.abs(render_res["camera_normals"]) < 0.01).all(dim=-1, keepdim=True).float()
        print(background_mask.shape, view.shape)
        masked_view = torch.clip(background_mask.squeeze(-1) + view.cuda(), 0, 1)
        torchvision.utils.save_image(masked_view.cpu(), f"{save_dir}/view{i:04d}.png")
        