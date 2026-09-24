# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import math
import torch
import torchvision
import matplotlib.pyplot as plt
import kaolin
import gloss
from gloss.utils.render_fast import custom_mesh_batched_render
from gloss.utils.single_view import backproject_render
from kornia.morphology import erosion
from dataclasses import dataclass


@dataclass
class CameraConfig:
    fov: float = 0.4
    resolution: int = 256
    dist: float = 0.75
    spacing: float = 0.25
    backproject_pix_count: int = 2
    backproject_max_angle: int = 90


class CameraLogger(object):
    """Utility class to log camera positions and create debug visualizations."""
    
    def __init__(self, h, w, save_dir, device, max_num=600):
        self.camera_debug_map = None
        self.camera_count_map = None 
        self.cameras = []
        self.max_num_camera = max_num
        self.save_dir = save_dir
        cmap = plt.get_cmap('tab20')
        self.camera_colors = [cmap(i % cmap.N)[:3] for i in range(self.max_num_camera)]
        self.device = device

    def log_camera(self, camera, update_area=None):
        idx = len(self.cameras)
        _camera_color = self.camera_colors[idx]
        _camera_color_tensor = torch.tensor(_camera_color, dtype=torch.float32).view(3, 1, 1).to(self.device)
        if update_area is not None:
            _camera_update_area = update_area
        _camera_debug_map = _camera_color_tensor.expand(3, 2048, 2048).unsqueeze(0) * _camera_update_area 
        if idx == 0:
            self.camera_debug_map = _camera_debug_map
            self.camera_count_map = _camera_update_area
        else:
            self.camera_debug_map += _camera_debug_map
            self.camera_count_map += _camera_update_area
        self.cameras.append(camera)
    
    def save_log(self):
        _save_img = self.camera_debug_map / (self.camera_count_map + 1e-8)
        torchvision.utils.save_image(_save_img, f"{self.save_dir}/camera_log_{len(self.cameras)}.png")
    
    def save_log_mp4(self):
        os.system(f"ffmpeg -y -framerate 4 -i {self.save_dir}/camera_log_%d.png {self.save_dir}/camera_log.mp4")
        
    def clear(self):
        self.camera_debug_map = None
        self.camera_count_map = None
        self.cameras = []


def get_camera_from_face(mesh, face_idx, camera_config, device, u=2/3, v=1/2):
    """Generate camera positioned on a specific mesh face."""
    vertices = mesh.face_vertices[face_idx]
    normals = mesh.face_normals[face_idx]
    w0 = 1 - u
    w1 = u * (1 - v)
    w2 = u * v

    points = w0 * vertices[0] + w1 * vertices[1] + w2 * vertices[2]
    normals = w0 * normals[0] + w1 * normals[1] + w2 * normals[2]

    cam_pos = points.unsqueeze(0)
    cam_normals = normals.unsqueeze(0)

    up = torch.tensor([[0.0, 1.0, 0.0]]).to(device)
    eye = cam_pos + torch.nn.functional.normalize(cam_normals) * camera_config.dist
    lookat = cam_pos
    camera = torch.cat([eye, lookat, up], dim=1)
    camera = gloss.utils.render.make_camera_from_extr_intr(camera[0],
                                                           camera_config.fov,
                                                           resolution=camera_config.resolution,
                                                           device=mesh.faces.device)
    return camera


def get_face_uv_pixel_counts(mesh, h, w, visualize=False):
    """Count pixels per face in UV space."""
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
    all_face_ids, counts = torch.unique(tex_face_idx[tex_face_idx != -1], return_counts=True)

    if visualize:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.imshow(tex_face_idx[0].cpu().numpy())
        plt.show()

    return all_face_ids, counts, tex_face_idx != -1


def backproject_view_to_texture(camera, view, mesh, texture_map, camera_config, device, 
                                latent=False, margin=False, view_margin=0, return_face_idx=False):
    """Backproject a view onto the mesh texture."""
    view_h, view_w = camera.height, camera.width
    pad = 1  # must be at least 1
    update_mask = torch.ones((1, view_h, view_w, 1), device=device, dtype=torch.float32) * -1
    update_mask[0, pad:view_h - pad, pad:view_w - pad] = 1

    batch_camera = kaolin.render.camera.Camera.cat([camera])
    normal_map = mesh.materials[0].hwc().normals_texture
    tex_h, tex_w = texture_map.shape[:2]
    if normal_map is not None:
        normal_map = torchvision.transforms.Resize((tex_h, tex_w))(normal_map.permute(2, 0, 1)).permute(1, 2, 0)
    render_res = custom_mesh_batched_render(batch_camera, mesh, texture_map, normal_map,
                                            requires_positions=True, process_as_albedo=False, backend="cuda")
    render_res["geo_camera_normals"] = render_res["camera_normals"]
    render_res["face_idx"] = render_res["face_idx"][..., 0]

    if margin and view_margin > 0:
        update_mask = update_mask == 1
        # mask removes areas too close to the edge of the foreground
        fg_mask = 1 - (torch.abs(render_res["geo_camera_normals"]) < 0.01).all(dim=-1, keepdim=True).float()
        if (fg_mask == 0).sum() > 0:
            kernel = torch.ones(view_margin, view_margin).to(device)
            fg_mask = erosion(fg_mask.permute(0, 3, 1, 2), kernel).permute(0, 2, 3, 1)
            update_mask &= (fg_mask == 1)
        # mask ignores positions too far away
        pos_mask = torch.norm(render_res["relative_positions"], dim=-1, keepdim=True) < 1.0
        update_mask &= pos_mask
        update_mask = update_mask.float() * 2 - 1
    
    if not latent:
        view = torch.cat([view, update_mask], dim=-1)
    render_res['albedo'] = view  # scaled from -1 to 1, BHWC

    # compute cos angle weights
    row, col = view_h // 2, view_w // 2
    center_normal = render_res['normals'][:, row, col, :].unsqueeze(1).unsqueeze(1)
    camera_dist = camera_config.dist
    view_direction = camera_dist * center_normal - render_res['relative_positions'][..., :3]
    view_direction = torch.nn.functional.normalize(view_direction, dim=-1)
    tex_normals = render_res['normals'][..., :3]
    cos_weight = torch.nn.functional.cosine_similarity(view_direction, tex_normals, dim=-1)[..., None]
    render_res['cos_weight'] = cos_weight

    if latent:
        min_pix = 1
        angle_dev = math.pi/2
    else:
        min_pix = camera_config.backproject_pix_count
        angle_dev = math.pi*camera_config.backproject_max_angle/180.0
        
    backprojection, mask, tex_face_idx = backproject_render(
        mesh, camera, render_res, ['albedo', 'cos_weight'],
        tex_h, tex_w,
        min_pixel_count=min_pix,
        max_angle_deviation=angle_dev,
        return_face_idx=True
    )

    if tex_face_idx is None:
        return None, None, None

    cos_weight = backprojection['cos_weight']
    if margin is not None and not latent:
        mask = backprojection['albedo'][..., -1:] / 2 + 0.5
    if return_face_idx:
        return backprojection['albedo'].permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2), tex_face_idx
    return backprojection['albedo'].permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2), cos_weight.permute(0, 3, 1, 2)


def check_camera_coverage(camera, mesh, curr_texture, camera_config, device, view_margin=0):
    """Check which areas of texture a camera can cover."""
    output = torch.ones(1, 3, camera_config.resolution, camera_config.resolution).to(device)
    backprojection, mask, tex_face_idx = backproject_view_to_texture(
        camera, output.permute(0, 2, 3, 1), mesh, curr_texture.squeeze(0).permute(1, 2, 0).contiguous(), 
        camera_config, device, margin=True, view_margin=view_margin, return_face_idx=True)
    if backprojection is None:
        return None, None
    return mask, tex_face_idx


def load_texture_image(path, device, h, w):
    """Load and process texture image."""
    albedo = kaolin.io.utils.read_image(path).to(device).permute(2, 0, 1)
    resize = torchvision.transforms.Resize((h, w))
    albedo = resize(albedo)
    if albedo.shape[0] == 4:
        mask = (albedo[3:] > 0.9).float()
        albedo[:3] = albedo[:3] * mask
        albedo[3:] = albedo[3:] * 2 - 1
        return albedo[None].contiguous(), mask[None].int()
    else:
        return albedo[None].contiguous() * 2 - 1, None


def setup_logging_directories(save_dir):
    """Create necessary logging directories."""
    uv_logs = []
    save_texture_dir = os.path.join(save_dir, 'texture')
    os.makedirs(save_texture_dir, exist_ok=True)
    model_debug_dir = os.path.join(save_dir, 'log')
    os.makedirs(model_debug_dir, exist_ok=True)
    cache_dir = os.path.join(save_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    camera_selection_info_dir = os.path.join(save_dir, 'camera_selection_info')
    os.makedirs(camera_selection_info_dir, exist_ok=True)
    
    return {
        'uv_logs': uv_logs,
        'save_texture_dir': save_texture_dir,
        'model_debug_dir': model_debug_dir,
        'cache_dir': cache_dir,
        'camera_selection_info_dir': camera_selection_info_dir
    }


def save_video_from_tensors(images, path, fps=5):
    """Save list of image tensors as video."""
    processed_images = []
    for img in images:
        # Ensure tensor is on CPU and convert to proper format
        img = img.detach().cpu().squeeze(0)[:3]
        img = (img.detach().cpu() * 255).clip(0, 255).to(torch.uint8).permute(1, 2, 0)
        processed_images.append(img)
    
    if processed_images:
        images_stack = torch.stack(processed_images, 0)
        torchvision.io.write_video(path, images_stack, fps=fps)