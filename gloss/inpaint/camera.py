# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import kaolin
from gloss.utils.single_view import cartesian2spherical
import numpy as np
import torch.nn.functional as F

def get_camera_from_spherical_param(azi, elev, dist, resolution, fov, device):
    eye = kaolin.render.lighting.sg_direction_from_azimuth_elevation(azi, elev).squeeze(-1).to(device)
    world_up = torch.tensor([0.0, 1.0, 0.0]).to(device)
    up = (
        world_up - eye[:, 1][:, None] * eye
    )  # TODO: examine if camera up direction makes a difference
    _camera = kaolin.render.camera.Camera(
        kaolin.render.camera.CameraExtrinsics.from_lookat(
            eye=eye * dist,
            at=torch.tensor([0.0, 0.0, 0.0]),
            up=up,
            dtype=torch.float32,
            device=device,
        ),
        kaolin.render.camera.PinholeIntrinsics.from_fov(
            width=resolution,
            height=resolution,
            fov=fov,
            device=device,
            dtype=torch.float32)
    )
    _camera = _camera.to(device)
    return _camera


def camera_strategy(init_camera, cone_angles, n_samples, dist, resolution, fov, device):
    """_summary_

    Args:
        init_camera (_type_): _description_
        cone_angles (_type_): _description_
        n_samples (_type_): _description_
        dist (_type_): _description_
        resolution (_type_): _description_
        fov (_type_): _description_
        device (_type_): _description_
    """


    num_rings = len(cone_angles)

    init_camera_cartesian = init_camera.extrinsics.cam_pos().flatten()
    #print(init_camera_cartesian)
    init_params = cartesian2spherical(init_camera_cartesian)
    init_a, init_e = init_params[0], init_params[1]
    
    cameras = [init_camera]
    for i in range(num_rings):
        azelevs = generate_cone_azelev(init_a, init_e, cone_angles[i], n_samples[i], device)
        for ae in azelevs:
            a, e = ae[0], ae[1]
            cam = get_camera_from_spherical_param(a, e, dist, resolution, fov, device)
            cameras.append(cam)
    
    final_e, final_a = -init_e, -init_a 
    cameras.append(get_camera_from_spherical_param(final_a, final_e, dist, resolution, fov, device))
    return cameras


def generate_cone_azelev(
    azi: float,
    elev: float,
    cone_angle: float,
    n_samples: int = 16,
    device='cuda',
    dtype=torch.float32,
):
    """_summary_

    Args:
        azi (float): given vector's azimuth
        elev (float): given vector's elevation
        cone_angle (float): cone angle in degree
        n_samples (int, optional): number of smaples
        device (str, optional): _description_. Defaults to 'cuda'.
        dtype (_type_, optional): _description_. Defaults to torch.float32.

    Raises:
        ValueError: _description_

    Returns:
        _type_: a list of [a, e] pairs
    """
    axis = kaolin.render.lighting.sg_direction_from_azimuth_elevation(azi, elev).squeeze(0)
    norm_axis = axis.norm(p=2)
    if norm_axis < 1e-9:
        raise ValueError("Invalid axis vector. Are az/el out of range?")

    v0 = axis / norm_axis  # shape (3,)
    alpha = torch.tensor(cone_angle, dtype=dtype).to(device) / 180 * torch.pi

    eps = 1e-7
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype).to(device)
    cross_v0_z = torch.cross(v0, z_axis)
    if cross_v0_z.norm(p=2) < eps:
        b1 = torch.tensor([1.0, 0.0, 0.0], dtype=dtype).to(device)
    else:
        b1 = z_axis

    proj = torch.dot(b1, v0)
    b1 = b1 - proj * v0
    b1 = b1 / b1.norm(p=2)

    b2 = torch.cross(v0, b1)
    b2 = b2 / b2.norm(p=2)
    
    ring_azelevs = []
    for i in range(n_samples):
        phi = torch.tensor(2.0 * torch.pi * i / n_samples)
        cosphi = torch.cos(phi)
        sinphi = torch.sin(phi)
        v = v0 * torch.cos(alpha) + (b1 * cosphi + b2 * sinphi) * torch.sin(alpha)
        v = v / v.norm(p=2)  # normalize for numerical safety
        spherical_params = cartesian2spherical(v)
        ring_azelevs.append([spherical_params[0].cuda(), spherical_params[1].cuda()])
    return ring_azelevs

def sample_sphere_by_rings(spacing=0.25, elevations=None):
    """
    Sample sphere by creating rings of points at various elevations
    The distance between points in each ring is determined by the spacing
    """
    points = []

    if elevations is None:
        num_rings = int(np.pi / spacing)
        elevation_angles = np.linspace(-np.pi / 2, np.pi / 2, num_rings)
    else:
        elevation_angles = elevations

    for theta in elevation_angles:
        ring_radius = np.cos(theta)
        if ring_radius < 1e-5:
            # x = 0
            # y = 0
            # z = np.sin(theta)
            # points.append([x, y, z])
            continue

        circumference = 2 * np.pi * ring_radius
        num_points = max(1, int(circumference / spacing))
        azimuth_angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        for phi in azimuth_angles:
            x = np.cos(phi) * ring_radius
            y = np.sin(phi) * ring_radius
            z = np.sin(theta)
            points.append([x, y, z])

    return np.array(points)


def rotation_between_z(vec):
    """
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    Args:
        vec: [..., 3]

    Returns:
        R: [..., 3, 3]

    """
    v1 = -vec[..., 1]
    v2 = vec[..., 0]
    v3 = torch.zeros_like(v1)
    v11 = v1 * v1
    v22 = v2 * v2
    v33 = v3 * v3
    v12 = v1 * v2
    v13 = v1 * v3
    v23 = v2 * v3
    cos_p_1 = (vec[..., 2] + 1).clamp_min(1e-7)
    R = torch.zeros(vec.shape[:-1] + (3, 3,), dtype=torch.float32, device="cuda")
    R[..., 0, 0] = 1 + (-v33 - v22) / cos_p_1
    R[..., 0, 1] = -v3 + v12 / cos_p_1
    R[..., 0, 2] = v2 + v13 / cos_p_1
    R[..., 1, 0] = v3 + v12 / cos_p_1
    R[..., 1, 1] = 1 + (-v33 - v11) / cos_p_1
    R[..., 1, 2] = -v1 + v23 / cos_p_1
    R[..., 2, 0] = -v2 + v13 / cos_p_1
    R[..., 2, 1] = v1 + v23 / cos_p_1
    R[..., 2, 2] = 1 + (-v22 - v11) / cos_p_1
    R = torch.where((vec[..., 2] + 1 > 0)[..., None, None], R,
                    -torch.eye(3, dtype=torch.float32, device="cuda").expand_as(R))
    return R


def uniform_sphere_camera_strategy(init_camera, dist, spacing, resolution, fov, device):
    lookat = torch.tensor([0.0, 0.0, 0.0]).to(device)
    sampled_eye = torch.from_numpy(sample_sphere_by_rings(spacing)).float().to(device)
    if init_camera is not None:
        up = lookat - init_camera.extrinsics.cam_pos().reshape(3)
        forward = -1 * init_camera.extrinsics.cam_forward().reshape(1, 3).to(device)
        rotation_matrix = rotation_between_z(forward)
        incident_dirs = rotation_matrix @ sampled_eye.permute(1, 0)
        sampled_eye = F.normalize(incident_dirs, dim=-2).transpose(-1, -2).reshape(-1, 3)
    else:
        up = torch.tensor([0.0, 1.0, 0.0]).to(device)

    cameras = []
    for eye in sampled_eye:
        forward = lookat - eye
        right = torch.cross(up, forward)
        cam_up = torch.cross(forward, right)
        camera = kaolin.render.camera.Camera(
            kaolin.render.camera.CameraExtrinsics.from_lookat(
                    eye=eye * dist,
                    at=lookat,
                    up=cam_up,
                    dtype=torch.float32,
                    device=device),
            kaolin.render.camera.PinholeIntrinsics.from_fov(
                    width=resolution,
                    height=resolution,
                    fov=fov,
                    device=device,
                    dtype=torch.float32)
        )
        cameras.append(camera)
    return cameras


def turntable_camera_strategy(dist, spacing, resolution, fov, device):
    lookat = torch.tensor([0.0, 0.0, 0.0]).to(device)
    sampled_eye = torch.from_numpy(sample_sphere_by_rings(spacing, elevations=[30/180*np.pi, -30/180*np.pi])).float().to(device)
    sampled_eye = sampled_eye[..., [0, 2, 1]]
    up = torch.tensor([0.0, 1.0, 0.0]).to(device)

    cameras = []
    for eye in sampled_eye:
        camera = kaolin.render.camera.Camera(
            kaolin.render.camera.CameraExtrinsics.from_lookat(
                    eye=eye * dist,
                    at=lookat,
                    up=up,
                    dtype=torch.float32,
                    device=device),
            kaolin.render.camera.PinholeIntrinsics.from_fov(
                    width=resolution,
                    height=resolution,
                    fov=fov,
                    device=device,
                    dtype=torch.float32)
        )
        cameras.append(camera)
    return cameras


if __name__ == "__main__":
    import open3d as o3d
    import numpy as np

    res = 256
    init_camera = kaolin.render.easy_render.default_camera(res).cuda()

    cameras = uniform_sphere_camera_strategy(init_camera, 10.0, 0.25, res, 0.3, "cuda")
    print(len(cameras))
    vizualizer = o3d.visualization.Visualizer()
    vizualizer.create_window()

    _, _, fx, fy = init_camera.intrinsics.parameters()[0]
    cx, cy = res / 2, res / 2
    intrinsics = np.array([[fx.item(), 0, cx], [0, fy.item(), cy], [0, 0, 1]])
    extrinsics = init_camera.extrinsics.view_matrix().reshape(4, 4).cpu().numpy()
    extrinsics[1:3] *= -1  # opengl to opencv
    cameraLines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=res, view_height_px=res,
                                                                   intrinsic=intrinsics, extrinsic=extrinsics)
    cameraLines.paint_uniform_color(np.array([0, 1, 0]).reshape(3, 1))
    vizualizer.add_geometry(cameraLines)

    for camera in cameras:
        _, _, fx, fy = init_camera.intrinsics.parameters()[0]
        cx, cy = res / 2, res / 2
        intrinsics = np.array([[fx.item(), 0, cx], [0, fy.item(), cy], [0, 0, 1]])
        extrinsics = camera.extrinsics.view_matrix().reshape(4, 4).cpu().numpy()
        extrinsics[1:3] *= -1  # opengl to opencv
        cameraLines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=res, view_height_px=res,
                                                                       intrinsic=intrinsics, extrinsic=extrinsics)
        cameraLines.paint_uniform_color(np.array([0, 0, 1]).reshape(3, 1))
        vizualizer.add_geometry(cameraLines)

    vizualizer.run()
if __name__ == "__main__":
    import open3d as o3d
    import numpy as np

    res = 256
    init_camera = kaolin.render.easy_render.default_camera(res).cuda()

    cone_angles = [15, 30, 60, 90, 120, 150, 165]
    n_samples = [8, 8, 8, 8, 8, 8, 8]
    cameras = camera_strategy(init_camera, cone_angles, n_samples, 10.0, res, 0.3, "cuda")
    print(len(cameras))
    vizualizer = o3d.visualization.Visualizer()
    vizualizer.create_window()

    _, _, fx, fy = init_camera.intrinsics.parameters()[0]
    cx, cy = res / 2, res / 2
    intrinsics = np.array([[fx.item(), 0, cx], [0, fy.item(), cy], [0, 0, 1]])
    extrinsics = init_camera.extrinsics.view_matrix().reshape(4, 4).cpu().numpy()
    extrinsics[1:3] *= -1  # opengl to opencv
    cameraLines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=res, view_height_px=res,
                                                                   intrinsic=intrinsics, extrinsic=extrinsics)
    cameraLines.paint_uniform_color(np.array([0, 1, 0]).reshape(3, 1))
    vizualizer.add_geometry(cameraLines)

    for camera in cameras:
        _, _, fx, fy = init_camera.intrinsics.parameters()[0]
        cx, cy = res / 2, res / 2
        intrinsics = np.array([[fx.item(), 0, cx], [0, fy.item(), cy], [0, 0, 1]])
        extrinsics = camera.extrinsics.view_matrix().reshape(4, 4).cpu().numpy()
        extrinsics[1:3] *= -1  # opengl to opencv
        cameraLines = o3d.geometry.LineSet.create_camera_visualization(view_width_px=res, view_height_px=res,
                                                                       intrinsic=intrinsics, extrinsic=extrinsics)
        cameraLines.paint_uniform_color(np.array([0, 0, 1]).reshape(3, 1))
        vizualizer.add_geometry(cameraLines)

    vizualizer.run()
