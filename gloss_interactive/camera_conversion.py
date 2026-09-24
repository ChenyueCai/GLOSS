import json
import os
from typing import List, Optional, Sequence

import torch
import kaolin
import numpy as np


def camera_to_meta_dict(camera: "kaolin.render.camera.Camera") -> dict:
    """Serialize a single kaolin Camera into a dict that round-trips through
    ``intrinsics_from_meta`` / ``extrinsics_from_meta`` (vertical fov, radians).

    Includes a few redundant fields (fov_x, focal_x, focal_y) for downstream
    inspection, but only ``width/height/fov/shift_x/shift_y/near/far`` and
    ``view_matrix`` are needed to rebuild the camera.
    """
    intr = camera.intrinsics
    extr = camera.extrinsics
    fov_y = float(intr.fov(kaolin.render.camera.CameraFOV.VERTICAL, in_degrees=False).reshape(-1)[0].item())
    fov_x = float(intr.fov(kaolin.render.camera.CameraFOV.HORIZONTAL, in_degrees=False).reshape(-1)[0].item())
    view_matrix = extr.view_matrix().detach().cpu().reshape(4, 4).tolist()
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fov": fov_y,
        "shift_x": float(intr.x0.reshape(-1)[0].item()),
        "shift_y": float(intr.y0.reshape(-1)[0].item()),
        "near": float(intr.near),
        "far": float(intr.far),
        "view_matrix": view_matrix,
        "fov_x": fov_x,
        "focal_x": float(intr.focal_x.reshape(-1)[0].item()),
        "focal_y": float(intr.focal_y.reshape(-1)[0].item()),
    }


def dump_cameras_json(
    fp: str,
    cameras: Sequence["kaolin.render.camera.Camera"],
    camera_configs: Optional[Sequence[object]] = None,
    extra: Optional[dict] = None,
) -> None:
    """Write cameras.json next to a logged inference's images.

    ``camera_configs`` is an optional parallel list of ``CameraConfig`` (used
    by backproject downstream); we save its public attrs verbatim. ``extra``
    is merged into the top-level dict (e.g. cam_source, use_syncmvd).
    """
    records: List[dict] = []
    for i, cam in enumerate(cameras):
        rec = camera_to_meta_dict(cam)
        if camera_configs is not None and i < len(camera_configs):
            cfg = camera_configs[i]
            rec["camera_config"] = {
                k: getattr(cfg, k) for k in (
                    "fov", "resolution", "dist", "spacing",
                    "backproject_pix_count", "backproject_max_angle",
                ) if hasattr(cfg, k)
            }
        records.append(rec)
    payload = {"cameras": records}
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w") as f:
        json.dump(payload, f, indent=2)


def intrinsics_from_meta(meta):
    """
    Expects parsed json like the following:
    {
        "width": 512,
        "height": 512,
        "fov": 0.988324761390686,
        "shift_x": 0.0,
        "shift_y": 0.0,
        "near": 0.10000000149011612,
        "far": 100.0
    }

    Returns:
        (kaolin.render.camera.PinholeIntrinsics)
    """
    return kaolin.render.camera.PinholeIntrinsics.from_fov(
        width=meta['width'], height=meta['height'],
        fov=meta['fov'],  # bpy.data.cameras['Camera'].angle
        x0=meta['shift_x'],  # bpy.data.cameras['Camera'].shift_x
        y0=meta['shift_y'],  # bpy.data.cameras['Camera'].shift_y
        near=meta['near'],  # bpy.data.cameras['Camera'].clip_start
        far=meta['far']  # bpy.data.cameras['Camera'].clip_end
    )

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

# TODO:
def get_kaolin_camera_from_json(meta):
    intrinsics, extrinsics = intrinsics_from_meta(meta), extrinsics_from_meta(meta)
    
