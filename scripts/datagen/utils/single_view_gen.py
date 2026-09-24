#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Generate ControlNet-guided single views of a mesh.

For a given mesh this script:
  1. Loads the (optionally yaw-rotated) mesh and normalizes it to a unit cube.
  2. Renders normal / geonormal / depth / canny conditions at a user-specified
     list of (azimuth, elevation) camera poses.
  3. Runs a diffusers-native Stable-Diffusion + ControlNet pipeline driven by
     those conditions and writes the generated images + condition inputs under

         <mesh_root>/<mesh>/single_view/<config_name>/run-<run_id>/
             meta.json
             view<NN>/
                 <azi>_<elev>.png                 # generated sample
                 cond_normal.png                  # condition fed to pipeline
                 cond_depth.png
                 cond_canny.png
                 ...

The configs live in ``single_view_configs.yaml`` (see
``data/material-superres/metadata/single_view_configs.yaml``).  Default values
for strength / prompt / steps come from that file; the CLI can override.

Rating + orientation files are expected next to the generated run dir:

    <mesh_root>/<mesh>/single_view/orientation.json   # {"yaw": 0.0}
    <mesh_root>/<mesh>/single_view/<config>/ratings.json

The orientation file is read automatically; ratings are updated by the UI.
"""

from __future__ import annotations

import argparse
import json
import locale
import logging
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# ---- locale / pygltflib UTF-8 patch (same rationale as render_previews.py) ----
try:
    locale.setlocale(locale.LC_CTYPE, "C.UTF-8")
except locale.Error:
    pass

import pygltflib  # noqa: E402


def _utf8_load_json(cls, fname):  # type: ignore[no-redef]
    path = Path(fname)
    with open(fname, "r", encoding="utf-8") as f:
        obj = cls.gltf_from_json(f.read())
    obj._path = path.parent
    obj._name = path.name
    return obj


pygltflib.GLTF2.load_json = classmethod(_utf8_load_json)

warnings.filterwarnings("ignore", category=UserWarning, module=r"pygltflib.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"kaolin\.io\.gltf")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
import yaml  # noqa: E402

import kaolin  # noqa: E402
import kaolin.render.easy_render as easy_render  # noqa: E402

from gloss.utils.render import render_all_features  # noqa: E402
from gloss.utils.single_view import make_single_view_cam  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ----------------------- device / mesh helpers --------------------------------


def _pick_free_cuda_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    best_idx, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        try:
            free, _total = torch.cuda.mem_get_info(i)
        except Exception:
            continue
        if free > best_free:
            best_idx, best_free = i, free
    return f"cuda:{best_idx}"


def _force_materials_to_device(mesh, device: str) -> None:
    mats = getattr(mesh, "materials", None)
    if not mats:
        return

    def _migrate(m):
        try:
            return m.to(device)
        except Exception:
            return easy_render.default_material().to(device)

    if isinstance(mats[0], list):
        mesh.materials = [[_migrate(m) for m in group] for group in mats]
    else:
        mesh.materials = [_migrate(m) for m in mats]


def _apply_euler(mesh, pitch: float, yaw: float, roll: float) -> None:
    """Rotate mesh vertices in place by Euler angles, order XYZ (Three.js default).

    pitch = rotation around +X, yaw = rotation around +Y, roll = rotation around +Z.
    Final matrix is R = Rx(pitch) @ Ry(yaw) @ Rz(roll), applied as ``v' = R @ v``.
    """
    if abs(pitch) < 1e-7 and abs(yaw) < 1e-7 and abs(roll) < 1e-7:
        return
    dtype, device = mesh.vertices.dtype, mesh.vertices.device
    cx, sx = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    cz, sz = math.cos(roll),  math.sin(roll)
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=dtype, device=device)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=dtype, device=device)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=dtype, device=device)
    R = Rx @ Ry @ Rz
    mesh.vertices = mesh.vertices @ R.T


# --------------------------- condition helpers --------------------------------


def _depth_from_raw(raw_depth: torch.Tensor) -> torch.Tensor:
    """Convert raw_depth (B,H,W,1) in NDC (bg=0) to a 3-channel BCHW image in [0,1].

    Mirrors ``gloss.utils.diffusion_render.get_depth`` but operates on a local
    copy so repeated calls are side-effect free.
    """
    raw = raw_depth.clone()
    bg = torch.all(raw == 0.0, dim=3)
    raw[bg] = 1.0
    d = 1.0 / raw
    d = d.clamp(min=0).permute(0, 3, 1, 2)
    d = (d - d.amin()) / (d.amax() - d.amin() + 1e-8)
    return d.repeat(1, 3, 1, 1)


def _canny_from_rgb(rgb01_bhwc: torch.Tensor, low: int = 100, high: int = 200) -> torch.Tensor:
    """Canny edges from a BHWC [0,1] tensor → BCHW [0,1], 3-channel."""
    arr = (rgb01_bhwc.clip(0, 1).detach().cpu().squeeze(0).numpy() * 255).astype(np.uint8)
    edges = cv2.Canny(arr, low, high)
    edges3 = np.tile(edges[:, :, None], (1, 1, 3))
    t = torch.from_numpy(edges3).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t


def _normal_to_01(normal_bhwc_m11: torch.Tensor) -> torch.Tensor:
    """BHWC in [-1,1] → BCHW in [0,1]."""
    n = (normal_bhwc_m11 + 1.0) * 0.5
    return n.clamp(0, 1).permute(0, 3, 1, 2)


def _render_to_01(render_bhwc_m11: torch.Tensor) -> torch.Tensor:
    r = (render_bhwc_m11 + 1.0) * 0.5
    r = r.clamp(0, 1).permute(0, 3, 1, 2)
    if r.shape[1] == 4:
        r = r[:, :3]
    return r


def _render_conditions(mesh, azi: float, elev: float, viewdist: float, fov: float,
                       resolution: int, device: str) -> dict[str, torch.Tensor]:
    """Render all passes we may need as ControlNet conditions, in BCHW [0,1]."""
    cam = make_single_view_cam(
        azi_range=[azi, azi], elev_range=[elev, elev],
        view_dist_range=[viewdist, viewdist], fov_range=[fov, fov],
        resolution=resolution, device=device,
    )
    lighting = easy_render.default_lighting().to(device)
    passes = ["render", "camera_normals", "geo_camera_normals", "raw_depth", "mask"]
    res = render_all_features(cam, mesh, lighting=lighting, required_passes=passes)

    # Canny prefers a uint8 RGB in [0,1] in BHWC; normals/renders are in [-1,1] BHWC.
    normal01 = _normal_to_01(res["camera_normals"])
    geonormal01 = _normal_to_01(res["geo_camera_normals"])
    render01 = _render_to_01(res["render"])
    depth01 = _depth_from_raw(res["raw_depth"])
    canny = _canny_from_rgb(res["camera_normals"].add(1).mul_(0.5))
    canny_geo = _canny_from_rgb(res["geo_camera_normals"].add(1).mul_(0.5))
    canny_render = _canny_from_rgb(res["render"].add(1).mul_(0.5))
    mask01 = ((res["mask"] + 1) * 0.5).clamp(0, 1).permute(0, 3, 1, 2)
    if mask01.shape[1] == 1:
        mask01 = mask01.repeat(1, 3, 1, 1)

    return {
        "normal":       normal01,
        "geonormal":    geonormal01,
        "render":       render01,
        "depth":        depth01,
        "canny":        canny,
        "canny_geo":    canny_geo,
        "canny_render": canny_render,
        "mask":         mask01,
    }


COND_TYPE_TO_PASS = {
    "normal":       "normal",
    "geonormal":    "geonormal",
    "depth":        "depth",
    "canny":        "canny",
    "canny_geo":    "canny_geo",
    "canny_render": "canny_render",
}


def camera_phrase(azi: float, elev: float) -> str:
    """Return a natural-language description of the camera viewpoint.

    Maps (azimuth, elevation) → one of 8 labels: ``{upper,lower} {left,right} {front,back}``.
    Assumes the mesh's canonical forward axis is +Z (set by the orientation slider),
    and that the eye position is
    ``(cos(elev)cos(azi), sin(elev), cos(elev)sin(azi))`` — kaolin's convention.

    Thresholds:
      elev >  +0.15 rad → "upper"
      elev <  −0.15 rad → "lower"
      otherwise         → "level"

    Azimuth (mod 2π):
      (0, π/2)      → "right front"    (eye at +X, +Z)
      (π/2, π)      → "left front"     (eye at −X, +Z)
      (π, 3π/2)     → "left back"      (eye at −X, −Z)
      (3π/2, 2π)    → "right back"     (eye at +X, −Z)
    """
    if elev > 0.15:
        vert = "upper"
    elif elev < -0.15:
        vert = "lower"
    else:
        vert = "level"
    a = azi % (2 * math.pi)
    if a < math.pi / 2:
        horiz = "right front"
    elif a < math.pi:
        horiz = "left front"
    elif a < 3 * math.pi / 2:
        horiz = "left back"
    else:
        horiz = "right back"
    return f"{vert} {horiz}"


# ---------------------- diffusion pipeline loading ----------------------------
#
# Supported `sd_variant` values:
#   sd15        StableDiffusionControlNetPipeline + N × ControlNetModel
#   sd21        same (SD 2.1 base)
#   sdxl        StableDiffusionXLControlNetPipeline + N × ControlNetModel
#   sdxl_union  StableDiffusionXLControlNetUnionPipeline + ControlNetUnionModel
#               Config provides `controlnet_model` (single union id); each entry in
#               `controlnets` supplies {type, control_mode, strength}.
#   flux_union  FluxControlNetPipeline + FluxMultiControlNetModel([FluxControlNetModel])
#               Config provides `controlnet_model` (single union id); each entry in
#               `controlnets` supplies {type, control_mode, strength}.
#   flux_multi  FluxControlNetPipeline + FluxMultiControlNetModel([...])
#               Each entry in `controlnets` loads a distinct FluxControlNetModel.
#               Entries may still carry `control_mode` (None for single-mode models).
#   sd3         StableDiffusion3ControlNetPipeline + N × SD3ControlNetModel


def _parse_dtype_cfg(config: dict, device: str) -> torch.dtype:
    s = config.get("dtype", "float16")
    if not device.startswith("cuda"):
        return torch.float32
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(
        s, torch.float16,
    )


def _load_pipeline(config: dict[str, Any], device: str):
    variant = config.get("sd_variant", "sd15")
    dtype = _parse_dtype_cfg(config, device)
    log.info("Variant=%s  dtype=%s  device=%s", variant, dtype, device)

    if variant in ("sd15", "sd21"):
        return _load_sd_legacy(config, device, dtype, xl=False)
    if variant == "sdxl":
        return _load_sd_legacy(config, device, dtype, xl=True)
    if variant == "sdxl_union":
        return _load_sdxl_union(config, device, dtype)
    if variant == "flux_union":
        return _load_flux(config, device, dtype, union=True)
    if variant == "flux_multi":
        return _load_flux(config, device, dtype, union=False)
    if variant == "sd3":
        return _load_sd3(config, device, dtype)
    raise ValueError(f"unknown sd_variant: {variant!r}")


def _load_sd_legacy(config: dict, device: str, dtype: torch.dtype, xl: bool):
    from diffusers import (
        ControlNetModel, StableDiffusionControlNetPipeline,
        StableDiffusionXLControlNetPipeline, UniPCMultistepScheduler, AutoencoderKL,
    )
    cnets = []
    for c in config["controlnets"]:
        log.info("  loading ControlNet %s (%s)", c["type"], c["model"])
        cnets.append(ControlNetModel.from_pretrained(c["model"], torch_dtype=dtype))
    log.info("  loading base %s", config["base_model"])
    # Passing a list to `controlnet=` makes diffusers wrap it in a
    # MultiControlNetModel, which then requires list-typed `image`/`scale`
    # at call time. Unwrap when there's only one ControlNet so `_run_pipe`'s
    # scalar-image branch matches.
    cnet_arg = cnets[0] if len(cnets) == 1 else cnets
    if xl:
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            config["base_model"], controlnet=cnet_arg, vae=vae,
            torch_dtype=dtype, use_safetensors=True,
        )
    else:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            config["base_model"], controlnet=cnet_arg, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False,
        )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device.startswith("cuda"):
        pipe.to(device)
    return pipe


def _load_sdxl_union(config: dict, device: str, dtype: torch.dtype):
    from diffusers import (
        StableDiffusionXLControlNetUnionPipeline, AutoencoderKL, UniPCMultistepScheduler,
    )
    from diffusers.models.controlnets import ControlNetUnionModel

    union_id = config["controlnet_model"]
    log.info("  loading ControlNetUnion %s", union_id)
    union = ControlNetUnionModel.from_pretrained(union_id, torch_dtype=dtype)
    log.info("  loading base %s", config["base_model"])
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)
    pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
        config["base_model"], controlnet=union, vae=vae,
        torch_dtype=dtype, use_safetensors=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if device.startswith("cuda"):
        pipe.to(device)
    return pipe


def _load_flux(config: dict, device: str, dtype: torch.dtype, union: bool):
    from diffusers import (
        FluxControlNetModel, FluxControlNetPipeline, FluxMultiControlNetModel,
    )
    if union:
        cnet_id = config["controlnet_model"]
        log.info("  loading FLUX union ControlNet %s", cnet_id)
        single = FluxControlNetModel.from_pretrained(cnet_id, torch_dtype=dtype)
        # FluxMultiControlNetModel.forward zips nets with control_image, so we must
        # replicate the *same* module reference once per control entry for the
        # pipeline to apply it at each mode.  This is O(1) memory since we hold the
        # same nn.Module reference, not a copy.
        n = len(config.get("controlnets", []))
        controlnet = FluxMultiControlNetModel([single] * max(1, n))
        n_distinct = 1
    else:
        cnets = []
        for c in config["controlnets"]:
            log.info("  loading FLUX ControlNet %s (%s)", c["type"], c["model"])
            cnets.append(FluxControlNetModel.from_pretrained(c["model"], torch_dtype=dtype))
        controlnet = FluxMultiControlNetModel(cnets)
        n_distinct = len(cnets)
    log.info("  loading FLUX base %s", config["base_model"])
    pipe = FluxControlNetPipeline.from_pretrained(
        config["base_model"], controlnet=controlnet, torch_dtype=dtype,
    )
    # FLUX.1-dev (~24 GB) + N distinct ControlNets (~6.6 GB each) + 1024² activations
    # can exceed 44 GB on L40.  Auto-enable CPU offload when the total weight budget
    # is tight — triggered when we have >1 distinct ControlNet module or the config
    # explicitly asks for it.
    want_offload = bool(config.get("cpu_offload")) or n_distinct > 1
    if device.startswith("cuda"):
        if want_offload:
            log.info("  enabling model CPU offload (n_distinct_cnets=%d)", n_distinct)
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
    return pipe


def _load_sd3(config: dict, device: str, dtype: torch.dtype):
    from diffusers import (
        StableDiffusion3ControlNetPipeline, SD3ControlNetModel,
    )
    cnets = []
    for c in config["controlnets"]:
        log.info("  loading SD3 ControlNet %s (%s)", c["type"], c["model"])
        cnets.append(SD3ControlNetModel.from_pretrained(c["model"], torch_dtype=dtype))
    log.info("  loading SD3 base %s", config["base_model"])
    from diffusers.models.controlnets.controlnet_sd3 import SD3MultiControlNetModel
    controlnet = cnets[0] if len(cnets) == 1 else SD3MultiControlNetModel(cnets)
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        config["base_model"], controlnet=controlnet, torch_dtype=dtype,
    )
    if device.startswith("cuda"):
        pipe.to(device)
    return pipe


def _run_pipe(pipe, config: dict, *, full_prompt: str, negative_prompt: str,
              cn_images: list[torch.Tensor], cn_modes: list[int | None],
              cn_scales: list[float], resolution: int, steps: int,
              guidance: float, generator: torch.Generator):
    """Dispatch to the right pipe(...) kwargs depending on variant."""
    variant = config.get("sd_variant", "sd15")
    common = dict(
        prompt=full_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
        height=resolution, width=resolution,
    )
    if variant in ("sd15", "sd21", "sdxl"):
        kw = dict(common)
        kw["negative_prompt"] = negative_prompt
        kw["image"] = cn_images if len(cn_images) > 1 else cn_images[0]
        kw["controlnet_conditioning_scale"] = (
            cn_scales if len(cn_scales) > 1 else cn_scales[0]
        )
        return pipe(**kw).images[0]
    if variant == "sdxl_union":
        kw = dict(common)
        kw["negative_prompt"] = negative_prompt
        kw["control_image"] = cn_images          # union pipe uses control_image
        kw["control_mode"] = [int(m) for m in cn_modes]
        kw["controlnet_conditioning_scale"] = cn_scales
        return pipe(**kw).images[0]
    if variant in ("flux_union", "flux_multi"):
        kw = dict(common)
        # FLUX pipelines take control_image as a list (one entry per (image, mode) pair
        # or per ControlNet in multi case).  control_mode is a list of ints (ignored for
        # single-mode models); for flux_multi where an entry has no mode, we pass -1
        # which diffusers interprets as "no union mode".
        kw["control_image"] = cn_images
        kw["control_mode"] = [(int(m) if m is not None else -1) for m in cn_modes]
        kw["controlnet_conditioning_scale"] = cn_scales
        # FLUX doesn't use CFG negative prompts the same way; skip negative_prompt.
        return pipe(**kw).images[0]
    if variant == "sd3":
        kw = dict(common)
        kw["negative_prompt"] = negative_prompt
        kw["control_image"] = cn_images if len(cn_images) > 1 else cn_images[0]
        kw["controlnet_conditioning_scale"] = (
            cn_scales if len(cn_scales) > 1 else cn_scales[0]
        )
        return pipe(**kw).images[0]
    raise ValueError(f"unknown sd_variant at run time: {variant!r}")


# ------------------------------ main loop -------------------------------------


def _save_png(tensor_bchw: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = tensor_bchw.detach().cpu().clamp(0, 1)
    if img.dim() == 4:
        img = img.squeeze(0)
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    if img.shape[0] == 4:
        img = img[:3]
    torchvision.io.write_png((img * 255).to(torch.uint8), str(path))


def _load_configs(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["configs"]


def _find_config(configs: list[dict], name: str) -> dict:
    for c in configs:
        if c["name"] == name:
            return c
    raise KeyError(f"config '{name}' not found; available: {[c['name'] for c in configs]}")


def _parse_strength_overrides(s: str | None) -> dict[str, float]:
    if not s:
        return {}
    out = {}
    for tok in s.split(","):
        if not tok.strip():
            continue
        k, v = tok.split("=")
        out[k.strip()] = float(v)
    return out


def _parse_views(s: str) -> list[tuple[float, float]]:
    """Parse ``"0.0,0.3;1.57,0.3"`` (azi,elev pairs) into [(azi, elev), ...]."""
    views = []
    for pair in s.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        a, e = pair.split(",")
        views.append((float(a), float(e)))
    return views


def _read_orientation(mesh_dir: Path) -> dict:
    """Return ``{"yaw": .., "pitch": .., "roll": ..}`` (radians).

    Back-compat: older orientation.json files with only ``yaw`` work;
    missing axes default to 0.
    """
    p = mesh_dir / "single_view" / "orientation.json"
    out = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    if not p.exists():
        return out
    try:
        data = json.loads(p.read_text())
        for k in ("yaw", "pitch", "roll"):
            if k in data:
                out[k] = float(data[k])
        return out
    except Exception as exc:
        log.warning("orientation.json unreadable (%s); defaulting to zero", exc)
        return out


def load_mesh_for_gen(mesh_root: Path, mesh_name: str, device: str):
    """Load + orient + center+normalize a mesh.  Returns (mesh, orient_dict)."""
    mesh_dir = mesh_root / mesh_name
    gltf = mesh_dir / "scene.gltf"
    if not gltf.is_file():
        raise FileNotFoundError(f"mesh not found: {gltf}")
    mesh = kaolin.io.import_mesh(str(gltf), triangulate=True).to(device)
    _force_materials_to_device(mesh, device)
    orient = _read_orientation(mesh_dir)
    if any(abs(orient[k]) > 1e-7 for k in ("yaw", "pitch", "roll")):
        _apply_euler(mesh, orient["pitch"], orient["yaw"], orient["roll"])
    mesh.vertices = kaolin.ops.pointcloud.center_points(
        mesh.vertices.unsqueeze(0), normalize=True,
    ).squeeze(0)
    return mesh, orient


def generate_for_mesh(
    *, pipe, config: dict, device: str, mesh_root: Path, mesh_name: str,
    prompt: str, views: list[tuple[float, float]],
    strengths: dict[str, float] | None = None,
    viewdist: float = 2.8, fov: float = 0.55,
    resolution: int | None = None,
    steps: int | None = None, guidance: float | None = None,
    num_per_view: int = 1, seed: int = 0,
    run_id: str | None = None,
    camera_suffix: bool = False,
    prompts_per_view: list[str] | None = None,
) -> Path:
    """Run one (mesh, config) generation pass with *pipe* already loaded.

    Returns the run directory.  Used by the CLI and by run_experiment.py.
    """
    mesh_dir = mesh_root / mesh_name
    variant = config.get("sd_variant", "sd15")
    default_res = 1024 if variant in ("sdxl", "sdxl_union", "flux_union", "flux_multi", "sd3") else 512
    resolution = resolution or config.get("resolution", default_res)
    steps = steps or config.get("steps", 25)
    guidance = guidance if guidance is not None else config.get("guidance", 7.5)
    prompt_suffix = (config.get("prompt_suffix") or "").strip()
    negative_prompt = config.get("negative_prompt", "")
    cfg_strengths = {c["type"]: float(c["strength"]) for c in config["controlnets"]}
    if strengths:
        cfg_strengths.update(strengths)

    log.info("Loading mesh %s", mesh_dir)
    mesh, orient = load_mesh_for_gen(mesh_root, mesh_name, device)

    run_id = run_id or time.strftime("%Y%m%dT%H%M%S")
    out_dir = mesh_dir / "single_view" / config["name"] / f"run-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if prompts_per_view is not None and len(prompts_per_view) != len(views):
        raise ValueError(
            f"prompts_per_view length {len(prompts_per_view)} != views length {len(views)}"
        )

    per_view_records = []
    for v_idx, (azi, elev) in enumerate(views):
        # Per-view prompt assembly: base + optional camera-suffix + config suffix.
        base = (prompts_per_view[v_idx] if prompts_per_view else prompt).strip()
        view_label = camera_phrase(azi, elev)
        if camera_suffix:
            base = f"{base}, {view_label} view"
        full_prompt = f"{base}, {prompt_suffix}" if prompt_suffix else base

        log.info("[%s:%s] [%d/%d] azi=%.3f elev=%.3f (%s)",
                 mesh_name, config["name"], v_idx + 1, len(views), azi, elev, view_label)
        conds = _render_conditions(mesh, azi, elev, viewdist, fov, resolution, device)

        cn_images, cn_modes, cn_scales = [], [], []
        for c in config["controlnets"]:
            pass_name = COND_TYPE_TO_PASS[c["type"]]
            cond = conds[pass_name].to(device).to(torch.float32)
            cn_images.append(cond)
            cn_modes.append(c.get("control_mode"))
            cn_scales.append(float(cfg_strengths.get(c["type"], c["strength"])))

        vdir = out_dir / f"view{v_idx:02d}"
        vdir.mkdir(parents=True, exist_ok=True)
        for c in config["controlnets"]:
            _save_png(conds[COND_TYPE_TO_PASS[c["type"]]], vdir / f"cond_{c['type']}.png")
        _save_png(conds["render"], vdir / "cond_render.png")

        samples = []
        for s in range(num_per_view):
            gen_seed = seed + v_idx * 1000 + s
            gen = torch.Generator(device=device).manual_seed(gen_seed)
            with torch.inference_mode():
                img = _run_pipe(
                    pipe, config,
                    full_prompt=full_prompt, negative_prompt=negative_prompt,
                    cn_images=cn_images, cn_modes=cn_modes, cn_scales=cn_scales,
                    resolution=resolution, steps=steps, guidance=guidance,
                    generator=gen,
                )
            fname = f"s{s:02d}_seed{gen_seed}.png"
            img.save(vdir / fname)
            samples.append({"file": fname, "seed": gen_seed, "prompt": full_prompt})

        per_view_records.append({
            "view_idx": v_idx,
            "azi": azi,
            "elev": elev,
            "viewdist": viewdist,
            "fov": fov,
            "camera_label": view_label,
            "samples": samples,
        })

    meta = {
        "mesh": mesh_name,
        "config": config["name"],
        "run_id": run_id,
        "prompt": prompt,
        "prompts_per_view": prompts_per_view,
        "camera_suffix": bool(camera_suffix),
        "negative_prompt": negative_prompt,
        "strengths": cfg_strengths,
        "steps": steps,
        "guidance": guidance,
        "resolution": resolution,
        "viewdist": viewdist,
        "fov": fov,
        "orientation": orient,
        "views": per_view_records,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Wrote %s", out_dir)
    # Free mesh memory before next one.
    del mesh
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mesh-root", type=Path, required=True)
    mg = ap.add_mutually_exclusive_group(required=True)
    mg.add_argument("--mesh", type=str, help="Single mesh folder name under --mesh-root.")
    mg.add_argument("--meshes", type=str, nargs="+",
                    help="Multiple mesh folder names; pipeline is loaded once and reused.")
    ap.add_argument("--configs-file", type=Path,
                    default=Path(__file__).resolve().parents[3] / "configs" / "single_view_configs.yaml")
    ap.add_argument("--config", type=str, required=True,
                    help="Name of a config block in --configs-file")
    ap.add_argument("--prompt", type=str, required=True,
                    help="Prompt string.  Applies to every mesh unless --prompts-file is set.")
    ap.add_argument("--prompts-file", type=Path, default=None,
                    help="JSON mapping mesh_name → prompt; overrides --prompt per mesh.")
    ap.add_argument("--views", type=str, required=True,
                    help="Semicolon-separated azi,elev radians pairs, e.g. '0.0,0.3;1.57,0.3'")
    ap.add_argument("--camera-suffix", action="store_true",
                    help="Append an auto-generated ``<upper|lower> <left|right> <front|back> view`` phrase to the prompt per view.")
    ap.add_argument("--viewdist", type=float, default=2.8)
    ap.add_argument("--fov", type=float, default=0.55)
    ap.add_argument("--resolution", type=int, default=None,
                    help="Override config.resolution")
    ap.add_argument("--strengths", type=str, default=None,
                    help="Comma-separated overrides like 'normal=0.7,depth=0.5'")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-per-view", type=int, default=1,
                    help="Samples drawn per viewpoint (different seeds).")
    ap.add_argument("--run-id", type=str, default=None,
                    help="Custom run id; defaults to timestamp.")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = args.device or _pick_free_cuda_device()
    if device.startswith("cuda"):
        idx = int(device.split(":", 1)[1]) if ":" in device else 0
        torch.cuda.set_device(idx)
    log.info("Device: %s", device)

    configs = _load_configs(args.configs_file)
    config = _find_config(configs, args.config)
    strength_overrides = _parse_strength_overrides(args.strengths)
    views = _parse_views(args.views)

    mesh_list = [args.mesh] if args.mesh else args.meshes
    prompt_map: dict[str, str] = {}
    if args.prompts_file is not None:
        prompt_map = json.loads(args.prompts_file.read_text())

    log.info("Loading diffusion pipeline (%s) — shared across %d mesh(es)",
             config["name"], len(mesh_list))
    pipe = _load_pipeline(config, device)

    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%S")
    for mesh_name in mesh_list:
        try:
            generate_for_mesh(
                pipe=pipe, config=config, device=device,
                mesh_root=args.mesh_root, mesh_name=mesh_name,
                prompt=prompt_map.get(mesh_name, args.prompt),
                views=views, strengths=strength_overrides,
                viewdist=args.viewdist, fov=args.fov,
                resolution=args.resolution, steps=args.steps, guidance=args.guidance,
                num_per_view=args.num_per_view, seed=args.seed,
                run_id=run_id, camera_suffix=args.camera_suffix,
            )
        except Exception as exc:
            log.exception("[%s] FAILED: %s", mesh_name, exc)
            continue


if __name__ == "__main__":
    main()
