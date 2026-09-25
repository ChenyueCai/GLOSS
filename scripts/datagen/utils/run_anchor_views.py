#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Single-view generation driven by anchors saved in ``orientation.json``.

Each anchor has ``position`` (in the normGroup-local, i.e. centered-but-
unscaled mesh frame), plus ``viewdist``, ``fov``, ``fov_min``, ``fov_max``.
This driver sweeps fov across ``[fov_min, fov_max]`` (configurable number of
samples), places the camera at ``viewdist`` along +Z from the anchor, and
writes images into the standard
``<mesh_root>/<mesh>/single_view/<config>/run-<tag>-<ts>/viewNN/`` layout so
the existing /sv and /compare UIs pick them up automatically.

Anchors are addressed by 0-based index; each anchor gets its own prompt on
the CLI.  Example::

    python run_anchor_views.py --mesh-root .../test_mesh/automatic --mesh gecko_1 \
      --config sd15_rv_normal_depth_canny --tag anchors-exp01 \
      --normal-strength 0.7 --depth-strength 0.7 --canny-strength 0.5 \
      --prompt "close-up of a gecko's head" \
      --prompt "close-up of a gecko's lower body" \
      --fov-samples 3 --device cuda:0
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import single_view_gen as svg  # noqa: E402

import kaolin  # noqa: E402
import kaolin.render.easy_render as easy_render  # noqa: E402

from gloss.utils.render import render_all_features  # noqa: E402
from gloss.utils.single_view import make_single_view_cam  # noqa: E402

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")

# Orbit around the anchor along the +Z axis (three.js "front" after
# yaw/pitch/roll is applied to the mesh). kaolin's azi/elev gives direction
# (cos(elev)cos(azi), sin(elev), cos(elev)sin(azi)), so azi=π/2, elev=0
# places the eye at (0, 0, r) relative to the look-at point.
DEFAULT_AZI = math.pi / 2
DEFAULT_ELEV = 0.0


def _load_mesh_with_anchor_frame(mesh_root: Path, mesh_name: str, device: str):
    """Load mesh like ``svg.load_mesh_for_gen`` while recording the pre-rotation
    bbox so anchor positions (stored in the raw, centered frame) can be mapped
    into the final normalized mesh frame.

    Returns (mesh, orient, anchor_transform_fn):
        anchor_transform_fn(p_json_xyz) -> 3-tensor in normalized-mesh frame
    """
    mesh_dir = mesh_root / mesh_name
    gltf = mesh_dir / "scene.gltf"
    if not gltf.is_file():
        raise FileNotFoundError(f"mesh not found: {gltf}")
    mesh = kaolin.io.import_mesh(str(gltf), triangulate=True).to(device)
    svg._force_materials_to_device(mesh, device)
    orient = svg._read_orientation(mesh_dir)

    # Pre-rotation bbox (matches the bbox the UI used when saving anchor pos).
    vmin_raw = mesh.vertices.amin(dim=0)
    vmax_raw = mesh.vertices.amax(dim=0)
    center_raw = (vmin_raw + vmax_raw) * 0.5

    if any(abs(orient[k]) > 1e-7 for k in ("yaw", "pitch", "roll")):
        svg._apply_euler(mesh, orient["pitch"], orient["yaw"], orient["roll"])

    # Post-rotation bbox → used by the final normalize step, and what we need
    # to express the anchor in the same normalized frame the renderer sees.
    vmin_rot = mesh.vertices.amin(dim=0)
    vmax_rot = mesh.vertices.amax(dim=0)
    center_rot = (vmin_rot + vmax_rot) * 0.5
    max_dim_rot = (vmax_rot - vmin_rot).amax()

    # Normalize vertices → this is what kaolin.ops.pointcloud.center_points does.
    mesh.vertices = (mesh.vertices - center_rot) / max_dim_rot

    # Build rotation R to map anchor (in raw-centered frame) through the same
    # rotation the mesh vertices went through.
    cx, sx = math.cos(orient["pitch"]), math.sin(orient["pitch"])
    cy, sy = math.cos(orient["yaw"]),   math.sin(orient["yaw"])
    cz, sz = math.cos(orient["roll"]),  math.sin(orient["roll"])
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], device=device, dtype=torch.float32)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], device=device, dtype=torch.float32)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], device=device, dtype=torch.float32)
    R = Rx @ Ry @ Rz

    def anchor_to_norm(p_json):
        # anchor.position stored in JS as (V_clicked - center_raw); recover
        # V_clicked, rotate, re-center + normalize.
        p = torch.as_tensor(p_json, dtype=torch.float32, device=device)
        v_clicked = p + center_raw
        v_rot = (R @ v_clicked.unsqueeze(-1)).squeeze(-1)
        return (v_rot - center_rot) / max_dim_rot

    return mesh, orient, anchor_to_norm


def _render_conditions_at(mesh, at_xyz: torch.Tensor, azi: float, elev: float,
                          viewdist: float, fov: float, resolution: int,
                          device: str) -> dict[str, torch.Tensor]:
    """Same passes as svg._render_conditions but with a custom look-at target."""
    cam = make_single_view_cam(
        azi_range=[azi, azi], elev_range=[elev, elev],
        view_dist_range=[viewdist, viewdist], fov_range=[fov, fov],
        resolution=resolution, device=device,
        at=at_xyz.detach().cpu(),
    )
    lighting = easy_render.default_lighting().to(device)
    passes = ["render", "camera_normals", "geo_camera_normals", "raw_depth", "mask"]
    res = render_all_features(cam, mesh, lighting=lighting, required_passes=passes)

    normal01 = svg._normal_to_01(res["camera_normals"])
    geonormal01 = svg._normal_to_01(res["geo_camera_normals"])
    render01 = svg._render_to_01(res["render"])
    depth01 = svg._depth_from_raw(res["raw_depth"])
    canny = svg._canny_from_rgb(res["camera_normals"].add(1).mul_(0.5))
    canny_geo = svg._canny_from_rgb(res["geo_camera_normals"].add(1).mul_(0.5))
    canny_render = svg._canny_from_rgb(res["render"].add(1).mul_(0.5))
    mask01 = ((res["mask"] + 1) * 0.5).clamp(0, 1).permute(0, 3, 1, 2)
    if mask01.shape[1] == 1:
        mask01 = mask01.repeat(1, 3, 1, 1)

    return {
        "normal": normal01, "geonormal": geonormal01, "render": render01,
        "depth": depth01, "canny": canny, "canny_geo": canny_geo,
        "canny_render": canny_render, "mask": mask01,
    }


def _fov_samples(fov_min: float, fov_max: float, n: int, fallback: float) -> list[float]:
    if fov_min is None or fov_max is None or n <= 1:
        return [fallback]
    if fov_max <= fov_min or n == 1:
        return [fallback or fov_min or fov_max]
    step = (fov_max - fov_min) / (n - 1)
    return [fov_min + i * step for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh-root", type=Path, required=True)
    ap.add_argument("--mesh", type=str, required=True)
    ap.add_argument("--configs-file", type=Path,
                    default=Path(__file__).resolve().parents[3] / "configs" / "single_view_configs.yaml")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--tag", type=str, default="anchors")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-per-view", type=int, default=1)
    ap.add_argument("--fov-samples", type=int, default=3,
                    help="Samples across [fov_min, fov_max] per anchor. "
                         "Set to 1 to use each anchor's saved fov.")
    ap.add_argument("--azi", type=float, default=DEFAULT_AZI,
                    help="Orbital azimuth in radians (default π/2 = +Z front).")
    ap.add_argument("--elev", type=float, default=DEFAULT_ELEV)
    ap.add_argument("--prompt", action="append", default=[],
                    help="Prompt for the Nth anchor; pass once per anchor in order.")
    ap.add_argument("--anchors", type=int, nargs="*", default=None,
                    help="0-based anchor indices to use (default: all).")
    ap.add_argument("--normal-strength", type=float, default=None)
    ap.add_argument("--depth-strength", type=float, default=None)
    ap.add_argument("--canny-strength", type=float, default=None)
    args = ap.parse_args()

    device = args.device or svg._pick_free_cuda_device()
    if device.startswith("cuda"):
        torch.cuda.set_device(int(device.split(":", 1)[1]) if ":" in device else 0)
    log.info("device=%s", device)

    configs = svg._load_configs(args.configs_file)
    by_name = {c["name"]: c for c in configs}
    if args.config not in by_name:
        sys.exit(f"config '{args.config}' not in {args.configs_file}")
    config = by_name[args.config]

    cfg_strengths = {c["type"]: float(c["strength"]) for c in config["controlnets"]}
    if args.normal_strength is not None and "normal" in cfg_strengths:
        cfg_strengths["normal"] = args.normal_strength
    if args.depth_strength is not None and "depth" in cfg_strengths:
        cfg_strengths["depth"] = args.depth_strength
    if args.canny_strength is not None and "canny" in cfg_strengths:
        cfg_strengths["canny"] = args.canny_strength
    log.info("strengths=%s", cfg_strengths)

    mesh_dir = args.mesh_root / args.mesh
    orient_path = mesh_dir / "single_view" / "orientation.json"
    if not orient_path.is_file():
        sys.exit(f"missing {orient_path}")
    orient_raw = json.loads(orient_path.read_text("utf-8"))
    anchors = orient_raw.get("anchors") or []
    if not anchors:
        sys.exit(f"no anchors saved in {orient_path}")

    idxs = args.anchors if args.anchors is not None else list(range(len(anchors)))
    sel_anchors = [anchors[i] for i in idxs if 0 <= i < len(anchors)]
    if not sel_anchors:
        sys.exit("anchor selection yielded nothing")

    prompts = args.prompt
    if len(prompts) < len(sel_anchors):
        sys.exit(f"got {len(prompts)} prompts but {len(sel_anchors)} anchors selected")

    mesh, orient, anchor_to_norm = _load_mesh_with_anchor_frame(
        args.mesh_root, args.mesh, device)
    log.info("mesh loaded, orient=%s", orient)

    variant = config.get("sd_variant", "sd15")
    default_res = 1024 if variant in ("sdxl", "sdxl_union", "flux_union", "flux_multi", "sd3") else 512
    resolution = config.get("resolution", default_res)
    steps = config.get("steps", 25)
    guidance = config.get("guidance", 7.5)
    prompt_suffix = (config.get("prompt_suffix") or "").strip()
    negative_prompt = config.get("negative_prompt", "")

    pipe = svg._load_pipeline(config, device)

    run_id = f"{args.tag}-{time.strftime('%Y%m%dT%H%M%S')}"
    out_dir = mesh_dir / "single_view" / config["name"] / f"run-{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_view_records = []
    view_counter = 0
    for local_idx, (anchor, base_prompt) in enumerate(zip(sel_anchors, prompts)):
        anchor_norm = anchor_to_norm(anchor["position"])
        fov_min = float(anchor.get("fov_min") or anchor.get("fov") or 0.55)
        fov_max = float(anchor.get("fov_max") or anchor.get("fov") or 0.55)
        fov_list = _fov_samples(fov_min, fov_max, args.fov_samples,
                                float(anchor.get("fov") or (fov_min + fov_max) / 2))
        viewdist = float(anchor.get("viewdist") or 0.8)
        log.info("[anchor %d] %s @ norm=%s  viewdist=%.3f  fov=%s",
                 local_idx, anchor.get("name", f"anchor_{local_idx}"),
                 [round(x, 4) for x in anchor_norm.tolist()], viewdist,
                 [round(f, 4) for f in fov_list])

        for fov in fov_list:
            full_prompt = f"{base_prompt.strip()}, {prompt_suffix}" if prompt_suffix else base_prompt.strip()
            log.info("  view%02d  fov=%.3f  prompt=%s",
                     view_counter, fov, full_prompt[:140])
            conds = _render_conditions_at(
                mesh, anchor_norm, args.azi, args.elev,
                viewdist=viewdist, fov=fov, resolution=resolution, device=device,
            )

            cn_images, cn_modes, cn_scales = [], [], []
            for c in config["controlnets"]:
                pass_name = svg.COND_TYPE_TO_PASS[c["type"]]
                cn_images.append(conds[pass_name].to(device).to(torch.float32))
                cn_modes.append(c.get("control_mode"))
                cn_scales.append(float(cfg_strengths.get(c["type"], c["strength"])))

            vdir = out_dir / f"view{view_counter:02d}"
            vdir.mkdir(parents=True, exist_ok=True)
            for c in config["controlnets"]:
                svg._save_png(conds[svg.COND_TYPE_TO_PASS[c["type"]]],
                              vdir / f"cond_{c['type']}.png")
            svg._save_png(conds["render"], vdir / "cond_render.png")

            samples = []
            for s in range(args.num_per_view):
                gen_seed = args.seed + view_counter * 1000 + s
                gen = torch.Generator(device=device).manual_seed(gen_seed)
                with torch.inference_mode():
                    img = svg._run_pipe(
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
                "view_idx": view_counter,
                "anchor_idx": idxs[local_idx],
                "anchor_name": anchor.get("name"),
                "anchor_position": anchor["position"],
                "anchor_position_norm": [float(x) for x in anchor_norm.tolist()],
                "azi": args.azi, "elev": args.elev,
                "viewdist": viewdist, "fov": fov,
                "prompt": full_prompt,
                "samples": samples,
            })
            view_counter += 1

    meta = {
        "mesh": args.mesh,
        "config": config["name"],
        "run_id": run_id,
        "tag": args.tag,
        "anchors_source": str(orient_path),
        "anchors_selected": idxs,
        "strengths": cfg_strengths,
        "steps": steps, "guidance": guidance, "resolution": resolution,
        "orientation": orient,
        "views": per_view_records,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Wrote %s", out_dir)

    del pipe
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
