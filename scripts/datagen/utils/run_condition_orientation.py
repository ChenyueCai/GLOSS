#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render single-view geometry conditions driven by ``orientation.json``.

For a given mesh this script renders two independent sets of views:

  * ``origin``   — camera orbits the mesh origin. ``viewdist`` and
                   ``[fov_min, fov_max]`` taken from the top-level fields of
                   ``orientation.json``. Random azi/elev within wide defaults.
  * ``anchors``  — camera orbits each anchor in ``orientation.json:anchors``.
                   Per-anchor ``viewdist`` and ``[fov_min, fov_max]`` are
                   used. The ``--num_views`` budget is split equally across
                   anchors (``floor(num_views / N)`` each); any remainder is
                   appended to the last anchor.

Both modes write the same channel layout as ``generate_condition.py``::

    <output_dir>/condition_output/{normal, geonormal, canny-normal,
                                   canny-geonormal, depth, mask}/<chan>NNNN.png
    <output_dir>/meta/viewNNNN.yml      # camera + (anchor_id for anchor mode)

Anchor positions in ``orientation.json`` live in the pre-rotation, raw-
centered mesh frame. We transform them through the same Euler rotation +
bbox normalization the renderer applies (see
``run_anchor_views._load_mesh_with_anchor_frame``).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torchvision

UTILS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(UTILS_DIR))
import single_view_gen as svg  # noqa: E402
from run_anchor_views import _load_mesh_with_anchor_frame  # noqa: E402

import kaolin.render.easy_render as easy_render  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from gloss.utils.diffusion_render import get_canny, get_depth  # noqa: E402
from gloss.utils.kaolin_utils import camera_to_meta  # noqa: E402
from gloss.utils.render import render_all_features  # noqa: E402
from gloss.utils.single_view import make_single_view_cam  # noqa: E402

log = logging.getLogger(__name__)

CHANNEL_DIRS = (
    "normal", "geonormal", "canny-normal", "canny-geonormal", "depth", "mask",
)
RENDER_PASSES = ["camera_normals", "geo_camera_normals", "raw_depth", "mask"]

# Default azi/elev ranges (full orbit, mostly upper hemisphere). Match
# generate_condition.py defaults.
AZI_RANGE = [0.0, 2 * math.pi]
ELEV_RANGE = [-1.0, 1.7]
VIEWDIST_JITTER = 0.05  # +/- 5% around the orientation.json viewdist


def _ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    cond_dir = output_dir / "condition_output"
    meta_dir = output_dir / "meta"
    for sub in CHANNEL_DIRS:
        (cond_dir / sub).mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    return cond_dir, meta_dir


def _save_view(
    cam,
    feats: dict,
    *,
    view_idx: int,
    cond_dir: Path,
    meta_dir: Path,
    anchor_id: Optional[str] = None,
) -> None:
    """Save one view's six channels and the meta yaml. Mirrors
    ``generate_condition.py:160-198`` exactly."""
    cam_normal = feats["camera_normals"]
    geo_cam_normal = feats["geo_camera_normals"]
    raw_depth = feats["raw_depth"]
    mask = feats["mask"]

    canny_normal = get_canny(cam_normal)
    canny_geonormal = get_canny(geo_cam_normal)
    depth = get_depth(raw_depth.clone())  # get_depth mutates input

    s = lambda t, p: torchvision.utils.save_image(t, str(p))
    s(cam_normal.permute(0, 3, 1, 2) / 2 + 0.5,
      cond_dir / "normal" / f"normal{view_idx:04d}.png")
    s(geo_cam_normal.permute(0, 3, 1, 2) / 2 + 0.5,
      cond_dir / "geonormal" / f"geonormal{view_idx:04d}.png")
    s(depth, cond_dir / "depth" / f"depth{view_idx:04d}.png")
    s(canny_normal,
      cond_dir / "canny-normal" / f"canny{view_idx:04d}.png")
    s(canny_geonormal,
      cond_dir / "canny-geonormal" / f"canny{view_idx:04d}.png")
    s(mask.permute(0, 3, 1, 2),
      cond_dir / "mask" / f"mask{view_idx:04d}.png")

    meta = {"camera": camera_to_meta(cam)}
    if anchor_id is not None:
        meta["anchor_id"] = anchor_id
    OmegaConf.save(config=OmegaConf.create(meta), f=meta_dir / f"view{view_idx:04d}.yml")


def _render_one(
    mesh,
    *,
    at: Optional[torch.Tensor],
    azi_range: List[float],
    elev_range: List[float],
    viewdist_range: List[float],
    fov_range: List[float],
    resolution: int,
    device: str,
    lighting,
):
    cam = make_single_view_cam(
        azi_range=azi_range,
        elev_range=elev_range,
        view_dist_range=viewdist_range,
        fov_range=fov_range,
        resolution=resolution,
        device=device,
        at=(at.detach().cpu() if at is not None else None),
    )
    feats = render_all_features(cam, mesh,
                                lighting=lighting,
                                required_passes=RENDER_PASSES)
    return cam, feats


def render_origin(
    mesh, *, num_views: int, viewdist: float, fov_min: float, fov_max: float,
    resolution: int, device: str, lighting, output_dir: Path,
) -> int:
    cond_dir, meta_dir = _ensure_dirs(output_dir)
    vd_lo = viewdist * (1.0 - VIEWDIST_JITTER)
    vd_hi = viewdist * (1.0 + VIEWDIST_JITTER)
    log.info("[origin] %d views | viewdist=[%.3f,%.3f] fov=[%.3f,%.3f] -> %s",
             num_views, vd_lo, vd_hi, fov_min, fov_max, output_dir)
    written = 0
    for i in range(num_views):
        view_yml = meta_dir / f"view{i:04d}.yml"
        if view_yml.exists() and (cond_dir / "normal" / f"normal{i:04d}.png").exists():
            continue
        cam, feats = _render_one(
            mesh, at=None,
            azi_range=AZI_RANGE, elev_range=ELEV_RANGE,
            viewdist_range=[vd_lo, vd_hi],
            fov_range=[fov_min, fov_max],
            resolution=resolution, device=device, lighting=lighting,
        )
        _save_view(cam, feats, view_idx=i,
                   cond_dir=cond_dir, meta_dir=meta_dir, anchor_id=None)
        written += 1
        if (i + 1) % 25 == 0 or i == num_views - 1:
            log.info("  origin progress %d/%d", i + 1, num_views)
    return written


def _split_budget(num_views: int, n_anchors: int) -> List[int]:
    """floor(num_views/N) per anchor with the remainder dumped on the last anchor."""
    base = num_views // n_anchors
    rem = num_views - base * n_anchors
    counts = [base] * n_anchors
    counts[-1] += rem
    return counts


def render_anchors(
    mesh, *, anchors_raw: list, anchor_to_norm, num_views: int,
    resolution: int, device: str, lighting, output_dir: Path,
) -> int:
    cond_dir, meta_dir = _ensure_dirs(output_dir)
    counts = _split_budget(num_views, len(anchors_raw))
    log.info("[anchors] %d total | per-anchor split=%s -> %s",
             num_views, counts, output_dir)
    view_idx = 0
    written = 0
    for anchor_idx, (anchor, count) in enumerate(zip(anchors_raw, counts)):
        anchor_id = anchor.get("id") or anchor.get("name") or f"a{anchor_idx+1}"
        at_xyz = anchor_to_norm(anchor["position"])
        viewdist = float(anchor["viewdist"])
        vd_lo = viewdist * (1.0 - VIEWDIST_JITTER)
        vd_hi = viewdist * (1.0 + VIEWDIST_JITTER)
        fov_min = float(anchor["fov_min"])
        fov_max = float(anchor["fov_max"])
        log.info("  anchor[%d] id=%s pos=%s viewdist=%.3f fov=[%.3f,%.3f] count=%d",
                 anchor_idx, anchor_id,
                 [round(float(x), 3) for x in anchor["position"]],
                 viewdist, fov_min, fov_max, count)
        for j in range(count):
            view_yml = meta_dir / f"view{view_idx:04d}.yml"
            if view_yml.exists() and (cond_dir / "normal" / f"normal{view_idx:04d}.png").exists():
                view_idx += 1
                continue
            cam, feats = _render_one(
                mesh, at=at_xyz,
                azi_range=AZI_RANGE, elev_range=ELEV_RANGE,
                viewdist_range=[vd_lo, vd_hi],
                fov_range=[fov_min, fov_max],
                resolution=resolution, device=device, lighting=lighting,
            )
            _save_view(cam, feats, view_idx=view_idx,
                       cond_dir=cond_dir, meta_dir=meta_dir,
                       anchor_id=anchor_id)
            written += 1
            view_idx += 1
            if (j + 1) % 25 == 0 or j == count - 1:
                log.info("    anchor[%d] progress %d/%d (global view %d)",
                         anchor_idx, j + 1, count, view_idx)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh-root", type=Path, required=True,
                    help="Directory containing per-mesh subdirs (e.g. "
                         ".../mesh/test_mesh/automatic).")
    ap.add_argument("--mesh", type=str, required=True,
                    help="Mesh subdir name under --mesh-root.")
    ap.add_argument("--num_views", type=int, default=500,
                    help="Number of views per mode (origin and anchors).")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--device", type=str, default=None,
                    help="cuda:N (default: pick a free GPU).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Torch RNG seed for camera sampling.")
    ap.add_argument("--modes", type=str, nargs="+",
                    default=["origin", "anchors"],
                    choices=["origin", "anchors"],
                    help="Which mode(s) to run.")
    ap.add_argument("--origin_subdir", type=str, default="origin")
    ap.add_argument("--anchors_subdir", type=str, default="anchors")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip mode if its output_dir/condition_output/normal/ "
                         "already has >= num_views pngs.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s",
                        datefmt="%H:%M:%S")

    device = args.device or svg._pick_free_cuda_device()
    if device.startswith("cuda") and ":" in device:
        torch.cuda.set_device(int(device.split(":", 1)[1]))
    log.info("device=%s mesh=%s", device, args.mesh)

    torch.manual_seed(args.seed)

    mesh_dir = args.mesh_root / args.mesh
    orient_path = mesh_dir / "single_view" / "orientation.json"
    if not orient_path.is_file():
        sys.exit(f"missing orientation.json: {orient_path}")
    orient_raw = json.loads(orient_path.read_text("utf-8"))

    log.info("loading mesh + applying yaw/pitch/roll + computing anchor frame")
    mesh, _orient, anchor_to_norm = _load_mesh_with_anchor_frame(
        args.mesh_root, args.mesh, device,
    )
    lighting = easy_render.default_lighting().to(device)

    t_start = time.time()

    # ---- ORIGIN ------------------------------------------------------------
    if "origin" in args.modes:
        out_origin = mesh_dir / "single_view" / args.origin_subdir
        if args.skip_existing and len(list(
                (out_origin / "condition_output" / "normal").glob("*.png"))) >= args.num_views:
            log.info("[origin] skip_existing -> already %d+ views in %s",
                     args.num_views, out_origin)
        else:
            viewdist = float(orient_raw.get("viewdist", 1.0))
            fov_min = float(orient_raw.get("fov_min", 0.4))
            fov_max = float(orient_raw.get("fov_max", 0.8))
            n_written = render_origin(
                mesh, num_views=args.num_views,
                viewdist=viewdist, fov_min=fov_min, fov_max=fov_max,
                resolution=args.resolution, device=device, lighting=lighting,
                output_dir=out_origin,
            )
            log.info("[origin] wrote %d new views (existing skipped)", n_written)

    # ---- ANCHORS -----------------------------------------------------------
    if "anchors" in args.modes:
        anchors_raw = orient_raw.get("anchors") or []
        if not anchors_raw:
            log.info("[anchors] no anchors in orientation.json -> skipping")
        else:
            out_anchors = mesh_dir / "single_view" / args.anchors_subdir
            if args.skip_existing and len(list(
                    (out_anchors / "condition_output" / "normal").glob("*.png"))) >= args.num_views:
                log.info("[anchors] skip_existing -> already %d+ views in %s",
                         args.num_views, out_anchors)
            else:
                n_written = render_anchors(
                    mesh, anchors_raw=anchors_raw, anchor_to_norm=anchor_to_norm,
                    num_views=args.num_views, resolution=args.resolution,
                    device=device, lighting=lighting, output_dir=out_anchors,
                )
                log.info("[anchors] wrote %d new views (existing skipped)", n_written)

    log.info("done in %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
