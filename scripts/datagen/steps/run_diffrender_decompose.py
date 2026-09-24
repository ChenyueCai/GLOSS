#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Decompose a flat directory of view images using diffusion-renderer's inverse
rendering model. Each image is treated as the first frame of a video; the model
fills the remaining frames by repeating the last frame (chunk_mode=first).

Input layout (flat; any mix of .png / .jpg):
    {input_dir}/viewNNNN.png
    {input_dir}/viewNNNN.png
    ...

Output layout:
    {output_dir}/viewNNNN.basecolor.png
    {output_dir}/viewNNNN.normal.png
    {output_dir}/viewNNNN.roughness.png
    {output_dir}/viewNNNN.metallic.png
    {output_dir}/viewNNNN.depth.png
    ...

Run this script from inside the diff-render conda environment:
    conda run -n diff-render python run_diffrender_decompose.py \
        --input_dir <gen_view_masked dir> \
        --output_dir <gen_view_decomposite dir> \
        --diffrender_dir <path/to/diffusion-renderer>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
VIEW_INDEX_RE = re.compile(r"view(\d+)")


def _excluded_indices(meta_json: Path | None) -> set[int]:
    """Return the set of view indices flagged as excluded in meta.json
    (key: ``exclude_from_training_indices``). Empty set if meta.json is
    None or has no key."""
    if meta_json is None:
        return set()
    p = Path(meta_json)
    if not p.is_file():
        return set()
    with p.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return set(int(i) for i in meta.get("exclude_from_training_indices") or [])


def _view_index(path: Path) -> int | None:
    """Pull NNNN out of viewNNNN[.<channel>].png. Returns None if no match."""
    m = VIEW_INDEX_RE.search(path.stem)
    return int(m.group(1)) if m else None
DEFAULT_PASSES = ["basecolor", "metallic", "roughness", "normal", "depth"]
DEFAULT_FALLBACK_INFERENCE_RES = ["384,384", "256,256"]


def parse_inference_res(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(
            f"Invalid inference resolution {value!r}; expected 'height,width'."
        )
    return int(parts[0]), int(parts[1])


def build_inference_cmd(
    diffrender_dir: Path,
    staging: Path,
    raw_out: Path,
    inference_res: str,
    inference_n_steps: int,
    model_passes,
) -> list[str]:
    h, w = parse_inference_res(inference_res)
    passes_cfg = "[" + ",".join(f"'{p}'" for p in model_passes) + "]"
    return [
        sys.executable,
        str(diffrender_dir / "inference_svd_rgbx.py"),
        "--config",
        str(diffrender_dir / "configs" / "rgbx_inference.yaml"),
        f"inference_input_dir={staging}",
        f"inference_save_dir={raw_out}",
        f"inference_res=[{h},{w}]",
        f"inference_n_steps={inference_n_steps}",
        "chunk_mode=first",
        "overlap_n_frames=0",
        "save_image=true",
        "save_video=false",
        f"model_passes={passes_cfg}",
    ]


def looks_like_cuda_oom(output: str) -> bool:
    lowered = output.lower()
    oom_markers = (
        "cuda out of memory",
        "torch.outofmemoryerror",
        "cublas_status_alloc_failed",
        "outofmemoryerror",
    )
    return any(marker in lowered for marker in oom_markers)


def run_inference_with_fallback(
    diffrender_dir: Path,
    staging: Path,
    raw_out: Path,
    inference_resolutions,
    inference_n_steps: int,
    model_passes,
    env,
) -> None:
    last_returncode = 1
    for index, inference_res in enumerate(inference_resolutions):
        if raw_out.exists():
            shutil.rmtree(raw_out)
        raw_out.mkdir(parents=True, exist_ok=True)

        cmd = build_inference_cmd(
            diffrender_dir=diffrender_dir,
            staging=staging,
            raw_out=raw_out,
            inference_res=inference_res,
            inference_n_steps=inference_n_steps,
            model_passes=model_passes,
        )
        print(f"Running diffusion-renderer inference at {inference_res} ...")
        proc = subprocess.run(
            cmd,
            cwd=str(diffrender_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = proc.stdout or ""
        if output:
            print(output, end="" if output.endswith("\n") else "\n")
        if proc.returncode == 0:
            return

        last_returncode = proc.returncode
        if index == len(inference_resolutions) - 1:
            sys.exit(proc.returncode)
        if not looks_like_cuda_oom(output):
            print(
                f"Inference failed at {inference_res} with a non-OOM error; not retrying.",
                file=sys.stderr,
            )
            sys.exit(proc.returncode)

        next_res = inference_resolutions[index + 1]
        print(
            f"CUDA OOM at inference_res={inference_res}; retrying with {next_res}.",
            file=sys.stderr,
        )

    sys.exit(last_returncode)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diffusion-renderer inverse rendering on a flat image directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--input_dir",
        required=True,
        type=Path,
        help="Directory containing flat view images (e.g. gen_view_masked/)",
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where decomposed images will be written (e.g. gen_view_decomposite/)",
    )
    ap.add_argument(
        "--diffrender_dir",
        required=True,
        type=Path,
        help="Path to the diffusion-renderer repository root",
    )
    ap.add_argument(
        "--hf_home",
        default=None,
        help="Override HF_HOME to redirect the HuggingFace model cache away from ~",
    )
    ap.add_argument(
        "--inference_res",
        default="512,512",
        help="Primary Height,Width for inference (comma-separated)",
    )
    ap.add_argument(
        "--fallback_inference_res",
        nargs="*",
        default=DEFAULT_FALLBACK_INFERENCE_RES,
        help="Lower Height,Width values to retry on CUDA OOM",
    )
    ap.add_argument(
        "--inference_n_steps",
        type=int,
        default=20,
        help="Number of denoising steps",
    )
    ap.add_argument(
        "--model_passes",
        nargs="+",
        default=DEFAULT_PASSES,
        help="Decomposition passes to run",
    )
    ap.add_argument(
        "--meta_json",
        type=Path,
        default=None,
        help="Optional single_view/meta.json. Views listed in "
             "exclude_from_training_indices are skipped (they will not appear "
             "in {output_dir}, so steps 6/7/8 cascade-filter automatically).",
    )
    args = ap.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    diffrender_dir = args.diffrender_dir.resolve()

    excluded = _excluded_indices(args.meta_json)
    if excluded:
        print(f"Excluding {len(excluded)} view(s) per {args.meta_json}: "
              f"{sorted(excluded)[:10]}{'...' if len(excluded) > 10 else ''}")

    candidates = sorted(
        f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    images = []
    for f in candidates:
        idx = _view_index(f)
        if idx is not None and idx in excluded:
            continue
        images.append(f)
    if not images:
        print(f"ERROR: No images found in {input_dir}"
              + (f" after excluding {len(excluded)} view(s)" if excluded else ""),
              file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(images)} of {len(candidates)} image(s) "
          f"in {input_dir} (excluded {len(candidates) - len(images)})")

    with tempfile.TemporaryDirectory(prefix="diffrender_decompose_") as tmp:
        staging = Path(tmp) / "input"
        raw_out = Path(tmp) / "output"
        staging.mkdir()
        raw_out.mkdir()

        for img in images:
            sub = staging / img.stem
            sub.mkdir()
            shutil.copy(img, sub / img.name)

        env = os.environ.copy()
        if args.hf_home:
            env["HF_HOME"] = args.hf_home
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        inference_resolutions = [args.inference_res, *args.fallback_inference_res]
        run_inference_with_fallback(
            diffrender_dir=diffrender_dir,
            staging=staging,
            raw_out=raw_out,
            inference_resolutions=inference_resolutions,
            inference_n_steps=args.inference_n_steps,
            model_passes=args.model_passes,
            env=env,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        missing = []
        for img in images:
            for pass_name in args.model_passes:
                src = raw_out / img.stem / f"0000.0000.{pass_name}.png"
                if src.exists():
                    shutil.copy(src, output_dir / f"{img.stem}.{pass_name}.png")
                else:
                    missing.append(str(src))

        if missing:
            print("WARNING: some expected outputs were not produced:", file=sys.stderr)
            for missing_path in missing:
                print(f"  {missing_path}", file=sys.stderr)

    print(f"Decomposition complete. Results in {output_dir}")


if __name__ == "__main__":
    main()
