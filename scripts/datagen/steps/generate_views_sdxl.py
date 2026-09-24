#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Generate single-view images using SDXL + ControlNet Union.

Alternative to the ComfyUI-based step 3 in the data generation pipeline.
Produces 1024×1024 PNG images from pre-rendered conditioning images
(normal, depth, canny) via a multi-type ControlNet applied to SDXL.

Models (downloaded automatically from HuggingFace on the first run):
  - stabilityai/stable-diffusion-xl-base-1.0
  - xinsir/controlnet-union-sdxl-1.0
    task types: depth=1, canny/lineart=3, normal=4

Conditioning layout (produced by step 2 – generate_condition.py):
  <single_view_dir>/condition_output/
    depth/      depth%04d.png
    normal/     normal%04d.png
    canny-geonormal/  canny%04d.png   (or canny-normal/ with --canny_normal)
    mask/       mask%04d.png

Output  (1024×1024):
  <single_view_dir>/gen_view/view%04d.png

Usage:
    python generate_views_sdxl.py \\
        --mesh_name <name> \\
        --exp_dir   <path/to/expr_root> \\
        [--canny_normal] \\
        [--depth_weight 0.5] \\
        [--canny_weight 0.3] \\
        [--normal_weight 0.5] \\
        [--hf_home /path/to/hf_cache]
"""

import argparse
import gc
import logging
import os
import random
from pathlib import Path

import torch
import torchvision
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ---------------------------------------------------------------------------
# Default model identifiers
# ---------------------------------------------------------------------------
DEFAULT_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_CONTROLNET_MODEL = "xinsir/controlnet-union-sdxl-1.0"

# ControlNet Union task-type IDs for xinsir/controlnet-union-sdxl-1.0:
#   0 openpose | 1 depth | 2 hed/pidi/scribble | 3 canny/lineart | 4 normal | 5 segment
DEFAULT_TASK_DEPTH = 1
DEFAULT_TASK_CANNY = 3
DEFAULT_TASK_NORMAL = 4

# Negative prompt matching the ComfyUI pipeline style
NEGATIVE_PROMPT = (
    "(hands), text, error, cropped, (worst quality:1.2), (low quality:1.2), "
    "normal quality, (jpeg artifacts:1.3), signature, watermark, username, "
    "blurry, artist name, monochrome, sketch, censorship, censor, backlit "
    "(copyright:1.2)"
)

# Style suffix appended to every positive prompt (matches ComfyUI workflow)
STYLE_SUFFIX = (
    "studio product photo, centered, seamless white background, soft diffuse "
    "lighting, sharp focus, soft lighting, high quality, cinematic, photograph"
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def image_is_valid(path: Path) -> bool:
    """Return True iff *path* exists and its pixels can be fully decoded."""
    if not path.is_file():
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
    except Exception:
        return False
    return True


def load_condition_image(path: Path, target_size: int = 1024) -> Image.Image:
    """Load a conditioning PNG and resize (nearest-neighbour) to *target_size*."""
    img = Image.open(path).convert("RGB")
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.NEAREST)
    return img


def sorted_pngs(directory: Path) -> list[Path]:
    """Return all *.png files in *directory* sorted by name."""
    return sorted(directory.glob("*.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate view images with SDXL + ControlNet Union "
            "(alternative to ComfyUI step 3)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Paths ---
    parser.add_argument(
        "--mesh_name", required=True,
        help="Mesh name (sub-folder under exp_dir)",
    )
    parser.add_argument(
        "--exp_dir", required=True,
        help="Experiment root directory (same as --exp_dir / <expr_tag> in the pipeline)",
    )
    parser.add_argument(
        "--out_prefix", default="view",
        help="Output filename prefix",
    )
    parser.add_argument(
        "--sv_subdir", default="civitai2.0",
        help="Sub-folder under <mesh>/single_view/ to read conditions from "
             "and write generated views to. Use 'origin' or 'anchors' for "
             "orientation.json-driven runs.",
    )

    # --- Conditioning ---
    parser.add_argument(
        "--canny_normal", action="store_true",
        help="Use canny-normal instead of canny-geonormal as the canny condition",
    )
    parser.add_argument("--normal_weight", type=float, default=0.5,
                        help="ControlNet strength for normal map")
    parser.add_argument("--depth_weight", type=float, default=0.5,
                        help="ControlNet strength for depth map")
    parser.add_argument("--canny_weight", type=float, default=0.3,
                        help="ControlNet strength for canny edges")

    # --- Diffusion sampling ---
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=25,
                        help="Number of denoising steps")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Fixed random seed for every view (default: random per view)",
    )
    parser.add_argument("--output_size", type=int, default=1024,
                        help="Output image resolution (height = width)")

    # --- Model selection ---
    parser.add_argument(
        "--sdxl_model", default=DEFAULT_SDXL_MODEL,
        help="HuggingFace repo ID or local path for the SDXL base model",
    )
    parser.add_argument(
        "--controlnet_model", default=DEFAULT_CONTROLNET_MODEL,
        help="HuggingFace repo ID or local path for the ControlNet Union model",
    )
    parser.add_argument(
        "--hf_home", default=None,
        help="Override HF_HOME for the HuggingFace model cache",
    )

    # --- ControlNet Union task types ---
    parser.add_argument("--condition_task_depth", type=int, default=DEFAULT_TASK_DEPTH,
                        help="ControlNet Union task type for depth")
    parser.add_argument("--condition_task_canny", type=int, default=DEFAULT_TASK_CANNY,
                        help="ControlNet Union task type for canny/lineart")
    parser.add_argument("--condition_task_normal", type=int, default=DEFAULT_TASK_NORMAL,
                        help="ControlNet Union task type for normal map")

    args = parser.parse_args()

    # Override HuggingFace cache location before any diffusers import
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    # -----------------------------------------------------------------------
    # Resolve paths
    # -----------------------------------------------------------------------
    sv_dir = (
        Path(args.exp_dir) / args.mesh_name / "single_view" / args.sv_subdir
    )
    # Prompts file: prefer per-sv-subdir <sv>/prompts.txt; fall back to legacy
    # <mesh>/prompts/<mesh>.txt for older datasets.
    prompt_fp_new = sv_dir / "prompts.txt"
    prompt_fp_legacy = Path(args.exp_dir) / args.mesh_name / "prompts" / f"{args.mesh_name}.txt"
    prompt_fp = prompt_fp_new if prompt_fp_new.is_file() else prompt_fp_legacy
    output_dir = sv_dir / "gen_view"
    condition_dir = sv_dir / "condition_output"
    depth_dir = condition_dir / "depth"
    normal_dir = condition_dir / "normal"
    canny_dir = condition_dir / (
        "canny-normal" if args.canny_normal else "canny-geonormal"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load prompts
    # -----------------------------------------------------------------------
    with open(prompt_fp) as fh:
        prompts = [line.strip() for line in fh if line.strip()]
    num_views = len(prompts)
    logger.info("Loaded %d prompts from %s", num_views, prompt_fp)

    # -----------------------------------------------------------------------
    # Validate condition directories
    # -----------------------------------------------------------------------
    for d in (depth_dir, normal_dir, canny_dir):
        if not d.is_dir():
            raise FileNotFoundError(
                f"Condition directory not found: {d}\n"
                "Run step 2 (generate_condition) first."
            )

    depth_files = sorted_pngs(depth_dir)
    normal_files = sorted_pngs(normal_dir)
    canny_files = sorted_pngs(canny_dir)

    min_cond = min(len(depth_files), len(normal_files), len(canny_files))
    if min_cond < num_views:
        logger.warning(
            "Only %d condition images found (expected %d). "
            "Will generate the first %d views.",
            min_cond, num_views, min_cond,
        )
        num_views = min_cond

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    from diffusers import ControlNetUnionModel
    from diffusers.pipelines.controlnet import StableDiffusionXLControlNetUnionPipeline

    logger.info("Loading ControlNet Union: %s", args.controlnet_model)
    controlnet = ControlNetUnionModel.from_pretrained(
        args.controlnet_model,
        torch_dtype=torch.float16,
    )

    logger.info("Loading SDXL base: %s", args.sdxl_model)
    pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
        args.sdxl_model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    # Offload individual model components to CPU between uses to reduce peak VRAM
    pipe.enable_model_cpu_offload()

    target_size = args.output_size

    # -----------------------------------------------------------------------
    # Generate views
    # -----------------------------------------------------------------------
    for i in range(num_views):
        dst_fp = output_dir / f"{args.out_prefix}{i:04d}.png"
        if image_is_valid(dst_fp):
            logger.info("View %d already exists – skipping.", i)
            continue

        prompt = prompts[i] + ", " + STYLE_SUFFIX
        logger.info("View %d / %d  prompt: %s", i + 1, num_views, prompt)

        # Load and upscale condition images from 512 → target_size
        depth_img = load_condition_image(depth_files[i], target_size)
        normal_img = load_condition_image(normal_files[i], target_size)
        canny_img = load_condition_image(canny_files[i], target_size)

        seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # control_mode maps each image to a ControlNet Union task type
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            control_image=[depth_img, canny_img, normal_img],
            control_mode=[
                args.condition_task_depth,
                args.condition_task_canny,
                args.condition_task_normal,
            ],
            controlnet_conditioning_scale=[
                args.depth_weight,
                args.canny_weight,
                args.normal_weight,
            ],
            height=target_size,
            width=target_size,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )

        img: Image.Image = result.images[0]
        img.save(dst_fp)
        logger.info("Saved → %s", dst_fp)

        # Release GPU memory between views
        del depth_img, normal_img, canny_img, result, img
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("Done.  Generated views in %s", output_dir)


if __name__ == "__main__":
    main()
