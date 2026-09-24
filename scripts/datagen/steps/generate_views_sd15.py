#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate single-view images with SD1.5 (Realistic Vision) + per-modality ControlNets.

Default config: ``sd15_rv_normal_depth_canny`` from
``data/material-superres/metadata/single_view_configs.yaml`` (SG161222
Realistic Vision V5.1 + normal/depth/canny ControlNets, strengths
1.0 / 0.7 / 0.5).

I/O contract is identical to ``generate_views_sdxl.py`` so the two are
drop-in interchangeable from ``pipeline.py``::

    in:  <exp_dir>/<mesh_name>/single_view/<sv_subdir>/condition_output/{normal,depth,canny-normal,canny-geonormal}/...
         <exp_dir>/<mesh_name>/prompts/<mesh_name>.txt        # one prompt per line, indexed by view
    out: <exp_dir>/<mesh_name>/single_view/<sv_subdir>/gen_view/viewNNNN.png

Pipeline loading is reused from ``scripts/datagen/utils/single_view_gen.py``
(``_load_pipeline`` + ``_run_pipe``); we feed it PIL images for each
ControlNet and per-view prompts.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from PIL import Image

UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
sys.path.insert(0, str(UTILS_DIR))
import single_view_gen as svg  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# Maps a ControlNet `type` (from single_view_configs.yaml) to the
# condition_output sub-folder + filename prefix used by generate_condition.py
# / run_condition_orientation.py.
TYPE_TO_DIR = {
    "normal":       ("normal",            "normal"),
    "geonormal":    ("geonormal",         "geonormal"),
    "depth":        ("depth",             "depth"),
    "canny":        ("canny-normal",      "canny"),
    "canny_geo":    ("canny-geonormal",   "canny"),
    "canny_render": ("canny-render",      "canny"),
}

DEFAULT_CONFIG_NAME = "sd15_rv_normal_depth_canny"
DEFAULT_CONFIGS_FILE = Path(__file__).resolve().parents[3] / "configs" / "single_view_configs.yaml"


def _load_condition_image(condition_dir: Path, ctype: str, idx: int,
                          resolution: int) -> Image.Image:
    if ctype not in TYPE_TO_DIR:
        raise ValueError(f"unsupported ControlNet type: {ctype!r}")
    sub, prefix = TYPE_TO_DIR[ctype]
    fp = condition_dir / sub / f"{prefix}{idx:04d}.png"
    if not fp.is_file():
        raise FileNotFoundError(f"missing condition image: {fp}")
    img = Image.open(fp).convert("RGB")
    if img.size != (resolution, resolution):
        img = img.resize((resolution, resolution), Image.NEAREST)
    return img


def _image_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
    except Exception:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mesh_name", required=True)
    ap.add_argument("--exp_dir", required=True,
                    help="Experiment root (the dir that contains <mesh_name>/).")
    ap.add_argument("--sv_subdir", default="civitai2.0",
                    help="Sub-folder under <mesh>/single_view/ (e.g. origin, anchors).")
    ap.add_argument("--out_prefix", default="view")

    # config / model
    ap.add_argument("--config", default=DEFAULT_CONFIG_NAME,
                    help="Name of the entry in --configs_file to use "
                         "(default: %(default)s).")
    ap.add_argument("--configs_file", type=Path, default=DEFAULT_CONFIGS_FILE)

    # per-controlnet strength overrides (None = use the value from configs_file)
    ap.add_argument("--normal_weight", type=float, default=None)
    ap.add_argument("--depth_weight", type=float, default=None)
    ap.add_argument("--canny_weight", type=float, default=None)

    # sampling
    ap.add_argument("--num_inference_steps", type=int, default=None,
                    help="Override config 'steps'.")
    ap.add_argument("--guidance_scale", type=float, default=None,
                    help="Override config 'guidance'.")
    ap.add_argument("--output_size", type=int, default=None,
                    help="Override config 'resolution'.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Fixed seed for every view (default: random per view).")

    # env
    ap.add_argument("--hf_home", default=None,
                    help="Override HF_HOME for the HuggingFace model cache.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    # ---- resolve paths ----------------------------------------------------
    sv_dir = Path(args.exp_dir) / args.mesh_name / "single_view" / args.sv_subdir
    cond_dir = sv_dir / "condition_output"
    out_dir = sv_dir / "gen_view"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Prompts live per-sv-subdir at <sv>/prompts.txt (so origin and anchors
    # don't share). Fall back to the legacy per-mesh location for
    # backwards compatibility with datasets generated before that change.
    prompt_fp_new = sv_dir / "prompts.txt"
    prompt_fp_legacy = Path(args.exp_dir) / args.mesh_name / "prompts" / f"{args.mesh_name}.txt"
    prompt_fp = prompt_fp_new if prompt_fp_new.is_file() else prompt_fp_legacy

    # ---- load config ------------------------------------------------------
    configs = svg._load_configs(args.configs_file)
    config = svg._find_config(configs, args.config)
    logger.info("config=%s base=%s variant=%s",
                config["name"], config["base_model"], config.get("sd_variant", "sd15"))

    # apply CLI strength overrides in-place on the config dict
    overrides = {"normal": args.normal_weight, "depth": args.depth_weight,
                 "canny": args.canny_weight}
    for c in config["controlnets"]:
        new_s = overrides.get(c["type"])
        if new_s is not None:
            logger.info("  override strength: %s %.2f -> %.2f",
                        c["type"], c["strength"], new_s)
            c["strength"] = float(new_s)

    resolution = args.output_size or int(config.get("resolution", 512))
    steps = args.num_inference_steps or int(config.get("steps", 25))
    guidance = args.guidance_scale or float(config.get("guidance", 7.5))
    suffix = config.get("prompt_suffix", "")
    negative = config.get("negative_prompt", "")

    # ---- load prompts -----------------------------------------------------
    if not prompt_fp.is_file():
        sys.exit(f"prompts file not found: {prompt_fp}\n"
                 f"Run step 2 (generate_view_prompts.py) first.")
    with prompt_fp.open() as fh:
        prompts = [l.strip() for l in fh if l.strip()]
    num_views = len(prompts)
    logger.info("Loaded %d prompts from %s", num_views, prompt_fp)

    # validate condition channel availability up front
    for c in config["controlnets"]:
        sub, _ = TYPE_TO_DIR[c["type"]]
        d = cond_dir / sub
        if not d.is_dir():
            sys.exit(f"condition channel missing: {d}\n"
                     f"Run step 1 (generate_condition / run_condition_orientation) first.")
    # cap num_views by least-populated channel
    min_cond = min(len(list((cond_dir / TYPE_TO_DIR[c["type"]][0]).glob("*.png")))
                   for c in config["controlnets"])
    if min_cond < num_views:
        logger.warning("only %d condition images available (%d prompts); "
                       "generating %d views",
                       min_cond, num_views, min_cond)
        num_views = min_cond

    # ---- load pipeline ----------------------------------------------------
    pipe = svg._load_pipeline(config, args.device)

    # ---- per-view loop ----------------------------------------------------
    cn_modes = [c.get("control_mode") for c in config["controlnets"]]
    cn_scales = [float(c["strength"]) for c in config["controlnets"]]
    cn_types = [c["type"] for c in config["controlnets"]]
    logger.info("ControlNets=%s scales=%s", cn_types, cn_scales)

    for i in range(num_views):
        out_fp = out_dir / f"{args.out_prefix}{i:04d}.png"
        if _image_is_valid(out_fp):
            logger.info("View %d already exists – skipping.", i)
            continue

        cn_images = [_load_condition_image(cond_dir, c["type"], i, resolution)
                     for c in config["controlnets"]]
        full_prompt = prompts[i]
        if suffix:
            full_prompt = f"{full_prompt}, {suffix}"

        seed = args.seed if args.seed is not None else int(torch.seed() % (2**32 - 1))
        gen = torch.Generator(device=args.device).manual_seed(int(seed))
        logger.info("View %d / %d  seed=%d  prompt: %s",
                    i + 1, num_views, seed, full_prompt[:140] + ("…" if len(full_prompt) > 140 else ""))

        img = svg._run_pipe(
            pipe, config,
            full_prompt=full_prompt, negative_prompt=negative,
            cn_images=cn_images, cn_modes=cn_modes, cn_scales=cn_scales,
            resolution=resolution, steps=steps, guidance=guidance,
            generator=gen,
        )
        img.save(out_fp)
        logger.info("Saved -> %s", out_fp)

    logger.info("Done. Generated views in %s", out_dir)


if __name__ == "__main__":
    main()
