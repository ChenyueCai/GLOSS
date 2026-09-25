# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a training checkpoint into the fp16 UNet safetensors file the inference code loads.

    python scripts/export_checkpoint.py <chkpt_*.ckpt> <out>/model.safetensors

Takes the EMA weights when present, keeps only UNet keys, checks they match a freshly built
17-channel UNet, casts to fp16, and writes safetensors with the training step in the metadata.
"""
import argparse
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import load_file, save_file

from gloss.model.standard import load_checkpoint_state_dict
from gloss.utils.paths import get_base_model_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="Training checkpoint (.ckpt or .safetensors)")
    ap.add_argument("dst", help="Output model.safetensors")
    ap.add_argument("--in_channels", type=int, default=17)
    a = ap.parse_args()
    cfg = UNet2DConditionModel.load_config(get_base_model_dir(), subfolder="unet")
    cfg["in_channels"] = a.in_channels
    target = set(UNet2DConditionModel.from_config(cfg).state_dict().keys())
    sd, step = load_checkpoint_state_dict(a.src, map_location="cpu")
    stripped = {(".".join(k.split(".")[1:]) if "." in k else k): v for k, v in sd.items()}
    best = max([sd, stripped], key=lambda d: sum(k in target for k in d))
    best = {k: v for k, v in best.items() if k in target}
    missing = target - set(best)
    if missing:
        raise SystemExit(f"{a.src}: {len(missing)} UNet weights missing, e.g. {sorted(missing)[:3]}")
    best = {k: (v.half() if v.is_floating_point() else v).contiguous() for k, v in best.items()}
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    save_file(best, a.dst, metadata={"global_step": str(step), "source": Path(a.src).name, "dtype": "float16"})
    assert set(load_file(a.dst)) == target
    print(f"wrote {a.dst}: {len(best)} tensors, step {step}")


if __name__ == "__main__":
    main()
