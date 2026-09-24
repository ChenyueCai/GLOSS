# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Tuple


def count_files(dir_path: Path, exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".yml")) -> int:
    """Return number of files with given extensions inside *dir_path* (non‑recursive)."""
    if not dir_path.is_dir():
        return 0
    return sum(1 for p in dir_path.iterdir() if p.suffix.lower() in exts and p.is_file())


def count_prompt_lines(prompt_file: Path) -> int:
    """Return number of non‑blank lines in prompt txt (0 if missing)."""
    if not prompt_file.is_file():
        return 0
    with prompt_file.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def human_name(path: Path) -> str:
    """Last component without trailing slash."""
    return path.name.rstrip("/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check data‑generation output.")
    parser.add_argument("--data_root", type=Path, required=True,
                        help="Parent folder that contains per‑mesh sub‑directories")
    

    cond_dir = "condition_output"
    conditions = ["canny-geonormal", "canny-normal", "depth", "mask", "normal", "geonormal"]
    views_dir = "gen_view"
    masked_dir = "gen_view_masked"
    decomposite_dir = "gen_view_decomposite"
    sr_dir = "gen_view_super"
    texture_dir = "textures_sr"
    meta_dir = "meta"
    
    args = parser.parse_args()
    if not args.data_root.is_dir():
        sys.exit(f"[!] data_root {args.data_root} does not exist or is not a directory.")
    
    header = [
        "mesh",
        "prompt_lines",
        "n_views",
        "n_masked",
        "n_basecolor",
        "n_sr",
        "n_backproj",
        "n_meta"
    ] 
    header.extend([f"n_{c}" for c in conditions])
    
    rows: list[list[int | str]] = []

    for mesh_dir in sorted(p for p in args.data_root.iterdir() if p.is_dir()):
        
        mesh = human_name(mesh_dir)
        prompt_file = mesh_dir / "prompts" / f"{mesh}.txt"
        single_view_dir = mesh_dir / "single_view/civitai2.0"
        count_info = [
            mesh,
            count_prompt_lines(prompt_file),
            count_files(single_view_dir / views_dir),
            count_files(single_view_dir / masked_dir),
            count_files(single_view_dir / decomposite_dir) / 7,
            count_files(single_view_dir / sr_dir),
            count_files(single_view_dir / texture_dir),
            count_files(single_view_dir / meta_dir),
        ]
        conds = [count_files(single_view_dir / cond_dir / c) for c in conditions]
        count_info.extend(conds)
        rows.append(count_info)
        
        
    # pretty‑print
    col_w = [max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))]
    fmt = "  ".join(f"{{:{w}}}" for w in col_w)
    print(fmt.format(*header))
    print("-" * (sum(col_w) + 2 * (len(header) - 1)))
    for row in rows:
        print(fmt.format(*row))


if __name__ == "__main__":
    main()