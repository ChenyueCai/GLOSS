# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse, json, pathlib, yaml   # pip install pyyaml

DEFAULT_MORE_ARGS = {
    "fov_min":      1.0,
    "fov_max":      1.0,
    "azi_min":      0.0,
    "azi_max":      3.14,
    "elev_min":    -1.1,
    "elev_max":     0.8,
    "viewdist_min": 1.25,
    "viewdist_max": 1.35,
}


def load_meshes_json(path: pathlib.Path) -> dict:
    """Return a dict mapping mesh name -> meshes.json entry."""
    with path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    return {e["name"]: e for e in entries}


def build_entries(root: pathlib.Path, mesh_info: dict):
    """Yield one YAML entry per direct sub-directory."""
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            args = dict(DEFAULT_MORE_ARGS)
            info = mesh_info.get(sub.name)
            if info:
                azi = info.get("valid_azi_angle")
                elv = info.get("valid_elv_angle")
                if azi and len(azi) == 2:
                    args["azi_min"] = azi[0]
                    args["azi_max"] = azi[1]
                if elv and len(elv) == 2:
                    args["elev_min"] = elv[0]
                    args["elev_max"] = elv[1]
            yield {"mesh_name": sub.name, "more_args": args}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path,
                    help="directory whose sub-folders are the mesh names")
    ap.add_argument("-o", "--output", default="config.yaml",
                    help="output YAML file name")
    ap.add_argument("--meshes_json", type=pathlib.Path, default=None,
                    help="path to meshes.json; if provided, valid_azi_angle and "
                         "valid_elv_angle override the default azi/elev values")
    args = ap.parse_args()

    mesh_info = load_meshes_json(args.meshes_json) if args.meshes_json else {}
    entries = list(build_entries(args.root, mesh_info))
    if not entries:
        raise SystemExit(f"No sub‑directories found in {args.root!s}")

    with open(args.output, "w", encoding="utf-8") as fh:
        yaml.safe_dump(entries, fh, sort_keys=False)

    print(f"✓ Wrote {args.output} with {len(entries)} mesh entries.")

if __name__ == "__main__":
    main()
