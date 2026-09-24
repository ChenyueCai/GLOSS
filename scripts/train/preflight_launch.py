#!/usr/bin/env python
"""Validate launch.sh inputs without submitting a Slurm job."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path


VIEW_TAR_PATTERN = re.compile(r"^view(\d+)(?:-(\d+))?\.tar$")


@dataclass
class PreflightSummary:
    object_name: str
    mesh_path: str
    single_view_dir: str
    data_root_dir: str
    available_view_count: int
    max_train_views_after_eval: int
    min_shards_per_view: int
    max_shards_per_view: int
    inferred_num_local_views: int
    requested_train_view_limit: str
    requested_eval_views: str
    inferred_eval_views: int
    predefined_eval_count: int
    filtered_low_quality_count: int


def _env_default(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global_root_dir", default=_env_default("GLOBAL_ROOT_DIR"))
    parser.add_argument("--data_mesh", default=_env_default("DATA_MESH"))
    parser.add_argument("--data_base_dir", default=_env_default("DATA_BASE_DIR"))
    parser.add_argument("--single_view_dir", default=_env_default("SINGLE_VIEW_DIR"))
    parser.add_argument("--data_train_view_limit", default=_env_default("DATA_TRAIN_VIEW_LIMIT", ""))
    parser.add_argument("--data_eval_views", default=_env_default("DATA_EVAL_VIEWS", "auto"))
    return parser.parse_args()


def require_path(path: Path, description: str, *, is_dir: bool = False) -> None:
    if is_dir:
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {description}: {path}")
        return
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def discover_shards(data_root_dir: Path) -> dict[int, list[Path]]:
    shard_map: dict[int, list[Path]] = {}
    for shard_path in sorted(data_root_dir.glob("view*.tar")):
        match = VIEW_TAR_PATTERN.match(shard_path.name)
        if match is None:
            continue
        view_idx = int(match.group(1))
        shard_map.setdefault(view_idx, []).append(shard_path)
    if not shard_map:
        raise FileNotFoundError(f"No WebDataset shards found under {data_root_dir}")
    return shard_map


def discover_eval_ids(global_root_dir: Path, object_name: str) -> list[int]:
    eval_dir = global_root_dir / "eval_data" / "eval_cond_views" / object_name
    if not eval_dir.is_dir():
        return []

    eval_ids: list[int] = []
    for path in sorted(eval_dir.iterdir()):
        match = re.search(r"view(\d+)", path.name)
        if match is not None:
            eval_ids.append(int(match.group(1)))
    return eval_ids


def infer_auto_eval_views(view_count: int) -> int:
    if view_count <= 1:
        return 1
    if view_count < 50:
        return view_count - 1
    return 50


def count_single_view_images(single_view_dir: Path) -> int:
    return len(glob.glob(str(single_view_dir / "view*.basecolor.png")))


def main() -> int:
    args = parse_args()
    missing = [
        flag
        for flag in ("global_root_dir", "data_mesh", "data_base_dir", "single_view_dir")
        if not getattr(args, flag)
    ]
    if missing:
        raise SystemExit(f"Missing required arguments/env vars: {', '.join(missing)}")

    global_root_dir = Path(args.global_root_dir).resolve()
    mesh_path = global_root_dir / args.data_mesh
    data_root_dir = global_root_dir / args.data_base_dir
    single_view_dir = global_root_dir / args.single_view_dir

    require_path(mesh_path, "mesh file")
    require_path(data_root_dir, "WDS directory", is_dir=True)
    require_path(single_view_dir, "single-view directory", is_dir=True)

    single_view_count = count_single_view_images(single_view_dir)
    if single_view_count == 0:
        raise FileNotFoundError(f"No single-view basecolor images found under {single_view_dir}")

    shard_map = discover_shards(data_root_dir)
    shard_counts = [len(paths) for paths in shard_map.values()]
    available_view_ids = sorted(shard_map)
    object_name = Path(args.data_mesh).parent.name

    predefined_eval_ids = discover_eval_ids(global_root_dir, object_name)
    missing_eval_ids = [view_id for view_id in predefined_eval_ids if view_id not in shard_map]
    if missing_eval_ids:
        raise ValueError(
            "eval_cond_views contains ids missing from WDS shards: "
            + ", ".join(str(view_id) for view_id in missing_eval_ids[:10])
        )

    data_meta_fp = data_root_dir / "data_meta.json"
    filtered_low_quality_count = 0
    if data_meta_fp.is_file():
        with data_meta_fp.open("r", encoding="utf-8") as handle:
            data_meta = json.load(handle)
        filtered_low_quality_count = len(data_meta.get("low_quality_sample_indices", []))

    inferred_eval_views = infer_auto_eval_views(len(available_view_ids))
    if args.data_eval_views != "auto":
        inferred_eval_views = int(args.data_eval_views)
    max_train_views_after_eval = len(available_view_ids) - len(predefined_eval_ids)
    if not predefined_eval_ids:
        max_train_views_after_eval = len(available_view_ids) - inferred_eval_views

    if args.data_train_view_limit:
        requested_train_view_limit = int(args.data_train_view_limit)
        if requested_train_view_limit > max_train_views_after_eval:
            raise ValueError(
                f"Requested {requested_train_view_limit} train views, but at most "
                f"{max_train_views_after_eval} are available after reserving eval views"
            )
    else:
        requested_train_view_limit = ""

    summary = PreflightSummary(
        object_name=object_name,
        mesh_path=str(mesh_path),
        single_view_dir=str(single_view_dir),
        data_root_dir=str(data_root_dir),
        available_view_count=len(available_view_ids),
        max_train_views_after_eval=max_train_views_after_eval,
        min_shards_per_view=min(shard_counts),
        max_shards_per_view=max(shard_counts),
        inferred_num_local_views=max(shard_counts) * 100,
        requested_train_view_limit=str(requested_train_view_limit),
        requested_eval_views=args.data_eval_views,
        inferred_eval_views=inferred_eval_views,
        predefined_eval_count=len(predefined_eval_ids),
        filtered_low_quality_count=filtered_low_quality_count,
    )

    print("Launch preflight passed")
    print(json.dumps(summary.__dict__, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
