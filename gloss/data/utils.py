# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


DEFAULT_EXCLUDED_TRAIN_VIEW_IDS_KEY = "exclude_from_training_indices"


def repair_degenerate_albedo_alpha(
    albedo_alpha: torch.Tensor,
    background_mask: torch.Tensor,
    *,
    opaque_threshold: float = 0.98,
) -> torch.Tensor:
    """Repair invalid albedo alpha by falling back to the visible-geometry mask.

    Some imported mesh materials expose a bogus alpha channel for opaque RGB textures.
    In the cached training data this shows up as `albedo_alpha` with no near-opaque
    pixels anywhere on the visible object. That collapses the training loss mask.

    When that happens, treat the sample as fully opaque over visible geometry.
    Inputs are expected to be in the signed `[-1, 1]` representation used by the
    dataloaders, and `background_mask` uses `1` for background pixels.
    """
    if albedo_alpha.numel() == 0 or background_mask.numel() == 0:
        return albedo_alpha

    foreground_mask = (1.0 - background_mask).to(albedo_alpha.dtype)
    if torch.count_nonzero(foreground_mask) == 0:
        return albedo_alpha

    foreground_values = albedo_alpha.masked_select(foreground_mask.expand_as(albedo_alpha) > 0)
    if foreground_values.numel() == 0:
        return albedo_alpha

    if foreground_values.max().item() >= opaque_threshold:
        return albedo_alpha

    return foreground_mask * 2.0 - 1.0


def generate_split_indices(num_samples: int, num_eval: int = 50, num_test: int = 50, random_seed: int = 42) -> dict:
    """
    Generates train eval and test indices for a dataset of a given size.

    Args:
        num_samples (int): The total number of samples in the dataset.
        num_eval (int): Eval count, or a fraction in `(0, 1]`.
        random_seed (int): Seed for reproducible shuffling.

    Returns:
        dict: A dictionary with keys 'train_indices' and 'eval_indices'.
    """
    # For reproducibility
    np.random.seed(random_seed)
    
    # Create a list of all indices
    all_indices = np.arange(num_samples)
    
    # Shuffle the indices
    np.random.shuffle(all_indices)
    
    if isinstance(num_eval, float) and 0.0 < num_eval <= 1.0:
        num_eval = int(round(num_samples * num_eval))
    num_eval = max(1, min(int(num_eval), max(num_samples - 1, 1)))

    # Create train and eval splits
    train_indices = all_indices[num_eval:].tolist()
    eval_indices = all_indices[:num_eval].tolist()
    
    return {
        "train_indices": train_indices,
        "eval_indices": eval_indices,
    }


def get_object_name_from_mesh_path(mesh_path: str) -> Optional[str]:
    if not mesh_path:
        return None
    mesh_path_obj = Path(mesh_path)
    parent_name = mesh_path_obj.parent.name
    return parent_name or None


def load_eval_cond_view_ids(global_root_dir: Optional[str], mesh_path: Optional[str]) -> Optional[list[int]]:
    object_name = get_object_name_from_mesh_path(mesh_path)
    if not global_root_dir or not object_name:
        return None

    cond_dir = Path(global_root_dir) / "eval_data" / "eval_cond_views" / object_name
    if not cond_dir.is_dir():
        return None

    view_ids = []
    for path in sorted(cond_dir.iterdir()):
        match = re.search(r"view(\d+)", path.name)
        if match is not None:
            view_ids.append(int(match.group(1)))

    if not view_ids:
        raise ValueError(f"No eval view ids found under {cond_dir}")
    return view_ids


def resolve_single_view_meta_path(path_like: Optional[Union[str, os.PathLike]]) -> Optional[Path]:
    if not path_like:
        return None

    path = Path(path_like)
    if path.is_file():
        if path.name == "meta.json":
            return path
        return None

    if not path.is_dir():
        return None

    direct_meta = path / "meta.json"
    if direct_meta.is_file():
        return direct_meta

    parent_meta = path.parent / "meta.json"
    if parent_meta.is_file():
        return parent_meta

    return None


def load_excluded_train_view_ids(
    path_like: Optional[Union[str, os.PathLike]],
    *,
    key: str = DEFAULT_EXCLUDED_TRAIN_VIEW_IDS_KEY,
) -> list[int]:
    meta_path = resolve_single_view_meta_path(path_like)
    if meta_path is None or not meta_path.is_file():
        return []

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    if not isinstance(meta, dict):
        raise ValueError(f"Expected a JSON object in {meta_path}, found {type(meta).__name__}")

    raw_ids = meta.get(key, [])
    if raw_ids is None:
        return []
    if not isinstance(raw_ids, list):
        raise ValueError(f"Expected {key} to be a JSON list in {meta_path}")

    excluded_ids: set[int] = set()
    for raw_id in raw_ids:
        if isinstance(raw_id, bool):
            raise ValueError(f"Boolean value {raw_id!r} is not a valid view id in {meta_path}")
        excluded_ids.add(int(raw_id))
    return sorted(excluded_ids)


def apply_excluded_train_view_ids(split_dict: dict, excluded_train_view_ids: list[int]) -> dict:
    if not excluded_train_view_ids:
        return {
            "train_indices": [int(idx) for idx in split_dict.get("train_indices", [])],
            "eval_indices": [int(idx) for idx in split_dict.get("eval_indices", [])],
        }

    excluded_set = set(int(idx) for idx in excluded_train_view_ids)
    return {
        "train_indices": [int(idx) for idx in split_dict.get("train_indices", []) if int(idx) not in excluded_set],
        "eval_indices": [int(idx) for idx in split_dict.get("eval_indices", [])],
    }


def build_split_dict_for_view_ids(
    available_view_ids: list[int],
    *,
    num_eval: int = 50,
    random_seed: int = 42,
    global_root_dir: Optional[str] = None,
    mesh_path: Optional[str] = None,
) -> tuple[dict, bool]:
    """
    Build train/eval split ids for the provided available view ids.

    If `eval_data/eval_cond_views/<object>` exists, that predefined list is used
    as the eval split strictly and all remaining available ids become the train split.
    Otherwise, fall back to the original seeded random split.
    """
    predefined_eval_ids = load_eval_cond_view_ids(global_root_dir, mesh_path)
    if predefined_eval_ids is not None:
        available_view_ids = list(available_view_ids)
        available_view_set = set(available_view_ids)
        missing_eval_ids = [view_id for view_id in predefined_eval_ids if view_id not in available_view_set]
        if missing_eval_ids:
            # Fallback: take the last num_eval view ids (by sorted id) as eval.
            sorted_ids = sorted(available_view_ids)
            eval_count = min(num_eval, max(len(sorted_ids) - 1, 1))
            eval_indices = sorted_ids[-eval_count:]
            eval_view_set = set(eval_indices)
            train_indices = [v for v in available_view_ids if v not in eval_view_set]
            if not train_indices:
                raise ValueError(f"No train views remain after applying last-{eval_count} eval split for mesh={mesh_path}")
            return {
                "train_indices": train_indices,
                "eval_indices": eval_indices,
            }, True

        eval_view_set = set(predefined_eval_ids)
        train_indices = [view_id for view_id in available_view_ids if view_id not in eval_view_set]
        if not train_indices:
            raise ValueError(f"No train views remain after applying eval_cond_views for mesh={mesh_path}")

        return {
            "train_indices": train_indices,
            "eval_indices": list(predefined_eval_ids),
        }, True

    split_dict = generate_split_indices(len(available_view_ids), num_eval, random_seed=random_seed)
    return {
        "train_indices": [available_view_ids[idx] for idx in split_dict["train_indices"]],
        "eval_indices": [available_view_ids[idx] for idx in split_dict["eval_indices"]],
    }, False
    
    
def save_split_indices(split_dict: dict, file_path: str) -> None:
    """
    Saves the split indices (train/eval) to a JSON file.

    Args:
        split_dict (dict): A dictionary with keys 'train_indices' and 'eval_indices'.
        file_path (str): Destination file path for the JSON file.
    """
    file_dir = os.path.dirname(file_path) or "."
    os.makedirs(file_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".split-", suffix=".json", dir=file_dir)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(split_dict, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, file_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_split_indices(file_path: str) -> dict:
    """
    Loads the train/eval indices from a JSON file.

    Args:
        file_path (str): Source file path of the JSON file.

    Returns:
        dict: A dictionary with keys 'train_indices' and 'eval_indices'.
    """
    with open(file_path, 'r') as f:
        split_dict = json.load(f)
    return split_dict
