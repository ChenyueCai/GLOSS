# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import glob
import json
import numpy as np
import io
import random
import math
import re
import torch
import torchvision
import imageio.v3 as imageio

import itertools
from functools import partial
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union
from braceexpand import braceexpand

import webdataset as wds
from webdataset.tariterators import (
    tar_file_expander,
    group_by_keys,
    url_opener,
    valid_sample,
)

from gloss.data.utils import (
    apply_excluded_train_view_ids,
    build_split_dict_for_view_ids,
    load_excluded_train_view_ids,
    load_split_indices,
    repair_degenerate_albedo_alpha,
    save_split_indices,
)

FILE_SUFFIXES = [
    "_albedo.png",
    "_albedo_alpha.png",
    "_geo_camera_normals.png",
    "_camera_normals.png",
    "_relative_positions.png",
    "_positions.png",
]

VIEW_TAR_PATTERN = re.compile(r"^view(\d+)(?:-(\d+))?\.tar$")


@dataclass
class EnsembleWebdatasetSourceSpec:
    """Configuration for one object-level WDS source in an ensemble dataset."""

    name: str
    data_dir: Union[str, list[str]]
    num_views: int
    num_local_views: int
    mesh_path: Optional[str] = None
    view_dir: Optional[str] = None
    weight: float = 1.0
    train_view_limit: int = 0
    train_view_selection: str = "random"
    eval_views: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleWebdatasetResolvedSource:
    """Resolved runtime state for one ensemble source."""

    source_index: int
    spec: EnsembleWebdatasetSourceSpec
    dataset: "MultiviewLocalMeshRendersWebdataset"

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def num_train_examples(self) -> int:
        return self.dataset.num_train_examples


@dataclass
class EnsembleWebdatasetDiscoveryResult:
    """Filesystem discovery result for ensemble WDS sources."""

    sources: list[EnsembleWebdatasetSourceSpec]
    skipped: list[dict[str, str]] = field(default_factory=list)


def _close_iterator_if_possible(iterator) -> None:
    close_fn = getattr(iterator, "close", None)
    if callable(close_fn):
        close_fn()


class _ClosableIterableDataset(torch.utils.data.IterableDataset):
    """Wrap an iterable factory and close the created iterator when iteration finishes."""

    def __init__(self, iterable_factory, *, length: Optional[int] = None) -> None:
        super().__init__()
        self.iterable_factory = iterable_factory
        self.length = length

    def __iter__(self):
        iterator = iter(self.iterable_factory())
        try:
            for sample in iterator:
                yield sample
        finally:
            _close_iterator_if_possible(iterator)

    def __len__(self):
        if self.length is None:
            raise TypeError("Length is not defined for this iterable dataset")
        return self.length


class _EnsembleWebdatasetIterable(torch.utils.data.IterableDataset):
    """Simple iterable wrapper that mixes already-decoded per-source WDS datasets."""

    def __init__(
        self,
        sources: Sequence[EnsembleWebdatasetResolvedSource],
        *,
        mixing_mode: str = "round_robin",
        max_samples: Optional[int] = None,
        shuffle_sources: bool = False,
        include_metadata: bool = True,
    ) -> None:
        super().__init__()
        self.sources = list(sources)
        self.mixing_mode = mixing_mode
        self.max_samples = max_samples
        self.shuffle_sources = shuffle_sources
        self.include_metadata = include_metadata

        if self.mixing_mode not in {"round_robin", "by_object"}:
            raise ValueError(f"Unsupported ensemble mixing mode: {self.mixing_mode}")

    def _annotate_sample(self, sample: dict, source: EnsembleWebdatasetResolvedSource) -> dict:
        annotated = dict(sample)
        dataset_id = annotated.get("dataset_id")
        if dataset_id is not None:
            sample_uid = f"{source.name}:view{int(dataset_id):04d}"
        else:
            sample_uid = source.name

        annotated["source_name"] = source.name
        annotated["source_index"] = source.source_index
        annotated["source_weight"] = float(source.spec.weight)
        annotated["sample_uid"] = sample_uid
        if source.spec.metadata:
            annotated["source_metadata"] = dict(source.spec.metadata)
        if not self.include_metadata:
            annotated = {
                key: value
                for key, value in annotated.items()
                if isinstance(value, torch.Tensor)
            }
        return annotated

    def __iter__(self):
        if not self.sources:
            return

        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        active_indices = list(range(len(self.sources)))
        source_iterators = {
            index: iter(source.dataset.get_dataset())
            for index, source in enumerate(self.sources)
        }

        produced_global = 0
        rng = random.Random()
        try:
            while active_indices:
                current_indices = active_indices.copy()
                if self.shuffle_sources:
                    rng.shuffle(current_indices)

                made_progress = False
                for source_index in current_indices:
                    if self.max_samples is not None and produced_global >= self.max_samples:
                        return

                    source = self.sources[source_index]
                    iterator = source_iterators.get(source_index)
                    if iterator is None:
                        continue

                    try:
                        sample = next(iterator)
                    except StopIteration:
                        _close_iterator_if_possible(iterator)
                        del source_iterators[source_index]
                        active_indices.remove(source_index)
                        continue

                    made_progress = True
                    if produced_global % num_workers == worker_id:
                        yield self._annotate_sample(sample, source)
                    produced_global += 1

                if not made_progress:
                    return
        finally:
            for iterator in source_iterators.values():
                _close_iterator_if_possible(iterator)

    def __len__(self):
        if self.max_samples is not None:
            return self.max_samples
        return sum(source.num_train_examples for source in self.sources)


def limit_train_views(train_indices, train_view_limit, random_seed, selection="random"):
    if train_view_limit is None or int(train_view_limit) <= 0:
        return list(train_indices)

    train_view_limit = int(train_view_limit)
    train_indices = list(train_indices)
    if train_view_limit >= len(train_indices):
        return train_indices

    rng = np.random.RandomState(random_seed)
    candidate_indices = np.array(sorted(train_indices), dtype=np.int64)
    if selection == "sequential":
        return [int(idx) for idx in candidate_indices[:train_view_limit].tolist()]
    if selection != "random":
        raise ValueError(f"Unsupported train-view selection mode: {selection}")

    chosen = rng.choice(candidate_indices, size=train_view_limit, replace=False)
    return sorted(int(idx) for idx in chosen.tolist())


def discover_available_webdataset_shards(data_dir):
    shard_map = {}
    shard_glob = os.path.join(data_dir, "view*.tar")
    for shard_path in sorted(glob.glob(shard_glob)):
        match = VIEW_TAR_PATTERN.match(os.path.basename(shard_path))
        if match is None:
            continue
        view_idx = int(match.group(1))
        shard_idx = int(match.group(2) or 0)
        shard_map.setdefault(view_idx, []).append((shard_idx, shard_path))
    return {
        view_idx: [path for _, path in sorted(shards)]
        for view_idx, shards in sorted(shard_map.items())
    }


def get_tar_files(prefixes, tars_per_view=None):
    matching_files = []
    for prefix in prefixes:
        if tars_per_view is None:
            pattern = f"{prefix}*.tar"
            matching_files.extend(glob.glob(pattern))
        else:
            for i in range(tars_per_view):
                if os.path.exists(f"{prefix}-{i}.tar"):
                    matching_files.append(f"{prefix}-{i}.tar")
    return matching_files


def _resolve_path_under_root(global_root_dir, path_like):
    if path_like is None:
        return None
    if isinstance(path_like, (list, tuple)):
        return [_resolve_path_under_root(global_root_dir, item) for item in path_like]
    if os.path.isabs(path_like):
        return path_like
    if not global_root_dir:
        return path_like
    return os.path.join(global_root_dir, path_like)


def _infer_num_local_views_from_shards(available_shards, local_views_per_shard=100):
    if not available_shards:
        return 0
    max_shards_per_view = max(len(shards) for shards in available_shards.values())
    return max_shards_per_view * int(local_views_per_shard)


def discover_ensemble_webdataset_source_specs(
    objects_root_dir,
    *,
    object_names: Optional[Sequence[str]] = None,
    exclude_object_names: Optional[Sequence[str]] = None,
    mesh_filename: str = "scene.gltf",
    multiview_root_name: str = "multi_view",
    multiview_wds_dirname: Optional[str] = None,
    single_view_root_name: str = "single_view",
    single_view_variant: str = "civitai2.0",
    single_view_image_dir_candidates: Sequence[str] = ("gen_view_decomposite", "gen_view_super"),
    num_local_views: Optional[int] = None,
    local_views_per_shard: int = 100,
    default_weight: float = 1.0,
    strict: bool = False,
):
    """
    Discover per-object WDS sources from a train-mesh style root directory.

    Expected layout per object:
    - <object>/<mesh_filename>
    - <object>/<multiview_root_name>/<something-ending-in-'-wds'>
    - <object>/<single_view_root_name>/<single_view_variant>/<image_subdir>/view####.basecolor.png
    """
    exclude_object_names = set(exclude_object_names or [])
    if object_names is None:
        candidate_names = sorted(
            entry
            for entry in os.listdir(objects_root_dir)
            if os.path.isdir(os.path.join(objects_root_dir, entry))
        )
    else:
        candidate_names = [name for name in object_names]

    sources = []
    skipped = []

    for object_name in candidate_names:
        if object_name in exclude_object_names:
            continue

        object_root = os.path.join(objects_root_dir, object_name)
        if not os.path.isdir(object_root):
            reason = f"missing object directory: {object_root}"
            if strict:
                raise FileNotFoundError(reason)
            skipped.append({"name": object_name, "reason": reason})
            continue

        mesh_path = os.path.join(object_root, mesh_filename)
        if not os.path.isfile(mesh_path):
            reason = f"missing mesh file: {mesh_path}"
            if strict:
                raise FileNotFoundError(reason)
            skipped.append({"name": object_name, "reason": reason})
            continue

        multiview_root = os.path.join(object_root, multiview_root_name)
        if not os.path.isdir(multiview_root):
            reason = f"missing multiview directory: {multiview_root}"
            if strict:
                raise FileNotFoundError(reason)
            skipped.append({"name": object_name, "reason": reason})
            continue

        if multiview_wds_dirname is not None:
            wds_dir = os.path.join(multiview_root, multiview_wds_dirname)
            if not os.path.isdir(wds_dir):
                reason = f"missing WDS directory: {wds_dir}"
                if strict:
                    raise FileNotFoundError(reason)
                skipped.append({"name": object_name, "reason": reason})
                continue
        else:
            wds_candidates = sorted(
                path
                for path in glob.glob(os.path.join(multiview_root, "*-wds"))
                if os.path.isdir(path) and ".partial-" not in os.path.basename(path)
            )
            if not wds_candidates:
                reason = f"no WDS directories found under {multiview_root}"
                if strict:
                    raise FileNotFoundError(reason)
                skipped.append({"name": object_name, "reason": reason})
                continue
            if len(wds_candidates) > 1:
                reason = f"multiple WDS directories found under {multiview_root}: {wds_candidates}"
                if strict:
                    raise ValueError(reason)
                skipped.append({"name": object_name, "reason": reason})
                continue
            wds_dir = wds_candidates[0]

        available_shards = discover_available_webdataset_shards(wds_dir)
        if not available_shards:
            reason = f"no view tar shards found under {wds_dir}"
            if strict:
                raise FileNotFoundError(reason)
            skipped.append({"name": object_name, "reason": reason})
            continue

        single_view_root = os.path.join(object_root, single_view_root_name, single_view_variant)
        if not os.path.isdir(single_view_root):
            reason = f"missing single-view root: {single_view_root}"
            if strict:
                raise FileNotFoundError(reason)
            skipped.append({"name": object_name, "reason": reason})
            continue

        resolved_view_dir = None
        resolved_view_subdir = None
        for candidate in single_view_image_dir_candidates:
            candidate_dir = os.path.join(single_view_root, candidate)
            if not os.path.isdir(candidate_dir):
                continue
            if glob.glob(os.path.join(candidate_dir, "view*.basecolor.png")):
                resolved_view_dir = candidate_dir
                resolved_view_subdir = candidate
                break
        if resolved_view_dir is None:
            reason = (
                f"no single-view image directory found under {single_view_root} "
                f"for candidates={list(single_view_image_dir_candidates)}"
            )
            if strict:
                raise FileNotFoundError(reason)
            skipped.append({"name": object_name, "reason": reason})
            continue

        inferred_num_local_views = _infer_num_local_views_from_shards(
            available_shards,
            local_views_per_shard=local_views_per_shard,
        )
        spec_num_local_views = inferred_num_local_views if num_local_views is None else int(num_local_views)
        if spec_num_local_views <= 0:
            reason = f"invalid inferred num_local_views={spec_num_local_views} for {wds_dir}"
            if strict:
                raise ValueError(reason)
            skipped.append({"name": object_name, "reason": reason})
            continue

        sources.append(
            EnsembleWebdatasetSourceSpec(
                name=object_name,
                data_dir=wds_dir,
                num_views=len(available_shards),
                num_local_views=spec_num_local_views,
                mesh_path=mesh_path,
                view_dir=resolved_view_dir,
                weight=default_weight,
                metadata={
                    "object_root": object_root,
                    "wds_dir": wds_dir,
                    "single_view_root": single_view_root,
                    "single_view_image_dir": resolved_view_subdir,
                    "available_view_count": len(available_shards),
                    "inferred_num_local_views": inferred_num_local_views,
                },
            )
        )

    if strict and not sources:
        raise ValueError(f"No valid ensemble WDS sources discovered under {objects_root_dir}")
    return EnsembleWebdatasetDiscoveryResult(sources=sources, skipped=skipped)


def resolve_multi_view_webdataset_split(
    *,
    data_dir,
    indices_fp,
    num_eval=50,
    random_seed=0,
    mask=None,
    global_root_dir=None,
    mesh_path=None,
    view_dir=None,
    train_view_limit=0,
    train_view_selection="random",
):
    available_shards = discover_available_webdataset_shards(data_dir)
    if not available_shards:
        raise FileNotFoundError(f"No WebDataset tar shards found under {data_dir}")

    available_view_ids = list(available_shards.keys())
    if mask is not None:
        available_view_ids = [view_idx for view_idx in mask if view_idx in available_shards]
    if not available_view_ids:
        raise ValueError(f"No requested WebDataset views are present under {data_dir}")

    excluded_train_view_ids = load_excluded_train_view_ids(view_dir)
    available_view_set = set(available_view_ids)

    def regenerate_split():
        if len(available_view_ids) == 1:
            return {
                "train_indices": available_view_ids.copy(),
                "eval_indices": available_view_ids.copy(),
            }, False
        eval_count = min(num_eval, len(available_view_ids) - 1)
        return build_split_dict_for_view_ids(
            available_view_ids,
            num_eval=eval_count,
            random_seed=random_seed,
            global_root_dir=global_root_dir,
            mesh_path=mesh_path,
        )

    if not os.path.exists(indices_fp):
        split_dict, _ = regenerate_split()
    else:
        try:
            split_dict = load_split_indices(indices_fp)
        except (json.JSONDecodeError, OSError):
            split_dict, _ = regenerate_split()
            save_split_indices(split_dict, indices_fp)
        expected_split_dict, used_predefined_eval = regenerate_split()
        train_indices = [idx for idx in split_dict.get("train_indices", []) if idx in available_view_set]
        eval_indices = [idx for idx in split_dict.get("eval_indices", []) if idx in available_view_set]
        if used_predefined_eval:
            split_dict = expected_split_dict
            save_split_indices(split_dict, indices_fp)
        elif not train_indices or not eval_indices:
            split_dict, _ = regenerate_split()
            save_split_indices(split_dict, indices_fp)
        else:
            split_dict = {
                "train_indices": train_indices,
                "eval_indices": eval_indices,
            }

    split_dict = apply_excluded_train_view_ids(split_dict, excluded_train_view_ids)
    if not split_dict["train_indices"]:
        raise ValueError(
            "No training views remain after applying excluded training ids from "
            f"{os.path.join(view_dir or '', 'meta.json')}"
        )

    split_dict = {
        "train_indices": limit_train_views(
            split_dict["train_indices"],
            train_view_limit,
            random_seed,
            train_view_selection,
        ),
        "eval_indices": split_dict["eval_indices"],
    }
    save_split_indices(split_dict, indices_fp)

    train_indices = split_dict["train_indices"]
    eval_indices = split_dict["eval_indices"]
    train_tars = [shard for idx in train_indices for shard in available_shards[idx]]
    eval_tars = [shard for idx in eval_indices for shard in available_shards[idx]]

    return {
        "data_dir": data_dir,
        "indices_fp": indices_fp,
        "available_shards": available_shards,
        "available_view_ids": available_view_ids,
        "excluded_train_view_ids": excluded_train_view_ids,
        "train_indices": train_indices,
        "eval_indices": eval_indices,
        "train_tars": train_tars,
        "eval_tars": eval_tars,
        "view_dir": view_dir,
        "mesh_path": mesh_path,
    }


def configure_multi_view_webdataset(
    data_config,
    global_root_dir,
    indices_fp,
    batch_size,
    num_eval=50,
    random_seed=0,
    mask=None,
    eval_shuffle=True,
):
    data_dir = os.path.join(global_root_dir, data_config.data_base_dir)
    single_view_dir = os.path.join(global_root_dir, data_config.single_view_dir)
    split_result = resolve_multi_view_webdataset_split(
        data_dir=data_dir,
        indices_fp=indices_fp,
        num_eval=num_eval,
        random_seed=random_seed,
        mask=mask,
        global_root_dir=global_root_dir,
        mesh_path=data_config.mesh,
        view_dir=single_view_dir,
        train_view_limit=getattr(data_config, "train_view_limit", 0),
        train_view_selection=getattr(data_config, "train_view_selection", "random"),
    )

    train_indices = split_result["train_indices"]
    eval_indices = split_result["eval_indices"]
    train_tars = split_result["train_tars"]
    eval_tars = split_result["eval_tars"]
    available_view_ids = split_result["available_view_ids"]
    excluded_train_view_ids = split_result["excluded_train_view_ids"]

    print("Discovered shard views", len(available_view_ids))
    if excluded_train_view_ids:
        print("Excluded train views", len(excluded_train_view_ids))
    if getattr(data_config, "train_view_limit", 0):
        print("Train view limit", data_config.train_view_limit)
    print("Train views", len(train_indices), "Train shards", len(train_tars))
    print("Eval views", len(eval_indices), "Eval shards", len(eval_tars))
    if data_config.normal_cond == "geonormal":
        in_channels = ["geo_camera_normals", "relative_positions", "albedo"]
    else:
        in_channels = ["camera_normals", "relative_positions", "albedo"]
    train_dataset = MultiviewLocalMeshRendersWebdataset(
        data_dir=train_tars,
        num_views=len(train_indices),
        num_local_views=data_config.num_local_views,
        batch_size=batch_size,
        view_dir=single_view_dir,
        input_channels=in_channels,
    )
    eval_dataset = MultiviewLocalMeshRendersWebdataset(
        data_dir=eval_tars,
        num_views=len(eval_indices),
        num_local_views=data_config.num_local_views,
        batch_size=batch_size,
        view_dir=single_view_dir,
        input_channels=in_channels,
        shuffle=eval_shuffle,
    )
    return train_dataset, eval_dataset


def configure_ensemble_multi_view_webdataset(
    source_specs: Sequence[EnsembleWebdatasetSourceSpec],
    *,
    indices_dir: str,
    batch_size: int,
    num_eval: int = 50,
    random_seed: int = 0,
    masks_by_name: Optional[dict[str, Sequence[int]]] = None,
    global_root_dir: Optional[str] = None,
    input_channels=["geo_camera_normals", "relative_positions", "albedo"],
    exclude_labels=None,
    shuffle: bool = True,
    eval_shuffle: bool = False,
    mixing_mode: str = "round_robin",
    include_metadata: bool = True,
    **kwargs,
):
    """
    Resolve train/eval splits independently for every enlisted source and return:
    - one ensemble train dataset
    - per-source eval datasets
    - per-source split metadata
    """
    os.makedirs(indices_dir, exist_ok=True)
    masks_by_name = masks_by_name or {}

    train_source_specs = []
    eval_datasets = {}
    split_results = {}

    for spec in source_specs:
        resolved_data_dir = _resolve_path_under_root(global_root_dir, spec.data_dir)
        resolved_view_dir = _resolve_path_under_root(global_root_dir, spec.view_dir)
        resolved_mesh_path = _resolve_path_under_root(global_root_dir, spec.mesh_path)
        source_indices_fp = os.path.join(indices_dir, f"{spec.name}.json")
        source_num_eval = spec.eval_views if spec.eval_views is not None else num_eval

        split_result = resolve_multi_view_webdataset_split(
            data_dir=resolved_data_dir,
            indices_fp=source_indices_fp,
            num_eval=source_num_eval,
            random_seed=random_seed,
            mask=masks_by_name.get(spec.name),
            global_root_dir=global_root_dir,
            mesh_path=resolved_mesh_path,
            view_dir=resolved_view_dir,
            train_view_limit=spec.train_view_limit,
            train_view_selection=spec.train_view_selection,
        )
        split_results[spec.name] = split_result

        train_source_specs.append(
            EnsembleWebdatasetSourceSpec(
                name=spec.name,
                data_dir=split_result["train_tars"],
                num_views=len(split_result["train_indices"]),
                num_local_views=spec.num_local_views,
                mesh_path=resolved_mesh_path,
                view_dir=resolved_view_dir,
                weight=spec.weight,
                train_view_limit=spec.train_view_limit,
                train_view_selection=spec.train_view_selection,
                eval_views=source_num_eval,
                metadata=dict(spec.metadata),
            )
        )
        eval_datasets[spec.name] = MultiviewLocalMeshRendersWebdataset(
            data_dir=split_result["eval_tars"],
            num_views=len(split_result["eval_indices"]),
            num_local_views=spec.num_local_views,
            batch_size=batch_size,
            view_dir=resolved_view_dir,
            input_channels=list(input_channels),
            exclude_labels=exclude_labels,
            shuffle=eval_shuffle,
            **kwargs,
        )

    train_dataset = EnsembleMultiviewLocalMeshRendersWebdataset(
        train_source_specs,
        batch_size=batch_size,
        input_channels=list(input_channels),
        exclude_labels=exclude_labels,
        shuffle=shuffle,
        mixing_mode=mixing_mode,
        include_metadata=include_metadata,
        **kwargs,
    )
    return train_dataset, eval_datasets, split_results


def custom_key_fn(path):
    for suffix in FILE_SUFFIXES:
        if path.endswith(suffix):
            key = path.split("/")[0]
            base = path[: -len(suffix)]
            channel = suffix.lstrip("_").removesuffix(".png")
            return key, f"{base}:{channel}"
    return None, None


class MultiviewLocalMeshRendersWebdataset:
    def __init__(
        self,
        data_dir,
        num_views,
        num_local_views,
        batch_size,
        view_dir=None,
        input_channels=["geo_camera_normals", "relative_positions", "albedo"],
        exclude_labels=None,
        shuffle=True,
        **kwargs,
    ) -> None:
        if not isinstance(data_dir, str):
            data_dir = [list(braceexpand(urls)) for urls in data_dir]
            data_dir = list(itertools.chain.from_iterable(data_dir))
        else:
            data_dir = list(braceexpand(data_dir))

        url_opener_custom = partial(url_opener)

        self.num_train_examples = num_views * num_local_views
        if "albedo" in input_channels:
            input_channels += ["albedo_alpha"]
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.view_dir = view_dir
        self.shuffle = shuffle

        filter_labels = self.input_channels

        def wds_file_selector(sample):
            if exclude_labels is not None:
                for k in exclude_labels:
                    if k in sample:
                        return False
            for k in filter_labels:
                if k in sample:
                    return True
            return False

        def tarfile_to_samples(src, handler=wds.warn_and_continue):
            streams = url_opener_custom(src, handler=handler)
            files = tar_file_expander(streams, handler=handler, select_files=wds_file_selector)
            samples = group_by_keys(files, handler=handler, keys=custom_key_fn)
            return samples

        def custom_collate(samples):
            res = {}
            for k in samples[0].keys():
                if k == "prompt_embed" or k == "view_embed":
                    res[k] = torch.cat([x[k] for x in samples], dim=0)
                elif isinstance(samples[0][k], torch.Tensor):
                    res[k] = torch.cat([x[k] for x in samples], dim=0).permute(0, 3, 1, 2)
                else:
                    res[k] = [x[k] for x in samples]
            return res

        def data_process(example):
            local_views = [k.split(":")[0] for k in example.keys() if not k.startswith("__")]
            local_views = sorted(set(local_views))
            if not local_views:
                raise ValueError("Encountered a WebDataset shard with no decodable local views")

            if self.shuffle:
                if len(local_views) >= self.batch_size:
                    batch_views = random.sample(local_views, self.batch_size)
                else:
                    batch_views = local_views.copy()
                    batch_views.extend(random.choices(local_views, k=self.batch_size - len(local_views)))
                    random.shuffle(batch_views)
            else:
                batch_views = local_views[:self.batch_size]

            dataset_id = int(batch_views[0].split("/")[0][4:])
            batch_data = []

            for view in batch_views:
                view_data = {}
                for k in self.input_channels:
                    data = example[f"{view}:{k}"]
                    decoded_image = imageio.imread(io.BytesIO(data))
                    decoded_image = decoded_image[..., None] if decoded_image.ndim == 2 else decoded_image
                    view_data[k] = torch.from_numpy(decoded_image.astype(np.float32) / 255.0).unsqueeze(0) * 2 - 1

                normal_key = [k for k in view_data.keys() if "normal" in k][0]
                normal = view_data[normal_key]
                mask = (torch.abs(normal) < 0.01).all(dim=-1, keepdim=True).float()
                if mask.sum() == 0:
                    mask = (normal == -1).all(dim=-1, keepdim=True).float()
                for k in view_data.keys():
                    view_data[k] = view_data[k] * (1.0 - mask) + (torch.ones_like(view_data[k]) * -1.0) * mask
                view_data["background_alpha"] = (1.0 - mask) * 2.0 - 1.0

                batch_data.append(view_data)

            sample = custom_collate(batch_data)
            sample["dataset_id"] = dataset_id
            sample["local_views"] = batch_views
            if self.view_dir is not None:
                sample["view"] = self.get_view(dataset_id).repeat(len(batch_views), 1, 1, 1)

            return sample

        if self.shuffle:
            tarfiles = wds.ResampledShards(data_dir)
        else:
            tarfiles = wds.SimpleShardList(data_dir)

        self.pipeline = [
            tarfiles,
            tarfile_to_samples,
            wds.map(data_process),
        ]

        self._train_dataset = None
        self._train_dataloader = None

    def get_view(self, idx):
        _view_path = os.path.join(self.view_dir, "view%04d.basecolor.png" % idx)
        return torchvision.io.read_image(_view_path).unsqueeze(0) / 255

    def get_dataset(self):
        return _ClosableIterableDataset(
            lambda: wds.DataPipeline(*self.pipeline),
            length=self.num_train_examples,
        )

    def get_dataloader(self, global_batch_size, num_workers, **kwargs):
        num_worker_batches = math.ceil(self.num_train_examples / (global_batch_size * max(1, num_workers)))
        num_batches = num_worker_batches * max(1, num_workers)
        num_samples = num_batches * global_batch_size

        self._dataset = _ClosableIterableDataset(
            lambda: wds.DataPipeline(*self.pipeline).with_epoch(num_worker_batches),
            length=num_samples,
        )
        self._dataloader = wds.WebLoader(
            self._dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs,
        ).with_length(num_batches)
        self._dataloader.num_batches = num_batches
        self._dataloader.num_samples = num_samples

        return self._dataloader

    def __len__(self):
        return self.num_train_examples

    @property
    def dataset(self):
        return self._dataset

    @property
    def dataloader(self):
        return self._dataloader


class EnsembleMultiviewLocalMeshRendersWebdataset:
    """Combine multiple object-level WDS datasets into one source-aware iterable."""

    def __init__(
        self,
        sources: Sequence[EnsembleWebdatasetSourceSpec],
        batch_size: int,
        input_channels=["geo_camera_normals", "relative_positions", "albedo"],
        exclude_labels=None,
        shuffle: bool = True,
        mixing_mode: str = "round_robin",
        include_metadata: bool = True,
        **kwargs,
    ) -> None:
        if not sources:
            raise ValueError("EnsembleMultiviewLocalMeshRendersWebdataset requires at least one source")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mixing_mode = mixing_mode
        self.include_metadata = include_metadata
        self.input_channels = list(input_channels)
        self.exclude_labels = exclude_labels
        self.kwargs = dict(kwargs)

        self.sources: list[EnsembleWebdatasetResolvedSource] = []
        self.source_map: dict[str, EnsembleWebdatasetResolvedSource] = {}
        self.num_train_examples = 0

        for source_index, spec in enumerate(sources):
            if spec.name in self.source_map:
                raise ValueError(f"Duplicate ensemble source name: {spec.name}")
            if spec.weight <= 0:
                raise ValueError(f"Ensemble source {spec.name} has non-positive weight: {spec.weight}")

            dataset = MultiviewLocalMeshRendersWebdataset(
                data_dir=spec.data_dir,
                num_views=spec.num_views,
                num_local_views=spec.num_local_views,
                batch_size=batch_size,
                view_dir=spec.view_dir,
                input_channels=list(self.input_channels),
                exclude_labels=exclude_labels,
                shuffle=shuffle,
                **self.kwargs,
            )
            resolved = EnsembleWebdatasetResolvedSource(
                source_index=source_index,
                spec=spec,
                dataset=dataset,
            )
            self.sources.append(resolved)
            self.source_map[spec.name] = resolved
            self.num_train_examples += resolved.num_train_examples

        self._dataset = None
        self._dataloader = None

    def get_source_dataset(self, source_name: str) -> MultiviewLocalMeshRendersWebdataset:
        return self.source_map[source_name].dataset

    def get_source_datasets(self) -> dict[str, MultiviewLocalMeshRendersWebdataset]:
        return {
            source.name: source.dataset
            for source in self.sources
        }

    def get_dataset(self, max_samples: Optional[int] = None):
        return _EnsembleWebdatasetIterable(
            self.sources,
            mixing_mode=self.mixing_mode,
            max_samples=max_samples,
            shuffle_sources=self.shuffle,
            include_metadata=self.include_metadata,
        )

    def get_dataloader(self, global_batch_size, num_workers, **kwargs):
        num_worker_batches = math.ceil(self.num_train_examples / (global_batch_size * max(1, num_workers)))
        num_batches = num_worker_batches * max(1, num_workers)
        num_samples = num_batches * global_batch_size

        self._dataset = self.get_dataset(max_samples=num_samples)
        self._dataloader = torch.utils.data.DataLoader(
            self._dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs,
        )
        self._dataloader.num_batches = num_batches
        self._dataloader.num_samples = num_samples

        return self._dataloader

    def __len__(self):
        return self.num_train_examples

    @property
    def dataset(self):
        return self._dataset

    @property
    def dataloader(self):
        return self._dataloader
