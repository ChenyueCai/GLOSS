# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Locate GLOSS data, base model weights, and checkpoints without machine-specific paths.

A release install needs at most one environment variable, ``GLOSS_DATA_DIR``, pointing
at the example-data bundle. Everything else is derived from it or can be overridden:

    GLOSS_DATA_DIR         root of the data bundle (meshes/, single_views/, metas/, textures/, ...);
                           defaults to <repo>/data/interactive when that folder exists
    GLOSS_MODEL_DIR        local copy of the SD 2.1 unCLIP base weights; when unset,
                           diffusers pulls them from GLOSS_BASE_MODEL_REPO on the Hub
    GLOSS_BASE_MODEL_REPO  Hugging Face repo with the base weights
    GLOSS_HF_REPO          Hugging Face model repo that hosts the per-mesh checkpoints
    GLOSS_INTERACTIVE_DIR  where the interactive backend writes everything it generates:
                           sessions/, brushes/, caches/, logs/ (default: <GLOSS_DATA_DIR> itself)
    GLOSS_SESSION_DIR      override for just the sessions folder
                           (default: <GLOSS_INTERACTIVE_DIR>/sessions)

"""

import os
from pathlib import Path
from typing import Iterable, Optional, Union

PathLike = Union[str, os.PathLike]

DATA_DIR_ENV = "GLOSS_DATA_DIR"
MODEL_DIR_ENV = "GLOSS_MODEL_DIR"
HF_REPO_ENV = "GLOSS_HF_REPO"
SESSION_DIR_ENV = "GLOSS_SESSION_DIR"
BASE_MODEL_REPO_ENV = "GLOSS_BASE_MODEL_REPO"
INTERACTIVE_DIR_ENV = "GLOSS_INTERACTIVE_DIR"


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a GLOSS_* setting from the environment."""
    return os.environ.get(name) or default

DEFAULT_HF_REPO = "chenyuec/gloss-checkpoints"
# stabilityai/stable-diffusion-2-1-unclip was removed from the Hub, so the release hosts a copy
# (CreativeML Open RAIL++-M, license shipped alongside the weights).
SD21_UNCLIP_MODEL_ID = "chenyuec/gloss-sd21-unclip"
CHECKPOINT_FILENAME = "model.safetensors"      # released checkpoints (fp16 UNet weights)
LEGACY_CHECKPOINT_FILENAMES = ("chkpt_80000.ckpt",)  # training output; still accepted locally

REPO_ROOT = Path(__file__).resolve().parents[2]
# Where scripts/download_example_data.sh puts the bundle; used when GLOSS_DATA_DIR is unset.
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "interactive"
DEFAULT_SINGLE_VIEW_CONFIGS = REPO_ROOT / "configs" / "single_view_configs.yaml"
# Keys of the interactive backend config that hold paths. Relative values of the
# input keys resolve against GLOSS_DATA_DIR; the output keys (folders the backend
# writes to) resolve against GLOSS_INTERACTIVE_DIR. See ``resolve_config_paths``.
BACKEND_INPUT_PATH_KEYS = (
    "mesh_folder",
    "inpaint_model_dir",
    "single_views_folder",
    "single_views_cam_folder",
    "single_views_texture_folder",
    "brush_presets_folder",
)
BACKEND_OUTPUT_PATH_KEYS = (
    "brushes_folder",
    "brushes_meta_folder",
    "cache_folder",
)
BACKEND_PATH_KEYS = BACKEND_INPUT_PATH_KEYS + BACKEND_OUTPUT_PATH_KEYS


def get_data_dir(required: bool = True) -> Optional[Path]:
    """Return ``GLOSS_DATA_DIR`` as a Path, or raise a clear error when unset."""
    value = env(DATA_DIR_ENV)
    if value:
        return Path(value).expanduser()
    if DEFAULT_DATA_DIR.is_dir():
        return DEFAULT_DATA_DIR
    if required:
        raise EnvironmentError(
            f"{DATA_DIR_ENV} is not set. Point it at the GLOSS data bundle (the folder "
            "containing meshes/, single_views/, metas/, ...) or pass an explicit path."
        )
    return None


def resolve_path(value: Optional[PathLike], default_relative: Optional[str] = None) -> Path:
    """Absolute paths pass through; relative ones are joined onto ``GLOSS_DATA_DIR``."""
    if value is None:
        if default_relative is None:
            raise ValueError("resolve_path needs a value or a default_relative")
        value = default_relative
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return get_data_dir() / path


def get_interactive_dir() -> Path:
    """Root for everything the interactive backend generates.

    Defaults to ``GLOSS_DATA_DIR`` itself: sessions/, brushes/, caches/ and logs/
    sit next to the example data (meshes/, single_views/, ...) they were made from.
    """
    value = env(INTERACTIVE_DIR_ENV)
    return Path(value).expanduser() if value else get_data_dir()


def resolve_config_paths(config: dict, keys: Iterable[str] = BACKEND_PATH_KEYS) -> dict:
    """Return a copy of ``config`` with relative path entries made absolute.

    Input folders resolve under GLOSS_DATA_DIR, output folders under
    GLOSS_INTERACTIVE_DIR. Absolute values are used as-is.
    """
    resolved = dict(config)
    for key in keys:
        if resolved.get(key):
            if key in BACKEND_OUTPUT_PATH_KEYS:
                path = Path(resolved[key]).expanduser()
                resolved[key] = str(path if path.is_absolute() else get_interactive_dir() / path)
            else:
                resolved[key] = str(resolve_path(resolved[key]))
    return resolved


def get_base_model_dir(default: str = SD21_UNCLIP_MODEL_ID) -> str:
    """Local directory of the base diffusion weights, or a Hub model id."""
    return env(MODEL_DIR_ENV) or env(BASE_MODEL_REPO_ENV) or default


def get_log_dir() -> Path:
    """Fallback for per-inference logs (<GLOSS_INTERACTIVE_DIR>/logs).

    The backend normally routes them into ``sessions/<name>/inference-log``; this
    catches brushes used before (or without) a session instead of the process CWD.
    """
    return get_interactive_dir() / "logs"


def get_session_dir() -> Path:
    value = env(SESSION_DIR_ENV)
    return Path(value).expanduser() if value else get_interactive_dir() / "sessions"


def resolve_checkpoint(object_name: str,
                       ckpt_dir: Optional[PathLike] = None,
                       explicit: Optional[PathLike] = None,
                       filename: Optional[str] = None) -> str:
    """Find the per-mesh inpainting checkpoint for ``object_name``.

    Order: an explicit path, then ``<ckpt_dir>/<object_name>/<filename>`` (with
    ``ckpt_dir`` defaulting to ``<GLOSS_DATA_DIR>/ckpts`` and ``filename`` trying
    ``model.safetensors`` then ``chkpt_80000.ckpt``), then a download from the
    Hugging Face repo named by ``GLOSS_HF_REPO``.
    """
    if explicit:
        return str(explicit)
    if ckpt_dir is None:
        data_dir = get_data_dir(required=False)
        ckpt_dir = data_dir / "ckpts" if data_dir else None
    names = (filename,) if filename else (CHECKPOINT_FILENAME,) + LEGACY_CHECKPOINT_FILENAMES
    if ckpt_dir is not None:
        for name in names:
            local = Path(ckpt_dir) / object_name / name
            if local.is_file():
                return str(local)
    repo = env(HF_REPO_ENV) or DEFAULT_HF_REPO
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo, filename=f"ckpts/{object_name}/{names[0]}")
    except Exception as exc:
        raise FileNotFoundError(
            f"No checkpoint for '{object_name}': not found under {ckpt_dir} and the download "
            f"from Hugging Face repo '{repo}' failed ({exc}). Pass an explicit checkpoint "
            f"path or set {HF_REPO_ENV}."
        ) from exc


def relativize(path: Optional[PathLike], base: Optional[PathLike]) -> Optional[str]:
    """Path written to a saved file: relative to ``base`` when it lives under it.

    Paths outside ``base`` (or with no base) are returned unchanged, so a
    user-supplied absolute location still round-trips.
    """
    if path is None or base is None:
        return None if path is None else str(path)
    p, b = Path(path).expanduser(), Path(base).expanduser()
    try:
        return str(p.resolve().relative_to(b.resolve()))
    except ValueError:
        return str(path)


def rebase(path: Optional[PathLike], base: Optional[PathLike]) -> Optional[str]:
    """Inverse of :func:`relativize`: relative paths are joined onto ``base``.

    Absolute paths (older session and brush files) pass through unchanged.
    """
    if path is None:
        return None
    p = Path(path).expanduser()
    if p.is_absolute() or base is None:
        return str(p)
    return str(Path(base).expanduser() / p)
