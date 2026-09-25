#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Shared settings for the top-level scripts/*.sh wrappers. Source it, do not run it.
#
# Everything can be overridden from the environment:
#   GLOSS_DATA_DIR    data bundle root                    (default: <repo>/data/interactive)
#   GLOSS_ENV         conda env for GLOSS                 (default: gloss)
#   DIFFRENDER_ENV    conda env for datagen step 5        (default: diff-render)
#   INVSR_ENV         conda env for datagen step 6        (default: invsr)
#   GLOSS_HF_REPO     Hugging Face model repo with checkpoints
#   GLOSS_HF_DATA_REPO Hugging Face dataset repo with the example data
GLOSS_REPO_ROOT="${GLOSS_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export GLOSS_REPO_ROOT
export GLOSS_DATA_DIR="${GLOSS_DATA_DIR:-${GLOSS_REPO_ROOT}/data/interactive}"
GLOSS_ENV="${GLOSS_ENV:-gloss}"
DIFFRENDER_ENV="${DIFFRENDER_ENV:-diff-render}"
INVSR_ENV="${INVSR_ENV:-invsr}"
export GLOSS_HF_REPO="${GLOSS_HF_REPO:-chenyuec/gloss-checkpoints}"
export GLOSS_HF_DATA_REPO="${GLOSS_HF_DATA_REPO:-chenyuec/gloss-example-data}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl-cache-${USER:-gloss}}"
export PYTHONUNBUFFERED=1

log() { printf '[gloss] %s\n' "$*" >&2; }
die() { printf '[gloss] ERROR: %s\n' "$*" >&2; exit 1; }

require_conda() {
    command -v conda >/dev/null 2>&1 || die "conda not found on PATH. Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
}

# Run a command inside a conda env without requiring it to be activated.
#   in_env <env> <cmd...>
in_env() {
    local env="$1"; shift
    if [[ "${CONDA_DEFAULT_ENV:-}" == "${env}" ]]; then
        "$@"
    else
        require_conda
        conda run --no-capture-output -n "${env}" "$@"
    fi
}

env_exists() { command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$1"; }

# Fetch third-party submodules that are not checked out yet, then apply the local patches.
ensure_submodules() {
    local s missing=()
    for s in "$@"; do [[ -e "${GLOSS_REPO_ROOT}/${s}/.git" ]] || missing+=("${s}"); done
    if (( ${#missing[@]} )); then
        log "fetching ${missing[*]}"
        git -C "${GLOSS_REPO_ROOT}" submodule update --init "${missing[@]}"
    fi
    bash "${GLOSS_REPO_ROOT}/thirdparty/apply_patches.sh" >/dev/null
}

# Python in the gloss env, run from the repo root, with the repo root on PYTHONPATH
# (gloss_interactive has no __init__.py, so the editable install does not expose it).
gloss_python() { (cd "${GLOSS_REPO_ROOT}" && PYTHONPATH="${GLOSS_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" in_env "${GLOSS_ENV}" python "$@"); }
