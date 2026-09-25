#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Generate a training dataset for one textured mesh (8-stage pipeline). Needs a CUDA GPU,
# the datagen envs (bash scripts/setup_env.sh --datagen), and OPENAI_API_KEY for prompting.
#
#   bash scripts/generate_data.sh data/interactive/meshes/croissant                     # folder holding scene.gltf
#   bash scripts/generate_data.sh data/interactive/meshes/croissant --condition_mode debug --num_prompts 10
#
# Output: $GLOSS_DATA_DIR/generated/<mesh>/ (views, textures, WebDataset shards under multi_view/).
# Extra flags go to scripts/datagen/pipeline.py. Env: MESH_NAME, EXPR_TAG. Details: docs/data-generation-pipeline.md
set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${_here}/common.sh" ]] || _here="${GLOSS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}/scripts"
source "${_here}/common.sh"

[[ $# -ge 1 ]] || { sed -n 2,10p "${BASH_SOURCE[0]}"; exit 1; }
MESH_DIR="$(cd "$1" && pwd)" || die "mesh folder not found: $1"
shift
[[ -f "${MESH_DIR}/scene.gltf" ]] || die "${MESH_DIR} has no scene.gltf"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set (used to caption views and write prompts)}"
MESH_NAME="${MESH_NAME:-$(basename "${MESH_DIR}")}"
EXPR_TAG="${EXPR_TAG:-generated}"

for e in "${GLOSS_ENV}" "${DIFFRENDER_ENV}" "${INVSR_ENV}"; do
    env_exists "${e}" || die "conda env '${e}' not found; run: bash scripts/setup_env.sh --datagen"
done
ensure_submodules thirdparty/diffusion-renderer thirdparty/InvSR
DR="${GLOSS_REPO_ROOT}/thirdparty/diffusion-renderer"
if [[ ! -d "${DR}/checkpoints/diffusion_renderer-inverse-svd" ]]; then
    log "downloading diffusion-renderer inverse weights"
    (cd "${DR}" && in_env "${DIFFRENDER_ENV}" python utils/download_weights.py --repo_id nexuslrf/diffusion_renderer-inverse-svd)
fi

# Pipeline layout: <data_dir>/<expr_tag>/<mesh_name>/mesh/scene.gltf
OUT="${GLOSS_DATA_DIR}/${EXPR_TAG}/${MESH_NAME}"
mkdir -p "${OUT}"
[[ -e "${OUT}/mesh" ]] || ln -s "${MESH_DIR}" "${OUT}/mesh"
log "generating data for '${MESH_NAME}' into ${OUT}"

gloss_python scripts/datagen/pipeline.py \
    --data_dir "${GLOSS_DATA_DIR}" --expr_tag "${EXPR_TAG}" --mesh_name "${MESH_NAME}" \
    --gloss_env "${GLOSS_ENV}" \
    --diffrender_dir "${GLOSS_REPO_ROOT}/thirdparty/diffusion-renderer" --diffrender_env "${DIFFRENDER_ENV}" \
    --invsr_dir "${GLOSS_REPO_ROOT}/thirdparty/InvSR" --invsr_env "${INVSR_ENV}" \
    "$@"
log "done. Train on it with: bash scripts/train.sh ${OUT}"
