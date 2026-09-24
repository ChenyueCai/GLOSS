#!/usr/bin/env bash
# Train the texture-completion model on a dataset made by scripts/generate_data.sh.
# Uses every visible GPU (one process per GPU).
#
#   bash scripts/train.sh data/interactive/generated/croissant
#   ITERS=20000 BATCH_SIZE=8 bash scripts/train.sh data/interactive/generated/croissant
#   sbatch --gpus-per-node=4 --cpus-per-task=16 --mem=64G --time=24:00:00 scripts/train.sh data/interactive/generated/croissant
#
# Checkpoints: $GLOSS_DATA_DIR/experiments/<group>/<name>-<objective>/. Details: docs/training.md
set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${_here}/common.sh" ]] || _here="${GLOSS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}/scripts"
source "${_here}/common.sh"

[[ $# -ge 1 ]] || { sed -n 2,11p "${BASH_SOURCE[0]}"; exit 1; }
MESH_OUT="$(cd "$1" && pwd)" || die "dataset folder not found: $1"
DATA_ROOT="$(cd "${GLOSS_DATA_DIR}" && pwd)"
[[ "${MESH_OUT}" == "${DATA_ROOT}"/* ]] || die "$1 must live under GLOSS_DATA_DIR (${DATA_ROOT})"
REL="${MESH_OUT#"${DATA_ROOT}"/}"
SV_SUBDIR="${SV_SUBDIR:-civitai2.0}"

shopt -s nullglob
wds_dirs=("${MESH_OUT}"/multi_view/*-wds)
shopt -u nullglob
(( ${#wds_dirs[@]} > 0 )) || die "no WebDataset shards under ${MESH_OUT}/multi_view/*-wds; run scripts/generate_data.sh first"
WDS_REL="${wds_dirs[0]#"${DATA_ROOT}"/}"

if [[ -z "${GPUS:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then GPUS="$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | grep -c .)"
    else GPUS="$(nvidia-smi -L 2>/dev/null | grep -c GPU || true)"; fi
fi
(( GPUS > 0 )) || die "no GPU visible"

export GLOBAL_ROOT_DIR="${DATA_ROOT}"
export DATA_MESH="${REL}/mesh/scene.gltf"
export DATA_BASE_DIR="${WDS_REL}"
export SINGLE_VIEW_DIR="${REL}/single_view/${SV_SUBDIR}/gen_view_super"
export EXP_NAME="${EXP_NAME:-$(basename "${MESH_OUT}")}"
export EXP_GROUP="${EXP_GROUP:-gloss}"
export TRAIN_TOTAL_ITERATIONS="${ITERS:-${TRAIN_TOTAL_ITERATIONS:-80000}}"
export TRAIN_BATCH_SIZE="${BATCH_SIZE:-${TRAIN_BATCH_SIZE:-8}}"
export GPUS_PER_NODE="${GPUS}"
export GLOSS_ENV WANDB_MODE="${WANDB_MODE:-offline}"
log "training on ${DATA_BASE_DIR} with ${GPUS} GPU(s), ${TRAIN_TOTAL_ITERATIONS} iterations"
bash "${GLOSS_REPO_ROOT}/scripts/train/launch.sh"
