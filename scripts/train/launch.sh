#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=train-diffusion-wds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=train-diffusion-wds-%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${GLOSS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}}"
cd "${REPO_ROOT}"
# Data/experiment root; defaults to the release data bundle.
GLOBAL_ROOT_DIR="${GLOBAL_ROOT_DIR:-${GLOSS_DATA_DIR:-}}"
EXP_NAME="${EXP_NAME:-}"
EXP_GROUP="${EXP_GROUP:-default}"
DATA_MESH="${DATA_MESH:-}"
DATA_BASE_DIR="${DATA_BASE_DIR:-}"
SINGLE_VIEW_DIR="${SINGLE_VIEW_DIR:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# Required inputs. Override them with environment variables before calling sbatch.
: "${GLOBAL_ROOT_DIR:?Set GLOBAL_ROOT_DIR to the data/experiment root.}"
: "${EXP_NAME:?Set EXP_NAME to the experiment run name.}"
: "${DATA_MESH:?Set DATA_MESH to the mesh path relative to GLOBAL_ROOT_DIR.}"
: "${DATA_BASE_DIR:?Set DATA_BASE_DIR to the WDS shard directory relative to GLOBAL_ROOT_DIR.}"
: "${SINGLE_VIEW_DIR:?Set SINGLE_VIEW_DIR to the single-view basecolor directory relative to GLOBAL_ROOT_DIR.}"

# Optional inputs with repo-friendly defaults.
GLOSS_ENV="${GLOSS_ENV:-gloss}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/multigpu.yaml}"
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-scripts/train/train_diffusion_wds.py}"
EXP_GROUP="${EXP_GROUP:-default}"
EXP_BASE_DIR="${EXP_BASE_DIR:-experiments}"
TRAIN_TOTAL_ITERATIONS="${TRAIN_TOTAL_ITERATIONS:-100000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_LOG_LPIPS="${TRAIN_LOG_LPIPS:-True}"
TRAIN_LOG_FID="${TRAIN_LOG_FID:-False}"
TRAIN_OBJECTIVE="${TRAIN_OBJECTIVE:-diffusion}" #-flow_matching
FLOW_SHIFT="${FLOW_SHIFT:-1.0}"
EXP_TAG="${EXP_TAG:-${TRAIN_OBJECTIVE}}"
EXP_EVAL_EVERY="${EXP_EVAL_EVERY:-1000}"
EXP_LOG_EVERY="${EXP_LOG_EVERY:-100}"
EXP_LOG_METRIC_EVERY="${EXP_LOG_METRIC_EVERY:-250}"
EXP_CHECKPOINT_EVERY="${EXP_CHECKPOINT_EVERY:-1000}"
EXP_PERSISTENT_CHECKPOINT_EVERY="${EXP_PERSISTENT_CHECKPOINT_EVERY:-5000}"
EXP_SAVE_LATEST_CHECKPOINT="${EXP_SAVE_LATEST_CHECKPOINT:-True}"
EXP_PERSISTENT_CHECKPOINT_STEPS="${EXP_PERSISTENT_CHECKPOINT_STEPS:-}"
DATA_NUM_VIEWS="${DATA_NUM_VIEWS:-auto}"
DATA_NUM_LOCAL_VIEWS="${DATA_NUM_LOCAL_VIEWS:-auto}"
DATA_TRAIN_VIEW_LIMIT="${DATA_TRAIN_VIEW_LIMIT:-}"
DATA_TRAIN_VIEW_SELECTION="${DATA_TRAIN_VIEW_SELECTION:-random}"
DATA_EVAL_VIEWS="${DATA_EVAL_VIEWS:-auto}"
DATA_CAMERA_DIST="${DATA_CAMERA_DIST:-0.25}"
DATA_FOV_MIN="${DATA_FOV_MIN:-0.4}"
DATA_FOV_MAX="${DATA_FOV_MAX:-0.8}"
DATA_SEED="${DATA_SEED:-0}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
JOB_LOCAL_ROOT="${JOB_LOCAL_ROOT:-${SLURM_TMPDIR:-/tmp/${USER}/slurm-${SLURM_JOB_ID:-$$}}}"
# Base SD 2.1 unCLIP weights. Leave both unset to let diffusers pull them from
# GLOSS_BASE_MODEL_REPO on the Hugging Face Hub; set SOURCE_MODEL_DIR to stage a local copy onto job scratch.
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:-}"
GLOSS_MODEL_DIR="${GLOSS_MODEL_DIR:-}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-cache-${USER}}"
PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/python-pycache-${USER}}"
TMPDIR="${TMPDIR:-${JOB_LOCAL_ROOT}/tmp}"
WANDB_DIR="${WANDB_DIR:-${JOB_LOCAL_ROOT}/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${JOB_LOCAL_ROOT}/wandb-cache}"
WANDB_DATA_DIR="${WANDB_DATA_DIR:-${JOB_LOCAL_ROOT}/wandb-data}"
WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-120}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}"
PERSISTENT_WEIGHTS_ONLY="${PERSISTENT_WEIGHTS_ONLY:-0}"
PERSISTENT_WEIGHTS_DTYPE="${PERSISTENT_WEIGHTS_DTYPE:-float16}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${JOB_LOCAL_ROOT}"
mkdir -p "${TMPDIR}"
mkdir -p "${MPLCONFIGDIR}"
mkdir -p "${PYTHONPYCACHEPREFIX}"
mkdir -p "${WANDB_DIR}"
mkdir -p "${WANDB_CACHE_DIR}"
mkdir -p "${WANDB_DATA_DIR}"

if [[ -n "${SOURCE_MODEL_DIR}" ]]; then
    GLOSS_MODEL_DIR="${GLOSS_MODEL_DIR:-${JOB_LOCAL_ROOT}/stable-diffusion-2-1-unclip}"
    if [[ ! -f "${GLOSS_MODEL_DIR}/image_encoder/model.safetensors" ]]; then
        echo "Staging diffusion model to local scratch: ${GLOSS_MODEL_DIR}"
        rsync -a --delete "${SOURCE_MODEL_DIR}/" "${GLOSS_MODEL_DIR}/"
    fi
fi

DATA_ROOT_DIR="${GLOBAL_ROOT_DIR}/${DATA_BASE_DIR}"
if [[ "${DATA_NUM_VIEWS}" == "auto" || "${DATA_NUM_LOCAL_VIEWS}" == "auto" || "${DATA_EVAL_VIEWS}" == "auto" ]]; then
    shopt -s nullglob
    shard_files=("${DATA_ROOT_DIR}"/view*.tar)
    shopt -u nullglob
    if (( ${#shard_files[@]} == 0 )); then
        echo "No WebDataset shards found under ${DATA_ROOT_DIR}" >&2
        exit 1
    fi

    declare -A shard_counts=()
    for shard_path in "${shard_files[@]}"; do
        shard_name="$(basename "${shard_path}")"
        if [[ "${shard_name}" =~ ^view([0-9]+)(-([0-9]+))?\.tar$ ]]; then
            view_id="${BASH_REMATCH[1]}"
            shard_counts["${view_id}"]=$(( ${shard_counts["${view_id}"]:-0} + 1 ))
        fi
    done

    available_view_count="${#shard_counts[@]}"
    max_shards_per_view=0
    for view_id in "${!shard_counts[@]}"; do
        if (( shard_counts["${view_id}"] > max_shards_per_view )); then
            max_shards_per_view="${shard_counts["${view_id}"]}"
        fi
    done

    if [[ "${DATA_NUM_VIEWS}" == "auto" ]]; then
        DATA_NUM_VIEWS="${available_view_count}"
    fi
    if [[ "${DATA_NUM_LOCAL_VIEWS}" == "auto" ]]; then
        DATA_NUM_LOCAL_VIEWS="$(( max_shards_per_view * 100 ))"
    fi
    if [[ "${DATA_EVAL_VIEWS}" == "auto" ]]; then
        if (( available_view_count <= 1 )); then
            DATA_EVAL_VIEWS=1
        elif (( available_view_count < 50 )); then
            DATA_EVAL_VIEWS="$(( available_view_count - 1 ))"
        else
            DATA_EVAL_VIEWS=50
        fi
    fi
fi

if command -v conda >/dev/null 2>&1; then
    # Cluster conda activate/deactivate hooks may reference unset vars.
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "${GLOSS_ENV}"
    set -u
else
    echo "conda not found in PATH" >&2
    exit 1
fi

export MPLCONFIGDIR
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX
export TMPDIR
if [[ -n "${WANDB_ENTITY}" ]]; then export WANDB_ENTITY; fi
export WANDB_MODE
export GLOSS_MODEL_DIR
export WANDB_DIR
export WANDB_CACHE_DIR
export WANDB_DATA_DIR
export WANDB__SERVICE_WAIT

echo "Launching multi-GPU training from ${REPO_ROOT}"
echo "  GLOBAL_ROOT_DIR=${GLOBAL_ROOT_DIR}"
echo "  EXP_NAME=${EXP_NAME}"
echo "  DATA_BASE_DIR=${DATA_BASE_DIR}"
echo "  DATA_ROOT_DIR=${DATA_ROOT_DIR}"
echo "  DATA_NUM_VIEWS=${DATA_NUM_VIEWS}"
echo "  DATA_NUM_LOCAL_VIEWS=${DATA_NUM_LOCAL_VIEWS}"
echo "  DATA_TRAIN_VIEW_LIMIT=${DATA_TRAIN_VIEW_LIMIT}"
echo "  DATA_TRAIN_VIEW_SELECTION=${DATA_TRAIN_VIEW_SELECTION}"
echo "  DATA_EVAL_VIEWS=${DATA_EVAL_VIEWS}"
echo "  SINGLE_VIEW_DIR=${SINGLE_VIEW_DIR}"
echo "  GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "  ACCELERATE_CONFIG=${ACCELERATE_CONFIG}"
echo "  TRAIN_ENTRYPOINT=${TRAIN_ENTRYPOINT}"
echo "  TRAIN_OBJECTIVE=${TRAIN_OBJECTIVE}"
echo "  EXP_TAG=${EXP_TAG}"
echo "  EXP_EVAL_EVERY=${EXP_EVAL_EVERY}"
echo "  EXP_LOG_EVERY=${EXP_LOG_EVERY}"
echo "  EXP_LOG_METRIC_EVERY=${EXP_LOG_METRIC_EVERY}"
echo "  EXP_CHECKPOINT_EVERY=${EXP_CHECKPOINT_EVERY}"
echo "  EXP_PERSISTENT_CHECKPOINT_EVERY=${EXP_PERSISTENT_CHECKPOINT_EVERY}"
echo "  EXP_SAVE_LATEST_CHECKPOINT=${EXP_SAVE_LATEST_CHECKPOINT}"
echo "  EXP_PERSISTENT_CHECKPOINT_STEPS=${EXP_PERSISTENT_CHECKPOINT_STEPS}"
echo "  WANDB_ENTITY=${WANDB_ENTITY}"
echo "  WANDB_MODE=${WANDB_MODE}"
echo "  SOURCE_MODEL_DIR=${SOURCE_MODEL_DIR}"
echo "  GLOSS_MODEL_DIR=${GLOSS_MODEL_DIR}"
echo "  JOB_LOCAL_ROOT=${JOB_LOCAL_ROOT}"
echo "  TMPDIR=${TMPDIR}"
echo "  WANDB_DIR=${WANDB_DIR}"
echo "  WANDB_CACHE_DIR=${WANDB_CACHE_DIR}"
echo "  WANDB_DATA_DIR=${WANDB_DATA_DIR}"
echo "  WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT}"
echo "  MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT}"
echo "  PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX}"
echo "  PERSISTENT_WEIGHTS_ONLY=${PERSISTENT_WEIGHTS_ONLY}"
echo "  PERSISTENT_WEIGHTS_DTYPE=${PERSISTENT_WEIGHTS_DTYPE}"

EXP_NAME_WITH_TAG="${EXP_NAME}"
if [[ -n "${EXP_TAG}" ]]; then
  EXP_NAME_WITH_TAG="${EXP_NAME}-${EXP_TAG}"
fi

COMMON_ARGS=(
  --global_root_dir="${GLOBAL_ROOT_DIR}"
  --exp.name="${EXP_NAME_WITH_TAG}"
  --exp.base_dir="${EXP_BASE_DIR}"
  --exp.group="${EXP_GROUP}"
  --exp.eval_every="${EXP_EVAL_EVERY}"
  --exp.log_every="${EXP_LOG_EVERY}"
  --exp.log_metric_every="${EXP_LOG_METRIC_EVERY}"
  --exp.checkpoint_every="${EXP_CHECKPOINT_EVERY}"
  --exp.persistent_checkpoint_every="${EXP_PERSISTENT_CHECKPOINT_EVERY}"
  --exp.save_latest_checkpoint="${EXP_SAVE_LATEST_CHECKPOINT}"
  --train.total_iterations="${TRAIN_TOTAL_ITERATIONS}"
  --train.batch_size="${TRAIN_BATCH_SIZE}"
  --train.log_lpips="${TRAIN_LOG_LPIPS}"
  --train.log_fid="${TRAIN_LOG_FID}"
  --train.objective="${TRAIN_OBJECTIVE}"
  --train.flow_shift="${FLOW_SHIFT}"
  --data.mesh="${DATA_MESH}"
  --data.num_views="${DATA_NUM_VIEWS}"
  --data.num_local_views="${DATA_NUM_LOCAL_VIEWS}"
  --data.data_base_dir="${DATA_BASE_DIR}"
  --data.single_view_dir="${SINGLE_VIEW_DIR}"
  --data.eval_views="${DATA_EVAL_VIEWS}"
  --data.camera_dist="${DATA_CAMERA_DIST}"
  --data.fov_min="${DATA_FOV_MIN}"
  --data.fov_max="${DATA_FOV_MAX}"
  --data.seed="${DATA_SEED}"
)

if [[ -n "${DATA_TRAIN_VIEW_LIMIT}" ]]; then
  COMMON_ARGS+=("--data.train_view_limit=${DATA_TRAIN_VIEW_LIMIT}")
fi

if [[ -n "${DATA_TRAIN_VIEW_SELECTION}" ]]; then
  COMMON_ARGS+=("--data.train_view_selection=${DATA_TRAIN_VIEW_SELECTION}")
fi

if [[ -n "${EXP_PERSISTENT_CHECKPOINT_STEPS}" ]]; then
  COMMON_ARGS+=("--exp.persistent_checkpoint_steps=${EXP_PERSISTENT_CHECKPOINT_STEPS}")
fi

if [[ "${PERSISTENT_WEIGHTS_ONLY}" == "1" ]]; then
  COMMON_ARGS+=(
    --persistent_weights_only
    --persistent_weights_dtype="${PERSISTENT_WEIGHTS_DTYPE}"
  )
fi

if [[ "${GPUS_PER_NODE}" == "1" ]]; then
  python "${TRAIN_ENTRYPOINT}" "${COMMON_ARGS[@]}" ${EXTRA_ARGS}
else
  accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --num_processes "${GPUS_PER_NODE}" \
    --num_machines 1 \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    "${TRAIN_ENTRYPOINT}" \
    "${COMMON_ARGS[@]}" \
    ${EXTRA_ARGS}
fi
