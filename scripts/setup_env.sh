#!/usr/bin/env bash
# Create the conda environment(s) for GLOSS.
#
#   bash scripts/setup_env.sh             # gloss env: completion, training, Blender backend
#   bash scripts/setup_env.sh --datagen   # also the diff-render and invsr envs for data generation
#
# Details: docs/setup.md
set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${_here}/common.sh" ]] || _here="${GLOSS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}/scripts"
source "${_here}/common.sh"

WITH_DATAGEN=0
for arg in "$@"; do
    case "${arg}" in
        --datagen) WITH_DATAGEN=1 ;;
        -h|--help) sed -n 2,8p "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown argument: ${arg}" ;;
    esac
done
require_conda
cd "${GLOSS_REPO_ROOT}"

create_env() {  # create_env <name> <python-version>
    if env_exists "$1"; then log "conda env '$1' exists, reusing it."
    else log "creating conda env '$1' (python $2)"; conda create -y -q -n "$1" "python=$2"; fi
}
pip_in() { local env="$1"; shift; in_env "${env}" python -m pip install -q "$@"; }

# --- submodules -------------------------------------------------------------
submodules=(thirdparty/cmmd-pytorch)
(( WITH_DATAGEN )) && submodules+=(thirdparty/diffusion-renderer thirdparty/InvSR)
log "fetching submodules: ${submodules[*]}"
git submodule update --init "${submodules[@]}"
bash thirdparty/apply_patches.sh

# --- gloss env ----------------------------------------------------------------
# Tested with python 3.9, torch 2.1.1 + CUDA 11.8, kaolin 0.17.0.
create_env "${GLOSS_ENV}" 3.9
pip_in "${GLOSS_ENV}" torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
pip_in "${GLOSS_ENV}" kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.1_cu118.html
pip_in "${GLOSS_ENV}" -r requirements.txt
pip_in "${GLOSS_ENV}" -e .

log "checking the gloss env"
in_env "${GLOSS_ENV}" python - <<'PY'
import torch, kaolin, kaolin.render.easy_render, diffusers
import gloss, gloss.inpaint.models, gloss.utils.paths, gloss_interactive.server
print(f"torch {torch.__version__} (cuda available: {torch.cuda.is_available()}), "
      f"kaolin {kaolin.__version__}, diffusers {diffusers.__version__}")
PY

# --- datagen envs -------------------------------------------------------------
if (( WITH_DATAGEN )); then
    # Step 5: intrinsic decomposition with diffusion-renderer.
    create_env "${DIFFRENDER_ENV}" 3.10
    pip_in "${DIFFRENDER_ENV}" torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
    # nvdiffrast is only needed by diffusion-renderer's forward renderer, and building it needs a
    # CUDA compiler; step 5 runs the inverse model, so install everything else.
    reqs="$(mktemp)"; grep -v nvdiffrast thirdparty/diffusion-renderer/requirements.txt > "${reqs}"
    pip_in "${DIFFRENDER_ENV}" -r "${reqs}"; rm -f "${reqs}"
    if [[ ! -d thirdparty/diffusion-renderer/checkpoints/diffusion_renderer-inverse-svd ]]; then
        log "downloading diffusion-renderer inverse weights"
        (cd thirdparty/diffusion-renderer && in_env "${DIFFRENDER_ENV}" python utils/download_weights.py --repo_id nexuslrf/diffusion_renderer-inverse-svd)
    fi

    # Step 6: super-resolution with InvSR.
    create_env "${INVSR_ENV}" 3.10
    pip_in "${INVSR_ENV}" torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    pip_in "${INVSR_ENV}" -U xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
    (cd thirdparty/InvSR && pip_in "${INVSR_ENV}" -e ".[torch]" && pip_in "${INVSR_ENV}" -r requirements.txt)

    log "checking the datagen envs"
    in_env "${DIFFRENDER_ENV}" python -c "import torch, diffusers; print('diff-render: torch', torch.__version__)"
    in_env "${INVSR_ENV}" python -c "import torch, xformers; print('invsr: torch', torch.__version__)"
fi

log "done. Next: bash scripts/download_example_data.sh"
