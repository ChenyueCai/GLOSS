#!/usr/bin/env bash
# Automatic texture completion from one reference view (stage 1 guidance, then stage 2). Needs a CUDA GPU.
#
#   bash scripts/run_completion.sh croissant 48      # <mesh> <view id> from the example data
#   NUM_CAMERAS=300 MAX_ANGLE=60 bash scripts/run_completion.sh cabbage 33
#
# Inputs:  $GLOSS_DATA_DIR/{meshes,single_views,textures}/<mesh>/, checkpoint from ckpts/<mesh>/ or Hugging Face.
# Output:  expr/completion/<mesh>/view<id>/all-fast_*/texture/uv_final.png
# Env: NUM_CAMERAS (600), MAX_NUM_CAMERA (350), MAX_ANGLE, VIEW_MARGIN, SEED (0), EXPR_DIR. Details: docs/completion.md
set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${_here}/common.sh" ]] || _here="${GLOSS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}/scripts"
source "${_here}/common.sh"

[[ $# -eq 2 ]] || { sed -n 2,10p "${BASH_SOURCE[0]}"; exit 1; }
MESH="$1"; VIEW="$((10#$2))"
EXPR_DIR="${EXPR_DIR:-${GLOSS_REPO_ROOT}/expr/completion}"
NUM_CAMERAS="${NUM_CAMERAS:-600}"
view_tag="view$(printf '%04d' "${VIEW}")"
for f in "meshes/${MESH}/scene.gltf" "single_views/${MESH}/${view_tag}.basecolor.png" "textures/${MESH}/${view_tag}.png"; do
    [[ -f "${GLOSS_DATA_DIR}/${f}" ]] || die "missing ${GLOSS_DATA_DIR}/${f}"
done

shared=(--object_name "${MESH}" --view_id "${VIEW}"
        --data_dir "${GLOSS_DATA_DIR}" --expr_dir "${EXPR_DIR}"
        --cond_view_root "${GLOSS_DATA_DIR}/single_views" --texture_root "${GLOSS_DATA_DIR}/textures")
[[ -n "${MAX_ANGLE:-}" ]] && shared+=(--max_angle "${MAX_ANGLE}")
[[ -n "${VIEW_MARGIN:-}" ]] && shared+=(--view_margin "${VIEW_MARGIN}")

log "stage 1/2: guidance texture (${NUM_CAMERAS} cameras)"
gloss_python scripts/completion/complete_stage_1.py "${shared[@]}" --num_cameras "${NUM_CAMERAS}"
log "stage 2/2: final texture"
gloss_python scripts/completion/complete_stage_2.py "${shared[@]}" --guidance_num_cameras "${NUM_CAMERAS}" \
    --max_num_camera "${MAX_NUM_CAMERA:-350}" --seed "${SEED:-0}"

result="$(find "${EXPR_DIR}/${MESH}/${view_tag}" -maxdepth 3 -path '*/all-fast_*/texture/uv_final.png' -newer "${GLOSS_DATA_DIR}/textures/${MESH}/${view_tag}.png" 2>/dev/null | head -1 || true)"
[[ -n "${result}" ]] || die "stage 2 finished but wrote no all-fast_*/texture/uv_final.png under ${EXPR_DIR}/${MESH}/${view_tag}"
log "done: ${result}"
