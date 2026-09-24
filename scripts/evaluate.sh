#!/usr/bin/env bash
# Patch-based evaluation of completed textures: LPIPS, FID, and CMMD. Needs a CUDA GPU.
#
#   bash scripts/evaluate.sh croissant 48            # after: bash scripts/run_completion.sh croissant 48
#   bash scripts/evaluate.sh croissant 48 53 60      # several views; FID and CMMD pool their patches
#
# Scores each completion against the reference texture of the same view, on random close-up
# patches rendered from the mesh (docs/completion.md#evaluation).
# Output: expr/completion/<mesh>/metrics/summary.json. Env: NUM_PATCHES (100), CAMERA_DIST (0.25), EXPR_DIR.
set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${_here}/common.sh" ]] || _here="${GLOSS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}/scripts"
source "${_here}/common.sh"

[[ $# -ge 2 ]] || { sed -n 2,10p "${BASH_SOURCE[0]}"; exit 1; }
MESH="$1"; shift
EXPR_DIR="${EXPR_DIR:-${GLOSS_REPO_ROOT}/expr/completion}"
NUM_PATCHES="${NUM_PATCHES:-100}"
CAMERA_DIST="${CAMERA_DIST:-0.25}"
MESH_PATH="${GLOSS_DATA_DIR}/meshes/${MESH}/scene.gltf"
REF_DIR="${GLOSS_DATA_DIR}/textures/${MESH}"
OUT="${EXPR_DIR}/${MESH}/metrics"
FLAT="${OUT}/completed"
[[ -f "${MESH_PATH}" ]] || die "missing ${MESH_PATH}"
if [[ ! -f "${GLOSS_REPO_ROOT}/thirdparty/cmmd-pytorch/main.py" ]]; then
    log "fetching thirdparty/cmmd-pytorch"
    git -C "${GLOSS_REPO_ROOT}" submodule update --init thirdparty/cmmd-pytorch
fi
bash "${GLOSS_REPO_ROOT}/thirdparty/apply_patches.sh" >/dev/null
rm -rf "${FLAT}" "${OUT}/cmmd" "${OUT}/lpips-fid.csv"
mkdir -p "${FLAT}"

# Gather the stage 2 result of each view into the flat viewNNNN.png layout the metric scripts read.
views=()
for v in "$@"; do
    v="$((10#${v}))"; tag="view$(printf '%04d' "${v}")"
    src="$(find "${EXPR_DIR}/${MESH}/${tag}" -maxdepth 3 -path '*/all-fast_*/texture/uv_final.png' 2>/dev/null | head -1 || true)"
    [[ -n "${src}" ]] || die "no completion for ${MESH} ${tag}; run: bash scripts/run_completion.sh ${MESH} ${v}"
    [[ -f "${REF_DIR}/${tag}.png" ]] || die "missing reference texture ${REF_DIR}/${tag}.png"
    ln -s "${src}" "${FLAT}/${tag}.png"
    views+=("${v}")
done
ids="$(IFS=,; echo "${views[*]}")"
log "evaluating ${MESH} views ${ids} with ${NUM_PATCHES} patches per view"

log "LPIPS + FID"
gloss_python scripts/metrics/test_completion_metrics.py \
    --expr-dir "${EXPR_DIR}" --object-name "${MESH}" --data-dir "${GLOSS_DATA_DIR}" \
    --mesh-path "${MESH_PATH}" --completed-dir "${FLAT}" --gt-texture-dir "${REF_DIR}" \
    --view-ids "${ids}" --num_samples "${NUM_PATCHES}" --camera_dist "${CAMERA_DIST}"

log "CMMD: rendering patches"
gloss_python scripts/metrics/gen_cmmd_patches.py \
    --mesh-path "${MESH_PATH}" --completed-dir "${FLAT}" --gt-texture-dir "${REF_DIR}" \
    --out-dir "${OUT}/cmmd" --view-ids "${ids}" --num-samples "${NUM_PATCHES}" --camera-dist "${CAMERA_DIST}"
log "CMMD: scoring"
(cd "${GLOSS_REPO_ROOT}/thirdparty/cmmd-pytorch" && \
    in_env "${GLOSS_ENV}" python main.py "${OUT}/cmmd/gt" "${OUT}/cmmd/pred" --batch_size=32) | tee "${OUT}/cmmd.txt"

MESH="${MESH}" OUT="${OUT}" IDS="${ids}" NUM_PATCHES="${NUM_PATCHES}" gloss_python - <<'PY'
import csv, json, os, re
out = os.environ["OUT"]
row = list(csv.DictReader(open(f"{out}/lpips-fid.csv")))[-1]
cmmd = float(re.search(r"CMMD value is:\s*([-\d.eE+]+)", open(f"{out}/cmmd.txt").read()).group(1))
summary = {"mesh": os.environ["MESH"], "views": [int(v) for v in os.environ["IDS"].split(",")],
           "patches_per_view": int(os.environ["NUM_PATCHES"]),
           "lpips": float(row["avg-lpips"]), "fid": float(row["final-fid"]), "cmmd": cmmd}
json.dump(summary, open(f"{out}/summary.json", "w"), indent=2)
print("\n  LPIPS  {lpips:.4f}   (lower is better; known region vs. reference)\n"
      "  FID    {fid:.2f}   (lower is better; completed region vs. reference patches)\n"
      "  CMMD   {cmmd:.4f}   (lower is better; same patches, CLIP embeddings)".format(**summary))
PY
log "done: ${OUT}/summary.json"
