#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Place a generated dataset and its trained model where the Blender backend, completion, and
# evaluation look for meshes (the example-data layout under $GLOSS_DATA_DIR).
#
#   bash scripts/import_generated.sh data/interactive/generated/croissant                 # newest trained checkpoint
#   bash scripts/import_generated.sh data/interactive/generated/croissant data/interactive/ckpts/croissant/model.safetensors
#   MESH_NAME=my_croissant bash scripts/import_generated.sh data/interactive/generated/croissant
#
# Creates, for mesh <name> (default: the folder name):
#   meshes/<name>       -> the generated mesh folder
#   single_views/<name>/ viewNNNN.basecolor.png   super-resolved reference views
#   textures/<name>/    viewNNNN.png             their backprojected textures
#   metas/<name>/       viewNNNN.yml             their cameras
#   ckpts/<name>/       model.safetensors        converted from the training checkpoint
# Files are relative symlinks into the generated folder, except the checkpoint. Views excluded
# during review (exclude_from_training_indices) are skipped. Details: docs/training.md
set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${_here}/common.sh" ]] || _here="${GLOSS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}/scripts"
source "${_here}/common.sh"

[[ $# -ge 1 ]] || { sed -n 2,17p "${BASH_SOURCE[0]}"; exit 1; }
GEN="$(cd "$1" && pwd)" || die "generated folder not found: $1"
CKPT="${2:-}"
SV_SUBDIR="${SV_SUBDIR:-civitai2.0}"
MESH="${MESH_NAME:-$(basename "${GEN}")}"
SV="${GEN}/single_view/${SV_SUBDIR}"
D="$(mkdir -p "${GLOSS_DATA_DIR}" && cd "${GLOSS_DATA_DIR}" && pwd)"
[[ -f "${GEN}/mesh/scene.gltf" ]] || die "${GEN}/mesh/scene.gltf not found; is this a generate_data.sh output?"
for sub in gen_view_super textures_sr meta; do [[ -d "${SV}/${sub}" ]] || die "missing ${SV}/${sub}; did all 8 datagen steps finish?"; done
for t in "${D}/meshes/${MESH}" "${D}/single_views/${MESH}" "${D}/textures/${MESH}" "${D}/metas/${MESH}" "${D}/ckpts/${MESH}"; do
    if [[ -e "${t}" || -L "${t}" ]] && [[ "${FORCE:-0}" != 1 ]]; then
        die "${t} already exists; pick another MESH_NAME, or set FORCE=1 to replace it"
    fi
done

# Views with an image, a texture, and a camera, minus the ones excluded in review.
mapfile -t VIEWS < <(SV="${SV}" gloss_python - <<'PY'
import json, os, re
sv = os.environ["SV"]
ids = lambda d, suf: {m.group(1) for f in os.listdir(f"{sv}/{d}") if (m := re.fullmatch(r"view(\d{4})" + re.escape(suf), f))}
ok = ids("gen_view_super", ".basecolor.png") & ids("textures_sr", ".png") & ids("meta", ".yml")
meta = f"{sv}/meta.json"
excluded = set()
if os.path.isfile(meta):
    excluded = {f"{int(i):04d}" for i in json.load(open(meta)).get("exclude_from_training_indices") or []}
print("\n".join(sorted(ok - excluded)))
PY
)
(( ${#VIEWS[@]} > 0 )) || die "no view in ${SV} has an image, a texture, and a camera"

link() {  # link <target> <link path>: relative symlink
    mkdir -p "$(dirname "$2")"; rm -f "$2"
    ln -s "$(realpath --relative-to="$(dirname "$2")" "$1")" "$2"
}
rm -rf "${D}/single_views/${MESH}" "${D}/textures/${MESH}" "${D}/metas/${MESH}"
link "$(realpath "${GEN}/mesh")" "${D}/meshes/${MESH}"
for v in "${VIEWS[@]}"; do
    link "${SV}/gen_view_super/view${v}.basecolor.png" "${D}/single_views/${MESH}/view${v}.basecolor.png"
    link "${SV}/textures_sr/view${v}.png" "${D}/textures/${MESH}/view${v}.png"
    link "${SV}/meta/view${v}.yml" "${D}/metas/${MESH}/view${v}.yml"
done
log "linked mesh and ${#VIEWS[@]} views as '${MESH}' under ${D}"

# Checkpoint: explicit, or the newest one train.sh wrote for this dataset.
if [[ -z "${CKPT}" ]]; then
    CKPT="$(ls -t "${D}"/experiments/*/"$(basename "${GEN}")"-*/checkpts/chkpt_*.{ckpt,safetensors} 2>/dev/null | head -1 || true)"
fi
if [[ -n "${CKPT}" ]]; then
    [[ -f "${CKPT}" ]] || die "checkpoint not found: ${CKPT}"
    log "exporting ${CKPT}"
    gloss_python scripts/export_checkpoint.py "$(realpath "${CKPT}")" "${D}/ckpts/${MESH}/model.safetensors"
else
    log "no trained checkpoint found; train with: bash scripts/train.sh ${GEN}"
fi

first="$((10#${VIEWS[0]}))"
log "done. Use it with:"
log "  bash scripts/run_completion.sh ${MESH} ${first}"
log "  bash scripts/evaluate.sh ${MESH} ${first}"
log "  bash scripts/run_backend.sh            (the Blender add-on now lists '${MESH}')"
