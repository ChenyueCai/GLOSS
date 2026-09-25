#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Package the GLOSS Blender add-on (submodule blender/gloss-blender) and install it into Blender.
#
#   bash scripts/setup_blender.sh                          # finds Blender on PATH or in the usual install locations
#   BLENDER=/path/to/blender bash scripts/setup_blender.sh
#
# Writes build/gloss_blender.zip. The zip carries the add-on's data/config.yaml, which the add-on
# loads on its own, with its folders pointed at blender/gloss-blender/data/ and server_url at PORT.
# The example data itself is not packaged; fill it with `python blender/gloss-blender/data/download.py`.
# Without Blender it only builds the zip. Details: docs/blender.md
set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${_here}/common.sh" ]] || _here="${GLOSS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}/scripts"
source "${_here}/common.sh"

ADDON_DIR="${GLOSS_ADDON_DIR:-${GLOSS_REPO_ROOT}/blender/gloss-blender}"
PORT="${PORT:-10017}"
BUILD="${GLOSS_REPO_ROOT}/build"
if [[ ! -f "${ADDON_DIR}/__init__.py" && -z "${GLOSS_ADDON_DIR:-}" ]]; then
    log "fetching the add-on submodule blender/gloss-blender"
    git -C "${GLOSS_REPO_ROOT}" submodule update --init blender/gloss-blender
fi
[[ -f "${ADDON_DIR}/__init__.py" ]] || die "add-on not found at ${ADDON_DIR}.
  Get it with:  git submodule update --init blender/gloss-blender"
mkdir -p "${BUILD}"

PYTHON="$(command -v python3 || command -v python)" || die "python3 not found"
log "packaging ${ADDON_DIR} -> ${BUILD}/gloss_blender.zip"
"${PYTHON}" - "${ADDON_DIR}" "${BUILD}/gloss_blender.zip" "${PORT}" <<'PY'
import os, re, sys, zipfile
from pathlib import Path
src, out, port = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), sys.argv[3]
skip_dirs = {".git", "tests", "__pycache__", ".codex"}
skip_files = {".gitignore", ".DS_Store", "AGENTS.md"}
keep_data = {"config.yaml", "download.py"}  # the downloaded example data stays in the submodule
folder_keys = ("mesh_folder", "single_views_folder", "single_views_texture_folder", "brushes_folder")

def packaged_config(path):
    # Relative folders are resolved against the YAML file, which moves into Blender's add-ons
    # directory on install, so anchor them to the submodule's data/ instead.
    lines = []
    for line in path.read_text().splitlines():
        m = re.match(r"(\s*)(\w+):\s*(.*?)\s*$", line)
        if m and m.group(2) in folder_keys and m.group(3) and not m.group(3).startswith(("/", "~")):
            line = f"{m.group(1)}{m.group(2)}: {os.path.normpath(path.parent / m.group(3))}"
        elif m and m.group(2) == "server_url":
            line = f"{m.group(1)}server_url: ws://localhost:{port}/websocket"
        lines.append(line)
    return "\n".join(lines) + "\n"

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if p.is_dir() or skip_dirs & set(rel.parts) or p.name in skip_files or p.suffix == ".pyc":
            continue
        if rel.parts[0] == "data" and (len(rel.parts) != 2 or rel.name not in keep_data):
            continue
        if rel == Path("data/config.yaml"):
            zf.writestr(str(Path("gloss_blender") / rel), packaged_config(p))
        else:
            zf.write(p, Path("gloss_blender") / rel)
print(f"  {len(zf.namelist())} files")
PY
[[ -d "${ADDON_DIR}/data/mesh" ]] || log "example data missing; fetch it with: python ${ADDON_DIR}/data/download.py"

find_blender() {
    [[ -n "${BLENDER:-}" ]] && { echo "${BLENDER}"; return; }
    command -v blender 2>/dev/null && return
    for c in /Applications/Blender.app/Contents/MacOS/Blender /snap/bin/blender "/c/Program Files/Blender Foundation/"*/blender.exe; do
        [[ -x "${c}" ]] && { echo "${c}"; return; }
    done
}
BLENDER_BIN="$(find_blender || true)"
if [[ -z "${BLENDER_BIN}" ]]; then
    log "Blender not found. Install manually: Edit > Preferences > Add-ons > Install > ${BUILD}/gloss_blender.zip"
    log "then install the add-on's Python deps into Blender's Python: ${ADDON_DIR}/requirement.txt"
    exit 0
fi

log "installing into $("${BLENDER_BIN}" --version | head -1)"
"${BLENDER_BIN}" --background --factory-startup --python-exit-code 1 --python-expr "
import subprocess, sys, bpy
subprocess.check_call([sys.executable, '-m', 'ensurepip'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', r'${ADDON_DIR}/requirement.txt', 'websockets', 'pyyaml'])
bpy.ops.preferences.addon_install(filepath=r'${BUILD}/gloss_blender.zip', overwrite=True)
bpy.ops.preferences.addon_enable(module='gloss_blender')
bpy.ops.wm.save_userpref()
print('gloss_blender installed and enabled')
"
log "done. Start bash scripts/run_backend.sh, then in Blender: View3D > Sidebar > Gloss > Reconnect (data/config.yaml loads automatically)"
