#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Start the GLOSS backend that the Blender add-on connects to. Needs a CUDA GPU.
#
#   bash scripts/run_backend.sh                 # ws://localhost:10017/websocket, session "default"
#   PORT=8080 SESSION=my-scene bash scripts/run_backend.sh
#
# Details: docs/blender.md
set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${_here}/common.sh" ]] || _here="${GLOSS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}/scripts"
source "${_here}/common.sh"

PORT="${PORT:-10017}"
SESSION="${SESSION:-default}"
[[ -d "${GLOSS_DATA_DIR}/meshes" ]] || die "no data under ${GLOSS_DATA_DIR}; run: bash scripts/download_example_data.sh"
log "backend on port ${PORT}, session '${SESSION}', data ${GLOSS_DATA_DIR}"
log "if Blender runs on another machine: ssh -N -L ${PORT}:$(hostname):${PORT} <this-host>"
gloss_python scripts/interactive/run.py --port "${PORT}" --session_name "${SESSION}" "$@"
