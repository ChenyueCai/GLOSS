#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Download the inference example data and model checkpoints from Hugging Face.
#
#   bash scripts/download_example_data.sh                     # into ./data/interactive
#   bash scripts/download_example_data.sh --skip-checkpoints  # data only (e.g. the Blender laptop)
#
# Set GLOSS_DATA_DIR to download somewhere else. Details: docs/setup.md
set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${_here}/common.sh" ]] || _here="${GLOSS_REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}/scripts"
source "${_here}/common.sh"

SKIP_CKPTS=0
for arg in "$@"; do
    case "${arg}" in
        --skip-checkpoints) SKIP_CKPTS=1 ;;
        -h|--help) sed -n 2,8p "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown argument: ${arg}" ;;
    esac
done
mkdir -p "${GLOSS_DATA_DIR}"
log "example data  ${GLOSS_HF_DATA_REPO} -> ${GLOSS_DATA_DIR}"
(( SKIP_CKPTS )) || log "checkpoints   ${GLOSS_HF_REPO} -> ${GLOSS_DATA_DIR}/ckpts"

SKIP_CKPTS="${SKIP_CKPTS}" gloss_python - <<'PY'
import os, random, sys, time
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError
dest = os.environ["GLOSS_DATA_DIR"]
api = HfApi()
ATTEMPTS = 10

def pause(attempt, why):
    wait = min(300, 20 * 2 ** attempt) * (0.8 + 0.4 * random.random())
    print(f"[gloss] {why}; retrying in {wait:.0f}s (huggingface-cli login raises the rate limit)", file=sys.stderr, flush=True)
    time.sleep(wait)

def rate_limited(e):
    return getattr(getattr(e, "response", None), "status_code", None) == 429

def expected_files(**kw):
    for attempt in range(ATTEMPTS):
        try:
            return [f for f in api.list_repo_files(**kw) if f != ".gitattributes"]
        except HfHubHTTPError as e:
            if not rate_limited(e) or attempt == ATTEMPTS - 1: raise
            pause(attempt, "Hugging Face rate limit reached while listing files")

def fetch(patterns=None, **kw):
    # snapshot_download can return a partial folder without raising when it is rate limited,
    # so compare what arrived against the repo's file list and resume until nothing is missing.
    want = [f for f in expected_files(**kw) if not patterns or any(f.startswith(p.rstrip("*")) for p in patterns)]
    for attempt in range(ATTEMPTS):
        try:
            snapshot_download(local_dir=dest, max_workers=2, allow_patterns=patterns, **kw)
        except HfHubHTTPError as e:
            if not rate_limited(e): raise
        missing = [f for f in want if not os.path.isfile(os.path.join(dest, f))]
        if not missing:
            print(f"[gloss] {kw['repo_id']}: all {len(want)} files present", file=sys.stderr, flush=True)
            return
        if attempt == ATTEMPTS - 1:
            sys.exit(f"[gloss] ERROR: {len(missing)} of {len(want)} files from {kw['repo_id']} are still missing "
                     f"(e.g. {missing[:3]}). Rerun this script later, or log in with huggingface-cli login.")
        pause(attempt, f"{len(missing)} of {len(want)} files from {kw['repo_id']} still missing")

fetch(repo_id=os.environ["GLOSS_HF_DATA_REPO"], repo_type="dataset")
if os.environ["SKIP_CKPTS"] != "1":
    fetch(patterns=["ckpts/*"], repo_id=os.environ["GLOSS_HF_REPO"])
PY
log "done. Layout: $(ls "${GLOSS_DATA_DIR}" | tr '\n' ' ')"
