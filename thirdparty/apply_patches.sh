#!/usr/bin/env bash
# Apply the local modifications this project needs on top of the pinned
# third-party submodules. Run once after `git submodule update --init`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for name in ComfyUI InvSR cmmd-pytorch; do
    repo="$HERE/$name"
    patch="$HERE/patches/$name.patch"
    if [[ ! -e "$repo/.git" ]]; then
        echo "$name: submodule not checked out, skipping."
        continue
    fi
    if git -C "$repo" apply --check --reverse "$patch" >/dev/null 2>&1; then
        echo "$name: patch already applied, skipping."
        continue
    fi
    git -C "$repo" apply "$patch"
    echo "$name: patch applied."
done
