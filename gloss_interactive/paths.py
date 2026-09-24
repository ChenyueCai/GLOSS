"""Canonical paths for the gloss_interactive package.

Single source of truth so scripts under ``scripts/interactive/`` don't each
hardcode the same config path. Change ``INTERACTIVE_MESH_CONFIG`` here to
re-point every entrypoint at once.
"""

import os

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSET_DIR = os.path.join(_PACKAGE_DIR, "asset")

# Active config for the interactive_mesh workflow (mesh/interactive_mesh/,
# ckpts/interactive_mesh/, etc.). Imported by scripts/interactive/run.py and
# friends.
INTERACTIVE_MESH_CONFIG = os.path.join(_ASSET_DIR, "example_config.yaml")
