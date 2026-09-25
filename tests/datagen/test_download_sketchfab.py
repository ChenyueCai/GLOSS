# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "datagen" / "utils" / "download_sketchfab.py"
SPEC = spec_from_file_location("download_sketchfab", MODULE_PATH)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_input_file_supports_direct_mesh_registry_format(tmp_path):
    registry = tmp_path / "test_mesh.txt"
    registry.write_text(
        "\n".join(
            [
                "# comment",
                "rusty_barrel_metal: https://sketchfab.com/3d-models/rusty-barrel-0123456789abcdef0123456789abcdef",
                "sea_star: https://sketchfab.com/3d-models/sea-star-fedcba9876543210fedcba9876543210",
            ]
        ),
        encoding="utf-8",
    )

    entries = MODULE.parse_input_file(registry)

    assert entries == [
        {
            "url": "https://sketchfab.com/3d-models/rusty-barrel-0123456789abcdef0123456789abcdef",
            "uid": "0123456789abcdef0123456789abcdef",
            "label": "rusty_barrel_metal",
            "folder_name": "rusty_barrel_metal",
        },
        {
            "url": "https://sketchfab.com/3d-models/sea-star-fedcba9876543210fedcba9876543210",
            "uid": "fedcba9876543210fedcba9876543210",
            "label": "sea_star",
            "folder_name": "sea_star",
        },
    ]


def test_parse_input_file_rejects_duplicate_direct_mesh_names(tmp_path):
    registry = tmp_path / "test_mesh.txt"
    registry.write_text(
        "\n".join(
            [
                "rusty_barrel_metal: https://sketchfab.com/3d-models/rusty-barrel-0123456789abcdef0123456789abcdef",
                "rusty_barrel_metal: https://sketchfab.com/3d-models/rusty-barrel-fedcba9876543210fedcba9876543210",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate mesh name"):
        MODULE.parse_input_file(registry)


def test_parse_input_file_keeps_legacy_label_format(tmp_path):
    registry = tmp_path / "assets.txt"
    registry.write_text(
        "\n".join(
            [
                "Savoy cabbage",
                "https://sketchfab.com/3d-models/savoy-cabbage-0123456789abcdef0123456789abcdef",
                "https://sketchfab.com/3d-models/savoy-cabbage-fedcba9876543210fedcba9876543210",
            ]
        ),
        encoding="utf-8",
    )

    entries = MODULE.parse_input_file(registry)

    assert [entry["folder_name"] for entry in entries] == ["savoy_cabbage_0", "savoy_cabbage_1"]
