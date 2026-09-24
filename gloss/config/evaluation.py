# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvaluationConfig:
    name: str
    """"name for the evaluation"""
    data_dir: str
    """"data directory for the experiment"""
    expr_dir: str
    """"root directory for the experiment"""
    save_dir: str
    """dir to save evaluation metrics"""
    model_fp: str
    """path to the evaluation model"""
    """number of channels"""
    mesh: str
    """mesh filepath"""
    num_source_views: int
    """number of source views"""
    num_target_views: int
    """number of target views"""
    fov_min: float
    """fov configuration"""
    fov_max: float
    """fov configuration"""
    cam_dist: float
    """cam configuration"""
    single_view_fp: Optional[str] = None
    """filepath to the single diffusion view """
    custom_material: Optional[str] = None
    """file path to custom material """
    v_emb_fp: Optional[str] = None
    """filepath to the view embedding"""
    p_emb_fp: Optional[str] = None
    """filepath to the prompt embedding"""
    in_channels: list[str] = field(
        default_factory=lambda: [
            "geo_camera_normals",
            "relative_positions",
            "albedo",
            "inpaint_mask",
        ]
    )
    """The input channels to the transformer."""
    num_in_channels: int = 17
    """The number of input channels to the transformer."""
    seed: int = 0
    """experiment seed"""
    log: bool = True
    """logging intermediates"""
    overwrite: bool = False
    """whether to overwriting csv"""
    cache_camera: bool = True
    """whether to cache cameras"""
    mode: str = "gloss"
    bootstrap_iter: int= 3
