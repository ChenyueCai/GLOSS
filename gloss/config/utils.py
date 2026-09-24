# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for config management.
"""

import argparse
from omegaconf import OmegaConf, DictConfig

from .specification import ExperimentConfig

## Register some convenience resolvers
OmegaConf.register_new_resolver("load_cfg", OmegaConf.load)


def load_config(args: argparse.Namespace) -> ExperimentConfig:
    """Load the config from the CLI arguments.

    Args:
        args (argparse.Namespace): Arguments from parsing with argparser.
            Expected to contain:
                - A `config` argument pointing to the config file
                - An `opts` argument containing a list of config overrides
                - A `distributed` boolean overriding the config distributed flag

    Returns:
        OmegaConf: _description_
    """
    # Load the config file and update with CLI
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    OmegaConf.resolve(cfg)

    assert isinstance(cfg, DictConfig), "Config must be a DictConfig"

    # Parse into an experiment config
    cfg = OmegaConf.structured(ExperimentConfig(**cfg))

    # Override the distributed flag
    cfg.system.distributed = args.distributed
    # Make immutable once loaded
    OmegaConf.set_readonly(cfg, True)
    return cfg
