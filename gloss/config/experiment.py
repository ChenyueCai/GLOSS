# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, asdict
import os
import logging
import shutil
import re

import gloss.logging

import csv
from pathlib import Path
from typing import Union


def write_metric_to_log(
    log_fp: Union[str, Path],
    log: float,
    global_step: int,
    *,
    with_header: bool = False,
) -> None:
    """
    Append or update a (step, metric) pair in a CSV log file.

    Parameters
    ----------
    log_fp : str | pathlib.Path
        Path to the CSV file (created if it does not exist).
    log : float
        The metric value to record.  It is formatted with four digits
        after the decimal point (``{log:.4f}``).
    global_step : int
        Training / evaluation step (serves as the primary key).
    with_header : bool, optional
        Write a single‑line header ``step,metric`` when the file is
        created and empty.  Ignored on subsequent calls.

    Notes
    -----
    * If *global_step* already exists in the file, its metric value is
      **overwritten**.
    * Rows are sorted by *global_step* each time the function rewrites
      the file, keeping the log tidy.
    """
    log_fp = Path(log_fp)
    log_fp.parent.mkdir(parents=True, exist_ok=True)

    rows: dict[int, str] = {}

    # --- read existing contents (if any) ---------------------------------
    if log_fp.exists() and log_fp.stat().st_size > 0:
        with log_fp.open(newline="") as f:
            reader = csv.reader(f)
            first_row = next(reader, None)
            # Detect and preserve an existing header
            if first_row and first_row[0].lower() == "step":
                header = first_row
            else:
                header = None
                # first_row might be data, put it back
                if first_row:
                    reader = [first_row, *reader]

            for row in reader:
                if len(row) >= 2:
                    try:
                        step = int(row[0])
                        rows[step] = row[1]
                    except ValueError:
                        # Skip malformed rows
                        continue
    else:
        header = ["step", "metric"] if with_header else None

    # --- update / insert the metric for this step ------------------------
    rows[global_step] = f"{log:.4f}"

    # --- rewrite the file -------------------------------------------------
    with log_fp.open("w", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        for step in sorted(rows):
            writer.writerow([step, rows[step]])


@dataclass
class ExperimentConfig:
    name: str
    """Unique name for this experiment"""
    base_dir: str
    """Base directory for all the experiments"""
    group: str = 'default'
    """Experiment group for this experiment"""
    overwrite: bool = False
    """Set to true to overwrite the experiment folder"""

    # Training iterations between evaluations
    eval_every: int = 1000
    
    # Training iteration between saving tests
    test_every: int = 1000
    
    # Training iterations between log events
    log_every: int = 100 # TODO:increase

    # Training iterations between metric logging
    log_metric_every: int = 250 #TODO:increase

    # Training iterations between "latest" checkpointing
    checkpoint_every: int = 1000 #TODO:increase

    # Training iterations between persistent checkpointing
    persistent_checkpoint_every: int = 5000
    # Whether to write/update chkpt_latest.ckpt during training.
    save_latest_checkpoint: bool = True
    # Additional persistent checkpoint steps expressed as a comma/space separated list, e.g. "1000,2000,5000"
    persistent_checkpoint_steps: str = ""


class ExperimentHelper:
    def __init__(self, config: ExperimentConfig, args):
        global_root_dir = args.global_root_dir
        exp_base_dir = os.path.join(global_root_dir, config.base_dir)
        os.makedirs(exp_base_dir, exist_ok=True)
        assert os.path.isdir(exp_base_dir), f'DNE: {exp_base_dir}'
        run_dir = os.path.join(exp_base_dir, config.group, config.name)

        self.config = config
        self.run_dir = run_dir
        self.log_dir = os.path.join(run_dir, 'logs')
        self.checkpt_dir = os.path.join(run_dir, 'checkpts')
        self.config_dir = os.path.join(run_dir, 'config')
        self.viz_dir = os.path.join(run_dir, 'viz')
        self.eval_dir = os.path.join(run_dir, 'eval')
        self.persistent_checkpoint_steps = self._parse_checkpoint_steps(config.persistent_checkpoint_steps)
        self.setup(config.overwrite, log_level=args.log_level)
        self.start_global_step = 0
        print(f'Setup experiment in dir {self.run_dir}')

    @staticmethod
    def _parse_checkpoint_steps(raw_steps: str) -> set[int]:
        if raw_steps is None:
            return set()
        raw_steps = raw_steps.strip()
        if not raw_steps:
            return set()

        steps = set()
        for token in re.split(r"[\s,]+", raw_steps):
            if not token:
                continue
            step = int(token)
            if step <= 0:
                raise ValueError(f"Checkpoint steps must be positive integers, got {step}")
            steps.add(step)
        return steps

    def purge(self):
        if os.path.exists(self.run_dir):
            # Note: this is print because this is typically called before logger is set up
            print('Purging past run directory: %s' % self.run_dir)
            # TODO: add checks to make sure we don't do anything catastrophic
            shutil.rmtree(self.run_dir)

    def setup(self, purge=False, log_level=logging.INFO):
        if purge:
            self.purge()

        os.makedirs(os.path.abspath(self.run_dir), exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpt_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(os.path.join(self.viz_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.viz_dir, 'eval'), exist_ok=True)
        os.makedirs(os.path.join(self.viz_dir, 'test'), exist_ok=True)
     
    def start_training_at(self, global_step):
        self.start_global_step = global_step

    def should_log(self, global_step):
        early_steps = (global_step - self.start_global_step) < 100
        return global_step % self.config.log_every == 0 or (early_steps and (global_step % 5 == 0))

    def should_eval(self, global_step):
        early_steps = (global_step - self.start_global_step) < 5
        if early_steps:
            return False
        warming_stage = (global_step - self.start_global_step) < 10000
        return global_step % self.config.eval_every == 0 or (warming_stage and (global_step % 1000 == 0))

    def should_checkpoint(self, global_step):
        early_steps = (global_step - self.start_global_step) < 50
        if early_steps:
            return False
        if global_step in self.persistent_checkpoint_steps:
            return True
        return global_step % self.config.checkpoint_every == 0

    def should_persist_checkpoint(self, global_step):
        if global_step in self.persistent_checkpoint_steps:
            return True
        return global_step % self.config.persistent_checkpoint_every == 0

    def should_test(self, global_step):
        early_steps = (global_step - self.start_global_step) < 50
        if early_steps:
            return False
        return global_step % self.config.test_every == 0
    
    def should_log_metric(self, global_step):
        early_steps = (global_step - self.start_global_step) < 50
        if early_steps:
            return False
        return global_step % self.config.log_metric_every == 0
    
    def log(self, log_name, log, global_step):
        log_fp = os.path.join(self.log_dir, f"{log_name}.csv")
        write_metric_to_log(log_fp, log, global_step)
        
