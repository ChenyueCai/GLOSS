# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import omegaconf

from gloss.config import ExperimentConfig

class TestConfigDefaults(unittest.TestCase):
    """Tests that the default values are maintained as expected

    These tests should only check critical settings, as others should be freely tweaked.
    """

    def test_basic(self):
        """Tests that basic critical values are set correctly by default.
        """
        cfg: ExperimentConfig = omegaconf.OmegaConf.structured(ExperimentConfig)

        self.assertEqual(cfg.checkpoint, None)
        self.assertEqual(cfg.enable_profiling, False)
        self.assertEqual(
            cfg.system.distributed_backend,
            'nccl',
            'Distributed back end default should be NCCL.'
        )
        self.assertRaises(
            omegaconf.MissingMandatoryValue,
            getattr,
            cfg.model,
            'cls_path'
        )
        self.assertEqual(len(cfg.model.args), 0)
        self.assertEqual(len(cfg.model.kwargs), 0)
        self.assertRaises(
            omegaconf.MissingMandatoryValue,
            getattr,
            cfg.loss,
            'cls_path'
        )
        self.assertEqual(len(cfg.loss.args), 0)
        self.assertEqual(len(cfg.loss.kwargs), 0)
        self.assertRaises(
            omegaconf.MissingMandatoryValue,
            getattr,
            cfg.trainer,
            'cls_path'
        )
        self.assertEqual(len(cfg.trainer.trainer_kwargs), 0)

        self.assertEqual(
            cfg.trainer.mixed_precision,
            True,
            "Mixed precision should be enabled by default"
        )

        self.assertRaises(
            omegaconf.MissingMandatoryValue,
            getattr,
            cfg.trainer.optimizer,
            'cls_path'
        )
        self.assertEqual(len(cfg.trainer.optimizer.args), 0)
        self.assertEqual(len(cfg.trainer.optimizer.kwargs), 0)

        self.assertIsNone(cfg.trainer.grad_clipping, "Grad clipping should be disabled by default")

        self.assertIsNone(cfg.trainer.scheduler, "Scheduler is set to None by default")

        self.assertEqual(len(cfg.dataset.train_sets), 0)
        self.assertEqual(len(cfg.dataset.test_sets), 0)


if __name__ == '__main__':
    unittest.main()
