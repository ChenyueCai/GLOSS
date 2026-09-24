# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from argparse import Namespace

import omegaconf

from gloss.config import ExperimentConfig, load_config

class TestConfigLoad(unittest.TestCase):
    """Test loading functionality via the `your_project_name.config.load_config` function.
    """

    @classmethod
    def _edit_cfg(cls, cfg: ExperimentConfig, name, value):
        setattr(cfg, name, value)

    def _load_test_cfg(self, fname, opts=None):
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yaml', fname)
        self.assertTrue(os.path.exists(cfg_path), f"Missing {fname} test config file")

        if opts is None:
            opts = []
        dummy = Namespace(config=cfg_path, distributed=False, opts=opts)
        return load_config(dummy)

    def test_from_file(self):
        """Test that loading from file works.
        """
        cfg = self._load_test_cfg('test.yaml')

        # Check values are correct
        self.assertEqual(cfg.checkpoint, None)
        self.assertEqual(cfg.enable_profiling, True)
        self.assertEqual(cfg.system.num_cpu, 1)
        self.assertEqual(cfg.system.num_gpu, 1)
        self.assertEqual(cfg.system.distributed, False)
        self.assertEqual(cfg.model.cls_path, "torch.nn.Linear")
        self.assertEqual(cfg.model.args[0], 1024)
        self.assertEqual(cfg.model.args[1], 128)
        self.assertEqual(len(cfg.model.args), 2)
        self.assertEqual(cfg.model.kwargs['bias'], True)
        self.assertEqual(len(cfg.model.kwargs), 1)
        self.assertEqual(cfg.loss.cls_path, "torch.nn.CrossEntropyLoss")
        self.assertEqual(len(cfg.loss.args), 0)
        self.assertEqual(len(cfg.loss.kwargs), 0)
        self.assertEqual(cfg.trainer.cls_path, "gloss.logging.timer.QuickTimer")
        self.assertEqual(cfg.trainer.optimizer.cls_path, "torch.optim.Adam")
        self.assertEqual(cfg.trainer.optimizer.kwargs['lr'], 0.1)
        self.assertListEqual(list(cfg.trainer.optimizer.kwargs['betas']), [0.9, 0.99])
        self.assertIsNone(cfg.trainer.scheduler)
        self.assertIsNotNone(cfg.trainer.grad_clipping)
        self.assertEqual(cfg.trainer.grad_clipping.norm_type, float('inf'))
        self.assertEqual(
            cfg.trainer.grad_clipping.max_norm,
            10.0,
            "Grad clipping max norm was not set automatically"
        )



    def test_from_file_readonly(self):
        """Test that loading from file works.
        """
        cfg = self._load_test_cfg('test.yaml')

        # Check readonly
        self.assertRaises(
            omegaconf.ReadonlyConfigError,
            TestConfigLoad._edit_cfg,
            cfg, 'enable_profiling', False
        )


    def test_from_broken_file(self):
        """Test that loading from a bad config throws an error.
        """
        self.assertRaises(omegaconf.ValidationError, self._load_test_cfg, 'test_broken.yaml')


    def test_with_opts(self):
        """Tests that file loading with option overrides works
        """
        cfg = self._load_test_cfg(
            'test.yaml',
            opts=[
                'trainer.optimizer.cls_path=torch.optim.SGD',
                'trainer.optimizer.kwargs.lr=0.1',
                'trainer.optimizer.kwargs.betas=[0.1,0.5]',
            ]
        )
        self.assertEqual(cfg.trainer.optimizer.cls_path, "torch.optim.SGD")
        self.assertEqual(cfg.trainer.optimizer.kwargs['lr'], 0.1)
        self.assertEqual(
            cfg.trainer.optimizer.kwargs.lr, 0.1,
            "Failed to retrieve kwarg argument by attribute"
        )
        self.assertListEqual(
            list(cfg.trainer.optimizer.kwargs['betas']),
            [0.1, 0.5]
        )
    
    def test_nested(self):
        cfg = self._load_test_cfg('test_nested.yaml')
        self.assertEqual(cfg.system.num_cpu, 4)
        self.assertEqual(cfg.system.num_gpu, 1)
        self.assertEqual(cfg.system.distributed, False)


if __name__ == '__main__':
    unittest.main()
