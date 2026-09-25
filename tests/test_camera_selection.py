# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the target-camera selection rule.

Only the greedy cover is exercised here: it is pure tensor math, so it runs on
CPU without kaolin, a GPU, or a mesh. Camera placement and rasterization
(``camera_from_face_normal`` / ``build_face_coverage_matrix``) need a real mesh
and are not covered by this module.
"""

import unittest

import torch

from gloss_interactive.utils import greedy_camera_cover


class TestGreedyCameraCover(unittest.TestCase):
    def test_picks_largest_coverage_first(self):
        # cam0 covers 1 face, cam1 covers 3, cam2 covers 2.
        coverage = torch.tensor([
            [1, 1, 0],
            [0, 1, 1],
            [0, 1, 1],
        ], dtype=torch.float32)
        weights = torch.ones(3)
        chosen, _ = greedy_camera_cover(coverage, weights, max_cameras=1)
        self.assertEqual(chosen, [1])

    def test_second_pick_maximizes_newly_covered_area(self):
        # cam0 covers faces 0-1. cam1 covers face 1 only (fully redundant after
        # cam0). cam2 covers face 2 only. cam2 must win the second slot.
        coverage = torch.tensor([
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 1],
        ], dtype=torch.float32)
        weights = torch.ones(3)
        chosen, remaining = greedy_camera_cover(coverage, weights, max_cameras=2)
        self.assertEqual(chosen, [0, 2])
        self.assertEqual(remaining, 0.0)

    def test_stops_when_nothing_new_is_covered(self):
        # Two cameras see the same single face; the second adds nothing, so
        # only one is returned even though max_cameras allows five.
        coverage = torch.tensor([[1.0, 1.0]])
        chosen, _ = greedy_camera_cover(coverage, torch.ones(1), max_cameras=5)
        self.assertEqual(chosen, [0])

    def test_respects_max_cameras(self):
        coverage = torch.eye(6)
        chosen, remaining = greedy_camera_cover(coverage, torch.ones(6), max_cameras=2)
        self.assertEqual(len(chosen), 2)
        self.assertEqual(remaining, 4.0)  # four faces left uncovered

    def test_area_weights_beat_face_count(self):
        # cam0 covers two tiny faces; cam1 covers one large face. Weighted by
        # area the single large face wins, which is the point of the weights.
        coverage = torch.tensor([
            [1, 0],
            [1, 0],
            [0, 1],
        ], dtype=torch.float32)
        weights = torch.tensor([1.0, 1.0, 10.0])
        chosen, _ = greedy_camera_cover(coverage, weights, max_cameras=1)
        self.assertEqual(chosen, [1])

    def test_no_coverage_returns_nothing(self):
        chosen, remaining = greedy_camera_cover(torch.zeros(3, 4), torch.ones(3), max_cameras=3)
        self.assertEqual(chosen, [])
        self.assertEqual(remaining, 3.0)


if __name__ == "__main__":
    unittest.main()
