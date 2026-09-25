# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for ``gloss_interactive.syncmvd.run_syncmvd_inference``.

Run from material-superres-private:

    conda activate gloss
    python -m unittest tests.test_syncmvd_inference

The test loads a small mesh + the cabbage ckpt, builds a 2-ref + 3-target
batch, runs 5-step SyncMVD denoising, and checks that:
  * inference returns one decoded RGB view per batch slot
  * the output dtype, shape, and value range are reasonable
  * the per-target outputs differ from each other (sync produced view-specific
    content, not collapsed to one repeated image)

The test is skipped automatically when CUDA or the fixture files are missing,
so it is safe to leave in the suite.
"""
import os
import unittest
import warnings
import math

import torch
import kaolin
import torchvision
from kornia.morphology import erosion

from gloss.utils.kaolin_utils import load_mesh
from gloss.utils.render_fast import custom_mesh_batched_render
from gloss.model.attention import SamplewiseAttnProcessor2_0
from gloss.inpaint.models import TextureInpaintStandardModel
from gloss.inpaint.completion_utils import CameraConfig, get_camera_from_face

from gloss_interactive.syncmvd import run_syncmvd_inference


warnings.filterwarnings("ignore", category=UserWarning)

_DATA_DIR = os.environ.get("GLOSS_DATA_DIR", "")
MESH_PATH = os.path.join(_DATA_DIR, "mesh/transfer_mesh/cabbage/cabbage_1/scene.gltf")
CKPT_PATH = os.path.join(_DATA_DIR, "ckpts/cabbage/chkpt_80000.ckpt")

IN_CHANNELS = ["camera_normals", "relative_positions", "albedo", "inpaint_mask"]
NUM_IN_CHANNELS = 17


def _build_view_data(camera, mesh, mesh_texture, in_channels, device):
    """Render a single view through ``camera`` and pack channels in the format
    the inpaint model expects (matches ``ReferenceBrush._render_views_to_data``).
    """
    s = mesh_texture.shape[0]
    # The brush call site uses ``process_as_albedo=False`` and assumes a
    # 4-channel RGBA texture (last channel = alpha). Pad with alpha=1 if the
    # mesh ships a 3-channel diffuse texture (cabbage_1 does).
    if mesh_texture.shape[-1] == 3:
        mesh_texture = torch.cat(
            [mesh_texture, torch.ones_like(mesh_texture[..., :1])], dim=-1
        )
    normal_map = mesh.materials[0].hwc().normals_texture
    if normal_map is not None:
        normal_map = (
            torchvision.transforms.Resize((s, s))(normal_map.permute(2, 0, 1))
            .permute(1, 2, 0)
            .to(device)
        )
    batch_camera = kaolin.render.camera.Camera.cat([camera.to(device)])
    r = custom_mesh_batched_render(
        batch_camera, mesh, mesh_texture, normal_map,
        requires_positions=True, process_as_albedo=False, backend="cuda",
    )
    r["albedo"] = r["textured"][..., :3] * 2 - 1
    r["albedo_alpha"] = r["textured"][..., 3:4]
    bg_mask = (torch.abs(r["camera_normals"]) < 0.01).all(dim=-1, keepdim=True).float()
    out = {}
    for ch in in_channels:
        if ch == "inpaint_mask":
            kernel = torch.ones(15, 15, device=device)
            out["inpaint_mask"] = erosion(
                r["albedo_alpha"].permute(0, 3, 1, 2), kernel
            ).permute(0, 2, 3, 1)
        else:
            out[ch] = r[ch] * (1 - bg_mask) + bg_mask * -1
    out["albedo_alpha"] = out["inpaint_mask"]
    out["background_alpha"] = (1.0 - bg_mask) * 2.0 - 1.0
    return out


class TestSyncMVDInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        if not os.path.exists(MESH_PATH):
            raise unittest.SkipTest(f"missing test mesh: {MESH_PATH}")
        if not os.path.exists(CKPT_PATH):
            raise unittest.SkipTest(f"missing ckpt: {CKPT_PATH}")

        cls.device = torch.device("cuda")
        cls.cam_config = CameraConfig()
        cls.mesh = load_mesh(MESH_PATH).to(cls.device)
        cls.model = TextureInpaintStandardModel(
            NUM_IN_CHANNELS, IN_CHANNELS, CKPT_PATH, use_fp16=True
        )
        cls.model.set_attention_proc(SamplewiseAttnProcessor2_0)
        # The unCLIP image embedding is required by ``model.step``; a uniform
        # gray dummy works for a smoke test (we don't validate aesthetics).
        dummy = torch.full((1, 3, 224, 224), 0.5, device=cls.device)
        cls.model.set_embedding("", dummy)
        cls.mesh_texture = cls.mesh.materials[0].diffuse_texture.to(cls.device)

    def test_runs_and_returns_per_view_outputs(self):
        n_faces = self.mesh.faces.shape[0]
        # Three target cameras and two reference patches anchored at distinct
        # faces to keep the views meaningfully different.
        target_face_ids = [n_faces // 4, n_faces // 2, 3 * n_faces // 4]
        ref_face_ids = [n_faces // 8, 5 * n_faces // 8]
        target_cameras = [
            get_camera_from_face(self.mesh, fid, self.cam_config, self.device)
            for fid in target_face_ids
        ]
        target_data = [
            _build_view_data(c, self.mesh, self.mesh_texture, IN_CHANNELS, self.device)
            for c in target_cameras
        ]
        ref_data = [
            _build_view_data(
                get_camera_from_face(self.mesh, fid, self.cam_config, self.device),
                self.mesh, self.mesh_texture, IN_CHANNELS, self.device,
            )
            for fid in ref_face_ids
        ]

        torch.manual_seed(0)
        out = run_syncmvd_inference(
            self.model, self.mesh,
            target_cameras=target_cameras,
            inpaint_data=ref_data + target_data,
            num_refs=len(ref_data),
            camera_config=self.cam_config,
            num_inference_steps=5,         # short for speed
            multiview_diffusion_end=0.6,   # keep most steps in sync mode
        )

        # Shape: (num_refs + num_targets, 3, H, W)
        expected_n = len(ref_data) + len(target_cameras)
        self.assertEqual(out.shape[0], expected_n)
        self.assertEqual(out.shape[1], 3)
        self.assertEqual(out.shape[2], self.cam_config.resolution)
        self.assertEqual(out.shape[3], self.cam_config.resolution)

        # Values: VAE-decoded RGB postprocessed to [0, 1]
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLessEqual(out.max().item(), 1.0)
        self.assertFalse(torch.isnan(out).any().item(), "NaNs in output")

        # Sanity: the three target outputs aren't byte-identical (sync should
        # still produce per-view content; only the latent UV is shared).
        targets_out = out[len(ref_data):]
        for i in range(targets_out.shape[0]):
            for j in range(i + 1, targets_out.shape[0]):
                diff = (targets_out[i] - targets_out[j]).abs().mean().item()
                self.assertGreater(diff, 1e-3,
                    f"target views {i} and {j} are unexpectedly identical")

    def test_falls_back_when_single_camera(self):
        """A single-camera batch should still return valid output (the brush
        skips SyncMVD in that case but ``run_syncmvd_inference`` itself should
        not crash if called directly with one target)."""
        n_faces = self.mesh.faces.shape[0]
        target_cameras = [get_camera_from_face(self.mesh, n_faces // 2,
                                               self.cam_config, self.device)]
        target_data = [
            _build_view_data(target_cameras[0], self.mesh, self.mesh_texture,
                             IN_CHANNELS, self.device)
        ]
        ref_data = [
            _build_view_data(
                get_camera_from_face(self.mesh, n_faces // 4, self.cam_config, self.device),
                self.mesh, self.mesh_texture, IN_CHANNELS, self.device,
            )
        ]
        torch.manual_seed(0)
        out = run_syncmvd_inference(
            self.model, self.mesh,
            target_cameras=target_cameras,
            inpaint_data=ref_data + target_data,
            num_refs=1,
            camera_config=self.cam_config,
            num_inference_steps=3,
            multiview_diffusion_end=0.4,
        )
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], 3)
        self.assertFalse(torch.isnan(out).any().item())


if __name__ == "__main__":
    unittest.main()
