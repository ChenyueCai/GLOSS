# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for scripts/datagen/pipeline.py.

All tests are pure unit tests – no gloss, kaolin, or GPU is required.
External processes are never spawned; subprocess.run is always mocked.
"""

import os
import argparse
import importlib
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

# ---------------------------------------------------------------------------
# Make pipeline importable regardless of cwd
# ---------------------------------------------------------------------------
PIPELINE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "scripts"
    / "datagen"
    / "pipeline.py"
)

import importlib.util

spec = importlib.util.spec_from_file_location("pipeline", PIPELINE_PATH)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def _make_args(tmp_path: Path, **overrides) -> SimpleNamespace:
    """Return a namespace that mimics the parsed argparse args."""
    defaults = dict(
        data_dir=tmp_path / "data",
        expr_tag="exp01",
        mesh_name="barrel",
        mesh_subject=None,
        num_prompts=500,
        openai_model="o3",
        comfyui_dir=None,
        comfyui_install_requirements=False,
        canny_normal=False,
        canny_weight=None,
        comfyui_visualize=False,
        comfyui_port=8188,
        decomp_config=None,
        decomp_script=None,
        decomp_weights=None,
        diffrender_dir=None,
        diffrender_env="diff-render",
        hf_home=None,
        diffrender_inference_res="512,512",
        diffrender_fallback_inference_res=["384,384", "256,256"],
        diffrender_inference_n_steps=20,
        invsr_dir=None,
        invsr_env="invsr",
        texture_size=4096,
        dataset_tag="cam0.25-fov0.4-0.8",
        start_view=0,
        end_view=-1,
        fov_min=0.4,
        fov_max=0.8,
        camera_dist=0.25,
        mesh_subpath="mesh/scene.gltf",
        gloss_env="gloss",
        steps=list(range(1, 9)),
        force=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _ok_proc() -> MagicMock:
    """Fake successful subprocess.CompletedProcess."""
    p = MagicMock()
    p.returncode = 0
    return p


def _fail_proc() -> MagicMock:
    """Fake failed subprocess.CompletedProcess."""
    p = MagicMock()
    p.returncode = 1
    return p


# ===========================================================================
# Unit tests – helper functions
# ===========================================================================

class TestDoneMarker(unittest.TestCase):

    def test_path_is_DONE_file(self, ):
        d = Path("/some/dir")
        self.assertEqual(pipeline.done_marker(d), Path("/some/dir/DONE"))

    def test_is_done_false_when_missing(self, tmp_path=None):
        with unittest.mock.patch("pathlib.Path.is_file", return_value=False):
            import tempfile, os
            with tempfile.TemporaryDirectory() as td:
                self.assertFalse(pipeline.is_done(Path(td) / "step"))

    def test_is_done_true_when_file_exists(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "step"
            d.mkdir()
            (d / "DONE").touch()
            self.assertTrue(pipeline.is_done(d))

    def test_mark_done_creates_dir_and_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "a" / "b"
            self.assertFalse(d.exists())
            pipeline.mark_done(d)
            self.assertTrue(d.is_dir())
            self.assertTrue((d / "DONE").is_file())

    def test_mark_done_idempotent(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "step"
            pipeline.mark_done(d)
            pipeline.mark_done(d)   # must not raise
            self.assertTrue((d / "DONE").is_file())


# ===========================================================================

class TestPythonCmd(unittest.TestCase):

    def test_contains_conda_run(self):
        cmd = pipeline.python_cmd("myenv", Path("/scripts/foo.py"), "--bar 1")
        self.assertIn("conda run -n myenv", cmd)

    def test_contains_script_path(self):
        cmd = pipeline.python_cmd("myenv", Path("/scripts/foo.py"), "--bar 1")
        self.assertIn("/scripts/foo.py", cmd)

    def test_contains_args(self):
        cmd = pipeline.python_cmd("myenv", Path("/s.py"), "--alpha 42 --beta x")
        self.assertIn("--alpha 42 --beta x", cmd)


# ===========================================================================

class TestRunHelper(unittest.TestCase):

    @patch("subprocess.run", return_value=_ok_proc())
    def test_success_does_not_exit(self, mock_run):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "log.txt"
            pipeline.run("echo hi", Path(td), log)   # should not raise / call sys.exit
        mock_run.assert_called_once()

    @patch("subprocess.run", return_value=_fail_proc())
    def test_failure_calls_sys_exit(self, mock_run):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "log.txt"
            with self.assertRaises(SystemExit) as cm:
                pipeline.run("false", Path(td), log)
            self.assertEqual(cm.exception.code, 1)

    @patch("subprocess.run", return_value=_ok_proc())
    def test_log_file_parent_created(self, mock_run):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "subdir" / "log.txt"
            pipeline.run("echo hi", Path(td), log)
            self.assertTrue(log.parent.is_dir())

    @patch("subprocess.run", return_value=_ok_proc())
    def test_cwd_passed_to_subprocess(self, mock_run):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "log.txt"
            pipeline.run("echo hi", Path(td), log)
            _, kwargs = mock_run.call_args
            self.assertEqual(kwargs["cwd"], str(Path(td)))


# ===========================================================================

class TestBuildPaths(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def _paths(self, **kw):
        return pipeline.build_paths(_make_args(self.tmp, **kw))

    def test_data_dir_is_absolute(self):
        p = self._paths()
        self.assertTrue(p["data_dir"].is_absolute())

    def test_expr_root(self):
        p = self._paths()
        self.assertEqual(p["expr_root"], p["data_dir"] / "exp01")

    def test_expr_dir(self):
        p = self._paths()
        self.assertEqual(p["expr_dir"], p["data_dir"] / "exp01" / "barrel")

    def test_mesh_fp(self):
        p = self._paths()
        self.assertEqual(p["mesh_fp"], p["data_dir"] / "exp01" / "barrel" / "mesh" / "scene.gltf")

    def test_prompts_fp_uses_mesh_name(self):
        p = self._paths()
        self.assertEqual(p["prompts_fp"].name, "barrel.txt")
        self.assertEqual(p["prompts_fp"].parent.name, "prompts")

    def test_single_view_dir(self):
        p = self._paths()
        expected_suffix = Path("barrel") / "single_view" / "civitai2.0"
        self.assertTrue(str(p["single_view_dir"]).endswith(str(expected_suffix)))

    def test_config_yaml(self):
        p = self._paths()
        self.assertEqual(p["config_yaml"], p["expr_root"] / "config.yaml")

    def test_different_mesh_names(self):
        for name in ("croissant", "rusty_barrel_metal", "sea_urchin_shell"):
            p = pipeline.build_paths(_make_args(self.tmp, mesh_name=name))
            self.assertIn(name, str(p["mesh_fp"]))
            self.assertEqual(p["prompts_fp"].name, f"{name}.txt")


# ===========================================================================
# Step runner tests  (subprocess.run fully mocked)
# ===========================================================================

class _StepTestBase(unittest.TestCase):
    """Base class – sets up a tmp_path, args, and paths for each test."""

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.args = _make_args(self.root)
        self.paths = pipeline.build_paths(self.args)

    def tearDown(self):
        self._td.cleanup()

    def _mark_done(self, subdir: Path):
        pipeline.mark_done(subdir)

    @staticmethod
    def _patch_subprocess():
        return patch("subprocess.run", return_value=_ok_proc())

    @staticmethod
    def _patch_sys_exit():
        return patch("sys.exit", side_effect=SystemExit)


class TestStep1GeneratePrompts(_StepTestBase):

    def test_skips_when_done(self):
        self._mark_done(self.paths["prompts_dir"])
        with self._patch_subprocess() as mock_run:
            pipeline.step_generate_prompts(self.args, self.paths)
        mock_run.assert_not_called()

    def test_runs_when_not_done(self):
        with self._patch_subprocess() as mock_run:
            pipeline.step_generate_prompts(self.args, self.paths)
        mock_run.assert_called_once()

    def test_marks_done_after_run(self):
        with self._patch_subprocess():
            pipeline.step_generate_prompts(self.args, self.paths)
        self.assertTrue(pipeline.is_done(self.paths["prompts_dir"]))

    def test_uses_mesh_subject_when_given(self):
        self.args.mesh_subject = "rusty metal barrel"
        captured = []
        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_generate_prompts(self.args, self.paths)
        self.assertIn("rusty metal barrel", captured[0])

    def test_falls_back_to_mesh_name(self):
        self.args.mesh_subject = None
        captured = []
        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_generate_prompts(self.args, self.paths)
        self.assertIn(self.args.mesh_name, captured[0])


class TestStep2GenerateCondition(_StepTestBase):

    def test_skips_when_done(self):
        self._mark_done(self.paths["single_view_dir"] / "condition_output")
        with self._patch_subprocess() as mock_run:
            pipeline.step_generate_condition(self.args, self.paths)
        mock_run.assert_not_called()

    def test_runs_when_not_done(self):
        # create config.yaml so we skip _ensure_config_yaml
        self.paths["config_yaml"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["config_yaml"].touch()
        with self._patch_subprocess() as mock_run:
            pipeline.step_generate_condition(self.args, self.paths)
        mock_run.assert_called_once()

    def test_marks_done_after_run(self):
        self.paths["config_yaml"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["config_yaml"].touch()
        with self._patch_subprocess():
            pipeline.step_generate_condition(self.args, self.paths)
        self.assertTrue(pipeline.is_done(self.paths["single_view_dir"] / "condition_output"))

    def test_auto_creates_config_yaml_when_missing(self):
        """If config.yaml is absent, _ensure_config_yaml must be called first."""
        self.assertFalse(self.paths["config_yaml"].exists())
        call_order = []
        def _capture(cmd, **kw):
            call_order.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_generate_condition(self.args, self.paths)
        # First call should be generate_config_yml.py, second generate_condition.py
        self.assertGreaterEqual(len(call_order), 2)
        self.assertIn("generate_config_yml.py", call_order[0])
        self.assertIn("generate_condition.py", call_order[1])

    def test_cmd_contains_mesh_fp(self):
        self.paths["config_yaml"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["config_yaml"].touch()
        captured = []
        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_generate_condition(self.args, self.paths)
        self.assertIn("scene.gltf", captured[0])

    def test_force_flag_sets_overwrite_true(self):
        self.args.force = True
        self.paths["config_yaml"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["config_yaml"].touch()
        captured = []
        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_generate_condition(self.args, self.paths)
        self.assertIn("--overwrite=True", captured[0])

    def test_no_force_sets_overwrite_false(self):
        self.args.force = False
        self.paths["config_yaml"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["config_yaml"].touch()
        captured = []
        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_generate_condition(self.args, self.paths)
        self.assertIn("--overwrite=False", captured[0])

    def test_condition_mode_debug(self):
        self.args.condition_mode = "debug"
        self.paths["config_yaml"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["config_yaml"].touch()
        captured = []
        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_generate_condition(self.args, self.paths)
        self.assertIn("--mode=debug", captured[0])

    def test_condition_mode_generate(self):
        self.args.condition_mode = "generate"
        self.paths["config_yaml"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["config_yaml"].touch()
        captured = []
        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_generate_condition(self.args, self.paths)
        self.assertIn("--mode=generate", captured[0])

    def test_cmd_contains_prompts_fp(self):
        self.paths["config_yaml"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["config_yaml"].touch()
        captured = []
        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_generate_condition(self.args, self.paths)
        self.assertIn(str(self.paths["prompts_fp"]), captured[0])

    def test_cmd_contains_output_dir(self):
        self.paths["config_yaml"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["config_yaml"].touch()
        captured = []
        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_generate_condition(self.args, self.paths)
        self.assertIn(str(self.paths["single_view_dir"]), captured[0])

    def test_cmd_contains_configs_path(self):
        self.paths["config_yaml"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["config_yaml"].touch()
        captured = []
        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_generate_condition(self.args, self.paths)
        self.assertIn("--configs=", captured[0])
        self.assertIn("config.yaml", captured[0])

    def test_single_view_dir_created_before_run(self):
        """single_view_dir must exist when the subprocess is called."""
        self.paths["config_yaml"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["config_yaml"].touch()
        sv = self.paths["single_view_dir"]
        self.assertFalse(sv.exists())
        with self._patch_subprocess():
            pipeline.step_generate_condition(self.args, self.paths)
        self.assertTrue(sv.is_dir())

    def test_done_dir_is_condition_output_not_single_view(self):
        """DONE marker must land in condition_output/, not in single_view_dir itself."""
        self.paths["config_yaml"].parent.mkdir(parents=True, exist_ok=True)
        self.paths["config_yaml"].touch()
        with self._patch_subprocess():
            pipeline.step_generate_condition(self.args, self.paths)
        cond_out = self.paths["single_view_dir"] / "condition_output"
        self.assertTrue((cond_out / "DONE").is_file())
        self.assertFalse((self.paths["single_view_dir"] / "DONE").is_file())


class TestStep3ComfyuiViews(_StepTestBase):

    def test_skips_when_done(self):
        self._mark_done(self.paths["single_view_dir"] / "gen_view")
        with self._patch_subprocess() as mock_run:
            pipeline.step_comfyui_views(self.args, self.paths)
        mock_run.assert_not_called()

    def test_exits_when_comfyui_dir_missing(self):
        self.args.comfyui_dir = None
        with self.assertRaises(SystemExit) as cm:
            with self._patch_subprocess():
                pipeline.step_comfyui_views(self.args, self.paths)
        self.assertEqual(cm.exception.code, 0)

    def test_runs_when_comfyui_dir_given(self):
        import tempfile
        with tempfile.TemporaryDirectory() as comfy:
            self.args.comfyui_dir = comfy
            with self._patch_subprocess() as mock_run:
                pipeline.step_comfyui_views(self.args, self.paths)
        mock_run.assert_called_once()

    def test_marks_done_when_comfyui_succeeds(self):
        import tempfile
        with tempfile.TemporaryDirectory() as comfy:
            self.args.comfyui_dir = comfy
            with self._patch_subprocess():
                pipeline.step_comfyui_views(self.args, self.paths)
        self.assertTrue(pipeline.is_done(self.paths["single_view_dir"] / "gen_view"))

    def test_cmd_contains_mesh_name(self):
        import tempfile
        with tempfile.TemporaryDirectory() as comfy:
            self.args.comfyui_dir = comfy
            captured = []
            def _capture(cmd, **kw):
                captured.append(cmd)
                return _ok_proc()
            with patch("subprocess.run", side_effect=_capture):
                pipeline.step_comfyui_views(self.args, self.paths)
        self.assertIn(self.args.mesh_name, captured[0])

    def test_canny_normal_appended_when_set(self):
        import tempfile
        with tempfile.TemporaryDirectory() as comfy:
            self.args.comfyui_dir = comfy
            self.args.canny_normal = True
            captured = []
            def _capture(cmd, **kw):
                captured.append(cmd)
                return _ok_proc()
            with patch("subprocess.run", side_effect=_capture):
                pipeline.step_comfyui_views(self.args, self.paths)
        self.assertIn("--canny_normal", captured[0])

    def test_canny_normal_absent_when_not_set(self):
        import tempfile
        with tempfile.TemporaryDirectory() as comfy:
            self.args.comfyui_dir = comfy
            self.args.canny_normal = False
            captured = []
            def _capture(cmd, **kw):
                captured.append(cmd)
                return _ok_proc()
            with patch("subprocess.run", side_effect=_capture):
                pipeline.step_comfyui_views(self.args, self.paths)
        self.assertNotIn("--canny_normal", captured[0])

    def test_canny_weight_appended_when_given(self):
        import tempfile
        with tempfile.TemporaryDirectory() as comfy:
            self.args.comfyui_dir = comfy
            self.args.canny_weight = 0.2
            captured = []
            def _capture(cmd, **kw):
                captured.append(cmd)
                return _ok_proc()
            with patch("subprocess.run", side_effect=_capture):
                pipeline.step_comfyui_views(self.args, self.paths)
        self.assertIn("--canny_weight 0.2", captured[0])

    def test_canny_weight_absent_when_none(self):
        import tempfile
        with tempfile.TemporaryDirectory() as comfy:
            self.args.comfyui_dir = comfy
            self.args.canny_weight = None
            captured = []
            def _capture(cmd, **kw):
                captured.append(cmd)
                return _ok_proc()
            with patch("subprocess.run", side_effect=_capture):
                pipeline.step_comfyui_views(self.args, self.paths)
        self.assertNotIn("--canny_weight", captured[0])

    def test_visualize_flag_and_port_added_when_set(self):
        import tempfile
        with tempfile.TemporaryDirectory() as comfy:
            self.args.comfyui_dir = comfy
            self.args.comfyui_visualize = True
            self.args.comfyui_port = 8188
            captured = []
            def _capture(cmd, **kw):
                captured.append(cmd)
                return _ok_proc()
            with patch("subprocess.run", side_effect=_capture):
                pipeline.step_comfyui_views(self.args, self.paths)
        self.assertIn("--visualize", captured[0])
        self.assertIn("--port 8188", captured[0])

    def test_visualize_absent_when_not_set(self):
        import tempfile
        with tempfile.TemporaryDirectory() as comfy:
            self.args.comfyui_dir = comfy
            self.args.comfyui_visualize = False
            captured = []
            def _capture(cmd, **kw):
                captured.append(cmd)
                return _ok_proc()
            with patch("subprocess.run", side_effect=_capture):
                pipeline.step_comfyui_views(self.args, self.paths)
        self.assertNotIn("--visualize", captured[0])

    def test_custom_comfyui_port(self):
        import tempfile
        with tempfile.TemporaryDirectory() as comfy:
            self.args.comfyui_dir = comfy
            self.args.comfyui_visualize = True
            self.args.comfyui_port = 9090
            captured = []
            def _capture(cmd, **kw):
                captured.append(cmd)
                return _ok_proc()
            with patch("subprocess.run", side_effect=_capture):
                pipeline.step_comfyui_views(self.args, self.paths)
        self.assertIn("--port 9090", captured[0])

    def test_requirements_install_skipped_by_default(self):
        import tempfile
        with tempfile.TemporaryDirectory() as comfy:
            self.args.comfyui_dir = comfy
            captured = []
            def _capture(cmd, **kw):
                captured.append(cmd)
                return _ok_proc()
            with patch("subprocess.run", side_effect=_capture):
                pipeline.step_comfyui_views(self.args, self.paths)
        self.assertNotIn("pip install -r requirements.txt", captured[0])
        self.assertIn("conda run -n gloss python generate_views.py", captured[0])

    def test_pip_install_runs_before_generation_when_enabled(self):
        import tempfile
        with tempfile.TemporaryDirectory() as comfy:
            self.args.comfyui_dir = comfy
            self.args.comfyui_install_requirements = True
            captured = []
            def _capture(cmd, **kw):
                captured.append(cmd)
                return _ok_proc()
            with patch("subprocess.run", side_effect=_capture):
                pipeline.step_comfyui_views(self.args, self.paths)
        self.assertIn("pip install -r requirements.txt", captured[0])
        pip_pos = captured[0].index("pip install")
        gen_pos = captured[0].index("generate_views.py")
        self.assertLess(pip_pos, gen_pos)


class TestStep4PostprocessMask(_StepTestBase):

    def test_skips_when_done(self):
        self._mark_done(self.paths["single_view_dir"] / "gen_view_masked")
        with self._patch_subprocess() as mock_run:
            pipeline.step_postprocess_mask(self.args, self.paths)
        mock_run.assert_not_called()

    def test_runs_when_not_done(self):
        with self._patch_subprocess() as mock_run:
            pipeline.step_postprocess_mask(self.args, self.paths)
        mock_run.assert_called_once()

    def test_marks_done_after_run(self):
        with self._patch_subprocess():
            pipeline.step_postprocess_mask(self.args, self.paths)
        self.assertTrue(pipeline.is_done(self.paths["single_view_dir"] / "gen_view_masked"))

    def test_cmd_passes_output_dir(self):
        captured = []
        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_postprocess_mask(self.args, self.paths)
        self.assertIn("--output_dir=", captured[0])
        self.assertIn(str(self.paths["single_view_dir"]), captured[0])


class TestStep5Decompose(_StepTestBase):

    def test_skips_when_done(self):
        self._mark_done(self.paths["single_view_dir"] / "gen_view_decomposite")
        with self._patch_subprocess() as mock_run:
            pipeline.step_decompose(self.args, self.paths)
        mock_run.assert_not_called()

    def test_exits_when_args_missing(self):
        # none of decomp_config/script/weights are set
        with self.assertRaises(SystemExit) as cm:
            pipeline.step_decompose(self.args, self.paths)
        self.assertEqual(cm.exception.code, 0)

    def test_exits_when_only_some_args_missing(self):
        self.args.decomp_config = "/cfg.yaml"
        # decomp_script and decomp_weights still None
        with self.assertRaises(SystemExit):
            pipeline.step_decompose(self.args, self.paths)

    def test_runs_when_all_args_given(self):
        self.args.decomp_config = "/cfg.yaml"
        self.args.decomp_script = "/infer.py"
        self.args.decomp_weights = "/weights.pt"
        with self._patch_subprocess() as mock_run:
            pipeline.step_decompose(self.args, self.paths)
        mock_run.assert_called_once()

    def test_marks_done_after_successful_run(self):
        self.args.decomp_config = "/cfg.yaml"
        self.args.decomp_script = "/infer.py"
        self.args.decomp_weights = "/weights.pt"
        with self._patch_subprocess():
            pipeline.step_decompose(self.args, self.paths)
        self.assertTrue(pipeline.is_done(self.paths["single_view_dir"] / "gen_view_decomposite"))

    def test_cmd_contains_input_and_save_dirs(self):
        self.args.decomp_config = "/cfg.yaml"
        self.args.decomp_script = "/infer.py"
        self.args.decomp_weights = "/weights.pt"
        sv = self.paths["single_view_dir"]
        captured = []
        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_decompose(self.args, self.paths)
        self.assertIn(str(sv / "gen_view_masked"), captured[0])
        self.assertIn(str(sv / "gen_view_decomposite"), captured[0])

    def test_diffrender_cmd_includes_retry_resolutions(self):
        self.args.diffrender_dir = "/diffusion-renderer"
        self.args.hf_home = "/hf-cache"
        self.args.diffrender_inference_res = "448,448"
        self.args.diffrender_fallback_inference_res = ["384,384", "256,256"]
        self.args.diffrender_inference_n_steps = 12
        captured = []

        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()

        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_decompose(self.args, self.paths)

        self.assertIn("conda run -n diff-render python", captured[0])
        self.assertIn("run_diffrender_decompose.py", captured[0])
        self.assertIn("--diffrender_dir=/diffusion-renderer", captured[0])
        self.assertIn("--inference_res=448,448", captured[0])
        self.assertIn("--fallback_inference_res 384,384 256,256", captured[0])
        self.assertIn("--inference_n_steps=12", captured[0])
        self.assertIn("--hf_home=/hf-cache", captured[0])


class TestStep6Superresolution(_StepTestBase):

    def test_skips_when_done(self):
        self._mark_done(self.paths["single_view_dir"] / "gen_view_super")
        with self._patch_subprocess() as mock_run:
            pipeline.step_superresolution(self.args, self.paths)
        mock_run.assert_not_called()

    def test_exits_when_invsr_dir_missing(self):
        self.args.invsr_dir = None
        with self.assertRaises(SystemExit) as cm:
            pipeline.step_superresolution(self.args, self.paths)
        self.assertEqual(cm.exception.code, 0)

    def test_runs_when_invsr_dir_given(self):
        import tempfile
        with tempfile.TemporaryDirectory() as invsr:
            self.args.invsr_dir = invsr
            with self._patch_subprocess() as mock_run:
                pipeline.step_superresolution(self.args, self.paths)
        mock_run.assert_called_once()

    def test_marks_done_after_run(self):
        import tempfile
        with tempfile.TemporaryDirectory() as invsr:
            self.args.invsr_dir = invsr
            with self._patch_subprocess():
                pipeline.step_superresolution(self.args, self.paths)
        self.assertTrue(pipeline.is_done(self.paths["single_view_dir"] / "gen_view_super"))

    def test_cmd_contains_decomposite_and_super_dirs(self):
        import tempfile
        sv = self.paths["single_view_dir"]
        with tempfile.TemporaryDirectory() as invsr:
            self.args.invsr_dir = invsr
            captured = []
            def _capture(cmd, **kw):
                captured.append(cmd)
                return _ok_proc()
            with patch("subprocess.run", side_effect=_capture):
                pipeline.step_superresolution(self.args, self.paths)
        self.assertIn("conda run -n invsr", captured[0])
        self.assertIn(str(sv / "gen_view_decomposite"), captured[0])
        self.assertIn(str(sv / "gen_view_super"), captured[0])


class TestStep7Backproject(_StepTestBase):

    def test_skips_when_done(self):
        self._mark_done(self.paths["single_view_dir"] / "textures_sr")
        with self._patch_subprocess() as mock_run:
            pipeline.step_backproject(self.args, self.paths)
        mock_run.assert_not_called()

    def test_runs_when_not_done(self):
        with self._patch_subprocess() as mock_run:
            pipeline.step_backproject(self.args, self.paths)
        mock_run.assert_called_once()

    def test_marks_done_after_run(self):
        with self._patch_subprocess():
            pipeline.step_backproject(self.args, self.paths)
        self.assertTrue(pipeline.is_done(self.paths["single_view_dir"] / "textures_sr"))

    def test_cmd_contains_texture_size(self):
        self.args.texture_size = 2048
        captured = []
        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_backproject(self.args, self.paths)
        self.assertIn("--texture_height=2048", captured[0])
        self.assertIn("--texture_width=2048", captured[0])

    def test_cmd_contains_mesh_fp(self):
        captured = []
        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_backproject(self.args, self.paths)
        self.assertIn("scene.gltf", captured[0])


class TestStep8Datagen(_StepTestBase):

    def test_skips_when_done(self):
        out = self.paths["expr_dir"] / "multi_view" / self.args.dataset_tag
        self._mark_done(out)
        with self._patch_subprocess() as mock_run:
            pipeline.step_datagen(self.args, self.paths)
        mock_run.assert_not_called()

    def test_runs_when_not_done(self):
        with self._patch_subprocess() as mock_run:
            pipeline.step_datagen(self.args, self.paths)
        mock_run.assert_called_once()

    def test_marks_done_after_run(self):
        with self._patch_subprocess():
            pipeline.step_datagen(self.args, self.paths)
        out = self.paths["expr_dir"] / "multi_view" / self.args.dataset_tag
        self.assertTrue(pipeline.is_done(out))

    def test_cmd_contains_global_root_dir(self):
        captured = []
        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_datagen(self.args, self.paths)
        self.assertIn("--global_root_dir=", captured[0])
        self.assertIn(str(self.paths["data_dir"]), captured[0])

    def test_cmd_contains_fov_and_camera_dist(self):
        self.args.fov_min = 0.3
        self.args.fov_max = 0.9
        self.args.camera_dist = 0.5
        captured = []
        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_datagen(self.args, self.paths)
        self.assertIn("--data.fov_min=0.3", captured[0])
        self.assertIn("--data.fov_max=0.9", captured[0])
        self.assertIn("--data.camera_dist=0.5", captured[0])

    def test_relative_paths_used_inside_cmd(self):
        """datagen.py must receive paths relative to data_dir, not absolute."""
        captured = []
        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_datagen(self.args, self.paths)
        # the mesh path passed as --data.mesh should be relative: expr_tag/mesh_name/...
        rel_prefix = f"{self.args.expr_tag}/{self.args.mesh_name}"
        self.assertIn(rel_prefix, captured[0])

    def test_mesh_subpath_is_respected_inside_cmd(self):
        self.args.mesh_subpath = "scene.gltf"
        self.paths = pipeline.build_paths(self.args)
        captured = []
        def _capture(cmd, **kw):
            captured.append(cmd)
            return _ok_proc()
        with patch("subprocess.run", side_effect=_capture):
            pipeline.step_datagen(self.args, self.paths)
        self.assertIn("--data.mesh=exp01/barrel/scene.gltf", captured[0])

    def test_custom_dataset_tag(self):
        self.args.dataset_tag = "my-dataset"
        out = self.paths["expr_dir"] / "multi_view" / "my-dataset"
        with self._patch_subprocess():
            pipeline.step_datagen(self.args, self.paths)
        self.assertTrue(pipeline.is_done(out))


# ===========================================================================
# _clear_done
# ===========================================================================

class TestClearDone(_StepTestBase):

    def _run_clear(self, step_num, dataset_tag=None):
        pipeline._clear_done(step_num, self.paths, dataset_tag=dataset_tag)

    def test_clears_existing_marker(self):
        sv = self.paths["single_view_dir"]
        for step_num, subdir in {
            1: self.paths["prompts_dir"],
            2: sv / "condition_output",
            3: sv / "gen_view",
            4: sv / "gen_view_masked",
            5: sv / "gen_view_decomposite",
            6: sv / "gen_view_super",
            7: sv / "textures_sr",
        }.items():
            with self.subTest(step=step_num):
                pipeline.mark_done(subdir)
                self.assertTrue(pipeline.is_done(subdir))
                self._run_clear(step_num)
                self.assertFalse(pipeline.is_done(subdir))

    def test_no_error_when_marker_absent(self):
        # Should not raise even if marker does not exist
        for step_num in range(1, 8):
            with self.subTest(step=step_num):
                self._run_clear(step_num)   # should not raise

    def test_step_8_returns_without_error_when_dataset_tag_missing(self):
        self._run_clear(8)  # must not raise

    def test_step_8_clears_existing_marker_when_dataset_tag_given(self):
        out = self.paths["expr_dir"] / "multi_view" / self.args.dataset_tag
        pipeline.mark_done(out)
        self.assertTrue(pipeline.is_done(out))
        self._run_clear(8, dataset_tag=self.args.dataset_tag)
        self.assertFalse(pipeline.is_done(out))


# ===========================================================================
# Integration: main() / argument parsing
# ===========================================================================

class TestMain(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def _base_argv(self):
        return [
            "--data_dir", str(self.root / "data"),
            "--expr_tag", "exp",
            "--mesh_name", "barrel",
        ]

    def _run_main(self, extra_argv=None, mock_runners=True):
        argv = self._base_argv() + (extra_argv or [])
        runners = {i: MagicMock() for i in range(1, 9)}
        ctx = patch.object(pipeline, "STEP_RUNNERS", runners)
        with patch("sys.argv", ["pipeline.py"] + argv), ctx:
            pipeline.main()
        return runners

    def test_all_steps_run_by_default(self):
        runners = self._run_main()
        for step in range(1, 9):
            runners[step].assert_called_once()

    def test_subset_of_steps(self):
        runners = self._run_main(["--steps", "3", "7"])
        runners[3].assert_called_once()
        runners[7].assert_called_once()
        for s in [1, 2, 4, 5, 6, 8]:
            runners[s].assert_not_called()

    def test_steps_run_in_ascending_order(self):
        call_order = []
        runners = {i: MagicMock(side_effect=lambda a, p, _i=i: call_order.append(_i))
                   for i in range(1, 9)}
        argv = self._base_argv() + ["--steps", "5", "2", "8"]
        with patch("sys.argv", ["pipeline.py"] + argv), \
             patch.object(pipeline, "STEP_RUNNERS", runners):
            pipeline.main()
        self.assertEqual(call_order, sorted(call_order))

    def test_invalid_step_exits(self):
        with self.assertRaises(SystemExit):
            self._run_main(["--steps", "99"])

    def test_force_clears_done_markers(self):
        """With --force, _clear_done is called for each requested step."""
        with patch.object(pipeline, "STEP_RUNNERS", {i: MagicMock() for i in range(1, 9)}), \
             patch.object(pipeline, "_clear_done") as mock_clear, \
             patch("sys.argv", ["p.py"] + self._base_argv() + ["--force", "--steps", "4", "5"]):
            pipeline.main()
        clear_step_nums = [c.args[0] for c in mock_clear.call_args_list]
        self.assertIn(4, clear_step_nums)
        self.assertIn(5, clear_step_nums)

    def test_force_passes_dataset_tag_to_clear_done(self):
        with patch.object(pipeline, "STEP_RUNNERS", {i: MagicMock() for i in range(1, 9)}), \
             patch.object(pipeline, "_clear_done") as mock_clear, \
             patch("sys.argv", ["p.py"] + self._base_argv() + ["--force", "--steps", "8", "--dataset_tag", "my-tag"]):
            pipeline.main()
        _, kwargs = mock_clear.call_args
        self.assertEqual(kwargs["dataset_tag"], "my-tag")

    def test_default_dataset_tag(self):
        """Ensure the default dataset_tag is passed through to step 8."""
        parsed_args = []
        original_run = {i: pipeline.STEP_RUNNERS[i] for i in range(1, 9)}
        def capture_step8(args, paths):
            parsed_args.append(args)
        runners = {i: MagicMock() for i in range(1, 9)}
        runners[8] = capture_step8
        with patch.object(pipeline, "STEP_RUNNERS", runners), \
             patch("sys.argv", ["p.py"] + self._base_argv() + ["--steps", "8"]):
            pipeline.main()
        self.assertEqual(parsed_args[0].dataset_tag, "cam0.25-fov0.4-0.8")

    def test_missing_required_args_exits(self):
        with self.assertRaises(SystemExit):
            with patch("sys.argv", ["p.py", "--expr_tag", "e", "--mesh_name", "m"]):
                pipeline.main()


# ===========================================================================
# Output verification – step 2 cabbage reference run
# ===========================================================================

class TestGenerateConditionOutputs(unittest.TestCase):
    """
    Verify the outputs of step 2 (generate_condition) against the reference run on
    the 'cabbage' mesh.

    These tests do NOT invoke the pipeline – they inspect files that were already
    produced by:

        python scripts/datagen/steps/generate_condition.py \\
            --mesh  .../cabbage/scene.gltf \\
            --prompts .../cabbage/prompts/cabbage.txt \\
            --configs .../mesh/config.yaml \\
            --output_dir .../cabbage/single_view/civitai2.0 \\
            --mode debug

    All tests are skipped automatically when the reference data is absent.
    """

    _BASE = Path(os.environ.get("GLOSS_DATA_DIR", "")) / "meshes" / "cabbage"
    COND_OUT = _BASE / "single_view" / "civitai2.0" / "condition_output"
    META_DIR = _BASE / "single_view" / "civitai2.0" / "meta"
    META_JSON = _BASE / "single_view" / "civitai2.0" / "meta.json"
    NUM_VIEWS = 10  # cabbage was generated with --mode debug

    _COND_SUBDIRS = ("normal", "geonormal", "canny-normal", "canny-geonormal", "depth", "mask")
    _COND_PREFIXES = {
        "normal": "normal",
        "geonormal": "geonormal",
        "canny-normal": "canny",
        "canny-geonormal": "canny",
        "depth": "depth",
        "mask": "mask",
    }

    @classmethod
    def setUpClass(cls):
        if not cls.COND_OUT.exists():
            raise unittest.SkipTest(
                f"Reference cabbage data not found at {cls.COND_OUT}"
            )

    # ------------------------------------------------------------------
    # Top-level structure
    # ------------------------------------------------------------------

    def test_done_marker_present(self):
        self.assertTrue((self.COND_OUT / "DONE").is_file())

    def test_condition_subdirs_exist(self):
        for subdir in self._COND_SUBDIRS:
            with self.subTest(subdir=subdir):
                self.assertTrue((self.COND_OUT / subdir).is_dir())

    def test_meta_dir_exists(self):
        self.assertTrue(self.META_DIR.is_dir())

    def test_meta_json_exists(self):
        self.assertTrue(self.META_JSON.is_file())

    def test_generate_condition_log_exists(self):
        log = self._BASE / "single_view" / "civitai2.0" / "generate_condition_log.txt"
        self.assertTrue(log.is_file())

    # ------------------------------------------------------------------
    # PNG file counts and naming convention
    # ------------------------------------------------------------------

    def test_each_condition_subdir_has_correct_png_count(self):
        for subdir in self._COND_SUBDIRS:
            with self.subTest(subdir=subdir):
                count = len(list((self.COND_OUT / subdir).glob("*.png")))
                self.assertEqual(count, self.NUM_VIEWS)

    def test_png_files_follow_naming_convention(self):
        for subdir, prefix in self._COND_PREFIXES.items():
            for i in range(self.NUM_VIEWS):
                fname = f"{prefix}{i:04d}.png"
                with self.subTest(subdir=subdir, file=fname):
                    self.assertTrue((self.COND_OUT / subdir / fname).is_file())

    def test_all_png_files_are_nonempty(self):
        for subdir in self._COND_SUBDIRS:
            for f in sorted((self.COND_OUT / subdir).glob("*.png")):
                with self.subTest(file=f.relative_to(self.COND_OUT)):
                    self.assertGreater(f.stat().st_size, 0)

    # ------------------------------------------------------------------
    # meta/ directory
    # ------------------------------------------------------------------

    def test_view_yml_count(self):
        ymls = list(self.META_DIR.glob("view*.yml"))
        self.assertEqual(len(ymls), self.NUM_VIEWS)

    def test_view_yml_naming(self):
        for i in range(self.NUM_VIEWS):
            with self.subTest(view=i):
                self.assertTrue((self.META_DIR / f"view{i:04d}.yml").is_file())

    def test_extrinsics_pt_exists(self):
        self.assertTrue((self.META_DIR / "extrinsics.pt").is_file())

    def test_intrinsics_pt_exists(self):
        self.assertTrue((self.META_DIR / "intrinsics.pt").is_file())

    # ------------------------------------------------------------------
    # view yml content
    # ------------------------------------------------------------------

    def _load_yml(self, idx: int) -> dict:
        import yaml
        with (self.META_DIR / f"view{idx:04d}.yml").open() as f:
            return yaml.safe_load(f)

    def test_view_yml_has_prompt_key(self):
        data = self._load_yml(0)
        self.assertIn("prompt", data)

    def test_view_yml_prompt_is_nonempty_string(self):
        for i in range(self.NUM_VIEWS):
            with self.subTest(view=i):
                prompt = self._load_yml(i)["prompt"]
                self.assertIsInstance(prompt, str)
                self.assertGreater(len(prompt.strip()), 0)

    def test_view_yml_has_camera_key(self):
        data = self._load_yml(0)
        self.assertIn("camera", data)

    def test_view_yml_intrinsics_fields_present(self):
        required = ("width", "height", "focal_x", "focal_y", "near", "far")
        for i in range(self.NUM_VIEWS):
            with self.subTest(view=i):
                intr = self._load_yml(i)["camera"]["intrinsics"]
                for field in required:
                    self.assertIn(field, intr)

    def test_view_yml_resolution_is_512(self):
        for i in range(self.NUM_VIEWS):
            with self.subTest(view=i):
                intr = self._load_yml(i)["camera"]["intrinsics"]
                self.assertEqual(intr["width"], 512)
                self.assertEqual(intr["height"], 512)

    def test_view_yml_focal_lengths_positive(self):
        for i in range(self.NUM_VIEWS):
            with self.subTest(view=i):
                intr = self._load_yml(i)["camera"]["intrinsics"]
                self.assertGreater(intr["focal_x"], 0)
                self.assertGreater(intr["focal_y"], 0)

    def test_view_yml_near_less_than_far(self):
        for i in range(self.NUM_VIEWS):
            with self.subTest(view=i):
                intr = self._load_yml(i)["camera"]["intrinsics"]
                self.assertLess(intr["near"], intr["far"])

    def test_view_yml_view_matrix_is_4x4(self):
        """view_matrix is stored as a (1, 4, 4) nested list; vm[0] is the 4×4 matrix."""
        for i in range(self.NUM_VIEWS):
            with self.subTest(view=i):
                vm = self._load_yml(i)["camera"]["extrinsics"]["view_matrix"]
                self.assertEqual(len(vm), 1)
                matrix = vm[0]
                self.assertEqual(len(matrix), 4)
                for row in matrix:
                    self.assertEqual(len(row), 4)

    def test_view_yml_view_matrix_values_are_numeric(self):
        vm = self._load_yml(0)["camera"]["extrinsics"]["view_matrix"]
        for row in vm[0]:
            for val in row:
                self.assertIsInstance(val, (int, float))

    def test_all_views_have_distinct_prompts_or_cameras(self):
        """No two consecutive views should be fully identical."""
        data0 = self._load_yml(0)
        data1 = self._load_yml(1)
        # Either different prompt or different camera matrix
        same_prompt = data0["prompt"] == data1["prompt"]
        same_vm = (
            data0["camera"]["extrinsics"]["view_matrix"]
            == data1["camera"]["extrinsics"]["view_matrix"]
        )
        self.assertFalse(same_prompt and same_vm,
                         "view0000 and view0001 are completely identical")

    # ------------------------------------------------------------------
    # meta.json
    # ------------------------------------------------------------------

    def _load_meta_json(self) -> dict:
        import json
        with self.META_JSON.open() as f:
            return json.load(f)

    def test_meta_json_required_keys_present(self):
        meta = self._load_meta_json()
        for key in ("mesh", "prompts", "output_dir", "mode", "resolution",
                    "fov_min", "fov_max", "azi_min", "azi_max",
                    "elev_min", "elev_max", "viewdist_min", "viewdist_max"):
            with self.subTest(key=key):
                self.assertIn(key, meta)

    def test_meta_json_mode_is_debug(self):
        self.assertEqual(self._load_meta_json()["mode"], "debug")

    def test_meta_json_resolution_is_512(self):
        self.assertEqual(self._load_meta_json()["resolution"], 512)

    def test_meta_json_mesh_is_gltf(self):
        mesh = self._load_meta_json()["mesh"]
        self.assertTrue(mesh.endswith(".gltf"), f"Unexpected mesh format: {mesh}")

    def test_meta_json_mesh_points_to_cabbage(self):
        mesh = self._load_meta_json()["mesh"]
        self.assertIn("cabbage", mesh)

    def test_meta_json_output_dir_matches_cond_parent(self):
        meta = self._load_meta_json()
        self.assertEqual(Path(meta["output_dir"]).resolve(), self.COND_OUT.parent.resolve())

    def test_meta_json_view_range_params_valid(self):
        meta = self._load_meta_json()
        self.assertLessEqual(meta["fov_min"], meta["fov_max"])
        self.assertLessEqual(meta["azi_min"], meta["azi_max"])
        self.assertLessEqual(meta["elev_min"], meta["elev_max"])
        self.assertLessEqual(meta["viewdist_min"], meta["viewdist_max"])


# ===========================================================================

if __name__ == "__main__":
    unittest.main()
