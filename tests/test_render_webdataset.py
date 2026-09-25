# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import io
import json
import shutil
import tarfile
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gloss.data.render_webdataset import (
    EnsembleMultiviewLocalMeshRendersWebdataset,
    EnsembleWebdatasetSourceSpec,
    configure_ensemble_multi_view_webdataset,
    discover_ensemble_webdataset_source_specs,
)

warnings.filterwarnings("ignore", message="unclosed file .*", category=ResourceWarning)


def _png_bytes(value, *, channels=3, size=(2, 2)):
    if channels == 1:
        image = Image.new("L", size, color=value)
    elif channels == 3:
        image = Image.new("RGB", size, color=(value, value, value))
    else:
        raise ValueError(f"Unsupported channel count: {channels}")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _png_bytes_from_array(array: np.ndarray):
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_condition_image(path: Path, value: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(_png_bytes(value, channels=3))


def _add_tar_member(handle: tarfile.TarFile, name: str, payload: bytes):
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    handle.addfile(info, io.BytesIO(payload))


def _write_wds_tar(path: Path, canonical_view_id: int, local_view_names, *, albedo_value: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as handle:
        for local_view_name in local_view_names:
            prefix = f"view{canonical_view_id:04d}/{local_view_name}"
            _add_tar_member(handle, f"{prefix}_albedo.png", _png_bytes(albedo_value, channels=3))
            _add_tar_member(handle, f"{prefix}_albedo_alpha.png", _png_bytes(255, channels=1))
            _add_tar_member(handle, f"{prefix}_geo_camera_normals.png", _png_bytes(192, channels=3))
            _add_tar_member(handle, f"{prefix}_relative_positions.png", _png_bytes(96, channels=3))


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _tensor_to_uint8_image(tensor: torch.Tensor, *, normalize_signed: bool = False):
    image = tensor.detach().cpu()
    if normalize_signed:
        image = image / 2.0 + 0.5
    image = image.clamp(0.0, 1.0)
    image = (image * 255.0).round().to(torch.uint8)
    return image.permute(1, 2, 0).numpy()


class TestRenderWebdatasetEnsemble(unittest.TestCase):
    REAL_OBJECTS_ROOT = Path(os.environ.get("GLOSS_DATA_DIR", "")) / "meshes" / "train_mesh"
    REAL_OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "render_webdataset_real"

    def _clear_persistent_visualization_outputs(self, output_dir: Path):
        viz_dir = output_dir / "viz"
        shutil.rmtree(viz_dir, ignore_errors=True)
        log_path = output_dir / "batch_log.jsonl"
        if log_path.exists():
            log_path.unlink()

    def _write_step_visualization(self, out_dir: Path, step: int, batch: dict):
        out_dir.mkdir(parents=True, exist_ok=True)
        batch_size = int(batch["view"].shape[0])
        row_images = []
        for batch_index in range(batch_size):
            view_img = _tensor_to_uint8_image(batch["view"][batch_index], normalize_signed=False)
            albedo_img = _tensor_to_uint8_image(batch["albedo"][batch_index], normalize_signed=True)
            row = Image.new("RGB", (view_img.shape[1] + albedo_img.shape[1], view_img.shape[0]))
            row.paste(Image.fromarray(view_img), (0, 0))
            row.paste(Image.fromarray(albedo_img), (view_img.shape[1], 0))
            row_images.append(row)

        combined = Image.new("RGB", (row_images[0].width, row_images[0].height * len(row_images)))
        for row_index, row in enumerate(row_images):
            combined.paste(row, (0, row_index * row.height))
        combined.save(out_dir / f"step_{step:03d}.png")

    def _create_object_fixture(
        self,
        root: Path,
        object_name: str,
        *,
        train_views,
        eval_views=(),
        create_wds=True,
    ):
        object_root = root / object_name
        _touch(object_root / "scene.gltf")

        single_view_root = object_root / "single_view" / "civitai2.0"
        (single_view_root / "gen_view_decomposite").mkdir(parents=True, exist_ok=True)
        with (single_view_root / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump({"exclude_from_training_indices": []}, handle)

        for view_id in list(train_views) + list(eval_views):
            _write_condition_image(
                single_view_root / "gen_view_decomposite" / f"view{view_id:04d}.basecolor.png",
                value=view_id,
            )

        if not create_wds:
            return object_root

        wds_root = object_root / "multi_view" / "cam0.25-fov0.4-0.8-wds"
        for view_id in list(train_views) + list(eval_views):
            _write_wds_tar(
                wds_root / f"view{view_id:04d}-0.tar",
                view_id,
                local_view_names=["local00", "local01"],
                albedo_value=view_id,
            )
        return object_root

    def test_discover_source_specs_resolves_complete_objects_and_skips_incomplete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_object_fixture(root, "object_a", train_views=[1, 2])
            self._create_object_fixture(root, "object_b", train_views=[5, 6])
            self._create_object_fixture(root, "object_incomplete", train_views=[9], create_wds=False)

            result = discover_ensemble_webdataset_source_specs(
                str(root),
                num_local_views=1,
            )

            self.assertEqual([source.name for source in result.sources], ["object_a", "object_b"])
            self.assertEqual(len(result.skipped), 1)
            self.assertEqual(result.skipped[0]["name"], "object_incomplete")

            source_a = result.sources[0]
            self.assertTrue(source_a.data_dir.endswith("object_a/multi_view/cam0.25-fov0.4-0.8-wds"))
            self.assertTrue(
                source_a.view_dir.endswith("object_a/single_view/civitai2.0/gen_view_decomposite")
            )
            self.assertEqual(source_a.num_views, 2)
            self.assertEqual(source_a.num_local_views, 1)
            self.assertEqual(source_a.metadata["single_view_image_dir"], "gen_view_decomposite")

    def test_configure_ensemble_dataset_uses_per_object_split_and_correct_condition_view(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._create_object_fixture(root, "object_a", train_views=[1], eval_views=[2])
            self._create_object_fixture(root, "object_b", train_views=[5], eval_views=[6])

            eval_root = root / "eval_data" / "eval_cond_views"
            _touch(eval_root / "object_a" / "view0002.txt")
            _touch(eval_root / "object_b" / "view0006.txt")

            discovery = discover_ensemble_webdataset_source_specs(
                str(root),
                object_names=["object_a", "object_b"],
                num_local_views=1,
            )
            self.assertEqual(len(discovery.sources), 2)

            train_dataset, eval_datasets, split_results = configure_ensemble_multi_view_webdataset(
                discovery.sources,
                indices_dir=str(root / "indices"),
                batch_size=2,
                num_eval=1,
                random_seed=0,
                global_root_dir=str(root),
                shuffle=False,
                eval_shuffle=False,
            )

            self.assertEqual(split_results["object_a"]["train_indices"], [1])
            self.assertEqual(split_results["object_a"]["eval_indices"], [2])
            self.assertEqual(split_results["object_b"]["train_indices"], [5])
            self.assertEqual(split_results["object_b"]["eval_indices"], [6])

            train_loader = train_dataset.get_dataloader(global_batch_size=1, num_workers=0)
            train_samples = list(train_loader)
            self.assertEqual(len(train_samples), 2)

            first_sample, second_sample = train_samples
            self.assertEqual(first_sample["source_name"], "object_a")
            self.assertEqual(first_sample["dataset_id"], 1)
            self.assertEqual(first_sample["sample_uid"], "object_a:view0001")
            self.assertEqual(first_sample["local_views"], ["view0001/local00", "view0001/local01"])
            self.assertTrue(
                torch.allclose(
                    first_sample["view"],
                    torch.full((2, 3, 2, 2), 1.0 / 255.0, dtype=first_sample["view"].dtype),
                )
            )

            self.assertEqual(second_sample["source_name"], "object_b")
            self.assertEqual(second_sample["dataset_id"], 5)
            self.assertEqual(second_sample["sample_uid"], "object_b:view0005")
            self.assertEqual(second_sample["local_views"], ["view0005/local00", "view0005/local01"])
            self.assertTrue(
                torch.allclose(
                    second_sample["view"],
                    torch.full((2, 3, 2, 2), 5.0 / 255.0, dtype=second_sample["view"].dtype),
                )
            )

            eval_a_loader = eval_datasets["object_a"].get_dataloader(global_batch_size=1, num_workers=0)
            eval_a_sample = next(iter(eval_a_loader))
            self.assertEqual(eval_a_sample["dataset_id"], 2)
            self.assertEqual(eval_a_sample["local_views"], ["view0002/local00", "view0002/local01"])
            self.assertTrue(
                torch.allclose(
                    eval_a_sample["view"],
                    torch.full((2, 3, 2, 2), 2.0 / 255.0, dtype=eval_a_sample["view"].dtype),
                )
            )

            eval_b_loader = eval_datasets["object_b"].get_dataloader(global_batch_size=1, num_workers=0)
            eval_b_sample = next(iter(eval_b_loader))
            self.assertEqual(eval_b_sample["dataset_id"], 6)
            self.assertEqual(eval_b_sample["local_views"], ["view0006/local00", "view0006/local01"])
            self.assertTrue(
                torch.allclose(
                    eval_b_sample["view"],
                    torch.full((2, 3, 2, 2), 6.0 / 255.0, dtype=eval_b_sample["view"].dtype),
                )
            )

    def test_direct_ensemble_dataloader_preserves_source_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            view_dir_a = root / "object_a_views"
            view_dir_b = root / "object_b_views"
            _write_condition_image(view_dir_a / "view0003.basecolor.png", 33)
            _write_condition_image(view_dir_b / "view0007.basecolor.png", 77)

            tar_a = root / "object_a" / "view0003-0.tar"
            tar_b = root / "object_b" / "view0007-0.tar"
            _write_wds_tar(tar_a, 3, ["local00", "local01"], albedo_value=3)
            _write_wds_tar(tar_b, 7, ["local00", "local01"], albedo_value=7)

            dataset = EnsembleMultiviewLocalMeshRendersWebdataset(
                [
                    EnsembleWebdatasetSourceSpec(
                        name="object_a",
                        data_dir=[str(tar_a)],
                        num_views=1,
                        num_local_views=1,
                        view_dir=str(view_dir_a),
                        metadata={"family": "train"},
                    ),
                    EnsembleWebdatasetSourceSpec(
                        name="object_b",
                        data_dir=[str(tar_b)],
                        num_views=1,
                        num_local_views=1,
                        view_dir=str(view_dir_b),
                        metadata={"family": "train"},
                    ),
                ],
                batch_size=2,
                shuffle=False,
            )

            loader = dataset.get_dataloader(global_batch_size=1, num_workers=0)
            samples = list(loader)
            self.assertEqual([sample["source_name"] for sample in samples], ["object_a", "object_b"])
            self.assertEqual(samples[0]["source_metadata"], {"family": "train"})
            self.assertEqual(samples[0]["source_index"], 0)
            self.assertEqual(samples[1]["source_index"], 1)
            self.assertEqual(samples[0]["local_views"], ["view0003/local00", "view0003/local01"])
            self.assertEqual(samples[1]["local_views"], ["view0007/local00", "view0007/local01"])
            self.assertTrue(
                torch.allclose(
                    samples[0]["view"],
                    torch.full((2, 3, 2, 2), 33.0 / 255.0, dtype=samples[0]["view"].dtype),
                )
            )
            self.assertTrue(
                torch.allclose(
                    samples[1]["view"],
                    torch.full((2, 3, 2, 2), 77.0 / 255.0, dtype=samples[1]["view"].dtype),
                )
            )

    def test_degenerate_albedo_alpha_falls_back_to_visible_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            view_dir = root / "object_a_views"
            _write_condition_image(view_dir / "view0003.basecolor.png", 33)

            tar_path = root / "object_a" / "view0003-0.tar"
            tar_path.parent.mkdir(parents=True, exist_ok=True)
            normals = np.full((2, 2, 3), 192, dtype=np.uint8)
            normals[0, 0] = 128
            bad_alpha = np.zeros((2, 2), dtype=np.uint8)

            with tarfile.open(tar_path, "w") as handle:
                prefix = "view0003/local00"
                _add_tar_member(handle, f"{prefix}_albedo.png", _png_bytes(3, channels=3))
                _add_tar_member(handle, f"{prefix}_albedo_alpha.png", _png_bytes_from_array(bad_alpha))
                _add_tar_member(handle, f"{prefix}_geo_camera_normals.png", _png_bytes_from_array(normals))
                _add_tar_member(handle, f"{prefix}_relative_positions.png", _png_bytes(96, channels=3))

            dataset = EnsembleMultiviewLocalMeshRendersWebdataset(
                [
                    EnsembleWebdatasetSourceSpec(
                        name="object_a",
                        data_dir=[str(tar_path)],
                        num_views=1,
                        num_local_views=1,
                        view_dir=str(view_dir),
                    ),
                ],
                batch_size=1,
                shuffle=False,
            )

            sample = next(iter(dataset.get_dataloader(global_batch_size=1, num_workers=0)))
            self.assertTrue(torch.equal(sample["albedo_alpha"], sample["background_alpha"]))
            self.assertEqual(sample["albedo_alpha"][0, 0, 0, 0].item(), -1.0)
            self.assertTrue(torch.all(sample["albedo_alpha"][0, 0, :, 1:] == 1.0))
            self.assertTrue(torch.all(sample["albedo_alpha"][0, 0, 1, :] == 1.0))

    def test_ensemble_dataloader_logs_and_visualizes_50_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            view_dir_a = root / "object_a_views"
            view_dir_b = root / "object_b_views"
            _write_condition_image(view_dir_a / "view0003.basecolor.png", 33)
            _write_condition_image(view_dir_b / "view0007.basecolor.png", 77)

            tar_a = root / "object_a" / "view0003-0.tar"
            tar_b = root / "object_b" / "view0007-0.tar"
            _write_wds_tar(tar_a, 3, ["local00", "local01"], albedo_value=3)
            _write_wds_tar(tar_b, 7, ["local00", "local01"], albedo_value=7)

            dataset = EnsembleMultiviewLocalMeshRendersWebdataset(
                [
                    EnsembleWebdatasetSourceSpec(
                        name="object_a",
                        data_dir=[str(tar_a)],
                        num_views=1,
                        num_local_views=25,
                        view_dir=str(view_dir_a),
                        metadata={"family": "train"},
                    ),
                    EnsembleWebdatasetSourceSpec(
                        name="object_b",
                        data_dir=[str(tar_b)],
                        num_views=1,
                        num_local_views=25,
                        view_dir=str(view_dir_b),
                        metadata={"family": "train"},
                    ),
                ],
                batch_size=1,
                shuffle=True,
            )

            output_dir = root / "ensemble_step_logs"
            viz_dir = output_dir / "viz"
            log_path = output_dir / "batch_log.jsonl"
            output_dir.mkdir(parents=True, exist_ok=True)

            loader = dataset.get_dataloader(global_batch_size=1, num_workers=0)
            expected_local_views = {
                "object_a": {"view0003/local00", "view0003/local01"},
                "object_b": {"view0007/local00", "view0007/local01"},
            }
            expected_dataset_ids = {"object_a": 3, "object_b": 7}

            step_entries = []
            with log_path.open("w", encoding="utf-8") as handle:
                for step, batch in enumerate(loader):
                    entry = {
                        "step": step,
                        "source_name": batch["source_name"],
                        "dataset_id": int(batch["dataset_id"]),
                        "sample_uid": batch["sample_uid"],
                        "local_views": list(batch["local_views"]),
                    }
                    handle.write(json.dumps(entry) + "\n")
                    handle.flush()
                    step_entries.append(entry)
                    self._write_step_visualization(viz_dir, step, batch)

            self.assertEqual(len(step_entries), 50)
            self.assertEqual(len(list(viz_dir.glob("step_*.png"))), 50)

            seen_sources = {entry["source_name"] for entry in step_entries}
            self.assertEqual(seen_sources, {"object_a", "object_b"})

            for step, entry in enumerate(step_entries):
                source_name = entry["source_name"]
                self.assertEqual(entry["step"], step)
                self.assertEqual(entry["dataset_id"], expected_dataset_ids[source_name])
                self.assertEqual(entry["sample_uid"], f"{source_name}:view{expected_dataset_ids[source_name]:04d}")
                self.assertEqual(len(entry["local_views"]), 1)
                self.assertIn(entry["local_views"][0], expected_local_views[source_name])

            self.assertTrue((viz_dir / "step_000.png").is_file())
            self.assertTrue((viz_dir / "step_049.png").is_file())

    def test_real_data_ensemble_logs_and_visualizes_50_steps(self):
        if not self.REAL_OBJECTS_ROOT.is_dir():
            raise unittest.SkipTest(f"Real dataset root is unavailable: {self.REAL_OBJECTS_ROOT}")

        object_names = ["brick", "cabbage", "croissant"]
        discovery = discover_ensemble_webdataset_source_specs(
            str(self.REAL_OBJECTS_ROOT),
            object_names=object_names,
        )
        discovered_names = [source.name for source in discovery.sources]
        if discovered_names != object_names:
            raise unittest.SkipTest(
                f"Expected real sources {object_names}, discovered {discovered_names}, skipped={discovery.skipped}"
            )

        output_dir = self.REAL_OUTPUT_ROOT
        viz_dir = output_dir / "viz"
        log_path = output_dir / "batch_log.jsonl"
        indices_dir = output_dir / "indices"
        output_dir.mkdir(parents=True, exist_ok=True)
        self._clear_persistent_visualization_outputs(output_dir)

        train_dataset, eval_datasets, split_results = configure_ensemble_multi_view_webdataset(
            discovery.sources,
            indices_dir=str(indices_dir),
            batch_size=8,
            num_eval=50,
            random_seed=0,
            global_root_dir=str(self.REAL_OBJECTS_ROOT),
            shuffle=True,
            eval_shuffle=False,
        )

        self.assertEqual(set(eval_datasets.keys()), set(object_names))
        self.assertEqual(set(split_results.keys()), set(object_names))
        source_by_name = {source.name: source for source in discovery.sources}

        loader = train_dataset.get_dataloader(global_batch_size=1, num_workers=0)
        step_entries = []
        with log_path.open("w", encoding="utf-8") as handle:
            for step, batch in enumerate(loader):
                source_name = batch["source_name"]
                dataset_id = int(batch["dataset_id"])
                sample_uid = batch["sample_uid"]
                local_views = list(batch["local_views"])
                entry = {
                    "step": step,
                    "source_name": source_name,
                    "dataset_id": dataset_id,
                    "sample_uid": sample_uid,
                    "local_views": local_views,
                }
                handle.write(json.dumps(entry) + "\n")
                handle.flush()
                step_entries.append(entry)
                self._write_step_visualization(viz_dir, step, batch)

                self.assertIn(source_name, object_names)
                self.assertIn(dataset_id, split_results[source_name]["train_indices"])
                self.assertEqual(sample_uid, f"{source_name}:view{dataset_id:04d}")
                self.assertEqual(len(local_views), 8)
                self.assertTrue(all(local_view.startswith(f"view{dataset_id:04d}/") for local_view in local_views))

                condition_path = Path(source_by_name[source_name].view_dir) / f"view{dataset_id:04d}.basecolor.png"
                self.assertTrue(condition_path.is_file(), f"Missing condition image {condition_path}")
                expected_view = torch.from_numpy(
                    np.array(Image.open(condition_path).convert("RGB"), dtype="float32")
                ).permute(2, 0, 1) / 255.0
                self.assertTrue(
                    torch.allclose(batch["view"], expected_view.unsqueeze(0).repeat(8, 1, 1, 1), atol=1e-6),
                    f"Condition image mismatch for {source_name} view {dataset_id}",
                )

        self.assertEqual(len(step_entries), 50)
        self.assertEqual(len(list(viz_dir.glob("step_*.png"))), 50)
        self.assertTrue((viz_dir / "step_000.png").is_file())
        self.assertTrue((viz_dir / "step_049.png").is_file())

        seen_sources = {entry["source_name"] for entry in step_entries}
        self.assertEqual(seen_sources, set(object_names))


if __name__ == "__main__":
    unittest.main()
