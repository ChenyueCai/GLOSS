# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import math
import statistics
from pathlib import Path

import kaolin
import torch
import torchvision
import torchvision.transforms as transforms
from dreamsim import dreamsim
from omegaconf import OmegaConf
from tqdm import tqdm

from gloss.data.render_dataloader import FovSampler
from gloss.utils.eval_utils import extract_view_numbers, save_dict_to_csv
from gloss.utils.kaolin_utils import camera_from_meta, load_mesh
from gloss.utils.render import make_camera_from_extr_intr
from gloss.utils.render_fast import custom_mesh_batched_render
from gloss.utils.single_view import SingleViewCameraExtrinsicsSampler
from gloss.utils.paths import get_data_dir


OBJECT_LIST = [
    "brick",
    "cabbage",
    "croissant",
    "dirty_tire",
    "fire_hydrant",
    "gourd",
    "koi_fish",
    "rusty_barrel_metal",
    "sea_urchin_shell",
    "turtle",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute DreamSim for texture completion outputs.")
    parser.add_argument("--camera_dist", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--data-dir", type=str, default=None, help="Data root (default: $GLOSS_DATA_DIR)")
    parser.add_argument(
        "--test-data-dir",
        type=str,
        default=None,
        help="Root containing test_cond_views/, test_textures_sr/, and test_metas/. Defaults to <data-dir>/test_data.",
    )
    parser.add_argument("--expr-dir", type=str, required=True, help="Completion experiment root.")
    parser.add_argument("--object-name", type=str, default=None, help="Object to evaluate.")
    parser.add_argument(
        "--object-id",
        type=int,
        default=0,
        help="Legacy object index. Used only when --object-name is omitted.",
    )
    parser.add_argument(
        "--expr-name",
        type=str,
        default=None,
        help="Label written into the CSV. Defaults to the expr-dir basename.",
    )
    parser.add_argument(
        "--completed-dir",
        type=str,
        default=None,
        help="Optional directory containing flat completion textures as viewXXXX.png. Defaults to <expr-dir>/<object-name>.",
    )
    parser.add_argument(
        "--camera-cache-dir",
        type=str,
        default=None,
        help="Optional shared cache directory for per-view DreamSim camera samples.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--view-ids",
        type=str,
        default=None,
        help="Optional comma-separated list of view ids to restrict evaluation to.",
    )
    parser.add_argument(
        "--mesh-path",
        type=str,
        default=None,
        help="Optional explicit mesh path. Overrides <data-dir>/meshes/<object-name>/scene.gltf.",
    )
    return parser.parse_args()


def resolve_object_name(args):
    if args.object_name:
        return args.object_name
    try:
        return OBJECT_LIST[args.object_id]
    except IndexError as exc:
        raise ValueError(f"object_id {args.object_id} is out of range for {len(OBJECT_LIST)} objects") from exc


def load_cached_payload(cache_path: Path):
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu")
    return None


def save_cached_payload(cache_path: Path, payload) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)


def main():
    args = parse_args()

    data_dir = Path(args.data_dir or get_data_dir()).resolve()
    test_data_dir = Path(args.test_data_dir).resolve() if args.test_data_dir else data_dir / "test_data"
    expr_dir = Path(args.expr_dir).resolve()
    object_name = resolve_object_name(args)
    expr_name = args.expr_name or expr_dir.name

    completed_dir = Path(args.completed_dir).resolve() if args.completed_dir else expr_dir / object_name
    gt_texture_dir = test_data_dir / "test_textures_sr" / object_name
    cond_view_dir = test_data_dir / "test_cond_views" / object_name
    meta_dir = test_data_dir / "test_metas" / object_name
    mesh_path = Path(args.mesh_path).resolve() if args.mesh_path else data_dir / "meshes" / object_name / "scene.gltf"
    output_path = expr_dir / object_name / "metrics" / "dreamsim.csv"
    camera_cache_dir = (
        Path(args.camera_cache_dir).resolve()
        if args.camera_cache_dir
        else expr_dir / object_name / "metrics" / "camera_cache"
    )

    missing = [str(path) for path in [completed_dir, gt_texture_dir, cond_view_dir, meta_dir, mesh_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required DreamSim inputs:\n" + "\n".join(missing))

    view_ids = [f"{num:04d}" for num in extract_view_numbers(str(gt_texture_dir))]
    if args.view_ids:
        wanted = {f"{int(v.strip()):04d}" for v in args.view_ids.split(",") if v.strip()}
        view_ids = [v for v in view_ids if v in wanted]
    if not view_ids:
        raise ValueError(f"No test views found under {gt_texture_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = load_mesh(str(mesh_path))

    batch_size = args.batch_size
    device = args.device

    model, _ = dreamsim(pretrained=True, device=device)
    resize_to_model = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
    resize_view = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC)

    all_dreamsim = []
    for view_id in tqdm(view_ids):
        completed_texture_path = completed_dir / f"view{view_id}.png"
        gen_texture_path = gt_texture_dir / f"view{view_id}.png"
        view_path = cond_view_dir / f"view{view_id}.basecolor.png"
        meta_path = meta_dir / f"view{view_id}.yml"

        missing_view_inputs = [
            str(path)
            for path in [completed_texture_path, gen_texture_path, view_path, meta_path]
            if not path.exists()
        ]
        if missing_view_inputs:
            raise FileNotFoundError("Missing DreamSim inputs:\n" + "\n".join(missing_view_inputs))

        gt_view = torchvision.io.read_image(str(view_path)).to(mesh.faces.device)[None] / 255.0
        gt_view = resize_view(gt_view)
        view_camera_meta = OmegaConf.load(str(meta_path))
        view_camera = camera_from_meta(view_camera_meta["camera"]).to(device)

        albedo = kaolin.io.utils.read_image(str(completed_texture_path)).to(mesh.faces.device).contiguous() * 2 - 1
        gen_albedo = kaolin.io.utils.read_image(str(gen_texture_path)).to(mesh.faces.device).contiguous() * 2 - 1

        batch_cameras = kaolin.render.camera.Camera.cat([view_camera])
        view_render = custom_mesh_batched_render(
            batch_cameras,
            mesh,
            gen_albedo,
            None,
            requires_positions=False,
            process_as_albedo=False,
            backend="cuda",
        )
        view_mask = view_render["mask"] == 1
        gt_view = gt_view * view_mask.permute(0, 3, 1, 2)

        resolution = 256
        dreamsim_cache = camera_cache_dir / f"dreamsim_view{view_id}_n{args.num_samples}.pt"
        cached_dreamsim = load_cached_payload(dreamsim_cache)
        if cached_dreamsim is None:
            torch.manual_seed(int(view_id))

            intrinsics_sampler = FovSampler(1.0, 1.0)
            azim_range = [0.0, 3.14]
            elev_range = [-1.57, 1.57]
            view_dist_range = [0.9, 1.1]
            extrinsics_sampler = SingleViewCameraExtrinsicsSampler(azim_range, elev_range, view_dist_range)

            intr_samples, extr_samples = [], []
            resize = torchvision.transforms.Resize((gen_albedo.shape[0], gen_albedo.shape[1]))
            normal_map = resize(mesh.materials[0].chw().normals_texture).permute(1, 2, 0)
            count = 0

            while len(intr_samples) < args.num_samples and count < 500:
                count += 1
                intr_batch = intrinsics_sampler.generate(32)
                extr_batch = extrinsics_sampler.generate(32)
                cameras = [
                    make_camera_from_extr_intr(extr_batch[i], intr_batch[i], resolution=resolution, device=device)
                    for i in range(32)
                ]
                batch_cameras = kaolin.render.camera.Camera.cat(cameras)
                rendered = custom_mesh_batched_render(
                    batch_cameras,
                    mesh,
                    gen_albedo,
                    normal_map,
                    requires_positions=False,
                    process_as_albedo=False,
                )
                albedo_alpha = rendered["textured"][..., 3:] > -1
                background_alpha = (torch.abs(rendered["camera_normals"]) < 0.01).all(dim=-1, keepdim=True).float()
                fully_valid = albedo_alpha.sum((1, 2, 3)) < (resolution * resolution - background_alpha.sum((1, 2, 3))) * 0.3
                valid_camera_idx = torch.argwhere(fully_valid)
                print("valid patches", len(intr_samples), "new patches", len(valid_camera_idx))
                for idx in valid_camera_idx:
                    intr_samples.append(intr_batch[idx.item()])
                    extr_samples.append(extr_batch[idx.item()])
                    if len(intr_samples) == args.num_samples:
                        break

            if len(intr_samples) < args.num_samples:
                print("skipping view", view_id)
                continue

            intr_samples = torch.stack(intr_samples, 0)
            extr_samples = torch.stack(extr_samples, 0)
            save_cached_payload(
                dreamsim_cache,
                {"intr_samples": intr_samples, "extr_samples": extr_samples},
            )
        else:
            intr_samples = cached_dreamsim["intr_samples"]
            extr_samples = cached_dreamsim["extr_samples"]

        dreamsim_scores = []
        num_batches = int(math.ceil(args.num_samples / batch_size))
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, args.num_samples)
            batch_cameras = [
                make_camera_from_extr_intr(
                    extr_samples[j],
                    intr_samples[j],
                    resolution=resolution,
                    device=mesh.faces.device,
                )
                for j in range(start_idx, end_idx)
            ]
            num_samples = len(batch_cameras)
            batch_cameras = kaolin.render.camera.Camera.cat(batch_cameras)
            pred_batch = custom_mesh_batched_render(
                batch_cameras,
                mesh,
                albedo,
                None,
                requires_positions=False,
                process_as_albedo=False,
                backend="cuda",
            )
            gt = gt_view.repeat(num_samples, 1, 1, 1)
            pred_mask = pred_batch["mask"] == 1
            pred = (pred_batch["textured"][..., :3] / 2 + 0.5) * pred_mask
            pred = pred.permute(0, 3, 1, 2)
            pred, gt = resize_to_model(pred), resize_to_model(gt)
            if batch_idx == 0:
                log = torch.cat([gt[0], pred[0]], dim=-1)
                torchvision.utils.save_image(log, f"log-dreamsim-{batch_idx}.png")
            dreamsim_scores.append(model(pred, gt).mean().item())

        dreamsim_score = torch.mean(torch.tensor(dreamsim_scores)).item()
        print("DREAMSIM", dreamsim_score)
        all_dreamsim.append(dreamsim_score)

    save_dict = {
        "eval-name": f"{expr_name}-0.3",
        "avg-dreamsim": statistics.mean(all_dreamsim),
        "eval-dreamsim": all_dreamsim,
    }
    save_dict_to_csv(save_dict, str(output_path), False)


if __name__ == "__main__":
    main()
