# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import os
import statistics
from pathlib import Path

import kaolin
import lpips
import torch
import torchvision
from torcheval.metrics import FrechetInceptionDistance
from tqdm import tqdm

from gloss.data.render_dataloader import FovSampler, LocalCameraExtrinsicsSampler
from gloss.utils.eval_utils import extract_view_numbers, save_dict_to_csv
from gloss.utils.kaolin_utils import load_mesh
from gloss.utils.render import make_camera_from_extr_intr
from gloss.utils.render_fast import custom_mesh_batched_render
from gloss.utils.single_view import get_valid_faces_from_texture
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
    parser = argparse.ArgumentParser(description="Compute LPIPS and FID for texture completion outputs.")
    parser.add_argument("--camera_dist", type=float, default=0.25)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data-dir", type=str, default=None, help="Data root (default: $GLOSS_DATA_DIR)")
    parser.add_argument(
        "--test-data-dir",
        type=str,
        default=None,
        help="Root containing test_cond_views/, test_textures_sr/, and test_metas/. Defaults to <data-dir>/test_data.",
    )
    parser.add_argument(
        "--gt-texture-dir",
        type=str,
        default=None,
        help="Directory of reference textures viewXXXX.png. Overrides <test-data-dir>/test_textures_sr/<object-name>.",
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
        help="Optional shared cache directory for per-view sampled evaluation cameras.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--mesh-path",
        type=str,
        default=None,
        help="Optional explicit mesh path. Overrides <data-dir>/meshes/<object-name>/scene.gltf.",
    )
    parser.add_argument(
        "--view-ids",
        type=str,
        default=None,
        help="Optional comma-separated list of view ids (e.g. '200,201,205') to restrict evaluation to.",
    )
    return parser.parse_args()


def resolve_object_name(args):
    if args.object_name:
        return args.object_name
    try:
        return OBJECT_LIST[args.object_id]
    except IndexError as exc:
        raise ValueError(f"object_id {args.object_id} is out of range for {len(OBJECT_LIST)} objects") from exc


def cache_suffix(*, num_samples: int, camera_dist: float) -> str:
    return f"n{num_samples}_dist{camera_dist:g}".replace(".", "p")


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
    gt_texture_dir = (Path(args.gt_texture_dir).resolve() if args.gt_texture_dir
                      else test_data_dir / "test_textures_sr" / object_name)
    mesh_path = Path(args.mesh_path).resolve() if args.mesh_path else data_dir / "meshes" / object_name / "scene.gltf"
    output_dir = expr_dir / object_name / "metrics" / "lpips-fid"
    output_path = expr_dir / object_name / "metrics" / "lpips-fid.csv"
    camera_cache_dir = (
        Path(args.camera_cache_dir).resolve()
        if args.camera_cache_dir
        else expr_dir / object_name / "metrics" / "camera_cache"
    )

    missing = [str(path) for path in [completed_dir, gt_texture_dir, mesh_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required completion metric inputs:\n" + "\n".join(missing))

    view_ids = [f"{num:04d}" for num in extract_view_numbers(str(gt_texture_dir))]
    if args.view_ids:
        wanted = {f"{int(v.strip()):04d}" for v in args.view_ids.split(",") if v.strip()}
        view_ids = [v for v in view_ids if v in wanted]
    if not view_ids:
        raise ValueError(f"No test views found under {gt_texture_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = load_mesh(str(mesh_path))

    batch_size = args.batch_size
    device = args.device

    all_lpips = []
    for view_id in tqdm(view_ids):
        completed_texture_path = completed_dir / f"view{view_id}.png"
        gen_texture_path = gt_texture_dir / f"view{view_id}.png"
        if not completed_texture_path.exists():
            raise FileNotFoundError(f"Missing completed texture: {completed_texture_path}")

        albedo = kaolin.io.utils.read_image(str(completed_texture_path)).to(mesh.faces.device).contiguous() * 2 - 1
        gen_albedo = kaolin.io.utils.read_image(str(gen_texture_path)).to(mesh.faces.device).contiguous() * 2 - 1

        resolution = 256
        fov_min, fov_max = 0.4, 0.8
        lpips_cache = camera_cache_dir / f"lpips_view{view_id}_{cache_suffix(num_samples=args.num_samples, camera_dist=args.camera_dist)}.pt"
        cached_lpips = load_cached_payload(lpips_cache)
        if cached_lpips is None:
            torch.manual_seed(int(view_id))
            intrinsics_sampler = FovSampler(fov_min, fov_max)
            intr_samples = intrinsics_sampler.generate(args.num_samples)

            extrinsics_camera_sampler = LocalCameraExtrinsicsSampler(mesh, args.camera_dist)
            sampling_weights = torch.zeros(mesh.faces.shape[0])
            valid_face_ids = get_valid_faces_from_texture(mesh, gen_albedo.permute(2, 0, 1).unsqueeze(0), all_filled=True)
            sampling_weights[valid_face_ids] = 1.0
            if sampling_weights.sum() <= 0:
                raise ValueError(f"No valid face ids found for {object_name} view {view_id}")
            extrinsics_camera_sampler.set_sampling_weights(sampling_weights)
            extr_samples = extrinsics_camera_sampler.generate(args.num_samples)
            save_cached_payload(
                lpips_cache,
                {"intr_samples": intr_samples, "extr_samples": extr_samples},
            )
        else:
            intr_samples = cached_lpips["intr_samples"]
            extr_samples = cached_lpips["extr_samples"]

        lpip = lpips.LPIPS(net="vgg").to(device)
        lpip_scores = []
        num_batches = int(math.ceil(args.num_samples / batch_size))
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, args.num_samples)
            batch_cameras = [
                make_camera_from_extr_intr(extr_samples[j], intr_samples[j], resolution=resolution, device=mesh.faces.device)
                for j in range(start_idx, end_idx)
            ]
            batch_cameras = kaolin.render.camera.Camera.cat(batch_cameras)
            gt_batch = custom_mesh_batched_render(
                batch_cameras,
                mesh,
                gen_albedo,
                None,
                requires_positions=False,
                process_as_albedo=False,
                backend="cuda",
            )
            pred_batch = custom_mesh_batched_render(
                batch_cameras,
                mesh,
                albedo,
                None,
                requires_positions=False,
                process_as_albedo=False,
                backend="cuda",
            )
            mask = gt_batch["textured"][..., 3:].permute(0, 3, 1, 2) / 2 + 0.5
            gt = gt_batch["textured"][..., :3].permute(0, 3, 1, 2) / 2 + 0.5
            pred = pred_batch["textured"][..., :3].permute(0, 3, 1, 2) / 2 + 0.5
            gt = gt * mask
            pred = pred * mask

            if batch_idx == 0:
                log = torch.cat([gt[0], pred[0]], dim=-1)
                torchvision.utils.save_image(log, str(output_dir / "log-lpips.png"))

            lpip_scores.append(lpip(pred, gt).mean().item())

        lpips_score = torch.mean(torch.tensor(lpip_scores)).item()
        print("LPIPS", lpips_score)
        all_lpips.append(lpips_score)

    all_fid = []
    fid = FrechetInceptionDistance().to(device)
    for view_id in tqdm(view_ids):
        completed_texture_path = completed_dir / f"view{view_id}.png"
        gen_texture_path = gt_texture_dir / f"view{view_id}.png"

        albedo = kaolin.io.utils.read_image(str(completed_texture_path)).to(mesh.faces.device).contiguous() * 2 - 1
        gen_albedo = kaolin.io.utils.read_image(str(gen_texture_path)).to(mesh.faces.device).contiguous() * 2 - 1

        resolution = 256
        fov_min, fov_max = 0.4, 0.8
        fid_cache = camera_cache_dir / f"fid_view{view_id}_{cache_suffix(num_samples=args.num_samples, camera_dist=args.camera_dist)}.pt"
        cached_fid = load_cached_payload(fid_cache)
        if cached_fid is None:
            torch.manual_seed(int(view_id))

            gt_intrinsics_sampler = FovSampler(fov_min, fov_max)
            gt_extrinsics_camera_sampler = LocalCameraExtrinsicsSampler(mesh, args.camera_dist)
            sampling_weights = torch.zeros(mesh.faces.shape[0])
            valid_face_ids = get_valid_faces_from_texture(mesh, gen_albedo.permute(2, 0, 1).unsqueeze(0), all_filled=True)
            sampling_weights[valid_face_ids] = 1.0
            if sampling_weights.sum() <= 0:
                raise ValueError(f"No valid face ids found for {object_name} view {view_id}")
            gt_extrinsics_camera_sampler.set_sampling_weights(sampling_weights)

            intr_samples, extr_samples = [], []
            invalid_intr_samples, invalid_extr_samples = [], []
            resize = torchvision.transforms.Resize((gen_albedo.shape[0], gen_albedo.shape[1]))
            normal_map = resize(mesh.materials[0].chw().normals_texture).permute(1, 2, 0)
            count = 0

            while len(intr_samples) < args.num_samples and count < 500:
                count += 1
                intr_batch = gt_intrinsics_sampler.generate(32)
                extr_batch = gt_extrinsics_camera_sampler.generate(32)
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
                fully_valid = albedo_alpha.sum((1, 2, 3)) == (resolution * resolution - background_alpha.sum((1, 2, 3)))

                valid_camera_idx = torch.argwhere(fully_valid)
                print("fid patches", len(intr_samples), "new patches", len(valid_camera_idx))

                for idx in valid_camera_idx:
                    intr_samples.append(intr_batch[idx.item()])
                    extr_samples.append(extr_batch[idx.item()])
                    if len(intr_samples) == args.num_samples:
                        break

                if args.debug:
                    not_valid = albedo_alpha.sum((1, 2, 3)) != (resolution * resolution - background_alpha.sum((1, 2, 3)))
                    invalid_camera_idx = torch.argwhere(not_valid)
                    for idx in invalid_camera_idx:
                        invalid_intr_samples.append(intr_batch[idx.item()])
                        invalid_extr_samples.append(extr_batch[idx.item()])
                    break

            if len(intr_samples) < args.num_samples:
                print("skipping view", view_id)
                continue

            gt_intr_samples = torch.stack(intr_samples, 0)
            gt_extr_samples = torch.stack(extr_samples, 0)

            if args.debug:
                invalid_intr_samples = torch.stack(invalid_intr_samples, 0)
                invalid_extr_samples = torch.stack(invalid_extr_samples, 0)

            intrinsics_sampler = FovSampler(fov_min, fov_max)
            intr_samples = intrinsics_sampler.generate(args.num_samples)
            extrinsics_camera_sampler = LocalCameraExtrinsicsSampler(mesh, args.camera_dist)
            sampling_weights = torch.ones(mesh.faces.shape[0])
            valid_face_ids = get_valid_faces_from_texture(mesh, gen_albedo.permute(2, 0, 1).unsqueeze(0), all_filled=True)
            sampling_weights[valid_face_ids] = 0.0
            if sampling_weights.sum() <= 0:
                raise ValueError(f"No hidden-face samples available for {object_name} view {view_id}")
            extrinsics_camera_sampler.set_sampling_weights(sampling_weights)
            extr_samples = extrinsics_camera_sampler.generate(args.num_samples)
            save_cached_payload(
                fid_cache,
                {
                    "gt_intr_samples": gt_intr_samples,
                    "gt_extr_samples": gt_extr_samples,
                    "pred_intr_samples": intr_samples,
                    "pred_extr_samples": extr_samples,
                    "invalid_intr_samples": invalid_intr_samples if args.debug else None,
                    "invalid_extr_samples": invalid_extr_samples if args.debug else None,
                },
            )
        else:
            gt_intr_samples = cached_fid["gt_intr_samples"]
            gt_extr_samples = cached_fid["gt_extr_samples"]
            intr_samples = cached_fid["pred_intr_samples"]
            extr_samples = cached_fid["pred_extr_samples"]
            invalid_intr_samples = cached_fid.get("invalid_intr_samples")
            invalid_extr_samples = cached_fid.get("invalid_extr_samples")

        num_batches = int(math.ceil(args.num_samples / batch_size))

        gt_dir = output_dir / "gt"
        pred_dir = output_dir / "pred"
        gt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, args.num_samples)
            batch_cameras = [
                make_camera_from_extr_intr(
                    gt_extr_samples[j],
                    gt_intr_samples[j],
                    resolution=resolution,
                    device=mesh.faces.device,
                )
                for j in range(len(gt_extr_samples))
            ]
            batch_cameras = kaolin.render.camera.Camera.cat(batch_cameras)
            gt_batch = custom_mesh_batched_render(
                batch_cameras,
                mesh,
                gen_albedo,
                None,
                requires_positions=False,
                process_as_albedo=False,
                backend="cuda",
            )

            batch_cameras = [
                make_camera_from_extr_intr(extr_samples[j], intr_samples[j], resolution=resolution, device=mesh.faces.device)
                for j in range(start_idx, end_idx)
            ]
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

            gt_imgs = gt_batch["textured"]
            gt_mask = (torch.abs(gt_batch["camera_normals"]) < 0.01).all(dim=-1, keepdim=True)
            gt_imgs[..., :3][gt_mask.expand(-1, -1, -1, 3)] = -1.0
            gt_imgs = gt_imgs[..., :3]

            pred_imgs = pred_batch["textured"]
            pred_mask = (torch.abs(pred_batch["camera_normals"]) < 0.01).all(dim=-1, keepdim=True)
            pred_imgs[..., :3][pred_mask.expand(-1, -1, -1, 3)] = -1.0
            pred_imgs = pred_imgs[..., :3]

            gt = gt_imgs.permute(0, 3, 1, 2) / 2 + 0.5
            pred = pred_imgs.permute(0, 3, 1, 2) / 2 + 0.5

            fid.update(gt.clip(0, 1), is_real=True)
            fid.update(pred.clip(0, 1), is_real=False)

            if batch_idx == 0:
                log = torch.cat([gt[0], pred[0]], dim=-1)
                torchvision.utils.save_image(log, str(output_dir / "log-fid.png"))

            if args.debug:
                if invalid_intr_samples is not None and invalid_extr_samples is not None:
                    batch_cameras = [
                        make_camera_from_extr_intr(
                            invalid_extr_samples[j],
                            invalid_intr_samples[j],
                            resolution=resolution,
                            device=mesh.faces.device,
                        )
                        for j in range(len(invalid_extr_samples))
                    ]
                    batch_cameras = kaolin.render.camera.Camera.cat(batch_cameras)

                    invalid_gt_batch = custom_mesh_batched_render(
                        batch_cameras,
                        mesh,
                        gen_albedo,
                        None,
                        requires_positions=False,
                        process_as_albedo=False,
                        backend="cuda",
                    )
                    gt_debug = gt_batch["textured"][..., :3].permute(0, 3, 1, 2) / 2 + 0.5
                    invalid_gt = invalid_gt_batch["textured"][..., :3].permute(0, 3, 1, 2) / 2 + 0.5
                    for image_idx in range(len(gt_debug)):
                        torchvision.utils.save_image(gt_debug[image_idx], gt_dir / f"gt_v{view_id}_{batch_idx}-{image_idx}.png")
                    for image_idx in range(len(invalid_gt)):
                        torchvision.utils.save_image(invalid_gt[image_idx], pred_dir / f"pred_v{view_id}_{batch_idx}-{image_idx}.png")
                    break

        fid_score = fid.compute().item()
        print("FID", fid_score)
        all_fid.append(fid_score)

    save_dict = {
        "eval-name": f"{expr_name}_num_patch={args.num_samples}",
        "avg-lpips": statistics.mean(all_lpips),
        "final-fid": fid.compute().item(),
        "eval-lpips": all_lpips,
        "eval-fid": all_fid,
    }
    save_dict_to_csv(save_dict, str(output_path), False)


if __name__ == "__main__":
    main()
