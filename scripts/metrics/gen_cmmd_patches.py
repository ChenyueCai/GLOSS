"""Generate CMMD reference/eval patch images for a single (mesh, completion) run.

Writes
  <out_dir>/gt/gt_v<view>_<batch>-<j>.png    — patches rendered from GT texture
  <out_dir>/pred/pred_v<view>_<batch>-<j>.png — patches rendered from completion texture

The reference (gt) cameras are sampled from faces VISIBLE in the conditioning view
(`view_id`); the eval (pred) cameras are sampled from faces NOT visible in the
conditioning view, matching the original `test_completion_cmmd.py` recipe.
"""
import argparse
import math
import os
from pathlib import Path

import kaolin
import torch
import torchvision
from tqdm import tqdm

from gloss.data.render_dataloader import FovSampler, LocalCameraExtrinsicsSampler
from gloss.utils.eval_utils import extract_view_numbers
from gloss.utils.kaolin_utils import load_mesh
from gloss.utils.render import make_camera_from_extr_intr
from gloss.utils.render_fast import custom_mesh_batched_render
from gloss.utils.single_view import get_valid_faces_from_texture


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh-path", required=True)
    p.add_argument("--completed-dir", required=True,
                   help="Dir with view{NNNN}.png predicted textures.")
    p.add_argument("--gt-texture-dir", required=True,
                   help="Dir with view{NNNN}.png ground-truth textures (test_textures_sr).")
    p.add_argument("--out-dir", required=True,
                   help="Output dir for gt/ and pred/ subfolders.")
    p.add_argument("--view-ids", type=str, default=None,
                   help="CSV of view ids to use; defaults to all available in --gt-texture-dir.")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--camera-dist", type=float, default=0.25)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    mesh = load_mesh(args.mesh_path)
    completed_dir = Path(args.completed_dir)
    gt_dir_in = Path(args.gt_texture_dir)

    view_ids_all = [f"{n:04d}" for n in extract_view_numbers(str(gt_dir_in))]
    if args.view_ids:
        wanted = {f"{int(v.strip()):04d}" for v in args.view_ids.split(",") if v.strip()}
        view_ids = [v for v in view_ids_all if v in wanted]
    else:
        view_ids = view_ids_all
    if not view_ids:
        raise ValueError(f"No views to process under {gt_dir_in}")

    out_dir = Path(args.out_dir)
    gt_out = out_dir / "gt"
    pred_out = out_dir / "pred"
    gt_out.mkdir(parents=True, exist_ok=True)
    pred_out.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    resolution = args.resolution
    fov_min, fov_max = 0.4, 0.8

    for view_id in tqdm(view_ids):
        completed_path = completed_dir / f"view{view_id}.png"
        gt_path = gt_dir_in / f"view{view_id}.png"
        if not completed_path.exists() or not gt_path.exists():
            print(f"SKIP view{view_id} (missing files)")
            continue

        albedo = (kaolin.io.utils.read_image(str(completed_path))
                  .to(mesh.faces.device).contiguous()) * 2 - 1
        gen_albedo = (kaolin.io.utils.read_image(str(gt_path))
                      .to(mesh.faces.device).contiguous()) * 2 - 1

        torch.manual_seed(int(view_id))

        # Reference (gt) cameras: sample faces VISIBLE in the cond view
        gt_intrinsics_sampler = FovSampler(fov_min, fov_max)
        gt_extr_sampler = LocalCameraExtrinsicsSampler(mesh, args.camera_dist)
        sampling_weights = torch.zeros(mesh.faces.shape[0])
        valid_face_ids = get_valid_faces_from_texture(
            mesh, gen_albedo.permute(2, 0, 1).unsqueeze(0), all_filled=True)
        sampling_weights[valid_face_ids] = 1.0
        if sampling_weights.sum() == 0:
            print(f"SKIP view{view_id} (no visible faces)")
            continue
        gt_extr_sampler.set_sampling_weights(sampling_weights)

        intr_samples, extr_samples = [], []
        resize = torchvision.transforms.Resize((gen_albedo.shape[0], gen_albedo.shape[1]))
        normal_map = resize(mesh.materials[0].chw().normals_texture).permute(1, 2, 0)
        count = 0
        while len(intr_samples) < args.num_samples and count < 500:
            count += 1
            intr_batch = gt_intrinsics_sampler.generate(32)
            extr_batch = gt_extr_sampler.generate(32)
            cameras = [
                make_camera_from_extr_intr(extr_batch[i], intr_batch[i],
                                           resolution=resolution, device=device)
                for i in range(32)
            ]
            batch_cameras = kaolin.render.camera.Camera.cat(cameras)
            r = custom_mesh_batched_render(batch_cameras, mesh, gen_albedo, normal_map,
                                           requires_positions=False, process_as_albedo=False)
            albedo_alpha = r["textured"][..., 3:] > -1
            background_alpha = (torch.abs(r["camera_normals"]) < 0.01).all(dim=-1, keepdim=True).float()
            fully_valid = albedo_alpha.sum((1, 2, 3)) == (resolution * resolution - background_alpha.sum((1, 2, 3)))
            valid_camera_idx = torch.argwhere(fully_valid)
            for idx in valid_camera_idx:
                intr_samples.append(intr_batch[idx.item()])
                extr_samples.append(extr_batch[idx.item()])
                if len(intr_samples) == args.num_samples:
                    break
        if len(intr_samples) < args.num_samples:
            print(f"SKIP view{view_id} (only {len(intr_samples)} valid gt cameras)")
            continue
        gt_intr_samples = torch.stack(intr_samples, 0)
        gt_extr_samples = torch.stack(extr_samples, 0)

        # Eval (pred) cameras: sample faces NOT visible in the cond view
        intr_pred_sampler = FovSampler(fov_min, fov_max)
        intr_samples = intr_pred_sampler.generate(args.num_samples)
        extr_pred_sampler = LocalCameraExtrinsicsSampler(mesh, args.camera_dist)
        sw = torch.ones(mesh.faces.shape[0])
        sw[valid_face_ids] = 0.0
        if sw.sum() == 0:
            print(f"SKIP view{view_id} (no occluded faces)")
            continue
        extr_pred_sampler.set_sampling_weights(sw)
        extr_samples = extr_pred_sampler.generate(args.num_samples)

        num_batches = int(math.ceil(args.num_samples / args.batch_size))
        for bi in range(num_batches):
            s = bi * args.batch_size
            e = min(s + args.batch_size, args.num_samples)
            cams_gt = [
                make_camera_from_extr_intr(gt_extr_samples[j], gt_intr_samples[j],
                                           resolution=resolution, device=mesh.faces.device)
                for j in range(s, e)
            ]
            cams_gt = kaolin.render.camera.Camera.cat(cams_gt)
            gt_batch = custom_mesh_batched_render(cams_gt, mesh, gen_albedo, None,
                                                  requires_positions=False, process_as_albedo=False, backend="cuda")

            cams_pred = [
                make_camera_from_extr_intr(extr_samples[j], intr_samples[j],
                                           resolution=resolution, device=mesh.faces.device)
                for j in range(s, e)
            ]
            cams_pred = kaolin.render.camera.Camera.cat(cams_pred)
            pred_batch = custom_mesh_batched_render(cams_pred, mesh, albedo, None,
                                                    requires_positions=False, process_as_albedo=False, backend="cuda")

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

            for j in range(gt.shape[0]):
                torchvision.utils.save_image(gt[j], str(gt_out / f"gt_v{view_id}_{bi}-{j}.png"))
                torchvision.utils.save_image(pred[j], str(pred_out / f"pred_v{view_id}_{bi}-{j}.png"))

    print(f"DONE wrote patches to {out_dir}")


if __name__ == "__main__":
    main()
