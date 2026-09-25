# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that example-data reference views line up with their cameras and textures.

For every view of every mesh it renders the mesh from the view's camera meta and measures:
  leak           share of a thin band just outside the rendered silhouette that is not background;
                 high when the camera does not line up with the image
  texture cover  share of the rendered silhouette that the view's partial texture covers;
                 low when the texture is not the backprojection of this view
  color error    rendered texture vs. image colour where both exist (0-255 scale);
                 high when the texture belongs to a different view
  silhouette IoU against a background-colour mask, reported for information only: it
                 undercounts objects whose colour is close to the studio background

    python tests/check_example_data.py                         # every mesh under $GLOSS_DATA_DIR
    python tests/check_example_data.py --meshes croissant --out /tmp/check
    python tests/check_example_data.py --meshes croissant --views-dir A --metas-dir B --textures-dir C

Exits non-zero when any view falls below --min-iou or above --max-color-err, and writes a JSON
report plus overlay images for the
worst views (red: mesh without object, blue: object without mesh). Needs a CUDA GPU.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation
import yaml
import kaolin
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gloss.utils.kaolin_utils import load_mesh, camera_from_meta  # noqa: E402
from gloss.utils.paths import get_data_dir  # noqa: E402
from gloss.utils.render_fast import fast_batched_render  # noqa: E402


def fg_mask(img):
    """Object mask for a reference view on a flat studio background."""
    border = np.concatenate([img[:8].reshape(-1, 3), img[-8:].reshape(-1, 3),
                             img[:, :8].reshape(-1, 3), img[:, -8:].reshape(-1, 3)])
    return np.abs(img - np.median(border, 0)).max(-1) > 30


def iou(a, b):
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def check_view(mesh, view_fp, meta_fp, tex_fp, device):
    meta = yaml.safe_load(open(meta_fp))["camera"]
    cam = camera_from_meta(meta).to(device)
    w, h = int(meta["intrinsics"]["width"]), int(meta["intrinsics"]["height"])
    img = np.asarray(Image.open(view_fp).convert("RGB").resize((w, h), Image.BILINEAR)).astype(np.float32)
    fg = fg_mask(img)
    tex = kaolin.io.utils.read_image(str(tex_fp)).to(device)
    if tex.shape[-1] == 3:
        tex = torch.cat([tex, torch.ones_like(tex[..., :1])], -1)
    r = fast_batched_render(cam, mesh, tex * 2 - 1)
    sil = (r["face_idx"][0] >= 0).cpu().numpy()
    rgba = (r["textured"][0] / 2 + 0.5).clamp(0, 1).cpu().numpy()
    cov = sil & (rgba[..., 3] > 0.5)
    err = float(np.abs(rgba[..., :3] * 255 - img)[cov].mean()) if cov.any() else float("nan")
    band = binary_dilation(sil, iterations=6) & ~binary_dilation(sil, iterations=2)
    leak = float(fg[band].mean()) if band.any() else 0.0
    cover = float(cov.sum() / max(sil.sum(), 1))
    row = {"leak": round(leak, 4), "texture_cover": round(cover, 4), "color_err": round(err, 2),
           "silhouette_iou": round(iou(sil, fg), 4), "texture_px": int(tex.shape[0])}
    return row, (img, fg, sil, cov, rgba)


def overlay(img, fg, sil, cov, rgba, uv=None):
    """Panels: reference view | backprojected UV texture | texture re-rendered from the view's
    camera | silhouette outline and leak band (green: rendered outline, red: object leaking out)."""
    h = img.shape[0]
    outline = binary_dilation(sil, iterations=1) & ~sil
    band = binary_dilation(sil, iterations=6) & ~binary_dilation(sil, iterations=2)
    over = img.copy() / 255
    over[outline] = [0, 1, 0]
    over[band & fg] = [1, 0, 0]
    tex = np.where(cov[..., None], rgba[..., :3], 0.2)
    panels = [img / 255]
    if uv is not None:
        u = Image.open(uv).convert("RGBA").resize((h, h), Image.BILINEAR)
        bg = Image.new("RGBA", u.size, (51, 51, 51, 255)); bg.alpha_composite(u)
        panels.append(np.asarray(bg.convert("RGB")).astype(np.float32) / 255)
    panels += [tex, over]
    return Image.fromarray((np.concatenate(panels, 1) * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None, help="Example-data root (default: $GLOSS_DATA_DIR)")
    ap.add_argument("--meshes", nargs="*", default=None, help="Meshes to check (default: all under mesh/)")
    ap.add_argument("--views-dir", default=None, help="Override single_view/<mesh> (single mesh only)")
    ap.add_argument("--metas-dir", default=None, help="Override metas/<mesh> (single mesh only)")
    ap.add_argument("--textures-dir", default=None, help="Override texture/<mesh> (single mesh only)")
    ap.add_argument("--max-leak", type=float, default=0.25, help="Fail above this leak (camera misaligned; soft contact shadows reach about 0.16)")
    ap.add_argument("--min-cover", type=float, default=0.3, help="Fail below this texture cover (wrong texture)")
    ap.add_argument("--max-color-err", type=float, default=20.0, help="Fail above this color error (wrong texture)")
    ap.add_argument("--out", default="expr/check_example_data", help="Report and overlay folder")
    ap.add_argument("--overlays", type=int, default=3, help="Overlays saved per mesh (worst views)")
    ap.add_argument("--samples", type=int, default=0, help="Also save this many overlays of passing views per mesh")
    a = ap.parse_args()
    data = Path(a.data_dir or get_data_dir())
    meshes = a.meshes or sorted(p.name for p in (data / "meshes").iterdir() if (p / "scene.gltf").is_file())
    if (a.views_dir or a.metas_dir or a.textures_dir) and len(meshes) != 1:
        ap.error("--views-dir/--metas-dir/--textures-dir need exactly one --meshes entry")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    report, failed = {}, False
    for m in meshes:
        views = Path(a.views_dir or data / "single_views" / m)
        metas = Path(a.metas_dir or data / "metas" / m)
        texs = Path(a.textures_dir or data / "textures" / m)
        mesh = load_mesh(str(data / "meshes" / m / "scene.gltf")).to(device)
        ids = sorted(f[4:8] for f in os.listdir(views) if f.endswith(".basecolor.png"))
        rows, missing = [], []
        for v in ids:
            fps = (views / f"view{v}.basecolor.png", metas / f"view{v}.yml", texs / f"view{v}.png")
            if not all(p.is_file() for p in fps):
                missing.append(v); continue
            row, _ = check_view(mesh, *fps, device)
            row["view"] = v; rows.append(row)
        def failing(r):
            return r["leak"] > a.max_leak or r["texture_cover"] < a.min_cover or not (r["color_err"] <= a.max_color_err)
        bad = [r for r in rows if failing(r)]
        worst = sorted(bad, key=lambda r: -(r["color_err"] if r["color_err"] == r["color_err"] else 999))[: a.overlays]
        good = [r for r in rows if not failing(r)]
        picks = [(r, "fail") for r in worst] + [(r, "pass") for r in good[:: max(1, len(good) // max(a.samples, 1))][: a.samples]]
        for r, tag in picks:
            v = r["view"]
            _, imgs = check_view(mesh, views / f"view{v}.basecolor.png", metas / f"view{v}.yml", texs / f"view{v}.png", device)
            overlay(*imgs, uv=texs / f"view{v}.png").save(out / f"{m}_{tag}_view{v}_leak{r['leak']:.3f}_cover{r['texture_cover']:.2f}_err{r['color_err']:.0f}.png")
        err = [r["color_err"] for r in rows]; leak = [r["leak"] for r in rows]; cover = [r["texture_cover"] for r in rows]
        report[m] = {"views": len(rows), "missing_files": missing,
                     "leak_max": max(leak) if leak else None, "texture_cover_min": min(cover) if cover else None,
                     "color_err_median": float(np.nanmedian(err)) if err else None,
                     "texture_px": sorted({r["texture_px"] for r in rows}),
                     "failing_views": [{k: r[k] for k in ("view", "leak", "texture_cover", "color_err")} for r in bad]}
        failed |= bool(bad or missing)
        json.dump(rows, open(out / f"{m}_views.json", "w"), indent=1)
        print(f"{m}: {len(rows)} views, max leak {report[m]['leak_max']:.3f}, min texture cover {report[m]['texture_cover_min']:.3f}, "
              f"color err median {report[m]['color_err_median']:.1f}, failing {len(bad)}, missing {len(missing)}", flush=True)
    json.dump(report, open(out / "report.json", "w"), indent=1)
    print("FAIL" if failed else "OK", "- report:", out / "report.json")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
