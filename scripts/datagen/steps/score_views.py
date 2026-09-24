#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Score generated single views for automatic filtering.

Reads images from ``<single_view_dir>/gen_view_masked/viewNNNN.png``, their
rendered camera-space normals from ``condition_output/normal/normalNNNN.png``,
their masks from ``condition_output/mask/maskNNNN.png``, and their per-view
prompt + camera pose from ``meta/viewNNNN.yml``. Produces four scalar scores
per view and writes them into ``meta.json`` under a ``view_scores`` key.

Scores
------
clip_prompt        CLIP(ViT-L/14) cosine similarity between the generated
                   image and its per-view prompt. Catches the
                   "belly-of-turtle prompt, downtown image" case.
clip_viewpoint     CLIP cosine similarity against a viewpoint-conditioned
                   caption synthesized from the camera extrinsics
                   ("<subject> photographed from above/below/the side").
                   Flags "right subject, wrong angle" failures.
aesthetic          LAION aesthetic-predictor-v2 MLP applied on top of the
                   CLIP image embedding. Flags visually broken outputs.
normal_agreement   Masked mean cosine similarity between the rendered
                   camera-space normal map and the normal map predicted
                   from the generated RGB by Marigold-normals. Flags
                   geometry drift (SDXL ignored the ControlNet normal
                   hint). Axis-convention is auto-resolved by picking the
                   best of the four sign-flip variants.

Output
------
Writes ``view_scores``, ``view_scores_meta``, and ``auto_excluded_indices``
into the single-view ``meta.json``. With ``--apply`` the auto-excludes are
unioned into ``exclude_from_training_indices`` alongside any manually
rejected ids from ``review_training_views.py``; without ``--apply`` the
exclude list is left untouched (score-only).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("score_views")

DEFAULT_VIEW_RE = re.compile(r"^view(\d+)\.png$")
NORMAL_RE = re.compile(r"^normal(\d+)\.png$")
MASK_RE = re.compile(r"^mask(\d+)\.png$")
YAML_RE = re.compile(r"^view(\d+)\.ya?ml$")


def build_view_regex(channel: str | None) -> re.Pattern[str]:
    """Build a regex that matches ``viewNNNN.png`` (channel is None) or
    ``viewNNNN.<channel>.png`` (e.g. basecolor, roughness)."""
    if not channel:
        return DEFAULT_VIEW_RE
    return re.compile(rf"^view(\d+)\.{re.escape(channel)}\.png$")

# ---------------------------------------------------------------------------
# LAION aesthetic-predictor-v2 MLP
# ---------------------------------------------------------------------------
# Official checkpoint is a tiny (~8 MB) MLP on top of CLIP ViT-L/14 image
# embeddings. We construct the architecture ourselves and pull the weights
# directly via huggingface_hub so there is no extra pip dependency.
_AESTHETIC_REPO = "camenduru/improved-aesthetic-predictor"
_AESTHETIC_FILE = "sac+logos+ava1-l14-linearMSE.pth"


class AestheticMLP(nn.Module):
    """Architecture of LAION aesthetic-predictor-v2."""

    def __init__(self, in_dim: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


@dataclass
class ViewEntry:
    view_id: int
    image_path: Path
    normal_path: Path | None
    mask_path: Path | None
    yaml_path: Path | None
    prompt: str | None = None
    view_matrix: np.ndarray | None = None  # (4, 4)


@dataclass
class RunPaths:
    single_view_dir: Path
    view_dir: Path
    normal_dir: Path
    mask_dir: Path
    meta_dir: Path
    meta_json: Path


def resolve_paths(args) -> RunPaths:
    base = Path(args.path).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Path does not exist: {base}")

    if base.is_dir() and base.name == args.view_subdir:
        single_view_dir = base.parent
        view_dir = base
    elif base.is_dir():
        single_view_dir = base
        view_dir = base / args.view_subdir
    else:
        raise ValueError(f"Expected a directory, got file: {base}")

    if not view_dir.is_dir():
        raise FileNotFoundError(f"View directory not found: {view_dir}")

    return RunPaths(
        single_view_dir=single_view_dir,
        view_dir=view_dir,
        normal_dir=single_view_dir / args.normal_subdir,
        mask_dir=single_view_dir / args.mask_subdir,
        meta_dir=single_view_dir / args.meta_subdir,
        meta_json=single_view_dir / args.meta_filename,
    )


def discover_views(
    paths: RunPaths,
    limit: int | None,
    views: list[int] | None,
    view_re: re.Pattern[str] = DEFAULT_VIEW_RE,
) -> list[ViewEntry]:
    entries: list[ViewEntry] = []
    for p in sorted(paths.view_dir.iterdir()):
        m = view_re.match(p.name)
        if not m:
            continue
        vid = int(m.group(1))
        if views is not None and vid not in views:
            continue
        normal = paths.normal_dir / f"normal{vid:04d}.png"
        mask = paths.mask_dir / f"mask{vid:04d}.png"
        yaml_path = paths.meta_dir / f"view{vid:04d}.yml"
        if not yaml_path.exists():
            alt = paths.meta_dir / f"view{vid:04d}.yaml"
            yaml_path = alt if alt.exists() else yaml_path
        entries.append(
            ViewEntry(
                view_id=vid,
                image_path=p,
                normal_path=normal if normal.exists() else None,
                mask_path=mask if mask.exists() else None,
                yaml_path=yaml_path if yaml_path.exists() else None,
            )
        )
    if limit is not None:
        entries = entries[:limit]
    return entries


def load_view_metadata(entry: ViewEntry) -> None:
    if entry.yaml_path is None:
        return
    cfg = OmegaConf.load(entry.yaml_path)
    entry.prompt = str(cfg.get("prompt", "") or "")
    try:
        vm = np.asarray(cfg.camera.extrinsics.view_matrix, dtype=np.float32)
        if vm.ndim == 3:  # (1, 4, 4)
            vm = vm.squeeze(0)
        entry.view_matrix = vm
    except Exception:
        entry.view_matrix = None


# ---------------------------------------------------------------------------
# Camera direction → viewpoint caption
# ---------------------------------------------------------------------------


def camera_elevation_azimuth(view_matrix: np.ndarray) -> tuple[float, float]:
    """Return (elevation_deg, azimuth_deg) of the camera origin under Y-up.

    Convention: ``view_matrix`` is world→camera, so the camera origin in
    world space is ``-R^T @ t`` where ``R = view_matrix[:3, :3]`` and
    ``t = view_matrix[:3, 3]``. Elevation is the angle above the XZ plane,
    azimuth is measured around +Y from +Z.
    """
    R = view_matrix[:3, :3]
    t = view_matrix[:3, 3]
    origin = -R.T @ t
    r = float(np.linalg.norm(origin) + 1e-8)
    elev = math.degrees(math.asin(float(origin[1]) / r))
    azi = math.degrees(math.atan2(float(origin[0]), float(origin[2])))
    return elev, azi


def viewpoint_phrase(elev: float, azi: float) -> str:
    if elev > 60:
        vert = "from directly above (top-down view)"
    elif elev > 25:
        vert = "from a high angle above"
    elif elev > 10:
        vert = "from slightly above"
    elif elev > -10:
        vert = "from the side at eye level"
    elif elev > -25:
        vert = "from slightly below"
    elif elev > -60:
        vert = "from a low angle below"
    else:
        vert = "from directly below (worm's-eye view)"

    # Azimuth: relative to the object's +Z-facing (front). Object orientation
    # is unknown in general, so we only include a coarse hint.
    a = azi % 360
    if a < 0:
        a += 360
    if 315 <= a or a < 45:
        side = "front"
    elif 45 <= a < 135:
        side = "right side"
    elif 135 <= a < 225:
        side = "back"
    else:
        side = "left side"
    return f"{vert}, {side}"


def viewpoint_caption(subject: str, view_matrix: np.ndarray) -> str:
    elev, azi = camera_elevation_azimuth(view_matrix)
    return f"a photograph of {subject}, {viewpoint_phrase(elev, azi)}"


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


class CLIPScorer:
    def __init__(self, model_name: str, pretrained: str, device: str, dtype: torch.dtype):
        import open_clip

        log.info("Loading CLIP: %s / %s", model_name, pretrained)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval().to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        self.embed_dim = self.model.visual.output_dim

    @torch.no_grad()
    def embed_images(self, pil_images: list[Image.Image]) -> torch.Tensor:
        pixels = torch.stack([self.preprocess(img) for img in pil_images]).to(
            device=self.device, dtype=self.dtype
        )
        feats = self.model.encode_image(pixels)
        return F.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        return F.normalize(feats.float(), dim=-1)


class AestheticScorer:
    def __init__(self, clip: CLIPScorer, hf_home: str | None):
        from huggingface_hub import hf_hub_download

        kwargs = {"repo_id": _AESTHETIC_REPO, "filename": _AESTHETIC_FILE}
        if hf_home:
            kwargs["cache_dir"] = str(Path(hf_home) / "hub")
        log.info("Loading aesthetic predictor: %s/%s", _AESTHETIC_REPO, _AESTHETIC_FILE)
        ckpt_path = hf_hub_download(**kwargs)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model = AestheticMLP(in_dim=clip.embed_dim)
        self.model.load_state_dict(state)
        self.model.eval().to(device=clip.device, dtype=torch.float32)

    @torch.no_grad()
    def score(self, clip_image_feats: torch.Tensor) -> torch.Tensor:
        return self.model(clip_image_feats.float()).squeeze(-1)


class NormalAgreementScorer:
    """Marigold-based normal agreement. Handles axis-convention mismatch by
    trying all 4 sign-flip combinations of (X, Y, Z) and reporting the best
    masked-mean cosine similarity."""

    def __init__(self, device: str, dtype: torch.dtype, hf_home: str | None,
                 num_inference_steps: int = 1, resolution: int = 512):
        from diffusers import MarigoldNormalsPipeline

        log.info("Loading Marigold-normals (num_inference_steps=%d)", num_inference_steps)
        kwargs: dict[str, Any] = {"variant": "fp16"} if dtype == torch.float16 else {}
        if hf_home:
            kwargs["cache_dir"] = str(Path(hf_home) / "hub")
        self.pipe = MarigoldNormalsPipeline.from_pretrained(
            "prs-eth/marigold-normals-v1-1",
            torch_dtype=dtype,
            **kwargs,
        )
        self.pipe = self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.device = device
        self.dtype = dtype
        self.num_inference_steps = num_inference_steps
        self.resolution = resolution

    @torch.no_grad()
    def predict(self, pil_image: Image.Image) -> torch.Tensor:
        """Return predicted camera-space normals as (3, H, W) in [-1, 1]."""
        out = self.pipe(
            pil_image,
            num_inference_steps=self.num_inference_steps,
            processing_resolution=self.resolution,
            ensemble_size=1,
            output_type="pt",
        )
        n = out.prediction[0]  # (3, H, W) in [-1, 1]
        return F.normalize(n.float(), dim=0)

    @staticmethod
    def _agreement(pred: torch.Tensor, rendered: torch.Tensor, mask: torch.Tensor) -> float:
        # pred, rendered: (3, H, W) unit vectors; mask: (H, W) in {0, 1}
        cos = (pred * rendered).sum(dim=0)
        denom = mask.sum().clamp_min(1.0)
        return float((cos * mask).sum() / denom)

    @torch.no_grad()
    def score(self, pil_image: Image.Image, rendered_png: Path, mask_png: Path | None) -> float:
        pred = self.predict(pil_image)  # (3, H, W)

        rendered_arr = np.asarray(Image.open(rendered_png).convert("RGB")).astype(np.float32)
        rendered = torch.from_numpy(rendered_arr).permute(2, 0, 1) / 255.0
        rendered = rendered * 2.0 - 1.0  # [-1, 1]
        rendered = F.normalize(rendered.to(pred.device), dim=0)

        # Downsample the larger of the two to the smaller's resolution.
        H = min(pred.shape[1], rendered.shape[1])
        W = min(pred.shape[2], rendered.shape[2])
        if pred.shape[1:] != (H, W):
            pred = F.interpolate(pred.unsqueeze(0), size=(H, W), mode="bilinear",
                                 align_corners=False).squeeze(0)
            pred = F.normalize(pred, dim=0)
        if rendered.shape[1:] != (H, W):
            rendered = F.interpolate(rendered.unsqueeze(0), size=(H, W), mode="bilinear",
                                     align_corners=False).squeeze(0)
            rendered = F.normalize(rendered, dim=0)

        if mask_png is not None and mask_png.exists():
            m = np.asarray(Image.open(mask_png).convert("L")).astype(np.float32) / 255.0
            mask = torch.from_numpy(m).to(pred.device)
            if mask.shape != (H, W):
                mask = F.interpolate(mask[None, None], size=(H, W), mode="nearest")[0, 0]
            mask = (mask > 0.5).float()
        else:
            # Fall back: treat unit-length rendered normals as foreground.
            mask = (rendered.norm(dim=0) > 0.5).float()

        if mask.sum() < 10:
            return float("nan")

        best = -2.0
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                for sz in (1.0, -1.0):
                    sign = torch.tensor([sx, sy, sz], device=pred.device)[:, None, None]
                    score = self._agreement(pred * sign, rendered, mask)
                    if score > best:
                        best = score
        return best


# ---------------------------------------------------------------------------
# Scoring orchestration
# ---------------------------------------------------------------------------


@dataclass
class Scores:
    clip_prompt: float | None = None
    clip_viewpoint: float | None = None
    aesthetic: float | None = None
    normal_agreement: float | None = None

    def as_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in self.__dict__.items():
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                out[k] = float(v)
        return out


@dataclass
class ScoreConfig:
    subject: str
    skip_clip_prompt: bool
    skip_clip_viewpoint: bool
    skip_aesthetic: bool
    skip_normal: bool
    batch_size: int
    device: str
    dtype: torch.dtype
    clip_model: str
    clip_pretrained: str
    hf_home: str | None
    normal_steps: int
    normal_resolution: int


def score_all(entries: list[ViewEntry], cfg: ScoreConfig) -> dict[int, Scores]:
    clip_scorer: CLIPScorer | None = None
    aesthetic_scorer: AestheticScorer | None = None
    normal_scorer: NormalAgreementScorer | None = None

    want_clip = not (cfg.skip_clip_prompt and cfg.skip_clip_viewpoint and cfg.skip_aesthetic)
    if want_clip:
        clip_scorer = CLIPScorer(cfg.clip_model, cfg.clip_pretrained, cfg.device, cfg.dtype)
    if not cfg.skip_aesthetic:
        assert clip_scorer is not None
        aesthetic_scorer = AestheticScorer(clip_scorer, cfg.hf_home)
    if not cfg.skip_normal:
        normal_scorer = NormalAgreementScorer(
            device=cfg.device,
            dtype=cfg.dtype,
            hf_home=cfg.hf_home,
            num_inference_steps=cfg.normal_steps,
            resolution=cfg.normal_resolution,
        )

    results: dict[int, Scores] = {entry.view_id: Scores() for entry in entries}

    # -- CLIP (prompt + viewpoint) + aesthetic in batches -----------------
    if clip_scorer is not None:
        for start in range(0, len(entries), cfg.batch_size):
            batch = entries[start : start + cfg.batch_size]
            pil_images = [Image.open(e.image_path).convert("RGB") for e in batch]
            img_feats = clip_scorer.embed_images(pil_images)

            if not cfg.skip_clip_prompt:
                prompts = [e.prompt or "" for e in batch]
                has_prompt = [bool(p.strip()) for p in prompts]
                if any(has_prompt):
                    txt_feats = clip_scorer.embed_texts(prompts)
                    sims = (img_feats * txt_feats).sum(dim=-1).cpu().tolist()
                    for e, sim, ok in zip(batch, sims, has_prompt):
                        if ok:
                            results[e.view_id].clip_prompt = float(sim)

            if not cfg.skip_clip_viewpoint:
                captions: list[str] = []
                valid: list[bool] = []
                for e in batch:
                    if e.view_matrix is None:
                        captions.append("")
                        valid.append(False)
                    else:
                        captions.append(viewpoint_caption(cfg.subject, e.view_matrix))
                        valid.append(True)
                if any(valid):
                    txt_feats = clip_scorer.embed_texts(captions)
                    sims = (img_feats * txt_feats).sum(dim=-1).cpu().tolist()
                    for e, sim, ok in zip(batch, sims, valid):
                        if ok:
                            results[e.view_id].clip_viewpoint = float(sim)

            if aesthetic_scorer is not None:
                aes = aesthetic_scorer.score(img_feats).cpu().tolist()
                for e, a in zip(batch, aes):
                    results[e.view_id].aesthetic = float(a)

            log.info(
                "CLIP/aesthetic batch %d/%d done",
                min(start + cfg.batch_size, len(entries)),
                len(entries),
            )

    # -- Normal agreement, per-image (Marigold does not batch cleanly) ----
    if normal_scorer is not None:
        for i, e in enumerate(entries):
            if e.normal_path is None or not e.normal_path.exists():
                log.warning("view %d: no rendered normal at %s, skipping normal score",
                            e.view_id, e.normal_path)
                continue
            pil = Image.open(e.image_path).convert("RGB")
            score = normal_scorer.score(pil, e.normal_path, e.mask_path)
            results[e.view_id].normal_agreement = score
            if (i + 1) % 10 == 0 or i == len(entries) - 1:
                log.info("Normal-agreement %d/%d", i + 1, len(entries))

    return results


# ---------------------------------------------------------------------------
# meta.json I/O
# ---------------------------------------------------------------------------


def load_meta_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}, got {type(data).__name__}")
    return data


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
        tmp = Path(fh.name)
    tmp.replace(path)


def compute_auto_exclude(
    results: dict[int, Scores],
    thresholds: dict[str, float],
) -> list[int]:
    excluded: set[int] = set()
    for vid, s in results.items():
        for key, thresh in thresholds.items():
            val = getattr(s, key, None)
            if val is None:
                continue
            if val < thresh:
                excluded.add(vid)
                break
    return sorted(excluded)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Score generated single views for automatic filtering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("path", type=Path,
                    help="Either the single-view style directory (e.g. .../single_view/civitai2.0) "
                         "or the masked-view directory itself (.../gen_view_masked).")
    ap.add_argument("--view-subdir", default="gen_view_masked")
    ap.add_argument("--view-channel", default=None,
                    help="Channel suffix in the view filename. None (default) matches "
                         "viewNNNN.png; set e.g. 'basecolor' to match viewNNNN.basecolor.png "
                         "when scoring gen_view_decomposite/ instead of gen_view_masked/.")
    ap.add_argument("--normal-subdir", default="condition_output/normal")
    ap.add_argument("--mask-subdir", default="condition_output/mask")
    ap.add_argument("--meta-subdir", default="meta")
    ap.add_argument("--meta-filename", default="meta.json")
    ap.add_argument("--subject", default=None,
                    help="Subject noun for the viewpoint caption. Defaults to the "
                         "mesh-name directory (two levels above single_view_dir).")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--batch-size", type=int, default=16)

    ap.add_argument("--limit", type=int, default=None, help="Only score the first N views.")
    ap.add_argument("--views", type=int, nargs="*", default=None,
                    help="Specific view ids to score (default: all).")

    ap.add_argument("--clip-model", default="ViT-L-14")
    ap.add_argument("--clip-pretrained", default="openai")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"),
                    help="Hugging Face cache dir (default: HF_HOME or the library default).")
    ap.add_argument("--normal-steps", type=int, default=1,
                    help="Marigold-normals num_inference_steps (1 keeps it cheap).")
    ap.add_argument("--normal-resolution", type=int, default=512)

    ap.add_argument("--skip-clip-prompt", action="store_true")
    ap.add_argument("--skip-clip-viewpoint", action="store_true")
    ap.add_argument("--skip-aesthetic", action="store_true")
    ap.add_argument("--skip-normal", action="store_true")

    ap.add_argument("--apply", action="store_true",
                    help="Union auto-excluded ids into exclude_from_training_indices. "
                         "Without this flag, only scores are written (score-only).")
    ap.add_argument("--force", action="store_true",
                    help="Rescore views that already have an entry in view_scores.")

    ap.add_argument("--thresh-clip-prompt", type=float, default=0.20,
                    help="Exclude view if clip_prompt below this (only with --apply).")
    ap.add_argument("--thresh-clip-viewpoint", type=float, default=0.18)
    ap.add_argument("--thresh-aesthetic", type=float, default=3.5)
    ap.add_argument("--thresh-normal", type=float, default=0.30)

    return ap.parse_args()


def infer_subject(paths: RunPaths, override: str | None) -> str:
    if override:
        return override
    # .../<mesh_name>/single_view/<style>/  → mesh_name is 2 levels up.
    try:
        mesh_name = paths.single_view_dir.parent.parent.name
    except Exception:
        mesh_name = "object"
    return mesh_name.replace("_", " ").replace("-", " ")


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    subject = infer_subject(paths, args.subject)
    log.info("single_view_dir : %s", paths.single_view_dir)
    log.info("view_dir        : %s", paths.view_dir)
    log.info("normal_dir      : %s", paths.normal_dir)
    log.info("mask_dir        : %s", paths.mask_dir)
    log.info("meta_dir        : %s", paths.meta_dir)
    log.info("meta_json       : %s", paths.meta_json)
    log.info("subject         : %r", subject)

    view_re = build_view_regex(args.view_channel)
    entries = discover_views(paths, args.limit, args.views, view_re=view_re)
    if not entries:
        log.error("No viewNNNN.png files found under %s", paths.view_dir)
        sys.exit(1)
    log.info("Discovered %d views.", len(entries))

    meta = load_meta_json(paths.meta_json)
    existing_scores: dict[str, dict[str, float]] = meta.get("view_scores", {}) or {}

    requested = [
        name for name, skip in [
            ("clip_prompt", args.skip_clip_prompt),
            ("clip_viewpoint", args.skip_clip_viewpoint),
            ("aesthetic", args.skip_aesthetic),
            ("normal_agreement", args.skip_normal),
        ] if not skip
    ]
    if not args.force and requested:
        def has_all(vid: int) -> bool:
            rec = existing_scores.get(str(vid), {})
            return all(k in rec for k in requested)
        todo = [e for e in entries if not has_all(e.view_id)]
        if len(todo) != len(entries):
            log.info("Skipping %d views that already have all requested metrics "
                     "(use --force to rescore).", len(entries) - len(todo))
        entries = todo

    for e in entries:
        load_view_metadata(e)

    if entries:
        cfg = ScoreConfig(
            subject=subject,
            skip_clip_prompt=args.skip_clip_prompt,
            skip_clip_viewpoint=args.skip_clip_viewpoint,
            skip_aesthetic=args.skip_aesthetic,
            skip_normal=args.skip_normal,
            batch_size=args.batch_size,
            device=args.device,
            dtype=torch.float16 if args.dtype == "float16" and args.device == "cuda" else torch.float32,
            clip_model=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            hf_home=args.hf_home,
            normal_steps=args.normal_steps,
            normal_resolution=args.normal_resolution,
        )
        new_scores = score_all(entries, cfg)
    else:
        new_scores = {}

    # --- Merge into meta.json.view_scores --------------------------------
    merged_scores = dict(existing_scores)
    for vid, s in new_scores.items():
        entry = dict(merged_scores.get(str(vid), {}))
        entry.update(s.as_dict())
        merged_scores[str(vid)] = entry
    meta["view_scores"] = merged_scores

    # --- Compute auto-excluded ids across ALL scored views ---------------
    all_scores_typed: dict[int, Scores] = {}
    for vid_str, d in merged_scores.items():
        try:
            vid = int(vid_str)
        except ValueError:
            continue
        all_scores_typed[vid] = Scores(
            clip_prompt=d.get("clip_prompt"),
            clip_viewpoint=d.get("clip_viewpoint"),
            aesthetic=d.get("aesthetic"),
            normal_agreement=d.get("normal_agreement"),
        )
    thresholds = {
        "clip_prompt": args.thresh_clip_prompt,
        "clip_viewpoint": args.thresh_clip_viewpoint,
        "aesthetic": args.thresh_aesthetic,
        "normal_agreement": args.thresh_normal,
    }
    auto_excluded = compute_auto_exclude(all_scores_typed, thresholds)
    meta["auto_excluded_indices"] = auto_excluded

    # --- exclude_from_training_indices: union manual + auto if --apply ---
    prior_exclude = set(meta.get("exclude_from_training_indices") or [])
    prior_auto = set(meta.get("auto_excluded_indices_applied") or [])
    manual = prior_exclude - prior_auto  # best estimate of "what the user set"
    if args.apply:
        final = sorted(manual | set(auto_excluded))
        meta["exclude_from_training_indices"] = final
        meta["auto_excluded_indices_applied"] = auto_excluded
        log.info("Applied auto-exclude: %d manual + %d auto → %d total excluded.",
                 len(manual), len(auto_excluded), len(final))
    else:
        log.info("Score-only run (no --apply). Auto-exclude list has %d views "
                 "at current thresholds; exclude_from_training_indices left unchanged.",
                 len(auto_excluded))

    meta["view_scores_meta"] = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "scored_view_dir": args.view_subdir,
        "clip_model": f"{args.clip_model} / {args.clip_pretrained}",
        "aesthetic_model": f"{_AESTHETIC_REPO}/{_AESTHETIC_FILE}",
        "normal_model": None if args.skip_normal else
            f"prs-eth/marigold-normals-v1-1 (num_inference_steps={args.normal_steps})",
        "subject": subject,
        "thresholds": thresholds,
        "applied": bool(args.apply),
    }

    atomic_write_json(paths.meta_json, meta)
    log.info("Wrote %s", paths.meta_json)

    # --- Report summary --------------------------------------------------
    if merged_scores:
        def stat(key: str) -> str:
            vals = [d[key] for d in merged_scores.values() if d.get(key) is not None]
            if not vals:
                return f"{key:18s}  (no data)"
            arr = np.asarray(vals, dtype=np.float64)
            return (f"{key:18s}  n={len(arr):4d}  "
                    f"min={arr.min():+.3f}  med={np.median(arr):+.3f}  "
                    f"max={arr.max():+.3f}  mean={arr.mean():+.3f}")

        log.info("Score summary:")
        for k in ("clip_prompt", "clip_viewpoint", "aesthetic", "normal_agreement"):
            log.info("  %s", stat(k))


if __name__ == "__main__":
    main()
