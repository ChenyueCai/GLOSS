#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""View-conditioned prompt generation.

For each rendered view this script
  1. captions the rendered surface-normal map with a vision LLM to describe
     the visible silhouette / orientation / salient geometry,
  2. generates an appearance prompt for the subject via a multi-turn
     batched text-LLM chat (one prompt per view, grouped by subject so
     anchor-aware runs get coherent per-anchor batches),
  3. concatenates `appearance_prompt + ", " + caption` into the final
     prompt that feeds SDXL / ComfyUI step 3.

Outputs:
  * the flat file ``<prompts_dir>/<mesh_name>.txt`` (one prompt per line,
    indexed by view), which `generate_views_sdxl.py` reads directly,
  * the per-view yaml ``<single_view_dir>/meta/view{idx:04d}.yml``,
    where the ``prompt:`` key is added/overwritten without touching the
    existing ``camera:`` block.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import pathlib
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from omegaconf import OmegaConf
from openai import OpenAI, RateLimitError


# -------- caption (vision) ---------------------------------------------------

CAPTION_SYS_MSG = (
    "You are looking at a rendered camera-space surface-normal map of a 3D "
    "object (RGB channels encode XYZ surface orientation). In one sentence "
    "of 10-18 words, describe the visible silhouette, pose/orientation, and "
    "salient geometric features of the object. Do NOT mention colors, "
    "materials, lighting, or the fact that this is a normal map."
)


# -------- appearance prompts (text LLM, multi-turn batched) ------------------

APPEARANCE_SYS_MSG = (
    "Generate concise, diverse and vivid prompts (~15-25 words) for objects. "
    "Output one prompt per line, no numbering."
)


_client: "OpenAI | None" = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _retry_chat(messages, model: str, max_retries: int = 12, **kwargs):
    """Resilient chat-completion call with exponential back-off + jitter,
    capped at 60s per wait. Default 12 retries (~5 min worst case)."""
    import random as _random
    for n in range(max_retries):
        try:
            return _get_client().chat.completions.create(
                model=model, messages=messages, **kwargs
            )
        except RateLimitError:
            wait = min(60.0, (2 ** n) * (0.6 + 0.4 * _random.random()))
            print(f"429 - retrying in {wait:.1f}s (attempt {n + 1}/{max_retries})",
                  file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Too many retries against {model}")


# -----------------------------------------------------------------------------
# Appearance prompt generation (multi-turn batched chat, per subject)
# -----------------------------------------------------------------------------

def generate_appearance_prompts(subject: str, num: int, model: str) -> List[str]:
    """Multi-turn batched chat producing ``num`` appearance prompts for ``subject``.

    Mirrors the legacy ``scripts/datagen/steps/generate_prompts.py`` flow:
    seed with a request for a batch of 10 (or ``num`` if smaller), keep the
    assistant reply in the conversation, then repeatedly ask for the next
    rounded-up-to-10 chunk until ``num`` non-empty lines are accumulated.
    Trims to exactly ``num``.
    """
    if num <= 0:
        return []
    batch = min(10, num)
    messages = [
        {"role": "system", "content": APPEARANCE_SYS_MSG},
        {"role": "user", "content":
            f"Generate {batch} prompts for different kinds of **{subject}** "
            f"with different appearances in a studio setting. Use vivid short "
            f"object centric adjectives instead of long prose"},
    ]
    resp = _retry_chat(messages, model)
    first = resp.choices[0].message.content.strip()
    messages.append({"role": "assistant", "content": first})
    lines = [ln.strip() for ln in first.splitlines() if ln.strip()]

    while len(lines) < num:
        remaining = int(math.ceil((num - len(lines)) / 10.0) * 10)
        messages.append({"role": "user", "content":
            f"Great! Now generate {remaining} more prompts for **{subject}** "
            f"with different appearances just like that."})
        resp = _retry_chat(messages, model)
        extra = resp.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": extra})
        lines.extend(ln.strip() for ln in extra.splitlines() if ln.strip())
        print(f"  appearance prompts for {subject!r}: {len(lines)}/{num}")

    return lines[:num]


# -----------------------------------------------------------------------------
# Per-view captioning (vision)
# -----------------------------------------------------------------------------

def _png_to_data_url(path: pathlib.Path) -> str:
    b = path.read_bytes()
    enc = base64.b64encode(b).decode("ascii")
    return f"data:image/png;base64,{enc}"


def caption_normal(normal_path: pathlib.Path, subject: str, model: str) -> str:
    user_text = (
        f"Caption the geometry of this {subject} as visible from the rendered "
        f"angle, following the system instructions."
    )
    messages = [
        {"role": "system", "content": CAPTION_SYS_MSG},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url",
             "image_url": {"url": _png_to_data_url(normal_path)}},
        ]},
    ]
    resp = _retry_chat(messages, model)
    return resp.choices[0].message.content.strip().splitlines()[0].strip()


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def discover_normal_files(condition_dir: pathlib.Path) -> List[pathlib.Path]:
    normal_dir = condition_dir / "normal"
    if not normal_dir.is_dir():
        raise FileNotFoundError(
            f"normal directory not found: {normal_dir}. "
            f"Run generate_condition.py first."
        )
    files = sorted(p for p in normal_dir.iterdir()
                   if p.suffix.lower() == ".png")
    if not files:
        raise FileNotFoundError(f"no .png files under {normal_dir}")
    return files


def view_index_from_path(p: pathlib.Path) -> int:
    # filenames look like normal0007.png
    digits = "".join(ch for ch in p.stem if ch.isdigit())
    if not digits:
        raise ValueError(f"cannot extract view index from {p.name}")
    return int(digits)


def update_meta_yaml(meta_dir: pathlib.Path, idx: int, prompt: str) -> None:
    yml_path = meta_dir / f"view{idx:04d}.yml"
    if yml_path.exists():
        cfg = OmegaConf.load(yml_path)
    else:
        cfg = OmegaConf.create({})
    cfg.prompt = prompt
    OmegaConf.save(config=cfg, f=yml_path)


def load_anchor_name_map(orientation_json: Optional[pathlib.Path]) -> Dict[str, str]:
    # Map anchor id (e.g. "a1") -> human-readable name from orientation.json.
    if orientation_json is None or not orientation_json.is_file():
        return {}
    data = json.loads(orientation_json.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    for idx, a in enumerate(data.get("anchors") or []):
        aid = a.get("id") or f"a{idx+1}"
        nm = a.get("name")
        if nm:
            out[aid] = nm
    return out


def view_anchor_id(meta_dir: pathlib.Path, idx: int) -> Optional[str]:
    yml_path = meta_dir / f"view{idx:04d}.yml"
    if not yml_path.exists():
        return None
    cfg = OmegaConf.load(yml_path)
    aid = cfg.get("anchor_id") if hasattr(cfg, "get") else None
    return str(aid) if aid is not None else None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single_view_dir", type=pathlib.Path, required=True,
                    help="Path to the single-view dir, e.g. "
                         "<expr>/<mesh>/single_view/civitai2.0. Reads "
                         "condition_output/normal/, writes meta/view*.yml.")
    ap.add_argument("--prompts_fp", type=pathlib.Path, required=True,
                    help="Flat prompts file destination, "
                         "e.g. <expr>/<mesh>/prompts/<mesh>.txt. One prompt per line.")
    ap.add_argument("--subject", type=str, required=True,
                    help="Human-readable subject for prompt seeding "
                         "(e.g. 'rusty barrel').")
    ap.add_argument("--num_views", type=int, default=None,
                    help="Number of views to caption. Default: count of "
                         "condition_output/normal/*.png.")
    ap.add_argument("--vlm_model", type=str, default="gpt-5",
                    help="Vision-capable OpenAI model for normal captioning.")
    ap.add_argument("--text_model", type=str, default="gpt-5",
                    help="OpenAI model for the multi-turn batched appearance "
                         "prompt generation (one batched conversation per "
                         "unique subject).")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel VLM workers (threads) for normal captioning.")
    ap.add_argument("--force", action="store_true",
                    help="Re-generate prompts even if prompts_fp exists.")
    ap.add_argument("--orientation_json", type=pathlib.Path, default=None,
                    help="Optional path to single_view/orientation.json. When "
                         "set, views whose meta yaml carries an 'anchor_id' "
                         "compose the subject as '<subject> <anchor.name>' "
                         "(per-anchor naming for anchors-mode renders).")
    args = ap.parse_args()

    sv = args.single_view_dir
    condition_dir = sv / "condition_output"
    meta_dir = sv / "meta"
    if not meta_dir.is_dir():
        raise FileNotFoundError(
            f"meta dir not found: {meta_dir}. Run generate_condition.py first.")

    args.prompts_fp.parent.mkdir(parents=True, exist_ok=True)

    if args.prompts_fp.exists() and not args.force:
        print(f"{args.prompts_fp} already exists; pass --force to regenerate.")
        return

    normal_files = discover_normal_files(condition_dir)
    indices = [view_index_from_path(p) for p in normal_files]
    if args.num_views is not None:
        normal_files = normal_files[: args.num_views]
        indices = indices[: args.num_views]
    n = len(normal_files)
    print(f"Captioning {n} views from {condition_dir/'normal'}")

    anchor_names = load_anchor_name_map(args.orientation_json)
    per_view_subjects: List[str] = []
    for idx in indices:
        aid = view_anchor_id(meta_dir, idx) if anchor_names else None
        nm = anchor_names.get(aid) if aid else None
        per_view_subjects.append(f"{args.subject} {nm}" if nm else args.subject)
    if anchor_names:
        uniq = sorted(set(per_view_subjects))
        print(f"Anchor-aware subjects ({len(uniq)} variants): {uniq}")

    # Group views by subject so each unique subject gets its own batched
    # appearance-prompt conversation; then assign one prompt per view.
    subject_to_indices: Dict[str, List[int]] = defaultdict(list)
    for local_i, subj in enumerate(per_view_subjects):
        subject_to_indices[subj].append(local_i)

    print(f"Generating appearance prompts via {args.text_model} for "
          f"{len(subject_to_indices)} unique subject(s) ...")
    appearance_per_view: List[Optional[str]] = [None] * n
    for subj, idxs in subject_to_indices.items():
        prompts = generate_appearance_prompts(subj, len(idxs), args.text_model)
        for k, local_i in enumerate(idxs):
            appearance_per_view[local_i] = prompts[k]

    captions: List[str | None] = [None] * n

    def _caption_one(i: int) -> tuple[int, str]:
        cap = caption_normal(normal_files[i], per_view_subjects[i], args.vlm_model)
        return i, cap

    print(f"Captioning normals with {args.vlm_model} "
          f"(workers={args.workers}) ...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_caption_one, i) for i in range(n)]
        done = 0
        for fut in as_completed(futures):
            i, cap = fut.result()
            captions[i] = cap
            done += 1
            if done % 25 == 0 or done == n:
                print(f"  captioned {done}/{n}")

    # Combine and write outputs. Appearance prompts already mention the
    # subject (legacy mechanism), so we don't re-prepend `a {subject}`.
    final_prompts: List[str] = []
    for i in range(n):
        appearance = (appearance_per_view[i] or "").rstrip(".").strip()
        cap = (captions[i] or "").rstrip(".").strip()
        prompt = f"{appearance}, {cap}" if cap else appearance
        final_prompts.append(prompt)
        update_meta_yaml(meta_dir, indices[i], prompt)

    args.prompts_fp.write_text("\n".join(final_prompts) + "\n", encoding="utf-8")
    print(f"Wrote {n} prompts to {args.prompts_fp}")
    print(f"Updated meta yamls under {meta_dir}")


if __name__ == "__main__":
    main()
