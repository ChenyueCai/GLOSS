#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import os, argparse, pathlib, time, json, sys, shutil
import math
from typing import Iterable, List

from openai import OpenAI, RateLimitError        # openai‑python ≥ 1.3  :contentReference[oaicite:0]{index=0}


MODEL = "gpt-5"        # swap for gpt‑3.5‑turbo‑0125 if you’re cost‑sensitive
SYS_MSG = (
    "Generate concise, diverse and vivid prompts (~15‑25 words) for objects. Output *one prompt per line*, no numbering."
)

_client: "OpenAI | None" = None

def _get_client() -> "OpenAI":
    """Lazily create the OpenAI client so the module can be imported without OPENAI_API_KEY."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

# ---------- OpenAI helper with gentle exponential back‑off --------------------
def chat(messages, model="gpt-5", stream=False):
    for n in range(6):
        try:
            return _get_client().chat.completions.create(
                model=model, messages=messages, stream=stream
            )
        except RateLimitError as e:
            wait = 2**n
            print(f"429 – retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("Too many retries")

def generate(subject: str, out_path: pathlib.Path, model: str = "gpt-5", per_subject: int = 500):
    total = 0
    batch = min(10, per_subject)
    messages = [{"role": "system", "content": SYS_MSG}]
    # Turn 1 – ask for first batch
    messages.append({"role": "user",
                     "content": f"Generate {batch} prompts for different kinds of **{subject}** with different appearances in a studio setting. Use vivid short object centric adjectives instead of long prose"})
    resp = chat(messages, model)
    first_batch = resp.choices[0].message.content.strip()
    messages.append({"role": "assistant", "content": first_batch})

    print("Preview (first batch):\n" + first_batch, "\n---\nLooks good ✓")
    prompts = first_batch
    total = len(prompts.splitlines())
    # Continue batching until we reach per_subject
    while total < per_subject:
        remaining = int(math.ceil((per_subject - total) / 10.0) * 10)
        messages.append({"role": "user",
                        "content": f"Great! Now generate {remaining} more prompts for **{subject}** with different appearances just like that."})
        resp = chat(messages, model)
        extra = resp.choices[0].message.content.strip()
        prompts = prompts + "\n" + extra
        total = len(prompts.splitlines())
        print(f"Generating {total} prompts so far")

    # Save
    out_path.write_text(prompts, encoding="utf-8")
    print(f"Saved {out_path} ({len(prompts.splitlines())} lines)")

# ---------- CLI wrapper -------------------------------------------------------
def iter_subjects(args):
    # ----- plain CLI list -----
    if args.subjects:
        for s in args.subjects:
            yield (s.replace(" ", "_"), s)      # key = safe file stem

    # ----- .json file -----
    if args.subjects_json:
        import json, pathlib
        data = json.loads(pathlib.Path(args.subjects_json).read_text("utf-8"))

        # list → ["mushroom", "barrel", …]
        if isinstance(data, list):
            for idx, item in enumerate(data):
                subj = str(item)
                yield (f"item_{idx:03d}", subj)

        # dict → {"mesh_001": "mushroom", "mesh_002": "barrel"}
        elif isinstance(data, dict):
            for key, value in data.items():
                yield (str(key), str(value))

        else:
            raise ValueError("--subjects-json must be a list or dict")


def main() -> None:

    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", help="subjects (e.g., squash seashell barrel)")
    ap.add_argument("--subjects-json", type=str, default=None,
                    help="JSON file with subjects: list or dict of mesh_name→subject")
    ap.add_argument("--per-subject", type=int, default=500, help="# prompts per subject")
    ap.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("prompts"))
    ap.add_argument("--stream", action="store_true", help="stream responses (lower latency)")
    ap.add_argument("--model", default="o3")
    args = ap.parse_args()

    if not args.subjects and not args.subjects_json:
        ap.error("Provide --subjects or --subjects-json")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for key, subj in iter_subjects(args):
        out_path = args.out_dir / key / "prompts" / f"{key}.txt"
        if os.path.exists(out_path):
            dst_path = args.out_dir / key / "prompts" / f"{key}-v0.txt"
            shutil.move(out_path, dst_path)
        print(f"\n🔸 Generating {args.per_subject} prompts for {subj!r} → {out_path}")
        generate(subj, out_path, model=args.model, per_subject=args.per_subject)

if __name__ == "__main__":
    main()
