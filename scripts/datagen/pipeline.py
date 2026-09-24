#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end data generation pipeline for material super-resolution.

All intermediate paths are derived from --data_dir, --expr_tag, and --mesh_name,
so no hard-coded absolute paths are needed.

Pipeline steps
--------------
1  generate_condition   Render conditioning images (normals, depth, canny) and camera
                        metadata from the mesh. No prompts needed at this stage.
2  generate_view_prompts View-conditioned prompt generation: for each unique
                        subject (one or more, when anchor-aware), a multi-turn
                        batched text-LLM chat produces appearance prompts; for
                        each view a vision LLM captions the rendered
                        surface-normal map; the per-view final prompt is the
                        appearance prompt concatenated with the geometry
                        caption. Writes a flat prompts/<mesh>.txt and fills
                        the prompt field of each meta/view*.yml.
3  generate_views        Generate texture-stylised view images.
                        Default back-end: SD1.5 + Realistic Vision V5.1
                        with normal/depth/canny ControlNets at strengths
                        1.0 / 0.7 / 0.5 (config sd15_rv_normal_depth_canny).
                        Alternative back-ends:
                          - SDXL + ControlNet Union          (pass --use_sdxl)
                          - legacy ComfyUI (requires --comfyui_dir)
                                                             (pass --use_comfyui)
4  postprocess_mask     Apply foreground mask to generated view images.
   (critique pass)      Right after step 4 the score_views step runs by default:
                        scores each masked view (LAION aesthetic + CLIP prompt
                        alignment) and writes view_scores + auto_excluded_indices
                        into single_view/meta.json. Score-only: actually rejecting
                        views is a manual step via:
                          python scripts/datagen/utils/score_viewer.py        <sv>
                          python scripts/datagen/utils/review_training_views.py <sv>
                        which writes exclude_from_training_indices. Steps 5, 7,
                        and 8 honor that field (step 6 InvSR cascades). Pass
                        --scoring_apply to skip the manual review and use the
                        auto-thresholded list directly. Disable with --no_scoring.
5  decompose            Run intrinsic image decomposition (basecolor / metallic /
                        roughness / normal …) using diffusion-renderer.
                        Requires --diffrender_dir (and optionally --hf_home).
                        Falls back to an accelerate-based script when
                        --decomp_config / --decomp_script / --decomp_weights
                        are supplied instead.
6  superresolution      Upscale decomposed images with InvSR.
                        Requires --invsr_dir.
7  backproject          Backproject super-resolved basecolor onto the mesh UV space
                        and compute per-face sampling weights.
8  datagen              Generate multi-view training samples and pack them into
                        WebDataset .tar shards.

Usage
-----
    python scripts/datagen/pipeline.py \\
        --data_dir $GLOSS_DATA_DIR \\
        --expr_tag my_experiment \\
        --mesh_name rusty_barrel_metal

    # Run only backproject + datagen, force re-run even if DONE exists:
    python scripts/datagen/pipeline.py \\
        --data_dir /data/mat \\
        --expr_tag test \\
        --mesh_name barrel \\
        --steps 7 8 --force
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repo-relative constants
# ---------------------------------------------------------------------------
# pipeline.py lives at  <repo>/scripts/datagen/pipeline.py
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent   # scripts/datagen/
STEPS_DIR = SCRIPTS_DIR / "steps"               # scripts/datagen/steps/

ALL_STEPS = list(range(1, 9))

STEP_NAMES = {
    1: "generate_condition",
    2: "generate_view_prompts",
    3: "generate_views",   # comfyui_views (default) or sdxl_views (--use_sdxl)
    4: "postprocess_mask",
    5: "decompose",
    6: "superresolution",
    7: "backproject",
    8: "datagen",
}


MESHES_JSON = REPO_ROOT / "assets" / "meshes.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_mesh_info(mesh_name: str) -> Optional[dict]:
    """Return the meshes.json entry for *mesh_name*, or None if not found."""
    if not MESHES_JSON.exists():
        return None
    with open(MESHES_JSON) as f:
        entries = json.load(f)
    for entry in entries:
        if entry.get("name") == mesh_name:
            return entry
    return None


def run(cmd: str, cwd: Path, log_file: Path) -> None:
    """Run *cmd* in a shell, writing stdout+stderr to *log_file*."""
    log.info("Running in %s:\n  %s", cwd, cmd.strip())
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w") as fh:
        proc = subprocess.run(cmd, shell=True, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        log.error("Command failed (exit %d). See %s", proc.returncode, log_file)
        sys.exit(proc.returncode)


def done_marker(d: Path) -> Path:
    return d / "DONE"


def is_done(d: Path) -> bool:
    return done_marker(d).is_file()


def mark_done(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    done_marker(d).touch()


def python_cmd(env: str, script: Path, args_str: str) -> str:
    """Build a `conda run` python invocation."""
    return f"conda run -n {env} python {script} {args_str}"


def relative_to_data_root(path: Path, data_dir: Path) -> Path:
    """Return *path* relative to *data_dir*, exiting with a clear error otherwise."""
    try:
        return path.resolve().relative_to(data_dir.resolve())
    except ValueError:
        try:
            return path.absolute().relative_to(data_dir.absolute())
        except ValueError:
            log.error("Path %s is not inside data_dir %s", path, data_dir)
            sys.exit(1)


# ---------------------------------------------------------------------------
# Path builder
# ---------------------------------------------------------------------------

def build_paths(args) -> dict:
    """
    All paths are derived from --data_dir / --expr_tag / --mesh_name.

    Canonical layout
    ----------------
    {data_dir}/
      {expr_tag}/
        config.yaml                   # per-experiment view configs
        {mesh_name}/
          mesh/
            scene.gltf
          prompts/
            {mesh_name}.txt
          single_view/
            <sv_subdir>/              # default civitai2.0; --sv_subdir overrides
                                      # (e.g. origin / anchors for runs driven
                                      # by orientation.json)
              meta/                   # camera .yml files
              condition_output/       # normals, depth, canny, mask
              gen_view/               # raw diffusion-generated images
              gen_view_masked/        # foreground-masked images
              gen_view_decomposite/   # intrinsic decomposition channels
              gen_view_super/         # super-resolved channels
              textures_sr/            # backprojected UV textures
              sampling/               # per-face sampling weights
          multi_view/
            {dataset_tag}/            # per-view datacaches
            {dataset_tag}-wds/        # WebDataset .tar shards
    """
    data_dir = Path(args.data_dir).resolve()
    expr_root = data_dir / args.expr_tag
    expr_dir = expr_root / args.mesh_name
    sv_subdir = getattr(args, "sv_subdir", "civitai2.0")
    sv = expr_dir / "single_view" / sv_subdir
    mesh_name = args.mesh_name
    # Allow overriding the mesh subpath (default: mesh/scene.gltf)
    mesh_subpath = getattr(args, "mesh_subpath", "mesh/scene.gltf")
    # Prompts live INSIDE the per-sv-subdir folder so origin and anchors
    # passes for the same mesh don't clobber each other (the captions are
    # per-view and depend on the rendered normals, which differ by camera
    # set, so they MUST be per-sv-subdir). The DONE marker + log get their
    # own ``prompts_meta`` subdir to avoid colliding with other steps'
    # DONE markers.
    return {
        "data_dir": data_dir,
        "expr_root": expr_root,
        "expr_dir": expr_dir,
        "mesh_fp": expr_dir / mesh_subpath,
        "prompts_dir": sv / "prompts_meta",
        "prompts_fp": sv / "prompts.txt",
        "single_view_dir": sv,
        "config_yaml": expr_root / "config.yaml",
    }


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def step_generate_condition(args, paths: dict) -> None:
    """Step 1 – render conditioning images and camera metadata.

    Prompts are generated AFTER this step (step 2), so meta/view*.yml is
    written without a prompt field; step 2 fills it in.
    """
    cond_out = paths["single_view_dir"] / "condition_output"
    if is_done(cond_out):
        log.info("Step 1 (generate_condition): already done, skipping.")
        return

    config_yaml = paths["config_yaml"]
    log.info("Updating config.yaml from meshes.json …")
    _ensure_config_yaml(config_yaml, paths, args)

    overwrite = "True" if args.force else "False"
    mode = getattr(args, "condition_mode", "generate")
    cmd = python_cmd(
        args.gloss_env,
        STEPS_DIR / "generate_condition.py",
        (
            f"--mesh={paths['mesh_fp']} "
            f"--mesh_name={args.mesh_name} "
            f"--num_views={args.num_prompts} "
            f"--output_dir={paths['single_view_dir']} "
            f"--configs={config_yaml} "
            f"--overwrite={overwrite} "
            f"--mode={mode}"
        ),
    )
    paths["single_view_dir"].mkdir(parents=True, exist_ok=True)
    run(cmd, REPO_ROOT, paths["single_view_dir"] / "generate_condition_log.txt")
    mark_done(cond_out)


def step_generate_view_prompts(args, paths: dict) -> None:
    """Step 2 – view-conditioned prompt generation.

    For each rendered view (step 1 output), a vision LLM captions the
    surface-normal map; the caption is combined with a material/color/look
    modifier (from a text-LLM pool) into the final prompt. Writes both the
    flat prompts/<mesh>.txt (consumed by step 3) and the prompt field of
    each meta/view*.yml.
    """
    out = paths["prompts_dir"]
    if is_done(out):
        log.info("Step 2 (generate_view_prompts): already done, skipping.")
        return

    subject = args.mesh_subject or args.mesh_name
    extra = ""
    if args.force:
        extra += " --force"
    orientation_json = paths["single_view_dir"].parent / "orientation.json"
    if orientation_json.is_file():
        extra += f" --orientation_json={orientation_json}"
    cmd = python_cmd(
        args.gloss_env,
        STEPS_DIR / "generate_view_prompts.py",
        (
            f"--single_view_dir={paths['single_view_dir']} "
            f"--prompts_fp={paths['prompts_fp']} "
            f"--subject={subject!r} "
            f"--num_views={args.num_prompts} "
            f"--vlm_model={args.vlm_model} "
            f"--text_model={args.text_model} "
            f"--workers={args.prompt_workers}"
            + extra
        ),
    )
    out.mkdir(parents=True, exist_ok=True)
    run(cmd, REPO_ROOT, out / "log.txt")
    mark_done(out)


def _ensure_config_yaml(config_yaml: Path, paths: dict, args) -> None:
    """Auto-generate the per-experiment config.yaml with default view parameters."""
    config_yaml.parent.mkdir(parents=True, exist_ok=True)
    cmd = python_cmd(
        args.gloss_env,
        STEPS_DIR / "generate_config_yml.py",
        f"{paths['expr_root']} -o {config_yaml}"
        + (f" --meshes_json {MESHES_JSON}" if MESHES_JSON.is_file() else ""),
    )
    run(cmd, REPO_ROOT, config_yaml.parent / "generate_config_log.txt")


def step_sdxl_views(args, paths: dict) -> None:
    """Step 3 (SDXL path) – generate 1024×1024 views with SDXL + ControlNet Union."""
    out = paths["single_view_dir"] / "gen_view"
    if is_done(out):
        log.info("Step 3 (sdxl_views): already done, skipping.")
        return

    extra = ""
    if args.canny_normal:
        extra += " --canny_normal"
    if args.hf_home:
        extra += f" --hf_home {args.hf_home}"
    if args.sdxl_seed is not None:
        extra += f" --seed {args.sdxl_seed}"

    cmd = python_cmd(
        args.sdxl_env,
        STEPS_DIR / "generate_views_sdxl.py",
        (
            f"--mesh_name {args.mesh_name} "
            f"--exp_dir {paths['expr_root']} "
            f"--sdxl_model {args.sdxl_model} "
            f"--controlnet_model {args.controlnet_model} "
            f"--num_inference_steps {args.sdxl_steps} "
            f"--guidance_scale {args.sdxl_guidance_scale} "
            f"--normal_weight {args.sdxl_normal_weight} "
            f"--depth_weight {args.sdxl_depth_weight} "
            f"--canny_weight {args.sdxl_canny_weight} "
            f"--sv_subdir {args.sv_subdir}"
            + extra
        ),
    )
    out.mkdir(parents=True, exist_ok=True)
    run(cmd, REPO_ROOT, out / "log.txt")
    mark_done(out)


def step_comfyui_views(args, paths: dict) -> None:
    """Step 3 (ComfyUI path) – generate view images (requires --comfyui_dir)."""
    out = paths["single_view_dir"] / "gen_view"
    if is_done(out):
        log.info("Step 3 (comfyui_views): already done, skipping.")
        return

    if not args.comfyui_dir:
        log.warning(
            "Step 3 (comfyui_views): --comfyui_dir is not set.\n"
            "\n"
            "  Please run ComfyUI manually to generate views, then re-run the\n"
            "  pipeline starting from step 4:\n"
            "\n"
            "    cd <ComfyUI_dir>\n"
            "    conda run -n %s python generate_views.py --mesh_name %s --exp_dir %s\n"
            "\n"
            "  Optional one-time setup if that environment is not ready yet:\n"
            "    pip install -r requirements.txt\n"
            "\n"
            "  Useful optional flags:\n"
            "    --canny_normal          use canny-normal instead of canny-geonormal\n"
            "    --canny_weight 0.4      ControlNet canny strength\n"
            "    --visualize             start the ComfyUI web UI at localhost:8188\n"
            "\n"
            "  Output should appear in:\n"
            "    %s\n"
            "\n"
            "  Once done, touch a DONE marker:\n"
            "    touch %s/DONE\n"
            "\n"
            "  Then continue with:\n"
            "    python scripts/datagen/pipeline.py ... --steps 4 5 6 7 8",
            args.gloss_env,
            args.mesh_name,
            paths["expr_root"],
            out,
            out,
        )
        sys.exit(0)

    comfyui_dir = Path(args.comfyui_dir)
    gen_cmd = (
        f"conda run -n {args.gloss_env} python generate_views.py "
        f"--mesh_name {args.mesh_name} "
        f"--exp_dir {paths['expr_root']}"
    )
    if getattr(args, "canny_normal", False):
        gen_cmd += " --canny_normal"
    if getattr(args, "canny_weight", None) is not None:
        gen_cmd += f" --canny_weight {args.canny_weight}"
    if getattr(args, "sdxl_normal_weight", None) is not None:
        gen_cmd += f" --normal_weight {args.sdxl_normal_weight}"
    if getattr(args, "sdxl_depth_weight", None) is not None:
        gen_cmd += f" --depth_weight {args.sdxl_depth_weight}"
    if getattr(args, "comfyui_visualize", False):
        port = getattr(args, "comfyui_port", 8188)
        gen_cmd += f" --visualize --port {port}"
    if getattr(args, "comfyui_install_requirements", False):
        cmd = f"pip install -r requirements.txt && {gen_cmd}"
    else:
        cmd = gen_cmd
    out.mkdir(parents=True, exist_ok=True)
    run(cmd, comfyui_dir, out / "log.txt")
    mark_done(out)


def step_postprocess_mask(args, paths: dict) -> None:
    """Step 4 – mask out the background from generated views."""
    sv = paths["single_view_dir"]
    out = sv / "gen_view_masked"
    if is_done(out):
        log.info("Step 4 (postprocess_mask): already done, skipping.")
        return

    cmd = python_cmd(
        args.gloss_env,
        STEPS_DIR / "postprocess_mask.py",
        f"--output_dir={sv}",
    )
    out.mkdir(parents=True, exist_ok=True)
    run(cmd, REPO_ROOT, sv / "postprocess_mask_log.txt")
    mark_done(out)


def step_score_views(args, paths: dict) -> None:
    """Post-step after step 4 (default ON) — critique single views and write
    scores to meta.json. Score-only by default: filtering is a manual step
    via the browser UIs (see hint printed at the end).

    Writes ``view_scores`` and ``auto_excluded_indices`` into the single-view
    ``meta.json``. With ``--scoring_apply`` (off by default) auto-flagged ids
    are unioned into ``exclude_from_training_indices`` immediately; otherwise
    that field is updated manually via review_training_views.py.

    Steps 5, 7, and 8 honor whatever ``exclude_from_training_indices`` ends
    up containing (manual or auto). Step 6 InvSR cascades from step 5's
    filtered output dir.

    By default only LAION aesthetic + CLIP prompt-alignment run; Marigold
    normal-agreement and CLIP viewpoint-alignment are opt-in via flags.
    """
    sv = paths["single_view_dir"]
    done_dir = sv / "scores"
    if is_done(done_dir):
        log.info("Scoring (after step 4): already done, skipping.")
        _emit_review_hint(sv)
        return

    extra = ""
    if not args.scoring_normal_agreement:
        extra += " --skip-normal"
    if not args.scoring_clip_viewpoint:
        extra += " --skip-clip-viewpoint"
    if args.scoring_apply:
        extra += " --apply"
    extra += f" --thresh-clip-prompt {args.scoring_thresh_clip_prompt}"
    extra += f" --thresh-aesthetic {args.scoring_thresh_aesthetic}"
    if args.hf_home:
        extra += f" --hf-home {args.hf_home}"

    cmd = python_cmd(
        args.gloss_env,
        STEPS_DIR / "score_views.py",
        f"{sv}{extra}",
    )
    done_dir.mkdir(parents=True, exist_ok=True)
    run(cmd, REPO_ROOT, done_dir / "score_views_log.txt")
    mark_done(done_dir)
    _emit_review_hint(sv)


def _emit_review_hint(sv: Path) -> None:
    """Surface launch commands for the bucket viewer + review UI.

    score_viewer.py is read-only — it groups views into bad/weak/ok/good/
    excellent buckets per metric for inspection. review_training_views.py
    is the manual reject UI that writes exclude_from_training_indices.
    """
    log.info(
        "\nScores written to %s/meta.json. To filter views manually:\n"
        "  python scripts/datagen/utils/score_viewer.py %s --port 10014\n"
        "      (read-only; groups views into bad/weak/ok/good/excellent buckets)\n"
        "  python scripts/datagen/utils/review_training_views.py %s --port 10013\n"
        "      (writes exclude_from_training_indices on reject)\n"
        "Steps 5/7/8 will skip whatever ends up in exclude_from_training_indices.",
        sv, sv, sv,
    )


def step_decompose(args, paths: dict) -> None:
    """Step 5 – intrinsic image decomposition via diffusion-renderer (or a custom script)."""
    sv = paths["single_view_dir"]
    out = sv / "gen_view_decomposite"
    if is_done(out):
        log.info("Step 5 (decompose): already done, skipping.")
        return

    diffrender_dir = getattr(args, "diffrender_dir", None)
    diffrender_env = getattr(args, "diffrender_env", "diff-render")
    hf_home = getattr(args, "hf_home", None)
    diffrender_inference_res = getattr(args, "diffrender_inference_res", "512,512")
    diffrender_fallback_inference_res = getattr(
        args,
        "diffrender_fallback_inference_res",
        ["384,384", "256,256"],
    )
    diffrender_inference_n_steps = getattr(args, "diffrender_inference_n_steps", 20)

    if diffrender_dir:
        # Use diffusion-renderer: each masked view image is treated as the first
        # frame of a video; the model fills remaining frames by repetition.
        hf_flag = f" --hf_home={hf_home}" if hf_home else ""
        fallback_flag = ""
        if diffrender_fallback_inference_res:
            fallback_flag = (
                " --fallback_inference_res "
                + " ".join(str(res) for res in diffrender_fallback_inference_res)
            )
        meta_json_flag = ""
        if not args.no_scoring:
            meta_json_flag = f" --meta_json={sv / 'meta.json'}"
        cmd = (
            f"conda run -n {diffrender_env} python "
            f"{STEPS_DIR / 'run_diffrender_decompose.py'} "
            f"--input_dir={sv / 'gen_view_masked'} "
            f"--output_dir={out} "
            f"--diffrender_dir={diffrender_dir} "
            f"--inference_res={diffrender_inference_res} "
            f"--inference_n_steps={diffrender_inference_n_steps}"
            + fallback_flag
            + hf_flag
            + meta_json_flag
        )
        out.mkdir(parents=True, exist_ok=True)
        run(cmd, REPO_ROOT, sv / "decompose_log.txt")
        mark_done(out)
        return

    if not all([args.decomp_config, args.decomp_script, args.decomp_weights]):
        log.warning(
            "Step 5 (decompose): neither --diffrender_dir nor "
            "--decomp_config / --decomp_script / --decomp_weights are set.\n"
            "\n"
            "  Option A – use diffusion-renderer (recommended):\n"
            "\n"
            "    --diffrender_dir /path/to/diffusion-renderer\n"
            "    --diffrender_env diff-render          # conda env name\n"
            "    --hf_home /path/to/hf_cache           # optional, avoids filling ~/\n"
            "\n"
            "  Option B – manual accelerate launch:\n"
            "\n"
            "    accelerate launch --config_file DECOMP_CONFIG DECOMP_SCRIPT \\\n"
            "      --inference_model_weights DECOMP_WEIGHTS \\\n"
            "      --inference_input_dir %s/gen_view_masked \\\n"
            "      --inference_save_dir %s/gen_view_decomposite \\\n"
            "      --inference_res 512 512 --inference_n_repeat 8 \\\n"
            "      --model_passes basecolor metallic roughness normal "
            "diffuse_albedo depth --seed=0\n"
            "\n"
            "  Then touch %s/DONE and continue with --steps 6 7 8",
            sv, sv, out,
        )
        sys.exit(0)

    cmd = (
        f"accelerate launch --config_file {args.decomp_config} "
        f"{args.decomp_script} "
        f"--inference_model_weights={args.decomp_weights} "
        f"--inference_input_dir={sv / 'gen_view_masked'} "
        f"--inference_save_dir={sv / 'gen_view_decomposite'} "
        f"--inference_res 512 512 "
        f"--inference_n_repeat 8 "
        f"--model_passes basecolor metallic roughness normal diffuse_albedo depth "
        f"--seed=0"
    )
    out.mkdir(parents=True, exist_ok=True)
    run(cmd, REPO_ROOT, sv / "decompose_log.txt")
    mark_done(out)


def step_superresolution(args, paths: dict) -> None:
    """Step 6 – upsample decomposed images with InvSR."""
    sv = paths["single_view_dir"]
    out = sv / "gen_view_super"
    if is_done(out):
        log.info("Step 6 (superresolution): already done, skipping.")
        return

    if not args.invsr_dir:
        log.warning(
            "Step 6 (superresolution): --invsr_dir is not set.\n"
            "\n"
            "  Please run InvSR manually:\n"
            "\n"
            "    python3 <INVSR_DIR>/inference_invsr.py \\\n"
            "      -i %s/gen_view_decomposite \\\n"
            "      -o %s/gen_view_super --num_steps 1\n"
            "\n"
            "  Then touch %s/DONE and continue with --steps 7 8",
            sv, sv, out,
        )
        sys.exit(0)

    invsr_dir = Path(args.invsr_dir)
    cmd = (
        f"conda run -n {args.invsr_env} python {invsr_dir / 'inference_invsr.py'} "
        f"-i {sv / 'gen_view_decomposite'} "
        f"-o {sv / 'gen_view_super'} "
        f"--num_steps 1"
    )
    out.mkdir(parents=True, exist_ok=True)
    run(cmd, invsr_dir, sv / "superresolution_log.txt")
    mark_done(out)


def step_backproject(args, paths: dict) -> None:
    """Step 7 – backproject super-resolved basecolor onto the mesh UV atlas."""
    sv = paths["single_view_dir"]
    out = sv / "textures_sr"
    if is_done(out):
        log.info("Step 7 (backproject): already done, skipping.")
        return

    meta_json_flag = ""
    if not args.no_scoring:
        meta_json_flag = f" --meta_json={sv / 'meta.json'}"
    orientation_flag = ""
    orientation_json = sv.parent / "orientation.json"
    if orientation_json.is_file():
        orientation_flag = f" --orientation_json={orientation_json}"
    cmd = python_cmd(
        args.gloss_env,
        STEPS_DIR / "postprocess_backproject.py",
        (
            f"--mesh={paths['mesh_fp']} "
            f"--texture_height={args.texture_size} "
            f"--texture_width={args.texture_size} "
            f"--output_dir={sv} "
            f"--start_view={args.start_view} "
            f"--end_view={args.end_view}"
            + meta_json_flag
            + orientation_flag
        ),
    )
    out.mkdir(parents=True, exist_ok=True)
    run(cmd, REPO_ROOT, sv / "backproject_log.txt")
    mark_done(out)


def step_datagen(args, paths: dict) -> None:
    """Step 8 – render multi-view training samples and pack into WebDataset shards."""
    dataset_tag = args.dataset_tag
    out = paths["expr_dir"] / "multi_view" / dataset_tag
    if is_done(out):
        log.info("Step 8 (datagen): already done, skipping.")
        return

    # datagen.py expects paths relative to --global_root_dir (= data_dir).
    # NOTE: orientation.json is intentionally NOT forwarded — local patches
    # are sampled in mesh-local space and the single-view texture is in UV
    # space, so the local-rendering frame is independent of orientation.
    rel = Path(args.expr_tag) / args.mesh_name
    mesh_rel = relative_to_data_root(paths["mesh_fp"], paths["data_dir"])
    meta_json_flag = ""
    if not args.no_scoring:
        meta_json_flag = f" --meta_json={paths['single_view_dir'] / 'meta.json'}"
    cmd = python_cmd(
        args.gloss_env,
        STEPS_DIR / "datagen.py",
        (
            f"--global_root_dir={paths['data_dir']} "
            f"--start_view={args.start_view} "
            f"--end_view={args.end_view} "
            f"--data.mesh={mesh_rel} "
            f"--data.num_views=100 "
            f"--data.num_local_views=500 "
            f"--data.single_view_dir={rel}/single_view/{args.sv_subdir}/gen_view_super "
            f"--data.single_view_texture_dir={rel}/single_view/{args.sv_subdir}/textures_sr "
            f"--data.data_base_dir={rel}/multi_view/{dataset_tag} "
            f"--data.sampling_dir={rel}/single_view/{args.sv_subdir}/sampling "
            f"--data.seed=0 "
            f"--data.fov_min={args.fov_min} "
            f"--data.fov_max={args.fov_max} "
            f"--data.camera_dist={args.camera_dist}"
            + meta_json_flag
        ),
    )
    out.mkdir(parents=True, exist_ok=True)
    run(cmd, REPO_ROOT, out / "datagen_log.txt")
    mark_done(out)


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

def step_sd15_views(args, paths: dict) -> None:
    """Step 3 (SD1.5 path, DEFAULT) – generate views with SD1.5 + per-modality
    ControlNets. Default config ``sd15_rv_normal_depth_canny`` uses
    Realistic Vision V5.1 + normal/depth/canny at strengths 1.0/0.7/0.5.
    """
    out = paths["single_view_dir"] / "gen_view"
    if is_done(out):
        log.info("Step 3 (sd15_views): already done, skipping.")
        return

    extra = ""
    if args.hf_home:
        extra += f" --hf_home {args.hf_home}"
    if args.sd15_seed is not None:
        extra += f" --seed {args.sd15_seed}"
    if args.sd15_resolution is not None:
        extra += f" --output_size {args.sd15_resolution}"
    if args.sd15_steps is not None:
        extra += f" --num_inference_steps {args.sd15_steps}"
    if args.sd15_guidance_scale is not None:
        extra += f" --guidance_scale {args.sd15_guidance_scale}"

    cmd = python_cmd(
        args.gloss_env,
        STEPS_DIR / "generate_views_sd15.py",
        (
            f"--mesh_name {args.mesh_name} "
            f"--exp_dir {paths['expr_root']} "
            f"--sv_subdir {args.sv_subdir} "
            f"--config {args.sd15_config} "
            f"--configs_file {args.sd15_configs_file} "
            f"--normal_weight {args.sd15_normal_weight} "
            f"--depth_weight {args.sd15_depth_weight} "
            f"--canny_weight {args.sd15_canny_weight}"
            + extra
        ),
    )
    out.mkdir(parents=True, exist_ok=True)
    run(cmd, REPO_ROOT, out / "log.txt")
    mark_done(out)


def step_generate_views(args, paths: dict) -> None:
    """Step 3 dispatcher – default = SD1.5 + Realistic Vision; opt in to
    SDXL Union via --use_sdxl, or to the legacy ComfyUI flow via --use_comfyui."""
    if getattr(args, "use_sdxl", False):
        step_sdxl_views(args, paths)
    elif getattr(args, "use_comfyui", False):
        step_comfyui_views(args, paths)
    else:
        step_sd15_views(args, paths)


STEP_RUNNERS = {
    1: step_generate_condition,
    2: step_generate_view_prompts,
    3: step_generate_views,
    4: step_postprocess_mask,
    5: step_decompose,
    6: step_superresolution,
    7: step_backproject,
    8: step_datagen,
}


# ---------------------------------------------------------------------------
# Force-re-run helper
# ---------------------------------------------------------------------------

def _clear_done(step_num: int, paths: dict, dataset_tag: Optional[str] = None) -> None:
    """Remove the DONE marker for *step_num* so it will be re-executed."""
    sv = paths["single_view_dir"]
    step_done_dirs = {
        1: sv / "condition_output",
        2: paths["prompts_dir"],
        3: sv / "gen_view",
        4: sv / "gen_view_masked",
        5: sv / "gen_view_decomposite",
        6: sv / "gen_view_super",
        7: sv / "textures_sr",
        8: (
            paths["expr_dir"] / "multi_view" / dataset_tag
            if dataset_tag is not None
            else None
        ),
    }
    d = step_done_dirs.get(step_num)
    if d is None:
        return
    marker = done_marker(d)
    if marker.exists():
        marker.unlink()
        log.info("Cleared DONE marker: %s", marker)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end data generation pipeline for material super-resolution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- required ---
    ap.add_argument("--data_dir", required=True, type=Path,
                    help="Root data directory (e.g. /data/material-superres)")
    ap.add_argument("--expr_tag", required=True,
                    help="Experiment tag (sub-folder under data_dir)")
    ap.add_argument("--mesh_name", required=True,
                    help="Mesh name (sub-folder under expr_tag)")

    # --- step selection ---
    ap.add_argument("--steps", nargs="+", type=int, default=ALL_STEPS, metavar="N",
                    help="Steps to run (1-8).  Defaults to all.")
    ap.add_argument("--force", action="store_true",
                    help="Remove DONE markers and re-run the selected steps.")

    # --- step 1: generate_condition ---
    ap.add_argument("--num_prompts", type=int, default=500,
                    help="Number of views (and per-view prompts) to generate.")
    ap.add_argument("--condition_mode", default="generate", choices=["generate", "debug"],
                    help="Pass 'debug' to generate only 10 conditioning views.")

    # --- mesh location override ---
    ap.add_argument("--mesh_subpath", default="mesh/scene.gltf",
                    help="Path to the mesh file relative to the mesh_name directory "
                         "(default: mesh/scene.gltf; use scene.gltf if there is no mesh/ subdir)")
    ap.add_argument("--sv_subdir", default="civitai2.0",
                    help="Sub-folder under <mesh>/single_view/ that all stage "
                         "outputs (condition_output, gen_view, gen_view_*, "
                         "textures_sr, sampling, meta, ...) are read from / "
                         "written to. Use 'origin' or 'anchors' for runs "
                         "driven by orientation.json (see "
                         "scripts/datagen/utils/run_condition_orientation.py).")

    # --- step 2: generate_view_prompts ---
    ap.add_argument("--mesh_subject", default=None,
                    help="Human-readable subject for prompt generation "
                         "(defaults to mesh_name).")
    ap.add_argument("--vlm_model", default="gpt-5",
                    help="Vision-capable OpenAI model used to caption the "
                         "rendered surface-normal map per view in step 2.")
    ap.add_argument("--text_model", default="gpt-5",
                    help="Text OpenAI model for the multi-turn batched "
                         "appearance-prompt generation in step 2.")
    ap.add_argument("--prompt_workers", type=int, default=8,
                    help="Parallel VLM caption workers for step 2.")

    # --- step 3: view generation back-end ---
    # Default = SD1.5 + Realistic Vision (sd15_rv_normal_depth_canny).
    # Opt-in alternatives:
    ap.add_argument("--use_sdxl", action="store_true",
                    help="Use SDXL + ControlNet Union for step 3 (overrides default SD1.5 RV)")
    ap.add_argument("--use_comfyui", action="store_true",
                    help="Use the legacy ComfyUI back-end for step 3 "
                         "(overrides default SD1.5 RV; requires --comfyui_dir)")

    # --- step 3 (SD1.5 RV, default) ---
    ap.add_argument("--sd15_config", default="sd15_rv_normal_depth_canny",
                    help="Entry name in --sd15_configs_file (default: %(default)s).")
    ap.add_argument(
        "--sd15_configs_file",
        default=str(REPO_ROOT / "configs" / "single_view_configs.yaml"),
        help="YAML file with single-view configs (passed to "
             "generate_views_sd15.py).")
    ap.add_argument("--sd15_normal_weight", type=float, default=1.0,
                    help="ControlNet strength for normal (SD1.5).")
    ap.add_argument("--sd15_canny_weight", type=float, default=0.5,
                    help="ControlNet strength for canny (SD1.5).")
    ap.add_argument("--sd15_depth_weight", type=float, default=0.7,
                    help="ControlNet strength for depth (SD1.5).")
    ap.add_argument("--sd15_steps", type=int, default=None,
                    help="Override SD1.5 sampler steps (default: from configs file).")
    ap.add_argument("--sd15_guidance_scale", type=float, default=None,
                    help="Override SD1.5 CFG (default: from configs file).")
    ap.add_argument("--sd15_resolution", type=int, default=None,
                    help="Override SD1.5 output resolution (default: from configs file).")
    ap.add_argument("--sd15_seed", type=int, default=None,
                    help="Fixed seed for SD1.5 (default: random per view).")

    # --- step 3a: SDXL (--use_sdxl) ---
    ap.add_argument("--sdxl_env", default="gloss",
                    help="Conda environment with diffusers/transformers for SDXL step 3")
    ap.add_argument("--sdxl_model", default="stabilityai/stable-diffusion-xl-base-1.0",
                    help="SDXL base model repo ID or local path (step 3 SDXL)")
    ap.add_argument("--controlnet_model", default="xinsir/controlnet-union-sdxl-1.0",
                    help="ControlNet Union model repo ID or local path (step 3 SDXL)")
    ap.add_argument("--sdxl_steps", type=int, default=25,
                    help="Number of denoising steps for SDXL (step 3 SDXL)")
    ap.add_argument("--sdxl_guidance_scale", type=float, default=7.5,
                    help="CFG guidance scale for SDXL (step 3 SDXL)")
    ap.add_argument("--sdxl_seed", type=int, default=None,
                    help="Fixed random seed for all views in SDXL (default: random per view)")
    ap.add_argument("--sdxl_normal_weight", type=float, default=None,
                    help="ControlNet normal map strength (step 3 SDXL); "
                         "defaults to control_strength_normal from meshes.json (fallback 0.5)")
    ap.add_argument("--sdxl_depth_weight", type=float, default=None,
                    help="ControlNet depth map strength (step 3 SDXL); "
                         "defaults to control_strength_depth from meshes.json (fallback 0.5)")
    ap.add_argument("--sdxl_canny_weight", type=float, default=None,
                    help="ControlNet canny edge strength (step 3 SDXL); "
                         "defaults to control_strength_canny from meshes.json (fallback 0.3)")

    # --- step 3b: ComfyUI (default, requires --comfyui_dir) ---
    ap.add_argument("--comfyui_dir", default=None,
                    help="Path to ComfyUI directory (required for step 3 without --use_sdxl)")
    ap.add_argument("--canny_normal", action="store_true",
                    help="Use canny-normal conditioning instead of canny-geonormal (step 3)")
    ap.add_argument("--canny_weight", type=float, default=None,
                    help="ControlNet canny strength for ComfyUI step 3 (default: 0.1 inside generate_views.py)")
    ap.add_argument("--comfyui_visualize", action="store_true",
                    help="Start the ComfyUI HTTP server during step 3 for browser visualisation")
    ap.add_argument("--comfyui_port", type=int, default=8188,
                    help="Port for the ComfyUI visualisation server (default: 8188)")
    ap.add_argument("--comfyui_install_requirements", action="store_true",
                    help="Install ComfyUI requirements before step 3 generation "
                         "(disabled by default because the vendored requirements pin can drift)")

    # --- scoring (runs between step 4 and step 5; ON by default, score-only) ---
    # Scoring writes view_scores + auto_excluded_indices to single_view/meta.json
    # but does NOT auto-apply by default — exclusion is a manual decision made
    # via the browser UIs:
    #   python scripts/datagen/utils/score_viewer.py <single_view_dir>
    #   python scripts/datagen/utils/review_training_views.py <single_view_dir>
    # The latter writes exclude_from_training_indices, which steps 5, 7, and 8
    # honor (step 6 InvSR cascades automatically). Pass --scoring_apply to skip
    # the manual review and union the auto-thresholded ids in directly.
    ap.add_argument("--no_scoring", action="store_true",
                    help="Disable the scoring pass after step 4. "
                         "Default: scoring runs (score-only).")
    ap.add_argument("--scoring_apply", action="store_true",
                    help="Union auto-excluded ids into "
                         "exclude_from_training_indices automatically (skip "
                         "manual UI review). Default: score-only — manually "
                         "filter via score_viewer.py / review_training_views.py.")
    ap.add_argument("--scoring_clip_viewpoint", action="store_true",
                    help="Include the CLIP viewpoint-alignment score (off by "
                         "default; aesthetic + clip_prompt are the active "
                         "filters).")
    ap.add_argument("--scoring_normal_agreement", action="store_true",
                    help="Include the Marigold normal-agreement score (off by "
                         "default; expensive and not part of the default "
                         "aesthetic + clip_prompt filter set).")
    ap.add_argument("--scoring_thresh_clip_prompt", type=float, default=0.20,
                    help="Below this CLIP-prompt score, the view is auto-flagged "
                         "in auto_excluded_indices (whether it actually excludes "
                         "depends on --scoring_apply / manual UI review).")
    ap.add_argument("--scoring_thresh_aesthetic", type=float, default=3.5,
                    help="Below this LAION aesthetic score, the view is auto-flagged.")

    # --- step 5: decomposition (diffusion-renderer, preferred) ---
    ap.add_argument("--diffrender_dir", default=None,
                    help="Path to the diffusion-renderer repo root (enables diffusion-renderer "
                         "for step 5; see thirdparty/diffusion-renderer)")
    ap.add_argument("--diffrender_env", default="diff-render",
                    help="Conda environment with diffusion-renderer installed (default: diff-render)")
    ap.add_argument("--hf_home", default=None,
                    help="Override HF_HOME for the HuggingFace cache used in step 5 "
                         "(recommended: set to a path with plenty of disk space)")
    ap.add_argument("--diffrender_inference_res", default="512,512",
                    help="Primary Height,Width inference resolution for diffusion-renderer in step 5")
    ap.add_argument("--diffrender_fallback_inference_res", nargs="*",
                    default=["384,384", "256,256"], metavar="H,W",
                    help="Lower Height,Width values to retry on CUDA OOM during step 5")
    ap.add_argument("--diffrender_inference_n_steps", type=int, default=20,
                    help="Number of diffusion-renderer denoising steps for step 5")

    # --- step 5: decomposition (legacy accelerate-based fallback) ---
    ap.add_argument("--decomp_config", default=None,
                    help="accelerate config file for intrinsic decomposition (step 5, legacy)")
    ap.add_argument("--decomp_script", default=None,
                    help="Python inference script for intrinsic decomposition (step 5, legacy)")
    ap.add_argument("--decomp_weights", default=None,
                    help="Model weights for intrinsic decomposition (step 5, legacy)")

    # --- step 6: super-resolution ---
    ap.add_argument("--invsr_dir", default=None,
                    help="Path to InvSR directory (required for step 6)")
    ap.add_argument("--invsr_env", default="invsr",
                    help="Conda environment with InvSR installed (default: invsr)")

    # --- step 7: backproject ---
    ap.add_argument("--texture_size", type=int, default=4096,
                    help="UV texture resolution for backprojection (height = width)")

    # --- step 8: datagen ---
    ap.add_argument("--dataset_tag", default="cam0.25-fov0.4-0.8",
                    help="Sub-folder name for the generated multi-view dataset")
    ap.add_argument("--start_view", type=int, default=0,
                    help="First view index to process in step 8")
    ap.add_argument("--end_view", type=int, default=-1,
                    help="Last view index (exclusive) in step 8; -1 = all views")
    ap.add_argument("--fov_min", type=float, default=0.4,
                    help="Minimum FoV (radians) for multi-view camera sampling")
    ap.add_argument("--fov_max", type=float, default=0.8,
                    help="Maximum FoV (radians) for multi-view camera sampling")
    ap.add_argument("--camera_dist", type=float, default=0.25,
                    help="Camera distance for multi-view sampling")

    # --- environment ---
    ap.add_argument("--gloss_env", default="gloss",
                    help="Name of the conda environment that has gloss installed")

    args = ap.parse_args()

    # --- Apply meshes.json defaults ---
    _mesh_info = load_mesh_info(args.mesh_name)
    if _mesh_info:
        if args.mesh_subject is None:
            args.mesh_subject = _mesh_info.get("prompt_name", args.mesh_name)
        if args.sdxl_normal_weight is None:
            args.sdxl_normal_weight = _mesh_info.get("control_strength_normal", 1.5)
        if args.sdxl_depth_weight is None:
            args.sdxl_depth_weight = _mesh_info.get("control_strength_depth", 1.5)
        if args.sdxl_canny_weight is None:
            args.sdxl_canny_weight = _mesh_info.get("control_strength_canny", 0.3)
        if args.canny_weight is None:
            args.canny_weight = _mesh_info.get("control_strength_canny")
    else:
        log.warning("Mesh '%s' not found in %s; using CLI/default values.", args.mesh_name, MESHES_JSON)
        if args.sdxl_normal_weight is None:
            args.sdxl_normal_weight = 0.5
        if args.sdxl_depth_weight is None:
            args.sdxl_depth_weight = 0.5
        if args.sdxl_canny_weight is None:
            args.sdxl_canny_weight = 0.3

    # Validate steps
    invalid = [s for s in args.steps if s not in STEP_RUNNERS]
    if invalid:
        ap.error(f"Unknown step(s): {invalid}.  Valid range is 1–8.")

    paths = build_paths(args)

    log.info("Pipeline configuration:")
    log.info("  data_dir   : %s", paths["data_dir"])
    log.info("  expr_tag   : %s", args.expr_tag)
    log.info("  mesh_name  : %s", args.mesh_name)
    log.info("  mesh_fp    : %s", paths["mesh_fp"])
    log.info("  steps      : %s", args.steps)

    sorted_steps = sorted(args.steps)
    for step_num in sorted_steps:
        name = STEP_NAMES[step_num]
        log.info("\n===== Step %d: %s =====", step_num, name)
        if args.force:
            _clear_done(step_num, paths, dataset_tag=args.dataset_tag)
        STEP_RUNNERS[step_num](args, paths)

        # Scoring pass fires after step 4 (masked views are available) and
        # before step 5 (so excluded views are skipped during decomposition,
        # super-resolution, backproject, and final dataset packaging). On by
        # default — pass --no_scoring to disable.
        if step_num == 4 and not args.no_scoring:
            log.info("\n===== Scoring masked views =====")
            if args.force:
                marker = done_marker(paths["single_view_dir"] / "scores")
                if marker.exists():
                    marker.unlink()
                    log.info("Cleared DONE marker: %s", marker)
            step_score_views(args, paths)

    log.info("\nAll requested steps completed successfully.")


if __name__ == "__main__":
    main()
