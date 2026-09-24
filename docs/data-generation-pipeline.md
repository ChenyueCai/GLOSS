# Data Generation

Turns one textured mesh into a training set: rendered conditions, prompted and generated views, material decomposition, super-resolution, backprojected UV textures, and WebDataset shards.

```bash
OPENAI_API_KEY=<key> bash scripts/generate_data.sh <mesh folder with scene.gltf> [pipeline flags]
```

Prerequisites: `bash scripts/setup_env.sh --datagen`, a CUDA GPU, and an OpenAI API key (used in step 2).

## Output

With `GLOSS_DATA_DIR=./data/interactive` (the default) and `data/interactive/meshes/croissant` as input:

```text
data/interactive/generated/
  config.yaml                          view-sampling parameters, created on first run
  croissant/
    mesh -> data/interactive/meshes/croissant        input, linked
    single_view/civitai2.0/
      meta/                            per-view camera + prompt (viewNNNN.yml)
      condition_output/                normal, geonormal, depth, canny, mask renders
      gen_view/  gen_view_masked/      generated views, background removed
      gen_view_decomposite/            basecolor / metallic / roughness / normal / depth
      gen_view_super/                  super-resolved passes
      textures_sr/  sampling/          backprojected UV textures and per-face weights
      meta.json                        view scores and excluded views
    multi_view/cam0.25-fov0.4-0.8-wds/ viewNNNN-K.tar training shards
```

`MESH_NAME` and `EXPR_TAG` change `croissant` and `generated`.

## Stages

| Step | Name | Script | Env |
| --- | --- | --- | --- |
| 1 | generate_condition | `scripts/datagen/steps/generate_condition.py` | gloss |
| 2 | generate_view_prompts | `scripts/datagen/steps/generate_view_prompts.py` (OpenAI vision + text) | gloss |
| 3 | generate_views | `scripts/datagen/steps/generate_views_sd15.py` (SD1.5 Realistic Vision + ControlNets, config in `configs/single_view_configs.yaml`) | gloss |
| 4 | postprocess_mask, then scoring | `scripts/datagen/steps/postprocess_mask.py`, `scripts/datagen/steps/score_views.py` | gloss |
| 5 | decompose | `scripts/datagen/steps/run_diffrender_decompose.py` wrapping diffusion-renderer | diff-render |
| 6 | superresolution | `thirdparty/InvSR/inference_invsr.py` | invsr |
| 7 | backproject | `scripts/datagen/steps/postprocess_backproject.py` | gloss |
| 8 | datagen | `scripts/datagen/steps/datagen.py` | gloss |

`scripts/datagen/pipeline.py` is the orchestrator; `generate_data.sh` fills in the layout, envs, and third-party paths. The pipeline switches envs itself with `conda run`, so one call runs all eight stages.

Step 3 alternatives: `--use_sdxl` (SDXL + ControlNet Union) or `--use_comfyui` (legacy ComfyUI flow; needs `git submodule update --init thirdparty/ComfyUI` and `--comfyui_dir thirdparty/ComfyUI`).

## Reviewing views

After step 4, every view is scored into `meta.json`:

| Score | Measures |
| --- | --- |
| `clip_prompt` | CLIP ViT-L/14 similarity between the view and its prompt. |
| `clip_viewpoint` | CLIP similarity against a caption built from the camera pose (`"<subject>, from above"`), which catches the right subject at the wrong angle. |
| `aesthetic` | LAION aesthetic predictor v2, 0 to 10. |
| `normal_agreement` | Masked cosine similarity between the rendered normals and Marigold's normals predicted from the view. |
 Scores only mark views; nothing is dropped unless you pass `--scoring_apply` or review manually:

```bash
python scripts/datagen/utils/score_viewer.py data/interactive/generated/croissant/single_view/civitai2.0
python scripts/datagen/utils/review_training_views.py data/interactive/generated/croissant/single_view/civitai2.0
```

Steps 5, 7, and 8 skip views listed in `exclude_from_training_indices`. `--no_scoring` disables the scoring pass.

To score, or rescore, without the pipeline:

```bash
python scripts/datagen/steps/score_views.py data/interactive/generated/croissant/single_view/civitai2.0
```

| Flag | Effect |
| --- | --- |
| `--limit N`, `--views 0 1 2` | Score a subset. |
| `--skip-normal` | Skip `normal_agreement` (no Marigold download); `--skip-clip-prompt`, `--skip-clip-viewpoint`, `--skip-aesthetic` likewise. |
| `--apply` | Add views below any threshold to `exclude_from_training_indices`. |
| `--thresh-clip-prompt`, `--thresh-clip-viewpoint`, `--thresh-aesthetic`, `--thresh-normal` | Thresholds for `--apply` (defaults 0.20, 0.18, 3.5, 0.30); tune them from the printed score summary. |
| `--force` | Rescore views that already have scores. |

## Resume and rerun

Each stage writes a `DONE` marker and is skipped when it exists, so rerunning the same command resumes after a failure.

```bash
bash scripts/generate_data.sh data/interactive/meshes/croissant --steps 7 8          # only some stages
bash scripts/generate_data.sh data/interactive/meshes/croissant --steps 5 6 --force  # redo stages
```

## Useful flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--num_prompts` | 500 | Number of views. |
| `--condition_mode debug` | off | Only 10 views, for a quick trial. |
| `--texture_size` | 4096 | UV texture resolution. |
| `--start_view`, `--end_view` | all | Views packed into shards in step 8. |
| `--fov_min`, `--fov_max`, `--camera_dist`, `--dataset_tag` | 0.4, 0.8, 0.25 | Training camera sampling in step 8. |
| `--mesh_subject` | from name | Subject text used in prompts. |
| `--hf_home` | `HF_HOME` | Model cache for step 5. |

All flags: `python scripts/datagen/pipeline.py --help`.

## Utilities

Standalone helpers in `scripts/datagen/utils/`, not run by the pipeline:

| Script | Purpose |
| --- | --- |
| `download_sketchfab.py` | Download Sketchfab meshes listed as `name: url` lines, as in `scripts/datagen/test_mesh.txt`. |
| `render_mesh.py` | Render a mesh for inspection. |
| `preview_views.py` | Quick look at generated views. |
| `score_viewer.py` | Browse view scores grouped into quality buckets, to pick `--apply` thresholds. |
| `review_training_views.py` | Mark bad views by hand in the browser; auto-excluded views start rejected. |
| `check_datagen.py` | Sanity-check generated outputs. |
| `clean_rgb_bg.py` | Clean up RGB view backgrounds. |
| `shard_dataset.py`, `zip_datacache.py` | Repackage a generated dataset for training. |
| `single_view_gen.py` | Generate ControlNet single views at chosen (azimuth, elevation) poses. |
| `run_condition_orientation.py`, `run_anchor_views.py` | Render conditions and generate views around anchors saved in `single_view/orientation.json`. |

Next: [train on the result](training.md).
