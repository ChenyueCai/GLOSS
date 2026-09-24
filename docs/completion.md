# Automatic Completion

Completes a mesh's full UV texture from one reference view and its partial texture.

```bash
bash scripts/run_completion.sh <mesh> <view id>
bash scripts/run_completion.sh croissant 48
```

Result: `expr/completion/<mesh>/view<id>/all-fast_a=<angle>_e=<margin>_c=<cameras>/texture/uv_final.png`, with `completion.mp4` and `turnaround.mp4` previews one folder up.

## Inputs

Read from `GLOSS_DATA_DIR`, in the layout of the example data:

| File | What |
| --- | --- |
| `meshes/<mesh>/scene.gltf` | The mesh. |
| `single_views/<mesh>/view<id>.basecolor.png` | Reference view. |
| `textures/<mesh>/view<id>.png` | Partial UV texture from that view. |
| `ckpts/<mesh>/model.safetensors` | Model trained on that mesh. Downloaded from Hugging Face if missing; a trained `chkpt_80000.ckpt` there also works. |

To complete your own mesh, [generate data](data-generation-pipeline.md) and [train](training.md) first.

## Two stages

1. `scripts/completion/complete_stage_1.py` synchronizes denoising across many cameras (SyncMVD-style) to produce a coarse guidance texture, `syncmvd=0.4_a=<angle>_e=<margin>_c=<cameras>/texture/uv-final.png`.
2. `scripts/completion/complete_stage_2.py` refines that guidance into the final texture, `all-fast_*/texture/uv_final.png`.


## Options

Set as environment variables:

| Variable | Default | Effect |
| --- | --- | --- |
| `NUM_CAMERAS` | 600 | Stage 1 cameras. Fewer is faster and coarser. |
| `MAX_NUM_CAMERA` | 350 | Stage 2 camera budget. |
| `MAX_ANGLE` | per-mesh, else 90 | Largest view angle, in degrees, at which the reference is backprojected. |
| `VIEW_MARGIN` | per-mesh, else 40 | Pixels eroded from the edge of the reference view's mask before backprojection. |
| `SEED` | 0 | Stage 2 seed. |
| `EXPR_DIR` | `expr/completion` | Output root. |

For more control, call the two Python scripts directly; `--help` lists every flag, including `--ckpt_path`, `--mesh_path`, and `--guidance_texture_path`.

## Evaluation

```bash
bash scripts/evaluate.sh croissant 48          # one view
bash scripts/evaluate.sh croissant 48 53 60    # several views
```

Each completed texture is compared with the reference texture of the same view on random close-up patches: 256×256 renders of the mesh at distance `CAMERA_DIST` (0.25) with a field of view between 0.4 and 0.8 rad, `NUM_PATCHES` (100) per view.

| Metric | Patches compared | Measures |
| --- | --- | --- |
| LPIPS | Completion vs. reference, same cameras, on faces the reference view covers | Whether the known region is preserved. |
| FID | Reference patches on covered faces vs. completion patches on faces the reference view does not see | Whether the invented region looks like the real material. |
| CMMD | Same two patch sets as FID, compared with CLIP embeddings | Same question as FID, more stable with few patches. |

Lower is better for all three. FID and CMMD pool the patches of all listed views.

Output under `expr/completion/<mesh>/metrics/`:

- `summary.json` with the three scores.
- `lpips-fid.csv` with per-view LPIPS and FID.
- `cmmd/gt`, `cmmd/pred` with the rendered patches.
- `lpips-fid/log-*.png` with one example patch pair per metric.

The scripts behind it are `scripts/metrics/test_completion_metrics.py` (LPIPS, FID), `scripts/metrics/gen_cmmd_patches.py` with `thirdparty/cmmd-pytorch` (CMMD), and `scripts/metrics/test_completion_dreamsim.py` for DreamSim.
