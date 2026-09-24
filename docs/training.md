# Training

Trains the per-mesh texture-completion model on a dataset made by [data generation](data-generation-pipeline.md).

```bash
bash scripts/train.sh data/interactive/generated/croissant
```

It uses every visible GPU, one process each, and starts from the Stable Diffusion 2.1 unCLIP weights.

## Options

Set as environment variables:

| Variable | Default | Effect |
| --- | --- | --- |
| `ITERS` | 80000 | Training iterations. |
| `BATCH_SIZE` | 8 | Per-GPU batch size. Use at least 4. |
| `GPUS` | all visible | Number of GPUs. |
| `EXP_NAME`, `EXP_GROUP` | mesh name, `gloss` | Output folder names. |
| `WANDB_MODE` | `offline` | Set `online` after `wandb login` to stream to Weights & Biases. |
| `TRAIN_OBJECTIVE` | `diffusion` | Or `flow_matching`. |

Every other knob in `scripts/train/launch.sh`, such as checkpoint frequency and camera ranges, can be set the same way.

## Output

```text
data/interactive/experiments/<group>/<name>-<objective>/
  config/        resolved configs and the train/eval split (indices.json)
  checkpts/      periodic and persistent checkpoints
```

## Using the result

```bash
MESH_NAME=my_croissant bash scripts/import_generated.sh data/interactive/generated/croissant
MESH_NAME=my_croissant bash scripts/import_generated.sh data/interactive/generated/croissant data/interactive/ckpts/croissant/model.safetensors
```

This places the generated data and the trained model where the Blender backend, `run_completion.sh`, and `evaluate.sh` look for a mesh:

| Created | From |
| --- | --- |
| `data/interactive/meshes/<name>` | the generated mesh folder (link) |
| `data/interactive/single_views/<name>/viewNNNN.basecolor.png` | `single_view/civitai2.0/gen_view_super/` (links) |
| `data/interactive/textures/<name>/viewNNNN.png` | `single_view/civitai2.0/textures_sr/` (links) |
| `data/interactive/metas/<name>/viewNNNN.yml` | `single_view/civitai2.0/meta/` (links) |
| `data/interactive/ckpts/<name>/model.safetensors` | the newest `chkpt_*` under `experiments/`, or the one you pass, converted by `scripts/export_checkpoint.py` |

Only views with an image, a texture, and a camera are linked, and views excluded during review are skipped. `<name>` defaults to the dataset folder name. Pick a new one with `MESH_NAME` when that name is taken, as it is for the example croissant; the script refuses to overwrite an existing mesh unless `FORCE=1`. To confirm the views line up with their cameras, run `python tests/check_example_data.py --meshes <name>` on a GPU.

## Notes

- `train.sh` wraps `scripts/train/launch.sh`, which calls `scripts/train/train_diffusion_wds.py`.
- Very small datasets stop at an epoch boundary, so a run can go past `ITERS`.
- `scripts/train/train_diffusion.py` trains from the unpacked per-view folders instead of shards.

Code overview: [scripts/train/README.md](../scripts/train/README.md).
