# thirdparty

External projects used by the data-generation pipeline and the metric scripts.
They are git submodules pinned to the upstream commits this code was tested with.

| Submodule | Used by | Local patch |
|---|---|---|
| `ComfyUI` | datagen step 3 (view synthesis) | `patches/ComfyUI.patch` adds `generate_views.py` / `generate_views.sh` and small node and requirement changes |
| `diffusion-renderer` | datagen step 5 (intrinsic decomposition) | none |
| `InvSR` | datagen step 6 (super-resolution) | `patches/InvSR.patch` filters inputs to basecolor images and skips already-written outputs |
| `cmmd-pytorch` | `scripts/evaluate.sh` (CMMD) | `patches/cmmd-pytorch.patch` pins the CLIP revision that ships safetensors weights |

## Setup

```bash
git submodule update --init --recursive
./thirdparty/apply_patches.sh
```

Each submodule has its own environment; see `docs/data-generation-pipeline.md`.
