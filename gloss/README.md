# gloss

`gloss` is the core Python package for geometry-aware texture generation, completion, rendering, and training.

## Main Areas

- `config/`: experiment and trainer configuration objects.
- `data/`: dataset loading, rendering dataloaders, and WebDataset helpers.
- `inpaint/`: texture completion models, blending logic, camera helpers, and completion utilities.
- `model/`: diffusion, attention, EMA, and model-loading helpers.
- `utils/`: Kaolin mesh helpers, rendering, backprojection, masking, and assorted math utilities.
- `viz/`: visualization helpers used by training and debugging scripts.

## Major Functions And Classes

### `gloss.inpaint.models`

- `TextureInpaintBaseModel`: shared wrapper for loading and running completion models.
- `TextureInpaintStandardModel`: main texture-completion model used by the completion scripts.
- `TextureInpaintSingleStepModel`: single-step variant for lighter or specialized inpainting flows.
- `TextureInpaintImageDiffusionModel`: image-diffusion-backed completion wrapper.
- `TextureInpaintControlNetModel`: ControlNet-backed completion wrapper.

### `gloss.utils.kaolin_utils`

- `load_mesh(mesh_path, device=...)`: canonical mesh loader used across datagen, completion, and interactive tools.
- `compute_normalization(points, eps=...)` and `normalize_points(...)`: normalize mesh point clouds into a stable coordinate frame.
- `split_mesh_into_part_meshes(mesh, face_part_assignments, renormalize=True)`: split a mesh into smaller part meshes for downstream processing.
- `camera_to_meta(cam)` and `camera_from_meta(meta)`: convert cameras to and from metadata dictionaries saved on disk.

### `gloss.utils.render_fast`

- `custom_mesh_batched_render(...)`: fast batched renderer for textured meshes and auxiliary channels.
- `fast_batched_render(...)`: convenience wrapper for batched rendering with the available backend.
- `mesh_rasterize_interpolate_cuda(...)`: lower-level CUDA rasterization helper used by the fast renderer.

### `gloss.utils.single_view`

- `get_valid_faces(...)`: determine which faces are visible and valid in a rendered view.
- `get_valid_faces_from_texture(mesh, texture, all_filled=True)`: infer valid faces directly from a texture map.
- `backproject_render(...)`: backproject rendered view channels into texture space.
- `setup_mesh_single_view(...)`: assemble the per-view tensors used by single-view training and inference.
- `make_single_view_cam(...)`: sample a randomized single-view camera.
- `SingleViewCameraExtrinsicsSampler`: reusable camera sampler for single-view data generation.

### `gloss.model.attention`

- `replace_attention_processors(module, processor, ...)`: swap custom attention processors into a diffusion model.
- `SamplewiseAttnProcessor2_0`: sample-aware attention processor used by completion and training code.
- `CustomAttnProcessor2_0`: more configurable attention processor for experimental attention control.
- `AttentionGraph`: small container for storing attention-graph relationships.
- `diagnose_attention_nan(model, verbose=True)`: helper for debugging unstable attention layers.

## Where These APIs Show Up

- `scripts/completion/*` mainly builds on `gloss.inpaint.models`, `gloss.utils.single_view`, and `gloss.utils.kaolin_utils`.
- `scripts/datagen/*` leans on `gloss.data.*`, `gloss.utils.render_fast`, and `gloss.utils.single_view`.
- `scripts/train/*` uses `gloss.config.*`, `gloss.data.*`, `gloss.model.*`, and `gloss.viz.*`.
