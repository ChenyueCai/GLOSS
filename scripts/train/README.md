# scripts/train

This folder contains the main training entrypoints for the diffusion models used in the project.

## Main Scripts

- `train_diffusion.py`: training entrypoint for the standard file-based multi-view dataset pipeline.
- `train_diffusion_wds.py`: training entrypoint for the WebDataset-based pipeline.

## What These Scripts Do

Both scripts assemble the same core pieces from `gloss`:

- experiment and trainer configuration from `gloss.config.*`
- dataset loading from `gloss.data.*`
- model and attention setup from `gloss.model.*`
- logging and visualization from `gloss.logging.*` and `gloss.viz.*`

The main difference is the data source:

- `train_diffusion.py` reads directly from the local dataset layout.
- `train_diffusion_wds.py` reads sharded WebDataset tar files and includes mask/style-loss utilities tailored to that workflow.

## Related Scripts

More experimental or older training variants live under `scripts/experimental/`.
