<h1 align="center">GLOSS</h1>

<p align="center"><b>Geometric Local Self-Similarity Learning for Faithful Reference-Guided Texture Fill</b></p>

<p align="center">
  <a href="https://chenyuecai.github.io/gloss-page/"><img alt="Project Page" src="https://img.shields.io/badge/Project_Page-5DCB81?style=for-the-badge&logo=githubpages&logoColor=white"></a>
  <a href="https://arxiv.org/abs/2608.25461"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2608.25461-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white"></a>
  <a href="https://www.youtube.com/watch?v=waepcABSxEo"><img alt="Video" src="https://img.shields.io/badge/Video-FF0000?style=for-the-badge&logo=youtube&logoColor=white"></a>
  <a href="https://github.com/ChenyueCai/gloss-blender"><img alt="Blender Add-on" src="https://img.shields.io/badge/Blender_Add--on-F5792A?style=for-the-badge&logo=blender&logoColor=white"></a>
  <a href="https://asia.siggraph.org/2026/"><img alt="SIGGRAPH Asia 2026" src="https://img.shields.io/badge/SIGGRAPH_Asia-2026-1C221E?style=for-the-badge"></a>
</p>

<p align="center">
  <a href="https://chenyue-cai.com/">Chenyue Cai</a><sup>1*</sup> ·
  Anita Hu<sup>2</sup> ·
  <a href="https://www.cs.toronto.edu/~jlucas/">James Lucas</a><sup>2</sup> ·
  <a href="https://www.cs.princeton.edu/~smr/">Szymon Rusinkiewicz</a><sup>1</sup> ·
  <a href="https://shumash.com/">Masha Shugrina</a><sup>2</sup>
</p>

<p align="center"><sup>1</sup> Princeton University · <sup>2</sup> NVIDIA<br><sub><sup>*</sup> Work done during an internship at NVIDIA</sub></p>

<div align="center">
    <img src="assets/teaser.png" alt="One shape with a library of reference patches, and the same dragon textured two different ways from those references." width="100%">
</div>

---

## 🔥 News

- **[2026-09]** The **training, data-generation, and inference code** is released. 🚀
- **[2026-09]** The [example data](https://huggingface.co/datasets/chenyuec/gloss-example-data) and [per-mesh checkpoints](https://huggingface.co/chenyuec/gloss-checkpoints) are on Hugging Face. 🤗
- **[2026-09]** The [Blender add-on](https://github.com/ChenyueCai/gloss-blender) for interactive texture fill is released. 🎨
- **[2026-07-18]** GLOSS is accepted to SIGGRAPH Asia 2026! 🎉

## 🌐 Project Page

Interactive demos, turntables and full results: **[chenyuecai.github.io/gloss-page](https://chenyuecai.github.io/gloss-page/)**

## 🚀 Getting Started

Complete the texture of a 3D mesh from a single reference view, automatically or interactively in Blender. Every step below is a single script; each links to a page with the full options.

### 1. Environment Setup

```bash
git clone --recursive https://github.com/ChenyueCai/GLOSS.git && cd GLOSS
bash scripts/setup_env.sh             # add --datagen for the data-generation envs
bash scripts/download_example_data.sh # example data and checkpoints into ./data/interactive
```

The example data covers three meshes (croissant, cabbage, dragon_head), each with 50 reference views, their cameras, and 4096 px partial textures, plus a per-mesh checkpoint and a Blender demo session.

Needs Linux, conda, and an NVIDIA GPU with CUDA 11.8 drivers. Details: [docs/setup.md](docs/setup.md).

### 2. Blender Plugin Setup

```bash
bash scripts/run_backend.sh    # on the GPU machine
bash scripts/setup_blender.sh  # on the machine running Blender 4.x
```

The add-on is the [gloss-blender](https://github.com/ChenyueCai/gloss-blender) submodule at `blender/gloss-blender`; `setup_blender.sh` fetches it if you did not clone with `--recursive`.

To try the demo, resume the example session `demo-croissant-4k` (croissant mesh, 4K texture). The download puts it in `data/interactive/sessions/`, with its brush caches in `data/interactive/caches/`:

```bash
SESSION=demo-croissant-4k bash scripts/run_backend.sh                          # GPU machine
ssh -N -L 10017:localhost:10017 <server-name>                                  # Blender machine, only if the backend is remote
```

Details, including remote GPU servers: [docs/blender.md](docs/blender.md).

### 3. Dataset Generation

```bash
OPENAI_API_KEY=<key> bash scripts/generate_data.sh data/interactive/meshes/croissant
```

Details: [docs/data-generation-pipeline.md](docs/data-generation-pipeline.md).

### 4. Model Training

```bash
bash scripts/train.sh data/interactive/generated/croissant
MESH_NAME=my_croissant bash scripts/import_generated.sh data/interactive/generated/croissant   # use it in Blender and for completion
```

Details: [docs/training.md](docs/training.md).

### 5. Application: Automatic Completion

```bash
bash scripts/run_completion.sh croissant 48
bash scripts/evaluate.sh croissant 48   # patch-based LPIPS, FID, CMMD
```

Details: [docs/completion.md](docs/completion.md).

## 📝 Citation

```bibtex
@inproceedings{cai2026gloss,
  title     = {GLOSS: Geometric Local Self-Similarity Learning for
               Faithful Reference-Guided Texture Fill},
  author    = {Cai, Chenyue and Hu, Anita and Lucas, James and
               Rusinkiewicz, Szymon},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers},
  year      = {2026},
  doi       = {10.1145/3829340.3842197}
}
```

## ⚖️ License

Apache 2.0, see [LICENSE.txt](LICENSE.txt). The example meshes carry their own licenses in `data/interactive/meshes/<name>/license.txt`:

| Mesh | Author | License |
| --- | --- | --- |
| croissant | [Jason Dovey](https://sketchfab.com/jasondovey) | Sketchfab Standard, redistributed with the author's permission |
| cabbage | [Meerschaum Digital](https://sketchfab.com/meerschaumdigital) | [CC-BY-4.0](http://creativecommons.org/licenses/by/4.0/) |
| dragon_head | [dgeraci](https://sketchfab.com/dgeraci) | [CC-BY-4.0](http://creativecommons.org/licenses/by/4.0/) |

Many thanks to Jason Dovey for allowing us to share the croissant model.

Open-source components used, modified, or distributed as part of this project:

- [SyncMVD](https://github.com/LIU-Yuxin/SyncMVD/) ([MIT](https://github.com/LIU-Yuxin/SyncMVD/blob/main/LICENSE))
- [diffusers](https://github.com/huggingface/diffusers/) ([Apache 2.0](https://github.com/huggingface/diffusers/blob/main/LICENSE))
- [img2img-turbo](https://github.com/GaParmar/img2img-turbo/) ([MIT](https://github.com/GaParmar/img2img-turbo/blob/main/LICENSE))
- [LaMa](https://github.com/advimman/lama/) ([Apache 2.0](https://github.com/advimman/lama/blob/main/LICENSE))
- [ARF-svox2](https://github.com/Kai-46/ARF-svox2) ([BSD 2-Clause](https://github.com/Kai-46/ARF-svox2/blob/master/LICENSE))
- [deep-learning-v2-pytorch](https://github.com/udacity/deep-learning-v2-pytorch/) ([MIT](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/LICENSE))
- Data generation: [diffusion-renderer](https://github.com/nv-tlabs/diffusion-renderer), [InvSR](https://github.com/zsyOAOA/InvSR), [ComfyUI](https://github.com/comfyanonymous/ComfyUI), [cmmd-pytorch](https://github.com/sayakpaul/cmmd-pytorch). See [thirdparty/README.md](thirdparty/README.md).
