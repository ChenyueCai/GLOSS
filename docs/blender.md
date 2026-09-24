# Blender Plugin

The add-on ([gloss-blender](https://github.com/ChenyueCai/gloss-blender), shipped as the submodule `blender/gloss-blender`) is the user interface. It sends paint requests over a websocket to the GLOSS backend, which runs the model on a GPU. The two can run on the same machine or on different ones.

```bash
bash scripts/run_backend.sh    # GPU machine
```

Then install the add-on on the Blender machine following its README.

## 1. Start the backend (GPU machine)

```bash
bash scripts/run_backend.sh
PORT=8080 SESSION=my-scene bash scripts/run_backend.sh   # other port or session name
```

The backend serves `ws://localhost:10017/websocket` and reads its folders from `gloss_interactive/asset/example_config.yaml`. Relative input folders in that file (meshes, views, checkpoints, brush presets) are taken under `GLOSS_DATA_DIR`. Everything the backend writes goes into `$GLOSS_DATA_DIR` too, next to the inputs (override with `GLOSS_INTERACTIVE_DIR`): sessions in `sessions/<session>` (per-inference logs in `sessions/<session>/inference-log/`), saved brushes in `brushes/`, reference libraries in `caches/`, and inference logs from before a session is attached in `logs/`. A session is reloaded when you reuse its name, and a saved brush is reloaded by name after a restart.

## 2. Install the add-on (Blender machine)

The add-on is a submodule; fetch it if the clone was not `--recursive`:

```bash
git submodule update --init blender/gloss-blender
```

Then follow the [Installation section of the add-on README](../blender/gloss-blender/README.md#%EF%B8%8F-installation): link the folder into Blender's add-ons directory, install `requirement.txt` into Blender's Python, download the example data into `blender/gloss-blender/data/` with `python data/download.py`, and enable **gloss-blender**.

Shortcut: once the example data is downloaded, `bash scripts/setup_blender.sh` packages the add-on as `build/gloss_blender.zip` and installs it (with its Python dependencies) into the Blender found on `PATH`, or the one given as `BLENDER=/path/to/blender`. The zip's `data/config.yaml` points at the submodule's `data/` folders, and its `server_url` uses `PORT` (default `10017`). If no Blender is found, install the zip from *Edit > Preferences > Add-ons > Install*.

## 3. Connect

1. Open *View3D > Sidebar > Gloss*.
2. The add-on reads its folders and server URL from `blender/gloss-blender/data/config.yaml`, loaded automatically into any scene with no folders set. Its default `server_url` is `ws://localhost:10017/websocket`; if the backend uses another port, change `Server URL` in the panel (or the YAML). Use *Load Config* only to switch to a different config.
3. Press *Reconnect*. The connection status in the panel shows whether the add-on reached the backend.

### Backend on a remote machine

Forward the port from the Blender machine; `run_backend.sh` prints the exact command:

```bash
ssh -N -L 10017:<gpu-node>:10017 <login-host>
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Status stays disconnected | Check the backend log shows it listening, that the port matches the config, and that the tunnel is up. |
| `ModuleNotFoundError: websockets` in Blender | Install `blender/gloss-blender/requirement.txt` with Blender's Python. |
| `requested texture does not exist` | The Blender machine's `blender/gloss-blender/data/texture/<mesh>/` is missing that view; rerun `python data/download.py` in the add-on folder. |

Add-on installation, config, and panel reference: [blender/gloss-blender/README.md](../blender/gloss-blender/README.md). Wire protocol: [blender/gloss-blender/MESSAGING.md](../blender/gloss-blender/MESSAGING.md). Backend internals: [gloss_interactive/README.md](../gloss_interactive/README.md).
