# gloss_interactive

`gloss_interactive` contains the backend used by the interactive painting workflow and the `gloss-blender` addon.

## Main Areas

- `server.py`: Tornado websocket server and request routing.
- `session.py`: in-memory session state for paint and model sessions.
- `brush.py`: reference-brush abstractions and brush-library management.
- `utils.py`: backprojection, camera sampling, and coverage utilities used by the backend.
- `asset/example_config.yaml`: example backend config for mesh, checkpoint, view, and cache paths.

## Major Functions And Classes

### `gloss_interactive.server`

- `create_server(config_fp, session_name, debug_dir=None)`: main backend factory used by `scripts/interactive/run.py`.
- `GlobalWsQueues`: websocket queue manager for chunked outgoing and incoming messages.
- `GLOSSWebSocketHandler`: primary websocket handler that receives addon requests and sends results back to the client.

### `gloss_interactive.session`

- `PaintSession`: tracks user painting state and transient paint buffers.
- `PaintMesh`: wrapper for mesh-specific editable state used during interactive painting.
- `GlossSession`: top-level interactive session object that owns the active mesh, cameras, and completion state.

### `gloss_interactive.brush`

- `register_brush(cls)`: decorator used to register new brush implementations.
- `CameraConfig`: camera configuration shared by brush sampling logic.
- `SingleView`: container for a single conditioning/reference view.
- `References`: helper for packaging reference views and metadata.
- `ReferenceBrush`: base class for reference-driven brush behavior.
- `AutoSampledReferenceBrush`: brush that samples cameras automatically.
- `PreSampledReferenceBrush`: brush that reuses precomputed camera sets.
- `ReferenceBrushLibrary`: registry and loader for available brushes.

### `gloss_interactive.utils`

- `backproject(...)`: project a generated or edited view back into texture space.
- `get_face_uv_pixel_counts(...)`: count UV coverage per face. Used as the area
  weights for camera selection, cached per mesh by
  `ReferenceBrushLibrary.get_face_uv_weights(...)`.
- `camera_from_face_normal(...)`: construct a camera looking down one face's
  normal. The single definition of a per-face camera, shared by brush strokes
  and target-camera selection.
- `build_face_coverage_matrix(...)`: rasterize a list of cameras into a
  `(num_faces, num_cameras)` bool coverage matrix, rejecting bad renders
  (inside-the-mesh, single-triangle, flat-on) so they can never be selected.
- `select_cameras_for_faces(...)`: **the** target-camera selector. Places one
  candidate camera per target face, then greedily takes the camera covering the
  most target area, subtracts that area, and repeats until `max_cameras` or no
  camera adds anything new.
- `get_camera_from_face(...)`: older world-up camera constructor, kept for
  `scripts/debug/camera_debug.py`.

## Entry Point

Start the backend with `python scripts/interactive/run.py`. That script loads `asset/example_config.yaml`, creates the server via `create_server(...)`, and listens for websocket requests from Blender.
