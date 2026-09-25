# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import print_function

import logging
import os

from tornado.wsgi import WSGIContainer
from tornado.web import Application, FallbackHandler

import copy
import json
import logging
import torch, torchvision
import torch.nn.functional as F
import collections
import threading
from tornado.websocket import WebSocketHandler, WebSocketClosedError

import tornado.gen
import tornado.ioloop
from flask import Flask

from concurrent.futures import ThreadPoolExecutor

from tornado.concurrent import run_on_executor

from gloss_interactive.kaolin_tmp_io import *
from gloss_interactive.brush import *
from gloss_interactive.protocol import (
    DTYPE_FLOAT32,
    DTYPE_UINT8,
    MSG_TEXTURE_SYNCED,
    MSG_BRUSH_ICON,
    MSG_BRUSH_STATUS,
    MSG_ERROR,
    MSG_FILL_STATUS,
    MSG_STATUS,
    MSG_TEXTURE_CHUNK,
    STATE_ERROR,
    STATE_PREPARING,
    STATE_READY,
)
from gloss_interactive.utils import reclaim_cuda_memory, backproject, composite_inpaint, select_cameras_for_faces
from gloss_interactive.session import PaintSession, GlossSession
from gloss.utils.paths import get_session_dir

logger = logging.getLogger(__name__)

#: All GPU work runs here instead of on the Tornado IOLoop thread.
#:
#: A single worker keeps CUDA work serialized (two concurrent strokes would
#: fight over VRAM) while leaving the IOLoop free to read, write and answer
#: pings for the whole duration of a stroke or a brush preparation. Before
#: this, brush preparation -- which can load a diffusion checkpoint from NFS
#: and render 50 views -- blocked every other message on the socket.
GPU_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gloss-gpu")

#: Longest edge of a brush thumbnail sent to the client. The full-resolution
#: float32 render was ~12 MB for what the panel draws as a small icon.
BRUSH_ICON_SIZE = 256


class GlobalWsQueues:
    """Thread-safe queue manager for websocket message handling.
    
    Provides write and read queues for buffering messages with proper
    synchronization for concurrent access.
    """
    _singleton = None
    _singleton_lock = threading.Lock()

    def __init__(self):
        self._write_queue = collections.deque()
        self._read_queue = collections.deque()
        self._queue_lock = threading.Lock()

    @staticmethod
    def singleton():
        with GlobalWsQueues._singleton_lock:
            if GlobalWsQueues._singleton is None:
                GlobalWsQueues._singleton = GlobalWsQueues()
        return GlobalWsQueues._singleton

    @staticmethod
    def add_write_task(msgs):
        """Add a task (list of message chunks) to the write queue.
        
        Args:
            msgs: A list of message chunks that belong together as a single task.
                  All chunks must be sent successfully for the task to be complete.
        """
        GlobalWsQueues.singleton()._add_write_task(msgs)

    def _add_write_task(self, msgs):
        with self._queue_lock:
            # Store the entire list of chunks as a single task
            self._write_queue.append(msgs)

    @staticmethod
    def add_read_task(msg):
        """Add a message to the read queue."""
        GlobalWsQueues.singleton()._add_read_task(msg)

    def _add_read_task(self, msg):
        with self._queue_lock:
            self._read_queue.append(msg)

    @staticmethod
    def pop_write_task():
        """Pop and return the next task (list of message chunks), or None if queue is empty."""
        return GlobalWsQueues.singleton()._pop_write_task()

    def _pop_write_task(self):
        with self._queue_lock:
            if self._write_queue:
                return self._write_queue.popleft()
            return None

    @staticmethod
    def pop_read_task():
        """Pop and return the next read message, or None if queue is empty."""
        return GlobalWsQueues.singleton()._pop_read_task()

    def _pop_read_task(self):
        with self._queue_lock:
            if self._read_queue:
                return self._read_queue.popleft()
            return None

    @staticmethod
    def pop_all_write_tasks():
        """Pop and return all tasks currently in the write queue.
        
        Returns:
            A list of tasks, where each task is a list of message chunks.
        """
        return GlobalWsQueues.singleton()._pop_all_write_tasks()

    def _pop_all_write_tasks(self):
        with self._queue_lock:
            tasks = list(self._write_queue)
            self._write_queue.clear()
            return tasks

    @staticmethod
    def requeue_write_task(task):
        """Re-add a task (list of message chunks) to the front of the write queue (for retry on failure)."""
        GlobalWsQueues.singleton()._requeue_write_task(task)

    def _requeue_write_task(self, task):
        with self._queue_lock:
            self._write_queue.appendleft(task)

    @staticmethod
    def requeue_write_tasks(tasks):
        """Re-add multiple tasks to the front of the write queue (for retry on failure)."""
        GlobalWsQueues.singleton()._requeue_write_tasks(tasks)

    def _requeue_write_tasks(self, tasks):
        with self._queue_lock:
            # Add in reverse order so they end up in the original order at the front
            for task in reversed(tasks):
                self._write_queue.appendleft(task)

    @staticmethod
    def is_write_queue_empty():
        """Check if the write queue is empty."""
        return GlobalWsQueues.singleton()._is_write_queue_empty()

    def _is_write_queue_empty(self):
        with self._queue_lock:
            return len(self._write_queue) == 0

    @staticmethod
    def write_queue_size():
        """Return the current size of the write queue."""
        return GlobalWsQueues.singleton()._write_queue_size()

    def _write_queue_size(self):
        with self._queue_lock:
            return len(self._write_queue)


class GLOSSWebSocketHandler(WebSocketHandler):
    """ Handles websocket communication with the client.
    """

    def initialize(self, session:GlossSession, paint_session:PaintSession, brush_library:ReferenceBrushLibrary, debug_dir:str):
        print(f'Initializing')
        """ Takes TBD helper type that can actually run the model."""
        # Note: this is correct, __init__ method should not be written for this
        self.brush_library = brush_library
        self.paint_session = paint_session
        self.gloss_session = session
        self.debug_dir = debug_dir

        # Route per-inference logs into the session export folder so they live
        # alongside meta.json and texture-log/ instead of the process CWD.
        self.brush_library.set_log_root(
            os.path.join(str(self.gloss_session.export_folder), "inference-log")
        )

        # this is for large binary messages
        self.image_meta = {}

    #: Bound so ``run_on_executor`` finds it; shared process-wide.
    executor = GPU_EXECUTOR

    # ------------------------------------------------------------------
    # Tagged replies
    #
    # Every server -> client message carries a ``type`` so the add-on can
    # route it to exactly one consumer. Untagged bare strings used to be the
    # norm, which forced the client to guess and race.
    # ------------------------------------------------------------------

    def _send_json(self, msg_type, data):
        """Send one tagged JSON message, tolerating a closed socket."""
        try:
            self.write_message({"type": msg_type, "data": data}, binary=False)
        except WebSocketClosedError:
            logger.warning('WebSocket closed while sending %s', msg_type)
        except Exception as e:
            logger.error('Failed to send %s: %s', msg_type, e)

    def _send_status(self, message):
        """Send a human-readable progress line."""
        self._send_json(MSG_STATUS, {"message": str(message)})

    def _send_error(self, message, context=""):
        """Tell the client a request failed so it can clear its pending state.

        Without this the add-on has no way to distinguish 'still working' from
        'died', and its panel stays stuck until a timeout fires.
        """
        logger.error('Request %r failed: %s', context, message)
        self._send_json(MSG_ERROR, {"message": str(message), "context": context})

    def _send_brush_status(self, brush_name, state, message=""):
        """Report a brush lifecycle transition."""
        self._send_json(MSG_BRUSH_STATUS, {
            "brush_name": brush_name, "state": state, "message": message,
        })

    def _send_fill_status(self, stage, current=None, total=None):
        """Report fill progress between inference stages."""
        self._send_json(MSG_FILL_STATUS, {
            "stage": stage, "current": current, "total": total,
        })

    @tornado.gen.coroutine
    def open(self):
        """ Open socket connection and send information about available geometry."""
        logger.debug("Socket opened.")
        message = {"type": "modelinfo",
                   "data": "welcome to the world of gloss"}
        yield self._send_queue_messages()  # send any old messages that got left behind

    @tornado.gen.coroutine
    def send_sample_json_message(self):
        message = {"type": "test",
                   "data": {"message": "hello from the other side :)"}}
        self.write_message(message, binary=False)

    @tornado.gen.coroutine
    def on_message(self, message):
        """ Handles new messages on the socket."""
        logger.debug('Received message of type {}'.format(type(message)))  #, message))

        # These are coroutines; they MUST be yielded. Calling them bare left
        # the returned Future unretrieved, so any exception inside a fill or a
        # brush preparation vanished silently and the client hung until its
        # timeout.
        request_type = ""
        try:
            if type(message) == bytes:
                yield self._handle_binary_request(message)
            else:
                try:
                    request_type = json.loads(message).get("type", "")
                except Exception:
                    request_type = ""
                yield self._handle_json_request(message)
        except Exception as e:
            logger.exception('Failed to handle incoming message')
            self._send_error(f'{type(e).__name__}: {e}', context=request_type)
    
    @tornado.gen.coroutine
    def _send_queue_messages(self):
        """Iterate through queued tasks, send all chunks of each task.
        
        A task is only marked as complete if all its chunks are sent successfully.
        If any chunk fails, the task and all remaining tasks are re-queued for retry.
        """
        # Check if websocket is still connected
        if self.ws_connection is None:
            logger.warning('WebSocket connection closed, cannot send queued messages')
            return

        tasks = GlobalWsQueues.pop_all_write_tasks()
        print(f"writing {len(tasks)} chunked messages to client")
        for task_idx, task_chunks in enumerate(tasks):
            task_complete = True
            for chunk in task_chunks:
                # Check connection before each chunk
                if self.ws_connection is None:
                    logger.warning('WebSocket connection closed mid-send')
                    task_complete = False
                    break
                try:
                    yield self.write_message(chunk, binary=True)
                except WebSocketClosedError:
                    logger.warning('WebSocket closed while sending message')
                    task_complete = False
                    break
                except Exception as e:
                    logger.error(f'Failed to send queued message chunk: {e}')
                    task_complete = False
                    break

            if not task_complete:
                # Re-queue this task and all remaining tasks for retry
                GlobalWsQueues.requeue_write_tasks(tasks[task_idx:])
                break
    
    # handle binary request, json request
    @tornado.gen.coroutine
    def _handle_json_request(self, raw_message):
        texture_h, texture_w = 4096, 4096
        
        logger.debug('Decoding json message')
        message = json.loads(raw_message)
        msg_type, data = message.get("type"), message.get("data")
        
        if msg_type == "add_ref_mesh":
            mesh_name = data["mesh_name"]
            self._send_status(f"Loading {mesh_name}...")
            # Loads the mesh and a diffusion checkpoint from NFS; off-IOLoop.
            yield self._load_mesh_model_blocking(mesh_name, True)
            self.gloss_session.add_mesh(mesh_name)
            self.gloss_session.add_inpaint_model(mesh_name)
            self._send_status(f"Loaded ref {mesh_name}.")

        if msg_type == "add_pnt_mesh":
            print("adding pnt mesh")
            paint_mesh_name, mesh_name = data["paint_mesh_name"], data["mesh_name"]
            if not data["high_res"]:
                texture_h, texture_w = 1024, 1024
            self._send_status(f"Loading {mesh_name}...")
            # Loads the mesh; fill cameras are chosen on demand. Off-IOLoop.
            yield self._load_mesh_model_blocking(mesh_name, False)
            # In-memory texture is uniformly [0, 1]: alpha = 0 unpainted, alpha > 0 painted.
            init_texture = torch.zeros((texture_h, texture_w, 4)).cuda()
            paint_mesh_texture_dir = os.path.join(self.gloss_session.export_folder, "texture-log", paint_mesh_name)
            os.makedirs(paint_mesh_texture_dir, exist_ok=True)
            paint_mesh_texture_fp = os.path.join(paint_mesh_texture_dir, "texture.png")
            if os.path.exists(paint_mesh_texture_fp):
                init_texture = kaolin.io.utils.read_image(paint_mesh_texture_fp).cuda()
            else:
                init_texture = torch.zeros((texture_h, texture_w, 4)).cuda()
            paint_mesh = self.paint_session.add_mesh(paint_mesh_name, mesh_name, init_texture, paint_mesh_texture_fp)
            if not os.path.exists(paint_mesh_texture_fp):
                paint_mesh.save_texture()
            self.gloss_session.add_mesh(mesh_name)
            self.gloss_session.add_paint_mesh(paint_mesh)
            self._send_status(f"Loaded paint {mesh_name}.")
        
        if msg_type == "add_brush":
            brush_name = data["brush_name"]
            # Tell the client we started BEFORE the slow part, so its panel can
            # show progress instead of looking frozen.
            self._send_brush_status(brush_name, STATE_PREPARING,
                                    f"preparing {brush_name}")
            # Loading a checkpoint, reading the 4K albedo and rendering 50
            # reference views all happen off the IOLoop now.
            brush = yield self._prepare_brush_blocking(data)
            yield self._send_brush_icon(brush_name, brush)
            self._send_brush_status(brush_name, STATE_READY, f"ready: {brush_name}")
        
        if msg_type == "save_brush":
            # check if brush and brush libraries are all created # TODO:
            brush_name = data["brush_name"]
            self.brush_library.save_brush(brush_name, save_ref_lib=True)
            self.gloss_session.add_brush(self.brush_library.get_brush_by_name(brush_name)) 
                
        
        if msg_type == "fill":
            yield self._fill_face(data)

        if msg_type == "clear_face": #TODO: high res or 4k
            target_faces, mesh_name = data["target_faces"], data["mesh_name"]
            h, w = texture_h, texture_w
            if not data["high_res"]:
                h, w = 1024, 1024
            mesh_texture = self.paint_session.get_texture(mesh_name).cuda()
            mesh = self.brush_library.meshes[self.paint_session.get_mesh(mesh_name)]
    
            face_vertices = kaolin.ops.mesh.index_vertices_by_faces(mesh.vertices.unsqueeze(0), mesh.faces)
            face_uvs = mesh.face_uvs.tile((1, 1, 1, 1))
            face_uvs[..., 1] = 1 - face_uvs[..., 1]
            face_uvs = face_uvs * 2 - 1
            _, tex_face_idx = kaolin.render.mesh.rasterize(
                h, w,
                face_features=face_vertices[..., :2] / 2 + 0.5,
                face_vertices_z=torch.zeros_like(face_vertices[..., -1]),
                face_vertices_image=face_uvs,
            )
            mask = 1 - torch.isin(tex_face_idx.squeeze(0).unsqueeze(-1), torch.tensor(target_faces).cuda()).int()
            texture_cleared = mesh_texture * mask 
            msgs = send_large_image(
                "texture", "backproject", texture_cleared,
                msg_type=MSG_TEXTURE_CHUNK, dtype=DTYPE_UINT8,
                extra={"num_views": 1, "view_id": 0},
            )
            self._enqueue_messages([to_binary(msg) for msg in msgs])
            yield self._send_queue_messages()
            # Address the mesh the request named. Using paint_meshes[0] only
            # worked because there is usually exactly one paint mesh, and
            # silently cleared the wrong one otherwise.
            self.paint_session.update_texture(mesh_name, texture_cleared, save=True)
                
        if msg_type == "clear_all_texture":
            # Match the paint mesh's actual texture resolution (1024² or 4096²)
            # rather than trusting the function-scope default; otherwise a low-res
            # paint mesh gets replaced with a 4K zero texture.
            paint_mesh_name = data.get("paint_mesh_name")
            existing = self.paint_session.get_texture(paint_mesh_name)
            h, w = existing.shape[0], existing.shape[1]
            init_texture = torch.zeros((h, w, 4)).cuda()
            self.paint_session.update_texture(paint_mesh_name, init_texture, save=True)
            # The client zeroes its own copy, so both sides already agree.
            self._send_json(MSG_TEXTURE_SYNCED,
                            {"message": f"cleared {paint_mesh_name}"})
            
                
    @run_on_executor
    def _load_mesh_model_blocking(self, mesh_name, load_inpaint_model):
        """Load a mesh and optionally its inpainting checkpoint, off the IOLoop."""
        return self.brush_library.load_mesh_model(
            mesh_name,
            load_inpaint_model=load_inpaint_model,
        )

    @run_on_executor
    def _prepare_brush_blocking(self, data):
        """Create and prepare a brush. Runs on the GPU executor, not the IOLoop.

        This is the expensive path: it may load a diffusion checkpoint from
        NFS, read the reference view and its 4K albedo, run the image encoder,
        and render 50 sampled reference views.

        Returns:
            ReferenceBrush: the prepared brush.
        """
        brush_name, brush_mesh = data["brush_name"], data["brush_mesh"]
        brush_type, sv_id = data["brush_type"], data["sv_id"]
        cam_dist, cam_fov = data.get("cam_dist"), data.get("cam_fov")

        if brush_name not in self.brush_library.brushes.keys():
            self.brush_library.add_brush(name=brush_name, brush_type=brush_type,
                                         mesh_name=brush_mesh, sv_id=sv_id,
                                         cam_dist=cam_dist, cam_fov=cam_fov)
            brush = self.brush_library.get_brush_by_name(brush_name)
            if isinstance(brush, PreSampledReferenceBrush):
                brush.set_valid_face_ids(data["reference_faces"])
            self.brush_library.prepare_brush(brush_name)
        else:
            brush = self.brush_library.get_brush_by_name(brush_name)
            if not brush.single_view.sv_id == int(sv_id):
                raise ValueError(
                    f"brush {brush_name!r} already exists with a different "
                    f"reference view (sv_id={brush.single_view.sv_id}); "
                    f"choose another brush name"
                )
        return brush

    @tornado.gen.coroutine
    def _send_brush_icon(self, brush_name, brush):
        """Send a small uint8 thumbnail for ``brush``.

        The icon used to go out as a full-resolution float32 render (~12 MB)
        for something the panel draws at thumbnail size. Downsampling to
        ``BRUSH_ICON_SIZE`` and quantizing to uint8 is roughly a 60x cut.
        """
        icon = (brush.reference_library.albedo[0] / 2 + 0.5).clamp(0.0, 1.0)
        icon = F.interpolate(
            icon.permute(2, 0, 1).unsqueeze(0).float(),
            size=(BRUSH_ICON_SIZE, BRUSH_ICON_SIZE),
            mode="area",
        ).squeeze(0).permute(1, 2, 0)
        payload, dtype_tag = encode_pixels(icon, dtype=DTYPE_UINT8)
        message = {
            "type": MSG_BRUSH_ICON,
            "brush_name": brush_name,
            "dtype": dtype_tag,
            "image": payload,
        }
        print(f"sending brush icon for {brush_name} ({payload.nbytes} bytes)")
        # Routed through the retry queue like textures, so a mid-send
        # disconnect re-delivers instead of losing the icon.
        self._enqueue_messages([to_binary(message)])
        yield self._send_queue_messages()

    @tornado.gen.coroutine
    def _fill_face(self, data):
        """Handle one ``fill`` request without blocking the IOLoop.

        The GPU work is dispatched to the executor so the socket keeps
        servicing messages (and pings) for the whole stroke; progress messages
        are streamed back as each camera completes.
        """
        io_loop = tornado.ioloop.IOLoop.current()

        def progress(stage, current=None, total=None):
            # Called from the executor thread; marshal onto the IOLoop.
            io_loop.add_callback(self._send_fill_status, stage, current, total)

        self._send_fill_status(STATE_PREPARING)
        texture = yield self._run_fill_blocking(data, progress)
        if texture is None:
            self._send_fill_status(STATE_READY)
            return
        self._send_fill_status("sending texture")
        yield self._safe_send_texture(texture)
        self._send_fill_status(STATE_READY)

    @run_on_executor
    def _run_fill_blocking(self, data, progress):
        """Run one fill stroke end to end. Executes on the GPU executor.

        Args:
            data: the decoded ``fill`` request payload.
            progress: ``progress(stage, current, total)`` callback, safe to
                call from this thread -- it hops back to the IOLoop.

        Returns:
            torch.Tensor | None: the updated texture, or ``None`` if the
            request selected no faces to paint.
        """
        texture_h, texture_w = 4096, 4096

        target_faces, brush_name, mesh_name = data["target_faces"], data["brush_name"], data["mesh_name"]
        if not data["high_res"]:
            h, w = 1024, 1024
        else:
            h, w = texture_h, texture_w

        max_cameras = 5
        if "max_cameras" in data:
            max_cameras = data["max_cameras"]

        clip_fill_to_faces = True
        if "clip_fill_to_faces" in data:
            clip_fill_to_faces = data["clip_fill_to_faces"]
        print("face clipping value set to ", clip_fill_to_faces)

        do_dilate = False
        if "dilate" in data:
            do_dilate = data["dilate"]

        brush = self.brush_library.get_brush_by_name(brush_name)

        # Fill-face does NOT inherit the brush's per-stroke fov/dist stack — the
        # stack is reserved for actual brush strokes. Default here is a fresh
        # CameraConfig() (fov=0.4, dist=0.75); a per-stroke message override
        # (cam_dist / cam_fov) wins over that default.
        fill_camera_config = CameraConfig()
        if "cam_dist" in data:
            fill_camera_config.dist = data["cam_dist"]
        if "cam_fov" in data:
            fill_camera_config.fov = data["cam_fov"]
        print(f"[_fill_face] brush={brush_name} fill_cam=(fov={fill_camera_config.fov}, "
              f"dist={fill_camera_config.dist}) stroke.cam_dist={data.get('cam_dist')} "
              f"stroke.cam_fov={data.get('cam_fov')}")

        start = time.time()
        mesh_texture = self.paint_session.get_texture(mesh_name)
        mesh = self.brush_library.meshes[self.paint_session.get_mesh(mesh_name)]


        if len(target_faces) == 0:
            print(f'Zero faces sent over; sending current texture')
            return mesh_texture

        progress("selecting cameras")
        # Greedy max-coverage over per-face candidate cameras: take the camera
        # that fills the most target area, subtract what it covered, repeat.
        tar_cameras = select_cameras_for_faces(
            mesh, target_faces, fill_camera_config,
            max_cameras=max_cameras,
            face_weights=self.brush_library.get_face_uv_weights(
                self.paint_session.get_mesh(mesh_name)),
        )
        tar_camera_configs = [fill_camera_config] * len(tar_cameras)
        # Recorded in the per-inference log metadata; there is one selection
        # strategy now, so this is constant.
        cam_source = 'greedy-per-face'

        print(f'Selected {len(tar_cameras)} cameras for {len(target_faces)} target faces')
        print(f"Time for selecting cameras: {time.time() - start}")

        start = time.time()

        debug_imgs = []
        tmp_mesh_texture = mesh_texture
        resize_transform = torchvision.transforms.Resize((tmp_mesh_texture.shape[0], tmp_mesh_texture.shape[1]))

        # Per-stroke client opt-in: when "syncmvd": true is in the message AND
        # there are >=2 cameras, run all cameras through one synchronized
        # inference (latent UV fusion across the high-noise denoising steps),
        # then backproject each output independently. Otherwise fall back to
        # the original per-camera loop ("apply model one by one, to ensure
        # seamless inpainting").
        syncmvd = bool(data.get("syncmvd", False))
        brush.use_syncmvd = syncmvd
        print(f"[_fill_face] syncmvd={syncmvd} ncameras={len(tar_cameras)}")
        
        # Clip the fill to exactly the faces the user selected. This used to be
        # intersected with a hardcoded per-mesh valid-face list (the lizard's),
        # which silently discarded most of every selection on any other mesh and
        # hung the addon for TEXTURE_TIMEOUT_S when the intersection was empty.
        # A real per-mesh valid-face feature needs a config key, not a constant.
        _valid_faces = target_faces

        if syncmvd and len(tar_cameras) > 1:
            progress("syncmvd inference", 1, 1)
            inference_start = time.time()
            outputs, paint_region_img_masks = brush.apply_stroke_to_views(
                mesh, tmp_mesh_texture, tar_cameras,
                camera_configs=tar_camera_configs, cam_source=cam_source)
            print("Time for syncmvd inference: ", time.time() - inference_start)
            for i in range(len(tar_cameras)):
                backproject_start = time.time()
                texture, mask, _ = backproject(
                    mesh, outputs[i:i + 1, ...].permute(0, 2, 3, 1), tar_cameras[i],
                    tar_camera_configs[i],
                    h=h, w=w, valid_faces=_valid_faces if clip_fill_to_faces else None,
                    dilation=do_dilate,
                    paint_region_img_mask=paint_region_img_masks[i],
                )
                if texture is None:
                    print(f"[_fill_face] camera {i} produced no visible faces; skipping")
                    continue
                texture_output = torch.cat([texture, mask], dim=-1).squeeze(0)
                del texture, mask
                print("Time for backprojection: ", time.time() - backproject_start)
                if texture_output.shape[0] != tmp_mesh_texture.shape[0]:
                    texture_output = resize_transform(texture_output.permute(2, 0, 1)).permute(1, 2, 0)
                update_mask = (texture_output[..., 3:] > 0)
                tmp_mesh_texture = ~update_mask * tmp_mesh_texture + update_mask * texture_output
                del texture_output
                reclaim_cuda_memory()
            del outputs
            reclaim_cuda_memory()
        else:
            # Apply model one by one, to ensure seamless inpainting
            for i in range(len(tar_cameras)):
                progress("inference", i + 1, len(tar_cameras))
                inference_start = time.time()
                outputs, paint_region_img_masks = brush.apply_stroke_to_views(
                    mesh, tmp_mesh_texture, [tar_cameras[i]],
                    camera_configs=[tar_camera_configs[i]], cam_source=cam_source)
                print("Time for model inference: ", time.time() - inference_start)

                backproject_start = time.time()
                texture, mask, _ = backproject(mesh, outputs[0:1,...].permute(0, 2, 3, 1), tar_cameras[i],
                                               tar_camera_configs[i],
                                               h=h, w=w, valid_faces=_valid_faces if clip_fill_to_faces else None,
                                               dilation=do_dilate,
                                               paint_region_img_mask=paint_region_img_masks[0])
                if texture is None:
                    print(f"[_fill_face] camera {i} produced no visible faces; skipping")
                    del outputs
                    reclaim_cuda_memory()
                    continue
                texture_output = torch.cat([texture, mask], dim=-1).squeeze(0)
                del texture, mask, outputs
                reclaim_cuda_memory()
                print("Time for backprojection: ", time.time() - backproject_start)
                misc_start = time.time()
                # update the mesh texture, replace only nonzero values
                texture_update = texture_output
                if texture_update.shape[0] != tmp_mesh_texture.shape[0]:
                    texture_update = resize_transform(texture_update.permute(2, 0, 1)).permute(1, 2, 0)
                update_mask = (texture_update[..., 3:] > 0)
                tmp_mesh_texture = ~update_mask * tmp_mesh_texture + update_mask * texture_update
                if self.debug_dir:
                    # Per-camera intermediate texture; only with an explicit debug_dir,
                    # never into the process CWD (which was the repo root).
                    os.makedirs(self.debug_dir, exist_ok=True)
                    torchvision.utils.save_image(tmp_mesh_texture.unsqueeze(0).permute(0, 3, 1, 2),
                                                 os.path.join(self.debug_dir, f"fill_debug_{i}.png"))

                del texture_update
                reclaim_cuda_memory()
                print("Time for misc updates: ", time.time() - misc_start)

            # debug_imgs.append({'res': texture_output.unsqueeze(0).permute(0, 3, 1, 2).cpu(),
            #                    'tmp_tex': tmp_mesh_texture.unsqueeze(0).permute(0, 3, 1, 2).cpu()})
        print(f"Time for filling faces: {time.time() - start}")
        # Persist immediately. The client composites this result locally and
        # pushes its version straight back, which supersedes this write -- but
        # if that push is lost (disconnect, crash) the server still holds a
        # near-current texture instead of silently conditioning the next stroke
        # on a state the user stopped seeing several strokes ago.
        self.paint_session.update_texture(mesh_name, tmp_mesh_texture, save=True)
        return tmp_mesh_texture


        # fn = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # for i in range(len(debug_imgs)):
        #     for k, img in debug_imgs[i].items():
        #         fname = os.path.join(log_dir, f"{fn}_{k}" + ("_%02d.png" % i))
        #         torchvision.utils.save_image(img, fname)

    @tornado.gen.coroutine
    def _safe_send_texture(self, texture_img):
        # # send original size, dilate the texture, dilate the mask by small pix count ?
        msgs = send_large_image(
            "texture-view-0", "backproject", texture_img,
            msg_type=MSG_TEXTURE_CHUNK, dtype=DTYPE_UINT8,
            extra={"num_views": 1, "view_id": 0},
        )
        queued_messages = [to_binary(msg) for msg in msgs]
        total_bytes = sum(len(m) for m in queued_messages)
        print(f"Enqueueing texture: {len(queued_messages)} chunk(s), "
              f"{total_bytes / 1e6:.1f} MB")
        self._enqueue_messages(queued_messages)
        yield self._send_queue_messages()

    def _enqueue_messages(self, msgs):
        GlobalWsQueues.add_write_task(msgs)                
        
    @tornado.gen.coroutine
    def _handle_binary_request(self, raw_message):
        logger.debug('Decoding binary message')
        message = from_binary(raw_message)
        msg_type = message.get("type")

        if msg_type == "image":
            # check if tensor meta has the data
            image_name, task_name, chunk_index, chunk_total = message.get("name"), message.get("task_name"), \
                message.get("chunk_index"), message.get("chunk_total")
            mesh_name = message.get("mesh_name")
            _chunk_index = chunk_index + 1
            print(f"{image_name}: {_chunk_index} / {chunk_total}")
            image = None
            # load image
            _image = message.get("image")
            self.image_meta[chunk_index] = _image
            if len(self.image_meta) == chunk_total:
                print("GATHER IMAGE CHUNK")
                ordered = [self.image_meta[k] for k in sorted(self.image_meta.keys(), key=lambda x: int(x))]
                # Derive dimensions from the destination paint mesh so this path
                # works for any resolution (1024² low-res, 4096² high-res, …).
                # Hardcoding here previously broke the 4K regime: the reshape
                # would fail when the client sent a 4K texture.
                if mesh_name is not None and mesh_name in self.paint_session.paint_mesh_entries:
                    existing = self.paint_session.get_texture(mesh_name)
                    texture_h, texture_w = existing.shape[0], existing.shape[1]
                else:
                    texture_h, texture_w = 4096, 4096
                # The add-on uploads uint8 by default (4x smaller than the
                # old float32 payload); legacy float32 clients still work.
                image = decode_pixels(
                    torch.cat(ordered), message.get("dtype", DTYPE_FLOAT32)
                )
                image = image.reshape((texture_h, texture_w, 4)).cuda()
                image = torch.flip(image, dims=[0])
                self.image_meta = {}
                if task_name == "set_texture":
                    print("SET TEXTURE")
                    self.paint_session.update_texture(mesh_name, image, save=True)
                    self._send_json(
                        MSG_TEXTURE_SYNCED,
                        {"message": f"server texture updated ({mesh_name})",
                         "mesh_name": mesh_name},
                    )
                    torchvision.utils.save_image(
                        image.cpu().permute(2, 0, 1).unsqueeze(0),
                        os.path.join(str(self.gloss_session.export_folder), "set_texture.png"),
                    )
            
    def on_close(self):
        logger.info("Socket closed.")
   
   
def create_server(config_fp,
                  session_name,
                  debug_dir=None,
                  fill_fov=None,
                  fill_dist=None,
                  brush_fov=None,
                  brush_dist=None,
                  brush_fov_dist_stack=None):
    """ Create appropriate inference helper """

    # Flask for HTTP
    # _base_dir = os.path.dirname(__file__)
    # _template_dir = os.path.join(_base_dir, 'templates')
    # _static_dir = os.path.join(_base_dir, 'static')
    app = Flask('web server')
                # template_folder=_template_dir,
                # static_url_path='/static',
                # static_folder=_static_dir)

    brush_library = ReferenceBrushLibrary.from_config(
        config_fp,
        fill_fov=fill_fov,
        fill_dist=fill_dist,
        brush_fov=brush_fov,
        brush_dist=brush_dist,
        brush_fov_dist_stack=brush_fov_dist_stack,
    )
    session_dir = str(get_session_dir())
    os.makedirs(session_dir, exist_ok=True)
    if session_name in os.listdir(session_dir):
        gloss_session = GlossSession.load(session_name, session_dir, brush_library)
    else:
        gloss_session = GlossSession(session_name, session_dir,
                                     brush_rel_base=brush_library.cache_folder)
    paint_session = PaintSession()
    for pm in gloss_session.paint_meshes:
        paint_session.add_paint_mesh(pm)

    @app.route('/')
    def index():
        return 'Hello! This is a websocket server only.'

    # Tornado server to handle websockets
    container = WSGIContainer(app)
    server = Application([
        (r'/websocket', GLOSSWebSocketHandler, dict(session=gloss_session, paint_session=paint_session, brush_library=brush_library, debug_dir=debug_dir)),
        (r'.*', FallbackHandler, dict(fallback=container)),
    ], websocket_max_message_size=40*1024*1024)
    return server

