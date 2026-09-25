# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Dict, Callable, Any
from dataclasses import dataclass
import yaml
import os
import kaolin
from kornia.morphology import erosion
import torch, torchvision
import math
import logging
import time
from datetime import datetime

from torchvision.transforms.functional import resize

from gloss.model.attention import SamplewiseAttnProcessor2_0
from gloss.utils import reclaim_cuda_memory
from gloss.utils.render_fast import custom_mesh_batched_render
from gloss.utils.kaolin_utils import load_mesh, camera_from_meta
from gloss.data.render_dataloader import LocalCameraExtrinsicsSampler
from gloss.utils.single_view import get_valid_faces_from_texture, backproject_render, get_tex_face_idx
from gloss.inpaint.inpaint_utils import dilate_nonblack_pool
from gloss.utils.nnfm_loss import NNFMLoss, nn_feat_replace, cos_loss
from gloss.utils.render import make_camera_from_extr_intr
from gloss.inpaint.models import *
from gloss.logging import log_tensor, log_tensor_dict, default_log_setup

from gloss_interactive.utils import camera_from_face_normal, get_face_uv_pixel_counts, save_concat_log, texel_scale
from gloss_interactive.syncmvd import run_syncmvd_inference
from gloss_interactive.camera_conversion import dump_cameras_json
from gloss.utils.paths import resolve_config_paths, resolve_checkpoint, relativize, rebase, get_log_dir

logger = logging.getLogger(__name__)
default_log_setup(logging.DEBUG)

BRUSH_REGISTRY = {}

def register_brush(cls):
    BRUSH_REGISTRY[cls.__name__] = cls
    return cls


NNFM_LOSS = None


def default_log_root() -> str:
    """<GLOSS_INTERACTIVE_DIR>/logs, or ./logs when no data dir is configured."""
    try:
        return str(get_log_dir())
    except EnvironmentError:
        return "./logs"


def get_global_nnfm_loss():
    global NNFM_LOSS
    if NNFM_LOSS is None:
        NNFM_LOSS = NNFMLoss(torch.device("cuda"))
    return NNFM_LOSS

@dataclass
class CameraConfig:
    fov: float = 0.4
    resolution: int = 256
    dist: float = 0.75
    spacing: float = 0.25
    backproject_pix_count: int = 2
    backproject_max_angle: int = 60

class SingleView:
    def __init__(
        self,
        mesh_name,
        mesh,
        cam,
        name: str,
        sv_id: int,
        albedo=None,
        albedo_map=None,
        view_folder=None,
        texture_folder=None,
        cam_meta_folder=None):
        self.mesh_name = mesh_name
        self.mesh = mesh
        self.cam = cam
        self.name = name
        self.sv_id = sv_id
        
        # Store folders for future loading
        self.view_folder = view_folder
        self.texture_folder = texture_folder
        self.cam_meta_folder = cam_meta_folder
        
        self.views = {
            'albedo': albedo} if albedo is not None else {}
        self.texturemaps = {
            'albedo': albedo_map} if albedo_map is not None else {}
        self._loaded = albedo is not None  # Track if full data is loaded

        self.reference_lib = None

    @staticmethod
    def from_disk(
        mesh_name,
        mesh,
        sv_name,
        view_folder,
        cam_meta_folder,
        texture_folder,
        sv_id):
        albedo = kaolin.io.utils.read_image(
            os.path.join(
                view_folder,
                sv_name +
                '.basecolor.png'))  # TODO: other channel
        albedo_map = kaolin.io.utils.read_image(
            os.path.join(texture_folder, sv_name + '.png'))
        with open(os.path.join(cam_meta_folder, sv_name + '.yml'), 'r') as file:
            camera_meta =  yaml.safe_load(file)['camera']
        camera = camera_from_meta(camera_meta)
        return SingleView(
            mesh_name, mesh, camera, sv_name, sv_id,
            albedo=albedo,
            albedo_map=albedo_map,
            view_folder=view_folder,
            texture_folder=texture_folder,
            cam_meta_folder=cam_meta_folder
        )

    @staticmethod
    def from_id_only(
        mesh_name,
        mesh,
        sv_name,
        sv_id,
        cam_meta_folder,
        view_folder=None,
        texture_folder=None
    ):
        with open(os.path.join(cam_meta_folder, sv_name + '.yml'), 'r') as f:
            camera_meta = yaml.safe_load(f)['camera']
        camera = camera_from_meta(camera_meta)

        return SingleView(
            mesh_name, mesh, camera, sv_name, sv_id,
            albedo=None,
            albedo_map=None,
            view_folder=view_folder,
            texture_folder=texture_folder,
            cam_meta_folder=cam_meta_folder
        )

    # -------------------------------------------------------
    # Load full data using stored folders
    # -------------------------------------------------------
    def load_full_data(self):
        """Load images using stored folders, no arguments needed."""
        if self._loaded:
            return  # already loaded

        if not self.view_folder or not self.texture_folder:
            raise RuntimeError(
                "SingleView missing folder paths: call from_disk() or provide folders in from_id_only()."
            )

        albedo_fp = os.path.join(self.view_folder, self.name + '.basecolor.png')
        albedo_map_fp = os.path.join(self.texture_folder, self.name + '.png')

        self.views['albedo'] = kaolin.io.utils.read_image(albedo_fp)
        self.texturemaps['albedo'] = kaolin.io.utils.read_image(albedo_map_fp)

        self._loaded = True
    
    def generate_reference(self, mesh=None, num_samples=50, camera_config=CameraConfig(), device=torch.device('cuda'), **kwargs):
        """generate reference from valid faces, auto select from texture
        """
        new_mesh_edits = False
        if new_mesh_edits:
            mesh = mesh
        else:
            mesh = self.mesh
        # New Mesh Edits #
        if new_mesh_edits:
            albedo = mesh.materials[0].diffuse_texture
            ones = torch.ones(
                (*albedo.shape[:2], 1),
                dtype=albedo.dtype,
                device=albedo.device
            )
            albedo = torch.cat([albedo[:, :, :3], ones], dim=-1).to(device)
        else:
            albedo = self.texturemaps['albedo'].to(device)
            permuted_albedo = albedo[None].permute(0, 3, 1, 2)
       
        
        valid_face_ids = kwargs.get('valid_face_ids')
        if valid_face_ids is None:
            # New Mesh Edits #
            if new_mesh_edits:
                valid_face_ids = torch.arange(self.mesh.faces.shape[0])
                valid_face_ids = get_valid_faces_from_texture(
                mesh.to(device), albedo[None].permute(0, 3, 1, 2))
            else:
                print(self.texturemaps['albedo'][None].shape)
                valid_face_ids = get_valid_faces_from_texture(
                    mesh.to(device), permuted_albedo) #self.texturemaps['albedo'][None]
        inpaint_camera_sampler = LocalCameraExtrinsicsSampler(mesh, camera_dist=camera_config.dist)
        sampling_weights = torch.zeros(mesh.faces.shape[0]).to(device)
        sampling_weights[valid_face_ids] = 1.0
        inpaint_camera_sampler.set_sampling_weights(sampling_weights)
        extrinsics = inpaint_camera_sampler.generate(num_samples)
        cameras = []
        for i in range(num_samples):
            camera = make_camera_from_extr_intr(extrinsics[i], camera_config.fov, resolution=camera_config.resolution, 
                                             device=device)
            cameras.append(camera)
        # render patches
        s = albedo.shape[0]
        # s = self.mesh.materials[0].metallic_texture.shape[1]
        
        normal_map = mesh.materials[0].hwc().normals_texture
        if normal_map is not None:
            normal_map = torchvision.transforms.Resize((s, s))(normal_map.permute(2, 0, 1)).permute(1, 2, 0).to(device)
        
        #albedo_map = torchvision.transforms.Resize((s, s))(self.texturemaps['albedo'].permute(2, 0, 1)).permute(1, 2, 0).to(device)
        
        if new_mesh_edits:
            albedo = torchvision.transforms.Resize((s, s))(
                albedo.permute(2, 0, 1)
            ).permute(1, 2, 0)
            ones = torch.ones(
                (*albedo.shape[:2], 1),
                dtype=albedo.dtype,
                device=albedo.device
            )
            albedo = torch.cat([albedo[:, :, :3], ones], dim=-1)
        # New Mesh Edits #
        # expr: METALLIC REF
        #metallic, roughness = self.mesh.materials[0].metallic_texture, self.mesh.materials[0].roughness_texture
        #mr0 = torch.cat([metallic, roughness, torch.zeros_like(metallic), albedo_map[..., 3:4]], dim=-1)
        #alpha = albedo_map[..., 3:4]
        h, w = albedo.shape[0], albedo.shape[1]
        if valid_face_ids is not None:
            tex_face_idx = get_tex_face_idx(mesh.to(device), h, w) 
            valid_face_mask = torch.isin(tex_face_idx, torch.tensor(valid_face_ids).to(device)).float()
            valid_face_mask = dilate_nonblack_pool(valid_face_mask.unsqueeze(0), kernel_size=3, iterations=3).squeeze(0).permute(1, 2, 0)
            render_texture = valid_face_mask * albedo
        else:
            render_texture = albedo #self.texturemaps["albedo"].to(device)
        batch_size = 16
        num_batches = int(math.ceil(num_samples / batch_size))
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_cameras = kaolin.render.camera.Camera.cat([cameras[idx] for idx in range(start_idx, end_idx)])
            r = custom_mesh_batched_render(batch_cameras, mesh.to(device), render_texture, 
                                           normal_map, requires_positions=True, process_as_albedo=False, backend="cuda")
            nnfm_loss_fn = get_global_nnfm_loss() #NNFMLoss(torch.device("cuda"))

            with torch.no_grad():
                feats = nnfm_loss_fn.get_feats(r['camera_normals'].permute(0, 3, 1, 2) / 2.0 + 0.5, [11, 13, 15])
            _reference = References(camera_normals=r['camera_normals'].cpu(),
                                   relative_positions=r['relative_positions'].cpu(),
                                   albedo=r['textured'][..., :3].cpu() * 2 -1, 
                                   albedo_alpha=r['textured'][..., 3:].cpu(),
                                   features=torch.cat(feats, 1).cpu())
            del r, feats
            reclaim_cuda_memory()
            if i == 0:
                reference = _reference
            else:
                reference.concat(_reference)
        return reference


@dataclass
class References:
    camera_normals: Optional[torch.Tensor] = None  # (B, H, W, C) (0, 1)
    relative_positions: Optional[torch.Tensor] = None # (B, H, W, C) (0, 1)
    albedo: Optional[torch.Tensor] = None # (B, H, W, C) (-1, 1)
    albedo_alpha: Optional[torch.Tensor] = None # (B, H, W, C) (0, 1)
    features: Optional[torch.Tensor] = None # vgg feature from camera_normals

    # ---------- SAVE / LOAD ----------
    def save(self, folder: str):
        """Save all non-None tensors to the given folder."""
        os.makedirs(folder, exist_ok=True)
        for name, tensor in self.__dict__.items():
            if tensor is not None:
                path = os.path.join(folder, f"{name}.pt")
                torch.save(tensor, path)

    @classmethod
    def load(cls, folder: str) -> "References":
        """Load all .pt tensors from the given folder into a new References."""
        kwargs = {}
        for name in cls.__annotations__.keys():
            path = os.path.join(folder, f"{name}.pt")
            if os.path.exists(path):
                kwargs[name] = torch.load(path, map_location="cpu")
            else:
                kwargs[name] = None
        return cls(**kwargs)

    # ---------- CONCATENATION ----------
    def concat(self, other: "References") -> "References":
        """Concatenate two References along the batch (0th) dimension."""
        merged = {}
        for name in self.__annotations__.keys():
            a = getattr(self, name)
            b = getattr(other, name)
            if a is not None and b is not None:
                merged[name] = torch.cat([a, b], dim=0)
            elif a is not None:
                merged[name] = a
            else:
                merged[name] = b
        return References(**merged)

    # ---------- UTILS ----------
    def to(self, device: str) -> "References":
        """Move all tensors to a device."""
        for name, tensor in self.__dict__.items():
            if tensor is not None:
                setattr(self, name, tensor.to(device))
        return self
    
    def to_model_data(self) -> list:
        device = torch.device('cuda') # making sure on cuda
        reference_data = []
        for i in range(self.get_num_references()):
            render_channels = {}
            bg_mask = (torch.abs(self.camera_normals[i:i+1].to(device)) < 0.01).all(dim=-1, keepdim=True).float()
            render_channels['albedo'] = self.albedo[i:i+1].to(device) * (1 - bg_mask) + bg_mask * -1
            render_channels['camera_normals'] = self.camera_normals[i:i+1].to(device) * (1 - bg_mask) + bg_mask * -1
            render_channels['relative_positions'] = self.relative_positions[i:i+1].to(device) * (1 - bg_mask) + bg_mask * -1
            kernel = torch.ones(15, 15).cuda()
            render_channels['inpaint_mask'] = erosion(self.albedo_alpha[i:i+1].to(device), kernel).to(device)
            render_channels['albedo_alpha'] = render_channels['inpaint_mask'].to(device)
            render_channels['background_alpha'] = (1.0 - bg_mask) * 2.0 - 1.0
            # add augmentation 
            if i % 2 == 0:
                for k, v in render_channels.items():
                    render_channels[k] = torchvision.transforms.functional.hflip(v.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            reference_data.append(render_channels)
        return reference_data

    def get_num_references(self) -> int:
        """Get the number of reference patches stored."""
        for name, tensor in self.__dict__.items():
            if tensor is not None:
                return tensor.shape[0]
        return 0
    
    def get_references_from_ids(self, ids: torch.Tensor) -> "References":
        """Get a subset of references by their IDs."""
        selected = {}
        for name, tensor in self.__dict__.items():
            if tensor is not None:
                selected[name] = tensor[ids]
            else:
                selected[name] = None
        return References(**selected)
    

@register_brush
class ReferenceBrush:
    def __init__(self, name: str, single_view: SingleView, inpaint_model:Optional[TextureInpaintStandardModel]=None):
        self.name = name
        self.single_view = single_view
        self.references_dir: Optional[str] = None
        self.reference_library_dir: Optional[str] = None
        self.inpaint_model_path: Optional[str] = None
        # Where _model_inference dumps per-call logs. Defaults to
        # <GLOSS_INTERACTIVE_DIR>/logs/<timestamp>; the websocket handler retargets
        # this to <session_export_folder>/inference-log on session init.
        self.log_root = default_log_root()
        
        self.reference_library:  Optional[References] = None
        self.references: Optional[References] = None
        self.inpaint_model: Optional[TextureInpaintStandardModel] = inpaint_model
        self.is_prepared = False
        self.is_set= False
        
        # Stack of CameraConfigs applied per-stroke. camera_config is an alias to
        # camera_configs[0] (the primary entry) so legacy single-camera consumers
        # — resolution, backproject thresholds, generate_reference — still work.
        self.camera_configs: list[CameraConfig] = [CameraConfig()]
        self.camera_config = self.camera_configs[0]
        self.sv_embedding = None
        # When True, multi-camera strokes denoise jointly via SyncMVD multi-view
        # fusion before per-view decode (eliminates seams between overlapping
        # cameras). When False, fall back to the original per-batch
        # ``inpaint_model.inpaint`` call. Toggleable via ``set_syncmvd``.
        self.use_syncmvd: bool = True
        self.syncmvd_steps: int = 20
        self.syncmvd_multiview_end: float = 0.4

    def set_inpaint_model(self, inpaint_model:TextureInpaintStandardModel):
        self.inpaint_model = inpaint_model
        if self.sv_embedding is not None:
            if self.inpaint_model:
                self.inpaint_model.set_embedding("", self.sv_embedding)
        
    def prepare(self, force):
        """load single view in full data, if not loaded, load the single view"""
        # if reference patch fp is not None and the file exists
        if not self.single_view._loaded:
            self.single_view.load_full_data()
        self.single_view.mesh.to(torch.device('cuda'))
        
        # set the inference model view and prompt embeddding here
        albedo_view = self.single_view.views["albedo"].unsqueeze(0)
        self.inpaint_model.set_embedding("", resize(albedo_view[..., :3].permute(0, 3, 1, 2), [512, 512]))
        self.sv_embedding = resize(albedo_view[..., :3].permute(0, 3, 1, 2), [512, 512])
        if self.inpaint_model:
            self.inpaint_model.set_embedding("", self.sv_embedding)
        
        if self.references_dir is not None and os.path.isdir(self.references_dir) and not force:
            self.references = References.load(self.references_dir).to(torch.device('cuda'))
            self.is_set = True
            print(f"load reference dir from {self.references_dir}")
            
        if self.reference_library_dir is not None and os.path.isdir(self.reference_library_dir) and not force:
            self.reference_library = References.load(self.reference_library_dir).to(torch.device('cuda'))
            self.is_prepared = True
            print(f"load reference lib dir from {self.reference_library_dir}")

    def to_config(self, rel_base=None):
        """Brush config for a saved file. Reference dirs are written relative to
        ``rel_base`` (the library's cache folder) so saved brushes and sessions
        stay valid when the data bundle moves."""
        config = {
            'name': self.name,
            'mesh_name': self.single_view.mesh_name,
            'brush_type': type(self).__name__,
            "sv_name": self.single_view.name,
            'sv_id': self.single_view.sv_id,
            'reference_library_dir': relativize(self.reference_library_dir, rel_base),
            'references_dir': relativize(self.references_dir, rel_base),
            # cam_dist / cam_fov kept for legacy readers; cam_fov_dist_stack is
            # the authoritative per-stroke camera stack.
            'cam_dist': float(self.camera_config.dist),
            'cam_fov': float(self.camera_config.fov),
            # lists, not tuples: yaml.dump writes tuples as !!python/tuple, which
            # yaml.safe_load (used to read saved brushes back) rejects.
            'cam_fov_dist_stack': [[float(c.fov), float(c.dist)] for c in self.camera_configs],
            }
        return config
    
    @staticmethod
    def from_config(config: dict, single_view_lookup:Callable[..., Any], inpaint_model:Optional[TextureInpaintStandardModel]=None,
                    rel_base=None) -> "ReferenceBrush":
        
        brush_type = config.get("brush_type")
        if brush_type is None:
            raise ValueError("Config missing 'brush_type' key.")

        # Look up class in registry
        if brush_type not in BRUSH_REGISTRY:
            raise ValueError(f"Unknown brush type: {brush_type}")
        BrushClass = BRUSH_REGISTRY[brush_type]

        # Get SingleView instance
        sv_id = config.get("sv_id")
        mesh_name = config.get("mesh_name")
        sv = single_view_lookup(mesh_name, sv_id)

        # Instantiate brush
        brush = BrushClass(config["name"], sv, inpaint_model)

        # Set optional directories if present
        # Relative dirs resolve against the cache folder; absolute ones (older files) pass through.
        brush.references_dir = rebase(config.get("references_dir"), rel_base)
        brush.reference_library_dir = rebase(config.get("reference_library_dir"), rel_base)
        # Prefer the new stack form when present; otherwise fall back to the
        # legacy single (cam_fov, cam_dist) pair so old saved brushes load fine.
        stack = config.get("cam_fov_dist_stack")
        if stack:
            brush.camera_configs = []
            for fov, dist in stack:
                cfg = CameraConfig()
                cfg.fov = float(fov)
                cfg.dist = float(dist)
                brush.camera_configs.append(cfg)
            brush.camera_config = brush.camera_configs[0]
        else:
            if config.get("cam_dist") is not None:
                brush.camera_config.dist = float(config["cam_dist"])
            if config.get("cam_fov") is not None:
                brush.camera_config.fov = float(config["cam_fov"])
        return brush
    
    def apply_stroke_to_views(self, mesh, mesh_texture, cameras:list[kaolin.render.camera.Camera],
                              target_erode_kernel:int=40,
                              camera_configs:Optional[list]=None,
                              cam_source:Optional[str]=None)->tuple:
        """Returns (output_patches, paint_region_img_masks).

        paint_region_img_masks is a list of (1, H, W, 1) tensors in [0, 1]; 1 marks
        the pixels outside the eroded known mask AND inside the rendered foreground
        — i.e. the pixels the model actually repainted (eroded boundary ring +
        previously-unknown texels). Callers pass this through to ``backproject`` to
        avoid rewriting the known interior of the texture.
        """
        # Create the per-inference log dir up front (one per call to this fn,
        # not per-`_model_inference`) so SyncMVD and per-camera paths share
        # the same dump format. cameras.json round-trips through
        # ``intrinsics_from_meta`` / ``extrinsics_from_meta``.
        log_dir = os.path.join(self.log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))
        os.makedirs(log_dir, exist_ok=True)
        try:
            dump_cameras_json(
                os.path.join(log_dir, "cameras.json"),
                cameras,
                camera_configs=camera_configs,
                extra={
                    "mesh_name": getattr(self.single_view, "mesh_name", None),
                    "brush_name": self.name,
                    "use_syncmvd": bool(self.use_syncmvd and len(cameras) > 1),
                    "cam_source": cam_source,
                },
            )
        except Exception as e:
            logger.warning(f"failed to dump cameras.json to {log_dir}: {e}")

        render_start = time.time()
        targets_data = self._render_views_to_data(mesh, mesh_texture, cameras,
                                                  target_erode_kernel=target_erode_kernel)
        print("Time for rendering views: ", time.time() - render_start)
        ref_start = time.time()
        references = self._get_references(targets_data, color_reference=True)
        references_data = references.to_model_data()
        print("Time for getting references: ", time.time() - ref_start)
        infer_start = time.time()
        if self.use_syncmvd and len(cameras) > 1:
            # SyncMVD: shared latent UV synchronizes the *stroke* cameras
            # (`cameras`) across the high-noise denoising steps so overlapping
            # views agree on shared surface regions. Reference patches are
            # passed through `inpaint_data` for cross-attention conditioning
            # only — they do NOT participate in UV fusion (their batch slots
            # get bg_latent and they're skipped in the backprojection loop).
            output_patches = run_syncmvd_inference(
                self.inpaint_model, mesh,
                target_cameras=list(cameras),
                inpaint_data=references_data + targets_data,
                num_refs=len(references_data),
                camera_config=self.camera_config,
                num_inference_steps=self.syncmvd_steps,
                multiview_diffusion_end=self.syncmvd_multiview_end,
            )
        else:
            output_patches = self._model_inference(references_data + targets_data, log_dir=log_dir)
        print("Time for model inference: ", time.time() - infer_start)
        # background_alpha is in {-1, +1}; >0 marks foreground.
        # albedo_alpha is the eroded known mask in [0, 1] (matches the in-memory
        # texture convention: alpha=0 unpainted, alpha>0 painted); >0 marks "keep".
        # Paint region = foreground AND NOT eroded-known.
        paint_region_img_masks = []
        for t in targets_data:
            fg = (t['background_alpha'] > 0).float()
            keep = (t['albedo_alpha'] > 0).float()
            paint_region_img_masks.append(fg * (1.0 - keep))
        return output_patches[-len(cameras):], paint_region_img_masks
    
    def backproject_to_texture(self, camera, view, mesh, valid_faces=[], h=1024, w=1024) -> torch.Tensor:
        # 0.07s
        device = self.single_view.mesh.faces.device
        pad = 1
        res = self.camera_config.resolution
        update_mask = torch.ones((1, res, res, 1), device=device, dtype=torch.float32) * -1
        update_mask[0, pad:res - pad, pad:res - pad] = 1
        lighting = kaolin.render.easy_render.default_lighting().cuda()
        render_res = gloss.utils.render.render_all_features(camera, mesh, lighting)
        
        fg_mask = 1 - (torch.abs(render_res["geo_camera_normals"]) < 0.01).all(dim=-1, keepdim=True).float()
        if (fg_mask == 0).sum() > 0:
            kernel = torch.ones(15, 15).cuda()
            fg_mask = erosion(fg_mask.permute(0, 3, 1, 2), kernel).permute(0, 2, 3, 1)
            update_mask = (update_mask == 1) & (fg_mask == 1)
            update_mask = update_mask.float() * 2 - 1
        view = torch.cat([view, update_mask], dim=-1)
        # RGB in [0, 1] (model output) + alpha in {-1, +1} (write/skip mask).
        render_res['albedo'] = view
        backprojection, mask, tex_face_idx = backproject_render(mesh, camera, render_res, ['albedo'],
                                                                h, w, min_pixel_count=0, #self.camera_config.backproject_pix_count,
                                                                max_angle_deviation=math.pi*self.camera_config.backproject_max_angle*2/180.0,
                                                                return_face_idx=True, sample_mode='bilinear')
        # compound the mask with valid faces mask
        valid_face_mask = torch.isin(tex_face_idx, torch.tensor(valid_faces).to(device)).int()
        mask = backprojection['albedo'][..., 3:4] / 2 + 0.5
        valid_face_mask = dilate_nonblack_pool(valid_face_mask.unsqueeze(0), kernel_size=3, iterations=4).squeeze(0)
        mask = mask * (valid_face_mask.permute(1, 2, 0)) # TODO: also dilate the valid face mask
        return backprojection['albedo'][..., :3], mask, tex_face_idx
        
    
    def _render_views_to_data(self, mesh, mesh_texture, cameras:list, target_erode_kernel:int=40)->list[Dict]:
        device = torch.device('cuda')
        s = mesh_texture.shape[0]
        normal_map = mesh.materials[0].hwc().normals_texture
        if normal_map is not None:
            normal_map = torchvision.transforms.Resize((s, s))(normal_map.permute(2, 0, 1)).permute(1, 2, 0).to(device)

        view_data = []
        batch_cameras = kaolin.render.camera.Camera.cat([camera.to(device) for camera in cameras])
        render_res = custom_mesh_batched_render(batch_cameras, mesh.to(device), mesh_texture,
                                                normal_map, requires_positions=True, process_as_albedo=False, backend="cuda") #TODO:
        render_res['albedo'] = render_res['textured'][..., :3] * 2 - 1
        render_res['albedo_alpha'] = render_res['textured'][..., 3:]
        for i in range(len(cameras)):
            bg_mask = (torch.abs(render_res["camera_normals"][i:i+1]) < 0.01).all(dim=-1, keepdim=True).float()
            render_channel_res = {}
            for ch in self.inpaint_model.in_channels:
                if ch == 'inpaint_mask':
                    pass
                else:
                    render_channel_res[ch] = render_res[ch][i:i+1] * (1.0 - bg_mask) + (
                                torch.ones_like(render_res[ch][i:i+1]) * -1.0) * bg_mask
            # kornia.morphology.erosion expects (B, C, H, W); the alpha tensor here is (1, H, W, 1),
            # so permute in/out. (Prior to this fix the erosion was treating the H axis as channels
            # and a degenerate W=1, so the "erode" was a no-op-ish row reduction.)
            kernel = torch.ones(target_erode_kernel, target_erode_kernel).to(device)
            eroded = erosion(render_res["albedo_alpha"][i:i+1].permute(0, 3, 1, 2), kernel).permute(0, 2, 3, 1)
            render_res["albedo_alpha"][i:i+1] = eroded
            render_channel_res['inpaint_mask'] = render_res["albedo_alpha"][i:i+1]
            render_channel_res['albedo_alpha'] = render_res["albedo_alpha"][i:i+1]
            render_channel_res['background_alpha'] = (1.0 - bg_mask) * 2.0 - 1.0
            view_data.append(render_channel_res)
        return view_data
    
    def _model_inference(self, inpaint_data, log_dir:Optional[str]=None)->torch.Tensor:
        # 0.83s
        if log_dir is None:
            log_dir = os.path.join(self.log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f"))
            os.makedirs(log_dir, exist_ok=True)
        output = self.inpaint_model.inpaint(inpaint_data, log=True, log_dir=log_dir)
        save_concat_log(log_dir, rows=6)
        return output
        
    def _get_references(self, targets, num=7, mode:str='nearest', **kwargs) ->References:
        g = torch.Generator()
        g.manual_seed(torch.seed())
        if self.reference_library is None:
            raise RuntimeError("References not loaded. Call prepare() first.")
        if mode == 'random':
            reference_ids = torch.randint(low=0, high=self.reference_library.get_num_references(), size=(num,), generator=g)
            return self.reference_library.get_references_from_ids(reference_ids)
        if mode == 'nearest':
            start = time.time()
            reference_ids = []
            device = torch.device("cuda")
            nnfm_loss_fn = get_global_nnfm_loss()
            for target in targets:
                nnfm_loss = []
                feats = nnfm_loss_fn.get_feats(target['camera_normals'].permute(0, 3, 1, 2) / 2.0 + 0.5, [11, 13, 15])
                feats = torch.cat(feats, 1)
                for ref_feats in self.reference_library.features: #TODO:
                    target_feats = nn_feat_replace(feats, ref_feats[None].to(device))
                    nnfm_loss.append(cos_loss(feats, target_feats))
                matched_ref = torch.topk(torch.stack(nnfm_loss), k=1, largest=False).indices[0]
                #if kwargs.get("color_reference") is None:
                reference_ids.append(matched_ref)
                # else:
                #     if not kwargs.get("color_reference"):
                #         reference_ids.append(matched_ref)
                    # else:
                    #     for _ in range(8):
                    #         matched_ref = torch.topk(torch.stack(nnfm_loss), k=1, largest=False).indices[0]
                    #         nnfm_loss[matched_ref] = torch.tensor(1e6).cuda()
                    #         reference = self.reference_library.get_references_from_ids(matched_ref)
                    #         ref_albedo = (reference.albedo + 1) / 2
                    #         ref_mask = (reference.albedo_alpha > 0).float()
                    #         tar_albedo = (target["albedo"] + 1) / 2
                    #         tar_mask = (target["albedo_alpha"] > 0).float()
                    #         ref_hist = compute_color_histogram((ref_albedo * ref_mask).squeeze(0), ref_mask.squeeze(0))
                    #         inpaint_hist = compute_color_histogram((tar_albedo * tar_mask).squeeze(0), tar_mask.squeeze(0))
                    #         hist_diff = torch.norm(ref_hist.cpu() - inpaint_hist.cpu(), p=1).item()
                    #         print(f"📊 {hist_diff}")
                    #         if hist_diff < 3:  # threshold
                    #             reference_ids.append(matched_ref)
                    #             print("🏷️ matched")
                    #             break
            if len(reference_ids) == 0:
                print("⚠️ no nearest patch found, potential failure batch")
                reference_ids = torch.randint(low=0, high=self.reference_library.get_num_references(), size=(num,))
            elif len(reference_ids) < num:
                left_num = num - len(reference_ids)
                _ref_ids = torch.randint(low=0, high=self.reference_library.get_num_references(), size=(left_num,))
                reference_ids = torch.cat([torch.tensor(reference_ids),_ref_ids])
            else:
                reference_ids = torch.tensor(reference_ids)
            print(f"⏰: {time.time() - start}")
            return self.reference_library.get_references_from_ids(reference_ids)
                
    def save(self, folder, save_ref_lib=False, save_ref=False, force_save=False, rel_base=None, **kwargs):
        """Save reference library and brush config YAML."""
        os.makedirs(folder, exist_ok=True)
        
        if save_ref_lib:
            ref_lib_dir = kwargs.get('ref_lib_dir')
            if ref_lib_dir is not None:
                ref_lib_folder = os.path.join(ref_lib_dir, self.single_view.mesh_name, self.single_view.name, str(type(self).__name__))
                if str(type(self).__name__) == "PreSampledReferenceBrush":
                    ref_lib_folder = os.path.join(ref_lib_folder, self.name)
                self.reference_library_dir = ref_lib_folder
                if os.path.isdir(ref_lib_folder) and not force_save:
                    print(f"Reference library folder {ref_lib_folder} already exists, skipping save.")
                else:
                    os.makedirs(ref_lib_folder, exist_ok=True)
                    self.reference_library.save(ref_lib_folder)
                    print(f"Saved reference library to {ref_lib_folder}")
                    
        if save_ref:
            ref_folder = os.path.join(ref_lib_dir, self.name)
            self.references_dir = ref_folder
            self.references.save(ref_folder)
            print(f"Saved references to {ref_folder}")
        
        # Save config YAML after save operations since they modify the ref dir
        config_file = os.path.join(folder, f"{self.name}.yaml")
        with open(config_file, "w") as f:
            yaml.dump(self.to_config(rel_base=rel_base), f)

        print(f"Saved brush and references to {folder}")
    
    def create_test_camera(self, mesh, face_idx, camera_config, u=1/3, v=1/3, up_vidx=0):
        """Camera looking down ``face_idx``'s normal. See
        :func:`gloss_interactive.utils.camera_from_face_normal` -- shared with
        target-camera selection so both place cameras identically."""
        return camera_from_face_normal(mesh, face_idx, camera_config,
                                       u=u, v=v, up_vidx=up_vidx)

@register_brush
class AutoSampledReferenceBrush(ReferenceBrush):
    """automatically sample patches from the single view"""
    def __init__(self, name: str, single_view: SingleView, inpaint_model:Optional[TextureInpaintStandardModel]=None):
        super().__init__(name, single_view, inpaint_model)

    def prepare(self, force):
        super().prepare(force) 
        # if single view has a reference library load it 
        # else generate it
        if not self.is_prepared or force:
            self.reference_library = self.single_view.generate_reference(camera_config=self.camera_config)
    
        
@register_brush
class PreSampledReferenceBrush(ReferenceBrush):
    def __init__(self, name: str, single_view: SingleView, inpaint_model:Optional[TextureInpaintStandardModel]=None):
        super().__init__(name, single_view, inpaint_model)
        self.reference_faces = []
    
    def set_valid_face_ids(self, reference_faces):
        self.reference_faces = reference_faces
    
    def get_valid_face_ids(self):
        return self.reference_faces

    def prepare(self, force):
        super().prepare(force) 
        if not self.is_prepared or force:
            self.reference_library = self.single_view.generate_reference(num_samples=20, camera_config=self.camera_config, valid_face_ids=self.reference_faces)
        

class ReferenceBrushLibrary:
    def __init__(self, mesh_folder, inpaint_model_dir, single_views_folder, single_views_cam_folder, single_views_texture_folder,
                 brushes_folder, brushes_meta_folder, cache_folder,
                 brush_presets_folder=None,
                 fill_fov=None, fill_dist=None,
                 brush_fov=None, brush_dist=None,
                 brush_fov_dist_stack=None,
                 **kwargs):
        # Storage locations on disk
        self.mesh_folder = mesh_folder
        self.inpaint_model_dir = inpaint_model_dir
        self.single_views_folder = single_views_folder #view%04d.png
        self.single_views_cam_folder = single_views_cam_folder
        self.single_views_texture_folder = single_views_texture_folder
        self.brushes_folder = brushes_folder          # saved brushes (generated output)
        self.brush_presets_folder = brush_presets_folder  # shipped presets (data bundle), read-only
        self.brushes_meta_folder = brushes_meta_folder
        self.cache_folder = cache_folder

        # Two independent camera-config families:
        # - fill_*  : used when presampling the global "fill" camera pool per mesh.
        # - brush_* : used as defaults for each newly-created brush's camera_config
        #             stack (per-stroke cameras placed against face normals).
        # Fall back to CameraConfig defaults so older callers still work.
        _default_cam_config = CameraConfig()
        self.fill_fov = fill_fov if fill_fov is not None else _default_cam_config.fov
        self.fill_dist = fill_dist if fill_dist is not None else _default_cam_config.dist
        self.brush_fov = brush_fov if brush_fov is not None else _default_cam_config.fov
        self.brush_dist = brush_dist if brush_dist is not None else _default_cam_config.dist
        # Stack of (fov, dist) pairs applied per-stroke. When unspecified, behaves
        # like a single-element stack at (brush_fov, brush_dist) — preserving the
        # pre-stack behaviour.
        if brush_fov_dist_stack is None or len(brush_fov_dist_stack) == 0:
            self.brush_fov_dist_stack = [(self.brush_fov, self.brush_dist)]
        else:
            self.brush_fov_dist_stack = [(float(f), float(d)) for f, d in brush_fov_dist_stack]

        self.meshes = {}
        # Per-face UV pixel counts per mesh, filled lazily by
        # get_face_uv_weights and used to weight target-camera selection.
        self._face_uv_weights = {}
        self.in_channels = ["camera_normals", "relative_positions", "albedo", "inpaint_mask"]
        self.inpaint_models = {}
        # Where brushes write their per-inference logs. Set by the websocket
        # handler to point inside the session export folder; defaults to
        # <GLOSS_INTERACTIVE_DIR>/logs for direct library users.
        self.log_root = default_log_root()
        
        # Library has source that can generate brushes [choose with UI]
        # id -> single view / single view cam\
            
        # dictionary of str, Dict[int, SingleView]
        self.single_views: Dict[str, Dict[int, SingleView]] = {}
        for mesh_name in self.meshes.keys():
            self.single_views[mesh_name] = {}

        # Preload single views - choose between full loading or ID-only loading
        preload_mode = kwargs.get('preload_mode')  # 'full', 'ids_only', or 'none'
        self.preload_mode = preload_mode

        # Initialize brush storage
        self.brushes: Dict[str, ReferenceBrush] = {}
        self.brushes_folder = brushes_folder

        # TODO: set difference between ref and paint
        # self.load_mesh_model("koi_fish", load_inpaint_model=True)
        # self.load_mesh_model("croissant", load_inpaint_model=False)


    def _load_existing_brushes(self):
        # iterate through brush folder yaml files and load brushes
        if not os.path.isdir(self.brushes_folder):
            return
        for fn in os.listdir(self.brushes_folder):
            if fn.endswith('.yaml'):
                brush_config_path = os.path.join(self.brushes_folder, fn)
                with open(brush_config_path, "r") as f:
                    config = yaml.safe_load(f)
                if config["mesh_name"] not in self.inpaint_models:
                    self.load_mesh_model(config["mesh_name"])
                brush = ReferenceBrush.from_config(
                    config,
                    single_view_lookup=self.get_single_view_by_id,
                    inpaint_model=self.inpaint_models[config["mesh_name"]],
                    rel_base=self.cache_folder,
                )
                self.brushes[brush.name] = brush
                print(f"Loaded brush '{brush.name}' from {brush_config_path}")


    @staticmethod
    def from_config(config_fname,
                    fill_fov=None, fill_dist=None,
                    brush_fov=None, brush_dist=None,
                    brush_fov_dist_stack=None):
        with open(config_fname, "r") as f:
            config = yaml.safe_load(f)
        # Relative folders are taken under GLOSS_DATA_DIR.
        config = resolve_config_paths(config)
        # CLI overrides win over the yaml when supplied.
        if fill_fov is not None:
            config['fill_fov'] = fill_fov
        if fill_dist is not None:
            config['fill_dist'] = fill_dist
        if brush_fov is not None:
            config['brush_fov'] = brush_fov
        if brush_dist is not None:
            config['brush_dist'] = brush_dist
        if brush_fov_dist_stack is not None:
            config['brush_fov_dist_stack'] = brush_fov_dist_stack
        return ReferenceBrushLibrary(**config)

    def _resolve_mesh_path(self, mesh_name):
        """Find mesh file under mesh_folder. Try <mesh_name>/scene.gltf, then <mesh_name>.glb."""
        scene_path = os.path.join(self.mesh_folder, mesh_name, "scene.gltf")
        if os.path.isfile(scene_path):
            return scene_path
        glb_path = os.path.join(self.mesh_folder, f"{mesh_name}.glb")
        if os.path.isfile(glb_path):
            return glb_path
        raise FileNotFoundError(
            f"No mesh found for '{mesh_name}' under {self.mesh_folder} "
            f"(checked '{mesh_name}/scene.gltf' and '{mesh_name}.glb')"
        )

    def load_mesh_model(self, mesh_name, load_inpaint_model=False, device=torch.device('cuda')):
        if self.meshes.get(mesh_name) is None :
            mesh_path = self._resolve_mesh_path(mesh_name)
            self.meshes[mesh_name] = load_mesh(mesh_path).to(device)
            # if mesh_name in ['cabbage', 'croissant', 'dirty_tire', 'fire_hydrant', 'koi_fish', 'rusty_barrel_metal', 'gourd', 'brick', 'sea_urchin_shell', 'turtle', 'sea_dragon_body', 'sea_dragon_fin', 'sea_dragon_head_less']:
            self._preload_mesh_sv_data(mesh_name, self.preload_mode)
            print(f"preload {mesh_name} sv")
        else:
            print(f"already preload {mesh_name} sv")
        if load_inpaint_model:
            if self.inpaint_models.get(mesh_name) is not None:
                print(f"already load {mesh_name} inpainting model")
                return self.inpaint_models[mesh_name]
            else:
                inpaint_model_path = resolve_checkpoint(mesh_name, ckpt_dir=self.inpaint_model_dir)
                inpaint_model = TextureInpaintStandardModel(17, self.in_channels, inpaint_model_path, use_fp16=True)
                inpaint_model.set_attention_proc(SamplewiseAttnProcessor2_0)
                self.inpaint_models[mesh_name] = inpaint_model
                print(f"load {mesh_name} inpainting model")

    def get_face_uv_weights(self, mesh_name, resolution=1024):
        """Per-face UV pixel counts for ``mesh_name``, computed once and cached.

        Target-camera selection uses these as area weights, so "this camera
        covers the most area" is measured in texture pixels rather than in
        triangles -- a handful of large faces should outrank many slivers.

        Returns:
            torch.Tensor: ``(num_faces,)`` float weights, zero for faces that
            occupy no pixels in the UV atlas.
        """
        if mesh_name not in self._face_uv_weights:
            mesh = self.meshes[mesh_name]
            face_ids, counts, _ = get_face_uv_pixel_counts(mesh, resolution, resolution)
            weights = torch.zeros(mesh.faces.shape[0], device=mesh.vertices.device,
                                  dtype=torch.float32)
            weights[face_ids] = counts.float()
            self._face_uv_weights[mesh_name] = weights
            print(f'[brush_library] cached UV area weights for {mesh_name} '
                  f'({int((weights > 0).sum().item())}/{weights.numel()} faces in atlas)')
        return self._face_uv_weights[mesh_name]

    def _preload_mesh_sv_data(self, mesh_name, preload_mode):
        self.single_views[mesh_name] = {}
        single_views_folder = os.path.join(self.single_views_folder, mesh_name)
        if not os.path.isdir(single_views_folder):
            print(f"[warn] no single-view folder for mesh '{mesh_name}' at "
                  f"{single_views_folder}; skipping SV preload")
            return
        single_views_ids = [int(fn.split('.')[0][4:]) for fn in os.listdir(single_views_folder)
                                if fn.startswith('view')]
        if preload_mode == 'full':
                # Load full data including images (original behavior)
            for id in single_views_ids:
                self.add_single_view_full_data(mesh_name, id)
        elif preload_mode == 'ids_only':
            # Load only IDs and camera metadata, no images
            for id in single_views_ids:
                self.add_single_view_id_only(mesh_name, id)
        elif preload_mode == 'test':
            # Load 3 views for testing purposes
            for i, id in enumerate(single_views_ids):
                if i < 3:
                    self.add_single_view_full_data(mesh_name, id)   
                else:
                    self.add_single_view_id_only(mesh_name, id)

    def add_brush(self, name:str, brush_type:str, mesh_name:str, sv_id:str,
                  cam_dist:Optional[float]=None, cam_fov:Optional[float]=None):
        sv = self.get_single_view_by_id(mesh_name, sv_id)
        brush = self._add_brush(name, brush_type, sv, cam_dist=cam_dist, cam_fov=cam_fov)
        return brush

    def _get_or_load_inpaint_model(self, mesh_name):
        if mesh_name in self.inpaint_models:
            return self.inpaint_models[mesh_name]
        ckpt_path = os.path.join(self.inpaint_model_dir, f"{mesh_name}/chkpt_80000.ckpt")
        if not os.path.exists(ckpt_path):
            raise KeyError(f"No inpaint model loaded for mesh '{mesh_name}' (and no checkpoint at {ckpt_path})")
        model = TextureInpaintStandardModel(17, self.in_channels, ckpt_path, use_fp16=True)
        model.set_attention_proc(SamplewiseAttnProcessor2_0)
        self.inpaint_models[mesh_name] = model
        return model

    def _add_brush(self, name:str, brush_type:str, sv: SingleView,
                   cam_dist:Optional[float]=None, cam_fov:Optional[float]=None):
        # Per-call cam_fov / cam_dist override → single-entry stack at those values.
        # Otherwise inherit the library-level brush_fov_dist_stack.
        if cam_dist is not None or cam_fov is not None:
            eff_dist = cam_dist if cam_dist is not None else self.brush_dist
            eff_fov = cam_fov if cam_fov is not None else self.brush_fov
            stack = [(float(eff_fov), float(eff_dist))]
        else:
            stack = list(self.brush_fov_dist_stack)
        _config = {
            'name': name,
            'brush_type': brush_type,
            'sv_id': sv.sv_id,
            'mesh_name': sv.mesh_name,
            'cam_fov_dist_stack': stack,
        }
        if not self.brushes.get(name):
            brush = ReferenceBrush.from_config(_config, self.get_single_view_by_id, self._get_or_load_inpaint_model(sv.mesh_name))
            brush.log_root = self.log_root
            stack_str = ', '.join(f'(fov={f}, dist={d})' for f, d in stack)
            print(f"adding brush of name {name} cam_stack=[{stack_str}]")
            self.brushes[name] = brush
            return brush
        else:
            print(f"brush of name {name} already created, change rename brush")
            return self.brushes.get(name)

    def set_log_root(self, log_root: str):
        """Point library + every existing brush at ``log_root`` for inference logs."""
        self.log_root = log_root
        for brush in self.brushes.values():
            brush.log_root = log_root

    def save_brush(self, name, **kwargs):
        _ref_lib_dir = os.path.join(self.cache_folder, 'reference_library')
        self.brushes[name].save(folder=self.brushes_folder, ref_lib_dir=_ref_lib_dir, rel_base=self.cache_folder, **kwargs)

    def add_single_view_id_only(self, mesh_name, id):
        """Add a single view with only ID and camera metadata, without loading images from disk"""

        if self.single_views.get(id) is not None:
            return self.single_views.get(id)

        # Create new single view with ID only
        sv_name = 'view%04d' % id
        sv = SingleView.from_id_only(mesh_name, self.meshes[mesh_name], sv_name, id, 
                                     os.path.join(self.single_views_cam_folder, mesh_name),
                                     os.path.join(self.single_views_folder, mesh_name),
                                     os.path.join(self.single_views_texture_folder, mesh_name))
        self.single_views[mesh_name][id] = sv
        print(f"Added {mesh_name} single view ID {id} (metadata only)")
        return sv

    def clear_brushes(self):
        self.brushes = {}
    
    def get_single_view_by_id(self, mesh_name, id):
        """Get a single view by its ID, returns None if not found"""
        return self.single_views[mesh_name][id]
    
    def get_brush_by_name(self, name) -> ReferenceBrush:
        if name not in self.brushes:
            self._load_saved_brush(name)
        return self.brushes[name]

    def _load_saved_brush(self, name):
        """Load a brush by name on first use, so it survives a backend restart.

        Looks in the saved-brush folder (``<brushes_folder>/<name>.yaml``, written
        by Save Brush) first, then in the shipped presets (``brush_presets_folder``).
        A preset whose reference library is not on disk regenerates it in prepare().
        """
        candidates = [os.path.join(d, f"{name}.yaml")
                      for d in (self.brushes_folder, self.brush_presets_folder) if d]
        config_path = next((c for c in candidates if os.path.isfile(c)), None)
        if config_path is None:
            raise KeyError(f"Brush '{name}' is not loaded and has no saved or preset config "
                           f"(looked for {', '.join(candidates)})")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        mesh_name = config["mesh_name"]
        self.load_mesh_model(mesh_name, load_inpaint_model=True)
        brush = ReferenceBrush.from_config(config, single_view_lookup=self.get_single_view_by_id,
                                           inpaint_model=self.inpaint_models[mesh_name],
                                           rel_base=self.cache_folder)
        brush.log_root = self.log_root
        brush.prepare(force=False)
        self.brushes[name] = brush
        print(f"Loaded saved brush '{name}' from {config_path}")

    def add_single_view_full_data(self, mesh_name, id, force_reload=False):
        """Load single view with full data, handling both new loading and upgrading from ID-only"""
        assert self.single_views.get(mesh_name) is not None
        if self.single_views.get(mesh_name).get(id) is not None:
            sv = self.single_views.get(mesh_name).get(id)
            if sv._loaded:
                if not force_reload:
                    print(f"Single view with ID {id} already fully loaded")
                    return sv
                else:
                    sv.load_full_data()
                    self.single_views[mesh_name][id] = sv
                    return sv
            
        sv = SingleView.from_disk(mesh_name, self.meshes[mesh_name],
                                  'view%04d' % (id), os.path.join(self.single_views_folder, mesh_name), 
                                  os.path.join(self.single_views_cam_folder, mesh_name),
                                  os.path.join(self.single_views_texture_folder, mesh_name), id)
        self.single_views[mesh_name][id] = sv
        return sv


    def prepare_brush(self, brush_name, force=False):
        print("preparing brush ...")
        self.brushes[brush_name].prepare(force=force)


    def to_config(self):
        config = {'mesh_folder': self.mesh_folder,
                'single_views_folder': self.single_views_folder,  # "name" per view, might want name AND id
                'single_views_cam_folder': self.single_views_cam_folder,
                'single_views_texture_folder': self.single_views_texture_folder,
                'brushes_folder': self.brushes_folder,
                'brush_presets_folder': self.brush_presets_folder,
                'brushes_meta_folder': self.brushes_meta_folder,
                'cache_folder': self.cache_folder,
                'preload_mode': self.preload_mode,
                'fill_fov': self.fill_fov,
                'fill_dist': self.fill_dist,
                'brush_fov': self.brush_fov,
                'brush_dist': self.brush_dist,
                'brush_fov_dist_stack': [list(p) for p in self.brush_fov_dist_stack]}
        return config
