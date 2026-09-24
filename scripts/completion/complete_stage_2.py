import logging
import os
import math
import time
import copy
import argparse
import json
import random
from dataclasses import dataclass
from tqdm import tqdm

import torch, torchvision
from kornia.morphology import erosion
from diffusers import DDPMScheduler

import kaolin
import gloss
from gloss.model.attention import SamplewiseAttnProcessor2_0, CustomAttnProcessor2_0, AttentionGraph
from gloss.utils.single_view import backproject_render, get_valid_faces, get_valid_faces_from_texture
from gloss.utils.render_fast import custom_mesh_batched_render
from gloss.utils.kaolin_utils import load_mesh, camera_from_meta, camera_to_meta
from gloss.inpaint.blend import alpha_blend, poisson_blend, laplacian_blend
from gloss.inpaint.camera import turntable_camera_strategy
from gloss.logging.logging import log_tensor, default_log_setup
from gloss.data.render_dataloader import get_lighting_from_cam, RelativeToCenterValueProcessor, LocalCameraExtrinsicsSampler
from gloss.inpaint.models import *
from gloss.inpaint.syncmvd_utils import composite_rendered_view
from gloss.inpaint.inpaint_utils import expand_mask_soft, composite_inpaint, gaussian_blur
from gloss.utils.nnfm_loss import NNFMLoss, nn_feat_replace, cos_loss
from gloss.utils import reclaim_cuda_memory
from gloss.utils.parser import str2bool
from gloss.utils.paths import get_data_dir, resolve_checkpoint
from gloss.utils.misc import extract_view_numbers
from gloss.inpaint.inpaint_utils import dilate_nonblack_pool
from gloss.inpaint.color_correction import compute_color_histogram
from gloss.inpaint.completion_utils import (
    CameraConfig, CameraLogger, get_camera_from_face, get_face_uv_pixel_counts,
    backproject_view_to_texture, check_camera_coverage, load_texture_image,
    setup_logging_directories, save_video_from_tensors
)

import cv2
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont


logger = logging.getLogger(__name__)
default_log_setup(logging.DEBUG)

class TextureMap(object):
    def __init__(self, pipe, from_scratch=True):
        self.from_scratch = from_scratch
        self.h, self.w = pipe.h, pipe.w
        self.pipe = pipe
        self.init_texture()
        
    def update_texture_with_view(self, camera, view):
        # This function update the current texture with the new texture 
        # this function update the current mask
        backprojection, bkpj_mask, _ = self.pipe.backproject(camera, view.permute(0, 2, 3, 1).detach())
        mask_fill = (bkpj_mask - self.curr_mask).clip(0.0, 1.0)
        texture = composite_inpaint(self.curr_texture, backprojection, self.curr_mask, 
                                    bkpj_mask, mask_fill=mask_fill)
        self.curr_texture = texture
        self.curr_mask += mask_fill
        # composite inpaint
        
    def update_texture_from_compositing_views(self, cameras, views, _set_texture=False):
        # This function update the texture with composited views * weights
        n = len(cameras)
        for i in range(n):
            self._update_texture_from_compositing_views(cameras[i], (views[i:i+1]).permute(0, 2, 3, 1).detach())
        if _set_texture:
            self.set_texture(self.texture / (self.weights + 1e-8))
            
    def _update_texture_from_compositing_views(self, camera, view):
        texture, mask, weights = self.pipe.backproject(camera, view, margin=(self.pipe.view_margin!=0))
        if self.pipe.current_exp > 0:
            min_value = 1e-6 ** (1 / self.pipe.current_exp)
            weights[weights <= min_value] = min_value
        weights = mask * weights ** self.pipe.current_exp
        if self.texture is None:
            self.texture = texture * weights
            self.weights = weights
            self.mask = mask
        else:
            self.texture += texture * weights
            self.weights += weights
            self.mask = (self.mask + mask).clip(0.0, 1.0)
    
    def set_texture(self, texture):
        # This function set entire texture map
        self.curr_texture = texture 
        self.curr_mask = self.mask
        self.clear_texture()
    
    def clear_texture(self):
        self.texture, self.weights, self.mask = None, None, None
    
    def init_texture(self):
        if not self.from_scratch:
            self.curr_texture, self.curr_mask = pipe.init_texture, pipe.init_mask
        else:
            self.curr_texture = torch.ones(1, 4, self.h, self.w).cuda() * -1
            self.curr_mask = torch.zeros(1, 1, self.h, self.w).cuda()
        self.texture, self.weights, self.mask = None, None, None # keep new texture, weights and mask
        

def add_text_to_tensor(image_tensor, text="placeholder", position=(10, 10), font_size=20, color=(255, 255, 255)):
    """
    Add text to the top-left of a PyTorch image tensor.

    Args:
        image_tensor: torch.Tensor [C, H, W] with values in [0,1].
        text: str, the text to write.
        position: (x,y), position of text.
        font_size: int, size of the font.
        color: (R,G,B), text color.
    """
    # Convert tensor to PIL
    to_pil = transforms.ToPILImage()
    image_pil = to_pil(image_tensor.cpu().squeeze(0))

    # Draw text
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)

    # Convert back to tensor
    to_tensor = transforms.ToTensor()
    return to_tensor(image_pil)


class TextureCompletionPipeline(object):
    def __init__(self, model, mesh, 
                 texture_res=(2048, 2048), 
                 camera_args=type[CameraConfig], 
                 blend='alpha',
                 sync_mvd_start_exp=0, 
                 sync_mvd_end_exp=15,
                 sync_mvd_end=1.0,
                 soft_margin=20,
                 inpaint_overlap=0.4
                 ):
        
        self.inpaint_model = model
        self.mesh = mesh
        if texture_res is None:
            self.h, self.w = mesh.materials[0].diffuse_texture.shape[:2]
        else:
            self.h, self.w = texture_res

        self.camera_config = camera_args
        self.device = "cuda"

        self.inpaint_camera_sampler = None
        self.latent_texture = None
        self.inpaint_texture = None
        self.init_texture = None
        self.init_mask = None
        self.curr_texture = None
        self.mask = None
        self.selection_mask = None
        self.save_dir = None
        self.save_texture_dir = "./"
        self.model_debug_dir = None
        # minimum fraction of a reference patch that must carry texture (0 = accept any)
        self.min_reference_content = 0.0
        self.reference_camera_mode = 'uniform'  # or 'covered'
        self.reference_camera_dist = None       # default: camera_config.dist
        self.reference_content_max_attempts = 4
        self.reference_content_min_pool = 16
        self.reference_dist_decay = 0.65
        self.max_inpaint_patches = 0      # 0 = inpaint until coverage stops improving
        self.reference_content_fallback_k = 8
        num_inference_steps = 20

        self.sync_mvd_end = sync_mvd_end
        self.num_inference_steps = num_inference_steps
        self.sync_mvd_scheduler = [(sync_mvd_end_exp - sync_mvd_start_exp) * i / num_inference_steps 
                                   for i in range(num_inference_steps)]
        self.inpaint_overlap = inpaint_overlap
        
        # manual mode
        self.reference_patches = []

        if blend == 'laplacian':
            self.blend_fn = laplacian_blend
        else:
            self.blend_fn = alpha_blend

        self.soft_margin = soft_margin


    def backproject(self, camera, view, latent=False, margin=False, return_face_idx=False):
        if latent:
            texture_map = self.latent_texture
        else:
            texture_map = self.texture_map.curr_texture.squeeze(0).permute(1, 2, 0).contiguous()
        
        # Use utility function for basic backprojection
        result = backproject_view_to_texture(
            camera, view, self.mesh, texture_map, self.camera_config, self.device,
            latent=latent, margin=(self.view_margin != 0), view_margin=self.view_margin,
            return_face_idx=return_face_idx
        )
        
        if result[0] is None:
            return None, None, None
            
        backprojection, mask, tex_face_idx = result
        
        if return_face_idx:
            return backprojection, mask, tex_face_idx
            
        # Stage 2 specific: Apply distance transform weighting
        if self.view_margin != 0 and not latent:
            dist = cv2.distanceTransform(mask.squeeze(0).squeeze(-1).float().cpu().numpy().astype(np.uint8), cv2.DIST_L2, 5)
            dist_normalized = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
            weights = torch.tensor(dist_normalized, device=self.device).unsqueeze(0).unsqueeze(-1)
            
            # Get cos weights from utility function result 
            cos_weights = result[2] if len(result) > 2 else torch.ones_like(weights)
            weights = weights * cos_weights.permute(0, 2, 3, 1)
            weights = (weights - weights.min()) / (weights.max() - weights.min())
            return backprojection, mask, weights.permute(0, 3, 1, 2)
        
        return backprojection, mask, torch.ones_like(mask)

    def update_texture(self, texture, mask, from_scratch=False):
        self.curr_texture, self.mask = self.blend_fn(self.curr_texture, self.mask, texture, mask, 1.0)
        if not from_scratch:
            self.curr_texture, _ = self.blend_fn(self.init_texture, self.init_mask, self.curr_texture, self.mask, 1.0)
        self.update_mesh_material()

    def get_camera_from_face(self, face_idx, u=2/3, v=1/2):
        return get_camera_from_face(self.mesh, face_idx, self.camera_config, self.device, u, v)

    def get_face_uv_pixel_counts(self, visualize=False):
        all_face_ids, counts, tex_face_mask = get_face_uv_pixel_counts(self.mesh, self.h, self.w, visualize)
        # Store tex_face_idx for stage 2 specific functionality
        self.tex_face_idx = torch.where(tex_face_mask, 
                                      torch.arange(len(tex_face_mask.flatten()), device=self.device).view(tex_face_mask.shape), 
                                      -1)
        return all_face_ids, counts, tex_face_mask

    def check_camera_coverage(self, camera):
        """output binary camera coverage mask and face idx

        Args:
            camera (_type_): _description_

        Returns:
            _type_: _description_
        """
        return check_camera_coverage(camera, self.mesh, self.texture_map.curr_texture, self.camera_config, self.device, 
                                   getattr(self, 'view_margin', 0))

    # ------ Manual mode functions start ------
    def get_surface_camera_from_selection(self, camera, paint_selection, is_reference=False):
        custom_materials = copy.deepcopy(self.mesh.materials)
        custom_materials[0].diffuse_texture = paint_selection.to(self.device).contiguous()
        render_kwargs = {'custom_materials': custom_materials}
        face_idx_pass = kaolin.render.easy_render.RenderPass.face_idx
        lighting = kaolin.render.easy_render.default_lighting().to(self.device)
        render_res = gloss.utils.render.render_all_features(camera, self.mesh, lighting=lighting, **render_kwargs)
        alpha = render_res["albedo_alpha"][..., 0]
        render_res[face_idx_pass][alpha == -1] = -1  # remove unselected faces
        render_res["fid"] = render_res[face_idx_pass].unsqueeze(-1)
        surface_camera = []
        valid_faces, _, all_counts = get_valid_faces(render_res[face_idx_pass], render_res["geo_camera_normals"],
                                                     self.mesh.faces.shape[0], min_pixel_count=10,
                                                     max_angle_deviation=math.pi * 60 / 180.0)
        valid_face_ids = torch.where(valid_faces)[0]
        if len(valid_face_ids) < 1:
            return None
        if is_reference:
            for _ in range(7):
                sampled_fid = valid_face_ids[torch.randint(len(valid_face_ids), ())]
                # random point inside the face
                u = torch.sqrt(torch.rand((), device=self.device))
                v = torch.rand((), device=self.device)
                surface_camera.append(self.get_camera_from_face(sampled_fid, u, v))
        else:
            # use face with the most amount of pixels selected
            select_fid = torch.argmax(all_counts)
            camera = self.get_camera_from_face(select_fid)
            surface_camera.append(camera)
        return surface_camera

    def manual_reference(self, camera, reference_texture):
        surface_camera = self.get_surface_camera_from_selection(camera, reference_texture, is_reference=True)
        if surface_camera is None:
            print("Valid reference faces not found")
            return
        self.inpaint_texture = reference_texture[None].to(self.device).permute(0, 3, 1, 2)
        self.reference_patches = self.render_inpaint_data(surface_camera, cond_cameras=7)
        for i, patch in enumerate(self.reference_patches):
            for k, v in patch.items():
                if i % 2 == 0:
                    patch[k] = torchvision.transforms.functional.hflip(v.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        #         # if i % 4 == 0:
        #         #     patch[k] = torchvision.transforms.functional.vflip(v.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    def manual_complete(self, camera, update_mask):
        surface_camera = self.get_surface_camera_from_selection(camera, update_mask.repeat(1, 1, 4))
        assert self.init_texture is not None, "run set_init_texture first"
        self.curr_texture, self.mask = self.init_texture, self.init_mask
        self.update_mesh_material()
        print(len(surface_camera))
        inpaint_patch = self.render_inpaint_data(surface_camera, cond_cameras=0)
        inpaint_data = self.reference_patches + inpaint_patch
        log_model = True
        output = self.inpaint_model.inpaint(inpaint_data, log_model, self.model_debug_dir)
        backprojection, mask = self.backproject(surface_camera[0], output[-1:].permute(0, 2, 3, 1), margin=True)
        update_mask = update_mask[None].to(self.device).permute(0, 3, 1, 2)
        update_mask = ((update_mask / 2 + 0.5) > 0.9).int()
        mask = mask * update_mask
        self.tmp_texture, self.tmp_mask = backprojection, mask

    def manual_update(self):
        self.update_texture(self.tmp_texture, self.tmp_mask, False)
        self.log_texture(f"uv.png")
        self.init_texture, self.init_mask = self.curr_texture, self.mask
    
    def set_guidance_texture(self, guidance_path):
        self.guidance_texture, _ = self._load_texture_image(guidance_path)

    # ------ Manual mode functions end ------
    @torch.no_grad()
    def complete(
        self,
        batch_size,
        num_cond_views=None,
        min_num_camera=0,
        max_num_camera=350,
        num_references=200,
        paint_region=None,
        from_scratch=False,
        use_nnfm=False,
        use_color_ref=False,
        custom_sampling_weights=None,
        debug=False,
        log_model=False,
        log_final_texture_only=True,
        color_correction=True,
        view_margin=20,
        seed=0,
        guidance_path=None,
        kernel_size=9,
        t_forward=0,
        t_backward=0
    ):  
        torch.manual_seed(seed)
        self.seed = seed
        self.color_correction=color_correction
        self.use_nnfm = use_nnfm
        self.use_color_ref= use_color_ref
        self.view_margin = view_margin
        self.custom_sampling_weights = custom_sampling_weights
        if self.custom_sampling_weights is not None:
            self.custom_sampling_weights.to(self.device)
        self.max_num_camera, self.min_num_camera = max_num_camera, min_num_camera
        self.log_model=log_model
        
        self.batch_size, self.num_cond_views = batch_size, num_cond_views
        self.initialize_completion(from_scratch=from_scratch, 
                           paint_region=paint_region,
                           batch_size=batch_size,
                           min_num_camera=self.min_num_camera,
                           max_num_camera=self.max_num_camera,
                           precompute_cameras=True)
        # use tqdm to track progress
        # TODO: each step is an inpainting step from the latent obtained from syncmvd, need to set t_forward start time
        self.set_guidance_texture(guidance_path)
        self.ker_size = kernel_size
        self.texture_map.init_texture()
        self.update_mesh_material()
        
        self.t_forward = t_forward
        self.t_backward = t_backward
        self.progressive_inpaint(t_forward, self.timesteps, from_latent=True, log_model=log_model, early_stop=250) 
        
        self.update_mesh_material()
        self.log_texture(f"uv-final-raw.png")
        self.texture_map.curr_texture = self.texture_map.curr_texture.clip(0.0, 1.0)
        incomplete_mask = (self.filled_mask.unsqueeze(-1).float() - self.texture_map.curr_mask.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        curr_texture = dilate_nonblack_pool(self.texture_map.curr_texture, kernel_size=5, inpaint_mask=incomplete_mask, iterations=1000, mode='avg')
        self.texture_map.curr_texture = curr_texture
        self.update_mesh_material()
        self.log_texture(f"uv-final-inpaint.png")
        
        curr_texture = dilate_nonblack_pool(self.texture_map.curr_texture, kernel_size=5, iterations=3, mode='avg')
        self.texture_map.curr_texture = curr_texture
        self.update_mesh_material()
        self.log_texture(f"uv-final.png")
        torchvision.utils.save_image(self.texture_map.curr_texture, os.path.join(self.save_texture_dir, f"uv_final.png"))

    
    def sample_cameras(self, min_num_camera=0, max_num_camera=10, custom_sampling_weights=None):
        """presample cameras 
        #TODO: if custom sampling weights only on certain part of the object, keeps a camera confidence values [0/1]
        # if the value is high use the camera more compared to others
        # sample camera of high confidence, if cannot finish the texture, use really low confidence camera that are progressively
        growing from the high confidence region
        
        # as for camera selection, use camera of high confidence region, then low confidence regions

        Args:
            max_num_camera (int, optional): _description_. Defaults to 10.
        """
        torch.manual_seed(self.seed)
        start = time.time()
        sampling_weights = torch.ones(self.mesh.faces.shape[0]).to(self.device)
        if self.custom_sampling_weights is not None:
            sampling_weights += self.custom_sampling_weights
        else:
            sampling_weights += kaolin.ops.mesh.face_areas(self.mesh.vertices.unsqueeze(0), self.mesh.faces).squeeze(0) * 1000
        if custom_sampling_weights is not None:
            sampling_weights = custom_sampling_weights
        valid_face_ids, pixel_counts_per_face, filled_mask = self.get_face_uv_pixel_counts()
        all_counts = torch.zeros(self.mesh.faces.shape[0]).long().to(self.device)
        all_counts[valid_face_ids] = pixel_counts_per_face
        current_counts = torch.zeros_like(all_counts)
        texture_mask = self.texture_map.curr_mask.clone()
        presample_inpaint_cameras = {k: [] for k in ["update_area", "camera"]}
        
        while sampling_weights.sum() > 0:
            faces_to_sample = torch.where(sampling_weights > 0)[0]
            sampled_index = torch.multinomial(sampling_weights[faces_to_sample], num_samples=1).item()
            face_id = faces_to_sample[sampled_index]
            camera = self.get_camera_from_face(face_id)
            update_mask, tex_face_idx = self.check_camera_coverage(camera)
            update_area = (update_mask == 1) & ~(texture_mask == 1)
            sampling_weights[face_id] = 0

            if len(presample_inpaint_cameras["camera"]) > max_num_camera:
                print("Reached max camera limit")
                break

            if update_area.sum() > 10:
                texture_mask[update_area] = 1
                update_faces, update_counts = torch.unique(tex_face_idx[update_area[0]], return_counts=True)
                valid_mask = torch.isin(update_faces, valid_face_ids)
                update_faces = update_faces[valid_mask]
                update_counts = update_counts[valid_mask]
                current_counts[update_faces] += update_counts
                # check if other faces are completely filled
                filled_faces = torch.where(all_counts[update_faces] == current_counts[update_faces])
                sampling_weights[filled_faces] = 0
                presample_inpaint_cameras["camera"].append(camera.cpu())
                presample_inpaint_cameras["update_area"].append((update_mask == 1).cpu())
                # self.camera_logger.log_camera(camera, update_mask.float())
                # self.camera_logger.save_log()
                
        # if not hit min camera, sample again on faces until hit
        while len(presample_inpaint_cameras["camera"]) < min_num_camera:
            sampling_weights = torch.ones(self.mesh.faces.shape[0]).to(self.device)
            if self.custom_sampling_weights is not None:
                sampling_weights += self.custom_sampling_weights
            else:
                sampling_weights += kaolin.ops.mesh.face_areas(self.mesh.vertices.unsqueeze(0), self.mesh.faces).squeeze(0) * 1000
            face_id = torch.multinomial(sampling_weights, num_samples=1).item()
            camera = self.get_camera_from_face(face_id)
            update_mask, tex_face_idx = self.check_camera_coverage(camera)
            if update_mask is not None:
                presample_inpaint_cameras["camera"].append(camera.cpu())
                presample_inpaint_cameras["update_area"].append((update_mask == 1).cpu())
                
        num_inpaint_cameras = len(presample_inpaint_cameras["camera"])
        print("Precompute inpaint cameras", time.time() - start)
        print("Num inpaint cameras", num_inpaint_cameras)
        self.presample_inpaint_cameras = presample_inpaint_cameras["camera"]
        self.presample_inpaint_cameras_update_area = torch.cat(presample_inpaint_cameras["update_area"], dim=0)
        self.unused_inpainting_camera = torch.ones(len(presample_inpaint_cameras["camera"]))
        
        torch.save(presample_inpaint_cameras, os.path.join(self.camera_cache_dir, f"presample_inpaint_cameras_custom_weights.pt"))
    
    def sample_next_inpainting_camera(self, min_pixel_update=20):
        # remove cameras that don't update the texture
        amount_update = self.presample_inpaint_cameras_update_area * (1 - self.texture_map.curr_mask.float()).cpu()
        amount_update = amount_update.sum((1, 2, 3))
        self.unused_inpainting_camera[torch.argwhere(amount_update == 0)] = 0
        valid_cameras_mask = self.unused_inpainting_camera == 1
        if self.texture_map.curr_mask.sum() != 0:
            # choose camera with overlaps
            overlap_w_mask = self.presample_inpaint_cameras_update_area * self.texture_map.curr_mask.float().cpu()
            overlap_w_mask = overlap_w_mask.sum((1, 2, 3))
            valid_cameras_mask &= overlap_w_mask > (self.camera_config.resolution*self.camera_config.resolution*self.inpaint_overlap)
            if valid_cameras_mask.sum() == 0:
                valid_cameras_mask = self.unused_inpainting_camera == 1
        avail_cameras = torch.argwhere(valid_cameras_mask).view(-1)
        if len(avail_cameras) == 0:
            return None
        if torch.max(amount_update[avail_cameras]) < min_pixel_update: # TODO: set pix num
            return None
        
        first_idx = avail_cameras[torch.argmax(amount_update[avail_cameras])]
        self.unused_inpainting_camera[first_idx] = 0
        batch_camera_indices = [first_idx]
        
        # pick camera that updates the most area, then remove overalapping camera within the batch
        inpaint_cameras = [self.presample_inpaint_cameras[first_idx].to(self.device)]
        batch_mask_coverage = self.presample_inpaint_cameras_update_area[first_idx].clone()
        for _ in range(self.batch_size - self.num_cond_views - 1):
            amount_overlap = self.presample_inpaint_cameras_update_area[avail_cameras] * batch_mask_coverage
            amount_overlap = amount_overlap.sum((1, 2, 3))
            avail_cameras_filter = torch.argwhere((amount_overlap == 0)).view(-1)

            if len(avail_cameras_filter) == 0:
                break

            avail_cameras = avail_cameras[avail_cameras_filter]
            select_camera = torch.argmax(amount_update[avail_cameras])
            select_camera = avail_cameras[select_camera]
            amount_update[select_camera] = 0

            inpaint_cameras.append(self.presample_inpaint_cameras[select_camera].to(self.device))
            batch_camera_indices.append(select_camera)
            self.unused_inpainting_camera[select_camera] = 0
            batch_mask_coverage[self.presample_inpaint_cameras_update_area[select_camera]] = 1
        if self.log_model:
            print("inpaint camera indices: ", batch_camera_indices)           
        return inpaint_cameras
   
    def initialize_completion(self, 
                              batch_size, 
                              from_scratch=True, 
                              paint_region=None, 
                              min_num_camera=0,
                              max_num_camera=350,
                              num_references=200,
                              precompute_cameras=True):
        # 1. Initialize texture and mask
        # latent texture resolution same as SyncMVD
        self.latent_texture = torch.normal(0, 1, (4, 512, 512), device=self.device).permute(1, 2, 0)
        assert self.inpaint_texture is not None, "need to run set_reference first"
        
        self.texture_map = TextureMap(self)
        self.update_mesh_material()

        # 2. Compute reference views
        self.num_cond_views = batch_size // 2
        self.inpaint_cameras_per_batch = batch_size - self.num_cond_views
        self.precompute_references(num_references)

        # 3. Sample cameras:
        self.init_log_camera()
        if os.path.exists(self.presample_camera_path) is False:
            self.sample_cameras(min_num_camera=min_num_camera, max_num_camera=max_num_camera)
        else:
        # LOAD CAMERA FROM PATH: TODO:
            presample_inpaint_cameras = torch.load(self.presample_camera_path)
            self.presample_inpaint_cameras = presample_inpaint_cameras["camera"]
            self.presample_inpaint_cameras_update_area = torch.cat(presample_inpaint_cameras["update_area"], dim=0)
            self.unused_inpainting_camera = torch.ones(len(presample_inpaint_cameras["camera"]))
        
        self.valid_face_ids, pixel_counts_per_face, self.filled_mask = self.get_face_uv_pixel_counts()
        self.all_counts = torch.zeros(self.mesh.faces.shape[0]).long().to(self.device)
        self.all_counts[self.valid_face_ids] = pixel_counts_per_face
        self.current_counts = torch.zeros_like(self.all_counts)
        self.sampling_weights = torch.zeros(self.mesh.faces.shape[0]).to(self.device)
        
        self.num_batches = round(len(self.presample_inpaint_cameras) / self.inpaint_cameras_per_batch )

        # 4. Set inference timestep and schedulers
        self.num_timesteps = self.inpaint_model.noise_scheduler.config.num_train_timesteps
        self.inpaint_model.noise_scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        self.timesteps = self.inpaint_model.noise_scheduler.timesteps
        self.bg_latent = self.inpaint_model.background_latent(256).float()
        scheduler_config = dict(self.inpaint_model.noise_scheduler.config)

        scheduler_config["prediction_type"] = "sample"
        # used to update the latent texture
        self.latent_noise_scheduler = DDPMScheduler.from_config(scheduler_config)
        self.latent_noise_scheduler.set_timesteps(self.num_inference_steps, device=self.device)

    def progressive_inpaint(self, i, t, from_latent=True, log_model=True, early_stop=100):
        cam_idx = 0
        patches_done = 0
        not_sampled = False
        _min_pixel_update = 20
        while True:
            camera = self.sample_next_inpainting_camera(min_pixel_update=_min_pixel_update)
            if camera is None:
                if not_sampled:
                    print("resample camera >>>>>>>>>>>>>>>>")
                    self.sample_cameras(max_num_camera=100, custom_sampling_weights=self.sampling_weights) #TODO:
                    not_sampled = False
                    _min_pixel_update = 10
                else:
                    break
            if self.max_inpaint_patches:
                remaining = self.max_inpaint_patches - patches_done
                if remaining <= 0:
                    logger.info("reached max_inpaint_patches=%d", self.max_inpaint_patches)
                    break
                if camera is not None and len(camera) > remaining:
                    camera = camera[:remaining]   # last batch trimmed to land exactly on the cap
            inpaint_log_dir = os.path.join(self.model_debug_dir, f"iter-{i}-cam-{cam_idx}")
            if self.log_model:
                os.makedirs(inpaint_log_dir, exist_ok=True)
            if from_latent:
                self.inpaint_from_latent(i, camera, log=log_model, log_dir=inpaint_log_dir, 
                                         color_correct=self.color_correction, t_forward=self.t_forward, t_backward=self.t_backward)
            else:
                self.inpaint(camera, log=log_model, log_dir=inpaint_log_dir)
            patches_done += len(camera) if camera is not None else 0
            self.log_texture(f"uv-inpaint-{cam_idx}.png")
            self.update_mesh_material()
            reclaim_cuda_memory()  
            cam_idx += 1
            if early_stop is not None:
                if cam_idx > early_stop:
                    break

    def inpaint_from_latent(self, start_step, inpaint_cameras, log, log_dir, color_correct=True, t_forward=0, t_backward=0) :
        # inpainting camera and cond data from current texture
        inpaint_data, ref_cameras = self.get_inpaint_batch(inpaint_cameras, self.batch_size-len(inpaint_cameras), use_nnfm=self.use_nnfm, use_color_reference=self.use_color_ref)
        if inpaint_data is None:
            return None
        batch_data = {
            "cameras": kaolin.render.camera.Camera.cat(ref_cameras + inpaint_cameras),
            "cond_input": self.inpaint_model.process_input(inpaint_data),
        }
        syncmvd_guidance = custom_mesh_batched_render(batch_data["cameras"].to(self.device), self.mesh, custom_texture=self.guidance_texture.squeeze(0).permute(1, 2, 0),
                                                      requires_positions=False, process_as_albedo=False, backend="cuda")["textured"].permute(0, 3, 1, 2)
        if self.log_model:
            assert os.path.exists(log_dir)
            torchvision.utils.save_image(syncmvd_guidance, os.path.join(log_dir, "log-guidance.png"))
            
        batch_latent_cameras = batch_data["cameras"].to(self.device)
        batch_latent_cameras.width = 32
        batch_latent_cameras.height = 32
        r = custom_mesh_batched_render(batch_latent_cameras, self.mesh, self.latent_texture,
                                       requires_positions=False, process_as_albedo=False, backend="cuda")
        fg_latents = r["textured"].permute(0, 3, 1, 2)
        fg_masks = r["mask"].permute(0, 3, 1, 2) / 2 + 0.5
        batch_bg_latents = self.bg_latent.repeat(len(batch_data["cond_input"]), 1, 1, 1)
        latent_noise = composite_rendered_view(self.inpaint_model.noise_scheduler, batch_bg_latents, fg_latents,
                                            fg_masks, self.timesteps[self.t_forward-1])
        batch_data["latent"], guidance = self.get_noised_latent_from_img(syncmvd_guidance*2.0 -1.0, latent_noise, t=self.t_forward, g_ker=self.ker_size) # TODO: g_kernel #for
        if self.log_model:
             assert os.path.exists(log_dir)
             torchvision.utils.save_image(guidance/2.0+0.5, os.path.join(log_dir, "log-noised-guidance.png"))
        # diffusion steps
        with torch.no_grad():
            for i in range(start_step, self.num_inference_steps):
                t = self.timesteps[i]
                noise_pred = self.inpaint_model.step(batch_data["cond_input"].to(self.device), batch_data["latent"], t)
                latent, latent_image = self.inpaint_model.noise_scheduler.step(noise_pred, t, batch_data["latent"], return_dict=False)
                batch_data["latent"] = latent
            # backproject to the texture map
            batch_data["latent"] = batch_data["latent"].to(torch.float16)
            view_decoded = self.inpaint_model.pipe.vae.decode(
                    batch_data["latent"] / self.inpaint_model.pipe.vae.config.scaling_factor, return_dict=False)[0]
            view_decoded = self.inpaint_model.pipe.image_processor.postprocess(view_decoded, 'pt')
        raw_input = self.inpaint_model.set_input(inpaint_data)
        if self.log_model:
            assert os.path.exists(log_dir)
            for ch in self.inpaint_model.in_channels:
                torchvision.utils.save_image(
                    (raw_input[ch].permute(0, 3, 1, 2) / 2) + 0.5,
                    os.path.join(log_dir, f"log-{ch}.png"),
                    )
        # do color correction only when there are a good amount of overlaps
        if self.color_correction:
            if self.log_model:
                torchvision.utils.save_image(view_decoded, os.path.join(log_dir, f"log-output-raw.png"))
            alpha = (raw_input["inpaint_mask"] == 1).permute(0, 3, 1, 2)
            input_rgb = (raw_input["albedo"].permute(0, 3, 1, 2) / 2 + 0.5)
            for i in range(view_decoded.shape[0]):
                bg_alpha = (inpaint_data[i]["background_alpha"] == -1)
                alpha[i, bg_alpha[..., 0]] = 0
                if alpha[i].sum() > 256 * 256 * 0.2:
                    result = reinhard_color_transfer(input_rgb[i], view_decoded[i], alpha[i])
                    view_decoded[i] = result
            if self.log_model:
                torchvision.utils.save_image(view_decoded, os.path.join(log_dir, f"log-output.png")) 
        else:
            if self.log_model:
                torchvision.utils.save_image(view_decoded, os.path.join(log_dir, f"log-output.png"))
        if self.log_model:
            self.log_reference_target_pairs(log_dir, raw_input, view_decoded, inpaint_cameras)
        for j in range(len(inpaint_cameras)):
            idx = self.batch_size-len(inpaint_cameras)+j
            cam = inpaint_cameras[j]
            self.texture_map.update_texture_with_view(cam, view_decoded[idx:idx+1])

    def log_reference_target_pairs(self, log_dir, raw_input, view_decoded, inpaint_cameras):
        """Dump the reference/target pairing chosen by get_inpaint_batch for this batch.

        Writes ``pairing.json`` (matched reference index, NNFM distance, how much of the
        target patch the single reference view already covers, and both camera metas) plus
        one ``pair-<target>.png`` sheet per target patch laying the reference patch's
        texture and geometry next to the target patch's geometry and generated texture.
        """
        pairing = getattr(self, 'last_batch_pairing', None)
        if not pairing:
            return
        try:
            self._write_reference_target_pairs(log_dir, raw_input, view_decoded, inpaint_cameras, pairing)
        except Exception:  # logging must never take down an inference run
            logger.exception("failed to write reference/target pairing log to %s", log_dir)

    def _write_reference_target_pairs(self, log_dir, raw_input, view_decoded, inpaint_cameras, pairing):
        num_cond_views = getattr(self, 'last_num_cond_views', self.batch_size - len(inpaint_cameras))
        ref_cameras = getattr(self, 'last_reference_cameras', [])
        record = {
            'num_cond_views': num_cond_views,
            'num_target_patches': len(inpaint_cameras),
            'num_matched_references': getattr(self, 'last_num_matched_references', None),
            'color_reference': getattr(self, 'last_color_reference', None),
            'min_reference_content': self.min_reference_content,
            'eligible_reference_pool': int(getattr(self, 'last_eligible_pool', -1)),
            'pairs': [],
        }
        for entry in pairing:
            pair = dict(entry)
            slot = entry.get('reference_slot')
            if slot is not None and slot < len(ref_cameras):
                pair['reference_camera'] = camera_to_meta(ref_cameras[slot])
            ti = entry['target_index']
            if ti < len(inpaint_cameras):
                pair['target_camera'] = camera_to_meta(inpaint_cameras[ti])
            record['pairs'].append(pair)
        # written before the sheets so a rendering problem cannot cost us the pairing data
        with open(os.path.join(log_dir, 'pairing.json'), 'w') as f:
            json.dump(record, f, indent=1)

        def channel(name, idx):
            """One log tile, always 3-channel: inpaint_mask is single-channel."""
            tile = (raw_input[name][idx:idx + 1].permute(0, 3, 1, 2).float() / 2) + 0.5
            if tile.shape[1] == 1:
                tile = tile.expand(-1, 3, -1, -1)
            return tile[:, :3]

        for entry in pairing:
            slot = entry.get('reference_slot')
            if slot is None:
                continue
            ti = entry['target_index']
            tgt_idx = num_cond_views + ti
            if tgt_idx >= view_decoded.shape[0]:
                continue
            tiles = [channel('albedo', slot), channel('camera_normals', slot),
                     channel('relative_positions', slot),
                     channel('camera_normals', tgt_idx), channel('relative_positions', tgt_idx),
                     channel('inpaint_mask', tgt_idx), view_decoded[tgt_idx:tgt_idx + 1].float()[:, :3]]
            torchvision.utils.save_image(torch.cat(tiles, 0).clamp(0, 1),
                                         os.path.join(log_dir, f"pair-{ti}.png"), nrow=len(tiles))

    def inpaint(self, inpaint_cameras, log, log_dir):
        if log:
            os.makedirs(log_dir, exist_ok=True)
        inpaint_data, _ = self.get_inpaint_batch(inpaint_cameras, self.batch_size-len(inpaint_cameras), use_nnfm=self.use_nnfm)
        output = self.inpaint_model.inpaint(inpaint_data, log, log_dir, color_correct=False)
        self.texture_map.update_texture_from_compositing_views(inpaint_cameras, output)
        self.texture_map.set_texture(self.texture_map.texture / (self.texture_map.weights + 1e-8))  
        
    def init_log_camera(self):
        self.camera_logger = CameraLogger(self.h, self.w, self.camera_selection_info_dir, self.mesh.faces.device)

    def batch_data_to_views(self, batch_data, t):
        with torch.no_grad():
            if t == 1:
                inpaint_latents = batch_data["latent"].to(self.device)
            else:
                inpaint_latents = batch_data["latent_image"].to(self.device)
            inpaint_latents = inpaint_latents.to(torch.float16)
            view_decoded = self.inpaint_model.pipe.vae.decode(
                inpaint_latents / self.inpaint_model.pipe.vae.config.scaling_factor, return_dict=False)[0]
            view_decoded = self.inpaint_model.pipe.image_processor.postprocess(view_decoded, 'pt')
        return view_decoded[-self.num_cond_views:]
        
    def log_texture(self, fn):
        assert os.path.exists(self.save_texture_dir)
        fp = os.path.join(self.save_texture_dir, fn)
        # add text to image tensor
        uv_log = add_text_to_tensor(self.texture_map.curr_texture.detach().cpu(), text=fn.split()[0], font_size=150)
        self.uv_logs.append(uv_log)
        if self.log_model:
            torchvision.utils.save_image(uv_log, fp)

    def _load_texture_image(self, path):
        return load_texture_image(path, self.device, self.h, self.w)

    def set_init_texture(self, texture, texture_res=None):
        """ texture param can have either string type for a file path or a tensor of shape [h, w, 4] """
        if texture_res is not None:
            self.h, self.w = texture_res, texture_res

        if isinstance(texture, str):
            albedo, mask = self._load_texture_image(texture)
            self.init_texture, self.init_mask = albedo, mask
        else:
            texture = texture.to(self.device).permute(2, 0, 1)
            if texture_res is not None:
                resize = torchvision.transforms.Resize((self.h, self.w))
                texture = resize(texture)
            else:
                self.h, self.w = texture.shape[-2:]
            mask = (texture[None, 3:] > 0.9).float()
            self.init_texture = texture[None] * mask  # shape [1, 4, h, w]
            self.init_mask = mask  # shape [1, 1, h, w]

    def set_logging(self, save_dir):
        self.uv_logs = []
        self.save_dir = save_dir
        
        # Use utility function for directory setup
        dirs = setup_logging_directories(save_dir)
        self.save_texture_dir = dirs['save_texture_dir']
        self.model_debug_dir = dirs['model_debug_dir']
        self.cache_dir = dirs['cache_dir']
        self.camera_selection_info_dir = dirs['camera_selection_info_dir']
        
        # Stage 2 specific directory
        self.camera_info_dir = os.path.join(save_dir, 'camera_info')
        os.makedirs(self.camera_info_dir, exist_ok=True)

    def set_reference(self, texture=None, view=None, view_camera=None):
        """
        texture param can have either string type for a file path or a tensor of shape [h, w, 4]
        view param can have either string type for a file path or a tensor of shape [h, w, 3]
        if view is None, then will use view_camera or default camera to render view from mesh
        if texture is None, then will backproject the rendered view to be reference texture
        """
        if isinstance(view, str) and os.path.exists(view):  # load view info from image
            init_view = kaolin.io.utils.read_image(view).unsqueeze(0)
        elif isinstance(view, torch.Tensor):
            init_view = view[None].to(self.device)
        else:  # render view from mesh texture
            if view_camera is None:
                resolution = 512
                view_camera = kaolin.render.easy_render.default_camera(resolution).to(self.device)
                print("Using default camera for view embedding")
            lighting = kaolin.render.easy_render.default_lighting().to(self.device)
            init_view = kaolin.render.easy_render.render_mesh(view_camera, self.mesh, lighting=lighting)["albedo"]
        if self.inpaint_model is not None:
            self.inpaint_model.set_embedding("", init_view[..., :3].permute(0, 3, 1, 2))

        if isinstance(texture, str) and os.path.exists(texture):
            albedo, mask = self._load_texture_image(texture)
            self.inpaint_texture = albedo
        elif isinstance(texture, torch.Tensor):
            self.inpaint_texture = texture[None].to(self.device).permute(0, 3, 1, 2)
            mask = ((self.inpaint_texture[:, 3:] / 2 + 0.5) > 0.9).int()
        else:
            # backproject from rendered view
            self.inpaint_texture, mask = self.backproject(view_camera, init_view)
        if self.init_texture is None:  # if not set using set_init_texture
            self.h, self.w = self.inpaint_texture.shape[-2:]
            self.init_texture, self.init_mask = self.inpaint_texture, mask
        valid_face_ids = get_valid_faces_from_texture(self.mesh, self.inpaint_texture)
        self.inpaint_camera_sampler = LocalCameraExtrinsicsSampler(self.mesh, camera_dist=self.camera_config.dist)
        sampling_weights = torch.zeros(self.mesh.faces.shape[0]).to(self.device)
        sampling_weights[valid_face_ids] = 1.0
        self.inpaint_camera_sampler.set_sampling_weights(sampling_weights)

    def precompute_references(self, num_references=200, albedo_coverage=None):
        """Build the reference-patch pool, tightening the reference camera framing until enough
        candidates clear ``min_reference_content``.

        A patch rendered at the default distance spans far more surface than a single
        conditioning view covers, so at that framing almost nothing clears a 50% floor. Each
        retry pulls the reference camera closer, which raises content at the cost of showing
        the model a more magnified exemplar; the loosest framing that satisfies the floor wins.
        """
        dist = self.reference_camera_dist or self.camera_config.dist
        for attempt in range(self.reference_content_max_attempts):
            self._build_reference_pool(num_references, dist)
            if self.min_reference_content <= 0:
                return
            passing = int((self._reference_content >= self.min_reference_content).sum())
            last = attempt == self.reference_content_max_attempts - 1
            if passing >= self.reference_content_min_pool or last:
                logger.info("reference pool settled at dist=%.3f: %d/%d patches clear content %.2f",
                            dist, passing, num_references, self.min_reference_content)
                return
            logger.info("only %d/%d patches clear content %.2f at dist=%.3f; tightening framing",
                        passing, num_references, self.min_reference_content, dist)
            dist *= self.reference_dist_decay

    def _build_reference_pool(self, num_references, dist):
        reference_material_map = self.inpaint_texture.squeeze(0).permute(1, 2, 0).contiguous()
        normal_map = self.mesh.materials[0].hwc().normals_texture
        normal_map = torchvision.transforms.Resize((self.h, self.w))(normal_map.permute(2, 0, 1)).permute(1, 2, 0)
        cameras = self._sample_reference_cameras(num_references, dist)
        batch_size = 64
        num_batches = int(math.ceil(num_references / batch_size))
        nnfm_loss_fn = NNFMLoss(self.device)
        references = {k: [] for k in ["camera_normals", "relative_positions", "albedo", "albedo_alpha", "features"]}
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_references)
            batch_cameras = kaolin.render.camera.Camera.cat([cameras[idx] for idx in range(start_idx, end_idx)])
            r = custom_mesh_batched_render(batch_cameras, self.mesh, reference_material_map, normal_map,
                                           requires_positions=True, process_as_albedo=False, backend="cuda")
            references['camera_normals'].append(r['camera_normals'].cpu())
            references['relative_positions'].append(r['relative_positions'].cpu())
            references['albedo'].append(r['textured'][..., :3].cpu() * 2 - 1)
            references['albedo_alpha'].append(r['textured'][..., 3:].cpu())  # already -1, 1
            with torch.no_grad():
                feats = nnfm_loss_fn.get_feats(r['camera_normals'].permute(0, 3, 1, 2) / 2.0 + 0.5, [11, 13, 15])
            references['features'].append(torch.cat(feats, 1).cpu())
            del r, feats
            reclaim_cuda_memory()
        self._references = {k: torch.cat(v, dim=0) for k, v in references.items()}
        self._reference_cameras = cameras

        # Fraction of each reference patch's object footprint that actually carries texture.
        # Geometry-only NNFM matching cannot see this, so it is what min_reference_content filters on.
        alpha = self._references['albedo_alpha']
        obj = ~(self._references['camera_normals'].abs() < 0.01).all(dim=-1, keepdim=True)
        obj_px = obj.flatten(1).sum(1).clamp(min=1)
        filled = ((alpha > 0) & obj).flatten(1).sum(1)
        self._reference_content = filled.float() / obj_px.float()
        logger.info("reference pool: %d patches, texture content min=%.3f median=%.3f max=%.3f",
                    len(self._reference_content), float(self._reference_content.min()),
                    float(self._reference_content.median()), float(self._reference_content.max()))

    def _sample_reference_cameras(self, num_references, dist=None):
        """Cameras used to render the reference patch pool.

        ``uniform`` samples the whole object, which is what starves the pool when the
        conditioning view covers only part of it: most patches land on untextured surface.
        ``covered`` anchors the cameras on faces the reference texture actually fills, so the
        patches carry texture and a content floor becomes reachable.
        """
        cfg = self.camera_config
        if dist is not None and dist != cfg.dist:
            cfg = copy.copy(cfg)
            cfg.dist = dist
        if self.reference_camera_mode == "covered":
            valid_faces = get_valid_faces_from_texture(self.mesh, self.inpaint_texture)
            if valid_faces is not None and len(valid_faces) > 0:
                pick = valid_faces[torch.randint(len(valid_faces), (num_references,))]
                logger.info("reference cameras anchored on %d/%d textured faces",
                            len(valid_faces), self.mesh.faces.shape[0])
                return [get_camera_from_face(self.mesh, int(f), cfg, self.device,
                                             u=float(torch.rand(()) * 0.5 + 0.4),
                                             v=float(torch.rand(())))
                        for f in pick]
            logger.warning("reference texture fills no complete face; using uniform cameras")
        extrinsics = self.inpaint_camera_sampler.generate(num_references)
        return [gloss.utils.render.make_camera_from_extr_intr(
                    extrinsics[i], cfg.fov,
                    resolution=cfg.resolution,
                    device=self.mesh.faces.device)
                for i in range(num_references)]

    def _eligible_reference_indices(self):
        """Reference candidates carrying at least ``min_reference_content`` texture.

        Falls back to the best-covered candidates when nothing clears the bar, so a sparse
        reference view degrades the pool rather than stalling the batch.
        """
        content = getattr(self, '_reference_content', None)
        total = self._references['albedo'].shape[0]
        if content is None or self.min_reference_content <= 0:
            return torch.arange(total)
        eligible = torch.nonzero(content >= self.min_reference_content).view(-1)
        if eligible.numel() == 0:
            k = min(total, max(1, self.reference_content_fallback_k))
            eligible = torch.topk(content, k).indices
            logger.warning("no reference patch reaches content %.2f (best %.3f); falling back to "
                           "the %d best-covered patches", self.min_reference_content,
                           float(content.max()), k)
        return eligible

    def get_inpaint_batch(self, inpaint_cameras, num_cond_views, use_nnfm=False, use_color_reference=False):
        color_reference = False
        current_material_map = self.texture_map.curr_texture.squeeze(0).permute(1, 2, 0).contiguous()
        batched_cameras = kaolin.render.camera.Camera.cat(inpaint_cameras)
        normal_map = self.mesh.materials[0].hwc().normals_texture
        normal_map = torchvision.transforms.Resize((self.h, self.w))(normal_map.permute(2, 0, 1)).permute(1, 2, 0)
        r = custom_mesh_batched_render(batched_cameras, self.mesh, current_material_map, normal_map,
                                       requires_positions=True, process_as_albedo=False, backend="cuda")
        r['albedo'] = r['textured'][..., :3] * 2 - 1
        r['albedo_alpha'] = r['textured'][..., 3:]  # already -1, 1
        
        # # if albedo_alpha is large, we calculate the color histogram of albedo region
        # # calculate the number of pixels with albedo_alpha > 0.0
        if use_color_reference:
            valid_mask = (r['albedo_alpha'] > 0.0).float().sum((1, 2, 3))
            total_pixels = r['albedo_alpha'].shape[1] * r['albedo_alpha'].shape[2]
            valid_ratio = valid_mask / total_pixels
            if valid_ratio[0].item() > 0.3:
                color_reference = True

        if use_nnfm:
            nnfm_loss_fn = NNFMLoss(self.device)
            feats = nnfm_loss_fn.get_feats(r['camera_normals'].permute(0, 3, 1, 2) / 2.0 + 0.5, [11, 13, 15])
            feats = torch.cat(feats, 1)
            del nnfm_loss_fn

        reference_patches = []
        inpaint_patches = []
        reference_cameras = []
        pairing = []

        def get_input_data(res):
            render_channels = {}
            bg_mask = (torch.abs(res["camera_normals"]) < 0.01).all(dim=-1, keepdim=True).float()
            for ch in self.inpaint_model.in_channels:
                if ch != 'inpaint_mask':
                    render_channels[ch] = res[ch] * (1 - bg_mask) + bg_mask * -1
            # kernel = torch.ones(15, 15).to(self.device)
            render_channels['inpaint_mask'] = res["albedo_alpha"]#erosion(res["albedo_alpha"], kernel)
            render_channels['albedo_alpha'] = render_channels['inpaint_mask']
            render_channels['background_alpha'] = (1.0 - bg_mask) * 2.0 - 1.0
            return render_channels

        log_pairing = getattr(self, 'log_model', False)
        eligible_idx = self._eligible_reference_indices()
        eligible_mask = torch.zeros(self._references['albedo'].shape[0], dtype=torch.bool)
        eligible_mask[eligible_idx] = True
        ref_content = getattr(self, '_reference_content', None)

        def pick_random_reference():
            return eligible_idx[torch.randint(len(eligible_idx), ())]

        def content_of(idx):
            return float(ref_content[idx]) if ref_content is not None else None

        def patch_stats(patch):
            """Object footprint and how much of it the reference view already textured."""
            obj = (patch['background_alpha'] > 0).float()
            obj_px = obj.sum().item()
            covered_px = ((patch['albedo_alpha'] > 0).float() * obj).sum().item()
            return {'object_frac': obj_px / obj.numel(),
                    'covered_frac': (covered_px / obj_px) if obj_px > 0 else 0.0}

        for i in range(len(inpaint_cameras)):
            if use_nnfm and color_reference:
                x_feats = feats[i:i+1]
                nnfm_loss = []
                for s_feats in self._references["features"]:
                    target_feats = nn_feat_replace(x_feats, s_feats[None].to(self.device))
                    nnfm_loss.append(cos_loss(x_feats, target_feats))
                for j in range(len(nnfm_loss)):  # exclude patches below the content floor
                    if not eligible_mask[j]:
                        nnfm_loss[j] = torch.tensor(1e6).to(self.device)
                input_data = {k: v[i:i+1] for k, v in r.items()}
                inpaint_patch = get_input_data(input_data)
                inpaint_patches.append(inpaint_patch)
                rejected = []
                accepted = None
                # topk = 8, if pass if not hit
                for _ in range(8):
                    matched_ref = torch.topk(torch.stack(nnfm_loss), k=1, largest=False).indices[0]
                    nnfm_dist = float(nnfm_loss[matched_ref])
                    nnfm_loss[matched_ref] = torch.tensor(1e6).cuda()
                    ref_data = {k: v[matched_ref:matched_ref+1].to(self.device) for k, v in self._references.items()}
                    reference_patch = get_input_data(ref_data)
                    ref_albedo = (ref_data['albedo'] + 1) / 2
                    ref_mask = (ref_data['albedo_alpha'] > 0).float()
                    inpaint_albedo = (input_data['albedo'] + 1) / 2
                    inpaint_mask = (input_data['albedo_alpha'] > 0).float()
                    ref_hist = compute_color_histogram((ref_albedo * ref_mask).squeeze(0), ref_mask.squeeze(0))
                    inpaint_hist = compute_color_histogram((inpaint_albedo * inpaint_mask).squeeze(0), inpaint_mask.squeeze(0))
                    hist_diff = torch.norm(ref_hist - inpaint_hist, p=1).item()
                    if hist_diff < 3:  # threshold
                        accepted = (int(matched_ref), nnfm_dist, hist_diff, len(reference_patches))
                        reference_patches.append(reference_patch)
                        reference_cameras.append(self._reference_cameras[matched_ref])
                        break
                    rejected.append({'reference_index': int(matched_ref), 'nnfm_distance': nnfm_dist,
                                     'hist_diff': hist_diff})
                if log_pairing:
                    entry = {'target_index': i, 'selection': 'nnfm+color_hist',
                             'rejected_references': rejected, 'matched': accepted is not None}
                    if accepted is not None:
                        entry.update({'reference_index': accepted[0], 'nnfm_distance': accepted[1],
                                      'hist_diff': accepted[2], 'reference_slot': accepted[3],
                                      'reference_content': content_of(accepted[0])})
                    entry.update(patch_stats(inpaint_patch))
                    pairing.append(entry)
            elif use_nnfm and not color_reference:
                x_feats = feats[i:i+1]
                nnfm_loss = []
                for s_feats in self._references["features"]:
                    target_feats = nn_feat_replace(x_feats, s_feats[None].to(self.device))
                    nnfm_loss.append(cos_loss(x_feats, target_feats))
                stacked_nnfm = torch.stack(nnfm_loss)
                matched_ref = torch.argmin(stacked_nnfm.masked_fill(
                    ~eligible_mask.to(stacked_nnfm.device), float('inf')))
                input_data = {k: v[i:i+1] for k, v in r.items()}
                inpaint_patch = get_input_data(input_data)
                ref_data = {k: v[matched_ref:matched_ref+1].to(self.device) for k, v in self._references.items()}
                reference_patch = get_input_data(ref_data)
                if log_pairing:
                    entry = {'target_index': i, 'selection': 'nnfm', 'matched': True,
                             'reference_index': int(matched_ref),
                             'nnfm_distance': float(stacked_nnfm[matched_ref]),
                             'nnfm_distance_median': float(stacked_nnfm.median()),
                             'reference_content': content_of(matched_ref),
                             'reference_slot': len(reference_patches)}
                    entry.update(patch_stats(inpaint_patch))
                    pairing.append(entry)
                inpaint_patches.append(inpaint_patch)
                reference_patches.append(reference_patch)
                reference_cameras.append(self._reference_cameras[matched_ref])
            else:
                matched_ref = pick_random_reference()
                input_data = {k: v[i:i+1] for k, v in r.items()}
                inpaint_patch = get_input_data(input_data)
                ref_data = {k: v[matched_ref:matched_ref+1].to(self.device) for k, v in self._references.items()}
                reference_patch = get_input_data(ref_data)
                if log_pairing:
                    entry = {'target_index': i, 'selection': 'random', 'matched': True,
                             'reference_index': int(matched_ref),
                             'reference_content': content_of(matched_ref),
                             'reference_slot': len(reference_patches)}
                    entry.update(patch_stats(inpaint_patch))
                    pairing.append(entry)
                inpaint_patches.append(inpaint_patch)
                reference_patches.append(reference_patch)
                reference_cameras.append(self._reference_cameras[matched_ref])

        # won't process this batch if cannot find good references
        if use_nnfm and color_reference:
            if len(reference_patches) == 0:
                self.last_batch_pairing = pairing
                self.last_reference_cameras = []
                return None, None

        # fill the rest with random sampled patches
        num_matched_references = len(reference_patches)
        while len(reference_patches) < num_cond_views:
            i = pick_random_reference()
            ref_data = {k: v[i:i + 1].to(self.device) for k, v in self._references.items()}
            reference_patch = get_input_data(ref_data)
            reference_patches.append(reference_patch)
            reference_cameras.append(self._reference_cameras[i])

        self.last_batch_pairing = pairing
        self.last_eligible_pool = len(eligible_idx)
        self.last_reference_cameras = list(reference_cameras)
        self.last_num_cond_views = num_cond_views
        self.last_num_matched_references = num_matched_references
        self.last_color_reference = color_reference

        return reference_patches + inpaint_patches, reference_cameras

    def update_mesh_material(self):
        curr_materials = copy.deepcopy(self.mesh.materials)
        curr_materials[0].diffuse_texture = self.texture_map.curr_texture.squeeze(0).permute(1, 2, 0).contiguous()
        self.mesh.materials = curr_materials

    def save_completion_video(self, path):
        if self.save_dir is not None:
            path = os.path.join(self.save_dir, path)
        save_video_from_tensors(self.uv_logs, path)

    def render_turnaround_video(self, path, camera_dist=4.0, resolution=1024):
        cameras = turntable_camera_strategy(camera_dist, spacing=0.25, resolution=resolution, fov=0.4,
                                            device=self.device)
        images = []
        for cam in cameras:
            # Ensure tensor is on CPU and convert to PIL Image
            lighting = get_lighting_from_cam(cam)
            render_res = gloss.utils.render.render_all_features(cam, self.mesh, lighting, ["albedo"])[
                             "albedo"] / 2 + 0.5
            img = (render_res.detach().cpu() * 255).clip(0, 255).to(torch.uint8)
            images.append(img)

        if self.save_dir is not None:
            path = os.path.join(self.save_dir, path)
        images = torch.cat(images, 0)
        torchvision.io.write_video(path, images, fps=5)  # NOTE: pip install av==12.0.0
    
    def get_noised_latent_from_img(self, img, noise, t=0, g_ker=21):
        if g_ker is not None:
            #img = gaussian_blur(img, size=int(g_ker))
            img = img.to(torch.float16)
        with torch.no_grad():
            x_0_latent = self.inpaint_model.pipe.vae.encode(img[:, :3, ...], return_dict=False)[0].sample() * self.inpaint_model.pipe.vae.config.scaling_factor
        return self.inpaint_model.noise_scheduler.add_noise(x_0_latent, noise, self.timesteps[t]), img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--object_name", type=str, nargs="?", const="cabbage", default="cabbage", help="Name of the object to run on")
    parser.add_argument("--view_id", type=str, nargs="?", const="0151", default="0151", help="View ID to run on")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data root (default: $GLOSS_DATA_DIR)")
    parser.add_argument("--expr_dir", type=str, default="expr",
                        help="Directory to save results (default: ./expr)")
    parser.add_argument("--max_angle", type=int, default=None,
                        help="Max backprojection angle; default: per-object table, else 90")
    parser.add_argument("--view_margin", type=int, default=None,
                        help="Reference view margin; default: per-object table, else 40")
    parser.add_argument("--max_num_camera", type=int, nargs="?", const=350, default=350, help="Maximum number of cameras to use")
    parser.add_argument("--min_num_camera", type=int, nargs="?", const=0, default=0, help="Minimum number of cameras to use")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Optional explicit checkpoint path")
    parser.add_argument("--cond_view_root", type=str, default=None, help="Optional root containing <object_name>/viewXXXX.basecolor.png")
    parser.add_argument("--texture_root", type=str, default=None, help="Optional root containing <object_name>/viewXXXX.png")
    parser.add_argument("--sample_weight_root", type=str, default=None, help="Optional root containing <object_name>-sample_weight.pt")
    parser.add_argument("--guidance_texture_path", type=str, default=None, help="Optional explicit stage-1 guidance texture path")
    parser.add_argument("--guidance_num_cameras", type=int, default=600, help="num_cameras used in stage 1; selects c=N in default guidance path")
    parser.add_argument("--mesh_path", type=str, default=None, help="Optional explicit mesh path (overrides <data_dir>/meshes/<object_name>/scene.gltf)")
    parser.add_argument("--log", type=str2bool, nargs="?", const=False, default=False, help="Whether to log the model")
    parser.add_argument("--max_inpaint_patches", type=int, default=0,
                        help="Stop after this many target patches have been inpainted "
                             "(0 = run until coverage stops improving)")
    parser.add_argument("--reference_camera_dist", type=float, default=None,
                        help="Camera distance for reference patches (default: the completion "
                             "camera distance). Closer framing raises texture content per patch.")
    parser.add_argument("--reference_camera_mode", type=str, default="uniform",
                        choices=("uniform", "covered"),
                        help="How reference-patch cameras are sampled: 'uniform' over the object "
                             "(legacy) or 'covered', anchored on faces the reference texture fills")
    parser.add_argument("--min_reference_content", type=float, default=0.0,
                        help="Minimum fraction of a reference patch that must carry texture for it "
                             "to be selectable (0 = accept any, the legacy behaviour)")
    parser.add_argument("--seed", type=int, nargs="?", const=0, default=0, help="Random seed")
    args = parser.parse_args()

    backproject_angles = {"brick": 90, "cabbage": 90, "croissant": 30, "dirty_tire": 30, "fire_hydrant": 60,
                          "gourd": 90, "koi_fish": 60, "rusty_barrel_metal": 90, "sea_urchin_shell": 90, "turtle": 90,
                          "sea_dragon_body": 90, "sea_dragon_fin": 90, "sea_dragon_head_less": 90,
                          "carp_fish_1": 60, "barrel_2": 90, "bark_1": 30,
                          "conch_1": 90, "croissant_1_04": 30, "croissant_1_05": 30, "croissant_1_08": 30,
                          "ginger_root_1": 90, "pumpkin_1": 90, "turtle_1": 90,
                          "cabbage_1": 90, "cabbage_2": 90, "cabbage_3": 90,
                          "fish_1": 60, "fish_3": 60, "fish_4": 60,
                          "toad_on_tortoise_shell_1": 90, "tortoise_1": 90,
                          "cantaloup_1": 90, "cantaloup_2": 90, "snail_1": 90}
    view_margins = {"brick": 40, "cabbage": 40, "croissant": 20, "dirty_tire": 20, "fire_hydrant": 40, "gourd": 40,
                    "koi_fish": 40, "rusty_barrel_metal": 40, "sea_urchin_shell": 40, "turtle": 40,
                    "sea_dragon_body": 40, "sea_dragon_fin": 40, "sea_dragon_head_less": 40,
                    "carp_fish_1": 40, "barrel_2": 40, "bark_1": 20,
                    "conch_1": 40, "croissant_1_04": 20, "croissant_1_05": 20, "croissant_1_08": 20,
                    "ginger_root_1": 40, "pumpkin_1": 40, "turtle_1": 40,
                    "cabbage_1": 40, "cabbage_2": 40, "cabbage_3": 40,
                    "fish_1": 40, "fish_3": 40, "fish_4": 40,
                    "toad_on_tortoise_shell_1": 40, "tortoise_1": 40,
                    "cantaloup_1": 40, "cantaloup_2": 40, "snail_1": 40}
    
    data_dir = args.data_dir or str(get_data_dir())
    expr_dir = args.expr_dir
    object_name = args.object_name
    view_id = "%04d" % int(args.view_id)
    max_num_camera = args.max_num_camera
    min_num_camera = args.min_num_camera
    
    in_channels = ["camera_normals", "relative_positions", "albedo", "inpaint_mask"]
    ckpt_path = resolve_checkpoint(object_name, ckpt_dir=f"{data_dir}/ckpts", explicit=args.ckpt_path)
    num_in_channels = 17
    model = TextureInpaintStandardModel(num_in_channels, in_channels, ckpt_path, use_fp16=True)
    model.set_attention_proc(SamplewiseAttnProcessor2_0)
    mesh_path = args.mesh_path or f"{data_dir}/meshes/{object_name}/scene.gltf"
    mesh = load_mesh(mesh_path)
    max_angle = args.max_angle if args.max_angle is not None else backproject_angles.get(object_name, 90)
    view_margin = args.view_margin if args.view_margin is not None else view_margins.get(object_name, 40)
    
    expr_name = f"all-fast_a={max_angle}_e={view_margin}_c={max_num_camera}"
    cam_config = CameraConfig(backproject_max_angle=max_angle)
    pipe = TextureCompletionPipeline(model, mesh, camera_args=cam_config, sync_mvd_start_exp=0, 
                                    sync_mvd_end_exp=15, sync_mvd_end=0.4, soft_margin=50, inpaint_overlap=0.0)
    
    cond_view_root = args.cond_view_root or os.path.join(data_dir, "test_cond_views")
    texture_root = args.texture_root or os.path.join(data_dir, "test_textures_sr")
    sample_weight_root = args.sample_weight_root or os.path.join(data_dir, "sample_weights")

    pipe.min_reference_content = args.min_reference_content
    pipe.reference_camera_mode = args.reference_camera_mode
    pipe.reference_camera_dist = args.reference_camera_dist
    pipe.max_inpaint_patches = args.max_inpaint_patches

    custom_sampling_weights_fp = os.path.join(sample_weight_root, f"{object_name}-sample_weight.pt")
    if os.path.exists(custom_sampling_weights_fp):
        custom_sampling_weights = torch.load(custom_sampling_weights_fp)
    else:
        custom_sampling_weights = None

    reference_view_path = os.path.join(cond_view_root, object_name, f"view{view_id}.basecolor.png")
    reference_texture_path = os.path.join(texture_root, object_name, f"view{view_id}.png")
    pipe.set_reference(reference_texture_path, reference_view_path)

    pipe.set_logging(f"{expr_dir}/{object_name}/view{view_id}/{expr_name}")
    pipe.camera_cache_dir = f"{expr_dir}/{object_name}/camera_cache"
    os.makedirs(pipe.camera_cache_dir, exist_ok=True)
    pipe.presample_camera_path = f"{pipe.camera_cache_dir}/presample_inpaint_cameras_custom_weight.pt"
    
    guidance_texture_path = args.guidance_texture_path or (
        f"{expr_dir}/{object_name}/view{view_id}/"
        f"syncmvd=0.4_a={max_angle}_e={view_margin}_c={args.guidance_num_cameras}/texture/uv-final.png"
    )
    
    print(f"Starting completion for {object_name} view {view_id} max_angle {max_angle} view_margin {view_margin}")
    pipe.complete(16, from_scratch=True, use_nnfm=True, use_color_ref=True, min_num_camera=min_num_camera, 
                 max_num_camera=max_num_camera, custom_sampling_weights=custom_sampling_weights,
                 log_model=args.log, log_final_texture_only=True, color_correction=True,
                 view_margin=view_margin, seed=args.seed, guidance_path=guidance_texture_path,
                 t_forward=2, t_backward=2)
    
    pipe.render_turnaround_video("turnaround.mp4")
    pipe.save_completion_video("completion.mp4")
