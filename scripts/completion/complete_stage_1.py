# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
import math
import copy
import argparse
from tqdm import tqdm

import torch, torchvision
from diffusers import DDPMScheduler

import kaolin
import gloss
from gloss.utils.parser import str2bool
from gloss.utils.paths import get_data_dir, resolve_checkpoint
from gloss.utils.single_view import get_valid_faces, get_valid_faces_from_texture
from gloss.inpaint.completion_utils import *
from gloss.utils.render_fast import custom_mesh_batched_render
from gloss.utils.kaolin_utils import load_mesh
from gloss.utils.nnfm_loss import NNFMLoss, nn_feat_replace, cos_loss
from gloss.utils import reclaim_cuda_memory
from gloss.model.attention import SamplewiseAttnProcessor2_0
from gloss.inpaint.blend import alpha_blend, laplacian_blend
from gloss.inpaint.camera import turntable_camera_strategy
from gloss.inpaint.models import *
from gloss.inpaint.syncmvd_utils import composite_rendered_view
from gloss.inpaint.inpaint_utils import dilate_nonblack_pool
from gloss.logging.logging import default_log_setup
from gloss.data.render_dataloader import get_lighting_from_cam, LocalCameraExtrinsicsSampler

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
default_log_setup(logging.DEBUG)


class TextureCompletionPipeline(object):
    def __init__(self, model, mesh, texture_res=(2048, 2048), camera_args=type[CameraConfig], blend='alpha'):
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
        self.reference_content_fallback_k = 8

        # manual mode
        self.reference_patches = []

        if blend == 'laplacian':
            self.blend_fn = laplacian_blend
        else:
            self.blend_fn = alpha_blend


    def backproject(self, camera, view, latent=False, margin=False, return_face_idx=False):
        if latent:
            texture_map = self.latent_texture
        else:
            texture_map = self.curr_texture.squeeze(0).permute(1, 2, 0).contiguous()
        
        return backproject_view_to_texture(
            camera, view, self.mesh, texture_map, self.camera_config, self.device,
            latent=latent, margin=margin, view_margin=getattr(self, 'view_margin', 0),
            return_face_idx=return_face_idx
        )

    def update_texture(self, texture, mask, from_scratch=False):
        self.curr_texture, self.mask = self.blend_fn(self.curr_texture, self.mask, texture, mask, 1.0)
        if not from_scratch:
            self.curr_texture, _ = self.blend_fn(self.init_texture, self.init_mask, self.curr_texture, self.mask, 1.0)
        self.update_mesh_material()

    def get_camera_from_face(self, face_idx, u=2/3, v=1/2):
        return get_camera_from_face(self.mesh, face_idx, self.camera_config, self.device, u, v)

    def get_face_uv_pixel_counts(self, visualize=False):
        return get_face_uv_pixel_counts(self.mesh, self.h, self.w, visualize)

    def check_camera_coverage(self, camera):
        return check_camera_coverage(camera, self.mesh, self.curr_texture, self.camera_config, self.device, 
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

    # ------ Manual mode functions end ------
    @torch.no_grad()
    def complete(
        self,
        batch_size,
        num_cond_views=None,
        num_references=200,
        paint_region=None,
        from_scratch=False,
        use_nnfm=False,
        custom_sampling_weights=None,
        debug=False,
        log_model=False,
        log_final_texture_only=True,
        seed=0,
        num_avg_camera=0,
        view_margin=10
    ):
        #self.camera_logger = CameraLogger(self.h, self.w, self.camera_selection_info_dir, self.mesh.faces.device)
        # 1. Initialize texture and mask
        # latent texture resolution same as SyncMVD
        self.latent_texture = torch.normal(0, 1, (4, 512, 512), device=self.device).permute(1, 2, 0)

        assert self.inpaint_texture is not None, "need to run set_reference first"
        if from_scratch:
            self.curr_texture = torch.ones(1, 4, self.h, self.w).cuda() * -1
            self.mask = torch.zeros(1, 1, self.h, self.w).cuda()
        else:
            self.curr_texture, self.mask = self.init_texture, self.init_mask
        self.update_mesh_material()
        
        self.view_margin=view_margin

        # 2. Initialize mesh face sampling params
        torch.manual_seed(seed)
        if paint_region is not None:
            albedo, selection_mask = self._load_texture_image(paint_region)
            select_face_ids = get_valid_faces_from_texture(self.mesh, albedo, all_filled=False)
            valid_face_ids, pixel_counts_per_face, filled_mask = self.get_face_uv_pixel_counts()
            all_counts = torch.zeros(self.mesh.faces.shape[0]).long().to(self.device)
            all_counts[valid_face_ids] = pixel_counts_per_face
            sampling_weights = torch.zeros(self.mesh.faces.shape[0]).to(self.device)
            face_areas = kaolin.ops.mesh.face_areas(self.mesh.vertices.unsqueeze(0), self.mesh.faces).squeeze(0)
            sampling_weights[select_face_ids] = face_areas[select_face_ids]
        else:
            sampling_weights = torch.ones(self.mesh.faces.shape[0]).to(self.device)
            if custom_sampling_weights is not None:
                sampling_weights += custom_sampling_weights
            sampling_weights += kaolin.ops.mesh.face_areas(self.mesh.vertices.unsqueeze(0), self.mesh.faces).squeeze(0) * 1000
            valid_face_ids, pixel_counts_per_face, filled_mask = self.get_face_uv_pixel_counts()
            all_counts = torch.zeros(self.mesh.faces.shape[0]).long().to(self.device)
            all_counts[valid_face_ids] = pixel_counts_per_face
        current_counts = torch.zeros_like(all_counts)

        # 3. Get reference views
        if not debug:
            self.precompute_references(num_references)

        if num_cond_views is None:
            num_cond_views = batch_size // 2

        # 4. Sample cameras
        camera_generated = os.path.exists(os.path.join(self.camera_cache_dir, f"presample_inpaint_cameras_num={num_avg_camera}.pt"))
        if not camera_generated:
            # random sample camera, then cover the texture map
            sampling_weights = torch.ones(self.mesh.faces.shape[0]).to(self.device)
            sampling_weights += kaolin.ops.mesh.face_areas(self.mesh.vertices.unsqueeze(0), self.mesh.faces).squeeze(0) * 1000
            uncovered_faces = torch.ones(self.mesh.faces.shape[0]).to(self.device)
            valid_face_ids, pixel_counts_per_face, filled_mask = self.get_face_uv_pixel_counts()
            all_counts = torch.zeros(self.mesh.faces.shape[0]).long().to(self.device)
            all_counts[valid_face_ids] = pixel_counts_per_face
            current_counts = torch.zeros_like(all_counts)
            texture_mask = torch.zeros(1, 1, self.h, self.w).cuda()
            count = num_avg_camera
            presample_inpaint_cameras = {k: [] for k in ["update_area", "camera"]}
            while count > 0:
                faces_to_sample = torch.where(uncovered_faces > -1)[0]
                sampled_index = torch.multinomial(sampling_weights[faces_to_sample], num_samples=1).item()
                face_id = faces_to_sample[sampled_index]
                camera = self.get_camera_from_face(face_id)
                update_mask, tex_face_idx = self.check_camera_coverage(camera)
                if update_mask is None:
                    continue
                update_area = (update_mask == 1) & ~(texture_mask == 1)
                texture_mask[update_area] = 1
                presample_inpaint_cameras["camera"].append(camera)
                presample_inpaint_cameras["update_area"].append((update_mask == 1).cpu())
                uncovered_faces[face_id] = 0
                count -= 1
                ### track face coverage
                update_faces, update_counts = torch.unique(tex_face_idx[update_area[0]], return_counts=True)
                valid_mask = torch.isin(update_faces, valid_face_ids)
                update_faces = update_faces[valid_mask]
                update_counts = update_counts[valid_mask]
                current_counts[update_faces] += update_counts
                filled_faces = torch.where(all_counts[update_faces] == current_counts[update_faces])
                uncovered_faces[filled_faces] = 0
            presample_inpaint_cameras["texture_mask"] = texture_mask.cpu()
            torch.save(presample_inpaint_cameras, os.path.join(self.camera_cache_dir, f"presample_inpaint_cameras_num={num_avg_camera}.pt"))
            presample_inpaint_cameras = presample_inpaint_cameras["camera"]
        else:
            presample_inpaint_cameras = torch.load(os.path.join(self.camera_cache_dir, f"presample_inpaint_cameras_num={num_avg_camera}.pt"))["camera"]
            texture_mask = torch.load(os.path.join(self.camera_cache_dir, f"presample_inpaint_cameras_num={num_avg_camera}.pt"))["texture_mask"].to(self.device)
        num_inpaint_cameras = len(presample_inpaint_cameras)
        print(f"Using a total of {num_inpaint_cameras} camera")

        # 5. Generate views
        inpaint_cameras_per_batch = batch_size - num_cond_views
        num_batches = int(math.ceil(num_inpaint_cameras / inpaint_cameras_per_batch))
        num_inference_steps = 20
        num_timesteps = self.inpaint_model.noise_scheduler.config.num_train_timesteps
        self.inpaint_model.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.inpaint_model.noise_scheduler.timesteps
        bg_latent = self.inpaint_model.background_latent(256).float()
        scheduler_config = dict(self.inpaint_model.noise_scheduler.config)
        scheduler_config["prediction_type"] = "sample"
        # used to update the latent texture
        latent_noise_scheduler = DDPMScheduler.from_config(scheduler_config)
        latent_noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        multiview_diffusion_end = 0.4
        exp_start, exp_end = 0, 15

        print("Multiview timesteps", timesteps > (1 - multiview_diffusion_end) * num_timesteps)

        # NOTE: first iteration always takes much longer
        for j, t in tqdm(enumerate(timesteps)):
            if not debug:
                combined_texture, total_weights = None, None
                current_exp = ((exp_end - exp_start) * j / num_inference_steps) + exp_start
                do_multi_view_update = t > (1 - multiview_diffusion_end) * num_timesteps
                for b in range(num_batches):
                    start_idx = b * inpaint_cameras_per_batch
                    end_idx = min(start_idx + inpaint_cameras_per_batch, num_inpaint_cameras)

                    # 5.1 Prepare cond inputs
                    if j == 0:
                        inpaint_cameras = [presample_inpaint_cameras[cam_idx].to(self.device)
                                           for cam_idx in range(start_idx, end_idx)]
                        inpaint_data, ref_cameras = self.get_inpaint_batch(inpaint_cameras, num_cond_views, use_nnfm)
                        batch_data = {
                            "cameras": kaolin.render.camera.Camera.cat(ref_cameras + inpaint_cameras),
                            "cond_input": self.inpaint_model.process_input(inpaint_data)
                        }
                    else:
                        batch_data = torch.load(os.path.join(self.cache_dir, f"batch{b}.pt"))

                    actual_batch_size = len(batch_data["cond_input"])

                    # 5.2 Prepare latents
                    if do_multi_view_update:
                        batch_latent_cameras = batch_data["cameras"].to(self.device)
                        batch_latent_cameras.width = 32
                        batch_latent_cameras.height = 32
                        r = custom_mesh_batched_render(batch_latent_cameras, self.mesh, self.latent_texture,
                                                       requires_positions=False, process_as_albedo=False, backend="cuda")
                        fg_latents = r["textured"].permute(0, 3, 1, 2)
                        fg_masks = r["mask"].permute(0, 3, 1, 2) / 2 + 0.5

                        if j == 0:
                            init_latents = torch.randn((actual_batch_size, 4, 32, 32)).to(self.device)
                            latents = composite_rendered_view(self.inpaint_model.noise_scheduler, init_latents, fg_latents,
                                                              fg_masks, t+1)
                            batch_data["latent"] = latents
                        else:
                            prev_t = timesteps[j-1]
                            batch_bg_latents = bg_latent.repeat(actual_batch_size, 1, 1, 1)
                            batch_data["latent"] = composite_rendered_view(self.inpaint_model.noise_scheduler, batch_bg_latents,
                                                                           fg_latents, fg_masks, prev_t)
                        del r, fg_latents, fg_masks

                    # 5.3 Denoising step
                    batch_data["latent"] = batch_data["latent"].to(self.device)
                    noise_pred = self.inpaint_model.step(
                        batch_data["cond_input"].to(self.device), batch_data["latent"], t)

                    latent, latent_image = self.inpaint_model.noise_scheduler.step(noise_pred, t, batch_data["latent"],
                                                                                   return_dict=False)
                    batch_data["latent"] = latent.cpu()
                    batch_data["latent_image"] = latent_image.cpu()

                    # 5.4 Backproject latent to uv space
                    if do_multi_view_update:
                        # backproject and combine using cos angle weights
                        for cam_id in range(actual_batch_size):
                            # NOTE: this also includes the latents from reference views
                            cam = batch_latent_cameras[cam_id]
                            orig_tex, tex_mask, cos_weights = self.backproject(
                                cam, latent_image[cam_id:cam_id+1].permute(0, 2, 3, 1), latent=True)
                            # need to fill the weights and the texture
                            if cos_weights is not None:
                                if current_exp > 0:
                                    min_value = 1e-6**(1/current_exp)
                                    cos_weights[cos_weights <= min_value] = min_value  # deal with decimal precision issue
                                weights = tex_mask * cos_weights ** current_exp
                                if combined_texture is None:
                                    total_weights = weights
                                    combined_texture = orig_tex[:, :4] * weights
                                else:
                                    total_weights += weights
                                    combined_texture += orig_tex[:, :4] * weights
                                del orig_tex, weights
                            reclaim_cuda_memory()

                    torch.save(batch_data, os.path.join(self.cache_dir, f"batch{b}.pt"))
                    del noise_pred, batch_data
                    reclaim_cuda_memory()

                # 5.5 Update latent texture
                if do_multi_view_update:
                    combined_texture /= total_weights + 1E-8
                    # noise to t-1
                    sample = self.latent_texture.unsqueeze(0).permute(0, 3, 1, 2)
                    prev_tex = latent_noise_scheduler.step(combined_texture, t, sample, return_dict=False)[0]
                    self.latent_texture = prev_tex[0].permute(1, 2, 0)
                    del combined_texture, total_weights
                    reclaim_cuda_memory()

            # 5.6 Decode view latents, combine and bake rgb texture
            if t == 1 or (j % 1 == 0 and not log_final_texture_only) or debug:
                rgb_texture, total_weights = None, None
                for b in range(num_batches):
                    start_idx = b * inpaint_cameras_per_batch
                    end_idx = min(start_idx + inpaint_cameras_per_batch, num_inpaint_cameras)
                    inpaint_cameras = [presample_inpaint_cameras[cam_idx].to(self.device)
                                       for cam_idx in range(start_idx, end_idx)]
                    if not debug:
                        # Even though we do not use reference views here, the reference view latents are already fused with the rest of the views
                        if t == 1:
                            batch_data = torch.load(os.path.join(self.cache_dir, f"batch{b}.pt"))["latent"]
                            inpaint_latents = batch_data[-num_cond_views:].to(self.device)  # remove reference views
                        else:
                            batch_data = torch.load(os.path.join(self.cache_dir, f"batch{b}.pt"))["latent_image"]
                            inpaint_latents = batch_data[-num_cond_views:].to(self.device)
                        if self.inpaint_model.use_fp16:
                            inpaint_latents = inpaint_latents.to(torch.float16)
                        view_decoded = self.inpaint_model.pipe.vae.decode(
                            inpaint_latents / self.inpaint_model.pipe.vae.config.scaling_factor, return_dict=False)[0]
                        view_decoded = self.inpaint_model.pipe.image_processor.postprocess(view_decoded, 'pt')
                        if log_model:
                            torchvision.utils.save_image(
                                view_decoded,
                                os.path.join(self.model_debug_dir, f"log-{j}-view{b}.png"),
                            )
                        del inpaint_latents
                        reclaim_cuda_memory()
                    else:
                        view_decoded = torch.ones((len(inpaint_cameras), 3, 256, 256)).to(self.device)
                    for cam_id in range(len(inpaint_cameras)):
                        cam = inpaint_cameras[cam_id]
                        rgb_tex, tex_mask, cos_weights = self.backproject(cam, view_decoded[cam_id:cam_id + 1].permute(0, 2, 3, 1), margin=True)
                        if cos_weights is not None:
                            min_value = 1e-6 ** (1 / exp_end)
                            cos_weights[cos_weights <= min_value] = min_value  # deal with decimal precision issue
                            weights = tex_mask * cos_weights ** exp_end
                            if rgb_texture is None:
                                total_weights = weights
                                rgb_texture = rgb_tex * weights
                            else:
                                total_weights += weights
                                rgb_texture += rgb_tex * weights
                            del rgb_tex, weights, tex_mask, cos_weights
                            reclaim_cuda_memory()

                rgb_texture /= total_weights + 1E-8
                self.curr_texture = rgb_texture
                self.log_texture(f"uv-{j}.png")
                self.update_mesh_material()

                if debug:
                    break
        
        self.update_mesh_material()
        self.log_texture(f"uv-final-raw.png")
        incomplete_mask = (filled_mask.unsqueeze(-1).float() - texture_mask.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        curr_texture = dilate_nonblack_pool(self.curr_texture, kernel_size=5, inpaint_mask=incomplete_mask, iterations=1000, mode='avg')

        self.curr_texture = curr_texture
        self.update_mesh_material()
        self.log_texture(f"uv-final-inpaint.png")
        
        curr_texture = dilate_nonblack_pool(curr_texture, kernel_size=5, iterations=3, mode='avg')
        self.curr_texture = curr_texture.to(dtype=self.curr_texture.dtype, device=self.curr_texture.device)
        self.update_mesh_material()
        self.log_texture(f"uv-final.png")

        # delete the cache dir
        import shutil
        shutil.rmtree(self.cache_dir)

    def log_texture(self, fn):
        assert os.path.exists(self.save_texture_dir)
        fp = os.path.join(self.save_texture_dir, fn)
        self.uv_logs.append(self.curr_texture.detach().cpu())
        torchvision.utils.save_image(self.curr_texture, fp)

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
        logging_dirs = setup_logging_directories(save_dir)
        self.uv_logs = logging_dirs['uv_logs']
        self.save_dir = save_dir
        self.save_texture_dir = logging_dirs['save_texture_dir']
        self.model_debug_dir = logging_dirs['model_debug_dir']
        self.cache_dir = logging_dirs['cache_dir']
        self.camera_selection_info_dir = logging_dirs['camera_selection_info_dir']

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

    def precompute_references(self, num_references=200):
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
        """Reference candidates carrying at least ``min_reference_content`` texture."""
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

    def get_inpaint_batch(self, inpaint_cameras, num_cond_views, use_nnfm=False):
        current_material_map = self.curr_texture.squeeze(0).permute(1, 2, 0).contiguous()
        batched_cameras = kaolin.render.camera.Camera.cat(inpaint_cameras)
        normal_map = self.mesh.materials[0].hwc().normals_texture
        normal_map = torchvision.transforms.Resize((self.h, self.w))(normal_map.permute(2, 0, 1)).permute(1, 2, 0)
        r = custom_mesh_batched_render(batched_cameras, self.mesh, current_material_map, normal_map,
                                       requires_positions=True, process_as_albedo=False, backend="cuda")
        r['albedo'] = r['textured'][..., :3] * 2 - 1
        r['albedo_alpha'] = r['textured'][..., 3:]  # already -1, 1
        if use_nnfm:
            nnfm_loss_fn = NNFMLoss(self.device)
            feats = nnfm_loss_fn.get_feats(r['camera_normals'].permute(0, 3, 1, 2) / 2.0 + 0.5, [11, 13, 15])
            feats = torch.cat(feats, 1)
            del nnfm_loss_fn

        reference_patches = []
        inpaint_patches = []
        reference_cameras = []
        eligible_idx = self._eligible_reference_indices()
        eligible_mask = torch.zeros(self._references['albedo'].shape[0], dtype=torch.bool)
        eligible_mask[eligible_idx] = True

        def pick_random_reference():
            return eligible_idx[torch.randint(len(eligible_idx), ())]

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

        for i in range(len(inpaint_cameras)):
            if use_nnfm:
                x_feats = feats[i:i+1]
                nnfm_loss = []
                for s_feats in self._references["features"]:
                    target_feats = nn_feat_replace(x_feats, s_feats[None].to(self.device))
                    nnfm_loss.append(cos_loss(x_feats, target_feats))
                stacked_nnfm = torch.stack(nnfm_loss)
                matched_ref = torch.argmin(stacked_nnfm.masked_fill(
                    ~eligible_mask.to(stacked_nnfm.device), float('inf')))
            else:
                matched_ref = pick_random_reference()
            input_data = {k: v[i:i+1] for k, v in r.items()}
            inpaint_patch = get_input_data(input_data)
            ref_data = {k: v[matched_ref:matched_ref+1].to(self.device) for k, v in self._references.items()}
            reference_patch = get_input_data(ref_data)
            inpaint_patches.append(inpaint_patch)
            reference_patches.append(reference_patch)
            reference_cameras.append(self._reference_cameras[matched_ref])

        # fill the rest with random sampled patches
        while len(reference_patches) < num_cond_views:
            i = pick_random_reference()
            ref_data = {k: v[i:i + 1].to(self.device) for k, v in self._references.items()}
            reference_patch = get_input_data(ref_data)
            reference_patches.append(reference_patch)
            reference_cameras.append(self._reference_cameras[i])

        return reference_patches + inpaint_patches, reference_cameras

    def update_mesh_material(self):
        curr_materials = copy.deepcopy(self.mesh.materials)
        curr_materials[0].diffuse_texture = self.curr_texture.squeeze(0).permute(1, 2, 0).contiguous()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--object_name", type=str, nargs="?", const="cabbage", default="cabbage", help="Name of the object to run on")
    parser.add_argument("--view_id", type=int, nargs="?", const=0, default=0, help="Singe View ID index to run on")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data root (default: $GLOSS_DATA_DIR)")
    parser.add_argument("--expr_dir", type=str, default="expr",
                        help="Directory to save results (default: ./expr)")
    parser.add_argument("--max_angle", type=int, default=None,
                        help="Max backprojection angle; default: per-object table, else 90")
    parser.add_argument("--view_margin", type=int, default=None,
                        help="Reference view margin; default: per-object table, else 40")
    parser.add_argument("--num_cameras", type=int, nargs="?", const=600, default=600, help="Number of average cameras to use")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Optional explicit checkpoint path")
    parser.add_argument("--cond_view_root", type=str, default=None, help="Optional root containing <object_name>/viewXXXX.basecolor.png")
    parser.add_argument("--texture_root", type=str, default=None, help="Optional root containing <object_name>/viewXXXX.png")
    parser.add_argument("--sample_weight_root", type=str, default=None, help="Optional root containing <object_name>-sample_weight.pt")
    parser.add_argument("--mesh_path", type=str, default=None, help="Optional explicit mesh path (overrides <data_dir>/meshes/<object_name>/scene.gltf)")
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
    parser.add_argument("--log", type=str2bool, nargs="?", const=False, default=False, help="Whether to log the model")
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
    
    num_avg_camera = args.num_cameras
    data_dir = args.data_dir or str(get_data_dir())
    expr_dir = args.expr_dir
    object_name = args.object_name
    view_id = "%04d" % int(args.view_id)
    
    in_channels = ["camera_normals", "relative_positions", "albedo", "inpaint_mask"]
    ckpt_path = resolve_checkpoint(object_name, ckpt_dir=f"{data_dir}/ckpts", explicit=args.ckpt_path)
    num_in_channels = 17
    model = TextureInpaintStandardModel(num_in_channels, in_channels, ckpt_path, use_fp16=True)
    model.set_attention_proc(SamplewiseAttnProcessor2_0)
    mesh_path = args.mesh_path or f"{data_dir}/meshes/{object_name}/scene.gltf"
    mesh = load_mesh(mesh_path)
    max_angle = args.max_angle if args.max_angle is not None else backproject_angles.get(object_name, 90)
    view_margin = args.view_margin if args.view_margin is not None else view_margins.get(object_name, 40)

    expr_name = f"syncmvd=0.4_a={max_angle}_e={view_margin}_c={num_avg_camera}"
    cam_config = CameraConfig(backproject_max_angle=max_angle)
    pipe = TextureCompletionPipeline(model, mesh, camera_args=cam_config)
    cond_view_root = args.cond_view_root or os.path.join(data_dir, "test_cond_views")
    texture_root = args.texture_root or os.path.join(data_dir, "test_textures_sr")
    sample_weight_root = args.sample_weight_root or os.path.join(data_dir, "sample_weights")

    pipe.min_reference_content = args.min_reference_content
    pipe.reference_camera_mode = args.reference_camera_mode
    pipe.reference_camera_dist = args.reference_camera_dist

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
    print(f"Starting completion for {object_name} view {view_id} max_angle {max_angle} view_margin {view_margin}")
    pipe.complete(16, from_scratch=True, use_nnfm=False, custom_sampling_weights=custom_sampling_weights,
                log_model=False, log_final_texture_only=True, num_avg_camera=num_avg_camera, view_margin=view_margin)
    pipe.render_turnaround_video("turnaround.mp4")
    pipe.save_completion_video("completion.mp4")
                               
