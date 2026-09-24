import os
import glob
import logging
import math
import time
import torch, torchvision
import torch.nn.functional as F
from PIL import Image, ImageDraw

import kaolin
from kaolin.render.easy_render.mesh import mesh_rasterize_interpolate_nvdiffrast
from kaolin.render.mesh.nvdiffrast_context import nvdiffrast_is_available, default_nvdiffrast_context

if nvdiffrast_is_available():
    import nvdiffrast.torch

import torchvision

from gloss.utils import reclaim_cuda_memory
from gloss.utils.render_fast import mesh_rasterize_interpolate_cuda

import gloss
import gloss.utils.render
from gloss.utils import reclaim_cuda_memory
from gloss.utils.single_view import backproject_render, get_tex_face_idx
from gloss.inpaint.inpaint_utils import dilate_nonblack_pool, expand_mask_soft, composite_inpaint
from gloss.inpaint.color_correction import compute_color_histogram

from kornia.morphology import erosion

logger = logging.getLogger(__name__)


def save_concat_log(log_dir, max_height=192, gap=4, out_name="log-all.png", cols=None, rows=None):
    """Concat every per-channel `log-*.png` in ``log_dir`` into a single
    downsized image for quick visual inspection.

    Tiles are laid out in a grid. Pass ``rows`` to fix the row count (cols are
    computed as ceil(N/rows)); otherwise pass ``cols`` to fix the column count.
    If neither is given, picks a roughly-square grid (ceil(sqrt(N)))."""
    paths = sorted(p for p in glob.glob(os.path.join(log_dir, "log-*.png"))
                   if os.path.basename(p) != out_name)
    if not paths:
        return None
    tiles = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        new_w = max(1, int(round(img.width * max_height / img.height)))
        tiles.append((os.path.basename(p)[len("log-"):-len(".png")],
                      img.resize((new_w, max_height), Image.BILINEAR)))
    n = len(tiles)
    if rows is not None:
        rows = max(1, min(rows, n))
        cols = int(math.ceil(n / rows))
    else:
        if cols is None:
            cols = max(1, int(math.ceil(math.sqrt(n))))
        cols = min(cols, n)
        rows = int(math.ceil(n / cols))
    label_h = 16
    row_h = max_height + label_h
    # row width = widest row across the grid
    row_widths = []
    for r in range(rows):
        row_tiles = tiles[r * cols:(r + 1) * cols]
        w = sum(t.width for _, t in row_tiles) + gap * max(0, len(row_tiles) - 1)
        row_widths.append(w)
    total_w = max(row_widths)
    total_h = row_h * rows + gap * max(0, rows - 1)
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for r in range(rows):
        row_tiles = tiles[r * cols:(r + 1) * cols]
        x = 0
        y = r * (row_h + gap)
        for name, tile in row_tiles:
            canvas.paste(tile, (x, y + label_h))
            draw.text((x + 2, y + 1), name, fill=(0, 0, 0))
            x += tile.width + gap
    out_path = os.path.join(log_dir, out_name)
    canvas.save(out_path)
    return out_path


def texel_scale(h, w, base=1024):
    """Integer factor by which texel-space kernels grow so they cover the same UV
    extent as at ``base``x``base`` (1 at 1K, 4 at 4K)."""
    return max(1, int(round(max(h, w) / base)))


def backproject(mesh, view, camera, camera_config, h, w, valid_faces=None, ero_kernel=15, dilation=True,
                paint_region_img_mask=None):
    """_summary_

    Args:
        mesh (_type_): _description_
        view (_type_): _description_
        camera (_type_): _description_
        camera_config (_type_): _description_
        h (_type_): _description_
        w (_type_): _description_
        valid_faces (_type_, optional): _description_. Defaults to None.
        ero_kernel (int, optional): _description_. Defaults to 15.
        paint_region_img_mask (torch.Tensor, optional): image-space mask in shape
            (1, H, W, 1) with values in [0, 1] where 1 marks pixels the model just
            painted (i.e. the eroded ring + previously-unknown region). When supplied
            the backprojection update mask is restricted to these pixels so the
            previously-known interior of the texture is left untouched.

    Returns:
        _type_: B H W C TENSORS
    """
    begin = time.time()
    start = time.time()
    device = mesh.faces.device
    camera = camera.to(device)
    pad = 1
    res = camera_config.resolution
    update_mask = torch.ones((1, res, res, 1), device=device, dtype=torch.float32) * -1
    update_mask[0, pad:res - pad, pad:res - pad] = 1
    lighting = kaolin.render.easy_render.default_lighting().cuda()
    render_res = gloss.utils.render.render_all_features(camera, mesh, lighting)
    print("Render time:", time.time() - start)

    start = time.time()
    fg_mask = 1 - (torch.abs(render_res["geo_camera_normals"]) < 0.01).all(dim=-1, keepdim=True).float()
    if (fg_mask == 0).sum() > 0:
        kernel = torch.ones(ero_kernel, ero_kernel).cuda()
        fg_mask = erosion(fg_mask.permute(0, 3, 1, 2), kernel).permute(0, 2, 3, 1)
        update_mask = (update_mask == 1) & (fg_mask == 1)
        update_mask = update_mask.float() * 2 - 1
    print("Erosion time:", time.time() - start)

    if paint_region_img_mask is not None:
        prm = paint_region_img_mask.to(device).float()
        if prm.dim() == 3:
            prm = prm.unsqueeze(0)
        update_mask = ((update_mask > 0) & (prm > 0.5)).float() * 2 - 1
    
    start = time.time()
    # Upsample the model's patch before backprojection so each view pixel covers
    # about as many texels at 4K as it does at 1K (256 px view -> 1024 px at 4096^2).
    # Colours use bicubic; the {-1,+1} write mask stays hard (nearest).
    # backproject_render(upscale=True) then lifts face_idx / normals to match.
    up = texel_scale(h, w)
    if up > 1:
        size = (view.shape[1] * up, view.shape[2] * up)
        view = torch.nn.functional.interpolate(view.permute(0, 3, 1, 2), size=size, mode='bicubic',
                                               align_corners=False).clamp(0.0, 1.0).permute(0, 2, 3, 1)
        update_mask = torch.nn.functional.interpolate(update_mask.permute(0, 3, 1, 2), size=size,
                                                      mode='nearest').permute(0, 2, 3, 1)
    view = torch.cat([view, update_mask], dim=-1)
    # RGB in [0, 1] (model output) + alpha in {-1, +1} (write/skip mask).
    # mask is extracted via /2+0.5 before the [0, 1] clip below.
    render_res['albedo'] = view
    backprojection, mask, tex_face_idx = backproject_render(mesh, camera, render_res, ['albedo'],
                                                            h, w, min_pixel_count=0, #self.camera_config.backproject_pix_count,
                                                            max_angle_deviation=math.pi*camera_config.backproject_max_angle*2/180.0,
                                                            return_face_idx=True, sample_mode='bilinear',
                                                            upscale=up > 1)
    print("Backproject time:", time.time() - start)
    start = time.time()
    # compound the mask with valid faces mask
    if tex_face_idx is None:
        return None, None, None
    mask = backprojection['albedo'][..., 3:4] / 2 + 0.5
    if valid_faces is not None:
        valid_face_mask = torch.isin(tex_face_idx, torch.tensor(valid_faces).to(device)).int()
        mask = mask * (valid_face_mask.permute(1, 2, 0))
    
    # dilation for seams
    #torchvision.utils.save_image(backprojection['albedo'].clip(0.0, 1.0).permute(0, 3, 1, 2), "before dilation.png")
    if dilation:
        # Texels outside every UV face (the gutter) get zero rasterizer coordinates
        # and therefore sample one arbitrary view pixel (the image corner); the old
        # mask dilation then wrote that colour as a uniform rim around each island,
        # which reads as a seam (4x thinner in UV at 4K, hence visible there).
        # Blank the gutter first so the bleed takes real neighbouring island colours,
        # bleed only into the gutter, and scale the reach with texture resolution so
        # it covers the same UV extent at 4K as the 6 texels tuned at 1K.
        gutter = (tex_face_idx == -1).permute(1, 2, 0)                     # (H, W, 1)
        albedo = backprojection['albedo'].clip(0.0, 1.0)
        albedo = torch.where(gutter, torch.zeros_like(albedo), albedo)
        mask = torch.where(gutter, torch.zeros_like(mask), mask)
        # Stage 1: the original short bleed everywhere (fills pinholes / isolated
        # unpainted texels inside islands, as it always did).
        albedo_s = dilate_nonblack_pool(albedo.permute(0, 3, 1, 2), kernel_size=5, iterations=3).permute(0, 2, 3, 1)
        mask_s = dilate_nonblack_pool(mask.permute(0, 3, 1, 2), kernel_size=5, iterations=3).permute(0, 2, 3, 1)
        # Stage 2: extend the bleed into the gutter only, scaled with resolution.
        extra = 3 * texel_scale(h, w) - 3
        if extra > 0:
            albedo_d = dilate_nonblack_pool(albedo_s.permute(0, 3, 1, 2), kernel_size=5, iterations=extra).permute(0, 2, 3, 1)
            mask_d = dilate_nonblack_pool(mask_s.permute(0, 3, 1, 2), kernel_size=5, iterations=extra).permute(0, 2, 3, 1)
        else:
            albedo_d, mask_d = albedo_s, mask_s
        backprojection['albedo'] = torch.where(gutter, albedo_d, albedo_s)
        mask = torch.where(gutter, mask_d, mask_s)
    print("Dilation time:", time.time() - start)  
    print("Total backproject time:", time.time() - begin)
    return backprojection['albedo'][..., :3], mask, tex_face_idx


def get_face_uv_pixel_counts(mesh, h, w, mask=None):

    """Get pixel counts for each face in UV space."""
    tex_face_idx = get_tex_face_idx(mesh, h, w)
    if mask is not None:
        mask = F.interpolate(mask.unsqueeze(0),size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
        binary_mask = (mask > 0.8)
        tex_face_idx[binary_mask] = -1
    
    all_face_ids, counts = torch.unique(tex_face_idx[tex_face_idx != -1], return_counts=True)
    filled_mask = tex_face_idx != -1
    
    return all_face_ids, counts, filled_mask


def get_camera_from_face(mesh, face_idx, camera_config, device, u=2/3, v=1/2):
    """Generate camera positioned to view a specific face."""
    vertices = mesh.face_vertices[face_idx]
    normals = mesh.face_normals[face_idx]
    w0 = 1 - u
    w1 = u * (1 - v)
    w2 = u * v

    points = w0 * vertices[0] + w1 * vertices[1] + w2 * vertices[2]
    normals = w0 * normals[0] + w1 * normals[1] + w2 * normals[2]

    cam_pos = points.unsqueeze(0)
    cam_normals = normals.unsqueeze(0)

    up = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    eye = cam_pos + torch.nn.functional.normalize(cam_normals) * camera_config.dist
    lookat = cam_pos
    camera = torch.cat([eye, lookat, up], dim=1)
    camera = gloss.utils.render.make_camera_from_extr_intr(
        camera[0], camera_config.fov, 
        resolution=camera_config.resolution, device=device
    )
    return camera


def sample_face_point_normal(mesh, face_idx, u=2/3, v=1/2):
    vertices = mesh.face_vertices[face_idx]
    normals = mesh.face_normals[face_idx]
    w0 = 1 - u
    w1 = u * (1 - v)
    w2 = u * v

    points = w0 * vertices[0] + w1 * vertices[1] + w2 * vertices[2]
    normals = w0 * normals[0] + w1 * normals[1] + w2 * normals[2]
    return points, normals




def _get_rendered_face_idx(mesh, cameras, backend=None, compute_normals=False):
    if backend is None:
        backend = "nvdiffrast" if nvdiffrast_is_available() else "cuda"

    if backend == "nvdiffrast":
        nvdiffrast_context = default_nvdiffrast_context(device=mesh.vertices.device, raise_error=True)

        rast_out = mesh_rasterize_interpolate_nvdiffrast(
            mesh, cameras, nvdiffrast_context,
            normals_required=compute_normals, uvs_required=False, tangents_required=False, features_required=False)
    elif backend == "cuda":
        rast_out = mesh_rasterize_interpolate_cuda(mesh, cameras,
            normals_required=compute_normals, uvs_required=False, tangents_required=False, features_required=False)
    else:
        raise RuntimeError(f'unknown backend {backend}')

    face_idx = rast_out[0]
    im_normals = rast_out[1] if compute_normals else None

    vertices_camera = cameras.extrinsics.transform(mesh.vertices)
    vertices_image = cameras.intrinsics.transform(vertices_camera)

    face_vertices_camera = kaolin.ops.mesh.index_vertices_by_faces(vertices_camera, mesh.faces)
    face_vertices_image = kaolin.ops.mesh.index_vertices_by_faces(vertices_image, mesh.faces)[..., :2]

    # C x F x 3 x 2
    faces_within_viewport = torch.logical_and(face_vertices_image >= -1, face_vertices_image <= 1).sum(dim=-1).sum(dim=-1)
    faces_within_viewport = faces_within_viewport >= 6  # all vertices within

    return face_idx, faces_within_viewport, im_normals


#####################################
#   Target Camera Selection         #
#####################################


def camera_from_face_normal(mesh, face_idx, camera_config, u=1/3, v=1/3, up_vidx=0):
    """Place a camera looking straight down one face's interpolated normal.

    The anchor point is the barycentric ``(u, v)`` sample of the face; the eye
    sits ``camera_config.dist`` along the interpolated normal from it, and
    ``up`` points at vertex ``up_vidx`` so the roll is deterministic.

    This is the single definition of a "per-face camera" -- both the brush
    stroke path (``ReferenceBrush.create_test_camera``) and target-camera
    selection go through it, so the cameras a fill considers are exactly the
    cameras it can end up using.
    """
    device = mesh.vertices.device
    vertices = mesh.face_vertices[face_idx]
    normals = mesh.face_normals[face_idx]

    cam_pos = u * vertices[0] + v * vertices[1] + (1 - u - v) * vertices[2]
    normal = u * normals[0] + v * normals[1] + (1 - u - v) * normals[2]

    up_vertex = mesh.vertices[mesh.faces[face_idx][up_vidx]]
    up = torch.nn.functional.normalize((up_vertex - cam_pos).unsqueeze(0)).squeeze(0)
    eye = cam_pos + torch.nn.functional.normalize(normal.unsqueeze(0)).squeeze(0) * camera_config.dist

    return kaolin.render.camera.Camera(
        kaolin.render.camera.CameraExtrinsics.from_lookat(
            eye=eye, at=cam_pos, up=up, dtype=torch.float32, device=device),
        kaolin.render.camera.PinholeIntrinsics.from_fov(
            width=camera_config.resolution, height=camera_config.resolution,
            fov=camera_config.fov, device=device, dtype=torch.float32))


def build_face_coverage_matrix(mesh, cameras, batch_size=64, min_face_pixels=3,
                               min_normal_std=0.05, min_coverage_for_normal_filter=0.5,
                               min_unique_faces=3, max_largest_face_pct=0.6):
    """Rasterize ``cameras`` and record which faces each one properly renders.

    A face counts as covered by a camera when it draws more than
    ``min_face_pixels`` pixels *and* the whole triangle lands inside the
    viewport, so a camera is never credited with a face it merely clips.

    Cameras that produced a bad render get an all-False column and can
    therefore never be selected. Three failure modes are screened, in
    precedence order:

    * ``empty`` -- the camera saw no geometry at all.
    * ``few_unique_faces`` -- fewer than ``min_unique_faces`` triangles visible,
      which is what a camera stuck inside the mesh looks like.
    * ``dominant_face`` -- one triangle covers more than
      ``max_largest_face_pct`` of the rendered pixels. Only enforced below 100
      visible triangles so legitimate close-ups survive.
    * ``flat_normals`` -- per-pixel normal stdev below ``min_normal_std`` while
      the rendered area exceeds ``min_coverage_for_normal_filter`` of the
      image. Low-coverage views have low stdev too, but they are not the
      failure being screened for.

    Mirrors the heuristics in ``scripts/debug/camera_debug.py:score_render``.

    Every statistic is computed batch-wide on device: per-(camera, face) pixel
    counts come from one ``scatter_add_``, and the normal stdev from a two-pass
    mean/deviation. Nothing is read back per camera, so a stroke costs a single
    host sync at the end instead of several per candidate.

    Returns:
        tuple[torch.Tensor, dict]: ``(coverage, bad_reasons)`` where coverage is
        a ``(num_faces, num_cameras)`` bool tensor and ``bad_reasons`` counts
        the rejected cameras per failure mode.
    """
    device = mesh.vertices.device
    num_faces = mesh.faces.shape[0]
    coverage = torch.zeros((num_faces, len(cameras)), device=device, dtype=torch.bool)
    # reason codes: 0 good, 1 empty, 2 few_unique_faces, 3 dominant_face, 4 flat_normals
    reason_totals = torch.zeros(5, device=device, dtype=torch.long)

    for bstart in range(0, len(cameras), batch_size):
        cam_batch = kaolin.render.camera.Camera.cat(cameras[bstart:bstart + batch_size])
        face_idx, faces_within_viewport, im_normals = _get_rendered_face_idx(
            mesh, cam_batch, compute_normals=True)
        if face_idx.dim() > 3:
            face_idx = face_idx.squeeze(-1)                      # (B, H, W)
        b, h, w = face_idx.shape

        mask = face_idx >= 0
        flat_mask = mask.flatten(1)                              # (B, H*W)
        n_pixels = flat_mask.sum(1)                              # (B,)

        # Pixels per (camera, face). Background pixels are aimed at face 0 with
        # a zero increment so they contribute nothing.
        counts = torch.zeros((b, num_faces), device=device, dtype=torch.int32)
        counts.scatter_add_(
            1,
            torch.where(mask, face_idx, torch.zeros_like(face_idx)).flatten(1).long(),
            flat_mask.to(torch.int32),
        )

        num_unique = (counts > 0).sum(1)                         # (B,)
        safe_pixels = n_pixels.clamp(min=1)
        largest_pct = counts.max(1).values.float() / safe_pixels.float()
        cov_frac = n_pixels.float() / float(h * w)

        # Two-pass stdev: accumulating sums of squares in fp32 loses too much
        # precision near the 0.05 threshold.
        m = mask.unsqueeze(-1)
        n = safe_pixels.view(b, 1).float()
        mean = (im_normals * m).flatten(1, 2).sum(1) / n         # (B, 3)
        dev = (im_normals - mean.view(b, 1, 1, 3)) * m
        var = dev.pow(2).flatten(1, 2).sum(1) / (n - 1).clamp(min=1)
        normal_std = var.clamp(min=0).sqrt().mean(1)             # (B,)

        reason = torch.zeros(b, device=device, dtype=torch.long)
        def _flag(cond, code):
            return torch.where((reason == 0) & cond, torch.full_like(reason, code), reason)
        reason = _flag(n_pixels == 0, 1)
        reason = _flag(num_unique < min_unique_faces, 2)
        reason = _flag((largest_pct > max_largest_face_pct) & (num_unique < 100), 3)
        reason = _flag((normal_std < min_normal_std)
                       & (cov_frac > min_coverage_for_normal_filter), 4)
        reason_totals += torch.bincount(reason, minlength=5)

        cov = (counts > min_face_pixels) & faces_within_viewport  # (B, F)
        cov &= (reason == 0).unsqueeze(1)
        coverage[:, bstart:bstart + b] = cov.T

    totals = reason_totals.tolist()  # the one host sync per call
    bad_reasons = {'empty': totals[1], 'few_unique_faces': totals[2],
                   'dominant_face': totals[3], 'flat_normals': totals[4]}
    return coverage, bad_reasons


def greedy_camera_cover(target_coverage, weights, max_cameras):
    """Greedy weighted max-coverage: repeatedly take the best-covering camera.

    Pure tensor math, no rendering -- kept separate from camera placement so
    the selection rule itself can be tested directly.

    Picks the camera covering the most target area, zeroes the area it just
    covered, and repeats against what is left. Stops at ``max_cameras`` or as
    soon as no camera would add any new area (so a small selection does not
    return redundant near-duplicate views).

    Args:
        target_coverage: ``(num_targets, num_cameras)`` float/bool tensor;
            nonzero where a camera covers that target face.
        weights: ``(num_targets,)`` per-face area weights.
        max_cameras: hard cap on how many cameras to pick.

    Returns:
        tuple[list[int], float]: chosen camera indices, best first, and the
        total target area still uncovered afterwards.
    """
    remaining = weights.clone().float()
    target_coverage = target_coverage.float()
    chosen = []
    for _ in range(int(max_cameras)):
        gain = (target_coverage * remaining.unsqueeze(1)).sum(dim=0)
        best = int(torch.argmax(gain).item())
        if float(gain[best].item()) <= 0:
            break  # nothing left that any remaining camera can reach
        chosen.append(best)
        remaining[target_coverage[:, best] > 0] = 0
    return chosen, float(remaining.sum().item())


def select_cameras_for_faces(mesh, target_face_ids, camera_config, max_cameras=5,
                             max_candidates=None, face_weights=None):
    """Pick the cameras that cover the painted region, greedily, best first.

    One candidate camera is placed on each target face, looking down that
    face's normal. The candidates are rasterized once, then chosen greedily:
    the camera covering the most target area is picked first, the area it
    covered is subtracted, and each subsequent camera is the one covering the
    most of what is still uncovered -- until ``max_cameras`` are chosen or no
    remaining camera adds any new area.

    Args:
        mesh: kaolin mesh being painted.
        target_face_ids: face indices the user selected for this stroke.
        camera_config: ``CameraConfig`` supplying fov / dist / resolution.
        max_cameras: hard cap on how many cameras to return.
        max_candidates: cap on candidate cameras rasterized. Bigger selections
            are subsampled evenly down to this many anchors so the rasterize
            cost stays bounded regardless of stroke size. Defaults to
            ``max(24, 8 * max_cameras)``: rasterizing candidates dominates the
            cost of a stroke and scales with this number, not with
            ``max_cameras``, so a small ``max_cameras`` should not pay for a
            large candidate pool. The greedy still gets ~8 choices per slot.
        face_weights: optional ``(num_faces,)`` per-face area weights. Pass UV
            pixel counts so "most area" means texture area; ``None`` weights
            every target face equally, making the objective face count.

    Returns:
        list[kaolin.render.camera.Camera]: chosen cameras, highest coverage
        first. Empty only when ``target_face_ids`` is empty.
    """
    if max_candidates is None:
        max_candidates = max(24, 8 * int(max_cameras))

    device = mesh.vertices.device
    targets = torch.as_tensor(list(target_face_ids), device=device, dtype=torch.long).view(-1)
    targets = torch.unique(targets)  # sorted, deduplicated
    if targets.numel() == 0:
        return []

    # One candidate per target face, evenly subsampled when there are too many.
    if targets.numel() > max_candidates:
        idx = torch.linspace(0, targets.numel() - 1, max_candidates, device=device).round().long()
        anchors = targets[torch.unique(idx)]
    else:
        anchors = targets

    start = time.time()
    cameras = [camera_from_face_normal(mesh, int(f), camera_config) for f in anchors]
    coverage, bad_reasons = build_face_coverage_matrix(mesh, cameras)

    if face_weights is None:
        weights = torch.ones(targets.numel(), device=device, dtype=torch.float32)
    else:
        weights = torch.as_tensor(face_weights, device=device, dtype=torch.float32)[targets]

    target_coverage = coverage[targets].float()  # (num_targets, num_candidates)
    total_area = float(weights.sum().item())

    chosen, remaining_area = greedy_camera_cover(target_coverage, weights, max_cameras)
    selected = [cameras[c] for c in chosen]

    if not selected:
        # Every candidate was rejected or covered nothing -- still paint
        # something rather than dropping the stroke on the floor.
        fallback = int(targets[int(torch.argmax(weights).item())].item())
        print(f'[select_cameras_for_faces] no camera covered any target area '
              f'({bad_reasons}); falling back to a single camera on face {fallback}')
        return [camera_from_face_normal(mesh, fallback, camera_config)]

    covered = 1.0 - (remaining_area / total_area if total_area else 0.0)
    print(f'[select_cameras_for_faces] {len(selected)}/{max_cameras} cameras from '
          f'{len(cameras)} candidates cover {covered:.1%} of {targets.numel()} target faces '
          f'(rejected={ {k: v for k, v in bad_reasons.items() if v} }, '
          f'{time.time() - start:.2f}s)')
    return selected
