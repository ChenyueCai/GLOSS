from gloss_interactive.brush import *
import time
import kaolin.render.easy_render as easy_render
from gloss.utils.single_view import backproject_render
import torch
import copy

class Timer:
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, *args):
        print(f"{(time.time() - self.start)} s")
    
    
from gloss_interactive.utils import *
from gloss_interactive.session import *
from gloss_interactive.paths import INTERACTIVE_MESH_CONFIG
from gloss.utils.paths import get_session_dir, resolve_path


# Example usage:
if __name__ == "__main__":
    # Load mesh

    config_fp = INTERACTIVE_MESH_CONFIG
    brush_library = ReferenceBrushLibrary.from_config(config_fp)
    session_dir = str(get_session_dir())
    session_name = "koi-fish-base"
    gloss_session = GlossSession.load(session_name, session_dir, brush_library)
    
    mesh = str(resolve_path("meshes/koi_fish/scene.gltf"))
    mesh = load_mesh(mesh, torch.device("cuda:0"))
    # Define camera config
    camera_config = CameraConfig(
        fov=0.4,
        resolution=256,
        dist=0.75,
        spacing=0.25,
        backproject_pix_count=2,
        backproject_max_angle=90
    )
    brush = brush_library.get_brush_by_name("koi_fish-117-auto")
    mesh_texture = gloss_session.paint_meshes[0].get_texture().cuda()
    
    # Specify target faces to cover (example: first 100 faces)
    target_faces = [15, 23, 27, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 50, 52, 53, 54, 58, 59, 63, 67, 69, 72, 73, 74, 75, 76, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 95, 96, 99, 101]
    # Sample cameras
    camera_config = CameraConfig()
    tar_cameras = sample_cameras_for_faces(mesh, target_faces, camera_config)["cameras"]
    batch_size = 8
    num_tar_camera = len(tar_cameras)
    if num_tar_camera > 4:
        print("Choose a smaller region")
    # 2.0 inference with cameras
    # tar_cameras = sample_cameras(mesh, target_faces)
    
    outputs = brush.apply_stroke_to_views(mesh, mesh_texture, tar_cameras)
    for i in range(num_tar_camera):
        texture, mask, _ = backproject(mesh, outputs[i:i+1].permute(0, 2, 3, 1), tar_cameras[i], camera_config,
                                        h=1024,w=1024,valid_faces=target_faces)
        texture_output = torch.cat([texture, mask], dim=-1).squeeze(0)
        torchvision.utils.save_image(texture_output.unsqueeze(0).permute(0, 3, 1, 2), "test.png")