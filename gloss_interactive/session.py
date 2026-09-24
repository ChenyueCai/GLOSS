import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

from gloss_interactive.brush import ReferenceBrush, ReferenceBrushLibrary
from gloss.utils.paths import relativize, rebase
import torch, torchvision

class PaintSession:
    def __init__(self):
        self.paint_mesh_entries: Dict[str, PaintMesh]= {}  # mesh_name -> PaintMesh

    def add_mesh(self, paint_mesh_name, mesh, texture, texture_file):
        paint_mesh = PaintMesh(paint_mesh_name, mesh, texture, texture_file)
        self.paint_mesh_entries[paint_mesh_name] = paint_mesh
        return paint_mesh
    
    def add_paint_mesh(self, paint_mesh):
        self.paint_mesh_entries[paint_mesh.mesh_name] = paint_mesh

    def get_texture(self, paint_mesh_name):
        return self.paint_mesh_entries[paint_mesh_name].get_texture()
    
    def get_mesh(self, paint_mesh_name):
        return self.paint_mesh_entries[paint_mesh_name].get_mesh()
    
    def update_texture(self, paint_mesh_name, texture, save=True):
        self.paint_mesh_entries[paint_mesh_name].update_texture(texture, save=save)


class PaintMesh:
    """
    Represents a single mesh in a paint session, 
    storing the mesh object and its texture.
    """
    def __init__(self, mesh_name: str, mesh: any, texture:torch.Tensor, texture_file:str):
        self.mesh_name = mesh_name
        self.mesh = mesh       
        self.texture = texture.cuda() # h w c [0, 1]
        self.texture_file = texture_file

    def get_texture(self):
        return self.texture
    
    def get_mesh(self):
        return self.mesh

    def update_texture(self, texture, save):
        self.texture = texture
        if save:
            self.save_texture()
    
    def save_texture(self):
        torchvision.utils.save_image(self.texture.permute(2, 0, 1).unsqueeze(0), self.texture_file)
    
    def to_dict(self, rel_base=None):
        try:
            assert self.texture_file is not None
        except AssertionError:
            print("save the texture first")
        return {
            "mesh_name": self.mesh_name,
            "mesh": self.mesh,
            # PNG path, relative to the session folder when it lives inside it
            "texture_file": relativize(self.texture_file, rel_base)
        }
    
    @staticmethod
    def from_dict(data, rel_base=None):
        texture_file = rebase(data.get("texture_file"), rel_base)
        if rel_base is not None and not os.path.isfile(texture_file):
            # Older sessions stored an absolute path; if the session folder moved or
            # was renamed, fall back to the standard location inside it.
            texture_file = os.path.join(str(rel_base), "texture-log", data["mesh_name"], "texture.png")
        texture = torchvision.io.read_image(texture_file).permute(1, 2, 0) / 255.0
        pm =  PaintMesh(data["mesh_name"], data["mesh"], texture, texture_file)
        return pm


class GlossSession:
    """
    Represents a full painting session.
    Holds metadata, brushes, model, multiple meshes and their textures.
    """
    def __init__(self,
                 session_name: str,
                 session_folder: str,
                 inpaint_models: Optional[Any] = [],
                 meshes:  Optional[Any] = [],
                 brush_rel_base: Optional[str] = None,
                 ):
        self.session_name = session_name
        self.session_dir = session_folder
        self.export_folder = Path(os.path.join(session_folder, session_name))
        # Brush reference dirs in meta.json are written relative to this
        # (the brush library's cache folder).
        self.brush_rel_base = brush_rel_base
        self.inpaint_models = inpaint_models  # handle or model name/path
        self.meshes = meshes
        self.brushes: list[ReferenceBrush] = []  # brush_name → properties
        self.paint_meshes: list[PaintMesh] = []  # mesh_name → PaintMesh
        self.export_folder.mkdir(exist_ok=True)
        if not os.path.exists(os.path.join(self.export_folder, "meta.json")):
            self.export(self.export_folder)

    # -------------------------------------------------------------------------
    # Mesh & Brush Management
    # -------------------------------------------------------------------------
    def add_mesh(self, mesh:str):
        if mesh not in self.meshes:
            self.meshes.append(mesh)
            self.export(self.export_folder)

    def add_inpaint_model(self, inpaint_model:str):
        if inpaint_model not in self.inpaint_models:
            self.inpaint_models.append(inpaint_model)
            self.export(self.export_folder)
        
    def add_brush(self, brush:ReferenceBrush):
        all_brush_name = [b.name for b in self.brushes]
        if brush.name not in all_brush_name:
            self.brushes.append(brush)
            self.export(self.export_folder)
    
    def add_paint_mesh(self, mesh:PaintMesh):
        all_mesh_name = [pm.mesh_name for pm in self.paint_meshes]
        if mesh.mesh_name not in all_mesh_name:
            self.paint_meshes.append(mesh)
            self.export(self.export_folder)

    # -------------------------------------------------------------------------
    # Export / Save
    # -------------------------------------------------------------------------
    def export(self, folder_path: str):
        session_data = {
            "session_name": self.session_name,
            "inpaint_model": self.inpaint_models,
            "meshes": self.meshes, # string
            # No absolute paths: the session folder is wherever meta.json sits.
            "brush_configs": [brush.to_config(rel_base=self.brush_rel_base) for brush in self.brushes],
            "paint_mesh_dicts": [mesh.to_dict(rel_base=self.export_folder) for mesh in self.paint_meshes]
        }
        with open(folder_path / "meta.json", "w") as f:
            json.dump(session_data, f, indent=4)

    # -------------------------------------------------------------------------
    # Loading / Import
    # -------------------------------------------------------------------------
    @staticmethod
    def load(session_name, session_dir, session_brush_lib:ReferenceBrushLibrary):
        folder_path = os.path.join(session_dir, session_name)
        folder = Path(folder_path)
        with open(folder / "meta.json", "r") as f:
            data = json.load(f)
        print(data)
        # Use the folder we loaded from, not the (legacy, absolute) session_dir
        # recorded in the file, so a moved or renamed session still loads.
        session = GlossSession(
            session_name=session_name,
            session_folder=session_dir,
            brush_rel_base=session_brush_lib.cache_folder,
            inpaint_models=data["inpaint_model"],
            meshes=data["meshes"]
        )
        for mesh in data["meshes"]:
            if mesh in data["inpaint_model"]:
                session_brush_lib.load_mesh_model(mesh, load_inpaint_model=True)
            else:
                session_brush_lib.load_mesh_model(mesh, load_inpaint_model=False)
        for brush_config in data["brush_configs"]:
            brush_name = brush_config["name"]
            print(f"load brush {brush_name}")
            brush = ReferenceBrush.from_config(brush_config, session_brush_lib.get_single_view_by_id,
                                               session_brush_lib.inpaint_models[brush_config["mesh_name"]],
                                               rel_base=session_brush_lib.cache_folder)
            brush.prepare(force=False)
            session_brush_lib.brushes[brush_name] = brush
            session.brushes.append(brush)
        session.paint_meshes = [PaintMesh.from_dict(d, rel_base=session.export_folder) for d in data["paint_mesh_dicts"]]
        return session
