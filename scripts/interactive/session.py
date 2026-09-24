# test save session and load from session 
# 1. save base sessions: 
# add koi fish mesh
# add inpaint model
# add koi fish auto brush, koi fish tail brush
# add koi fish paint mesh
from gloss_interactive.brush import *
from gloss_interactive.session import *
from gloss_interactive.paths import INTERACTIVE_MESH_CONFIG
from gloss.utils.paths import get_session_dir, resolve_path


config_fp = INTERACTIVE_MESH_CONFIG
brush_library = ReferenceBrushLibrary.from_config(config_fp)
session_dir = str(get_session_dir())

# 1. save koi fish base session
# session_name = "koi-fish-base"
# gloss_session = GlossSession(session_name, session_dir)

# mesh_name = "koi_fish"
# brush_library.load_mesh_model(mesh_name, load_inpaint_model=False)
# gloss_session.add_mesh(mesh_name)
# gloss_session.add_inpaint_model(mesh_name)
# sv_id = 117
# brush_name = f"{mesh_name}-{sv_id}-auto"
# brush_library.add_brush(name=brush_name, brush_type="AutoSampledReferenceBrush", mesh_name=mesh_name, sv_id=sv_id)
# brush = brush_library.get_brush_by_name(brush_name)
# brush_library.prepare_brush(brush_name)
# brush_library.save_brush(brush_name, save_ref_lib=True, save_ref=False, force_save=True)
# gloss_session.add_brush(brush)



# 2. save channel transfer sessions: 
# add croissant mesh
# add inpaint model
# add croissant mr0 brush
# add croissant paint mesh
#session_name = "croissant-mr0"
# gloss_session = GlossSession(session_name, session_dir)
# mesh_name = "croissant"
# brush_library.load_mesh_model(mesh_name, load_inpaint_model=True)
# gloss_session.add_mesh(mesh_name)
# gloss_session.add_inpaint_model(mesh_name)
# sv_id = 324
# brush_name = f"{mesh_name}-{sv_id}-mr0"
# brush_library.add_brush(name=brush_name, brush_type="AutoSampledReferenceBrush", mesh_name=mesh_name, sv_id=sv_id)
# brush = brush_library.get_brush_by_name(brush_name)
# brush_library.prepare_brush(brush_name)

# brush_library.save_brush(brush_name, save_ref_lib=True, save_ref=False, force_save=True)
# gloss_session.add_brush(brush)

# 3. save object transfer sessions 
# add croissant , koi fish mesh
# add croissant inpaint model 
# add croissant auto brush 
# add koi fish paint mesh
session_name = "croissant-koi_fish"
# gloss_session = GlossSession(session_name, session_dir)
# mesh_name = "koi_fish"
# brush_library.load_mesh_model(mesh_name, load_inpaint_model=False)
# gloss_session.add_mesh(mesh_name)

# mesh_name = "croissant"
# brush_library.load_mesh_model(mesh_name, load_inpaint_model=True)
# gloss_session.add_mesh(mesh_name)
# gloss_session.add_inpaint_model(mesh_name)
# sv_id = 394
# brush_name = f"{mesh_name}-{sv_id}-auto"
# brush_library.add_brush(name=brush_name, brush_type="AutoSampledReferenceBrush", mesh_name=mesh_name, sv_id=sv_id)
# brush = brush_library.get_brush_by_name(brush_name)
# brush_library.prepare_brush(brush_name)
# brush_library.save_brush(brush_name, save_ref_lib=True, save_ref=False, force_save=True)
# gloss_session.add_brush(brush)

gloss_session = GlossSession.load(session_name, session_dir, brush_library)
print(gloss_session.meshes, gloss_session.paint_meshes, gloss_session.inpaint_models, gloss_session.brushes)