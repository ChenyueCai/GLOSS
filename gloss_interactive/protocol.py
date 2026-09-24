"""Wire-protocol constants shared by the add-on and the Gloss server.

Every server -> client message carries a ``type`` field so the client can route
it to exactly one consumer. Before this contract existed, replies were
untagged and two Blender timers raced to drain a single shared inbox, which
silently dropped brush icons (see MESSAGING.md).

The add-on mirrors these constants in
``blender/gloss-blender/protocol.py`` (the add-on submodule). Keep the two files
in sync; ``tests/test_protocol.py`` asserts the values match.
"""

# --- client -> server -------------------------------------------------------
REQ_ADD_REF_MESH = "add_ref_mesh"
REQ_ADD_PNT_MESH = "add_pnt_mesh"
REQ_ADD_BRUSH = "add_brush"
REQ_SAVE_BRUSH = "save_brush"
REQ_FILL = "fill"
REQ_CLEAR_FACE = "clear_face"
REQ_CLEAR_ALL_TEXTURE = "clear_all_texture"
# Binary upload of the Blender-side paint texture. Sent automatically after
# every operation that changes the client texture, so the two sides never
# diverge; this replaces the old manual "Sync to Server" button.
REQ_IMAGE = "image"

# --- server -> client -------------------------------------------------------
#: Free-form human-readable progress line. Replaces the old bare-string replies.
MSG_STATUS = "status"
#: A request failed. ``data`` carries ``message`` and ``context`` (request type).
MSG_ERROR = "error"
#: Brush lifecycle transitions: preparing -> ready (or error).
MSG_BRUSH_STATUS = "brush_status"
#: Binary. The brush thumbnail, uint8 HWC.
MSG_BRUSH_ICON = "brush_icon"
#: Fill lifecycle transitions, emitted between inference stages.
MSG_FILL_STATUS = "fill_status"
#: Binary. One slice of a texture. Reassembled by chunk_index/chunk_total.
MSG_TEXTURE_CHUNK = "texture_chunk"
#: Server acknowledging a texture push, so the panel can show sync state.
MSG_TEXTURE_SYNCED = "texture_synced"

# --- brush / fill states ----------------------------------------------------
STATE_PREPARING = "preparing"
STATE_SYNCING = "syncing"
STATE_SYNCED = "synced"
STATE_READY = "ready"
STATE_ERROR = "error"

#: Wire dtype for pixel payloads. uint8 is 4x smaller than float32 and is
#: lossless for the [0, 1] basecolor data actually being sent.
DTYPE_UINT8 = "uint8"
DTYPE_FLOAT32 = "float32"

#: Default chunk budget, in bytes, for splitting large pixel payloads.
#: Both repos MUST agree on this; they silently disagreed (1 MB vs 10 MB)
#: before this constant existed, which sent 4K textures as ~269 frames.
DEFAULT_CHUNK_BYTES = 10_000_000
