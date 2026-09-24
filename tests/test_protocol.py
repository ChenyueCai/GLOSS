"""The server and the Blender add-on must agree on every wire-protocol constant.

``gloss_interactive/protocol.py`` mirrors ``blender/gloss-blender/protocol.py``
(the add-on submodule). Both are plain constant modules, so they are loaded by
path without importing either package.
"""

import importlib.util
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PROTOCOL = REPO_ROOT / "gloss_interactive" / "protocol.py"
ADDON_PROTOCOL = REPO_ROOT / "blender" / "gloss-blender" / "protocol.py"


def _constants(path):
    spec = importlib.util.spec_from_file_location(f"_protocol_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {k: v for k, v in vars(module).items() if k.isupper()}


@unittest.skipUnless(ADDON_PROTOCOL.exists(), "add-on submodule not checked out: git submodule update --init blender/gloss-blender")
class TestProtocolInSync(unittest.TestCase):
    def test_constants_match(self):
        server, addon = _constants(SERVER_PROTOCOL), _constants(ADDON_PROTOCOL)
        self.assertEqual(sorted(server), sorted(addon), "constant names differ")
        for name in server:
            self.assertEqual(server[name], addon[name], name)


if __name__ == "__main__":
    unittest.main()
