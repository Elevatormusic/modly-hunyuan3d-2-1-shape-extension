# tests/test_convert_patch.py — verify the patch STRING in generator.py wires mr_export + fallback
import re, unittest, pathlib
class TestConvertPatch(unittest.TestCase):
    def test_patch_routes_to_mr_export_with_fallback(self):
        src = pathlib.Path("generator.py").read_text(encoding="utf-8")
        # the appended convert_obj_to_glb must try mr_export then fall back
        self.assertIn("import mr_export", src)
        self.assertIn("build_glb_with_mr", src)
        # fallback still constructs a flat PBRMaterial (albedo-only) on Exception
        block = src[src.index("def convert_obj_to_glb"):]
        self.assertIn("except Exception", block)
        self.assertIn("metallicFactor=0.0", block)   # flat fallback preserved
