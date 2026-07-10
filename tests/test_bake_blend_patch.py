"""Section 9 (bake-blend merge) durable-patch tests.

`_patch_gpu_accel` section 9 (`_patch_bake_blend`) copies `bake_blend.py` next to the
vendored paint code and replaces `ViewProcessor.bake_from_multiview`'s stock in-loop merge
with a body that routes to `bake_blend.bake_from_multiview_ex` (env `EB_BAKE_BLEND=legacy`
restores stock). These tests verify the patch lands, is idempotent, copies the helper, and
— when the live vendored file is present — that it carries the hand-applied edit.
No torch, no GPU: pure string surgery.
"""
import sys
import tempfile
import types
import unittest
from pathlib import Path


def _install_services_stub():
    """generator.py imports Modly's runtime `services.generators.base` at module top.
    Off-device (CI / this repo's tests) that package is absent, so stub it before import."""
    if "services.generators.base" in sys.modules:
        return
    services = types.ModuleType("services")
    gens = types.ModuleType("services.generators")
    base = types.ModuleType("services.generators.base")

    class BaseGenerator:
        pass

    def smooth_progress(*a, **k):
        pass

    class GenerationCancelled(Exception):
        pass

    base.BaseGenerator = BaseGenerator
    base.smooth_progress = smooth_progress
    base.GenerationCancelled = GenerationCancelled
    services.generators = gens
    gens.base = base
    sys.modules["services"] = services
    sys.modules["services.generators"] = gens
    sys.modules["services.generators.base"] = base


_install_services_stub()
import generator as gen_mod  # noqa: E402


STOCK_BODY = '''    def bake_from_multiview(self, views, camera_elevs, camera_azims, view_weights):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []

        for view, camera_elev, camera_azim, weight in zip(views, camera_elevs, camera_azims, view_weights):
            project_texture, project_cos_map, project_boundary_map = self.render.back_project(
                view, camera_elev, camera_azim
            )
            project_cos_map = weight * (project_cos_map**self.config.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)
            texture, ori_trust_map = self.render.fast_bake_texture(project_textures, project_weighted_cos_maps)
        return texture, ori_trust_map > 1e-8
'''


class TestBakeBlendPatch(unittest.TestCase):
    def _fake_paint_src(self):
        d = Path(tempfile.mkdtemp())
        (d / "utils").mkdir()
        (d / "utils" / "pipeline_utils.py").write_text(
            "import torch\nimport numpy as np\n\n\nclass ViewProcessor:\n" + STOCK_BODY,
            encoding="utf-8")
        return d

    def test_patch_applies_and_is_idempotent(self):
        d = self._fake_paint_src()
        gen = gen_mod.Hunyuan3DShapeV21Generator
        gen._patch_bake_blend(d)                      # the section-9 helper
        text = (d / "utils" / "pipeline_utils.py").read_text(encoding="utf-8")
        self.assertIn("bake_blend.bake_from_multiview_ex", text)
        self.assertIn("EB_BAKE_BLEND", text)
        self.assertNotIn("project_boundary_maps = []", text)   # stock body replaced
        before = text
        gen._patch_bake_blend(d)                      # second run: no change
        self.assertEqual(before, (d / "utils" / "pipeline_utils.py").read_text(encoding="utf-8"))

    def test_bake_blend_copied_next_to_vendored(self):
        d = self._fake_paint_src()
        gen_mod.Hunyuan3DShapeV21Generator._patch_bake_blend(d)
        self.assertTrue((d / "bake_blend.py").exists())

    def test_section9_wired_before_eb_accel_early_returns(self):
        import inspect
        src = inspect.getsource(gen_mod.Hunyuan3DShapeV21Generator._patch_gpu_accel)
        self.assertIn("self._patch_bake_blend(", src)
        # `helper = Path(...)` starts the eb_accel copy + early-return block. The call must sit
        # BEFORE it; fails if someone moves _patch_bake_blend back behind the early-returns.
        # (Anchoring on the plain "eb_accel.py" literal is wrong here — it also appears in the
        # method docstring, which precedes the call.)
        self.assertLess(
            src.index("self._patch_bake_blend("),
            src.index('helper = Path(__file__)'))

    def test_live_vendored_file_is_patched(self):
        live = Path(r"C:\Users\Shaya\OneDrive\Documents\Modly\models\hunyuan3d-2-1-shape"
                    r"\generate\_hy3dpaint_src\utils\pipeline_utils.py")
        if not live.exists():
            self.skipTest("live vendored file not present")
        text = live.read_text(encoding="utf-8")
        self.assertIn("bake_blend.bake_from_multiview_ex", text)


if __name__ == "__main__":
    unittest.main()
