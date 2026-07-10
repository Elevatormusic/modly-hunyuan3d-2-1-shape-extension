# tests/test_finishing_normals.py
import sys
import types
import unittest


class TestFinishingNormalsStage(unittest.TestCase):
    def _run_with_stubs(self, apply_impl):
        import importlib
        import finishing
        importlib.reload(finishing)
        calls = []
        # stub seam_fix, smooth_normals, capacity, glb_validate so finish() runs
        # without a real GLB and we can observe ordering.
        sf = types.ModuleType("seam_fix")
        sf.apply_to_glb = lambda p: calls.append("seam_fix")
        sn = types.ModuleType("smooth_normals")
        def _apply(p, crease_deg=45.0):
            calls.append("smooth_normals")
            return apply_impl(p)
        sn.apply_to_glb = _apply
        cap = types.ModuleType("capacity")
        cap.should_bake = lambda *a, **k: False
        gv = types.ModuleType("glb_validate")
        gv.validate_glb = lambda p: (calls.append("validate"), {"ok": True})[1]
        for name, mod in (("seam_fix", sf), ("smooth_normals", sn),
                          ("capacity", cap), ("glb_validate", gv)):
            sys.modules[name] = mod
        try:
            rep = finishing.finish("x.glb", "x.obj", dense_mesh=None, texture_size=2048,
                                   mesh_mode="isotropic", bake_normal_map=False,
                                   seam_fix=True, debug_sheet=False)
        finally:
            for name in ("seam_fix", "smooth_normals", "capacity", "glb_validate"):
                sys.modules.pop(name, None)
        return rep, calls

    def test_runs_after_seamfix_before_validate(self):
        rep, calls = self._run_with_stubs(lambda p: True)
        self.assertEqual(rep["normals"], "ok")
        self.assertIn("smooth_normals", calls)
        self.assertLess(calls.index("seam_fix"), calls.index("smooth_normals"))
        self.assertLess(calls.index("smooth_normals"), calls.index("validate"))

    def test_nonfatal_when_stage_raises(self):
        def _boom(p):
            raise RuntimeError("kaboom")
        rep, calls = self._run_with_stubs(_boom)
        self.assertTrue(rep["normals"].startswith("skipped"))   # recorded, not raised
        self.assertIn("validate", calls)                         # pipeline continued


if __name__ == "__main__":
    unittest.main()
