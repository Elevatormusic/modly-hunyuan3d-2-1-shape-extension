import unittest
from unittest import mock
import finishing


class TestFinish(unittest.TestCase):
    def _install(self, calls):
        import seam_fix, normal_bake, glb_validate, debug_sheet, capacity
        specs = [
            (seam_fix, "apply_to_glb", lambda p: calls.append("seam")),
            (normal_bake, "bake_normal_map", lambda dense, p, size=None: calls.append("bake")),
            (glb_validate, "validate_glb",
             lambda p: (calls.append("validate") or {"ok": True, "warnings": [], "info": {}})),
            (debug_sheet, "write_debug_sheet",
             lambda glb, obj, out, input_image_path=None: (calls.append("sheet") or out)),
            (capacity, "should_bake", lambda b, m: bool(b) and m != "bpt"),
        ]
        for mod, attr, fn in specs:
            patch = mock.patch.object(mod, attr, fn)
            patch.start()
            self.addCleanup(patch.stop)

    def _finish(self, **kw):
        calls = []
        self._install(calls)
        args = dict(dense_mesh=object(), texture_size=1024, mesh_mode="regular")
        args.update(kw)
        rep = finishing.finish("out.glb", "textured.obj", **args)
        return calls, rep

    def test_default_no_bake_full_qa(self):
        calls, rep = self._finish()
        self.assertIn("seam", calls)
        self.assertNotIn("bake", calls)
        self.assertIn("validate", calls)
        self.assertIn("sheet", calls)
        self.assertEqual(rep["seam_fix"], "ok")

    def test_bake_on(self):
        calls, rep = self._finish(bake_normal_map=True)
        self.assertEqual(calls[:2], ["seam", "bake"])
        self.assertEqual(rep["bake"], "ok")

    def test_toggles_off(self):
        calls, rep = self._finish(seam_fix=False, debug_sheet=False)
        self.assertNotIn("seam", calls)
        self.assertNotIn("sheet", calls)
        self.assertIn("validate", calls)

    def test_non_fatal(self):
        calls = []
        self._install(calls)
        import seam_fix

        def boom(p):
            raise RuntimeError("boom")

        patch = mock.patch.object(seam_fix, "apply_to_glb", boom)
        patch.start()
        self.addCleanup(patch.stop)
        rep = finishing.finish("out.glb", "textured.obj",
                               dense_mesh=object(), texture_size=1024, mesh_mode="regular")
        self.assertIsInstance(rep, dict)
        self.assertTrue(rep["seam_fix"].startswith("skipped"))


if __name__ == "__main__":
    unittest.main()
