import unittest
import finishing
import vibrance


class TestFinishingVibrance(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self._orig = vibrance.apply_to_glb
        vibrance.apply_to_glb = lambda p, s: (self.calls.append((p, s)) or True)

    def tearDown(self):
        vibrance.apply_to_glb = self._orig

    def _finish(self, strength):
        # seam_fix / smooth_normals / bake / validate all no-op on a fake path;
        # finish() never raises, so we only assert the vibrance call.
        finishing.finish("nope.glb", "nope.obj", dense_mesh=None, texture_size=1024,
                         mesh_mode="regular", seam_fix=False,
                         saturation_strength=strength)

    def test_calls_vibrance_when_strength_positive(self):
        self._finish(0.18)
        self.assertEqual(self.calls, [("nope.glb", 0.18)])

    def test_skips_vibrance_when_zero(self):
        self._finish(0.0)
        self.assertEqual(self.calls, [])

    def test_strength_map_default_is_subtle(self):
        self.assertEqual(vibrance.STRENGTH_MAP["subtle"], 0.18)


if __name__ == "__main__":
    unittest.main()
