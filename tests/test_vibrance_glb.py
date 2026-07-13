import os, tempfile, unittest
import numpy as np, trimesh
from PIL import Image
import vibrance


def _make_glb(path, color=(150, 120, 120)):
    m = trimesh.creation.box()
    img = Image.fromarray(np.full((32, 32, 3), color, np.uint8), "RGB")
    uv = np.random.rand(len(m.vertices), 2).astype(np.float32)
    from trimesh.visual.material import PBRMaterial
    m.visual = trimesh.visual.TextureVisuals(uv=uv, material=PBRMaterial(baseColorTexture=img))
    m.export(path)


class TestApplyToGlb(unittest.TestCase):
    def test_strength_zero_noop(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.glb"); _make_glb(p)
            before = open(p, "rb").read()
            self.assertFalse(vibrance.apply_to_glb(p, 0.0))
            self.assertEqual(open(p, "rb").read(), before)

    def test_applies_and_reloads(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.glb"); _make_glb(p)
            self.assertTrue(vibrance.apply_to_glb(p, 0.35))
            trimesh.load(p)  # still valid

    def test_missing_file_never_raises(self):
        self.assertFalse(vibrance.apply_to_glb("/no/such.glb", 0.35))


if __name__ == "__main__":
    unittest.main()
