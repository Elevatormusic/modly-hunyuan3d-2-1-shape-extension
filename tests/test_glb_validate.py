import os, tempfile, unittest
import numpy as np
import trimesh
from PIL import Image
import glb_validate


def _good(dirp):
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    m.visual = trimesh.visual.TextureVisuals(
        uv=uv, image=Image.fromarray(np.full((32, 32, 3), 128, np.uint8)))
    glb = os.path.join(dirp, "good.glb")
    m.export(glb)
    return glb


def _no_uv(dirp):
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], float)
    f = np.array([[0, 1, 2]])
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    glb = os.path.join(dirp, "nouv.glb")
    m.export(glb)
    return glb


class TestValidate(unittest.TestCase):
    def test_good(self):
        with tempfile.TemporaryDirectory() as d:
            rep = glb_validate.validate_glb(_good(d))
            self.assertTrue(rep["ok"], rep["warnings"])
            self.assertEqual(rep["warnings"], [])

    def test_missing_uv(self):
        with tempfile.TemporaryDirectory() as d:
            rep = glb_validate.validate_glb(_no_uv(d))
            self.assertFalse(rep["ok"])
            self.assertTrue(any("uv" in w.lower() for w in rep["warnings"]))


if __name__ == "__main__":
    unittest.main()
