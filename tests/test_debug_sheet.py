import os, tempfile, unittest
import numpy as np
import trimesh
from PIL import Image
import debug_sheet


def _synth(dirp, with_mr=True, with_normal=True):
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    base = os.path.join(dirp, "textured")
    with Image.fromarray(np.full((64, 64, 3), 128, np.uint8)) as al:
        al.save(base + ".png")
    with open(base + ".mtl", "w") as fh:
        fh.write("newmtl m\nmap_Kd textured.png\n")
    if with_mr:
        with Image.fromarray(np.full((64, 64), 30, np.uint8)) as m:
            m.save(base + "_metallic.png")
        with Image.fromarray(np.full((64, 64), 210, np.uint8)) as r:
            r.save(base + "_roughness.png")
    if with_normal:
        with Image.fromarray(np.full((64, 64, 3), (128, 128, 255), np.uint8)) as n:
            n.save(base + "_normal.png")
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    with Image.open(base + ".png") as tex:
        m.visual = trimesh.visual.TextureVisuals(uv=uv, image=tex.copy())
    glb = os.path.join(dirp, "out.glb")
    m.export(glb)
    return glb, base + ".obj"


class TestDebugSheet(unittest.TestCase):
    def test_full_sheet(self):
        with tempfile.TemporaryDirectory() as d:
            glb, obj = _synth(d)
            out = os.path.join(d, "out_qa.png")
            res = debug_sheet.write_debug_sheet(glb, obj, out)
            self.assertEqual(res, out)
            self.assertTrue(os.path.exists(out))
            with Image.open(out) as im:
                self.assertGreaterEqual(im.width, 512)
                self.assertGreaterEqual(im.height, 256)

    def test_missing_maps_graceful(self):
        with tempfile.TemporaryDirectory() as d:
            glb, obj = _synth(d, with_mr=False, with_normal=False)
            out = os.path.join(d, "out_qa.png")
            res = debug_sheet.write_debug_sheet(glb, obj, out)
            self.assertEqual(res, out)
            self.assertTrue(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
