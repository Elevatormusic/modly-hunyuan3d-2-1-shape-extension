import unittest
import numpy as np
from PIL import Image
import game_bake


def _plane():
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], int)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    nrm = np.tile([0.0, 0.0, 1.0], (4, 1))
    return verts, faces, uv, nrm


class TestGameBake(unittest.TestCase):
    def test_albedo_transfer_reproduces_source(self):
        v, f, uv, n = _plane()
        alb = np.zeros((64, 64, 3), np.uint8)
        alb[:, :32] = (255, 0, 0)   # left red
        alb[:, 32:] = (0, 0, 255)   # right blue
        out = game_bake.bake_maps(v, f, uv, Image.fromarray(alb, "RGB"), None,
                                  v, f, uv, n, size=64, ao_samples=8)
        got = np.asarray(out["albedo"])
        self.assertGreater(int(got[32, 8, 0]), 150)   # left is red
        self.assertLess(int(got[32, 8, 2]), 100)
        self.assertGreater(int(got[32, 56, 2]), 150)  # right is blue
        self.assertLess(int(got[32, 56, 0]), 100)

    def test_ao_white_on_open_plane(self):
        v, f, uv, n = _plane()
        alb = Image.fromarray(np.full((32, 32, 3), 200, np.uint8), "RGB")
        out = game_bake.bake_maps(v, f, uv, alb, None, v, f, uv, n, size=48, ao_samples=16)
        self.assertGreater(float(np.asarray(out["ao"]).mean()), 200.0)  # mostly unoccluded

    def test_ao_darker_under_occluder(self):
        # Dense = the plane + a ceiling over its LEFT half (z=0.2). Texels under
        # the ceiling must bake darker AO than the open right half.
        v, f, uv, n = _plane()
        ceil_v = np.array([[0, 0, 0.2], [0.5, 0, 0.2], [0.5, 1, 0.2], [0, 1, 0.2]], float)
        dv = np.vstack([v, ceil_v])
        df = np.vstack([f, np.array([[4, 5, 6], [4, 6, 7]], int)])
        duv = np.zeros((8, 2))
        alb = Image.fromarray(np.full((16, 16, 3), 180, np.uint8), "RGB")
        out = game_bake.bake_maps(dv, df, duv, alb, None, v, f, uv, n, size=48, ao_samples=32)
        ao = np.asarray(out["ao"]).astype(float)
        left, right = ao[:, :16].mean(), ao[:, 32:].mean()
        self.assertLess(left, right - 10.0)   # left half is occluded
        self.assertLess(left, 250.0)

    def test_mr_bytes_preserved_linear(self):
        v, f, uv, n = _plane()
        alb = Image.fromarray(np.full((32, 32, 3), 100, np.uint8), "RGB")
        mr = Image.fromarray(np.full((32, 32, 3), 128, np.uint8), "RGB")
        out = game_bake.bake_maps(v, f, uv, alb, mr, v, f, uv, n, size=48, ao_samples=8)
        self.assertIn("mr", out)
        self.assertAlmostEqual(int(np.asarray(out["mr"])[24, 24, 1]), 128, delta=4)

    def test_empty_low_returns_empty(self):
        v, f, uv, n = _plane()
        alb = Image.fromarray(np.full((16, 16, 3), 100, np.uint8), "RGB")
        out = game_bake.bake_maps(v, f, uv, alb, None,
                                  v, np.zeros((0, 3), int), uv, n, size=16)
        self.assertEqual(out, {})


if __name__ == "__main__":
    unittest.main()
