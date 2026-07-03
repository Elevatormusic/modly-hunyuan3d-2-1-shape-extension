import unittest
import numpy as np
import trimesh
from tests._fixtures import unit_uv_quad, icosphere
import normal_bake


class TestRasterizer(unittest.TestCase):
    def test_full_quad_coverage_and_positions(self):
        v, f, uv = unit_uv_quad()
        n = np.tile([0, 0, 1.0], (len(v), 1))
        pos, nrm, mask = normal_bake.rasterize_uv_atlas(v, f, uv, n, size=64)
        # a unit quad mapped to the full [0,1] atlas covers ~everything
        self.assertGreater(mask.mean(), 0.95)
        # center texel (u=v=0.5) maps to 3D ~ (0.5, 0.5, 0)
        c = pos[32, 32]
        self.assertTrue(np.allclose(c[:2], [0.5, 0.5], atol=0.05))
        # normals are +Z everywhere covered
        self.assertTrue(np.allclose(nrm[mask], [0, 0, 1.0], atol=1e-5))


class TestDenseSampling(unittest.TestCase):
    def test_identity_normals_match_surface(self):
        m = icosphere(4)
        pts, fid = trimesh.sample.sample_surface(m, 2000)
        pos = np.zeros((45, 45, 3), np.float32)
        mask = np.zeros((45, 45), bool)
        flat = pos.reshape(-1, 3)
        mflat = mask.reshape(-1)
        flat[:2000] = pts
        mflat[:2000] = True
        pos = flat.reshape(45, 45, 3)
        mask = mflat.reshape(45, 45)
        wn = normal_bake.sample_dense_normals(m, pos, mask)
        got = wn.reshape(-1, 3)[:2000]
        # icosphere at origin: outward normal ~= normalized position
        ref = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        dots = np.sum(got * ref, axis=1)
        self.assertGreater(np.median(dots), 0.98)


if __name__ == "__main__":
    unittest.main()
