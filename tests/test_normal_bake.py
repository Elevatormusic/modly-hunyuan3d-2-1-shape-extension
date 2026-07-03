import unittest
import numpy as np
import trimesh
from tests._fixtures import unit_uv_quad
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


if __name__ == "__main__":
    unittest.main()
