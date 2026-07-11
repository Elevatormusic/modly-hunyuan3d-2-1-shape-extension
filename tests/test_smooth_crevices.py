# tests/test_smooth_crevices.py
import os
import unittest
import numpy as np
import trimesh
import mesh_cleanup as mc


def _bumpy_sphere(subdiv=4, noise=0.0, seed=0):
    m = trimesh.creation.icosphere(subdivisions=subdiv, radius=1.0)
    if noise:
        rng = np.random.RandomState(seed)
        m.vertices += rng.normal(0, noise, m.vertices.shape)
    return m


class TestSmoothCrevices(unittest.TestCase):
    def test_returns_valid_mesh_same_face_count(self):
        m = _bumpy_sphere(noise=0.01)
        out = mc.smooth_crevices(m)
        self.assertEqual(len(out.faces), len(m.faces))
        self.assertTrue(np.all(np.isfinite(out.vertices)))

    def test_masked_convex_region_barely_moves(self):
        # a mostly-convex sphere: masking should move few/no vertices far
        m = _bumpy_sphere(noise=0.0)
        out = mc.smooth_crevices(m)
        disp = np.linalg.norm(out.vertices - m.vertices, axis=1)
        bb = float(np.linalg.norm(m.extents))
        # a convex sphere has ~no strong-concave band -> global displacement tiny
        self.assertLess(disp.max() / bb, 0.02)

    def test_env_off_is_identity(self):
        m = _bumpy_sphere(noise=0.01)
        os.environ["EB_CREVICE_SMOOTH"] = "off"
        try:
            out = mc.smooth_crevices(m)
            np.testing.assert_array_equal(out.vertices, m.vertices)
        finally:
            os.environ.pop("EB_CREVICE_SMOOTH", None)

    def test_never_raises_on_degenerate(self):
        m = trimesh.Trimesh(vertices=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]))
        out = mc.smooth_crevices(m)              # must not raise
        self.assertIsNotNone(out)

    def test_clean_mesh_invokes_smoothing(self):
        # clean_mesh should call smooth_crevices (spy via monkeypatch)
        m = _bumpy_sphere(noise=0.01)
        called = {"n": 0}
        orig = mc.smooth_crevices
        try:
            mc.smooth_crevices = lambda mesh, **k: (called.__setitem__("n", called["n"] + 1) or mesh)
            mc.clean_mesh(m, "regular", 5000)
            self.assertEqual(called["n"], 1)
        finally:
            mc.smooth_crevices = orig


if __name__ == "__main__":
    unittest.main()
