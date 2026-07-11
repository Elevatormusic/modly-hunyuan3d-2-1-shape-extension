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


def _crevice_sphere():
    """Deterministic discriminating fixture for the masked smoother.

    Icosphere carrying two well-separated features:
      * an inward-dented equatorial ring -> a REAL concave crevice band, the
        strongest-concave region, so it owns the 8th-percentile curvature
        threshold and the masked Taubin pass smooths it.
      * a single outward spike at the north pole -> a sharp CONVEX feature far
        (>2 dilation rings) from the crevice. Its curvature is positive, so the
        concave mask must NOT select it: masked smoothing leaves it exactly in
        place, whereas whole-mesh (masking-bypassed) smoothing rounds it off.
    Returns (mesh, equatorial-band mask, apex vertex index).
    """
    m = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    v = m.vertices.copy()
    z = v[:, 2]
    band = np.abs(z) < 0.12          # narrow equatorial ring of vertices
    v[band] *= 0.45                  # dent it inward -> concave crevice
    apex = int(np.argmax(v[:, 2]))   # the lone north-pole vertex, far from band
    v[apex] *= 1.8                   # push it out -> sharp convex spike
    return trimesh.Trimesh(vertices=v, faces=m.faces, process=False), band, apex


class TestSmoothCrevices(unittest.TestCase):
    def test_returns_valid_mesh_same_face_count(self):
        m = _bumpy_sphere(noise=0.01)
        out = mc.smooth_crevices(m)
        self.assertEqual(len(out.faces), len(m.faces))
        self.assertTrue(np.all(np.isfinite(out.vertices)))

    def test_masked_convex_region_barely_moves(self):
        # Masking must be LOCAL: smooth the concave crevice band, leave the far
        # convex spike untouched. Discriminates against both failure modes:
        #   * no-op            -> band never moves            -> assert (b) fails
        #   * whole-mesh smooth -> spike rounds off (~0.16*bb) -> assert (a) fails
        # (measured: correct masked path gives spike disp 0.0, band max 0.055*bb.)
        m, band, apex = _crevice_sphere()
        out = mc.smooth_crevices(m)
        self.assertEqual(len(out.vertices), len(m.vertices))  # order preserved
        disp = np.linalg.norm(out.vertices - m.vertices, axis=1)
        bb = float(np.linalg.norm(m.extents))
        # (a) far-away convex spike essentially unmoved by the masked smoother
        self.assertLess(disp[apex] / bb, 0.01)
        # (b) the concave crevice band actually got smoothed (not a no-op)
        self.assertGreater(disp[band].max() / bb, 0.01)

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
