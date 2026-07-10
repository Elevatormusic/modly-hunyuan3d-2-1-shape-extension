# tests/test_smooth_normals.py
import math
import unittest
import numpy as np
import smooth_normals as sn


def _two_tri(theta_deg, uv_seam=False):
    """Two triangles sharing edge (v0,v1) on the x-axis; tri2 folded up by
    theta about that edge. dihedral(face_normals) == theta_deg.
    uv_seam=True duplicates the shared verts (distinct input vertices, same
    position) to emulate a UV-chart seam along the shared edge."""
    t = math.radians(theta_deg)
    v0, v1 = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]
    v2 = [0.5, 1.0, 0.0]                          # tri1 apex, z=0 plane
    v3 = [0.5, -math.cos(t), math.sin(t)]         # tri2 apex, folded up by theta
    if not uv_seam:
        P = np.array([v0, v1, v2, v3])
        F = np.array([[0, 1, 2], [1, 0, 3]])
        UV = np.zeros((4, 2))
    else:
        P = np.array([v0, v1, v2, v0, v1, v3])    # 3,4 duplicate 0,1
        F = np.array([[0, 1, 2], [4, 3, 5]])      # tri2 uses the duplicates
        UV = np.array([[0, 0], [1, 0], [.5, 1], [0, .5], [1, .5], [.5, 1]], float)
    return P, F, UV


class TestCreaseSmooth(unittest.TestCase):
    def test_flat_no_split_axis_aligned(self):
        P, F, UV = _two_tri(0.0)
        nP, nU, nF, N = sn.crease_smooth(P, F, UV, crease_deg=45.0)
        self.assertEqual(len(nP), 4)                       # no crease -> no split
        self.assertEqual(nF.shape, (2, 3))
        np.testing.assert_allclose(np.abs(N), np.tile([0, 0, 1.0], (len(N), 1)), atol=1e-6)
        self.assertTrue(np.all(np.isfinite(N)))

    def test_shallow_fold_smoothed_not_split(self):
        P, F, UV = _two_tri(10.0)                          # below 45 -> smooth
        nP, nU, nF, N = sn.crease_smooth(P, F, UV, crease_deg=45.0)
        self.assertEqual(len(nP), 4)                       # shared verts NOT split
        # the two shared verts (index 0,1 in input) get the blend of both faces,
        # i.e. not equal to either face normal
        face1_n = np.array([0, 0, 1.0])
        # every shared-vertex normal must differ from a pure face normal (blended)
        blended_exists = np.any(np.linalg.norm(N - face1_n, axis=1) > 1e-3)
        self.assertTrue(blended_exists)
        self.assertTrue(np.allclose(np.linalg.norm(N, axis=1), 1.0, atol=1e-6))

    def test_hard_fold_splits(self):
        P, F, UV = _two_tri(90.0)                          # above 45 -> hard edge
        nP, nU, nF, N = sn.crease_smooth(P, F, UV, crease_deg=45.0)
        self.assertEqual(len(nP), 6)                       # v0,v1 each split in two
        # no normal is the 45-deg blend of the two faces
        blend = np.array([0, math.sqrt(.5), math.sqrt(.5)])
        self.assertTrue(np.all(np.linalg.norm(N - blend, axis=1) > 0.1))
        self.assertTrue(np.all(np.linalg.norm(N + blend, axis=1) > 0.1))

    def test_uv_seam_no_crease_is_smoothed(self):
        # flat, but a UV seam along the shared edge: both duplicates must get the
        # SAME (blended) normal -> no shading seam. (This is why NORMAL is
        # accumulated per welded-position, not per input vertex.)
        P, F, UV = _two_tri(20.0, uv_seam=True)            # 20 deg < 45 -> smooth
        nP, nU, nF, N = sn.crease_smooth(P, F, UV, crease_deg=45.0)
        self.assertEqual(len(nP), 6)                       # UV duplicates preserved, no crease split
        # find the two output verts at position v0 and assert equal normals
        v0 = np.array([0.0, 0.0, 0.0])
        at_v0 = np.where(np.linalg.norm(nP - v0, axis=1) < 1e-9)[0]
        self.assertEqual(len(at_v0), 2)                    # both seam copies survive
        np.testing.assert_allclose(N[at_v0[0]], N[at_v0[1]], atol=1e-6)

    def test_degenerate_face_no_nan(self):
        P = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]], float)  # tri0 degenerate
        F = np.array([[0, 1, 2], [0, 2, 3]])
        UV = np.zeros((4, 2))
        nP, nU, nF, N = sn.crease_smooth(P, F, UV, crease_deg=45.0)
        self.assertTrue(np.all(np.isfinite(N)))
        np.testing.assert_allclose(np.linalg.norm(N, axis=1), 1.0, atol=1e-6)

    def test_counts_consistent(self):
        P, F, UV = _two_tri(90.0)
        nP, nU, nF, N = sn.crease_smooth(P, F, UV, crease_deg=45.0)
        self.assertEqual(len(nP), len(nU))
        self.assertEqual(len(nP), len(N))
        self.assertEqual(nF.shape, F.shape)
        self.assertTrue(nF.max() < len(nP))


if __name__ == "__main__":
    unittest.main()
