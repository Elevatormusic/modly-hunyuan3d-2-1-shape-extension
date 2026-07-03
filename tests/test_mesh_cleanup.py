import unittest
import numpy as np
import trimesh
from tests._fixtures import box_dense, icosphere
import mesh_cleanup


def _edge_cv(m):
    e = np.vstack([m.faces[:, [0, 1]], m.faces[:, [1, 2]], m.faces[:, [2, 0]]])
    el = np.linalg.norm(m.vertices[e[:, 0]] - m.vertices[e[:, 1]], axis=1)
    return el.std() / el.mean()


class TestCleanMesh(unittest.TestCase):
    def test_regular_hits_target(self):
        lo = mesh_cleanup.clean_mesh(box_dense(), "regular", 500)
        self.assertGreater(len(lo.faces), 0)
        self.assertLessEqual(len(lo.faces), 700)  # decimation approximate but bounded

    def test_isotropic_produces_uniform_triangles(self):
        # Contract of isotropic remesh: the OUTPUT has uniform edge lengths.
        # (Also regularizes a non-uniform input: quadric decimation is irregular.)
        noisy = icosphere(4).simplify_quadric_decimation(face_count=1200)
        lo = mesh_cleanup.clean_mesh(noisy, "isotropic", 1200)
        self.assertGreater(len(lo.faces), 0)
        self.assertLess(_edge_cv(lo), 0.30)  # uniform triangles
        self.assertLess(_edge_cv(lo), _edge_cv(noisy) + 1e-9)  # no worse than the irregular input

    def test_scene_input_concatenated(self):
        sc = trimesh.Scene(box_dense())
        lo = mesh_cleanup.clean_mesh(sc, "regular", 500)
        self.assertIsInstance(lo, trimesh.Trimesh)
        self.assertGreater(len(lo.faces), 0)

    def test_isotropic_bounded_for_huge_target(self):
        # A huge target must not run remeshing away to millions of faces / minutes.
        import time
        t0 = time.time()
        lo = mesh_cleanup.clean_mesh(icosphere(3), "isotropic", 10_000_000)
        self.assertGreater(len(lo.faces), 0)
        self.assertLess(len(lo.faces), 500_000)   # bounded, no runaway
        self.assertLess(time.time() - t0, 30)      # and fast

    def test_bad_mode_falls_back_no_raise(self):
        lo = mesh_cleanup.clean_mesh(box_dense(), "nonsense", 500)
        self.assertIsInstance(lo, trimesh.Trimesh)
        self.assertGreater(len(lo.faces), 0)


if __name__ == "__main__":
    unittest.main()
