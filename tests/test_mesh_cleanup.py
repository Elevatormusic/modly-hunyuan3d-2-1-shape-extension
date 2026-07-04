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


class TestStripBackground(unittest.TestCase):
    def test_removes_large_ground_plane_keeps_object(self):
        obj = trimesh.creation.box(extents=(0.5, 0.6, 0.5)).subdivide().subdivide()
        plane = trimesh.creation.box(extents=(2.0, 0.02, 2.0)).subdivide().subdivide()
        plane.apply_translation([0, -0.5, 0])
        combined = trimesh.util.concatenate([obj, plane])
        out = mesh_cleanup.strip_background(combined)
        self.assertLess(len(out.faces), len(combined.faces))       # plane dropped
        self.assertGreater(len(out.faces), 0)
        e = out.vertices.max(0) - out.vertices.min(0)
        self.assertGreater(e.min() / e.max(), 0.2)                  # kept part is 3D, not flat

    def test_single_component_unchanged(self):
        obj = trimesh.creation.box(extents=(1, 1, 1))
        out = mesh_cleanup.strip_background(obj)
        self.assertEqual(len(out.faces), len(obj.faces))

    def test_lone_flat_object_not_nuked(self):
        # a genuinely flat object (e.g. a coin/plate) must survive
        plate = trimesh.creation.box(extents=(2.0, 0.02, 2.0))
        out = mesh_cleanup.strip_background(plate)
        self.assertGreater(len(out.faces), 0)

    def test_small_flat_detail_kept(self):
        # a small flat panel attached near a 3D body is NOT scene-spanning -> keep it
        body = trimesh.creation.box(extents=(1.0, 1.0, 1.0)).subdivide().subdivide()
        panel = trimesh.creation.box(extents=(0.3, 0.02, 0.3)).subdivide().subdivide()
        panel.apply_translation([0.8, 0, 0])
        combined = trimesh.util.concatenate([body, panel])
        out = mesh_cleanup.strip_background(combined)
        # both kept (panel is not wide/large enough to look like a ground plane)
        self.assertEqual(len(out.faces), len(combined.faces))


if __name__ == "__main__":
    unittest.main()
