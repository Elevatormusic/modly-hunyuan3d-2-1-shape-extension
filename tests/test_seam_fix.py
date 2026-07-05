import unittest
import numpy as np
import seam_fix


class TestSeamDetection(unittest.TestCase):
    def test_single_chart_no_seam(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
        faces = np.array([[0, 1, 2]], int)
        uvs = np.array([[0, 0], [1, 0], [0, 1]], float)
        self.assertEqual(seam_fix._find_seam_edges(vertices, faces, uvs), [])

    def test_seam_detected_on_split_uv(self):
        # square as 4 verts x2 (duplicated at the shared edge to carry 2 UVs)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],   # tri0
                             [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)  # tri1 (dups of v1,v3 pos)
        faces = np.array([[0, 1, 2], [3, 4, 5]], int)
        uvs = np.array([[0.1, 0.1], [0.4, 0.1], [0.1, 0.4],
                        [0.6, 0.6], [0.9, 0.6], [0.6, 0.9]], float)  # different island
        seams = seam_fix._find_seam_edges(vertices, faces, uvs)
        # shared 3D edge is (v1,v2)=(v3-pos, v5-pos): (1,0,0)-(0,1,0)
        self.assertEqual(len(seams), 1)
        a0, a1, b0, b1 = seams[0]
        self.assertFalse(np.allclose(a0, b0) and np.allclose(a1, b1))


class TestReconcile(unittest.TestCase):
    def test_seam_jump_drops_interior_preserved(self):
        # Build a 64x64 atlas: left half color A, right half color B, seam down the
        # middle mapped by two charts. Assert cross-seam delta drops, deep interior
        # (col 5 vs col 58) unchanged.
        import numpy as np, seam_fix
        atlas = np.zeros((64, 64, 3), np.uint8)
        atlas[:, :32] = [200, 40, 40]   # chart A
        atlas[:, 32:] = [40, 40, 200]   # chart B
        # two charts abutting at u=0.5; verts share the 3D edge, UVs differ across it
        vertices = np.array([[0,0,0],[0,1,0],[1,0,0],   # A tri (3D edge v0-v1)
                             [0,0,0],[0,1,0],[-1,0,0]], float)  # B tri, same 3D edge
        faces = np.array([[0,1,2],[3,4,5]], int)
        uvs = np.array([[0.49,0.1],[0.49,0.9],[0.1,0.5],
                        [0.51,0.1],[0.51,0.9],[0.9,0.5]], float)
        before = abs(int(atlas[32,31,2]) - int(atlas[32,32,2]))
        out = seam_fix._reconcile(atlas.copy(), faces, uvs,
                                  seam_fix._find_seam_edges(vertices,faces,uvs), 4)
        after = abs(int(out[32,31,2]) - int(out[32,32,2]))
        self.assertLess(after, before)              # seam jump reduced
        np.testing.assert_array_equal(out[:, :5], atlas[:, :5])   # deep interior A
        np.testing.assert_array_equal(out[:, 59:], atlas[:, 59:]) # deep interior B


class TestDilateAndCompose(unittest.TestCase):
    def test_gutter_filled_with_nearest_valid(self):
        import numpy as np, seam_fix
        atlas = np.zeros((32, 32, 3), np.uint8)
        # one triangle covering the top-left; the rest is gutter (black)
        faces = np.array([[0, 1, 2]], int)
        uvs = np.array([[0.05, 0.95], [0.45, 0.95], [0.05, 0.55]], float)
        atlas[2:14, 2:14] = [180, 20, 20]   # paint roughly where the tri lands
        out = seam_fix._dilate_gutter(atlas.copy(), faces, uvs, gutter_px=6)
        # a gutter texel just outside the island should now be island-colored, not black
        np.testing.assert_array_equal(out[15, 8], [180, 20, 20])

    def test_reconcile_and_dilate_noop_on_no_seam(self):
        import numpy as np, seam_fix
        atlas = np.full((16, 16, 3), 120, np.uint8)
        vertices = np.array([[0,0,0],[1,0,0],[0,1,0]], float)
        faces = np.array([[0,1,2]], int)
        uvs = np.array([[0.1,0.1],[0.9,0.1],[0.1,0.9]], float)
        out = seam_fix.reconcile_and_dilate(vertices, faces, uvs, atlas.copy())
        # no seam -> interior untouched (dilation only writes gutter)
        np.testing.assert_array_equal(out[3:6, 3:6], atlas[3:6, 3:6])


if __name__ == "__main__":
    unittest.main()
