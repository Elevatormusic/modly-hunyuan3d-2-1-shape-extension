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


if __name__ == "__main__":
    unittest.main()
