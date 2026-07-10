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

    def test_reconcile_and_dilate_noop_on_zero_vertices(self):
        # MINOR-2: empty input must return the atlas unchanged, not raise
        # (np.ptp on a 0-row array previously raised ValueError).
        import numpy as np, seam_fix
        atlas = np.full((16, 16, 3), 120, np.uint8)
        vertices = np.zeros((0, 3), float)
        faces = np.zeros((0, 3), int)
        uvs = np.zeros((0, 2), float)
        out = seam_fix.reconcile_and_dilate(vertices, faces, uvs, atlas.copy())
        np.testing.assert_array_equal(out, atlas)


class TestApplyToGlb(unittest.TestCase):
    def test_apply_preserves_albedo_and_mr(self):
        import os, tempfile, numpy as np, trimesh, mr_export, seam_fix
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            base = os.path.join(d, "textured")
            Image.fromarray(np.full((64,64,3),128,np.uint8)).save(base+".png")
            Image.fromarray(np.full((64,64),30,np.uint8)).save(base+"_metallic.png")
            Image.fromarray(np.full((64,64),210,np.uint8)).save(base+"_roughness.png")
            with open(base+".mtl","w") as f: f.write("newmtl m\nmap_Kd textured.png\n")
            with open(base+".obj","w") as f:
                f.write("mtllib textured.mtl\nusemtl m\nv 0 0 0\nv 1 0 0\nv 0 1 0\n"
                        "vt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n")
            glb = os.path.join(d, "out.glb")
            mr_export.build_glb_with_mr(base+".obj", glb)
            seam_fix.apply_to_glb(glb)     # must not raise
            g = list(trimesh.load(glb, process=False).geometry.values())[0]
            self.assertIsNotNone(g.visual.material.baseColorTexture)
            self.assertIsNotNone(g.visual.material.metallicRoughnessTexture)

    def test_apply_keeps_jpeg_albedo_jpeg(self):
        # C3: after seam-fix, a JPEG-sourced albedo must stay JPEG in the rebuilt
        # GLB. Image.fromarray(...) has format=None -> trimesh would re-encode PNG
        # (~10x bloat). The fix preserves the source format.
        import os, tempfile, numpy as np, trimesh, seam_fix
        from PIL import Image
        from tests._fixtures import unit_uv_quad
        with tempfile.TemporaryDirectory() as d:
            verts, faces, uv = unit_uv_quad()
            m = trimesh.Trimesh(verts, faces, process=False)
            img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), np.uint8))
            img.format = "JPEG"
            mat = trimesh.visual.material.PBRMaterial(baseColorTexture=img)
            m.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
            glb = os.path.join(d, "j.glb")
            m.export(glb)
            # sanity: source really embedded as jpeg
            self.assertIn(b"image/jpeg", open(glb, "rb").read())
            seam_fix.apply_to_glb(glb)
            raw = open(glb, "rb").read()
            self.assertIn(b"image/jpeg", raw)          # still jpeg
            self.assertNotIn(b"image/png", raw)        # not re-encoded to png
            g = list(trimesh.load(glb, process=False).geometry.values())[0]
            self.assertEqual((g.visual.material.baseColorTexture.format or "").upper(),
                             "JPEG")


class TestReviewFixes(unittest.TestCase):
    def test_twin_packed_toward_interior_not_worsened(self):
        # C-IMPORTANT-1: when xatlas packs the twin toward chart A's interior, the
        # old twin-based inward direction sampled A's EXTERIOR and drove the seam
        # texel the wrong way. The interior direction must come from A's own third
        # (opposite) vertex. Seam on the u=0.5 boundary, A interior to the RIGHT,
        # twin packed to the right (toward A's interior).
        atlas = np.empty((64, 64, 3), np.uint8)
        atlas[:, :32] = 200      # A exterior (u<0.5)
        atlas[:, 32:] = 100      # A interior (u>0.5); B is packed inside here too
        vertices = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0],
                             [0, 0, 0], [0, 1, 0], [2, 0, 0]], float)
        faces = np.array([[0, 1, 2], [3, 4, 5]], int)
        uvs = np.array([[0.5, 0.2], [0.5, 0.8], [0.9, 0.5],   # A: seam u=0.5, third RIGHT
                        [0.6, 0.2], [0.6, 0.8], [0.7, 0.5]],   # B: packed right of A's seam
                       float)
        seams = seam_fix._find_seam_edges(vertices, faces, uvs)
        self.assertEqual(len(seams), 1)
        out = seam_fix._reconcile(atlas.copy(), faces, uvs, seams, 4)
        # A's seam texels (col 32, interior=100) must stay ~100, not be dragged
        # toward the 200 exterior the old code sampled (it drove them to ~50).
        self.assertGreater(out[20:45, 32, 0].mean(), 90)

    def test_reconcile_skips_nonfinite_seam(self):
        # C-9: a NaN/inf UV must skip just that seam, not abort the whole stage.
        atlas = np.full((16, 16, 3), 100, np.uint8)
        seams = [(np.array([np.nan, 0.5]), np.array([0.5, 0.9]),
                  np.array([0.51, 0.1]), np.array([0.51, 0.9]))]
        out = seam_fix._reconcile(atlas.copy(), np.zeros((0, 3), int),
                                  np.zeros((0, 2), float), seams, 4)
        np.testing.assert_array_equal(out, atlas)   # unchanged, no raise

    def test_spray_touches_contiguous_columns(self):
        # C-9: integer round() (banker's) skipped columns (a seam at x=3.5 stepping
        # inward hit 4,4,6 -> col 5 lost). Round-half-up hits every column.
        corr = np.zeros((8, 8, 1), np.float64)
        wsum = np.zeros((8, 8), np.float64)
        seam_fix._spray(corr, wsum, np.array([3.5, 4.0]),
                        np.array([1.0, 0.0]), 3, np.array([10.0]))
        touched = sorted(int(x) for x in np.where(wsum[4] > 0)[0])
        self.assertEqual(touched, [4, 5, 6])   # contiguous, no skipped column

    def test_seam_samples_use_longer_side(self):
        # C-9: the along-seam sample count must come from the LONGER side so the
        # denser twin is covered without holes (old code used side A only).
        n = seam_fix._seam_samples(np.array([0.0, 0.0]), np.array([3.0, 0.0]),
                                   np.array([0.0, 0.0]), np.array([40.0, 0.0]))
        self.assertGreaterEqual(n, 41)

    def test_feather_band_clamped_per_seam(self):
        # C-IMPORTANT-2: the band clamps from EACH seam's own edge length; a tiny
        # 2px seam must not collapse a long 200px seam's band across the atlas.
        base = seam_fix._local_band(4096)
        self.assertEqual(base, 9)
        self.assertEqual(seam_fix._seam_band(base, 200), base)   # long keeps base
        self.assertEqual(seam_fix._seam_band(base, 2), 1)        # tiny clamps to 1

    def test_coverage_mask_no_matplotlib(self):
        # C-IMPORTANT-4: matplotlib is not a declared dependency; the point-in-poly
        # test must be pure-numpy barycentric with no matplotlib import.
        import inspect
        self.assertNotIn("matplotlib", inspect.getsource(seam_fix))

    def test_coverage_mask_barycentric_correct(self):
        # right triangle corners (0,0),(15,0),(0,15) in px (x,y) space
        faces = np.array([[0, 1, 2]], int)
        uvs = np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]], float)
        mask = seam_fix._coverage_mask(uvs, faces, 16, 16)
        self.assertTrue(mask[3, 3])         # interior (y,x)
        self.assertFalse(mask[14, 14])      # past the hypotenuse
        self.assertGreater(mask.sum(), 90)  # ~half of 256
        self.assertLess(mask.sum(), 170)


if __name__ == "__main__":
    unittest.main()
