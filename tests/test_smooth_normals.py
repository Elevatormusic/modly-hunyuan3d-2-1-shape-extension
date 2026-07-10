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


import io
import os
import pathlib
import shutil
import tempfile
import unittest.mock


def _build_no_normal_glb(path):
    """A minimal textured GLB with POSITION + TEXCOORD_0 + PBR material + image
    and NO NORMAL accessor (trimesh omits it when vertex_normals is never
    computed — RV-1 variant (i)). Returns the path."""
    import trimesh
    from trimesh.visual.material import PBRMaterial
    from PIL import Image
    P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [1, 0, 1], [1, 1, 1]], float)          # a 90-deg fold -> a real crease
    F = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]])
    UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [1, 0], [1, 1]], float)
    mesh = trimesh.Trimesh(vertices=P, faces=F, process=False)
    img = Image.new("RGB", (8, 8), (200, 160, 40))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=UV, material=PBRMaterial(baseColorTexture=img, metallicFactor=0.0))
    trimesh.Scene(mesh).export(path)                     # variant (i): no NORMAL written
    return path


def _first_prim_report(path):
    import pygltflib
    g = pygltflib.GLTF2().load(path)
    prim = g.meshes[0].primitives[0]
    a = prim.attributes
    pos = g.accessors[a.POSITION].count
    nrm = g.accessors[a.NORMAL].count if a.NORMAL is not None else None
    return dict(normal=a.NORMAL is not None, normal_count=nrm, pos_count=pos,
                uv=a.TEXCOORD_0 is not None, materials=len(g.materials or []),
                images=len(g.images or []))


class TestApplyToGlb(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.glb = os.path.join(self.d, "m.glb")
        _build_no_normal_glb(self.glb)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_precondition_no_normal(self):
        self.assertFalse(_first_prim_report(self.glb)["normal"])

    def test_writes_normal_preserves_material(self):
        ok = sn.apply_to_glb(self.glb, crease_deg=45.0)
        self.assertTrue(ok)
        r = _first_prim_report(self.glb)
        self.assertTrue(r["normal"])
        self.assertEqual(r["normal_count"], r["pos_count"])   # per-vertex
        self.assertTrue(r["uv"])
        self.assertGreaterEqual(r["materials"], 1)
        self.assertGreaterEqual(r["images"], 1)

    def test_idempotent(self):
        self.assertTrue(sn.apply_to_glb(self.glb))
        self.assertTrue(sn.apply_to_glb(self.glb))            # second run still valid
        r = _first_prim_report(self.glb)
        self.assertTrue(r["normal"] and r["normal_count"] == r["pos_count"])

    def test_malformed_returns_false_and_untouched(self):
        bad = os.path.join(self.d, "bad.glb")
        with open(bad, "wb") as f:
            f.write(b"not a glb at all")
        before = pathlib.Path(bad).read_bytes()
        self.assertFalse(sn.apply_to_glb(bad))
        self.assertEqual(pathlib.Path(bad).read_bytes(), before)  # byte-identical

    def test_verify_failure_leaves_file_byte_identical(self):
        # export succeeds but _verify_glb rejects it -> return False, original untouched
        before = pathlib.Path(self.glb).read_bytes()
        with unittest.mock.patch.object(sn, "_verify_glb", return_value=False):
            ok = sn.apply_to_glb(self.glb, crease_deg=45.0)
        self.assertFalse(ok)
        self.assertEqual(pathlib.Path(self.glb).read_bytes(), before)  # byte-identical


class TestJpegPreservation(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.glb = os.path.join(self.d, "jpeg.glb")
        self._build_jpeg_glb(self.glb)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    @staticmethod
    def _build_jpeg_glb(path):
        """Like _build_no_normal_glb but the baseColorTexture is a REAL
        JPEG-encoded PIL image (img.format == 'JPEG')."""
        import trimesh
        from trimesh.visual.material import PBRMaterial
        from PIL import Image
        P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                      [1, 0, 1], [1, 1, 1]], float)
        F = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]])
        UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [1, 0], [1, 1]], float)
        mesh = trimesh.Trimesh(vertices=P, faces=F, process=False)
        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (200, 160, 40)).save(buf, format="JPEG")
        img = Image.open(io.BytesIO(buf.getvalue()))
        assert img.format == "JPEG"                          # precondition
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=UV, material=PBRMaterial(baseColorTexture=img, metallicFactor=0.0))
        trimesh.Scene(mesh).export(path)
        return path

    def test_jpeg_texture_stays_jpeg(self):
        import pygltflib
        ok = sn.apply_to_glb(self.glb, crease_deg=45.0)
        self.assertTrue(ok)
        g = pygltflib.GLTF2().load(self.glb)
        self.assertEqual(g.images[0].mimeType, "image/jpeg")  # NOT image/png (no re-encode)
        prim = g.meshes[0].primitives[0]
        self.assertIsNotNone(prim.attributes.NORMAL)          # NORMAL now present


if __name__ == "__main__":
    unittest.main()
