import os
import tempfile
import unittest
import numpy as np
import trimesh
from PIL import Image
from tests._fixtures import unit_uv_quad, icosphere
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


class TestDenseSampling(unittest.TestCase):
    def test_identity_normals_match_surface(self):
        m = icosphere(4)
        pts, fid = trimesh.sample.sample_surface(m, 2000)
        pos = np.zeros((45, 45, 3), np.float32)
        mask = np.zeros((45, 45), bool)
        flat = pos.reshape(-1, 3)
        mflat = mask.reshape(-1)
        flat[:2000] = pts
        mflat[:2000] = True
        pos = flat.reshape(45, 45, 3)
        mask = mflat.reshape(45, 45)
        wn = normal_bake.sample_dense_normals(m, pos, mask)
        got = wn.reshape(-1, 3)[:2000]
        # icosphere at origin: outward normal ~= normalized position
        ref = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        dots = np.sum(got * ref, axis=1)
        self.assertGreater(np.median(dots), 0.98)


def _ridge_highpoly(height=0.15, n=48):
    """A fine tent/ridge over [0,1]^2 peaking at x=0.5 (z=height), edges z=0."""
    xs = np.linspace(0, 1, n)
    ys = np.linspace(0, 1, n)
    gx, gy = np.meshgrid(xs, ys)
    gz = height * (1.0 - np.abs(2 * gx - 1))
    verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            faces.append([a, a + 1, a + n])
            faces.append([a + 1, a + n + 1, a + n])
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)


class TestEncode(unittest.TestCase):
    def test_uncovered_texels_are_neutral_not_black(self):
        size = 8
        low_nrm = np.tile([0, 0, 1.0], (size, size, 1)).astype(np.float32)
        world_nrm = low_nrm.copy()
        tex_T = np.tile([1.0, 0, 0], (size, size, 1))
        tex_w = np.ones((size, size))
        mask = np.zeros((size, size), bool)
        mask[2:5, 2:5] = True  # only the center is covered
        rgb = normal_bake.encode_tangent_space(world_nrm, mask, tex_T, tex_w, low_nrm)
        # uncovered MUST be neutral (128,128,255), never black (0,0,0)
        self.assertTrue((rgb[~mask] == [128, 128, 255]).all())
        self.assertFalse((rgb[~mask] == [0, 0, 0]).all())
        # covered-and-flat is also ~neutral
        self.assertTrue(np.allclose(rgb[mask].mean(0), [128, 128, 255], atol=4))


class TestBakeEndToEnd(unittest.TestCase):
    def _write_uv_glb(self, path):
        v, f, uv = unit_uv_quad()
        m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        mat = trimesh.visual.material.PBRMaterial(baseColorFactor=[200, 200, 200, 255])
        m.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
        _ = m.vertex_normals
        m.export(path, include_normals=True)

    def test_identity_bake_is_flat_map(self):
        # baking a flat quad's own plane -> tangent normal ~ (0,0,1) -> rgb ~ (128,128,255)
        with tempfile.TemporaryDirectory() as d:
            glb = os.path.join(d, "low.glb")
            self._write_uv_glb(glb)
            flat = trimesh.Trimesh(*unit_uv_quad()[:2], process=False)
            ok = normal_bake.bake_normal_map(flat, glb, size=64)
            self.assertTrue(ok)
            png = glb[:-4] + "_normal.png"
            self.assertTrue(os.path.exists(png))
            arr = np.asarray(Image.open(png).convert("RGB")).reshape(-1, 3)
            covered = arr[(arr != 0).any(1)]
            self.assertTrue(np.allclose(covered.mean(0), [128, 128, 255], atol=12))

    def test_ridge_detail_direction(self):
        # A ridge peaking at x=0.5 baked onto a flat quad (tangent U = +X):
        # left slope normal.x<0 -> R<128; right slope normal.x>0 -> R>128.
        # Catches handedness/sign inversion that the flat identity test cannot.
        with tempfile.TemporaryDirectory() as d:
            glb = os.path.join(d, "low.glb")
            self._write_uv_glb(glb)
            hi = _ridge_highpoly(height=0.15, n=48)
            normal_bake.bake_normal_map(hi, glb, size=128)
            arr = np.asarray(Image.open(glb[:-4] + "_normal.png").convert("RGB")).astype(float)
            W = arr.shape[1]
            left_r = arr[:, : W // 3, 0].mean()
            right_r = arr[:, 2 * W // 3:, 0].mean()
            self.assertLess(left_r, 118)     # left slope tilts -X
            self.assertGreater(right_r, 138)  # right slope tilts +X

    def test_opposed_surface_rejected_to_flat(self):
        # A dense surface just above the low quad but facing DOWN (normal -Z) is a
        # wrong-surface (back-face) hit; the reject must fall back to the low +Z normal
        # -> flat (128,128,255), NOT an inverted (128,128,~0) bump.
        with tempfile.TemporaryDirectory() as d:
            glb = os.path.join(d, "low.glb")
            self._write_uv_glb(glb)
            v = np.array([[0, 0, 0.02], [1, 0, 0.02], [1, 1, 0.02], [0, 1, 0.02]], float)
            f = np.array([[0, 2, 1], [0, 3, 2]], np.int64)  # reversed winding -> normal -Z
            dense_down = trimesh.Trimesh(vertices=v, faces=f, process=False)
            self.assertLess(dense_down.face_normals[0][2], 0)  # confirm it faces -Z
            normal_bake.bake_normal_map(dense_down, glb, size=64)
            arr = np.asarray(Image.open(glb[:-4] + "_normal.png").convert("RGB")).reshape(-1, 3)
            covered = arr[(arr != 0).any(1)]
            self.assertGreater(covered[:, 2].mean(), 220)  # B ~ 255 (flat), not inverted

    def test_normaltexture_and_normal_attribute_survive(self):
        with tempfile.TemporaryDirectory() as d:
            glb = os.path.join(d, "low.glb")
            self._write_uv_glb(glb)
            flat = trimesh.Trimesh(*unit_uv_quad()[:2], process=False)
            normal_bake.bake_normal_map(flat, glb, size=64)
            reloaded = trimesh.load(glb, process=False)
            geom = (list(reloaded.geometry.values())[0]
                    if isinstance(reloaded, trimesh.Scene) else reloaded)
            self.assertIsNotNone(geom.visual.material.normalTexture)


if __name__ == "__main__":
    unittest.main()
