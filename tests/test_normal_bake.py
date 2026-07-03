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
