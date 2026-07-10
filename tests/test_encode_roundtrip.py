import os
import tempfile
import unittest
import numpy as np
import trimesh
from PIL import Image
import normal_bake


class TestEncodeRoundTrip(unittest.TestCase):
    def test_known_tbn_recovers_world_normal(self):
        # Known TBN: T=+X, w=-1, N=+Z  =>  B = w*cross(N,T) = -Y
        H = W = 2
        tex_T = np.tile([1.0, 0.0, 0.0], (H, W, 1))
        tex_N = np.tile([0.0, 0.0, 1.0], (H, W, 1))
        tex_w = -np.ones((H, W))
        mask = np.ones((H, W), bool)
        n = np.array([0.3, -0.5, 0.8])
        n /= np.linalg.norm(n)
        world = np.tile(n, (H, W, 1))
        rgb = normal_bake.encode_tangent_space(world, mask, tex_T, tex_w, tex_N)
        # decode the way a glTF viewer does: sample*2-1 in the SAME TBN
        T = tex_T[0, 0]
        N = tex_N[0, 0]
        B = tex_w[0, 0] * np.cross(N, T)
        # encoded channels must sit within 1/255 of the exact encoding
        target01 = np.array([n @ T, n @ B, n @ N]) * 0.5 + 0.5
        got01 = rgb.astype(np.float64) / 255.0
        self.assertTrue(np.all(np.abs(got01 - target01) <= 1.0 / 255.0 + 1e-9))
        # reconstructed world direction matches
        s = got01 * 2.0 - 1.0
        dec = s[..., 0:1] * T + s[..., 1:2] * B + s[..., 2:3] * N
        dec /= np.linalg.norm(dec, axis=2, keepdims=True)
        self.assertGreater(float((dec * world).sum(axis=2).min()), 0.9999)

    def test_handedness_w_participates_in_encode(self):
        # Same normal, opposite w: the G channel must flip around 128 —
        # proves B = w*cross(N,T) actually uses w.
        H = W = 2
        tex_T = np.tile([1.0, 0.0, 0.0], (H, W, 1))
        tex_N = np.tile([0.0, 0.0, 1.0], (H, W, 1))
        mask = np.ones((H, W), bool)
        n = np.array([0.3, -0.5, 0.8])
        n /= np.linalg.norm(n)
        world = np.tile(n, (H, W, 1))
        g_neg = normal_bake.encode_tangent_space(
            world, mask, tex_T, -np.ones((H, W)), tex_N)[0, 0, 1]
        g_pos = normal_bake.encode_tangent_space(
            world, mask, tex_T, np.ones((H, W)), tex_N)[0, 0, 1]
        self.assertGreater(int(g_neg), 160)   # n.(-Y flipped) = +0.505
        self.assertLess(int(g_pos), 96)       # n.(+cross)     = -0.505

    def test_neutral_fill_under_mask(self):
        H = W = 4
        tex_T = np.tile([1.0, 0.0, 0.0], (H, W, 1))
        tex_N = np.tile([0.0, 0.0, 1.0], (H, W, 1))
        tex_w = np.ones((H, W))
        world = tex_N.copy()
        mask = np.zeros((H, W), bool)
        mask[1:3, 1:3] = True
        rgb = normal_bake.encode_tangent_space(world, mask, tex_T, tex_w, tex_N)
        self.assertTrue((rgb[~mask] == [128, 128, 255]).all())


class TestVFlipCanary(unittest.TestCase):
    def test_atlas_placement_and_handedness_match_accessor_uvs(self):
        # Low quad textured into only the v in [0, 0.25] strip of the atlas
        # (authoring/trimesh convention); the GLB accessor carries v in [0.75, 1].
        # The bake must place detail where the ACCESSOR UVs say via the same
        # uv_to_texel helper the rasterizer uses, and the exported handedness
        # must reflect the accessor-space (V-flipped) UVs: w = -1.
        v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
        f = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
        uv = np.array([[0, 0], [1, 0], [1, 0.25], [0, 0.25]], float)
        m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        mat = trimesh.visual.material.PBRMaterial(baseColorFactor=[200, 200, 200, 255])
        m.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
        with tempfile.TemporaryDirectory() as d:
            glb = os.path.join(d, "low.glb")
            # production-style export: NO NORMAL accessor (mr_export/seam_fix path)
            m.export(glb)
            # dense: the same quad tilted along +X (z = 0.09x) -> world normals
            # tilt -X, so covered texels encode R clearly below 128.
            dv = np.array([[0, 0, 0], [1, 0, 0.09], [1, 1, 0.09], [0, 1, 0]], float)
            dense = trimesh.Trimesh(vertices=dv, faces=f, process=False)
            self.assertTrue(normal_bake.bake_normal_map(dense, glb, size=64))

            arr = np.asarray(Image.open(glb[:-4] + "_normal.png").convert("RGB")).astype(float)
            # the shared helper predicts where accessor UV (0.5, 0.875) lands
            col, row = normal_bake.uv_to_texel(np.array([[0.5, 0.875]]), 64)[0]
            lit = arr[int(round(row)), int(round(col))]
            self.assertLess(lit[0], 120.0)
            # the V-mirrored texel is far outside the strip: untouched neutral.
            # A V-flip bug swaps these two texels.
            mrow = int(round((1.0 - 0.875) * 63))
            self.assertTrue((arr[mrow, int(round(col))] == [128, 128, 255]).all())

            # exported TANGENT agrees with the accessor-space UVs
            import pygltflib
            g = pygltflib.GLTF2().load(glb)
            prim = g.meshes[0].primitives[0]
            self.assertIsNotNone(prim.attributes.TANGENT)
            a = g.accessors[prim.attributes.TANGENT]
            bv = g.bufferViews[a.bufferView]
            off = (bv.byteOffset or 0) + (a.byteOffset or 0)
            tw = np.frombuffer(g.binary_blob(), dtype="<f4",
                               count=a.count * 4, offset=off).reshape(-1, 4)
            np.testing.assert_allclose(tw[:, :3], np.tile([1.0, 0, 0], (a.count, 1)),
                                       atol=1e-5)
            np.testing.assert_allclose(tw[:, 3], -1.0)


if __name__ == "__main__":
    unittest.main()
