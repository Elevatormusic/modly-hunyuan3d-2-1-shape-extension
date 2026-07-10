import os
import tempfile
import unittest
import numpy as np
import normal_bake


def quad(mirror_u=False):
    # unit quad in XY plane, UVs axis-aligned -> analytic T=+X (or -X mirrored), w=+1 (or -1)
    P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    N = np.tile([0, 0, 1.0], (4, 1))
    u = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    if mirror_u:
        u[:, 0] = 1.0 - u[:, 0]
    F = np.array([[0, 1, 2], [0, 2, 3]])
    return P, N, u, F


class TestComputeUvTangents(unittest.TestCase):
    def test_axis_aligned_quad_tangent_plus_x(self):
        P, N, u, F = quad()
        T, w = normal_bake.compute_uv_tangents(P, N, u, F)
        np.testing.assert_allclose(T, np.tile([1, 0, 0.0], (4, 1)), atol=1e-6)
        np.testing.assert_allclose(w, np.ones(4), atol=1e-6)

    def test_mirrored_uv_flips_tangent_and_handedness(self):
        P, N, u, F = quad(mirror_u=True)
        T, w = normal_bake.compute_uv_tangents(P, N, u, F)
        np.testing.assert_allclose(T, np.tile([-1, 0, 0.0], (4, 1)), atol=1e-6)
        np.testing.assert_allclose(w, -np.ones(4), atol=1e-6)

    def test_degenerate_uv_gets_safe_fallback(self):
        P, N, u, F = quad()
        u[:] = 0.25                       # zero UV area everywhere
        T, w = normal_bake.compute_uv_tangents(P, N, u, F)
        self.assertTrue(np.all(np.isfinite(T)) and np.all(np.isfinite(w)))
        np.testing.assert_allclose(np.linalg.norm(T, axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose((T * N).sum(axis=1), 0.0, atol=1e-6)

    def test_invariants_unit_and_orthogonal(self):
        rng = np.random.default_rng(7)
        P = rng.normal(size=(30, 3)); N = rng.normal(size=(30, 3))
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        u = rng.uniform(size=(30, 2)); F = rng.integers(0, 30, size=(40, 3))
        T, w = normal_bake.compute_uv_tangents(P, N, u, F)
        np.testing.assert_allclose(np.linalg.norm(T, axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose((T * N).sum(axis=1), 0.0, atol=1e-5)
        self.assertTrue(np.all(np.isin(w, (-1.0, 1.0))))


class TestReadGlbArrays(unittest.TestCase):
    def test_roundtrip_trimesh_export(self):
        import trimesh
        from tests._fixtures import unit_uv_quad
        v, f, uv = unit_uv_quad()
        m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        mat = trimesh.visual.material.PBRMaterial(baseColorFactor=[200, 200, 200, 255])
        m.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
        _ = m.vertex_normals  # force NORMAL attribute on export
        with tempfile.TemporaryDirectory() as d:
            glb = os.path.join(d, "quad.glb")
            m.export(glb, include_normals=True)
            arr = normal_bake.read_glb_arrays(glb)
        self.assertEqual(len(arr["positions"]), len(v))
        self.assertEqual(len(arr["normals"]), len(v))
        self.assertEqual(len(arr["uvs"]), len(v))
        self.assertEqual(len(arr["faces"]), len(f))
        np.testing.assert_allclose(arr["positions"], v, atol=1e-6)
        # trimesh V-flips UVs on GLB export (glTF UV origin is top-left)
        np.testing.assert_allclose(arr["uvs"][:, 0], uv[:, 0], atol=1e-6)
        np.testing.assert_allclose(arr["uvs"][:, 1], 1.0 - uv[:, 1], atol=1e-6)

    def test_missing_normal_accessor_gets_trimesh_fallback(self):
        # Production GLBs (mr_export / seam_fix path) ship no NORMAL accessor;
        # read_glb_arrays must fall back to exactly the smooth normals
        # attach_normal_texture will ship at re-export.
        import trimesh
        from tests._fixtures import unit_uv_quad
        v, f, uv = unit_uv_quad()
        m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        mat = trimesh.visual.material.PBRMaterial(baseColorFactor=[200, 200, 200, 255])
        m.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
        with tempfile.TemporaryDirectory() as d:
            glb = os.path.join(d, "quad.glb")
            m.export(glb)  # no vertex_normals touched -> no NORMAL accessor
            import pygltflib
            g = pygltflib.GLTF2().load(glb)
            self.assertIsNone(g.meshes[0].primitives[0].attributes.NORMAL)
            arr = normal_bake.read_glb_arrays(glb)
        ref = trimesh.Trimesh(vertices=arr["positions"], faces=arr["faces"],
                              process=False).vertex_normals
        np.testing.assert_allclose(arr["normals"], ref, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(arr["normals"], axis=1), 1.0,
                                   atol=1e-6)

    def test_missing_texcoord_raises_valueerror(self):
        import trimesh
        from tests._fixtures import unit_uv_quad
        v, f, _ = unit_uv_quad()
        m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        with tempfile.TemporaryDirectory() as d:
            glb = os.path.join(d, "nouv.glb")
            m.export(glb)
            with self.assertRaises(ValueError):
                normal_bake.read_glb_arrays(glb)


if __name__ == "__main__":
    unittest.main()
