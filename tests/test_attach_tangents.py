import hashlib
import os
import tempfile
import unittest
import numpy as np
import trimesh
from PIL import Image
from tests._fixtures import unit_uv_quad
import normal_bake


def _write_textured_quad_glb(path):
    """Textured quad GLB with an embedded baseColor image (so the test can
    assert image bytes survive the TANGENT append untouched)."""
    v, f, uv = unit_uv_quad()
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    img = Image.new("RGB", (4, 4), (200, 120, 40))
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=img)
    m.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    _ = m.vertex_normals  # force NORMAL attribute on export
    m.export(path, include_normals=True)


def _acc_bytes(g, idx, elem_size):
    a = g.accessors[idx]
    bv = g.bufferViews[a.bufferView]
    off = (bv.byteOffset or 0) + (a.byteOffset or 0)
    return bytes(g.binary_blob()[off: off + a.count * elem_size])


def _image_bytes(g):
    bv = g.bufferViews[g.images[0].bufferView]
    off = bv.byteOffset or 0
    return bytes(g.binary_blob()[off: off + bv.byteLength])


class TestAttachTangents(unittest.TestCase):
    def test_appends_verified_tangent_accessor(self):
        import pygltflib
        with tempfile.TemporaryDirectory() as d:
            glb = os.path.join(d, "quad.glb")
            _write_textured_quad_glb(glb)

            before = pygltflib.GLTF2().load(glb)
            prim_b = before.meshes[0].primitives[0]
            n = before.accessors[prim_b.attributes.POSITION].count
            pos_b = _acc_bytes(before, prim_b.attributes.POSITION, 12)
            nrm_b = _acc_bytes(before, prim_b.attributes.NORMAL, 12)
            uv_b = _acc_bytes(before, prim_b.attributes.TEXCOORD_0, 8)
            img_b = _image_bytes(before)

            rng = np.random.default_rng(11)
            T = rng.normal(size=(n, 3))
            T /= np.linalg.norm(T, axis=1, keepdims=True)
            w = np.where(rng.uniform(size=n) < 0.5, -1.0, 1.0)

            self.assertTrue(normal_bake.attach_tangents(glb, T, w))

            after = pygltflib.GLTF2().load(glb)
            prim = after.meshes[0].primitives[0]
            self.assertIsNotNone(prim.attributes.TANGENT)
            acc = after.accessors[prim.attributes.TANGENT]
            self.assertEqual(acc.count, n)
            self.assertEqual(acc.type, "VEC4")
            self.assertEqual(acc.componentType, 5126)
            expected = np.hstack([T, w[:, None]]).astype("<f4").tobytes()
            self.assertEqual(_acc_bytes(after, prim.attributes.TANGENT, 16), expected)
            # every pre-existing attribute + the embedded image survive byte-identical
            self.assertEqual(_acc_bytes(after, prim.attributes.POSITION, 12), pos_b)
            self.assertEqual(_acc_bytes(after, prim.attributes.NORMAL, 12), nrm_b)
            self.assertEqual(_acc_bytes(after, prim.attributes.TEXCOORD_0, 8), uv_b)
            self.assertEqual(_image_bytes(after), img_b)

    def test_count_mismatch_returns_false_and_leaves_file_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            glb = os.path.join(d, "quad.glb")
            _write_textured_quad_glb(glb)
            h_before = hashlib.sha256(open(glb, "rb").read()).hexdigest()
            arr = normal_bake.read_glb_arrays(glb)
            n = len(arr["positions"])
            T = np.tile([1.0, 0.0, 0.0], (n + 1, 1))  # one too many
            w = np.ones(n + 1)
            self.assertFalse(normal_bake.attach_tangents(glb, T, w))
            h_after = hashlib.sha256(open(glb, "rb").read()).hexdigest()
            self.assertEqual(h_before, h_after)


if __name__ == "__main__":
    unittest.main()
