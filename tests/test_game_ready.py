import os, tempfile, unittest
from unittest import mock
import numpy as np, trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
import game_ready


def _make_dense_glb(path):
    m = trimesh.creation.icosphere(subdivisions=3)  # ~1280 faces
    img = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), np.uint8), "RGB")
    uv = np.random.rand(len(m.vertices), 2).astype(np.float32)
    m.visual = trimesh.visual.TextureVisuals(uv=uv, material=PBRMaterial(baseColorTexture=img))
    m.export(path)
    return len(m.faces)


class TestGameReady(unittest.TestCase):
    def test_produces_lowpoly_glb_with_textures(self):
        with tempfile.TemporaryDirectory() as d:
            dense = os.path.join(d, "dense.glb")
            nf = _make_dense_glb(dense)
            out = game_ready.to_game_ready(dense, target_triangles=400, tex_size=128)
            self.assertNotEqual(out, dense)
            self.assertTrue(os.path.exists(out))
            geom = trimesh.load(out, process=False)
            geom = list(geom.geometry.values())[0] if hasattr(geom, "geometry") else geom
            self.assertLess(len(geom.faces), nf)  # fewer faces than the dense input
            self.assertIsNotNone(getattr(geom.visual.material, "baseColorTexture", None))

    def test_xatlas_failure_returns_dense(self):
        with tempfile.TemporaryDirectory() as d:
            dense = os.path.join(d, "dense.glb")
            _make_dense_glb(dense)
            import xatlas
            with mock.patch.object(xatlas, "parametrize", side_effect=RuntimeError("boom")):
                out = game_ready.to_game_ready(dense, target_triangles=400, tex_size=128)
            self.assertEqual(out, dense)  # dense GLB, never the old atlas on new topology

    def test_no_albedo_returns_dense(self):
        with tempfile.TemporaryDirectory() as d:
            dense = os.path.join(d, "plain.glb")
            trimesh.creation.box().export(dense)  # no texture
            self.assertEqual(game_ready.to_game_ready(dense, 200, 64), dense)


if __name__ == "__main__":
    unittest.main()
