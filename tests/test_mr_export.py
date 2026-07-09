# tests/test_mr_export.py
import os, tempfile, unittest
import numpy as np
from PIL import Image
import trimesh
import mr_export

class TestMRExport(unittest.TestCase):
    def test_pack_channels(self):
        m = Image.fromarray(np.full((8, 8), 50, np.uint8))
        r = Image.fromarray(np.full((8, 8), 200, np.uint8))
        out = np.asarray(mr_export.pack_metallic_roughness(m, r))
        self.assertTrue((out[..., 0] == 255).all())   # R unused
        self.assertTrue((out[..., 1] == 200).all())    # G = roughness
        self.assertTrue((out[..., 2] == 50).all())      # B = metallic

    def _write_obj(self, d):
        base = os.path.join(d, "textured")
        Image.fromarray(np.full((8,8,3),128,np.uint8)).save(base+".png")
        Image.fromarray(np.full((8,8),30,np.uint8)).save(base+"_metallic.png")
        Image.fromarray(np.full((8,8),210,np.uint8)).save(base+"_roughness.png")
        with open(base+".mtl","w") as f:
            f.write("newmtl m\nmap_Kd textured.png\n")
        with open(base+".obj","w") as f:
            f.write("mtllib textured.mtl\nusemtl m\n"
                    "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                    "vt 0 0\nvt 1 0\nvt 0 1\n"
                    "f 1/1 2/2 3/3\n")
        return base+".obj"

    def test_build_glb_wires_mr(self):
        with tempfile.TemporaryDirectory() as d:
            obj = self._write_obj(d)
            glb = os.path.join(d, "out.glb")
            self.assertTrue(mr_export.build_glb_with_mr(obj, glb))
            scene = trimesh.load(glb, process=False)
            g = list(scene.geometry.values())[0]
            mat = g.visual.material
            self.assertIsNotNone(mat.metallicRoughnessTexture)
            mr = np.asarray(mat.metallicRoughnessTexture.convert("RGB"))
            self.assertTrue((mr[..., 1] == 210).all())   # roughness -> G
            self.assertTrue((mr[..., 2] == 30).all())     # metallic  -> B

    def test_build_glb_flat_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            obj = self._write_obj(d)
            os.remove(obj[:-4] + "_metallic.png")   # drop MR
            glb = os.path.join(d, "out.glb")
            self.assertTrue(mr_export.build_glb_with_mr(obj, glb))
            g = list(trimesh.load(glb, process=False).geometry.values())[0]
            self.assertIsNone(g.visual.material.metallicRoughnessTexture)

    def _write_obj_jpg(self, d):
        # Real paint writes .jpg MR siblings (vendored _save_texture_map default),
        # and the MTL map_Kd points at a .jpg albedo. This is the on-disk case the
        # .png-only tests missed.
        base = os.path.join(d, "textured")
        Image.fromarray(np.full((8,8,3),128,np.uint8)).save(base+".jpg")
        Image.fromarray(np.full((8,8),30,np.uint8)).save(base+"_metallic.jpg")
        Image.fromarray(np.full((8,8),210,np.uint8)).save(base+"_roughness.jpg")
        with open(base+".mtl","w") as f:
            f.write("newmtl m\nmap_Kd textured.jpg\n")
        with open(base+".obj","w") as f:
            f.write("mtllib textured.mtl\nusemtl m\n"
                    "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                    "vt 0 0\nvt 1 0\nvt 0 1\n"
                    "f 1/1 2/2 3/3\n")
        return base+".obj"

    def test_build_glb_wires_mr_from_jpg(self):
        # Regression for C1: .jpg MR siblings must still wire the MR atlas.
        with tempfile.TemporaryDirectory() as d:
            obj = self._write_obj_jpg(d)
            glb = os.path.join(d, "out.glb")
            self.assertTrue(mr_export.build_glb_with_mr(obj, glb))
            scene = trimesh.load(glb, process=False)
            g = list(scene.geometry.values())[0]
            mat = g.visual.material
            self.assertIsNotNone(mat.metallicRoughnessTexture)
            mr = np.asarray(mat.metallicRoughnessTexture.convert("RGB"))
            # jpg is lossy; allow a small tolerance on the near-uniform fill.
            self.assertLess(abs(int(mr[..., 1].mean()) - 210), 5)   # roughness -> G
            self.assertLess(abs(int(mr[..., 2].mean()) - 30), 5)    # metallic  -> B
