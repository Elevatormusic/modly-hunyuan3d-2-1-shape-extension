import unittest
import capacity


class TestShouldBake(unittest.TestCase):
    def test_bake_on_for_faithful_modes(self):
        self.assertTrue(capacity.should_bake(True, "isotropic"))
        self.assertTrue(capacity.should_bake(True, "regular"))

    def test_bake_skipped_for_bpt(self):
        self.assertFalse(capacity.should_bake(True, "bpt"))

    def test_bake_off_stays_off(self):
        self.assertFalse(capacity.should_bake(False, "isotropic"))
        self.assertFalse(capacity.should_bake(False, "bpt"))


class TestVramPlan(unittest.TestCase):
    def test_plenty_of_vram_no_change(self):
        octree, warn = capacity.vram_plan(23, 24, True, 512, 512, 6)
        self.assertEqual(octree, 512)
        self.assertIsNone(warn)

    def test_low_vram_caps_octree(self):
        # 6 GB free: 512 needs ~14, 384 ~10, 256 ~7 -> still >6 -> falls to 256
        octree, warn = capacity.vram_plan(6, 8, False, 512, 512, 6)
        self.assertEqual(octree, 256)
        self.assertIsNotNone(warn)
        self.assertIn("Mesh Resolution", warn)

    def test_mid_vram_caps_to_384(self):
        # 11 GB free: 512 needs ~14 (too big) -> 384 needs ~10 (fits) -> 384
        octree, warn = capacity.vram_plan(11, 12, False, 512, 512, 6)
        self.assertEqual(octree, 384)

    def test_texture_warning_when_tight(self):
        # 15 GB free, textures need ~22 -> warn (but octree 256 shape fits, no cap)
        octree, warn = capacity.vram_plan(15, 24, True, 256, 512, 6)
        self.assertEqual(octree, 256)
        self.assertIsNotNone(warn)
        self.assertIn("Textures need", warn)

    def test_no_texture_no_texture_warning(self):
        octree, warn = capacity.vram_plan(15, 24, False, 256, 512, 6)
        self.assertIsNone(warn)

    def test_paint_vram_scales_up(self):
        self.assertGreater(capacity.paint_vram(768, 9), capacity.paint_vram(512, 6))


if __name__ == "__main__":
    unittest.main()
