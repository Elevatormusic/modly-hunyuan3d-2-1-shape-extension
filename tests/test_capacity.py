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


class TestPlanTextureMemory(unittest.TestCase):
    def test_empty_card_picks_ceiling_high(self):
        p = capacity.plan_texture_memory(24, "high")
        self.assertEqual(p.tier, "high")
        self.assertEqual((p.render_size, p.texture_size, p.sr_chunk), (1536, 2048, 4))
        self.assertIsNone(p.warning)

    def test_ceiling_clamps_below_band(self):
        # 24 GB free would allow High, but the user's ceiling is Low.
        p = capacity.plan_texture_memory(24, "low")
        self.assertEqual(p.tier, "low")
        self.assertEqual((p.render_size, p.texture_size, p.sr_chunk), (1024, 1024, 1))

    def test_high_gated_when_tight_falls_to_balanced(self):
        # ~22 GB free: High (needs ~23.5) doesn't fit, Balanced (~21.5) does.
        p = capacity.plan_texture_memory(22, "high")
        self.assertEqual(p.tier, "balanced")
        self.assertEqual((p.render_size, p.texture_size, p.sr_chunk), (1024, 2048, 2))

    def test_below_floor_warns_and_hints_offload(self):
        p = capacity.plan_texture_memory(17, "balanced")
        self.assertEqual(p.tier, "low")           # best effort
        self.assertTrue(p.offload_hint)
        self.assertIsNotNone(p.warning)
        self.assertIn("VRAM", p.warning)

    def test_monotonic_more_vram_never_smaller(self):
        lo = capacity.plan_texture_memory(20, "high")
        hi = capacity.plan_texture_memory(24, "high")
        self.assertGreaterEqual(hi.render_size, lo.render_size)
        self.assertGreaterEqual(hi.texture_size, lo.texture_size)

    def test_unknown_ceiling_treated_as_balanced(self):
        p = capacity.plan_texture_memory(24, "nonsense")
        self.assertEqual(p.tier, "balanced")

    def test_unreadable_free_is_conservative(self):
        p = capacity.plan_texture_memory(None, "high")
        self.assertEqual(p.tier, "low")
        self.assertTrue(p.offload_hint)

    def test_hi_res_diffusion_costs_more(self):
        # texture_resolution=768 adds demand -> tighter fit than 512 at same free VRAM.
        p512 = capacity.plan_texture_memory(22, "high", tex_resolution=512)
        p768 = capacity.plan_texture_memory(22, "high", tex_resolution=768)
        order = {"low": 0, "balanced": 1, "high": 2}
        self.assertLessEqual(order[p768.tier], order[p512.tier])


class TestApplyTexturePlan(unittest.TestCase):
    def test_sets_conf_fields(self):
        plan = capacity.TexturePlan(1024, 2048, 2, "balanced", False, None)
        class _Conf: pass
        c = _Conf()
        capacity.apply_texture_plan(c, plan)
        self.assertEqual((c.render_size, c.texture_size, c.sr_chunk), (1024, 2048, 2))


if __name__ == "__main__":
    unittest.main()
