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
        # 24 GB free, ceiling High: High peak 13.5 + margin 6 = 19.5 <= 24 -> High,
        # which is the stock (render 2048 / texture 4096) tier. Fits pure VRAM -> no warning.
        p = capacity.plan_texture_memory(24, "high")
        self.assertEqual(p.tier, "high")
        self.assertEqual((p.render_size, p.texture_size, p.sr_chunk), (2048, 4096, 4))
        self.assertIsNone(p.warning)

    def test_ceiling_clamps_below_band(self):
        # 24 GB free would allow High, but the user's ceiling is Low.
        p = capacity.plan_texture_memory(24, "low")
        self.assertEqual(p.tier, "low")
        self.assertEqual((p.render_size, p.texture_size, p.sr_chunk), (1536, 2048, 2))

    def test_ceiling_gated_when_tight_falls_to_low(self):
        # The upper tiers (balanced/high/max) now share peak 13.5 -> threshold 13.5+6=19.5;
        # Low's threshold is 12.5+6=18.5. At 19 GB free the Balanced ceiling can't fit
        # (19.5 > 19) so the planner gates DOWN to Low (18.5 <= 19), the reduced fallback tier.
        p = capacity.plan_texture_memory(19, "balanced")
        self.assertEqual(p.tier, "low")
        self.assertEqual((p.render_size, p.texture_size, p.sr_chunk), (1536, 2048, 2))

    def test_below_floor_warns_and_hints_offload(self):
        p = capacity.plan_texture_memory(17, "balanced")
        self.assertEqual(p.tier, "low")           # best effort
        self.assertTrue(p.offload_hint)
        self.assertIsNotNone(p.warning)
        self.assertIn("available", p.warning)

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


class TestMaxTier(unittest.TestCase):
    def test_max_reachable_via_ceiling(self):
        # 24 GB free, ceiling Max: Max peak 13.5 + margin 6 = 19.5 <= 24 -> Max,
        # the stock tier (render 2048 / texture 4096).
        p = capacity.plan_texture_memory(24, "max")
        self.assertEqual(p.tier, "max")
        self.assertEqual((p.render_size, p.texture_size, p.sr_chunk), (2048, 4096, 4))

    def test_ceiling_high_never_gives_max(self):
        p = capacity.plan_texture_memory(24, "high")
        self.assertEqual(p.tier, "high")


class TestSharedRamAllowance(unittest.TestCase):
    def test_headroom_leg_when_ram_tight(self):
        # 64 total, 40 available -> min(32, 40-12=28) = 28
        self.assertAlmostEqual(capacity.shared_ram_allowance(64, 40), 28.0)

    def test_fifty_percent_cap_when_ram_plentiful(self):
        # 64 total, 64 available -> min(32, 52) = 32
        self.assertAlmostEqual(capacity.shared_ram_allowance(64, 64), 32.0)

    def test_floors_at_zero_when_no_headroom(self):
        # 64 total, 10 available -> min(32, -2) -> clamped 0
        self.assertEqual(capacity.shared_ram_allowance(64, 10), 0.0)

    def test_bad_input_returns_zero(self):
        self.assertEqual(capacity.shared_ram_allowance("x", 40), 0.0)


class TestExtraBudget(unittest.TestCase):
    def test_extra_budget_makes_max_reachable(self):
        # 10 GB free VRAM alone can't fit Max, but +20 GB shared can.
        p = capacity.plan_texture_memory(10, "max", extra_budget_gb=20)
        self.assertEqual(p.tier, "max")
        self.assertIsNotNone(p.warning)
        self.assertIn("page", p.warning.lower())

    def test_no_paging_warning_when_fits_pure_vram(self):
        p = capacity.plan_texture_memory(24, "max")   # fits in real VRAM
        self.assertEqual(p.tier, "max")
        self.assertIsNone(p.warning)

    def test_extra_budget_zero_is_unchanged(self):
        a = capacity.plan_texture_memory(22, "high")
        b = capacity.plan_texture_memory(22, "high", extra_budget_gb=0.0)
        self.assertEqual(a, b)

    def test_toggle_off_never_pages(self):
        # 22 GB free, no extra budget: High fits its budget (13.5+6=19.5 <= 22) and its
        # peak 13.5 <= 22 free VRAM, so it runs entirely in real VRAM -> no paging warning.
        p = capacity.plan_texture_memory(22, "high")
        self.assertEqual(p.tier, "high")
        self.assertIsNone(p.warning)


if __name__ == "__main__":
    unittest.main()
