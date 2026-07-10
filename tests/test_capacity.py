import re
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
        # 23 GB free: shape 512 needs ~14, full-GPU textures ~20.4 -> both fit, no warning.
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

    def test_texture_warning_reassures_when_reduced_fits(self):
        # 15 GB free: full-GPU textures need ~20 (too big) but the reduced-VRAM path
        # (~13) fits -> reassure that Auto will use it rather than scaring the user.
        octree, warn = capacity.vram_plan(15, 24, True, 256, 512, 6)
        self.assertEqual(octree, 256)
        self.assertIsNotNone(warn)
        self.assertIn("Textures need", warn)
        self.assertIn("reduced-VRAM path", warn)

    def test_texture_warning_scary_when_even_reduced_wont_fit(self):
        # 10 GB free: neither full-GPU (~20) nor reduced (~13) fits -> honest OOM advice.
        octree, warn = capacity.vram_plan(10, 24, True, 256, 512, 6)
        self.assertEqual(octree, 256)
        self.assertIsNotNone(warn)
        self.assertIn("Textures need", warn)
        self.assertIn("out-of-memory", warn)

    def test_no_texture_no_texture_warning(self):
        octree, warn = capacity.vram_plan(15, 24, False, 256, 512, 6)
        self.assertIsNone(warn)

    def test_paint_vram_scales_up(self):
        self.assertGreater(capacity.paint_vram(768, 9), capacity.paint_vram(512, 6))

    def test_paint_vram_is_measured_stock_peak(self):
        # 512/6v full-GPU peak is the measured 20.4 GB reserved.
        self.assertAlmostEqual(capacity.paint_vram(512, 6), 20.4, places=3)


class TestTierResolution(unittest.TestCase):
    def test_legacy_ids_map_forward(self):
        # balanced|high|max -> auto; low -> reduced; unknown -> auto.
        self.assertEqual(capacity.plan_texture_memory(24, "balanced").tier, "standard")
        self.assertEqual(capacity.plan_texture_memory(24, "high").tier, "standard")
        self.assertEqual(capacity.plan_texture_memory(24, "max").tier, "standard")
        low = capacity.plan_texture_memory(24, "low")
        self.assertEqual(low.tier, "reduced")
        self.assertTrue(low.offload)

    def test_unknown_ceiling_treated_as_auto(self):
        # nonsense -> auto -> on a 24 GB card auto resolves to standard.
        p = capacity.plan_texture_memory(24, "nonsense")
        self.assertEqual(p.tier, "standard")
        self.assertFalse(p.offload)

    def test_all_tiers_use_stock_sizes(self):
        for tier in ("auto", "standard", "reduced"):
            p = capacity.plan_texture_memory(24, tier)
            self.assertEqual((p.render_size, p.texture_size, p.sr_chunk), (2048, 4096, 4))


class TestAutoTier(unittest.TestCase):
    def test_auto_picks_standard_when_it_fits(self):
        # 24 GB free >= need_standard (~21.9) -> full-GPU standard, offload off, no warning.
        p = capacity.plan_texture_memory(24, "auto")
        self.assertEqual(p.tier, "standard")
        self.assertFalse(p.offload)
        self.assertIsNone(p.warning)

    def test_auto_picks_reduced_when_tight_but_fits_reduced(self):
        # 16 GB free < need_standard -> reduced (offload on); reduced peak ~13 <= 16 -> no warning.
        p = capacity.plan_texture_memory(16, "auto")
        self.assertEqual(p.tier, "reduced")
        self.assertTrue(p.offload)
        self.assertIsNone(p.warning)

    def test_auto_reduced_default_ceiling(self):
        # Default tier_ceiling is auto.
        p = capacity.plan_texture_memory(16)
        self.assertEqual(p.tier, "reduced")
        self.assertTrue(p.offload)

    def test_monotonic_more_vram_never_offloads_more(self):
        # More VRAM must never force MORE offloading (never step from standard back to reduced).
        lo = capacity.plan_texture_memory(18, "auto")
        hi = capacity.plan_texture_memory(24, "auto")
        self.assertFalse(hi.offload and not lo.offload)   # more VRAM -> not worse
        self.assertGreaterEqual(hi.render_size, lo.render_size)


class TestForcedTiers(unittest.TestCase):
    def test_forced_standard_respected(self):
        p = capacity.plan_texture_memory(24, "standard")
        self.assertEqual(p.tier, "standard")
        self.assertFalse(p.offload)

    def test_forced_reduced_respected_at_stock_sizes(self):
        p = capacity.plan_texture_memory(24, "reduced")
        self.assertEqual(p.tier, "reduced")
        self.assertTrue(p.offload)
        self.assertEqual((p.render_size, p.texture_size, p.sr_chunk), (2048, 4096, 4))

    def test_forced_standard_warns_honestly_when_over_budget(self):
        # 16 GB free, forced standard (~20.4 peak) -> exceeds budget -> hint + honest need.
        p = capacity.plan_texture_memory(16, "standard")
        self.assertEqual(p.tier, "standard")
        self.assertTrue(p.offload_hint)
        self.assertIsNotNone(p.warning)
        need = int(re.search(r"need ~(\d+) GB", p.warning).group(1))
        # printed need = peak(20.4)+margin(1.5) = ~22 -> honestly exceeds the 16 GB budget.
        self.assertGreaterEqual(need, 21)
        self.assertGreater(need, 16)


class TestBelowFloor(unittest.TestCase):
    def test_below_floor_warns_and_hints_offload(self):
        # 10 GB free: even the reduced path (~13 peak) can't fit -> best effort + hint.
        p = capacity.plan_texture_memory(10, "auto")
        self.assertEqual(p.tier, "reduced")          # best effort
        self.assertTrue(p.offload)
        self.assertTrue(p.offload_hint)
        self.assertIsNotNone(p.warning)
        self.assertIn("available", p.warning)

    def test_floor_warning_need_includes_margin(self):
        # The fit test uses peak+MARGIN, so the printed 'need' must include the margin too.
        p = capacity.plan_texture_memory(10, "auto")
        self.assertIsNotNone(p.warning)
        need = int(re.search(r"need ~(\d+) GB", p.warning).group(1))
        # reduced peak (13.0) + margin (1.5) = 14.5 -> rounds to 14; margin bumps it above
        # the bare peak (13) and above the budget (10), i.e. the advice is honest.
        self.assertGreaterEqual(need, 14)
        self.assertGreater(need, 10)

    def test_unreadable_free_is_conservative(self):
        # None free -> budget 0 -> reduced best-effort with the offload hint.
        p = capacity.plan_texture_memory(None, "high")
        self.assertEqual(p.tier, "reduced")
        self.assertTrue(p.offload)
        self.assertTrue(p.offload_hint)


class TestHiResDemand(unittest.TestCase):
    def test_hi_res_diffusion_costs_more(self):
        # tex_res 768 adds ~14 GB -> at 22 GB free, 512 -> standard, 768 -> reduced (tighter).
        p512 = capacity.plan_texture_memory(22, "auto", tex_resolution=512)
        p768 = capacity.plan_texture_memory(22, "auto", tex_resolution=768)
        order = {"reduced": 0, "standard": 1}
        self.assertLessEqual(order[p768.tier], order[p512.tier])

    def test_768_gates_to_shared_ram_advice(self):
        # 768 + staging (~27 GB) exceeds any consumer card -> shared-RAM advice.
        p = capacity.plan_texture_memory(22, "reduced", tex_resolution=768)
        self.assertIsNotNone(p.warning)
        self.assertIn("shared GPU memory", p.warning)


class TestApplyTexturePlan(unittest.TestCase):
    def test_sets_conf_fields(self):
        plan = capacity.TexturePlan(2048, 4096, 4, "standard", False, False, None)
        class _Conf: pass
        c = _Conf()
        capacity.apply_texture_plan(c, plan)
        self.assertEqual((c.render_size, c.texture_size, c.sr_chunk), (2048, 4096, 4))

    def test_plan_has_offload_field(self):
        self.assertIn("offload", capacity.TexturePlan._fields)


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
    def test_extra_budget_makes_standard_reachable(self):
        # 10 GB free VRAM alone can't fit standard, but +20 GB shared can -> pages.
        p = capacity.plan_texture_memory(10, "standard", extra_budget_gb=20)
        self.assertEqual(p.tier, "standard")
        self.assertIsNotNone(p.warning)
        self.assertIn("page", p.warning.lower())

    def test_no_paging_warning_when_fits_pure_vram(self):
        p = capacity.plan_texture_memory(24, "standard")   # fits in real VRAM
        self.assertEqual(p.tier, "standard")
        self.assertIsNone(p.warning)

    def test_extra_budget_zero_is_unchanged(self):
        a = capacity.plan_texture_memory(22, "auto")
        b = capacity.plan_texture_memory(22, "auto", extra_budget_gb=0.0)
        self.assertEqual(a, b)

    def test_toggle_off_never_pages(self):
        # 22 GB free, no extra budget: standard fits (21.9 <= 22) and peak 20.4 <= 22 -> no paging.
        p = capacity.plan_texture_memory(22, "auto")
        self.assertEqual(p.tier, "standard")
        self.assertIsNone(p.warning)


if __name__ == "__main__":
    unittest.main()
