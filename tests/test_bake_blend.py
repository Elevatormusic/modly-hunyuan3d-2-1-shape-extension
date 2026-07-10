# tests/test_bake_blend.py
import contextlib
import io
import unittest
import numpy as np
import torch
import bake_blend as bb


def _flat(h, w, rgb):
    t = torch.zeros(h, w, 3)
    t[..., 0], t[..., 1], t[..., 2] = rgb
    return t


class TestMerge(unittest.TestCase):
    def test_weighted_average_exact(self):
        h = w = 8
        tA, tB = _flat(h, w, (1.0, 0.0, 0.0)), _flat(h, w, (0.0, 1.0, 0.0))
        cA = torch.full((h, w, 1), 3.0)
        cB = torch.full((h, w, 1), 1.0)
        tex, trust = bb.merge([tA, tB], [cA, cB])
        np.testing.assert_allclose(tex[0, 0].numpy(), [0.75, 0.25, 0.0], atol=1e-6)
        np.testing.assert_allclose(trust.numpy(), 4.0, atol=1e-6)

    def test_no_view_skip(self):
        # view B covers only 1 texel that A already covers (<1% new coverage:
        # the stock skip would drop B entirely). merge must still blend it.
        h = w = 16
        tA, tB = _flat(h, w, (1.0, 1.0, 1.0)), _flat(h, w, (0.0, 0.0, 0.0))
        cA = torch.full((h, w, 1), 1.0)
        cB = torch.zeros(h, w, 1)
        cB[0, 0, 0] = 1.0
        tex, _ = bb.merge([tA, tB], [cA, cB])
        np.testing.assert_allclose(tex[0, 0].numpy(), [0.5, 0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(tex[1, 1].numpy(), [1.0, 1.0, 1.0], atol=1e-6)

    def test_uncovered_texels_zero_trust(self):
        h = w = 4
        t = _flat(h, w, (1.0, 0.0, 0.0))
        c = torch.zeros(h, w, 1)
        c[0, 0, 0] = 1.0
        tex, trust = bb.merge([t], [c])
        self.assertFalse(bool((trust[1:, :] > 1e-8).any()))


class TestComputeRamps(unittest.TestCase):
    def test_halfplane_monotone_ramp(self):
        # visible right half; ramp must rise 0->1 over feather_px moving right
        h = w = 256
        c = torch.zeros(h, w, 1)
        c[:, w // 2:, 0] = 1.0
        (ramp,) = bb.compute_ramps([c], feather_px=32.0, ref_dim=w)
        self.assertEqual(tuple(ramp.shape), (h, w, 1))
        row = ramp[h // 2, :, 0].numpy()
        self.assertLess(row[w // 2], 0.25)              # near edge: small
        self.assertGreater(row[w // 2 + 40], 0.99)      # past feather: full
        self.assertTrue(np.all(np.diff(row[w // 2: w // 2 + 40]) >= -1e-6))  # monotone

    def test_interior_unchanged_and_masks(self):
        h = w = 128
        c = torch.zeros(h, w, 1)
        c[:, w // 2:, 0] = 1.0
        (ramp,) = bb.compute_ramps([c], feather_px=8.0, ref_dim=w)
        self.assertGreater(float(ramp[h // 2, -1, 0]), 0.99)   # deep interior = 1
        self.assertEqual(float(ramp[h // 2, 0, 0]), 0.0)       # invisible side = 0

    def test_full_and_empty_masks_safe(self):
        h = w = 64
        full = torch.ones(h, w, 1)
        empty = torch.zeros(h, w, 1)
        rf, re = bb.compute_ramps([full, empty], feather_px=8.0, ref_dim=w)
        self.assertGreater(float(rf.mean()), 0.9)
        self.assertEqual(float(re.max()), 0.0)

    def test_downsampled_grid_used_for_large(self):
        # 4096-sized dim triggers the <=2048 EDT grid; just verify it runs fast
        # and returns the right shape (correctness covered above at small size).
        h, w = 4096, 64  # thin strip keeps memory small; max dim triggers ds
        c = torch.zeros(h, w, 1)
        c[h // 2:, :, 0] = 1.0
        (ramp,) = bb.compute_ramps([c], feather_px=32.0, ref_dim=4096)
        self.assertEqual(tuple(ramp.shape), (h, w, 1))


class TestRampCache(unittest.TestCase):
    def test_put_take_and_prune(self):
        bb._RAMP_CACHE.clear()
        bb._cache_put(("k1",), [torch.ones(2, 2, 1)])
        bb._cache_put(("k2",), [torch.ones(2, 2, 1)])
        bb._cache_put(("k3",), [torch.ones(2, 2, 1)])   # prunes oldest
        self.assertLessEqual(len(bb._RAMP_CACHE), 2)
        self.assertIsNotNone(bb._cache_take(("k3",)))
        self.assertIsNone(bb._cache_take(("k3",)))       # take removes


def _gradient(h, w):
    x = torch.linspace(0.2, 0.8, w).repeat(h, 1)
    return torch.stack([x, x * 0.8, x * 0.6], dim=-1)


class TestHarmonize(unittest.TestCase):
    def _views(self, h=96, w=96, a=(1.0, 1.3), b=(0.0, -0.08)):
        base = _gradient(h, w)
        t0 = torch.clamp(base * a[0] + b[0], 0, 1)
        t1 = torch.clamp(base * a[1] + b[1], 0, 1)
        c0 = torch.zeros(h, w, 1); c0[:, : 2 * w // 3, 0] = 1.0   # left 2/3
        c1 = torch.zeros(h, w, 1); c1[:, w // 3:, 0] = 1.0        # right 2/3 (middle overlaps)
        return [t0, t1], [c0, c1], base

    def test_recovers_injected_gain_offset(self):
        (t, c, base) = self._views()
        t0_before = t[0].clone()                                  # harmonize now mutates t in place
        t1_before = t[1].clone()
        out = bb.harmonize_views(t, c, anchor=0)
        ov = slice(96 // 3, 2 * 96 // 3)                          # overlap columns
        before = float((t0_before[:, ov] - t1_before[:, ov]).abs().mean())
        after = float((out[0][:, ov] - out[1][:, ov]).abs().mean())
        self.assertLess(after, before / 5.0)                      # >=5x agreement
        self.assertTrue(torch.equal(out[0], t0_before))           # anchor untouched

    def test_harmonize_mutates_in_place(self):
        # in-place contract (VRAM fix): NO per-view clone. The SAME list and the
        # SAME tensor objects come back, with the non-anchor view corrected in place.
        (t, c, _) = self._views()
        v1_before = t[1].clone()
        out = bb.harmonize_views(t, c, anchor=0)
        self.assertIs(out, t)                                     # same list object, not a copy
        self.assertIs(out[1], t[1])                               # same tensor, written in place
        self.assertFalse(torch.equal(out[1], v1_before))         # view 1 really corrected

    def _thin_overlap_views(self, h=96, w=96, a1=1.06, b1=-0.02, ov_cols=4):
        # A REAL thin overlap: left block [0,split), right block [split-ov_cols, w).
        # ov_cols=4 -> 4*96 = 384 overlap texels: passes the >=32 gate, ~4% of the
        # 9216-texel image, and modest injected distortion (gain 1.06, offset -0.02).
        base = _gradient(h, w)
        t0 = torch.clamp(base, 0, 1)
        t1 = torch.clamp(base * a1 + b1, 0, 1)
        split = w // 2
        c0 = torch.zeros(h, w, 1); c0[:, :split, 0] = 1.0
        c1 = torch.zeros(h, w, 1); c1[:, split - ov_cols:, 0] = 1.0
        return [t0, t1], [c0, c1]

    def test_thin_overlap_stays_near_identity(self):
        (t, c) = self._thin_overlap_views()
        t0_before = t[0].clone()                                  # harmonize now mutates t in place
        t1_before = t[1].clone()
        overlap = int(((c[0][..., 0] > 0) & (c[1][..., 0] > 0)).sum())
        self.assertGreaterEqual(overlap, 32)                      # gate not skipping the pair
        self.assertLess(overlap, 96 * 96 // 10)                   # genuinely thin (<10% of image)
        out = bb.harmonize_views(t, c, anchor=0)
        delta = float((out[1] - t1_before).abs().max())
        # non-vacuous: the ridge path actually ran (a skipped pair leaves out[1]==input
        # exactly, delta==0). Empirically delta==0.0107 at these params.
        self.assertGreater(delta, 1e-4)                           # correction is real, not skipped
        self.assertLess(delta, 0.03)                              # yet ridge keeps it near identity
        self.assertTrue(torch.equal(out[0], t0_before))           # anchor untouched

    def test_identical_calls_are_bit_identical(self):
        # two independent-but-identical input stacks (harmonize is in place now, so
        # re-running on the same object would double-correct - use fresh copies)
        (ta, ca, _) = self._views()
        (tb, cb, _) = self._views()
        ta1_before = ta[1].clone()
        out_a = bb.harmonize_views(ta, ca, anchor=0)
        out_b = bb.harmonize_views(tb, cb, anchor=0)
        self.assertFalse(torch.equal(out_a[1], ta1_before))      # non-vacuous: a real correction
        for xa, xb in zip(out_a, out_b):
            self.assertTrue(torch.equal(xa, xb))                  # deterministic to the bit

    def test_clamps_hold_on_adversarial_input(self):
        (t, c, _) = self._views(a=(1.0, 5.0), b=(0.0, 0.4))       # wild injected distortion
        t1_before = t[1].clone()                                  # harmonize now mutates t in place
        out = bb.harmonize_views(t, c, anchor=0)
        # correction applied to view 1 is a'*I + b' with a' in [0.5,2], |b'|<=64/255:
        # verify output stays a bounded transform of the (pre-harmonize) input
        ratio = (out[1][c[1][..., 0] > 0] + 1e-6) / (t1_before[c[1][..., 0] > 0] + 1e-6)
        self.assertLessEqual(float(ratio.max()), 2.6)             # 2.0 gain + offset slack

    def test_failure_returns_inputs(self):
        # Force the internal solve to raise on V=2 REAL overlapping views so the
        # try/except recovery path (not the V<2 early return) is exercised, and
        # assert the original tensors come back untouched with no exception escaping.
        (t, c, _) = self._views()
        t_ref = [x.clone() for x in t]
        orig_solve = bb.np.linalg.solve

        def _boom(*a, **k):
            raise RuntimeError("injected solve failure")

        try:
            bb.np.linalg.solve = _boom
            # harmonize_views logs the caught failure (its designed recovery); mute
            # that expected noise so a passing test doesn't print a scary traceback.
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                out = bb.harmonize_views(t, c, anchor=0)          # must NOT raise
        finally:
            bb.np.linalg.solve = orig_solve

        self.assertEqual(len(out), len(t))
        for o, ref, orig in zip(out, t_ref, t):
            self.assertTrue(torch.equal(o, ref))                 # returned inputs unchanged
            self.assertTrue(torch.equal(orig, ref))              # inputs never mutated


class _FakeRender:
    """Two 'views': left/right halves with a small overlap; view colors differ.
    front_cov/back_cov set the mask split; changing them simulates a different
    mesh's geometry (different visibility -> different cos maps)."""
    def __init__(self, h=64, w=64):
        self.h, self.w = h, w
        self.calls = 0
        self.front_cov = 0.6                   # 'front' covers left front_cov
        self.back_cov = 0.4                    # 'back' covers right of back_cov on

    def back_project(self, view, elev, azim):
        h, w = self.h, self.w
        tex = torch.zeros(h, w, 3)
        cos = torch.zeros(h, w, 1)
        if azim == 0:                          # 'front': left front_cov
            tex[..., :] = torch.tensor(view[0])
            cos[:, : int(w * self.front_cov), 0] = 0.9
        else:                                  # 'back': right of back_cov
            tex[..., :] = torch.tensor(view[1])
            cos[:, int(w * self.back_cov):, 0] = 0.7
        self.calls += 1
        return tex, cos, None


class _FakeVP:
    def __init__(self):
        self.render = _FakeRender()
        self.config = type("C", (), {"bake_exp": 4})()


class TestBakeEx(unittest.TestCase):
    def setUp(self):
        bb._RAMP_CACHE.clear()
        self.vp = _FakeVP()
        # each 'view' arg carries the flat colors both fake views paint
        self.views = [((0.8, 0.8, 0.8), (0.4, 0.4, 0.4))] * 2
        # MR pass: SAME cameras/geometry, DIFFERENT texture content (metallic-
        # roughness values differ from albedo) -> distinct texture fingerprint.
        self.mr_views = [((0.2, 0.2, 0.2), (0.6, 0.6, 0.6))] * 2
        self.elevs, self.azims, self.weights = [0, 0], [0, 180], [1.0, 0.5]

    def test_contract_and_frontier_smoothing(self):
        tex, mask = bb.bake_from_multiview_ex(
            self.vp, self.views, self.elevs, self.azims, self.weights)
        self.assertEqual(tuple(tex.shape), (64, 64, 3))
        self.assertEqual(tuple(mask.shape), (64, 64, 1))
        self.assertTrue(bool(mask.any()))
        # frontier: max horizontal step of the merged row must be far below the
        # raw tone gap (0.4) - harmonize+feather smooths the handoff
        row = tex[32, :, 0]
        step = float((row[1:] - row[:-1]).abs().max())
        self.assertLess(step, 0.10)

    def test_albedo_then_mr_pair_semantics(self):
        bb.bake_from_multiview_ex(self.vp, self.views, self.elevs, self.azims, self.weights)
        self.assertEqual(len(bb._RAMP_CACHE), 1)          # albedo stored ramps
        # MR pass: same geometry, different texture content -> takes the ramps
        bb.bake_from_multiview_ex(self.vp, self.mr_views, self.elevs, self.azims, self.weights)
        self.assertEqual(len(bb._RAMP_CACHE), 0)          # MR took them

    def test_mr_not_harmonized(self):
        # make harmonize explode if called on the 2nd (MR) pass
        bb.bake_from_multiview_ex(self.vp, self.views, self.elevs, self.azims, self.weights)
        orig = bb.harmonize_views
        try:
            def _boom(*a, **k):
                raise AssertionError("harmonize called on MR pass")
            bb.harmonize_views = _boom
            # different texture content, same geometry -> recognized as MR, skips harmonize
            bb.bake_from_multiview_ex(self.vp, self.mr_views, self.elevs, self.azims, self.weights)
        finally:
            bb.harmonize_views = orig

    def test_repeat_albedo_not_flipped_to_mr(self):
        # Residual poisoning (I2): a crash AFTER the albedo `_cache_put` but BEFORE
        # the MR `_cache_take` leaves an entry whose key ALSO matches an identical-
        # geometry re-run (same mesh, same cameras, same _cos_sig). The pre-fix
        # (key hit == MR) misreads that repeat-albedo as MR: harmonization skipped
        # and its later MR harmonized (design forbids). The stored texture
        # fingerprint distinguishes them - IDENTICAL content => repeat-albedo
        # (harmonize + re-store); DIFFERENT content => true MR (skip).
        calls = {"n": 0}
        orig = bb.harmonize_views

        def _counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        try:
            bb.harmonize_views = _counting
            # gen 1 albedo: harmonizes + stores ramps. Crash: NO MR call follows.
            bb.bake_from_multiview_ex(self.vp, self.views, self.elevs, self.azims, self.weights)
            self.assertEqual(calls["n"], 1)               # gen 1 albedo harmonized
            # repeat-albedo: IDENTICAL views + geometry (the crashed run re-run).
            # Must be treated as albedo, NOT MR -> harmonize runs AGAIN.
            bb.bake_from_multiview_ex(self.vp, self.views, self.elevs, self.azims, self.weights)
            self.assertEqual(calls["n"], 2)               # BOTH albedo calls harmonized
            # now the real MR call: SAME geometry, DIFFERENT texture content -> skips.
            bb.bake_from_multiview_ex(self.vp, self.mr_views, self.elevs, self.azims, self.weights)
        finally:
            bb.harmonize_views = orig

        self.assertEqual(calls["n"], 2)                   # MR pass did NOT harmonize

    def test_stale_entry_does_not_flip_next_albedo_to_mr(self):
        # Poisoning: gen 1's albedo call stores ramps, but its MR call never runs
        # (exception / CUDA-OOM between the two bakes). Gen 2 reuses the SAME vp
        # object + SAME cameras but a DIFFERENT mesh (different visibility masks).
        # Without a geometry fingerprint in the cache key, gen 2's ALBEDO call is
        # misread as an MR call -> harmonization silently skipped and gen 1's
        # ramps (from a different mesh) reused. The cos-map fingerprint makes the
        # stale entry's key miss, so gen 2 correctly takes the albedo path.
        calls = {"n": 0}
        orig = bb.harmonize_views

        def _counting(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        try:
            bb.harmonize_views = _counting
            # gen 1 albedo: harmonizes + stores ramps. NO MR call follows.
            bb.bake_from_multiview_ex(
                self.vp, self.views, self.elevs, self.azims, self.weights)
            self.assertEqual(calls["n"], 1)               # gen 1 albedo harmonized
            stale_keys = set(bb._RAMP_CACHE)
            self.assertEqual(len(stale_keys), 1)          # stale entry survives

            # gen 2: same vp + cameras, DIFFERENT geometry (shift the mask splits)
            self.vp.render.front_cov = 0.8
            self.vp.render.back_cov = 0.2
            bb.bake_from_multiview_ex(
                self.vp, self.views, self.elevs, self.azims, self.weights)
        finally:
            bb.harmonize_views = orig

        # gen 2's albedo was NOT flipped to MR: it harmonized again.
        self.assertEqual(calls["n"], 2)
        # a fresh entry appeared under a new fingerprinted key (exactly one), so
        # gen 2's own MR pass will find ITS ramps; the stale key is now
        # unreachable and the <=2 LRU evicts it on the next put.
        new_keys = set(bb._RAMP_CACHE) - stale_keys
        self.assertEqual(len(new_keys), 1)


if __name__ == "__main__":
    unittest.main()
