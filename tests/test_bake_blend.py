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
        out = bb.harmonize_views(t, c, anchor=0)
        ov = slice(96 // 3, 2 * 96 // 3)                          # overlap columns
        before = float((t[0][:, ov] - t[1][:, ov]).abs().mean())
        after = float((out[0][:, ov] - out[1][:, ov]).abs().mean())
        self.assertLess(after, before / 5.0)                      # >=5x agreement
        self.assertTrue(torch.equal(out[0], t[0]))                # anchor untouched

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
        overlap = int(((c[0][..., 0] > 0) & (c[1][..., 0] > 0)).sum())
        self.assertGreaterEqual(overlap, 32)                      # gate not skipping the pair
        self.assertLess(overlap, 96 * 96 // 10)                   # genuinely thin (<10% of image)
        out = bb.harmonize_views(t, c, anchor=0)
        delta = float((out[1] - t[1]).abs().max())
        # non-vacuous: the ridge path actually ran (a skipped pair leaves out[1]==t[1]
        # exactly, delta==0). Empirically delta==0.0107 at these params.
        self.assertGreater(delta, 1e-4)                           # correction is real, not skipped
        self.assertLess(delta, 0.03)                              # yet ridge keeps it near identity
        self.assertTrue(torch.equal(out[0], t[0]))                # anchor untouched

    def test_identical_calls_are_bit_identical(self):
        (t, c, _) = self._views()
        out_a = bb.harmonize_views(t, c, anchor=0)
        out_b = bb.harmonize_views(t, c, anchor=0)
        self.assertFalse(torch.equal(out_a[1], t[1]))             # non-vacuous: a real correction
        for xa, xb in zip(out_a, out_b):
            self.assertTrue(torch.equal(xa, xb))                  # deterministic to the bit

    def test_clamps_hold_on_adversarial_input(self):
        (t, c, _) = self._views(a=(1.0, 5.0), b=(0.0, 0.4))       # wild injected distortion
        out = bb.harmonize_views(t, c, anchor=0)
        # correction applied to view 1 is a'*I + b' with a' in [0.5,2], |b'|<=64/255:
        # verify output stays a bounded transform of the input
        ratio = (out[1][c[1][..., 0] > 0] + 1e-6) / (t[1][c[1][..., 0] > 0] + 1e-6)
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


if __name__ == "__main__":
    unittest.main()
