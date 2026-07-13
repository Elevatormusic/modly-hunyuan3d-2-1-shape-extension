import unittest
import numpy as np
from PIL import Image
import vibrance


def _lin(c):  # sRGB->linear for building expectations
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _oklab(rgb8):  # rgb8: (...,3) uint8 -> Oklab (...,3)
    lin = _lin(rgb8.astype(np.float64))
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return np.cbrt(lin @ M1.T) @ M2.T


def _chroma(rgb8):
    lab = _oklab(rgb8)
    return float(np.hypot(lab[..., 1], lab[..., 2]).mean())


def _hue(rgb8):
    lab = _oklab(rgb8)
    return float(np.degrees(np.arctan2(lab[..., 2], lab[..., 1]).mean()))


class TestVibrance(unittest.TestCase):
    def test_strength_map_values(self):
        self.assertEqual(vibrance.STRENGTH_MAP,
                         {"off": 0.0, "subtle": 0.18, "medium": 0.35, "strong": 0.60})

    def test_off_is_byte_identical(self):
        arr = np.random.randint(0, 256, (16, 16, 3), np.uint8)
        out = vibrance.apply_vibrance(arr, 0.0)
        np.testing.assert_array_equal(out, arr)

    def test_gray_is_invariant(self):
        arr = np.full((8, 8, 3), 128, np.uint8)
        out = vibrance.apply_vibrance(arr, 0.6)
        np.testing.assert_array_equal(out, arr)

    def test_muted_boosted_more_than_vivid(self):
        muted = np.full((4, 4, 3), 0, np.uint8); muted[..., 0] = 150; muted[..., 1] = 120; muted[..., 2] = 120
        vivid = np.full((4, 4, 3), 0, np.uint8); vivid[..., 0] = 220; vivid[..., 1] = 20; vivid[..., 2] = 20
        dm = _chroma(vibrance.apply_vibrance(muted, 0.35)) - _chroma(muted)
        dv = _chroma(vibrance.apply_vibrance(vivid, 0.35)) - _chroma(vivid)
        self.assertGreater(dm, dv)
        self.assertGreater(dm, 0.0)

    def test_monotonic_in_strength(self):
        px = np.tile(np.array([150, 120, 120], np.uint8), (4, 4, 1))
        c0, c1, c2 = (_chroma(vibrance.apply_vibrance(px, k)) for k in (0.18, 0.35, 0.60))
        self.assertLess(c0, c1); self.assertLess(c1, c2)

    def test_hue_preserved(self):
        px = np.tile(np.array([150, 120, 120], np.uint8), (4, 4, 1))
        self.assertLess(abs(_hue(vibrance.apply_vibrance(px, 0.35)) - _hue(px)), 1.0)

    def test_vivid_hue_preserved_through_gamut_reduction(self):
        px = np.tile(np.array([230, 12, 12], np.uint8), (4, 4, 1))  # near gamut edge
        self.assertLess(abs(_hue(vibrance.apply_vibrance(px, 0.60)) - _hue(px)), 1.0)

    def test_lightness_preserved(self):
        px = np.tile(np.array([150, 120, 120], np.uint8), (4, 4, 1))
        L0 = float(_oklab(px)[..., 0].mean())
        L1 = float(_oklab(vibrance.apply_vibrance(px, 0.35))[..., 0].mean())
        self.assertLess(abs(L1 - L0), 0.02)

    def test_outputs_in_range(self):
        arr = np.random.randint(0, 256, (32, 32, 3), np.uint8)
        out = vibrance.apply_vibrance(arr, 0.6)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, arr.shape)

    def test_rgba_alpha_preserved(self):
        arr = np.random.randint(0, 256, (8, 8, 4), np.uint8)
        out = vibrance.apply_vibrance(arr, 0.5)
        np.testing.assert_array_equal(out[..., 3], arr[..., 3])

    def test_pil_in_pil_out(self):
        im = Image.fromarray(np.full((8, 8, 3), 100, np.uint8), "RGB")
        out = vibrance.apply_vibrance(im, 0.3)
        self.assertIsInstance(out, Image.Image)

    def test_never_raises_on_bad_input(self):
        self.assertEqual(vibrance.apply_vibrance("not an image", 0.3), "not an image")


if __name__ == "__main__":
    unittest.main()
