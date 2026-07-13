"""Hue-preserving vibrance for the albedo/baseColor atlas.

Oklab (a,b) chroma scaling in linear light: muted pixels get the most boost,
already-vivid pixels a token bump, neutral gray is invariant, hue and perceived
lightness are held fixed. Out-of-gamut results are pulled back along constant
hue and lightness (never per-channel clipped). Pure numpy + PIL, torch-free.
"""
import numpy as np

STRENGTH_MAP = {"off": 0.0, "subtle": 0.18, "medium": 0.35, "strong": 0.60}

_C_REF = np.float32(0.40)
_GRAY_FLOOR = np.float32(0.02)

_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]], np.float32)
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]], np.float32)
_M1_INV = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010]], np.float32)
_M2_INV = np.array([[1.0, 0.3963377774, 0.2158037573],
                    [1.0, -0.1055613458, -0.0638541728],
                    [1.0, -0.0894841775, -1.2914855480]], np.float32)

# exact 8-bit sRGB->linear decode (input is uint8, so a LUT is exact + free)
_e = np.arange(256, dtype=np.float32) / 255.0
_DECODE_LUT = np.where(_e <= 0.04045, _e / 12.92,
                       ((_e + 0.055) / 1.055) ** 2.4).astype(np.float32)
del _e


def _log(msg):
    try:
        print(msg)
    except Exception:
        pass


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)


def _in_gamut(lab, eps=1e-4):
    lin = (lab @ _M2_INV.T) ** 3 @ _M1_INV.T
    return np.all((lin >= -eps) & (lin <= 1.0 + eps), axis=-1, keepdims=True)


def _boost(rgb_lin, strength):
    lab = np.cbrt(rgb_lin @ _M1.T) @ _M2.T          # linear -> Oklab
    L = lab[..., 0:1]
    ab = lab[..., 1:3]
    C = np.sqrt((ab ** 2).sum(-1, keepdims=True)) + 1e-12
    t = np.clip(C / _GRAY_FLOOR, 0.0, 1.0)
    floor = t * t * (3.0 - 2.0 * t)                 # smoothstep near-gray guard
    w = np.clip(1.0 - C / _C_REF, 0.0, 1.0)         # muted -> 1, vivid -> 0
    g = 1.0 + np.float32(strength) * w * floor
    ab_boost = ab * g
    lab_boost = np.concatenate([L, ab_boost], -1)
    full_ok = _in_gamut(lab_boost)
    # constant-L, constant-hue reduction: largest scale of ab_boost in [1/g, 1]
    lo = 1.0 / np.maximum(g, 1e-6)                   # returns original ab (in gamut)
    hi = np.ones_like(g)
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        ok = _in_gamut(np.concatenate([L, ab_boost * mid], -1))
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    scale = np.where(full_ok, 1.0, lo)
    lab_final = np.concatenate([L, ab_boost * scale], -1)
    return (lab_final @ _M2_INV.T) ** 3 @ _M1_INV.T  # Oklab -> linear


def apply_vibrance(img, strength):
    """Return img with albedo vibrance applied. strength<=0 -> unchanged input.
    Accepts PIL.Image or uint8 ndarray (RGB/RGBA); never raises."""
    try:
        strength = float(strength)
        if strength <= 0.0:
            return img
        is_pil = hasattr(img, "mode")
        mode = getattr(img, "mode", None)
        arr = np.asarray(img)
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] not in (3, 4):
            return img
        alpha = arr[..., 3:4] if arr.shape[2] == 4 else None
        rgb_lin = _DECODE_LUT[arr[..., :3]]
        out_lin = _boost(rgb_lin, strength)
        out8 = np.rint(_linear_to_srgb(out_lin) * 255.0).astype(np.uint8)
        if alpha is not None:
            out8 = np.concatenate([out8, alpha], -1)
        if is_pil:
            from PIL import Image
            out_img = Image.fromarray(out8, mode)
            # Preserve the source encoding (e.g. JPEG). Image.fromarray sets
            # format=None, which makes trimesh's export re-encode as PNG and
            # bloats the GLB ~10x for JPEG albedo — the same reason seam_fix
            # carries img.format forward.
            out_img.format = getattr(img, "format", None)
            return out_img
        return out8
    except Exception as exc:
        _log(f"[vibrance] skipped ({exc})")
        return img


def apply_to_glb(glb_path, strength):
    """Apply vibrance to a GLB's baseColorTexture in place. Never raises."""
    try:
        if float(strength) <= 0.0:
            return False
        import trimesh
        scene = trimesh.load(glb_path, process=False)
        geoms = scene.geometry.values() if hasattr(scene, "geometry") else [scene]
        changed = False
        for g in geoms:
            mat = getattr(getattr(g, "visual", None), "material", None)
            img = getattr(mat, "baseColorTexture", None) if mat else None
            if img is None:
                continue
            # Pass img straight through so apply_vibrance can carry its
            # .format forward (a .convert() here would drop it -> PNG bloat).
            mat.baseColorTexture = apply_vibrance(img, strength)
            changed = True
        if changed:
            scene.export(glb_path)
        return changed
    except Exception as exc:
        _log(f"[vibrance] apply_to_glb skipped ({exc})")
        return False
