# bake_blend.py
"""Harmonize + feather view-merge for the Hunyuan3D-2.1 texture bake.

The stock bake blends back-projected views with cos^4 weights, but each view's
visibility ends in a hard 1-texel cliff (depth test + cos threshold + erosion),
so where neighboring texels are painted by views that disagree in tone, the
handoff is a ragged visible frontier. This module (a) harmonizes per-view color
(gain+offset solved from overlap regions) before merging the albedo bake, and
(b) feathers each view's influence to zero over ~32 texels approaching its own
visibility edge, then (c) merges with an exact weighted average (no whole-view
skip). Wired via the patched pipeline_utils.bake_from_multiview; env
EB_BAKE_BLEND=legacy bypasses everything (stock output).

Pure torch + numpy/scipy. No pipeline imports; CPU-testable. Design + research
validation: private/specs/2026-07-10-bake-merge-{design,research}.md.
"""
from __future__ import annotations
import numpy as np
import torch

FEATHER_PX = 32.0          # at a 4096 atlas; scales with atlas size
RIDGE_LAMBDA = 1.0         # summed pairwise residuals -> ridge self-scales
SAMPLE_CAP = 100_000
GAIN_CLAMP = (0.5, 2.0)
OFFSET_CLAMP = 64.0 / 255.0
_EDT_MAX_GRID = 2048       # EDT cost: 1.05 s @4096^2 vs 0.26 s @2048^2 (measured)

_RAMP_CACHE = {}           # (id(vp), elevs, azims) -> list[ramp]; albedo stores, MR takes


def _cache_put(key, ramps):
    while len(_RAMP_CACHE) >= 2:                     # only one albedo/MR pair in flight
        _RAMP_CACHE.pop(next(iter(_RAMP_CACHE)))
    _RAMP_CACHE[key] = ramps


def _cache_take(key):
    return _RAMP_CACHE.pop(key, None)


def merge(textures, cos_maps):
    """Exact weighted average of views. NO whole-view skip (the stock >0.99
    short-circuit changes output; RV-3 proved removal safe: trust feeds only
    uv_inpaint and painted-texel deltas are bounded by the weighted average)."""
    channel = textures[0].shape[-1]
    h, w = cos_maps[0].shape[:2]
    device = textures[0].device
    texture_merge = torch.zeros(h, w, channel, device=device)
    trust = torch.zeros(h, w, 1, device=device)
    for tex, cos in zip(textures, cos_maps):
        texture_merge += tex * cos
        trust += cos
    return texture_merge / torch.clamp(trust, min=1e-8), trust


def compute_ramps(cos_maps, feather_px=FEATHER_PX, ref_dim=4096):
    """Per-view feather ramp in [0,1]: EDT distance into the visible mask,
    clipped at feather_px (defined at ref_dim scale). EDT runs on a <=2048 grid
    (strided downsample; upsample error mean 0.0025 per RV-4) and the ramp is
    bilinearly upsampled back. Returns tensors shaped like cos_maps."""
    from scipy import ndimage
    ramps = []
    for cos in cos_maps:
        h, w = cos.shape[:2]
        device, dtype = cos.device, cos.dtype
        mask = (cos[..., 0] > 0).detach().cpu().numpy()
        ds = max(1, int(np.ceil(max(h, w) / _EDT_MAX_GRID)))
        small = mask[::ds, ::ds]
        if not small.any():
            ramps.append(torch.zeros(h, w, 1, device=device, dtype=dtype))
            continue
        dist = ndimage.distance_transform_edt(small)
        # feather_px is defined at ref_dim; convert to downsampled-grid units
        f_units = max(1.0, feather_px * (max(h, w) / float(ref_dim)) / ds)
        ramp_small = np.clip(dist / f_units, 0.0, 1.0).astype(np.float32)
        r = torch.from_numpy(ramp_small)[None, None]           # 1,1,hs,ws
        r = torch.nn.functional.interpolate(
            r, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        r = r.to(device=device, dtype=dtype) * torch.from_numpy(
            mask.astype(np.float32)).to(device=device, dtype=dtype)  # hard-zero outside
        ramps.append(r[..., None])
    return ramps
