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

_RAMP_CACHE = {}           # (id(vp), elevs, azims, cos_sig) -> list[ramp]; albedo stores, MR takes


def _cache_put(key, ramps):
    while len(_RAMP_CACHE) >= 2:                     # only one albedo/MR pair in flight
        _RAMP_CACHE.pop(next(iter(_RAMP_CACHE)))
    _RAMP_CACHE[key] = ramps


def _cache_take(key):
    return _RAMP_CACHE.pop(key, None)


def _cos_sig(cos_maps):
    """Cheap geometry fingerprint: identical for the albedo/MR pair of one
    generation (cos maps are geometry-only), different across meshes."""
    return tuple(round(float(c.sum().item()), 4) for c in cos_maps)


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


def harmonize_views(textures, cos_maps, anchor=0, lam=RIDGE_LAMBDA, cap=SAMPLE_CAP):
    """Solve per-view per-channel gain a_v + offset b_v from overlap regions
    (anchor fixed at identity), correct IN PLACE, and return the SAME list.
    Ridge-regularized normal equations on summed pairwise residuals (lambda
    self-scales); clamps keep a pathological solve to at worst a mild tint.
    Deterministic. ALL channels are solved before ANY texture is written, so on
    any internal failure the inputs are returned byte-for-byte unchanged (never
    raises). In-place avoids a full second copy of the view stack (the ~V*4096^2*3
    fp32 the clone used to hold, +1.15-1.7 GB CUDA)."""
    try:
        V = len(textures)
        if V < 2:
            return list(textures)
        eps = 1e-4
        vis = [c[..., 0] > eps for c in cos_maps]
        n_unk = 2 * (V - 1)                     # (a_v, b_v) for v != anchor
        others = [v for v in range(V) if v != anchor]
        col = {v: i for i, v in enumerate(others)}
        coeffs = {}                             # (v, ch) -> (a, b); applied only AFTER every channel solves
        for ch in range(textures[0].shape[-1]):
            A = np.zeros((n_unk, n_unk))
            rhs = np.zeros(n_unk)
            n_pairs_used = 0
            for i in range(V):
                for j in range(i + 1, V):
                    ov = (vis[i] & vis[j])
                    idx = torch.nonzero(ov.reshape(-1), as_tuple=False).reshape(-1)
                    if idx.numel() < 32:
                        continue
                    stride = max(1, idx.numel() * (V * (V - 1) // 2) // max(cap, 1))
                    idx = idx[::stride]
                    Ii = textures[i].reshape(-1, textures[i].shape[-1])[idx, ch].double().cpu().numpy()
                    Ij = textures[j].reshape(-1, textures[j].shape[-1])[idx, ch].double().cpu().numpy()
                    n_pairs_used += 1
                    # residual r = (a_i I_i + b_i) - (a_j I_j + b_j); anchor: a=1, b=0
                    # accumulate normal equations for x = [a_o..., b_o...]
                    # build per-sample coefficient triples: (column, coeff)
                    cols_i = None if i == anchor else col[i]
                    cols_j = None if j == anchor else col[j]
                    # constant term (from anchor side): if i is anchor, r has +I_i; if j is anchor, -I_j
                    const = np.zeros_like(Ii)
                    terms = []                       # list of (a_col, coeff_vec, b_col, coeff_scalar)
                    if cols_i is None:
                        const += Ii
                    else:
                        terms.append((cols_i, Ii, V - 1 + cols_i, 1.0))
                    if cols_j is None:
                        const -= Ij
                    else:
                        terms.append((cols_j, -Ij, V - 1 + cols_j, -1.0))
                    # normal equations: for unknown columns p,q: A[p,q] += sum coeff_p*coeff_q
                    # rhs[p] += -sum coeff_p * const   (since residual = sum coeff*x + const)
                    flat = []
                    for (ac, avec, bc, bsc) in terms:
                        flat.append((ac, avec))
                        flat.append((bc, np.full_like(avec, bsc)))
                    for (p, cp) in flat:
                        rhs[p] += -np.dot(cp, const)
                        for (q, cq) in flat:
                            A[p, q] += np.dot(cp, cq)
            if n_pairs_used == 0:
                continue
            # ridge toward identity: a=1 (x_a = a-1 ... we solved for a directly), b=0.
            # We solved with unknowns a,b directly, so shift: penalize (a-1) and b.
            for v in others:
                pa, pb = col[v], V - 1 + col[v]
                A[pa, pa] += lam
                rhs[pa] += lam * 1.0
                A[pb, pb] += lam
            x = np.linalg.solve(A + 1e-9 * np.eye(n_unk), rhs)
            for v in others:
                a = float(np.clip(x[col[v]], GAIN_CLAMP[0], GAIN_CLAMP[1]))
                b = float(np.clip(x[V - 1 + col[v]], -OFFSET_CLAMP, OFFSET_CLAMP))
                coeffs[(v, ch)] = (a, b)
        # every channel solved without raising -> apply all corrections in place
        # (atomic: a mid-solve failure above leaves every texture untouched)
        for (v, ch), (a, b) in coeffs.items():
            textures[v][..., ch] = torch.clamp(
                textures[v][..., ch] * a + b, 0.0, 1.0)
        return textures
    except Exception:
        import traceback
        traceback.print_exc()
        print("[bake_blend] harmonize failed; using uncorrected views")
        return list(textures)


def bake_from_multiview_ex(vp, views, camera_elevs, camera_azims, view_weights):
    """Drop-in body for ViewProcessor.bake_from_multiview (patched call site).
    Albedo call (first with this key): harmonize + compute & store feather ramps
    alongside a fingerprint of THIS pass's textures. MR call (same cameras/
    geometry but DIFFERENT texture content): identity harmonization, reuse ramps
    (cos maps are geometry-only, identical between the two bakes - RV-4).

    A key hit alone is NOT proof of an MR call: a crash between the albedo
    `_cache_put` and the MR `_cache_take` leaves an entry that an identical-
    geometry re-run's ALBEDO call would also hit. So the stored texture
    fingerprint decides the role - different content => true MR (skip harmonize);
    identical content => repeat-albedo (harmonize + re-store so the real MR pass
    still pairs)."""
    textures, cos_maps = [], []
    for view, elev, azim, weight in zip(views, camera_elevs, camera_azims, view_weights):
        tex, cos, _boundary = vp.render.back_project(view, elev, azim)
        cos_maps.append(weight * (cos ** vp.config.bake_exp))
        textures.append(tex)
    key = (id(vp), tuple(camera_elevs), tuple(camera_azims), _cos_sig(cos_maps))
    # fingerprint the CALLING pass's textures BEFORE harmonize mutates them in place
    tex_sig = tuple(round(float(t.sum().item()), 2) for t in textures)
    cached = _cache_take(key)
    ramps = cached[0] if cached is not None else None
    ramps_ok = ramps is not None and len(ramps) == len(cos_maps)
    if ramps_ok and cached[1] != tex_sig:
        # true MR pass: same geometry, DIFFERENT texture content -> reuse ramps,
        # identity harmonization (do NOT re-store; this consumes the pair)
        pass
    else:
        # albedo pass (fresh key) OR repeat-albedo (key hit, IDENTICAL tex_sig =
        # a crashed run's leftovers). Both harmonize; a same-geometry hit's ramps
        # are still valid so reuse them, else compute. Re-store (with this pass's
        # tex_sig) so the following true-MR call still pairs correctly.
        if not ramps_ok:
            dim = max(cos_maps[0].shape[:2])
            ramps = compute_ramps(cos_maps, feather_px=FEATHER_PX,
                                  ref_dim=4096 if dim >= 2048 else dim)
        _cache_put(key, (ramps, tex_sig))
        textures = harmonize_views(textures, cos_maps)
    cos_maps = [c * r for c, r in zip(cos_maps, ramps)]
    texture, trust = merge(textures, cos_maps)
    return texture, trust > 1e-8
