"""Post-bake UV-seam reconciliation + gutter dilation for the Hunyuan3D-2.1 paint.

Reconciles color jumps at UV-chart boundaries left by the multiview bake, then
dilates gutters. Pure numpy/scipy core + a trimesh GLB wrapper. Called from
generator._run_texture after the paint, before the normal bake.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def _weld_index(vertices, tol_rel=1e-6):
    v = np.asarray(vertices, dtype=np.float64)
    if v.size == 0:
        return np.zeros(len(vertices), dtype=np.int64)
    scale = float(np.ptp(v, axis=0).max()) or 1.0
    keys = np.round(v / (scale * tol_rel)).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    return inv


def _find_seam_edges(vertices, faces, uvs):
    faces = np.asarray(faces)
    uvs = np.asarray(uvs, dtype=np.float64)
    weld = _weld_index(vertices)
    edge_map = defaultdict(list)
    for f in faces:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i, j = int(f[a]), int(f[b])
            wi, wj = int(weld[i]), int(weld[j])
            key = (min(wi, wj), max(wi, wj))
            edge_map[key].append((i, j, wi, wj))
    seams = []
    for recs in edge_map.values():
        if len(recs) != 2:
            continue
        (i0, j0, wi0, wj0), (i1, j1, wi1, wj1) = recs
        uvA0, uvA1 = uvs[i0], uvs[j0]
        if (wi1, wj1) == (wi0, wj0):
            uvB0, uvB1 = uvs[i1], uvs[j1]
        else:
            uvB0, uvB1 = uvs[j1], uvs[i1]
        if np.allclose(uvA0, uvB0) and np.allclose(uvA1, uvB1):
            continue
        seams.append((np.array(uvA0), np.array(uvA1),
                      np.array(uvB0), np.array(uvB1)))
    return seams


def _uv_to_px(uv, w, h):
    # UV origin bottom-left; atlas row 0 = top. v flipped.
    return np.array([uv[0] * (w - 1), (1.0 - uv[1]) * (h - 1)])


def _edge_key(p, q):
    """Order-independent key for a UV edge, rounded to fold float noise."""
    a = (round(float(p[0]), 6), round(float(p[1]), 6))
    b = (round(float(q[0]), 6), round(float(q[1]), 6))
    return (a, b) if a <= b else (b, a)


def _third_lookup(faces, uvs):
    """Map each undirected UV edge -> the UV of its triangle's opposite (third)
    corner. A chart's seam edge belongs to exactly one triangle in that chart, so
    the third vertex gives the true interior side of the seam (used by _reconcile
    instead of the twin's packed position, which xatlas may place toward this
    chart's interior)."""
    lut = {}
    for f in np.asarray(faces):
        i, j, k = int(f[0]), int(f[1]), int(f[2])
        for e0, e1, opp in ((i, j, k), (j, k, i), (k, i, j)):
            lut.setdefault(_edge_key(uvs[e0], uvs[e1]), np.asarray(uvs[opp], float))
    return lut


def _inward_perp(p0, p1, third):
    """Unit vector perpendicular to the seam edge (p0->p1), pointing toward the
    `third` (opposite) corner, i.e. into the triangle interior. Degenerate edges
    or a collinear third fall back to a safe direction rather than raising."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); third = np.asarray(third, float)
    e = p1 - p0
    el = np.hypot(*e)
    if el < 1e-9:                       # degenerate edge -> aim straight at third
        g = third - p0
        return g / (np.hypot(*g) or 1.0)
    ehat = e / el
    g = third - 0.5 * (p0 + p1)
    perp = g - np.dot(g, ehat) * ehat   # component of (mid->third) normal to edge
    pl = np.hypot(*perp)
    if pl < 1e-9:                       # third collinear with the edge -> rotate normal
        return np.array([-ehat[1], ehat[0]])
    return perp / pl


def _seam_band(base, seam_len_px):
    """Feather band for ONE seam: the base band clamped so it never exceeds a
    third of THAT seam's own edge length. Per-seam so a single sub-3px seam edge
    can't collapse the band for the whole atlas (Fix 2)."""
    return max(1, min(int(base), int(seam_len_px // 3)))


def _seam_samples(pa0, pa1, pb0, pb1):
    """Along-seam sample count taken from the LONGER of the two sides, so the
    denser (finer-packed) twin is covered without holes (Fix 9)."""
    la = np.hypot(*(np.asarray(pa1, float) - np.asarray(pa0, float)))
    lb = np.hypot(*(np.asarray(pb1, float) - np.asarray(pb0, float)))
    return max(2, int(max(la, lb)) + 1)


def _reconcile(atlas, faces, uvs, seams, seam_band_px):
    if not seams:
        return atlas
    h, w = atlas.shape[:2]
    base = max(1, int(seam_band_px))
    uvs = np.asarray(uvs, np.float64)
    third = _third_lookup(faces, uvs)
    # accumulate a per-texel additive correction, feathered by distance to seam.
    corr = np.zeros((h, w, atlas.shape[2]), np.float64)
    wsum = np.zeros((h, w), np.float64)
    atf = atlas.astype(np.float64)
    for a0, a1, b0, b1 in seams:
        pa0, pa1 = _uv_to_px(a0, w, h), _uv_to_px(a1, w, h)
        pb0, pb1 = _uv_to_px(b0, w, h), _uv_to_px(b1, w, h)
        # Fix 9: a non-finite UV (NaN/inf) would blow up hypot/int below and abort
        # the WHOLE stage — skip just this seam and carry on.
        if not (np.all(np.isfinite(pa0)) and np.all(np.isfinite(pa1))
                and np.all(np.isfinite(pb0)) and np.all(np.isfinite(pb1))):
            continue
        # Fix 1: interior direction from each owning triangle's THIRD vertex.
        ta = third.get(_edge_key(a0, a1))
        tb = third.get(_edge_key(b0, b1))
        # Fix 9 (review follow-up): the third-vertex UVs feed _inward_perp/_sample
        # below, so a non-finite third corner must skip this seam too — not just the
        # endpoints guarded above — or one bad seam aborts the whole reconcile.
        if (ta is None or tb is None
                or not (np.all(np.isfinite(ta)) and np.all(np.isfinite(tb)))):
            continue
        da = _inward_perp(pa0, pa1, _uv_to_px(ta, w, h))
        db = _inward_perp(pb0, pb1, _uv_to_px(tb, w, h))
        # Fix 2: clamp the band from this seam's own edge length.
        seam_len = max(np.hypot(*(pa1 - pa0)), np.hypot(*(pb1 - pb0)))
        band = _seam_band(base, seam_len)
        # Fix 9: sample density from the longer side so the denser twin is covered.
        for t in np.linspace(0.0, 1.0, _seam_samples(pa0, pa1, pb0, pb1)):
            ca = pa0 + t * (pa1 - pa0)
            cb = pb0 + t * (pb1 - pb0)
            sa = _sample(atf, ca + 3 * da)   # 3px into each chart interior
            sb = _sample(atf, cb + 3 * db)
            target = 0.5 * (sa + sb)
            _spray(corr, wsum, ca, da, band, target - sa)
            _spray(corr, wsum, cb, db, band, target - sb)
    m = wsum > 0
    out = atf.copy()
    out[m] += corr[m] / wsum[m, None]
    return np.clip(out, 0, 255).astype(atlas.dtype)


def _sample(atf, p):
    h, w = atf.shape[:2]
    x = int(np.clip(np.floor(p[0] + 0.5), 0, w - 1))
    y = int(np.clip(np.floor(p[1] + 0.5), 0, h - 1))
    return atf[y, x]


def _spray(corr, wsum, c, inward, band, delta):
    h, w = wsum.shape
    for d in range(band):
        p = c + d * inward
        # round-half-up (floor(x+0.5)); Python's banker's round() skipped columns
        # (a seam at x=31.5 stepping inward never touched column 33) (Fix 9).
        x = int(np.floor(p[0] + 0.5)); y = int(np.floor(p[1] + 0.5))
        if 0 <= x < w and 0 <= y < h:
            fw = 1.0 - d / float(band)   # feather: full at seam -> 0 at band edge
            corr[y, x] += fw * delta
            wsum[y, x] += fw


def _local_band(atlas_dim):
    """Base feather band in texels, scaled to the atlas resolution. The per-seam
    clamp now lives in _reconcile (Fix 2), so this returns the base only."""
    return max(1, round(9 * atlas_dim / 4096))


def _coverage_mask(uvs, faces, h, w):
    # Pure-numpy barycentric point-in-triangle test (same rasterizer as
    # normal_bake.rasterize_uv_atlas). Matplotlib is NOT a declared dependency —
    # it only reached us via an unpinned realesrgan->facexlib->filterpy chain, so
    # the old Path.contains_points made seam-fix silently no-op when it was absent
    # (Fix 4).
    uvs = np.asarray(uvs, np.float64)
    mask = np.zeros((h, w), bool)
    for f in np.asarray(faces):
        a, b, c = (_uv_to_px(uvs[int(f[0])], w, h),
                   _uv_to_px(uvs[int(f[1])], w, h),
                   _uv_to_px(uvs[int(f[2])], w, h))
        # restrict the test to the triangle's pixel bounding box (O(bbox) texels).
        x0 = int(np.floor(min(a[0], b[0], c[0]))); x1 = int(np.ceil(max(a[0], b[0], c[0])))
        y0 = int(np.floor(min(a[1], b[1], c[1]))); y1 = int(np.ceil(max(a[1], b[1], c[1])))
        x0 = max(0, min(x0, w - 1)); x1 = max(0, min(x1, w - 1))
        y0 = max(0, min(y0, h - 1)); y1 = max(0, min(y1, h - 1))
        if x1 < x0 or y1 < y0:
            continue
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-12:              # degenerate triangle covers no texel
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        px = xs.ravel().astype(np.float64); py = ys.ravel().astype(np.float64)
        wa = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / d
        wb = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / d
        wc = 1.0 - wa - wb
        inside = (wa >= -1e-6) & (wb >= -1e-6) & (wc >= -1e-6)
        mask[ys.ravel()[inside], xs.ravel()[inside]] = True
    return mask


def _dilate_gutter(atlas, faces, uvs, gutter_px):
    from scipy import ndimage
    h, w = atlas.shape[:2]
    valid = _coverage_mask(uvs, faces, h, w)
    if valid.all() or not valid.any():
        return atlas
    # nearest valid texel index for every texel
    idx = ndimage.distance_transform_edt(~valid, return_distances=True,
                                         return_indices=True)
    dist, (iy, ix) = idx[0], idx[1]
    fill = (~valid) & (dist <= gutter_px)
    out = atlas.copy()
    out[fill] = atlas[iy[fill], ix[fill]]
    return out


def reconcile_and_dilate(vertices, faces, uvs, atlas, *,
                         seam_band_px=None, gutter_px=None):
    atlas = np.array(atlas)
    h, w = atlas.shape[:2]
    dim = max(h, w)
    seams = _find_seam_edges(vertices, faces, uvs)
    if seam_band_px is None:
        seam_band_px = _local_band(dim)
    if gutter_px is None:
        gutter_px = max(1, round(16 * dim / 4096))
    if seams:
        atlas = _reconcile(atlas, faces, uvs, seams, seam_band_px)
    atlas = _dilate_gutter(atlas, faces, uvs, gutter_px)
    return atlas


def apply_to_glb(glb_path):
    import trimesh
    from PIL import Image
    scene = trimesh.load(glb_path, process=False)
    geoms = scene.geometry.values() if hasattr(scene, "geometry") else [scene]
    for g in geoms:
        v = getattr(g, "visual", None)
        mat = getattr(v, "material", None)
        uv = getattr(v, "uv", None)
        if mat is None or uv is None:
            continue
        verts = np.asarray(g.vertices)
        faces = np.asarray(g.faces)
        uvs = np.asarray(uv)
        for attr in ("baseColorTexture", "metallicRoughnessTexture"):
            img = getattr(mat, attr, None)
            if img is None:
                continue
            src_fmt = getattr(img, "format", None)   # capture BEFORE convert() drops it
            arr = np.asarray(img.convert("RGB"))
            fixed = reconcile_and_dilate(verts, faces, uvs, arr)
            new_img = Image.fromarray(fixed, "RGB")
            # Preserve the source encoding. Image.fromarray has format=None, which
            # makes trimesh re-encode the atlas as PNG on export — ~10x bloat when
            # the paint emitted a JPEG albedo (Fix 3).
            if src_fmt:
                new_img.format = src_fmt
            setattr(mat, attr, new_img)
    scene.export(glb_path)
