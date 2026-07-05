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


def _reconcile(atlas, faces, uvs, seams, seam_band_px):
    if not seams:
        return atlas
    h, w = atlas.shape[:2]
    band = max(1, int(seam_band_px))
    # accumulate a per-texel additive correction, feathered by distance to seam.
    corr = np.zeros((h, w, atlas.shape[2]), np.float64)
    wsum = np.zeros((h, w), np.float64)
    atf = atlas.astype(np.float64)
    for a0, a1, b0, b1 in seams:
        pa0, pa1 = _uv_to_px(a0, w, h), _uv_to_px(a1, w, h)
        pb0, pb1 = _uv_to_px(b0, w, h), _uv_to_px(b1, w, h)
        n = max(2, int(np.hypot(*(pa1 - pa0))) + 1)
        for t in np.linspace(0.0, 1.0, n):
            ca = pa0 + t * (pa1 - pa0)
            cb = pb0 + t * (pb1 - pb0)
            # inward normals (toward each chart interior): sample 3px in
            da = _inward(ca, cb)
            db = _inward(cb, ca)
            sa = _sample(atf, ca + 3 * da)
            sb = _sample(atf, cb + 3 * db)
            target = 0.5 * (sa + sb)
            _spray(corr, wsum, ca, da, band, target - sa)
            _spray(corr, wsum, cb, db, band, target - sb)
    m = wsum > 0
    out = atf.copy()
    out[m] += corr[m] / wsum[m, None]
    return np.clip(out, 0, 255).astype(atlas.dtype)


def _inward(c, other):
    d = c - other
    nrm = np.hypot(*d) or 1.0
    return d / nrm


def _sample(atf, p):
    h, w = atf.shape[:2]
    x = int(np.clip(round(p[0]), 0, w - 1))
    y = int(np.clip(round(p[1]), 0, h - 1))
    return atf[y, x]


def _spray(corr, wsum, c, inward, band, delta):
    h, w = wsum.shape
    for d in range(band):
        p = c + d * inward
        x = int(round(p[0])); y = int(round(p[1]))
        if 0 <= x < w and 0 <= y < h:
            fw = 1.0 - d / float(band)   # feather: full at seam -> 0 at band edge
            corr[y, x] += fw * delta
            wsum[y, x] += fw


def _local_band(uvs, faces, seams, atlas_dim):
    base = max(1, round(9 * atlas_dim / 4096))
    if not seams:
        return base
    # shortest incident UV edge length in texels among seam edges
    shortest = min(np.hypot(*((a1 - a0) * atlas_dim)) for a0, a1, _, _ in seams)
    clamp = max(1, int(shortest // 3))
    return min(base, clamp)


def _coverage_mask(uvs, faces, h, w):
    from matplotlib.path import Path  # available via trimesh dep chain
    mask = np.zeros((h, w), bool)
    for f in np.asarray(faces):
        tri = np.array([_uv_to_px(uvs[int(i)], w, h) for i in f])
        # restrict the point test to this triangle's pixel bounding box so we
        # test O(bbox) texels per face instead of the whole h*w grid. Result is
        # identical: texels outside the bbox can't be inside the triangle.
        x0 = int(np.floor(tri[:, 0].min())); x1 = int(np.ceil(tri[:, 0].max()))
        y0 = int(np.floor(tri[:, 1].min())); y1 = int(np.ceil(tri[:, 1].max()))
        x0 = max(0, min(x0, w - 1)); x1 = max(0, min(x1, w - 1))
        y0 = max(0, min(y0, h - 1)); y1 = max(0, min(y1, h - 1))
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        pts = np.column_stack([xs.ravel(), ys.ravel()])
        inside = Path(tri).contains_points(pts).reshape(ys.shape)
        mask[y0:y1 + 1, x0:x1 + 1] |= inside
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
        seam_band_px = _local_band(uvs, faces, seams, dim)
    if gutter_px is None:
        gutter_px = max(1, round(16 * dim / 4096))
    if seams:
        atlas = _reconcile(atlas, faces, uvs, seams, seam_band_px)
    atlas = _dilate_gutter(atlas, faces, uvs, gutter_px)
    return atlas
