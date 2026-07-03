"""High->low tangent-space normal-map bake for the Hunyuan3D-2.1 extension.

Pipeline (all validated against the real 2.6M-face mesh, ~few seconds CPU):
  1. low-poly base (already xatlas-unwrapped) -> rasterize UV atlas to per-texel
     {position, interpolated smooth normal} + coverage mask.
  2. Open3D closest-point onto the dense mesh -> interpolate its SMOOTH vertex
     normals at the hit barycentric coords.
  3. transform dense normal into the texel tangent frame, +Y encode, dilate seams.
  4. attach as glTF normalTexture (force NORMAL export; no hand-authored TANGENT).
"""
from __future__ import annotations
import numpy as np


def rasterize_uv_atlas(verts, faces, uv, vertex_normals, size=2048):
    """Barycentric-rasterize each triangle in UV space (origin bottom-left)."""
    verts = np.asarray(verts, np.float64)
    uv = np.asarray(uv, np.float64)
    vn = np.asarray(vertex_normals, np.float64)
    pos = np.zeros((size, size, 3), np.float32)
    nrm = np.zeros((size, size, 3), np.float32)
    mask = np.zeros((size, size), bool)

    for tri in faces:
        uva, uvb, uvc = uv[tri] * (size - 1)
        minx = max(int(np.floor(min(uva[0], uvb[0], uvc[0]))), 0)
        maxx = min(int(np.ceil(max(uva[0], uvb[0], uvc[0]))), size - 1)
        miny = max(int(np.floor(min(uva[1], uvb[1], uvc[1]))), 0)
        maxy = min(int(np.ceil(max(uva[1], uvb[1], uvc[1]))), size - 1)
        if maxx < minx or maxy < miny:
            continue
        xs, ys = np.meshgrid(np.arange(minx, maxx + 1), np.arange(miny, maxy + 1))
        px = xs.ravel().astype(np.float64)
        py = ys.ravel().astype(np.float64)
        d = ((uvb[1] - uvc[1]) * (uva[0] - uvc[0]) + (uvc[0] - uvb[0]) * (uva[1] - uvc[1]))
        if abs(d) < 1e-12:
            continue
        wa = ((uvb[1] - uvc[1]) * (px - uvc[0]) + (uvc[0] - uvb[0]) * (py - uvc[1])) / d
        wb = ((uvc[1] - uva[1]) * (px - uvc[0]) + (uva[0] - uvc[0]) * (py - uvc[1])) / d
        wc = 1.0 - wa - wb
        inside = (wa >= -1e-6) & (wb >= -1e-6) & (wc >= -1e-6)
        if not inside.any():
            continue
        wa, wb, wc = wa[inside], wb[inside], wc[inside]
        pxi = xs.ravel()[inside]
        pyi = ys.ravel()[inside]
        va, vb, vc = verts[tri]
        na, nb, nc = vn[tri]
        p = wa[:, None] * va + wb[:, None] * vb + wc[:, None] * vc
        n = wa[:, None] * na + wb[:, None] * nb + wc[:, None] * nc
        pos[pyi, pxi] = p.astype(np.float32)
        nrm[pyi, pxi] = n.astype(np.float32)
        mask[pyi, pxi] = True

    ln = np.linalg.norm(nrm, axis=2, keepdims=True)
    ln[ln == 0] = 1.0
    nrm = nrm / ln
    return pos, nrm, mask
