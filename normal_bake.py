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


def sample_dense_normals(dense, pos_map, mask, max_dist_frac=0.05):
    """For each covered texel, the dense mesh's SMOOTH normal at its closest point."""
    import open3d as o3d
    size = pos_map.shape[0]
    world = np.zeros((size, size, 3), np.float32)
    idx = np.where(mask.ravel())[0]
    if idx.size == 0:
        return world
    query = pos_map.reshape(-1, 3)[idx].astype(np.float32)

    dv = np.asarray(dense.vertices, np.float32)
    df = np.asarray(dense.faces, np.uint32)
    vnorm = np.asarray(dense.vertex_normals, np.float32)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(dv), o3d.core.Tensor(df))
    ans = scene.compute_closest_points(o3d.core.Tensor(query))
    prim = ans["primitive_ids"].numpy().astype(np.int64)
    bary = ans["primitive_uvs"].numpy().astype(np.float64)  # (u, v) -> weights for v1, v2
    tri = df[prim].astype(np.int64)
    w1, w2 = bary[:, 0], bary[:, 1]
    w0 = 1.0 - w1 - w2
    n = (w0[:, None] * vnorm[tri[:, 0]]
         + w1[:, None] * vnorm[tri[:, 1]]
         + w2[:, None] * vnorm[tri[:, 2]])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    n = n / ln
    flat = world.reshape(-1, 3)
    flat[idx] = n.astype(np.float32)
    return flat.reshape(size, size, 3)
