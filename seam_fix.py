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
