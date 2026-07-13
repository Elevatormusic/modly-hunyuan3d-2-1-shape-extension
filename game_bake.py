"""Multi-channel high->low bake for game-ready assets.

Transfers the painted PBR (albedo + metallic-roughness) from the dense textured
mesh onto the retopo mesh's new UVs, and bakes an AO map. Reuses normal_bake's
UV rasterization + gutter dilation; the tangent-space NORMAL map is produced
separately by normal_bake.bake_normal_map (which also ships the TANGENT accessor).

Transfer sampling is NEAREST — no source-seam colour bleed and no sRGB/linear
interpolation error: albedo (sRGB) and MR (linear) texels are copied byte-for-byte
to the new layout. AO is a CPU hemisphere ray-cast (embreex if trimesh finds it,
else the pure engine). Never raises — returns whatever channels succeeded.
"""
from __future__ import annotations
import numpy as np

import normal_bake


def _closest_source_uv(dv, df, duv, query):
    """Interpolated source UV at each query point's closest point on the dense
    mesh (+ the closest point). Uses open3d, like normal_bake."""
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(dv.astype(np.float32)),
                        o3d.core.Tensor(df.astype(np.uint32)))
    ans = scene.compute_closest_points(o3d.core.Tensor(query.astype(np.float32)))
    prim = ans["primitive_ids"].numpy().astype(np.int64)
    bary = ans["primitive_uvs"].numpy().astype(np.float64)
    closest = ans["points"].numpy().astype(np.float32)
    tri = df[prim].astype(np.int64)
    w1, w2 = bary[:, 0], bary[:, 1]
    w0 = 1.0 - w1 - w2
    uv = (w0[:, None] * duv[tri[:, 0]] + w1[:, None] * duv[tri[:, 1]]
          + w2[:, None] * duv[tri[:, 2]])
    return uv, closest


def _sample_nearest(img, uv):
    """Nearest-sample a PIL image at accessor-space UVs (v grows downward),
    clamped to [0,1] (painted atlases aren't tiled). Returns (M,3) uint8."""
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    col = np.clip((np.clip(uv[:, 0], 0.0, 1.0) * (w - 1)).round().astype(int), 0, w - 1)
    row = np.clip((np.clip(uv[:, 1], 0.0, 1.0) * (h - 1)).round().astype(int), 0, h - 1)
    return arr[row, col]


def _tangent_frame(N):
    ref = np.where(np.abs(N[:, 1:2]) > 0.9,
                   np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    T = np.cross(ref, N)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    B = np.cross(N, T)
    return T, B


def _hemisphere_dirs(n):
    """n cosine-weighted hemisphere directions (z>0), deterministic (Fibonacci)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(np.sqrt(1.0 - i / n))          # cosine-weighted polar angle
    theta = np.pi * (3.0 - np.sqrt(5.0)) * i        # golden-angle azimuth
    z = np.cos(phi)
    r = np.sin(phi)
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


def _bake_ao(dense_mesh, pos, nrm, idx, size, samples):
    """Hemisphere ray-cast AO for covered texels → (size,size) float in [0,1].
    Never raises; on any failure returns all-white (no occlusion)."""
    ao = np.ones((size * size,), np.float32)
    try:
        origins = pos.reshape(-1, 3)[idx].astype(np.float64)
        N = nrm.reshape(-1, 3)[idx].astype(np.float64)
        ln = np.linalg.norm(N, axis=1, keepdims=True)
        ln[ln == 0] = 1.0
        N = N / ln
        T, B = _tangent_frame(N)
        diag = float(np.linalg.norm(dense_mesh.bounds[1] - dense_mesh.bounds[0]))
        eps = 1e-3 * max(diag, 1e-6)
        maxd = 0.5 * max(diag, 1e-6)
        inter = dense_mesh.ray
        hits = np.zeros(len(origins))
        for dx, dy, dz in _hemisphere_dirs(samples):
            wd = dx * T + dy * B + dz * N
            locs, ray_idx, _ = inter.intersects_location(
                origins + eps * wd, wd, multiple_hits=False)
            if len(ray_idx):
                d = np.linalg.norm(locs - origins[ray_idx], axis=1)
                np.add.at(hits, ray_idx[d < maxd], 1.0)
        ao[idx] = (1.0 - hits / float(samples)).astype(np.float32)
    except Exception as exc:
        print(f"[game_bake] AO skipped ({exc}); leaving white")
    return ao.reshape(size, size)


def bake_maps(dense_verts, dense_faces, dense_uv, dense_albedo, dense_mr,
              low_verts, low_faces, low_uv, low_nrm, size=2048, ao_samples=48):
    """Bake albedo + MR transfer and AO from the dense textured mesh onto the low
    mesh's UVs. Returns {"albedo": Image, "mr": Image|None, "ao": Image}; never raises."""
    out = {}
    try:
        import trimesh
        from PIL import Image
        low_verts = np.asarray(low_verts, float)
        low_faces = np.asarray(low_faces, int)
        low_uv = np.asarray(low_uv, float)
        low_nrm = np.asarray(low_nrm, float)
        pos, nrm, mask = normal_bake.rasterize_uv_atlas(
            low_verts, low_faces, low_uv, low_nrm, size)
        idx = np.where(mask.ravel())[0]
        if idx.size == 0:
            return out

        query = pos.reshape(-1, 3)[idx]
        src_uv, _ = _closest_source_uv(
            np.asarray(dense_verts, float), np.asarray(dense_faces, int),
            np.asarray(dense_uv, float), query)

        alb = np.zeros((size * size, 3), np.uint8)
        alb[idx] = _sample_nearest(dense_albedo, src_uv)
        out["albedo"] = Image.fromarray(
            normal_bake.dilate_map(alb.reshape(size, size, 3), mask), "RGB")

        if dense_mr is not None:
            mr = np.zeros((size * size, 3), np.uint8)
            mr[idx] = _sample_nearest(dense_mr, src_uv)
            out["mr"] = Image.fromarray(
                normal_bake.dilate_map(mr.reshape(size, size, 3), mask), "RGB")

        dense_mesh = trimesh.Trimesh(
            vertices=np.asarray(dense_verts, float),
            faces=np.asarray(dense_faces, int), process=False)
        ao8 = (_bake_ao(dense_mesh, pos, nrm, idx, size, ao_samples) * 255.0
               ).clip(0, 255).astype(np.uint8)
        out["ao"] = Image.fromarray(
            normal_bake.dilate_map(np.repeat(ao8[..., None], 3, axis=2), mask), "RGB")
        return out
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[game_bake] bake_maps failed ({exc})")
        return out
