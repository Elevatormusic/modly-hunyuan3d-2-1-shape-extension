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


def compute_uv_tangents(positions, normals, uvs, faces):
    """Per-vertex UV-space tangents (Lengyel) + glTF handedness w.
    B_view = w * cross(N, T) — matches three.js/glTF exactly (spec RV-A/RV-C)."""
    P = np.asarray(positions, float); N = np.asarray(normals, float)
    UV = np.asarray(uvs, float); F = np.asarray(faces, int)
    tan = np.zeros_like(P); bit = np.zeros_like(P)
    p0, p1, p2 = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    w0, w1, w2 = UV[F[:, 0]], UV[F[:, 1]], UV[F[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    du1, dv1 = w1[:, 0] - w0[:, 0], w1[:, 1] - w0[:, 1]
    du2, dv2 = w2[:, 0] - w0[:, 0], w2[:, 1] - w0[:, 1]
    det = du1 * dv2 - du2 * dv1
    ok = np.abs(det) > 1e-12                      # degenerate UV tris contribute nothing
    r = np.zeros_like(det); r[ok] = 1.0 / det[ok]
    t_f = (dv2[:, None] * e1 - dv1[:, None] * e2) * r[:, None]
    b_f = (du1[:, None] * e2 - du2[:, None] * e1) * r[:, None]
    for c in range(3):
        np.add.at(tan, F[:, c], np.where(ok[:, None], t_f, 0.0))
        np.add.at(bit, F[:, c], np.where(ok[:, None], b_f, 0.0))
    # Gram-Schmidt vs N, normalize, fallback for zero accumulation
    T = tan - (tan * N).sum(axis=1, keepdims=True) * N
    ln = np.linalg.norm(T, axis=1)
    bad = ln < 1e-8
    if bad.any():                                  # any unit vector perpendicular to N
        ref = np.where(np.abs(N[bad, 1:2]) > 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
        Tf = np.cross(ref, N[bad])
        Tf /= np.linalg.norm(Tf, axis=1, keepdims=True)
        T[bad] = Tf; ln[bad] = 1.0
    T = T / ln[:, None]
    w = np.where((np.cross(N, T) * bit).sum(axis=1) < 0.0, -1.0, 1.0)
    w[bad] = 1.0
    return T, w


def read_glb_arrays(glb_path):
    """Positions/normals/uvs/faces straight from the GLB's first primitive
    accessors (pygltflib) — the single source of truth for tangent work."""
    import pygltflib
    g = pygltflib.GLTF2().load(str(glb_path))
    blob = g.binary_blob()
    prim = g.meshes[0].primitives[0]

    def acc(idx, ncomp, dtype):
        a = g.accessors[idx]
        bv = g.bufferViews[a.bufferView]
        off = (bv.byteOffset or 0) + (a.byteOffset or 0)
        out = np.frombuffer(blob, dtype=dtype,
                            count=a.count * ncomp, offset=off)
        return out.reshape(a.count, ncomp).astype(np.float64 if dtype != np.uint32 else np.int64)

    comp = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}
    ia = g.accessors[prim.indices]
    faces = np.frombuffer(blob, dtype=comp[ia.componentType], count=ia.count,
                          offset=(g.bufferViews[ia.bufferView].byteOffset or 0) + (ia.byteOffset or 0)
                          ).astype(np.int64).reshape(-1, 3)
    if prim.attributes.NORMAL is None or prim.attributes.TEXCOORD_0 is None:
        raise ValueError("GLB primitive lacks NORMAL or TEXCOORD_0")
    return dict(
        positions=acc(prim.attributes.POSITION, 3, np.float32),
        normals=acc(prim.attributes.NORMAL, 3, np.float32),
        uvs=acc(prim.attributes.TEXCOORD_0, 2, np.float32),
        faces=faces,
    )


def attach_tangents(glb_path, tangents, w):
    """Append the glTF TANGENT accessor (VEC4 float32, xyz + handedness w) to the
    GLB's first primitive via pygltflib (spec RV-B: trimesh underscore-prefixes
    custom vertex attributes, so it cannot ship a usable TANGENT).

    Returns True on verified success, False on any failure; never raises. The
    count checks run before any mutation, so a mismatch leaves the file untouched.
    """
    try:
        import pygltflib
        tan = np.asarray(tangents, np.float64).reshape(-1, 3)
        ww = np.asarray(w, np.float64).reshape(-1)
        if len(tan) != len(ww):
            print("[normal_bake] tangent/w count mismatch; not attaching TANGENT")
            return False
        g = pygltflib.GLTF2().load(str(glb_path))
        prim = g.meshes[0].primitives[0]
        pos_count = g.accessors[prim.attributes.POSITION].count
        if len(tan) != pos_count:
            print(f"[normal_bake] tangent count {len(tan)} != POSITION count "
                  f"{pos_count}; not attaching TANGENT")
            return False
        blob = g.binary_blob()
        if len(blob) % 4:  # keep the float32 view 4-byte aligned
            blob = blob + b"\x00" * (4 - len(blob) % 4)
        tb = np.hstack([tan, ww[:, None]]).astype("<f4").tobytes()
        g.set_binary_blob(blob + tb)
        g.bufferViews.append(pygltflib.BufferView(
            buffer=0, byteOffset=len(blob), byteLength=len(tb), target=34962))
        g.accessors.append(pygltflib.Accessor(
            bufferView=len(g.bufferViews) - 1, componentType=5126,
            count=len(tan), type="VEC4"))
        prim.attributes.TANGENT = len(g.accessors) - 1
        g.save(str(glb_path))
        # post-save verification: the accessor must round-trip byte-identically
        g2 = pygltflib.GLTF2().load(str(glb_path))
        p2 = g2.meshes[0].primitives[0]
        if p2.attributes.TANGENT is None:
            print("[normal_bake] TANGENT missing after save; export unverified")
            return False
        a2 = g2.accessors[p2.attributes.TANGENT]
        bv2 = g2.bufferViews[a2.bufferView]
        off = (bv2.byteOffset or 0) + (a2.byteOffset or 0)
        got = bytes(g2.binary_blob()[off: off + a2.count * 16])
        if a2.type != "VEC4" or a2.componentType != 5126 or got != tb:
            print("[normal_bake] TANGENT round-trip mismatch; export unverified")
            return False
        return True
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[normal_bake] TANGENT export failed ({exc})")
        return False


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


def sample_dense_normals(dense, pos_map, mask, low_nrm=None, max_dist_frac=0.05):
    """For each covered texel, the dense mesh's SMOOTH normal at its closest point.

    When `low_nrm` is given, reject wrong-surface hits on concave/thin-wall meshes:
    a closest point that is too far (> max_dist_frac * bbox diagonal) or whose dense
    normal OPPOSES the low normal (back-face snap) falls back to the low normal, so a
    bad correspondence bakes flat rather than an inverted bump.
    """
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
    closest = ans["points"].numpy().astype(np.float32)
    tri = df[prim].astype(np.int64)
    w1, w2 = bary[:, 0], bary[:, 1]
    w0 = 1.0 - w1 - w2
    n = (w0[:, None] * vnorm[tri[:, 0]]
         + w1[:, None] * vnorm[tri[:, 1]]
         + w2[:, None] * vnorm[tri[:, 2]])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    n = n / ln

    if low_nrm is not None:
        low_q = np.asarray(low_nrm, np.float32).reshape(-1, 3)[idx]
        diag = float(np.linalg.norm(dense.bounds[1] - dense.bounds[0]))
        dist = np.linalg.norm(query - closest, axis=1)
        opposed = np.sum(n * low_q, axis=1) < 0.0
        toofar = dist > (max_dist_frac * max(diag, 1e-9))
        bad = opposed | toofar
        n[bad] = low_q[bad]

    flat = world.reshape(-1, 3)
    flat[idx] = n.astype(np.float32)
    return flat.reshape(size, size, 3)


def encode_tangent_space(low_nrm, world_nrm, mask):
    """Transform world-space dense normals into each texel's tangent frame (+Y encode).

    Frame is built from the interpolated low normal + a world reference (T = ref x N,
    B = N x T); a glTF viewer recomputes MikkTSpace tangents at render, so an exact
    per-texel UV-gradient tangent is not required — only a stable, right-handed frame.
    """
    up = np.array([0.0, 1.0, 0.0])
    N = low_nrm
    ref = np.where(np.abs(N[..., 1:2]) > 0.99, np.array([1.0, 0.0, 0.0]), up)
    T = np.cross(ref, N)
    T /= (np.linalg.norm(T, axis=2, keepdims=True) + 1e-9)
    B = np.cross(N, T)
    nt = np.stack([
        np.sum(world_nrm * T, axis=2),
        np.sum(world_nrm * B, axis=2),
        np.sum(world_nrm * N, axis=2),
    ], axis=2)
    ln = np.linalg.norm(nt, axis=2, keepdims=True)
    ln[ln == 0] = 1.0
    nt = nt / ln
    rgb = ((nt * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
    # Uncovered texels MUST be the neutral tangent normal (128,128,255 = flat), NOT
    # black. Black (0,0,0) decodes to an invalid inward-facing normal that renders as
    # dark blotches wherever the mesh samples a seam/gap; neutral just shades flat.
    rgb[~mask] = (128, 128, 255)
    return rgb


def dilate_map(rgb, mask, iters=6):
    """Grow covered texels a few px past the atlas charts to kill UV-seam bleed."""
    from scipy import ndimage
    out = rgb.copy()
    m = mask.copy()
    for _ in range(iters):
        grown = ndimage.binary_dilation(m)
        edge = grown & ~m
        if not edge.any():
            break
        den = ndimage.uniform_filter(m.astype(np.float32), size=3) * 9
        for c in range(3):
            ch = out[..., c].astype(np.float32)
            ch[~m] = 0
            num = ndimage.uniform_filter(ch, size=3) * 9
            filled = np.where(den > 0, num / np.maximum(den, 1), 0)
            out[..., c] = np.where(edge, filled.clip(0, 255).astype(np.uint8), out[..., c])
        m = grown
    return out


def attach_normal_texture(glb_path, normal_png):
    """Attach a normal map to an existing GLB, forcing NORMAL export (no TANGENT)."""
    import trimesh
    from PIL import Image
    scene = trimesh.load(glb_path, process=False)
    geom = (list(scene.geometry.values())[0]
            if isinstance(scene, trimesh.Scene) else scene)
    mat = geom.visual.material
    if not isinstance(mat, trimesh.visual.material.PBRMaterial):
        to_pbr = getattr(mat, "to_pbr", None)
        mat = to_pbr() if to_pbr else trimesh.visual.material.PBRMaterial()
    mat.normalTexture = Image.open(normal_png)
    uv = geom.visual.uv
    geom.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    _ = geom.vertex_normals  # force NORMAL attribute on export
    geom.export(glb_path, include_normals=True)


def bake_normal_map(dense, low_glb_path, size=2048):
    """Bake dense-mesh detail onto the painted low base as a tangent-space normal map.

    Returns True on success (writes <base>_normal.png + attaches normalTexture),
    False on any failure (leaves the textured GLB unchanged).
    """
    try:
        import trimesh
        from PIL import Image
        scene = trimesh.load(low_glb_path, process=False)
        low = (trimesh.util.concatenate(tuple(scene.geometry.values()))
               if isinstance(scene, trimesh.Scene) else scene)
        if low.visual is None or getattr(low.visual, "uv", None) is None:
            print("[normal_bake] low mesh has no UVs; skipping")
            return False
        uv = np.asarray(low.visual.uv, np.float64)
        pos, low_nrm, mask = rasterize_uv_atlas(
            low.vertices, low.faces, uv, low.vertex_normals, size)
        world_nrm = sample_dense_normals(dense, pos, mask, low_nrm=low_nrm)
        rgb = encode_tangent_space(low_nrm, world_nrm, mask)
        rgb = dilate_map(rgb, mask)
        png = low_glb_path[:-4] + "_normal.png"
        Image.fromarray(rgb, "RGB").save(png)
        attach_normal_texture(low_glb_path, png)
        print(f"[normal_bake] wrote {png} and attached normalTexture")
        return True
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[normal_bake] bake failed ({exc}); leaving textured GLB unchanged")
        return False
