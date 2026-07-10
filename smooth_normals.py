# smooth_normals.py
"""Crease-aware smooth vertex normals for the final textured GLB.

The paint/export path ships a GLB with POSITION+TEXCOORD_0 and no NORMAL, so
glTF viewers (Modly's three.js) flat-shade it -> faceted. This writes per-vertex
smooth normals with crease-aware selection: hard edges (>= crease_deg +
CREASE_HYST_DEG) anchor crease chains, band edges join only via a chain that
contains a hard edge and spans >= MIN_CHAIN_EDGES, and everything else —
including dropped fragments — is smoothed (kept creases split vertices with
per-side normals).

Pure numpy + trimesh core (crease_smooth) + a trimesh re-export wrapper
(apply_to_glb). No torch, no GPU, no bpy. Wired into finishing.finish() after
seam_fix and before the bake. Design + research validation in
private/specs/2026-07-10-smooth-normals-{design,research}.md.
"""
from __future__ import annotations
import numpy as np

# Crease-selection hysteresis + chain-coherence knobs (see
# private/plans/2026-07-10-crease-coherence.md). A welded edge is a HARD crease
# only at >= crease_deg + CREASE_HYST_DEG; edges in the [crease_deg-HYST,
# crease_deg+HYST) BAND are creases only when they belong to a connected crease
# chain that (a) contains at least one hard edge and (b) spans >= MIN_CHAIN_EDGES
# edges. This suppresses the noisy 1-2-edge crevice fragments that otherwise
# flicker hard/soft along clump contact lines.
CREASE_HYST_DEG = 10.0
MIN_CHAIN_EDGES = 5


def _weld_index(vertices, tol_rel=1e-6):
    """Map each vertex -> a welded (by-position) id. Same rounding scheme as
    seam_fix._weld_index; duplicated here so this module needs no seam_fix import."""
    v = np.asarray(vertices, dtype=np.float64)
    if v.size == 0:
        return np.zeros(len(vertices), dtype=np.int64)
    scale = float(np.ptp(v, axis=0).max()) or 1.0
    keys = np.round(v / (scale * tol_rel)).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    return inv.reshape(-1)


def crease_smooth(positions, faces, uvs, *, crease_deg=45.0):
    """Crease-aware smooth per-vertex normals.

    Returns (new_positions, new_uvs, new_faces, normals). Output vertices are
    keyed by (input_vertex, smoothing_group) so existing UV seams are preserved
    and vertices are split only across KEPT crease chains (hysteresis + chain
    coherence — see the module constants; isolated fragments smooth). Each
    output vertex's normal is the area-weighted average of the faces incident to
    its (welded_position, smoothing_group) — so a UV seam with no crease keeps a
    single shared normal (smooth). Normals are unit and never NaN. new_faces has
    the same triangle count as faces.
    """
    import trimesh
    from trimesh.graph import connected_components
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as _cc_labels

    positions = np.asarray(positions, np.float64)
    uvs = np.asarray(uvs, np.float64)
    faces = np.asarray(faces, np.int64)
    n_faces = len(faces)
    if n_faces == 0:
        return positions.copy(), uvs.copy(), faces.copy(), np.zeros((len(positions), 3))

    # 1. weld by position -> geometric topology for adjacency
    weld = _weld_index(positions)
    wpos = np.zeros((int(weld.max()) + 1, 3), np.float64)
    wpos[weld] = positions
    wfaces = weld[faces]

    # 2. welded trimesh: robust adjacency + dihedral + areas + face normals
    m = trimesh.Trimesh(vertices=wpos, faces=wfaces, process=False)
    fn = np.asarray(m.face_normals, np.float64)        # zero-vector for degenerate faces
    fa = np.asarray(m.area_faces, np.float64)
    adj = np.asarray(m.face_adjacency)                 # (k,2) exactly-2-face welded edges
    ang = np.asarray(m.face_adjacency_angles, np.float64)

    # 3. crease selection: hysteresis + chain coherence. Hard edges anchor
    #    crease chains; a band edge is a crease only when its connected candidate
    #    chain contains a hard edge AND is >= MIN_CHAIN_EDGES long. crease_final is
    #    a boolean over adjacency rows (True == keep as a hard shading edge).
    crease_final = np.zeros(len(adj), dtype=bool)
    if len(adj):
        crease_rad = np.radians(crease_deg)
        hyst_rad = np.radians(CREASE_HYST_DEG)
        hard = ang >= (crease_rad + hyst_rad)
        band = (ang >= (crease_rad - hyst_rad)) & ~hard
        cand = hard | band
        if cand.any():
            fae = np.asarray(m.face_adjacency_edges, np.int64)  # welded vertex pair / adj row
            nv = len(wpos)
            ci = np.where(cand)[0]                          # candidate adjacency rows
            va, vb = fae[ci, 0], fae[ci, 1]
            # chain candidates via shared welded vertices: connected components of
            # the welded-vertex graph whose edges are the candidate creases.
            graph = coo_matrix((np.ones(len(ci)), (va, vb)), shape=(nv, nv))
            _, labels = _cc_labels(graph, directed=False)
            elab = labels[va]                              # chain id per candidate edge
            n_lab = int(labels.max()) + 1
            edges_per = np.bincount(elab, minlength=n_lab)
            hard_per = np.bincount(elab, weights=hard[ci].astype(np.float64), minlength=n_lab)
            kept_lab = (edges_per >= MIN_CHAIN_EDGES) & (hard_per > 0.0)
            crease_final[ci] = kept_lab[elab]

    # 4. smoothing groups: join every non-crease adjacency (incl. dropped
    #    candidates) so only surviving crease chains split vertices.
    keep = adj[~crease_final] if len(adj) else np.zeros((0, 2), np.int64)
    comps = connected_components(keep, nodes=np.arange(n_faces), min_len=1)
    face_group = np.zeros(n_faces, np.int64)
    for gi, comp in enumerate(comps):
        face_group[np.asarray(comp, np.int64)] = gi

    corner_g = np.repeat(face_group, 3)                # smoothing group per face corner
    wfn = fn * fa[:, None]                             # area-weighted face normals
    corner_wfn = np.repeat(wfn, 3, axis=0)             # per corner

    # 4a. NORMAL accumulation keyed by (welded_position, group) -> smooths UV seams
    corner_wid = weld[faces].reshape(-1)
    nkey = np.stack([corner_wid, corner_g], axis=1)
    nuniq, ninv = np.unique(nkey, axis=0, return_inverse=True)
    Nacc = np.zeros((len(nuniq), 3), np.float64)
    np.add.at(Nacc, ninv, corner_wfn)
    ln = np.linalg.norm(Nacc, axis=1)
    good = ln > 1e-12
    Nacc[good] /= ln[good, None]
    Nacc[~good] = np.array([0.0, 0.0, 1.0])            # degenerate-only vertex fallback

    # 4b. OUTPUT vertices keyed by (input_vertex, group) -> preserves UV seams
    vkey = np.stack([faces.reshape(-1), corner_g], axis=1)
    vuniq, vinv = np.unique(vkey, axis=0, return_inverse=True)
    src_v = vuniq[:, 0]
    new_positions = positions[src_v]
    new_uvs = uvs[src_v]
    new_faces = vinv.reshape(-1, 3)

    # 5. each output vertex inherits its (welded_position, group) normal
    #    map (welded_id, group) -> row in nuniq via a dict (O(V))
    nrow = {(int(a), int(b)): i for i, (a, b) in enumerate(nuniq)}
    out_wid = weld[src_v]
    out_g = vuniq[:, 1]
    normals = np.array([Nacc[nrow[(int(w), int(g))]] for w, g in zip(out_wid, out_g)],
                       dtype=np.float64)
    return new_positions, new_uvs, new_faces, normals


def _verify_glb(path):
    """True iff first primitive has NORMAL (count == POSITION), a TEXCOORD_0, and
    at least one material — the post-export invariants."""
    # NOTE: checks the FIRST primitive only (single-mesh assumption; the production asset is single-mesh).
    import pygltflib
    g = pygltflib.GLTF2().load(path)
    prim = g.meshes[0].primitives[0]
    a = prim.attributes
    if a.NORMAL is None or a.TEXCOORD_0 is None:
        return False
    if g.accessors[a.NORMAL].count != g.accessors[a.POSITION].count:
        return False
    return bool(g.materials)


def apply_to_glb(glb_path, *, crease_deg=45.0):
    """Rewrite the GLB in place with crease-aware smooth vertex normals via a
    trimesh re-export (RV-1: writes NORMAL, carries exact normals, preserves
    material + textures + UV + indices). Atomic + verified; never raises. Returns
    True on verified success, False otherwise (original left byte-identical)."""
    import os
    import tempfile
    try:
        import trimesh
        scene = trimesh.load(glb_path, process=False)
        geoms = scene.geometry if hasattr(scene, "geometry") else None
        names = list(geoms.keys()) if geoms is not None else []
        if not names:
            return False
        touched = False
        for name in names:
            g = geoms[name]
            v = getattr(g, "visual", None)
            uv = getattr(v, "uv", None)
            mat = getattr(v, "material", None)
            if uv is None or mat is None:
                continue
            P = np.asarray(g.vertices, np.float64)
            F = np.asarray(g.faces, np.int64)
            UV = np.asarray(uv, np.float64)
            newP, newUV, newF, N = crease_smooth(P, F, UV, crease_deg=crease_deg)
            g2 = trimesh.Trimesh(vertices=newP, faces=newF, process=False)
            g2.vertex_normals = N                          # assigned normals export verbatim (RV-1 iii)
            g2.visual = trimesh.visual.TextureVisuals(uv=newUV, material=mat)  # REUSE original material
            geoms[name] = g2
            touched = True
        if not touched:
            return False
        # atomic verified swap: export to temp, verify, then os.replace
        d = os.path.dirname(os.path.abspath(glb_path))
        fd, tmp = tempfile.mkstemp(suffix=".glb", dir=d)
        os.close(fd)
        try:
            scene.export(tmp)
            if not _verify_glb(tmp):
                os.remove(tmp)
                return False
            os.replace(tmp, glb_path)
            return True
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
    except Exception as exc:
        print(f"[smooth_normals] skipped ({exc})")
        return False
