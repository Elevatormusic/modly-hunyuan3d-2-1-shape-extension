# smooth_normals.py
"""Crease-aware smooth vertex normals for the final textured GLB.

The paint/export path ships a GLB with POSITION+TEXCOORD_0 and no NORMAL, so
glTF viewers (Modly's three.js) flat-shade it -> faceted. This writes per-vertex
smooth normals with a crease-angle threshold: shared edges below the threshold
are smoothed; edges >= threshold stay hard (vertices split, per-side normals).

Pure numpy + trimesh core (crease_smooth) + a trimesh re-export wrapper
(apply_to_glb). No torch, no GPU, no bpy. Wired into finishing.finish() after
seam_fix and before the bake. Design + research validation in
private/specs/2026-07-10-smooth-normals-{design,research}.md.
"""
from __future__ import annotations
import numpy as np


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
    and vertices are split only across creases (dihedral >= crease_deg). Each
    output vertex's normal is the area-weighted average of the faces incident to
    its (welded_position, smoothing_group) — so a UV seam with no crease keeps a
    single shared normal (smooth). Normals are unit and never NaN. new_faces has
    the same triangle count as faces.
    """
    import trimesh
    from trimesh.graph import connected_components

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

    # 3. smoothing groups: join faces only across sub-crease welded edges
    if len(adj):
        keep = adj[ang < np.radians(crease_deg)]
    else:
        keep = np.zeros((0, 2), np.int64)
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
