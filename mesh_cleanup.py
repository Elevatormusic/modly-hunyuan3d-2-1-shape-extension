"""Mesh-cleanup strategies for the Hunyuan3D-2.1 shape mesh.

regular  = quadric decimation (open3d wrapper) to an exact-ish face budget.
isotropic= pymeshlab adaptive isotropic remesh (uniform triangles).
bpt      = neural retopology via an isolated sub-venv subprocess (see bpt_runner).

clean_mesh never raises: any failure falls back to quadric decimation.
"""
from __future__ import annotations
import numpy as np
import trimesh


def _as_trimesh(mesh) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Scene):
        return trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _quadric(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= target_faces:
        return mesh
    return mesh.simplify_quadric_decimation(face_count=target_faces)


def _isotropic(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    import pymeshlab
    # Remeshing controls EDGE LENGTH, not face count, so a mis-estimated edge could
    # otherwise run away to millions of faces (a tiny targetlen splits forever).
    # Bound it three ways: clamp the request, pre-reduce dense input, floor the pct,
    # and decimate to budget if the remesh still overshoots.
    target = int(max(500, min(int(target_faces), 300000)))  # bound the request
    work = mesh
    if len(work.faces) > target * 3:                        # pre-reduce (speed + bound)
        work = _quadric(work, target * 2)
    # n_faces ~ 4*area / (sqrt(3) * edge^2)  ->  edge ~ sqrt(4*area / (sqrt(3)*n)).
    diag = float(np.linalg.norm(work.bounds[1] - work.bounds[0]))
    area = float(work.area)
    edge = float(np.sqrt(4.0 * area / (np.sqrt(3.0) * max(target, 1))))
    pct = float(np.clip(100.0 * edge / max(diag, 1e-9), 0.3, 50.0))  # floor bounds fineness
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=np.asarray(work.vertices, np.float64),
        face_matrix=np.asarray(work.faces, np.int32),
    ))
    ms.meshing_isotropic_explicit_remeshing(
        iterations=6,
        targetlen=pymeshlab.PercentageValue(pct),
    )
    out = ms.current_mesh()
    res = trimesh.Trimesh(
        vertices=out.vertex_matrix(), faces=out.face_matrix(), process=False)
    if len(res.faces) > target * 3:                         # final safety on overshoot
        res = _quadric(res, target)
    return res


def make_watertight(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Repair a cleaned mesh to a single WATERTIGHT shell.

    Hunyuan's marching-cubes surfaces are non-manifold/open, so isotropic remesh shatters
    them into many disconnected islands (verified: 161 components) -> xatlas then makes a
    fragmented, seamy UV layout. pymeshlab's remove-duplicates + repair-non-manifold +
    close-holes welds it back into ONE watertight component (verified: 161 -> 1, watertight
    True) -> one clean UV island, a cleaner normal bake, and a printable asset. Cheap: it
    runs on the ~50k cleaned mesh, not the millions-of-faces dense input. Never raises.
    """
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, np.float64),
            face_matrix=np.asarray(mesh.faces, np.int32)))
        for f in ("meshing_remove_duplicate_vertices",
                  "meshing_remove_unreferenced_vertices",
                  "meshing_remove_duplicate_faces",
                  "meshing_repair_non_manifold_edges",
                  "meshing_repair_non_manifold_vertices",
                  "meshing_re_orient_faces_coherently"):
            try:
                ms.apply_filter(f)
            except Exception:
                pass
        try:
            ms.meshing_close_holes(maxholesize=5000)
        except Exception:
            pass
        out = ms.current_mesh()
        rep = trimesh.Trimesh(vertices=out.vertex_matrix(),
                              faces=out.face_matrix(), process=False)
        if len(rep.faces) > 0:
            return rep
    except Exception as exc:
        print(f"[mesh_cleanup] watertight repair skipped ({exc})")
    return mesh


def strip_background(mesh) -> trimesh.Trimesh:
    """Remove large flat background/ground planes and tiny stray fragments.

    Image-to-3D models (Hunyuan included) often generate a big flat slab from the
    photo's backdrop. Left in, it dominates the mesh: isotropic remesh sizes triangles
    by area, so the plane eats ~90% of the face budget and starves the real object.
    Drop components that are flat AND span most of the scene AND are large; also drop
    sub-50-face specks. No-op if there's a single component or nothing non-flat remains
    (so a genuinely flat object — a coin, a plate — is never nuked).
    """
    hi = _as_trimesh(mesh)
    try:
        comps = hi.split(only_watertight=False)
    except Exception:
        return hi
    if len(comps) <= 1:
        return hi
    total = len(hi.faces)
    scene_span = float((hi.vertices.max(0) - hi.vertices.min(0)).max())
    keep = []
    for c in comps:
        e = c.vertices.max(0) - c.vertices.min(0)
        flat = (float(e.min()) / max(float(e.max()), 1e-9)) < 0.05
        wide = float(e.max()) > 0.40 * scene_span
        big = len(c.faces) > 0.08 * total
        is_plane = flat and wide and big
        tiny = len(c.faces) < 50
        if not is_plane and not tiny:
            keep.append(c)
    if not keep:
        return hi  # never return an empty mesh (e.g. a lone flat object)
    return trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]


def smooth_crevices(mesh, *, k_percentile=8, rings=2, steps=10):
    """Curvature-masked, non-shrinking Taubin smoothing of concave crevice bands
    (the MC staircase contact lines). Only strongly-concave vertices move; convex
    silhouettes are untouched. Honest scope: a mild polish (single-digit-% de-
    serration per RV) — the geometric fix is the face budget + DMC. Never raises;
    EB_CREVICE_SMOOTH=off returns the mesh unchanged."""
    import os
    if os.environ.get("EB_CREVICE_SMOOTH", "").strip().lower() == "off":
        return mesh
    try:
        import numpy as np
        import pymeshlab
        v = np.asarray(mesh.vertices, np.float64)
        f = np.asarray(mesh.faces, np.int64)
        if len(f) < 8:
            return mesh
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v, face_matrix=f))
        ms.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype="Mean Curvature")
        q = ms.current_mesh().vertex_scalar_array()
        # concave band = low (negative) mean curvature tail; K = |k_percentile pct|
        thr = float(np.percentile(q, k_percentile))
        if not np.isfinite(thr):
            return mesh
        ms.compute_selection_by_condition_per_vertex(condselect=f"q < {thr!r}")
        # dilate the VERTEX band via the v->f->dilate->f->v idiom (dilatation acts
        # on the FACE selection and clears the vertex selection — RV gotcha).
        ms.compute_selection_transfer_vertex_to_face(inclusive=False)
        for _ in range(max(0, int(rings))):
            ms.apply_selection_dilatation()
        ms.compute_selection_transfer_face_to_vertex(inclusive=True)
        ms.apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53,
                                        stepsmoothnum=int(steps), selected=True)
        out = ms.current_mesh()
        import trimesh
        return trimesh.Trimesh(vertices=out.vertex_matrix(),
                               faces=out.face_matrix(), process=False)
    except Exception as exc:
        print(f"[mesh_cleanup] smooth_crevices masked path failed ({exc}); trying global humphrey")
        try:
            import trimesh
            m2 = mesh.copy()
            trimesh.smoothing.filter_humphrey(m2, alpha=0.1, beta=0.5, iterations=10)
            return m2
        except Exception as exc2:
            print(f"[mesh_cleanup] smooth_crevices fallback failed ({exc2}); returning input")
            return mesh


def clean_mesh(mesh, mode: str = "regular", target_faces: int = 40000) -> trimesh.Trimesh:
    hi = _as_trimesh(mesh)
    try:
        if mode == "regular":
            low = _quadric(hi, target_faces)
        elif mode == "isotropic":
            low = _isotropic(hi, target_faces)
        elif mode == "bpt":
            from bpt_runner import retopo  # wired in Task 6
            low = retopo(hi)
            if low is None:
                raise RuntimeError("bpt retopo unavailable")
        else:
            raise ValueError(f"unknown mesh_mode {mode!r}")
    except Exception as exc:  # never break a generation
        print(f"[mesh_cleanup] mode {mode!r} failed ({exc}); quadric fallback")
        try:
            low = _quadric(hi, target_faces)
        except Exception as exc2:
            print(f"[mesh_cleanup] quadric fallback failed ({exc2}); returning raw mesh")
            return hi
    # Repair to a single watertight shell -> one clean UV island (fixes isotropic's
    # fragmentation for good; also helps quadric/bpt). Cheap on the ~50k cleaned mesh.
    sealed = make_watertight(low)
    # Curvature-masked crevice polish on the concave MC contact bands (never raises;
    # EB_CREVICE_SMOOTH=off makes it an identity). All mode paths converge here.
    return smooth_crevices(sealed)
