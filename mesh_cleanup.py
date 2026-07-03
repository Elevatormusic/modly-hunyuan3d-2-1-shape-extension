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


def clean_mesh(mesh, mode: str = "isotropic", target_faces: int = 40000) -> trimesh.Trimesh:
    hi = _as_trimesh(mesh)
    try:
        if mode == "regular":
            return _quadric(hi, target_faces)
        if mode == "isotropic":
            return _isotropic(hi, target_faces)
        if mode == "bpt":
            from bpt_runner import retopo  # wired in Task 6
            low = retopo(hi)
            if low is None:
                raise RuntimeError("bpt retopo unavailable")
            return low
        raise ValueError(f"unknown mesh_mode {mode!r}")
    except Exception as exc:  # never break a generation
        print(f"[mesh_cleanup] mode {mode!r} failed ({exc}); quadric fallback")
        try:
            return _quadric(hi, target_faces)
        except Exception as exc2:
            print(f"[mesh_cleanup] quadric fallback failed ({exc2}); returning raw mesh")
            return hi
