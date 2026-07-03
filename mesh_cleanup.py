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
    # Remeshing controls EDGE LENGTH, not face count. Estimate a target edge length
    # from the current mesh, then express it as a % of the bbox diagonal (required
    # wrapper type). n_faces ~ (2 * surface_area) / (sqrt(3)/2 * edge^2) for a
    # triangulated surface, so edge ~ sqrt(4*area / (sqrt(3) * n_faces)).
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    area = float(mesh.area)
    edge = float(np.sqrt(4.0 * area / (np.sqrt(3.0) * max(target_faces, 1))))
    pct = max(0.1, min(50.0, 100.0 * edge / max(diag, 1e-9)))
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, np.float64),
        face_matrix=np.asarray(mesh.faces, np.int32),
    ))
    ms.meshing_isotropic_explicit_remeshing(
        iterations=8,
        targetlen=pymeshlab.PercentageValue(pct),
    )
    out = ms.current_mesh()
    return trimesh.Trimesh(
        vertices=out.vertex_matrix(), faces=out.face_matrix(), process=False)


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
