"""Quad-dominant retopology with a tiered, never-raise fallback.

`retopo_quads(mesh, target_triangles)`:
  1. Instant Meshes (vendored BSD-3 exe, headless) — best edge flow.
  2. pymeshlab isotropic remesh + tri-to-quad pairing — no exe needed.
  3. quadric decimation — last resort.

Returns a triangulated trimesh (quad topology only lives in the intermediate OBJ;
GLB is triangles anyway). Instant Meshes `-v`/`-f` count QUADS, so we pass roughly
half the triangle target. Never raises.
"""
from __future__ import annotations
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh

import mesh_cleanup

# Vendored Instant Meshes (BSD-3-Clause). Resolved module-relative; absent in tests.
_EXE = Path(__file__).resolve().parent / "vendor" / "instant-meshes" / "Instant Meshes.exe"


def _instant_meshes(mesh, target_triangles):
    """Run the vendored Instant Meshes exe headlessly. Returns a trimesh or None."""
    if not _EXE.exists():
        return None
    # Instant Meshes' -v/-f is a SOFT target: output faces run ~8x the passed
    # value (measured, consistent). Aim near the budget here; retopo_quads then
    # hard-caps any residual overshoot by decimation.
    quads = max(50, round(target_triangles / 8))
    d = Path(tempfile.mkdtemp(prefix="im_"))
    try:
        in_obj, out_obj = d / "in.obj", d / "out.obj"
        mesh.export(str(in_obj))
        cmd = [str(_EXE), str(in_obj), "-o", str(out_obj),
               "-v", str(quads), "-d", "-c", "30"]
        subprocess.run(cmd, capture_output=True, timeout=600, check=False)
        if not out_obj.exists() or out_obj.stat().st_size == 0:
            return None
        res = trimesh.load(str(out_obj), process=False, force="mesh")
        if res is None or len(getattr(res, "faces", [])) == 0:
            return None
        return res
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _pymeshlab_quad(mesh, target_triangles):
    """Fallback: isotropic remesh to ~target, then pair triangles into quads.
    Returns a (triangulated-on-load) trimesh or None."""
    try:
        import pymeshlab
        iso = mesh_cleanup._isotropic(mesh_cleanup._as_trimesh(mesh), target_triangles)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=np.asarray(iso.vertices, np.float64),
            face_matrix=np.asarray(iso.faces, np.int32)))
        try:
            ms.meshing_tri_to_quad_by_smart_triangle_pairing()
        except Exception:
            pass  # pairing is a nicety; the isotropic tris are already usable
        out = ms.current_mesh()
        res = trimesh.Trimesh(vertices=out.vertex_matrix(),
                              faces=out.face_matrix(), process=False)
        return res if len(res.faces) > 0 else None
    except Exception:
        return None


def _cap(mesh, target_triangles):
    """Hard-cap the face budget: decimate to target if a tier overshoots >20%.
    Guarantees game-ready output is actually low-poly regardless of the engine."""
    if len(mesh.faces) > target_triangles * 1.2:
        try:
            capped = mesh_cleanup._quadric(mesh, target_triangles)
            print(f"[retopo] capped {len(mesh.faces)} -> {len(capped.faces)} faces (budget)")
            return capped
        except Exception:
            pass
    return mesh


def retopo_quads(mesh, target_triangles):
    """Quad-dominant retopo with fallbacks + a hard budget cap. Never raises."""
    hi = mesh_cleanup._as_trimesh(mesh)
    res = None
    try:
        res = _instant_meshes(hi, target_triangles)
        if res is not None:
            print(f"[retopo] instant-meshes -> {len(res.faces)} faces")
    except Exception as exc:
        print(f"[retopo] instant-meshes failed ({exc})")
        res = None
    if res is None:
        try:
            res = _pymeshlab_quad(hi, target_triangles)
            if res is not None:
                print(f"[retopo] pymeshlab isotropic+tri2quad -> {len(res.faces)} faces")
        except Exception as exc:
            print(f"[retopo] pymeshlab fallback failed ({exc})")
            res = None
    if res is None:
        print("[retopo] falling back to quadric decimation")
        try:
            res = mesh_cleanup._quadric(hi, target_triangles)
        except Exception as exc:
            print(f"[retopo] quadric fallback failed ({exc}); returning input")
            return hi
    return _cap(res, target_triangles)
