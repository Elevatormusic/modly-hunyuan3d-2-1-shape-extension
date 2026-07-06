"""Practical structural validation of a textured GLB using trimesh.

validate_glb(glb_path) -> {"ok": bool, "warnings": [str], "info": {...}}
Non-fatal QA in Branch 1 (Branch 3 upgrades this to the Khronos gate).
"""
import numpy as np


def _geoms(sc):
    import trimesh
    if isinstance(sc, trimesh.Scene):
        return list(sc.geometry.values())
    return [sc]


def validate_glb(glb_path):
    import trimesh
    warnings = []
    info = {}
    try:
        sc = trimesh.load(glb_path, process=False)
    except Exception as exc:
        return {"ok": False, "warnings": [f"load failed: {exc}"], "info": {}}

    geoms = _geoms(sc)
    info["geometries"] = len(geoms)
    if not geoms:
        return {"ok": False, "warnings": ["no geometry"], "info": info}

    nf = sum(len(getattr(g, "faces", [])) for g in geoms)
    info["faces"] = int(nf)
    if nf == 0:
        warnings.append("no faces")

    has_material = False
    has_uv = False
    for g in geoms:
        vis = getattr(g, "visual", None)
        mat = getattr(vis, "material", None)
        if mat is not None:
            has_material = True
        uv = getattr(vis, "uv", None)
        if uv is not None and len(uv) > 0:
            has_uv = True
            uv = np.asarray(uv, float)
            if not np.isfinite(uv).all():
                warnings.append("non-finite uv coordinates")
        verts = np.asarray(g.vertices, float)
        if verts.size and not np.isfinite(verts).all():
            warnings.append("non-finite vertices")
        for attr in ("baseColorTexture", "metallicRoughnessTexture", "normalTexture"):
            tex = getattr(mat, attr, None)
            if tex is not None:
                try:
                    tex.convert("RGB")
                except Exception:
                    warnings.append(f"{attr} does not decode")
    if not has_material:
        warnings.append("no material on any geometry")
    if not has_uv:
        warnings.append("no uv coordinates")

    info["has_material"] = has_material
    info["has_uv"] = has_uv
    return {"ok": len(warnings) == 0, "warnings": warnings, "info": info}
