"""Game-ready pipeline.

dense textured GLB -> quad retopo -> xatlas UV unwrap -> transferred PBR
(albedo/MR) + baked AO + baked tangent-space normal -> low-poly GLB.

Never raises. On any unrecoverable failure it returns the INPUT dense GLB path
unchanged — in particular, an xatlas failure returns the dense GLB rather than
reusing the dense mesh's atlas on the (differently-topologised) retopo mesh.
"""
from __future__ import annotations
import numpy as np
import trimesh

import retopo
import game_bake
import normal_bake


def _mesh_only(m):
    if isinstance(m, trimesh.Scene):
        return trimesh.util.concatenate(tuple(m.geometry.values()))
    return m


def _load_dense(glb_path):
    """Dense trimesh + per-vertex UV + albedo/MR PIL images from a textured GLB."""
    geom = _mesh_only(trimesh.load(glb_path, process=False))
    uv = np.asarray(getattr(geom.visual, "uv", None), float)
    mat = getattr(geom.visual, "material", None)
    albedo = getattr(mat, "baseColorTexture", None) if mat else None
    mr = getattr(mat, "metallicRoughnessTexture", None) if mat else None
    return geom, uv, albedo, mr


def to_game_ready(dense_textured_glb, target_triangles=30000, tex_size=2048):
    """Return a game-ready low-poly GLB path, or the input dense path on failure."""
    try:
        import xatlas
        dense, dense_uv, albedo, mr = _load_dense(dense_textured_glb)
        if albedo is None or dense_uv is None or dense_uv.size == 0:
            print("[game_ready] dense mesh lacks albedo/UVs; returning dense GLB")
            return dense_textured_glb

        low = _mesh_only(retopo.retopo_quads(dense, target_triangles))

        try:
            vmapping, indices, uvs = xatlas.parametrize(
                np.asarray(low.vertices, np.float32),
                np.asarray(low.faces, np.uint32))
        except Exception as exc:
            print(f"[game_ready] xatlas failed ({exc}); returning dense GLB")
            return dense_textured_glb

        low_verts = np.asarray(low.vertices)[np.asarray(vmapping)]
        low_faces = np.asarray(indices)
        low_uv = np.asarray(uvs)
        low_nrm = trimesh.Trimesh(low_verts, low_faces, process=False).vertex_normals

        maps = game_bake.bake_maps(
            np.asarray(dense.vertices), np.asarray(dense.faces), dense_uv,
            albedo, mr, low_verts, low_faces, low_uv, low_nrm, size=tex_size)
        if "albedo" not in maps:
            print("[game_ready] bake produced no albedo; returning dense GLB")
            return dense_textured_glb

        from trimesh.visual.material import PBRMaterial
        mat = PBRMaterial(baseColorTexture=maps["albedo"],
                          baseColorFactor=[255, 255, 255, 255])
        if maps.get("mr") is not None:
            mat.metallicRoughnessTexture = maps["mr"]
        if maps.get("ao") is not None:
            mat.occlusionTexture = maps["ao"]
        low_mesh = trimesh.Trimesh(low_verts, low_faces, process=False)
        low_mesh.visual = trimesh.visual.TextureVisuals(uv=low_uv, material=mat)
        out_path = dense_textured_glb[:-4] + "_gameready.glb"
        low_mesh.export(out_path)

        # normal + TANGENT baked from the dense mesh onto the low UVs (non-fatal)
        try:
            normal_bake.bake_normal_map(dense, out_path, size=tex_size)
        except Exception as exc:
            print(f"[game_ready] normal bake skipped ({exc})")

        print(f"[game_ready] wrote {out_path} ({len(low_faces)} faces)")
        return out_path
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[game_ready] failed ({exc}); returning dense GLB")
        return dense_textured_glb
