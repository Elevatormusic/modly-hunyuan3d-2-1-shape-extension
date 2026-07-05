# mr_export.py
"""Assemble the paint OBJ into a GLB that carries the baked metallic-roughness
atlas (glTF packs roughness in G, metallic in B). Replaces the albedo-only
convert_obj_to_glb. Pure trimesh/PIL/numpy; no GPU."""
from __future__ import annotations
import os
import numpy as np
from PIL import Image


def pack_metallic_roughness(metallic_img, roughness_img):
    m = np.asarray(metallic_img.convert("L"))
    r = np.asarray(roughness_img.convert("L"))
    h, w = m.shape
    out = np.zeros((h, w, 3), np.uint8)
    out[..., 0] = 255      # R unused
    out[..., 1] = r        # G = roughness
    out[..., 2] = m        # B = metallic
    return Image.fromarray(out, "RGB")


def build_glb_with_mr(obj_path, glb_path):
    import trimesh
    from trimesh.visual.material import PBRMaterial
    scene = trimesh.load(obj_path, process=False)
    base = os.path.splitext(obj_path)[0]
    mpath, rpath = base + "_metallic.png", base + "_roughness.png"
    mr = None
    if os.path.exists(mpath) and os.path.exists(rpath):
        mr = pack_metallic_roughness(Image.open(mpath), Image.open(rpath))
    geoms = scene.geometry.values() if hasattr(scene, "geometry") else [scene]
    for g in geoms:
        v = getattr(g, "visual", None)
        mat = getattr(v, "material", None)
        img = (getattr(mat, "baseColorTexture", None)
               or getattr(mat, "image", None)) if mat else None
        uv = getattr(v, "uv", None)
        if img is None or uv is None:
            continue
        kw = dict(baseColorTexture=img, baseColorFactor=[255, 255, 255, 255])
        if mr is not None:
            kw.update(metallicRoughnessTexture=mr, metallicFactor=1.0,
                      roughnessFactor=1.0)
        else:
            kw.update(metallicFactor=0.0, roughnessFactor=1.0)
        g.visual = trimesh.visual.TextureVisuals(
            uv=uv, material=PBRMaterial(**kw))
    scene.export(glb_path)
    return True
