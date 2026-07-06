"""Pure-PIL QA debug sheet for a textured GLB (no matplotlib / pygltflib).

write_debug_sheet(glb_path, obj_path, out_png, *, input_image_path=None) -> str|None
renders a labeled grid: input, albedo, metallic, roughness, normal, a UV
wireframe, and a stats text panel. Missing maps degrade to a placeholder;
any hard failure returns None (caller treats it as non-fatal).
"""
import os
import numpy as np
from PIL import Image, ImageDraw

_CELL = 384
_PAD = 12
_LABEL_H = 20
_COLS = 4


def _find_map(base, suffix):
    for ext in (".jpg", ".png", ".jpeg"):
        p = base + suffix + ext
        if os.path.exists(p):
            return p
    return None


def _albedo_path(obj_path):
    base = obj_path[:-4] if obj_path.lower().endswith(".obj") else obj_path
    mtl = base + ".mtl"
    if os.path.exists(mtl):
        try:
            with open(mtl) as fh:
                for line in fh:
                    s = line.strip()
                    if s.lower().startswith("map_kd"):
                        name = s.split(None, 1)[1].strip()
                        cand = os.path.join(os.path.dirname(base) or ".", name)
                        if os.path.exists(cand):
                            return cand
        except Exception:
            pass
    for ext in (".jpg", ".png", ".jpeg"):
        if os.path.exists(base + ext):
            return base + ext
    return None


def _thumb(path, size):
    try:
        with Image.open(path) as im0:
            im = im0.convert("RGB")
        im.thumbnail((size, size))
        return im
    except Exception:
        return None


def _placeholder(size, text="-"):
    im = Image.new("RGB", (size, size), (40, 40, 40))
    d = ImageDraw.Draw(im)
    d.text((size // 2 - 4, size // 2 - 6), text, fill=(160, 160, 160))
    return im


def _load_scene(glb_path):
    import trimesh
    return trimesh.load(glb_path, process=False)


def _geoms(sc):
    import trimesh
    if isinstance(sc, trimesh.Scene):
        return list(sc.geometry.values())
    return [sc]


def _uv_panel(glb_path, size):
    im = Image.new("RGB", (size, size), (20, 20, 20))
    d = ImageDraw.Draw(im)
    try:
        for g in _geoms(_load_scene(glb_path)):
            uv = getattr(getattr(g, "visual", None), "uv", None)
            if uv is None or len(uv) == 0:
                continue
            uv = np.asarray(uv, float)
            F = np.asarray(g.faces)
            px = np.clip(uv[:, 0], 0, 1) * (size - 1)
            py = (1 - np.clip(uv[:, 1], 0, 1)) * (size - 1)
            for tri in F:
                pts = [(px[tri[k]], py[tri[k]]) for k in (0, 1, 2)]
                d.line(pts + [pts[0]], fill=(80, 200, 120), width=1)
    except Exception:
        pass
    return im


def _stats_lines(glb_path, obj_path):
    lines = []
    try:
        geoms = _geoms(_load_scene(glb_path))
        lines.append(f"geometries : {len(geoms)}")
        lines.append(f"faces      : {sum(len(g.faces) for g in geoms)}")
        lines.append(f"vertices   : {sum(len(g.vertices) for g in geoms)}")
        g0 = geoms[0] if geoms else None
        mat = getattr(getattr(g0, "visual", None), "material", None)
        if mat is not None:
            lines.append(f"metallicF  : {getattr(mat, 'metallicFactor', None)}")
            lines.append(f"roughnessF : {getattr(mat, 'roughnessFactor', None)}")
    except Exception as exc:
        lines.append(f"(stats error: {exc})")
    base = obj_path[:-4] if obj_path.lower().endswith(".obj") else obj_path
    for label, path in (("albedo", _albedo_path(obj_path)),
                        ("metallic", _find_map(base, "_metallic")),
                        ("roughness", _find_map(base, "_roughness")),
                        ("normal", _find_map(base, "_normal"))):
        if path and os.path.exists(path):
            try:
                with Image.open(path) as _im:
                    w, h = _im.size
                lines.append(f"{label:10}: {w}x{h}")
            except Exception:
                lines.append(f"{label:10}: (unreadable)")
        else:
            lines.append(f"{label:10}: -")
    try:
        lines.append(f"glb bytes  : {os.path.getsize(glb_path)}")
    except Exception:
        pass
    return lines


def _text_panel(lines, size):
    im = Image.new("RGB", (size, size), (16, 16, 16))
    d = ImageDraw.Draw(im)
    y = 8
    for ln in lines:
        d.text((8, y), ln, fill=(220, 220, 220))
        y += 15
    return im


def write_debug_sheet(glb_path, obj_path, out_png, *, input_image_path=None):
    try:
        base = obj_path[:-4] if obj_path.lower().endswith(".obj") else obj_path
        cells = []
        if input_image_path and os.path.exists(input_image_path):
            cells.append(("input", _thumb(input_image_path, _CELL)))
        cells.append(("albedo", _thumb(_albedo_path(obj_path) or "", _CELL)))
        cells.append(("metallic", _thumb(_find_map(base, "_metallic") or "", _CELL)))
        cells.append(("roughness", _thumb(_find_map(base, "_roughness") or "", _CELL)))
        cells.append(("normal", _thumb(_find_map(base, "_normal") or "", _CELL)))
        cells.append(("uv layout", _uv_panel(glb_path, _CELL)))
        cells.append(("stats", _text_panel(_stats_lines(glb_path, obj_path), _CELL)))

        cols = _COLS
        rows = (len(cells) + cols - 1) // cols
        cw = _CELL + _PAD
        ch = _CELL + _LABEL_H + _PAD
        canvas = Image.new("RGB", (cols * cw + _PAD, rows * ch + _PAD), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        for i, (label, img) in enumerate(cells):
            r, c = divmod(i, cols)
            x = _PAD + c * cw
            y = _PAD + r * ch
            draw.text((x, y), label, fill=(230, 230, 230))
            canvas.paste(img if img is not None else _placeholder(_CELL), (x, y + _LABEL_H))
        canvas.save(out_png)
        return out_png
    except Exception as exc:
        print(f"[debug_sheet] failed ({exc})")
        return None
