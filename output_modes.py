"""Output-mode preset layer.

`resolve_params` overlays a fixed knob bundle for a non-Custom mode; Custom (or
an unknown mode) passes params through unchanged. UI-param keys are written only
if the node actually exposes them (so `saturation` is set only when the vibrance
knob exists); internal directive keys (leading underscore) are always written and
are consumed by the generator directly. Pure, torch-free.
"""

_BUNDLES = {
    "render_max": {
        "octree_resolution": 512, "enable_texture": 1, "texture_resolution": 768,
        "max_num_view": 8, "mesh_mode": "regular", "bake_normal_map": 1,
        "seam_fix": 1, "saturation": "subtle", "_face_target": 100000,
    },
    "render_balanced": {
        "octree_resolution": 384, "enable_texture": 1, "texture_resolution": 512,
        "max_num_view": 6, "mesh_mode": "regular", "bake_normal_map": 1,
        "seam_fix": 1, "saturation": "subtle", "_face_target": 100000,
    },
    "game_ready": {
        "octree_resolution": 512, "enable_texture": 1, "texture_resolution": 768,
        "max_num_view": 8, "mesh_mode": "regular", "bake_normal_map": 1, "seam_fix": 1,
        "saturation": "subtle", "_face_target": 100000, "_game_ready": True,
    },
}


def resolve_params(output_mode, raw, schema_ids):
    """Return a copy of `raw` with the mode's bundle overlaid.

    - Internal directive keys (starting with '_') are always written.
    - UI-param keys are written only if present in `schema_ids`.
    - Custom / unknown mode → `raw` returned unchanged.
    """
    bundle = _BUNDLES.get(str(output_mode))
    if not bundle:
        return raw
    ids = schema_ids or set()
    out = dict(raw)
    for k, v in bundle.items():
        if k.startswith("_") or k in ids:
            out[k] = v
    return out
