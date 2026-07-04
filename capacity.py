"""VRAM-aware planning + normal-bake gating for the Hunyuan3D-2.1 extension.

Pure logic (no torch import) so it is trivially testable; generator.py feeds it the
live VRAM numbers from torch.cuda.mem_get_info().
"""
from __future__ import annotations

# Rough peak-VRAM (GB) for the shape stage by octree resolution (RTX-class, fp16).
_SHAPE_VRAM = {256: 7.0, 384: 10.0, 512: 14.0}
# Rough peak-VRAM (GB) for the paint stage (multiview diffusion + upscale + bake).
# Measured ~23.6 GB at 512/6 views on a 3090, so treat textures as ~22 GB+.
_PAINT_BASE = 22.0


def should_bake(bake_normal_map: bool, mesh_mode: str) -> bool:
    """Normal baking assumes the low-poly base is a faithful lower-res of the dense
    mesh. BPT REGENERATES the surface, so a high->low bake misregisters and produces
    blotches — skip the bake for BPT (ship clean-but-flat instead)."""
    return bool(bake_normal_map) and mesh_mode != "bpt"


def shape_vram(octree_res: int) -> float:
    return _SHAPE_VRAM.get(int(octree_res), 10.0)


def paint_vram(tex_resolution: int, max_num_view: int) -> float:
    extra = (3.0 if int(tex_resolution) >= 768 else 0.0) + max(0, int(max_num_view) - 6) * 0.6
    return _PAINT_BASE + extra


def vram_plan(free_gb, total_gb, enable_texture, octree_res,
              tex_resolution=512, max_num_view=6):
    """Plan a run against available VRAM.

    Returns (octree_res_to_use, warning_message_or_None). Caps octree resolution so
    the shape stage fits in free VRAM (clear OOM prevention). Warns — without
    disabling — if the texture stage is likely to exceed free VRAM, since that is the
    user's call (close other apps / lower texture settings / turn textures off).
    """
    msgs = []
    octree = int(octree_res)

    if shape_vram(octree) > free_gb:
        octree = 256
        for r in (512, 384, 256):
            if r <= int(octree_res) and _SHAPE_VRAM[r] <= free_gb:
                octree = r
                break
        if octree != int(octree_res):
            msgs.append(
                f"Low VRAM (~{free_gb:.0f} GB free): capped Mesh Resolution to {octree} "
                f"to avoid running out of memory on the shape stage.")

    if enable_texture:
        need = paint_vram(tex_resolution, max_num_view)
        if need > free_gb:
            msgs.append(
                f"Textures need ~{need:.0f} GB but only ~{free_gb:.0f} GB is free — close "
                f"other GPU apps, lower Texture view resolution/views, or turn textures "
                f"off to avoid an out-of-memory error.")

    return octree, (" ".join(msgs) if msgs else None)
