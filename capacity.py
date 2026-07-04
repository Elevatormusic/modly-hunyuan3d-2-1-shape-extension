"""VRAM-aware planning + normal-bake gating for the Hunyuan3D-2.1 extension.

Pure logic (no torch import) so it is trivially testable; generator.py feeds it the
live VRAM numbers from torch.cuda.mem_get_info().
"""
from __future__ import annotations
from collections import namedtuple

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


TexturePlan = namedtuple(
    "TexturePlan", "render_size texture_size sr_chunk tier offload_hint warning")

# tier -> (render_size, texture_size, sr_chunk), low -> high -> max
_TEX_TIERS = {
    "low":      (1024, 1024, 1),
    "balanced": (1024, 2048, 2),
    "high":     (1536, 2048, 4),
    "max":      (1536, 4096, 4),
}
_TIER_ORDER = ["low", "balanced", "high", "max"]
# Seed peak-VRAM (GB) per tier on a 24 GB card, shape freed AND diffusion freed before
# the bake (free-before-bake, see the textureGenPipeline patch). Because texture_size lives
# in the (post-free) bake stage, Max ~= High in VRAM. Engineering seeds pending on-device
# calibration (see private/runbooks/2026-07-03-texture-memory-calibration.md). Anchored to
# the measured ~22-23.6 GB at render=2048/tex=2048/chunk=4, minus per-lever deltas (render
# 2048->1536 ~-0.7, ->1024 ~-1.5; chunk 4->2 ~-1.5, ->1 ~-2.0; texture 2048->1024 ~-0.4).
_TEX_PEAK = {"high": 21.5, "balanced": 19.5, "low": 18.5, "max": 21.5}
_TEX_MARGIN = 2.0   # WDDM/fragmentation headroom


def _tex_tier_index(tier):
    t = str(tier).lower()
    return _TIER_ORDER.index(t) if t in _TIER_ORDER else _TIER_ORDER.index("balanced")


def plan_texture_memory(free_gb, tier_ceiling="balanced",
                        tex_resolution=512, max_num_view=6):
    """Pick (render_size, texture_size, sr_chunk) that fit `free_gb` of VRAM without
    exceeding the user's `tier_ceiling`. Pure; generator.py feeds it
    torch.cuda.mem_get_info() taken AFTER the shape model is freed. Never raises."""
    try:
        free = float(free_gb)
    except (TypeError, ValueError):
        free = 0.0
    ceiling_i = _tex_tier_index(tier_ceiling)
    # same demand shape as paint_vram(): the two diffusion quality knobs cost extra.
    extra = (3.0 if int(tex_resolution) >= 768 else 0.0) + max(0, int(max_num_view) - 6) * 0.6

    chosen = None
    for i in range(ceiling_i, -1, -1):
        tier = _TIER_ORDER[i]
        if _TEX_PEAK[tier] + extra + _TEX_MARGIN <= free:
            chosen = tier
            break

    offload_hint = False
    warning = None
    if chosen is None:              # even Low won't fit — best effort + warn (no silent offload)
        chosen = "low"
        offload_hint = True
        need = _TEX_PEAK["low"] + extra
        warning = (f"Textures need ~{need:.0f} GB but only ~{free:.0f} GB VRAM is free — "
                   f"close other GPU apps, turn on Low VRAM mode, or turn textures off "
                   f"to avoid a slowdown or out-of-memory error.")

    r, t, c = _TEX_TIERS[chosen]
    return TexturePlan(r, t, c, chosen, offload_hint, warning)


def apply_texture_plan(conf, plan):
    """Set the paint config's memory knobs from a TexturePlan. Returns conf.

    render_size/texture_size are read by MeshRender at pipeline construction. sr_chunk is
    surfaced here for inspection/completeness, but the LIVE channel eb_accel reads is the
    EB_SR_CHUNK env var (set in generator._run_texture) — the vendored SR call site is
    already patched and doesn't pass conf.
    """
    conf.render_size = plan.render_size
    conf.texture_size = plan.texture_size
    conf.sr_chunk = plan.sr_chunk
    return conf


def shared_ram_allowance(total_gb, available_gb, headroom=12.0):
    """GB of system RAM safe to lend the GPU as shared memory when the user opts into
    'Use shared GPU memory'. Bounded by BOTH the Windows shared-GPU ceiling (50% of total
    RAM) AND leaving `headroom` GB of currently-available RAM free for the OS + the paint
    process's own growth + the paged GPU pages (which draw from the same pool). Pure."""
    try:
        alw = min(0.5 * float(total_gb), float(available_gb) - float(headroom))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, alw)


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
