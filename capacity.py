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
    extra = (8.0 if int(tex_resolution) >= 768 else 0.0) + max(0, int(max_num_view) - 6) * 0.6
    return _PAINT_BASE + extra


TexturePlan = namedtuple(
    "TexturePlan", "render_size texture_size sr_chunk tier offload_hint warning")

# tier -> (render_size, texture_size, sr_chunk), low -> high -> max
# render_size is the per-view BAKE resolution (MeshRender.back_project samples each
# painted view into the UV atlas at this resolution) — it is the single biggest driver
# of texture sharpness. STOCK Hunyuan bakes at 2048; the old tiers capped it to
# 1024-1536 to save VRAM and that is what made our textures muddy/blotchy vs stock.
# free-before-bake frees the ~15-18 GB diffusion net BEFORE the bake, so a full
# 2048 bake + 4096 atlas fits comfortably (~13-16 GB measured; Tencent quote 12 GB for
# the whole pipeline). So the default now delivers STOCK quality; 'low' is the single
# reduced tier for genuinely small (<12 GB) cards. Seeds recalibrated on-device.
_TEX_TIERS = {
    "low":      (1536, 2048, 2),   # reduced fallback for tight VRAM
    "balanced": (2048, 4096, 4),   # STOCK quality (render 2048 / texture 4096) — default
    "high":     (2048, 4096, 4),   # stock
    "max":      (2048, 4096, 4),   # stock (2048 is the paint's native bake res — the ceiling)
}
_TIER_ORDER = ["low", "balanced", "high", "max"]
# Seed peak-VRAM (GB) per tier = the paint-stage peak with free-before-bake (the diffusion
# net is deleted before SR/bake, so texture_size is nearly free and the tiers barely differ
# — Low/Balanced/Max measured 12.9/13.0/13.4 GB allocated, ~19-20 GB reserved, at render
# 1024-1536 / tex_res 512 on-device 2026-07-04). PROVISIONAL bumps for the new render=2048
# tiers, refined by an on-device run (the paint-peak telemetry logs allocated/reserved). The
# real memory driver is tex_res (512->768 adds ~7 GB), handled by `extra` in plan_texture_memory.
# Measured on-device 2026-07-04 (render 2048 / texture 4096, penguin, free-before-bake):
# tex_res 512 -> 13.4 GB allocated / 20.4 GB reserved; tex_res 768 -> 21.1 / 35.4 (shared).
# Seed = allocated base at tex_res 512 (render_size barely moves it — 1536 and 2048 both 13.4);
# the +6 margin covers the reserved-minus-allocated gap so the budget check ~= true footprint.
_TEX_PEAK = {"low": 12.5, "balanced": 13.5, "high": 13.5, "max": 13.5}
_TEX_MARGIN = 6.0   # reserved runs ~7 GB above allocated (torch caching); check ~= reserved footprint


def _tex_tier_index(tier):
    t = str(tier).lower()
    return _TIER_ORDER.index(t) if t in _TIER_ORDER else _TIER_ORDER.index("balanced")


def plan_texture_memory(free_gb, tier_ceiling="balanced",
                        tex_resolution=512, max_num_view=6, extra_budget_gb=0.0):
    """Pick (render_size, texture_size, sr_chunk) that fit the memory BUDGET without
    exceeding the user's `tier_ceiling`. budget = free VRAM + extra_budget_gb (the latter is
    the opt-in shared-system-RAM allowance). Pure; generator.py feeds it
    torch.cuda.mem_get_info() (free) + capacity.shared_ram_allowance() (extra). When a tier
    fits only via the shared budget (its peak exceeds free VRAM) the plan carries a loud
    paging warning. Never raises."""
    try:
        free = float(free_gb)
    except (TypeError, ValueError):
        free = 0.0
    try:
        budget = free + max(0.0, float(extra_budget_gb))
    except (TypeError, ValueError):
        budget = free
    ceiling_i = _tex_tier_index(tier_ceiling)
    # same demand shape as paint_vram(): the two diffusion quality knobs cost extra.
    extra = (8.0 if int(tex_resolution) >= 768 else 0.0) + max(0, int(max_num_view) - 6) * 0.6

    chosen = None
    for i in range(ceiling_i, -1, -1):
        tier = _TIER_ORDER[i]
        if _TEX_PEAK[tier] + extra + _TEX_MARGIN <= budget:
            chosen = tier
            break

    offload_hint = False
    warning = None
    if chosen is None:              # even Low won't fit the budget — best effort + warn
        chosen = "low"
        offload_hint = True
        # include _TEX_MARGIN so the printed 'need' matches the fit test above
        # (peak+extra+margin); otherwise it reads "need ~13 but ~15 available",
        # a false statement on ~16 GB cards (Fix 8).
        need = _TEX_PEAK["low"] + extra + _TEX_MARGIN
        warning = (f"Textures need ~{need:.0f} GB but only ~{budget:.0f} GB is available — "
                   f"close other GPU apps, turn on Use shared GPU memory, or turn textures "
                   f"off to avoid a slowdown or out-of-memory error.")
    else:
        peak = _TEX_PEAK[chosen] + extra
        if peak > free:             # fits only via shared budget -> pages to RAM over PCIe
            paged = peak - free
            warning = (f"{chosen.capitalize()} tier will page ~{paged:.0f} GB to system RAM "
                       f"over PCIe (shared GPU memory) — expect it to run several times "
                       f"slower (tens of minutes).")

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
