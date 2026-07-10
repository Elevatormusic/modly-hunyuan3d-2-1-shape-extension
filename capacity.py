"""VRAM-aware planning + normal-bake gating for the Hunyuan3D-2.1 extension.

Pure logic (no torch import) so it is trivially testable; generator.py feeds it the
live VRAM numbers from torch.cuda.mem_get_info().
"""
from __future__ import annotations
from collections import namedtuple

# Rough peak-VRAM (GB) for the shape stage by octree resolution (RTX-class, fp16).
_SHAPE_VRAM = {256: 7.0, 384: 10.0, 512: 14.0}

# --- Measured paint-stage peaks (RTX 3090, on-device 2026-07-09, tree input, seed 42).
# Both numbers are RESERVED GB (the true footprint torch holds), at the STOCK sizes
# (render 2048 / texture 4096 / sr_chunk 4), tex_res 512, 6 views:
#   * full-GPU (Standard):     20.4 reserved (13.3 allocated), 111 s
#   * component-staged path:   13.0 reserved ( 9.1 allocated), 116 s  (~4-5% slower)
# Quality is identical (view-space diff 2.24/255 = run-to-run noise) — the staged path
# runs the same weights through the same math, only moving idle components to CPU RAM
# at stage boundaries. See private/specs/2026-07-09-measured-tiers-phase-offload-design.md.
_PEAK_STOCK = 20.4      # reserved GB, full-GPU paint at stock sizes, 512/6v
_PEAK_OFFLOAD = 13.0    # reserved GB, component-staged paint at stock sizes, 512/6v

# Demand modifiers (added to either peak). Measured: 512/9v cost +1.9 GB over 512/6v
# (~0.65 GB per view above 6, linear-ish). tex_res 768 cost +14.3 GB at full-GPU.
# 768 + staging is PROVISIONAL (unmeasured): assume the same +14.3 delta — at ~27 GB
# it exceeds every consumer card and gates to the shared-RAM advice anyway.
_PER_VIEW = 0.65        # GB per view above 6
_TEX_768_DELTA = 14.3   # GB added for tex_resolution 768 (provisional for the staged path)

# Safety headroom (GB) added on top of the measured reserved peak when deciding whether
# a tier fits, and printed in the "need ~N GB" warnings so the advice is honest. Small
# because the peaks are already RESERVED (true footprint), not allocated.
_TEX_MARGIN = 1.5


def should_bake(bake_normal_map: bool, mesh_mode: str) -> bool:
    """Normal baking assumes the low-poly base is a faithful lower-res of the dense
    mesh. BPT REGENERATES the surface, so a high->low bake misregisters and produces
    blotches — skip the bake for BPT (ship clean-but-flat instead)."""
    return bool(bake_normal_map) and mesh_mode != "bpt"


def shape_vram(octree_res: int) -> float:
    return _SHAPE_VRAM.get(int(octree_res), 10.0)


def _extra_demand(tex_resolution, max_num_view) -> float:
    """Extra reserved GB beyond the base peak for the two quality knobs that cost VRAM:
    a higher per-view render resolution and more views. Shared by paint_vram() and
    plan_texture_memory() so the preflight estimate and the planner agree."""
    try:
        tr = int(tex_resolution)
    except (TypeError, ValueError):
        tr = 512
    try:
        nv = int(max_num_view)
    except (TypeError, ValueError):
        nv = 6
    return (_TEX_768_DELTA if tr >= 768 else 0.0) + _PER_VIEW * max(0, nv - 6)


def paint_vram(tex_resolution: int, max_num_view: int) -> float:
    """Full-GPU (Standard) paint-stage peak in reserved GB. This is the higher of the
    two paths; the reduced-VRAM (staged) path needs ~7 GB less at the same quality."""
    return _PEAK_STOCK + _extra_demand(tex_resolution, max_num_view)


TexturePlan = namedtuple(
    "TexturePlan", "render_size texture_size sr_chunk tier offload offload_hint warning")

# Every tier now paints at STOCK sizes (render 2048 / texture 4096 / sr_chunk 4) — the
# reduced-quality 1536/2048 tier was retired (it saved only ~1.1 GB while softening the
# texture). The ONLY difference between Standard and Reduced is `offload`: Reduced stages
# whole components (VAE / diffusion UNet / DINO) between CPU and GPU at stage boundaries,
# cutting the peak from ~20.4 to ~13.0 GB reserved at identical quality. render_size is the
# per-view BAKE resolution; STOCK Hunyuan bakes at 2048.
_STOCK_SIZES = (2048, 4096, 4)   # (render_size, texture_size, sr_chunk)
_TEX_TIERS = {
    "auto":     _STOCK_SIZES,
    "standard": _STOCK_SIZES,
    "reduced":  _STOCK_SIZES,
}

# Manifest offers auto/standard/reduced, but old saved workflows carry the retired ids —
# map them forward: the three full-quality ids collapse to Auto, the old low tier becomes
# the reduced-VRAM path, and anything unrecognised falls back to Auto (the safe default).
_LEGACY_TIER = {
    "balanced": "auto",
    "high":     "auto",
    "max":      "auto",
    "low":      "reduced",
}


def _resolve_requested(tier) -> str:
    """Normalise the requested tier id to one of auto/standard/reduced (unknown -> auto)."""
    t = str(tier).lower()
    if t in ("auto", "standard", "reduced"):
        return t
    return _LEGACY_TIER.get(t, "auto")


def plan_texture_memory(free_gb, tier_ceiling="auto",
                        tex_resolution=512, max_num_view=6, extra_budget_gb=0.0):
    """Pick a paint-memory plan that fits the VRAM budget.

    budget = free VRAM + extra_budget_gb (the latter is the opt-in shared-system-RAM
    allowance). Every tier paints at stock quality; the choice is WHERE the idle
    components wait:

      * Auto (default): full-GPU (Standard) when it fits the budget with headroom,
        otherwise the component-staged (Reduced) path — same quality, ~7 GB less VRAM.
      * Standard: forced full-GPU (offload off).
      * Reduced: forced component staging (offload on).

    Legacy ids (balanced/high/max -> auto, low -> reduced, unknown -> auto) keep old
    saved workflows working. Pure; generator.py feeds it torch.cuda.mem_get_info()
    (free) + capacity.shared_ram_allowance() (extra). Never raises."""
    try:
        free = float(free_gb)
    except (TypeError, ValueError):
        free = 0.0
    try:
        budget = free + max(0.0, float(extra_budget_gb))
    except (TypeError, ValueError):
        budget = free

    requested = _resolve_requested(tier_ceiling)
    extra = _extra_demand(tex_resolution, max_num_view)
    need_standard = _PEAK_STOCK + extra + _TEX_MARGIN

    if requested == "standard":
        chosen, offload = "standard", False
    elif requested == "reduced":
        chosen, offload = "reduced", True
    else:                                   # auto: full-GPU iff it fits with headroom
        if budget >= need_standard:
            chosen, offload = "standard", False
        else:
            chosen, offload = "reduced", True

    peak = (_PEAK_OFFLOAD if offload else _PEAK_STOCK) + extra
    offload_hint = False
    warning = None
    if peak > budget:               # won't fit even with the shared budget -> best effort
        offload_hint = True
        need = peak + _TEX_MARGIN    # print peak+margin so the advice leaves real headroom
        warning = (f"Textures need ~{need:.0f} GB but only ~{budget:.0f} GB is available — "
                   f"close other GPU apps, turn on Use shared GPU memory, or turn textures "
                   f"off to avoid a slowdown or out-of-memory error.")
    elif peak > free:               # fits only via the shared budget -> pages over PCIe
        paged = peak - free
        label = "Reduced" if offload else "Standard"
        warning = (f"{label} tier will page ~{paged:.0f} GB to system RAM over PCIe "
                   f"(shared GPU memory) — expect it to run several times slower.")

    r, t, c = _TEX_TIERS[chosen]
    return TexturePlan(r, t, c, chosen, offload, offload_hint, warning)


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
    user's call (close other apps / lower texture settings / turn textures off). When the
    full-GPU peak won't fit but the reduced-VRAM path would, the message reassures that
    Auto will use the reduced-VRAM path rather than scaring the user.
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
        need = paint_vram(tex_resolution, max_num_view)           # full-GPU (Standard) peak
        if need > free_gb:
            reduced_need = _PEAK_OFFLOAD + _extra_demand(tex_resolution, max_num_view)
            if reduced_need <= free_gb:
                msgs.append(
                    f"Textures need ~{need:.0f} GB at full quality, more than the ~{free_gb:.0f} GB "
                    f"free — Auto will use the reduced-VRAM path (~{reduced_need:.0f} GB) at the "
                    f"same quality.")
            else:
                msgs.append(
                    f"Textures need ~{need:.0f} GB but only ~{free_gb:.0f} GB is free — close "
                    f"other GPU apps, turn on Use shared GPU memory, lower Texture view "
                    f"resolution/views, or turn textures off to avoid an out-of-memory error.")

    return octree, (" ".join(msgs) if msgs else None)
