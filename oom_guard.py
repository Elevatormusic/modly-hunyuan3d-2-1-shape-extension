"""
Turn a CUDA out-of-memory failure in the paint pass into advice.

The texture pass is the memory peak of the whole extension, and when it runs
out the user gets a raw CUDA traceback from somewhere deep inside the VAE
decode — no indication of which setting caused it or which one to change.
This module classifies the failure and builds a message naming the specific
knobs, ordered by how much each one actually saves.

Torch-free and pure, so it stays unit-testable and can be called from an
except block whose CUDA context may already be dead.
"""
from __future__ import annotations

# Savings are the measured demand modifiers from capacity.py, restated here as
# user-facing numbers. Keep them in step with _TEX_768_DELTA / _PER_VIEW.
_VIEW_RES_SAVING_GB = 14
_PER_VIEW_SAVING_GB = 0.65


def is_cuda_oom(exc: BaseException) -> bool:
    """True if `exc` is a GPU out-of-memory failure.

    Both flavours qualify: the caching allocator's "CUDA out of memory. Tried
    to allocate ..." and the driver's "CUDA error: out of memory".

    Host RAM exhaustion is excluded by type, not just by wording. Requiring
    "cuda" in the text is not enough on its own: a MemoryError carrying a
    message that happens to mention CUDA would match and send the user after
    VRAM for a system-memory problem.

    Only the exception itself is inspected, never __cause__ or __context__.
    Walking the chain would catch an OOM re-raised inside a wrapper, but
    __context__ is set for any exception raised while handling another, so an
    unrelated failure that merely followed an OOM would be misreported as one
    — and a false positive here tells the user to restart and re-download.
    Nothing in the paint path wraps the OOM today; revisit only if that changes.
    """
    if isinstance(exc, MemoryError):
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text and "cuda" in text


def context_lost(exc: BaseException) -> bool:
    """True if the CUDA context is likely unusable for the rest of the process.

    "CUDA error: out of memory" comes from a failed driver call rather than
    the caching allocator, and it usually poisons the context: every later
    CUDA call in the same process fails too, including empty_cache(). The
    allocator's own OutOfMemoryError is recoverable and does not qualify.
    """
    return "cuda error" in f"{exc}".lower()


def advice(*, tex_resolution=512, max_num_view=6, shared_on=False,
           tier="auto", lost_context=False, planner_warning=None) -> str:
    """Build the user-facing message for a paint-stage OOM.

    Steps are ordered by measured saving, so the first thing the user tries is
    the thing most likely to work.
    """
    lines = ["Ran out of GPU memory while painting textures."]
    if planner_warning:
        lines.append(str(planner_warning))

    steps = []
    try:
        tr = int(tex_resolution)
    except (TypeError, ValueError):
        tr = 512
    try:
        nv = int(max_num_view)
    except (TypeError, ValueError):
        nv = 6

    # Ordered so everything that preserves output quality comes first. Spilling
    # into system memory is the extension's whole premise on small cards, so the
    # levers that make the spill work lead; reducing quality is the last resort.
    if not shared_on:
        steps.append("Turn on Use shared GPU memory to borrow system RAM.")
    steps.append("Check NVIDIA Control Panel > Manage 3D Settings > "
                 "CUDA - Sysmem Fallback Policy. If it is set to 'Prefer No Sysmem "
                 "Fallback', the driver refuses to spill into system RAM and this "
                 "error is the result — set it back to Driver Default.")
    steps.append("Make sure the Windows page file is large and on a fast drive — "
                 "the spill lands there once system RAM fills.")
    steps.append("Close other GPU apps — browsers and other AI tools hold VRAM that "
                 "cannot be spilled.")
    if str(tier) == "standard":
        steps.append("Set Texture memory to Reduced — same quality, about 7 GB less VRAM.")
    if nv > 6:
        steps.append(f"Reduce Views from {nv} to 6 (about "
                     f"{_PER_VIEW_SAVING_GB:.2f} GB per view above 6). This lowers "
                     f"quality, so try it only after the steps above.")
    if tr >= 768:
        steps.append(f"Last resort, and it does cost quality: lower View resolution "
                     f"from {tr} to 512, which frees about {_VIEW_RES_SAVING_GB} GB.")

    lines.append("Try, in order:")
    lines.extend(f"  {i}. {s}" for i, s in enumerate(steps, 1))

    if lost_context:
        lines.append("Restart Modly before the next run — this particular error leaves "
                     "the GPU context unusable, so retrying without a restart will fail "
                     "again regardless of settings.")
    return "\n".join(lines)
