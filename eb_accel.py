"""GPU-acceleration helpers injected into the Hunyuan3D-2.1 paint pipeline by the
Modly extension (see generator.py::_patch_gpu_accel, which copies this file into
`_hy3dpaint_src/` and rewires a few call sites to use it).

Design rules:
  * Pure-torch, on the venv's existing torch/CUDA — NO new native dependency.
  * Same-or-better quality (no resolution/quality shortcuts).
  * Every entry point falls back to the ORIGINAL CPU path on ANY error, so a bug
    here can only make things slower, never wrong or broken.
"""
import numpy as np


# --------------------------------------------------------------------------- #
# 0. Progress hook. generator._run_texture registers a callback here before the
# paint pipeline runs; the vendored multiview diffusion reports per denoise step
# and the post-diffusion stages report milestones, driving the REAL progress bar
# in place of the old cosmetic creep. report() swallows every error so progress
# can never break the paint compute; if no hook is set it is a no-op.
# --------------------------------------------------------------------------- #
_PROGRESS_HOOK = None


def set_progress_hook(fn):
    """Register fn(percent:int, label:str), or clear it with None."""
    global _PROGRESS_HOOK
    _PROGRESS_HOOK = fn


def report(percent, label):
    fn = _PROGRESS_HOOK
    if fn is None:
        return
    try:
        fn(int(percent), str(label))
    except Exception:
        pass


def _resolve_sr_chunk(chunk):
    """Resolve the SR batch size: explicit arg wins, else the EB_SR_CHUNK env var
    (set by the extension per texture-memory tier), else 4. Always >= 1."""
    if chunk is not None:
        try:
            return max(1, int(chunk))
        except (TypeError, ValueError):
            return 4
    import os
    try:
        return max(1, int(os.environ.get("EB_SR_CHUNK", "4")))
    except (TypeError, ValueError):
        return 4


# --------------------------------------------------------------------------- #
# 1. Batched RealESRGAN super-resolution (replaces the per-image SR loop)
# --------------------------------------------------------------------------- #
def super_resolve_batch(sr_net, pil_images, out_size, chunk=None):
    """Super-resolve a list of same-size PIL images in batched GPU forwards, then
    resize to (out_size, out_size) on the GPU, returning a list of PIL images
    (same interface the pipeline's per-image loop produced).

    Replaces both the per-image `super_model(img)` loop AND the follow-up resize
    loop (which upstream buggily bounded by `len(enhance_images)`==2, resizing
    only the first two views). RRDBNet has no BatchNorm, so batching is
    per-image identical; result is same-or-better (float path, one requantize).

    Falls back to the original per-image path on any error.
    """
    try:
        import torch
        import torch.nn.functional as F
        from PIL import Image

        if not pil_images:
            return []
        up = sr_net.upsampler
        model, device, scale = up.model, up.device, up.scale
        half = bool(getattr(up, "half", False))

        arrs = [np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0 for im in pil_images]
        H, W = arrs[0].shape[:2]
        if any(a.shape[:2] != (H, W) for a in arrs):
            raise ValueError("super_resolve_batch requires uniform image sizes")

        x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous().to(device)
        if half:
            x = x.half()
        # RealESRGANer.enhance reverses channels before the model (its
        # cv2.COLOR_BGR2RGB, ch 0<->2) and reverses them again on the output
        # ([[2,1,0]]). RRDBNet is not channel-swap-symmetric, so we MUST do both
        # to be per-image identical (otherwise R/B are subtly mis-processed).
        x = x[:, [2, 1, 0], :, :]
        # Pad H,W up to a multiple of `scale` (reflect), mirroring RealESRGANer's
        # mod-pad; cropped back off after. No-op when already a multiple.
        ph, pw = (scale - H % scale) % scale, (scale - W % scale) % scale
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")

        outs = []
        with torch.no_grad():
            i, cur = 0, _resolve_sr_chunk(chunk)
            while i < x.shape[0]:
                try:
                    outs.append(model(x[i:i + cur]).float())
                    i += cur
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if cur == 1:
                        raise
                    cur = max(1, cur // 2)   # VRAM guard: shrink and retry
        y = torch.cat(outs, 0).clamp_(0.0, 1.0)
        if ph or pw:
            y = y[:, :, : H * scale, : W * scale]
        y = y[:, [2, 1, 0], :, :]   # reverse channels back (enhance's [[2,1,0]])
        # Only resample if the SR output isn't already the target size — an
        # F.interpolate at scale 1 with antialias would needlessly low-pass
        # (blur) the result. antialias only helps when actually downsampling.
        out_size = int(out_size)
        if y.shape[-2] != out_size or y.shape[-1] != out_size:
            y = F.interpolate(y, size=(out_size, out_size), mode="bicubic",
                              align_corners=False, antialias=True).clamp_(0.0, 1.0)
        y = (y * 255.0 + 0.5).clamp_(0, 255).byte().permute(0, 2, 3, 1).contiguous().cpu().numpy()
        return [Image.fromarray(a) for a in y]
    except Exception as exc:  # noqa: BLE001 — deliberate: never fail the pipeline
        print(f"[eb_accel] super_resolve_batch fell back to per-image path: {exc}")
        return [sr_net(im).resize((int(out_size), int(out_size))) for im in pil_images]


# --------------------------------------------------------------------------- #
# 2. GPU push-pull hole fill (replaces cv2.INPAINT_NS; keeps meshVerticeInpaint)
# --------------------------------------------------------------------------- #
def inpaint_fill_gpu(texture, mask):
    """Fill the residual holes of a UV texture with a GPU push-pull (mip-pyramid)
    fill. This replaces ONLY the cv2 Navier-Stokes pass — the mesh-aware
    `meshVerticeInpaint` still runs first, so seams are untouched and only the
    same never-sampled UV gutters are filled here.

    Args:
        texture: float [H,W,C] in 0..1 (as passed to cv2 upstream).
        mask:    uint8 [H,W], >=128 == valid, <128 == hole (upstream's `mask`;
                 cv2 inpainted where `255-mask` != 0).
    Returns:
        uint8 [H,W,C] in 0..255 — matching cv2.inpaint's output contract.
    Falls back to cv2.INPAINT_NS on any error.
    """
    try:
        import torch
        import torch.nn.functional as F

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        t = torch.as_tensor(np.ascontiguousarray(texture), dtype=torch.float32, device=dev)
        squeeze_c = t.ndim == 2
        if squeeze_c:
            t = t[..., None]
        m = torch.as_tensor(np.ascontiguousarray(mask), dtype=torch.float32, device=dev)
        valid = (m >= 128).float()                                  # [H,W]

        color = (t * valid[..., None]).permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
        w = valid.unsqueeze(0).unsqueeze(0)                           # [1,1,H,W]

        # PULL: weighted-average pyramid down to 1x1 (guarantees full coverage
        # as long as ANY texel is valid).
        pyr = [(color, w)]
        c, ww = color, w
        while min(c.shape[-2], c.shape[-1]) > 1:
            c = F.avg_pool2d(c, 2, ceil_mode=True)
            ww = F.avg_pool2d(ww, 2, ceil_mode=True)
            pyr.append((c, ww))

        # PUSH: coarse -> fine, filling holes with upsampled coarse color.
        up_c, up_w = pyr[-1]
        for i in range(len(pyr) - 2, -1, -1):
            fc, fw = pyr[i]
            coarse = up_c / up_w.clamp_min(1e-8)
            coarse = F.interpolate(coarse, size=fc.shape[-2:], mode="bilinear", align_corners=False)
            has = (fw > 1e-6).float()
            fine = fc / fw.clamp_min(1e-8)
            up_c = has * fine + (1.0 - has) * coarse
            up_w = torch.ones_like(fw)

        out = up_c.squeeze(0).permute(1, 2, 0)                        # [H,W,C], fully defined
        out = torch.where(valid[..., None] > 0.5, t, out)            # keep valid texels exact
        out = (out.clamp(0.0, 1.0) * 255.0 + 0.5).clamp(0, 255).byte().cpu().numpy()
        if squeeze_c:
            out = out[..., 0]
        return out
    except Exception as exc:  # noqa: BLE001 — deliberate fallback
        import cv2
        print(f"[eb_accel] inpaint_fill_gpu fell back to cv2.INPAINT_NS: {exc}")
        return cv2.inpaint((texture * 255).astype(np.uint8),
                           (255 - mask).astype(np.uint8), 3, cv2.INPAINT_NS)
