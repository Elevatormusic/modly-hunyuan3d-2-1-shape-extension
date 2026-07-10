"""Section 8 (phase CPU offload) durable-patch tests.

_patch_gpu_accel section 8 reproduces the hook-free EB_CPU_OFFLOAD=phase edits onto a
freshly downloaded _hy3dpaint_src. These tests use small synthetic files that contain the
REAL upstream anchor strings and verify:
  * the phase edits apply (markers land) on pristine input,
  * the patch is idempotent (patch-twice == patch-once, i.e. a no-op once patched),
  * the patched output is still syntactically valid Python,
  * and — when the live vendored files are present on disk — the patch is a no-op against
    them (they already carry the hand-applied edits).
No torch, no GPU: the patch is pure string surgery.
"""
import inspect
import pathlib
import sys
import tempfile
import types
import unittest


def _install_services_stub():
    if "services.generators.base" in sys.modules:
        return
    services = types.ModuleType("services")
    gens = types.ModuleType("services.generators")
    base = types.ModuleType("services.generators.base")

    class BaseGenerator:
        pass

    def smooth_progress(*a, **k):
        pass

    class GenerationCancelled(Exception):
        pass

    base.BaseGenerator = BaseGenerator
    base.smooth_progress = smooth_progress
    base.GenerationCancelled = GenerationCancelled
    services.generators = gens
    gens.base = base
    sys.modules["services"] = services
    sys.modules["services.generators"] = gens
    sys.modules["services.generators.base"] = base


def _G():
    _install_services_stub()
    from generator import Hunyuan3DShapeV21Generator as G
    return G


# Synthetic PRISTINE upstream files: minimal but valid Python carrying the exact anchor
# lines section 8 keys on (indentation matches the real hy3dpaint sources).
_PRISTINE_MULTIVIEW = '''import os
import torch


class multiviewDiffusionNet:
    def __init__(self, config) -> None:
        self.device = config.device
        pipeline = _load_pipeline()
        pipeline.eval()
        setattr(pipeline, "view_size", cfg.model.params.get("view_size", 320))
        self.pipeline = pipeline.to(self.device)

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            from hunyuanpaintpbr.unet.modules import Dino_v2
            self.dino_v2 = Dino_v2(config.dino_ckpt_path).to(torch.float16)
            self.dino_v2 = self.dino_v2.to(self.device)

    def forward_one(self, input_images, control_images):
        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            dino_hidden_states = self.dino_v2(input_images[0])
            kwargs["dino_hidden_states"] = dino_hidden_states
        return kwargs
'''

_PRISTINE_PIPELINE = '''import torch
from diffusers import StableDiffusionPipeline


def to_rgb_image(maybe_rgba):
    return maybe_rgba


class HunyuanPaintPipeline(StableDiffusionPipeline):
    def __call__(self, images=None):
        images_vae = _cat(images)
        images_vae = images_vae.to(device=self.vae.device, dtype=self.unet.dtype)
        if self.unet.use_learned_text_clip:
            prompt_embeds = _learned_tokens(self)
        else:
            prompt_embeds = _encode(self)
        return self.denoise(prompt_embeds)

    def denoise(self, prompt_embeds, output_type="pil", latents=None, generator=None):
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        else:
            image = latents
        return image
'''


def _write_pristine(root: pathlib.Path):
    (root / "utils").mkdir(parents=True, exist_ok=True)
    (root / "hunyuanpaintpbr").mkdir(parents=True, exist_ok=True)
    (root / "utils" / "multiview_utils.py").write_text(_PRISTINE_MULTIVIEW, encoding="utf-8")
    (root / "hunyuanpaintpbr" / "pipeline.py").write_text(_PRISTINE_PIPELINE, encoding="utf-8")


class TestPhaseOffloadPatch(unittest.TestCase):
    def _patch(self, root):
        _G()._patch_phase_offload(root)

    def _read(self, root):
        mu = (root / "utils" / "multiview_utils.py").read_text(encoding="utf-8")
        pp = (root / "hunyuanpaintpbr" / "pipeline.py").read_text(encoding="utf-8")
        return mu, pp

    def test_pristine_gets_all_phase_edits(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            _write_pristine(root)
            self._patch(root)
            mu, pp = self._read(root)
            # multiview_utils.py markers (8a-8d)
            self.assertIn("PHASE CPU offload ENABLED", mu)                 # 8a
            self.assertIn('self.dino_v2 = self.dino_v2.to("cpu")', mu)     # 8b
            self.assertIn("_eb_dev = getattr(self.pipeline", mu)           # 8c
            self.assertIn("_eb_ph = os.environ.get", mu)                   # 8d
            # pipeline.py markers (8e-8h)
            self.assertIn("def _eb_phase_to(", pp)                         # 8e
            self.assertIn("_eb_phase_to(self.vae, _eb_dev)", pp)           # 8f
            self.assertIn(
                '_eb_phase_to(self.unet, getattr(self, "_execution_device", None) or "cuda")', pp)  # 8g
            self.assertIn(
                '_eb_phase_to(self.vae, getattr(self, "_execution_device", None) or "cuda")', pp)   # 8h
            # the encode_images device fix flipped self.vae.device -> _eb_dev
            self.assertIn("images_vae = images_vae.to(device=_eb_dev, dtype=self.unet.dtype)", pp)
            self.assertNotIn(
                "images_vae = images_vae.to(device=self.vae.device, dtype=self.unet.dtype)", pp)

    def test_patched_output_is_valid_python(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            _write_pristine(root)
            self._patch(root)
            mu, pp = self._read(root)
            compile(mu, "multiview_utils.py", "exec")   # raises SyntaxError on failure
            compile(pp, "pipeline.py", "exec")

    def test_idempotent_patch_twice_equals_once(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            _write_pristine(root)
            self._patch(root)
            once = self._read(root)
            self._patch(root)                            # second pass must be a no-op
            twice = self._read(root)
            self.assertEqual(once, twice)

    def test_default_off_pristine_lines_preserved(self):
        # The stock full-GPU path must survive: the else-branches still carry the original
        # to(self.device) moves, so with EB_CPU_OFFLOAD unset behavior is unchanged.
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            _write_pristine(root)
            self._patch(root)
            mu, _ = self._read(root)
            self.assertIn("self.pipeline = pipeline.to(self.device)", mu)
            self.assertIn("self.dino_v2 = self.dino_v2.to(self.device)", mu)

    def test_run_texture_wires_offload_env(self):
        # _run_texture must gate EB_CPU_OFFLOAD=phase on plan.offload and log offload=on|off.
        src = inspect.getsource(_G()._run_texture)
        self.assertIn('os.environ["EB_CPU_OFFLOAD"] = "phase"', src)
        self.assertIn('os.environ.pop("EB_CPU_OFFLOAD", None)', src)
        self.assertIn("_plan.offload", src)
        self.assertIn("offload=", src)


_LIVE_ROOT = pathlib.Path(
    r"C:\Users\Shaya\OneDrive\Documents\Modly\models\hunyuan3d-2-1-shape\generate\_hy3dpaint_src")


class TestNoOpOnLiveVendoredFiles(unittest.TestCase):
    @unittest.skipUnless(
        (_LIVE_ROOT / "utils" / "multiview_utils.py").exists()
        and (_LIVE_ROOT / "hunyuanpaintpbr" / "pipeline.py").exists(),
        "live vendored _hy3dpaint_src not present")
    def test_patch_is_noop_on_live_files(self):
        import shutil
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "utils").mkdir(parents=True)
            (root / "hunyuanpaintpbr").mkdir(parents=True)
            shutil.copyfile(_LIVE_ROOT / "utils" / "multiview_utils.py",
                            root / "utils" / "multiview_utils.py")
            shutil.copyfile(_LIVE_ROOT / "hunyuanpaintpbr" / "pipeline.py",
                            root / "hunyuanpaintpbr" / "pipeline.py")
            before = ((root / "utils" / "multiview_utils.py").read_text(encoding="utf-8"),
                      (root / "hunyuanpaintpbr" / "pipeline.py").read_text(encoding="utf-8"))
            _G()._patch_phase_offload(root)
            after = ((root / "utils" / "multiview_utils.py").read_text(encoding="utf-8"),
                     (root / "hunyuanpaintpbr" / "pipeline.py").read_text(encoding="utf-8"))
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
