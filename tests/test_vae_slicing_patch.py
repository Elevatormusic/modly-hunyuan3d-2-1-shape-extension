"""Section 8e (VAE slicing) durable-patch tests.

The multiview decode runs every view in one batch, and that single decode is the
paint stage's memory peak -- it is where tight-VRAM cards run out. Section 8e
turns on AutoencoderKL slicing so the batch is decoded a view at a time.

Same string-surgery harness as test_phase_offload_patch: synthetic files carrying
the real upstream anchors, no torch, no GPU.
"""
import ast
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

    base.BaseGenerator = BaseGenerator
    base.smooth_progress = lambda *a, **k: None
    base.GenerationCancelled = type("GenerationCancelled", (Exception,), {})
    services.generators = gens
    gens.base = base
    sys.modules["services"] = services
    sys.modules["services.generators"] = gens
    sys.modules["services.generators.base"] = base


def _G():
    _install_services_stub()
    from generator import Hunyuan3DShapeV21Generator as G
    return G


# Pristine upstream shape: the anchor is the use_dino line, which survives whether
# or not 8a has already rewritten the assignment block above it.
_PRISTINE = '''import os
import torch


class multiviewDiffusionNet:
    def __init__(self, config) -> None:
        setattr(pipeline, "view_size", cfg.model.params.get("view_size", 320))
        self.pipeline = pipeline.to(self.device)

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            from hunyuanpaintpbr.unet.modules import Dino_v2
            self.dino_v2 = Dino_v2(config.dino_ckpt_path).to(torch.float16)
            self.dino_v2 = self.dino_v2.to(self.device)

    def forward_one(self, input_images, control_images, **kwargs):
        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))
        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            dino_hidden_states = self.dino_v2(input_images[0])
            kwargs["dino_hidden_states"] = dino_hidden_states
        return self.pipeline(**kwargs)
'''


class TestVaeSlicingPatch(unittest.TestCase):
    def _patch(self, text=_PRISTINE, times=1):
        d = pathlib.Path(tempfile.mkdtemp())
        (d / "utils").mkdir()
        mu = d / "utils" / "multiview_utils.py"
        mu.write_text(text, encoding="utf-8")
        g = _G().__new__(_G())
        for _ in range(times):
            g._patch_gpu_accel(d)
        return mu.read_text(encoding="utf-8")

    def test_slicing_is_enabled(self):
        out = self._patch()
        self.assertIn("self.pipeline.vae.enable_slicing()", out)
        self.assertIn("paint VAE: slicing ENABLED", out)

    def test_patch_output_is_valid_python(self):
        ast.parse(self._patch())

    def test_idempotent(self):
        once, twice = self._patch(times=1), self._patch(times=2)
        self.assertEqual(once, twice)
        self.assertEqual(twice.count("enable_slicing()"), 1)

    def test_slicing_call_is_guarded(self):
        # It runs inside the paint pipeline's construction; a failure here must
        # never take the whole texture pass down.
        out = self._patch()
        idx = out.index("self.pipeline.vae.enable_slicing()")
        self.assertIn("try:", out[max(0, idx - 200):idx])
        self.assertIn("except Exception", out[idx:idx + 300])

    def test_applies_ahead_of_the_dino_block(self):
        # Anchored on the __init__ use_dino line, so slicing is on before any
        # forward runs -- not inserted into forward_one's copy of that line.
        out = self._patch()
        self.assertLess(out.index("enable_slicing()"), out.index("def forward_one"))

    def test_no_op_when_already_present(self):
        pre = _PRISTINE.replace(
            '        if hasattr(self.pipeline.unet, "use_dino")',
            '        # [eb_accel] paint VAE: slicing ENABLED\n'
            '        if hasattr(self.pipeline.unet, "use_dino")', 1)
        self.assertEqual(self._patch(text=pre).count("slicing ENABLED"), 1)


if __name__ == "__main__":
    unittest.main()
