import sys
import types
import unittest


def _install_services_stub():
    """generator.py imports Modly's runtime `services.generators.base` at module top.
    That module only exists inside the Modly process, so stub it to import the class
    and exercise the pure params_schema() classmethod in a plain test run."""
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


class TestParams(unittest.TestCase):
    def _schema(self):
        _install_services_stub()
        from generator import Hunyuan3DShapeV21Generator as G
        return {p["id"]: p for p in G.params_schema()}

    def test_mesh_mode_present_default_regular(self):
        schema = self._schema()
        self.assertIn("mesh_mode", schema)
        self.assertEqual(schema["mesh_mode"]["default"], "regular")
        values = {o["value"] for o in schema["mesh_mode"]["options"]}
        self.assertEqual(values, {"regular", "isotropic", "bpt"})

    def test_bake_normal_map_present_default_off(self):
        schema = self._schema()
        self.assertIn("bake_normal_map", schema)
        self.assertEqual(schema["bake_normal_map"]["default"], 0)

    def test_target_faces_still_present(self):
        # Pre-existing CAD/print decimation param must not be clobbered.
        self.assertIn("target_faces", self._schema())

    def test_texture_memory_present_default_auto(self):
        schema = self._schema()
        self.assertIn("texture_memory", schema)
        self.assertEqual(schema["texture_memory"]["default"], "auto")
        values = {o["value"] for o in schema["texture_memory"]["options"]}
        self.assertEqual(values, {"auto", "standard", "reduced"})

    def test_low_vram_mode_removed(self):
        # low_vram_mode (broken CPU offload) was removed in favor of use_shared_vram.
        self.assertNotIn("low_vram_mode", self._schema())

    def test_manifest_and_schema_agree_on_new_knobs(self):
        import json, pathlib
        manifest = json.loads((pathlib.Path(__file__).resolve().parents[1] / "manifest.json").read_text())
        mparams = {p["id"]: p for p in manifest["nodes"][0]["params_schema"]}
        schema = self._schema()
        for pid in ("texture_memory",):
            self.assertIn(pid, mparams)
            self.assertEqual(mparams[pid]["default"], schema[pid]["default"])

    def test_run_texture_accepts_memory_knobs(self):
        _install_services_stub()
        import inspect
        from generator import Hunyuan3DShapeV21Generator as G
        params = inspect.signature(G._run_texture).parameters
        self.assertIn("texture_memory", params)
        self.assertNotIn("low_vram_mode", params)
        self.assertEqual(params["texture_memory"].default, "auto")

    def test_texture_memory_has_reduced_option(self):
        values = {o["value"] for o in self._schema()["texture_memory"]["options"]}
        self.assertIn("reduced", values)
        # retired legacy ids no longer offered in the manifest/schema
        self.assertNotIn("max", values)
        self.assertNotIn("balanced", values)

    def test_use_shared_vram_present_default_off(self):
        schema = self._schema()
        self.assertIn("use_shared_vram", schema)
        self.assertEqual(schema["use_shared_vram"]["default"], 0)

    def test_use_shared_vram_manifest_parity(self):
        import json, pathlib
        manifest = json.loads((pathlib.Path(__file__).resolve().parents[1] / "manifest.json").read_text())
        mparams = {p["id"]: p for p in manifest["nodes"][0]["params_schema"]}
        self.assertIn("use_shared_vram", mparams)
        self.assertEqual(mparams["use_shared_vram"]["default"], 0)
        mvals = {o["value"] for o in mparams["texture_memory"]["options"]}
        self.assertEqual(mvals, {"auto", "standard", "reduced"})

    def test_run_texture_accepts_use_shared_vram(self):
        _install_services_stub()
        import inspect
        from generator import Hunyuan3DShapeV21Generator as G
        params = inspect.signature(G._run_texture).parameters
        self.assertIn("use_shared_vram", params)
        self.assertEqual(params["use_shared_vram"].default, False)


class TestBackgroundRemoval(unittest.TestCase):
    def _mod(self):
        _install_services_stub()
        import generator
        return generator

    def test_matte_coverage_ok_boundaries(self):
        m = self._mod()
        self.assertTrue(m._matte_coverage_ok(0.30))     # a normal object
        self.assertFalse(m._matte_coverage_ok(0.0))     # object erased
        self.assertFalse(m._matte_coverage_ok(1.0))     # background kept -> floor
        self.assertFalse(m._matte_coverage_ok(0.95))    # near-opaque -> floor risk
        self.assertFalse(m._matte_coverage_ok(0.02))    # lower bound exclusive
        self.assertFalse(m._matte_coverage_ok(0.92))    # upper bound exclusive

    def test_bg_models_u2net_first_isnet_fallback(self):
        m = self._mod()
        self.assertEqual(m._BG_MODELS[0], "u2net")
        self.assertIn("isnet-general-use", m._BG_MODELS)

    def test_remove_background_and_debug_methods_exist(self):
        self._mod()
        from generator import Hunyuan3DShapeV21Generator as G
        self.assertTrue(hasattr(G, "_remove_background"))
        self.assertTrue(hasattr(G, "_save_matte_debug"))

    def test_preprocess_forces_cpu_and_drops_backgroundremover(self):
        import inspect
        self._mod()
        from generator import Hunyuan3DShapeV21Generator as G
        # Matting must never touch the broken CUDA provider.
        self.assertIn("CPUExecutionProvider", inspect.getsource(G._remove_background))
        # The old vendored BackgroundRemover path (RGBA-in, no sanity guard) is gone.
        self.assertNotIn("BackgroundRemover", inspect.getsource(G._preprocess))


if __name__ == "__main__":
    unittest.main()
