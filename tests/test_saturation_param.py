import json, pathlib, sys, types, unittest


def _install_services_stub():
    """Same stub tests/test_generator_params.py uses so generator.py imports
    outside the Modly process."""
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


class TestSaturationParam(unittest.TestCase):
    def _schema(self):
        _install_services_stub()
        from generator import Hunyuan3DShapeV21Generator as G
        return {p["id"]: p for p in G.params_schema()}

    def _manifest_params(self):
        p = pathlib.Path(__file__).resolve().parents[1] / "manifest.json"
        return {q["id"]: q for q in json.loads(p.read_text())["nodes"][0]["params_schema"]}

    def test_manifest_has_saturation_default_subtle(self):
        mp = self._manifest_params()
        self.assertIn("saturation", mp)
        self.assertEqual(mp["saturation"]["default"], "subtle")
        self.assertEqual({o["value"] for o in mp["saturation"]["options"]},
                         {"off", "subtle", "medium", "strong"})

    def test_schema_has_saturation_default_subtle(self):
        schema = self._schema()
        self.assertIn("saturation", schema)
        self.assertEqual(schema["saturation"]["default"], "subtle")
        self.assertEqual({o["value"] for o in schema["saturation"]["options"]},
                         {"off", "subtle", "medium", "strong"})

    def test_manifest_and_schema_agree(self):
        self.assertEqual(self._manifest_params()["saturation"]["default"],
                         self._schema()["saturation"]["default"])

    def test_run_texture_accepts_saturation(self):
        _install_services_stub()
        import inspect
        from generator import Hunyuan3DShapeV21Generator as G
        params = inspect.signature(G._run_texture).parameters
        self.assertIn("saturation", params)
        self.assertEqual(params["saturation"].default, "subtle")


if __name__ == "__main__":
    unittest.main()
