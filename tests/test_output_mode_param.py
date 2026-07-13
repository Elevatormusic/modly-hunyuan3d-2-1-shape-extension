import json, pathlib, sys, types, unittest, inspect


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


class TestOutputModeParam(unittest.TestCase):
    def _schema(self):
        _install_services_stub()
        from generator import Hunyuan3DShapeV21Generator as G
        return G.params_schema()

    def _manifest_params(self):
        p = pathlib.Path(__file__).resolve().parents[1] / "manifest.json"
        return json.loads(p.read_text())["nodes"][0]["params_schema"]

    def test_output_mode_first_in_schema(self):
        s = self._schema()
        self.assertEqual(s[0]["id"], "output_mode")
        self.assertEqual(s[0]["default"], "custom")
        self.assertEqual({o["value"] for o in s[0]["options"]},
                         {"custom", "render_balanced", "render_max", "game_ready"})

    def test_output_mode_first_in_manifest(self):
        m = self._manifest_params()
        self.assertEqual(m[0]["id"], "output_mode")
        self.assertEqual(m[0]["default"], "custom")

    def test_manifest_and_schema_options_agree(self):
        s = {o["value"] for o in self._schema()[0]["options"]}
        m = {o["value"] for o in self._manifest_params()[0]["options"]}
        self.assertEqual(s, m)

    def test_run_texture_accepts_face_target_and_game_ready(self):
        _install_services_stub()
        from generator import Hunyuan3DShapeV21Generator as G
        params = inspect.signature(G._run_texture).parameters
        self.assertIn("face_target", params)
        self.assertIn("game_ready", params)
        self.assertEqual(params["game_ready"].default, False)


if __name__ == "__main__":
    unittest.main()
