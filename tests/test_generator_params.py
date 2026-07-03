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

    def test_mesh_mode_present_default_isotropic(self):
        schema = self._schema()
        self.assertIn("mesh_mode", schema)
        self.assertEqual(schema["mesh_mode"]["default"], "isotropic")
        values = {o["value"] for o in schema["mesh_mode"]["options"]}
        self.assertEqual(values, {"regular", "isotropic", "bpt"})

    def test_bake_normal_map_present_default_on(self):
        schema = self._schema()
        self.assertIn("bake_normal_map", schema)
        self.assertEqual(schema["bake_normal_map"]["default"], 1)

    def test_target_faces_still_present(self):
        # Pre-existing CAD/print decimation param must not be clobbered.
        self.assertIn("target_faces", self._schema())


if __name__ == "__main__":
    unittest.main()
