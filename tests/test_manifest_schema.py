import json
import pathlib
import sys
import types
import unittest


def _install_services_stub():
    """generator.py imports Modly's runtime services.generators.base at module top;
    stub it so params_schema() can be exercised in a plain test run."""
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


class TestManifestSchemaLockstep(unittest.TestCase):
    def _manifest(self):
        repo = pathlib.Path(__file__).resolve().parents[1]
        man = json.loads((repo / "manifest.json").read_text(encoding="utf-8"))
        return man, {p["id"]: p for p in man["nodes"][0]["params_schema"]}

    def _schema(self):
        _install_services_stub()
        from generator import Hunyuan3DShapeV21Generator as G
        return {p["id"]: p for p in G.params_schema()}

    def test_version_1_6_0(self):
        man, _ = self._manifest()
        self.assertEqual(man["version"], "1.6.0")

    def test_bake_default_off_both_sides(self):
        _, mp = self._manifest()
        sp = self._schema()
        self.assertEqual(mp["bake_normal_map"]["default"], 0)
        self.assertEqual(sp["bake_normal_map"]["default"], 0)

    def test_debug_sheet_present_and_lockstep(self):
        _, mp = self._manifest()
        sp = self._schema()
        self.assertIn("debug_sheet", mp)
        self.assertIn("debug_sheet", sp)
        self.assertEqual(mp["debug_sheet"]["type"], sp["debug_sheet"]["type"])
        self.assertEqual(mp["debug_sheet"]["default"], sp["debug_sheet"]["default"])
        mvals = {o["value"] for o in mp["debug_sheet"]["options"]}
        svals = {o["value"] for o in sp["debug_sheet"]["options"]}
        self.assertEqual(mvals, svals)
        self.assertEqual(svals, {0, 1})


if __name__ == "__main__":
    unittest.main()
