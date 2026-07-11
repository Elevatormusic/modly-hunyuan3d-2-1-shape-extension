# tests/test_mc_algo.py
import os
import sys
import types
import unittest


# import the generator module with the services stub (repo convention)
def _install_services_stub():
    if "services.generators.base" in sys.modules:
        return
    base = types.ModuleType("services.generators.base")

    class BaseGenerator:  # minimal stand-in
        def __init__(self, *a, **k):
            pass

    def smooth_progress(*a, **k):
        pass

    class GenerationCancelled(Exception):
        pass

    base.BaseGenerator = BaseGenerator
    base.smooth_progress = smooth_progress
    base.GenerationCancelled = GenerationCancelled
    for name in ("services", "services.generators", "services.generators.base"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["services.generators.base"] = base


_install_services_stub()
import generator as gmod

G = gmod.Hunyuan3DShapeV21Generator


class TestSelectMcAlgo(unittest.TestCase):
    def setUp(self):
        self.g = G.__new__(G)          # no __init__ needed for these helpers
        os.environ.pop("EB_MC_ALGO", None)

    def tearDown(self):
        os.environ.pop("EB_MC_ALGO", None)
        sys.modules.pop("diso", None)

    def test_env_override_wins(self):
        os.environ["EB_MC_ALGO"] = "mc"
        self.assertEqual(self.g._select_mc_algo(), "mc")
        os.environ["EB_MC_ALGO"] = "dmc"
        self.assertEqual(self.g._select_mc_algo(), "dmc")

    def test_env_override_ignores_garbage(self):
        os.environ["EB_MC_ALGO"] = "banana"
        # garbage -> fall through to availability detection (diso absent -> mc)
        sys.modules.pop("diso", None)
        self.assertIn(self.g._select_mc_algo(), ("mc", "dmc"))

    def test_dmc_when_diso_importable(self):
        sys.modules["diso"] = types.ModuleType("diso")   # pretend installed
        self.assertEqual(self.g._select_mc_algo(), "dmc")

    def test_mc_when_diso_absent(self):
        sys.modules.pop("diso", None)
        # ensure a real import would fail: block it
        sys.modules["diso"] = None                        # import diso -> ImportError
        try:
            self.assertEqual(self.g._select_mc_algo(), "mc")
        finally:
            sys.modules.pop("diso", None)


if __name__ == "__main__":
    unittest.main()
