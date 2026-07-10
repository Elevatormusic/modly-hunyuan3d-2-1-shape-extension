import json
import pathlib
import unittest


def _iter_params(man):
    # Real manifest shape: params live under nodes[0].params_schema and are
    # keyed by "id" (not "name"). Yield each param dict.
    for node in man.get("nodes", []):
        for p in node.get("params_schema", []):
            yield p


class TestWiring(unittest.TestCase):
    def _repo(self):
        return pathlib.Path(__file__).resolve().parents[1]

    def test_manifest_and_schema_lockstep(self):
        man = json.loads((self._repo() / "manifest.json").read_text(encoding="utf-8"))
        ids = {p["id"] for p in _iter_params(man)}
        self.assertIn("seam_fix", ids)

        src = (self._repo() / "generator.py").read_text(encoding="utf-8")
        self.assertIn('"seam_fix"', src)  # present in params_schema()

        # _run_texture accepts seam_fix and delegates the post-paint tail to
        # finishing.finish (which owns the seam reconcile + bake + QA stages).
        rt = src[src.index("def _run_texture"):]
        self.assertIn("seam_fix", rt)
        self.assertIn("finishing.finish", rt)

    def test_manifest_and_schema_default_agree(self):
        # Lockstep: manifest and params_schema() must agree on the seam_fix default.
        man = json.loads((self._repo() / "manifest.json").read_text(encoding="utf-8"))
        mparams = {p["id"]: p for p in _iter_params(man)}
        self.assertIn("seam_fix", mparams)
        self.assertEqual(mparams["seam_fix"]["default"], 1)

    def test_paint_tmpdir_is_cleaned_up(self):
        # Fix 10: the tempfile.mkdtemp() paint scratch dir (~15-25 MB/gen) must be
        # removed after finishing.finish() returns, in the _run_texture finally.
        src = (self._repo() / "generator.py").read_text(encoding="utf-8")
        rt = src[src.index("def _run_texture"):src.index("def _ensure_hy3dpaint")]
        self.assertIn("tmp_dir = Path(tempfile.mkdtemp())", rt)
        self.assertIn("shutil.rmtree(tmp_dir", rt)   # cleanup wired
        # the cleanup must live in the finally (after finishing.finish reads tmp_dir)
        self.assertGreater(rt.index("shutil.rmtree(tmp_dir"), rt.index("finishing.finish"))


if __name__ == "__main__":
    unittest.main()
