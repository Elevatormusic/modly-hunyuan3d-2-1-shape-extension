"""Unit tests for the prebuilt-native-modules feature.

Covered (all hermetic — no GPU, no toolchain, no real pip):
  * ABI gate: exact win32 / cp311 / torch 2.7.0+cu128 match; every mismatch → False.
  * sha256 verification against PROVENANCE (mismatch + corruption → clean False).
  * transactional rollback: a step-3 (inpaint copy) failure after a step-2
    (wheel install) success must pip-uninstall the wheel and return False.
  * happy path applies both artifacts and returns True with no rollback.
  * _build_paint_extensions still runs the source build when the prebuilt path
    returns False.
  * the committed artifacts match the sha256s embedded in PROVENANCE.md.
  * the .gitignore *.pyd/*.whl negations exist (and sit after the *.pyd rule).

torch and the native imports are stubbed exactly as the other generator tests
stub `services.generators.base`; pip/subprocess/copyfile are mocked.
"""
import contextlib
import hashlib
import importlib
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


_REPO = pathlib.Path(__file__).resolve().parents[1]


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


class _FakeCP:
    """Stand-in for subprocess.CompletedProcess."""
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@contextlib.contextmanager
def _stub_torch(version):
    """Install a fake `torch` with the given __version__ for the duration."""
    saved = sys.modules.get("torch")
    m = types.ModuleType("torch")
    m.__version__ = version
    sys.modules["torch"] = m
    try:
        yield
    finally:
        if saved is not None:
            sys.modules["torch"] = saved
        else:
            sys.modules.pop("torch", None)


def _fake_import_module(state):
    """A drop-in for importlib.import_module that fakes the two native modules
    according to `state` and delegates everything else to the real importer.

    state keys: kernel_ok (custom_rasterizer_kernel importable),
                cr_ok (custom_rasterizer importable),
                cr_has_rasterize (default True)."""
    real = importlib.import_module

    def _imp(name, *a, **k):
        if name == "custom_rasterizer_kernel":
            if state.get("kernel_ok"):
                return types.ModuleType("custom_rasterizer_kernel")
            raise ImportError("stub: no custom_rasterizer_kernel")
        if name == "custom_rasterizer":
            if state.get("cr_ok"):
                mod = types.ModuleType("custom_rasterizer")
                if state.get("cr_has_rasterize", True):
                    mod.rasterize = lambda *a, **k: None
                return mod
            raise ImportError("stub: no custom_rasterizer")
        return real(name, *a, **k)

    return _imp


def _make_bundle(root: pathlib.Path, *, wheel_bytes=b"WHEEL-BYTES",
                 pyd_bytes=b"PYD-BYTES", bad_hashes=False):
    """Write a synthetic prebuilt bundle (wheel + pyd + PROVENANCE.md).

    PROVENANCE lists the true sha256 of each file unless bad_hashes=True, in
    which case the wheel's recorded hash is wrong (simulating a mismatch)."""
    G = _G()
    bundle = root / G._PREBUILT_DIRNAME
    bundle.mkdir(parents=True, exist_ok=True)
    wheel = bundle / G._PREBUILT_WHEEL
    pyd = bundle / G._PREBUILT_INPAINT
    wheel.write_bytes(wheel_bytes)
    pyd.write_bytes(pyd_bytes)
    wsha = hashlib.sha256(wheel_bytes).hexdigest()
    psha = hashlib.sha256(pyd_bytes).hexdigest()
    if bad_hashes:
        wsha = "0" * 64
    (bundle / "PROVENANCE.md").write_text(
        "## Checksums (sha256)\n\n```\n"
        f"{wsha}  {G._PREBUILT_WHEEL}\n"
        f"{psha}  {G._PREBUILT_INPAINT}\n"
        "```\n",
        encoding="utf-8",
    )
    return bundle


class _Base(unittest.TestCase):
    def setUp(self):
        self.G = _G()
        # bare instance: bypass BaseGenerator.__init__ (needs Modly runtime args)
        self.g = self.G.__new__(self.G)

    def tearDown(self):
        for m in ("custom_rasterizer", "custom_rasterizer_kernel"):
            sys.modules.pop(m, None)

    @contextlib.contextmanager
    def _good_gate(self):
        with _stub_torch("2.7.0+cu128"), \
             mock.patch.object(sys, "platform", "win32"), \
             mock.patch.object(sys, "version_info", (3, 11, 0, "final", 0)):
            yield


class TestABIGate(_Base):
    def test_non_win32_returns_false(self):
        with _stub_torch("2.7.0+cu128"), \
             mock.patch.object(sys, "platform", "linux"), \
             mock.patch.object(sys, "version_info", (3, 11, 0, "final", 0)), \
             mock.patch("subprocess.run") as run:
            self.assertFalse(self.g._try_prebuilt_extensions(pathlib.Path(".")))
            run.assert_not_called()

    def test_wrong_python_returns_false(self):
        with _stub_torch("2.7.0+cu128"), \
             mock.patch.object(sys, "platform", "win32"), \
             mock.patch.object(sys, "version_info", (3, 10, 0, "final", 0)), \
             mock.patch("subprocess.run") as run:
            self.assertFalse(self.g._try_prebuilt_extensions(pathlib.Path(".")))
            run.assert_not_called()

    def test_wrong_torch_version_returns_false(self):
        # EXACT match required — cu126/cpu builds and near-miss versions all fail.
        for ver in ("2.6.0+cu124", "2.7.0+cu126", "2.7.1+cu128", "2.7.0+cpu",
                    "2.7.0"):
            with self.subTest(ver=ver):
                with _stub_torch(ver), \
                     mock.patch.object(sys, "platform", "win32"), \
                     mock.patch.object(sys, "version_info", (3, 11, 0, "final", 0)), \
                     mock.patch("subprocess.run") as run:
                    self.assertFalse(
                        self.g._try_prebuilt_extensions(pathlib.Path(".")))
                    run.assert_not_called()

    def test_torch_missing_returns_false(self):
        # No torch importable at all → gate declines, nothing attempted.
        real_import = __import__

        def no_torch(name, *a, **k):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *a, **k)

        saved = sys.modules.pop("torch", None)
        try:
            with mock.patch.object(sys, "platform", "win32"), \
                 mock.patch.object(sys, "version_info", (3, 11, 0, "final", 0)), \
                 mock.patch("builtins.__import__", side_effect=no_torch), \
                 mock.patch("subprocess.run") as run:
                self.assertFalse(
                    self.g._try_prebuilt_extensions(pathlib.Path(".")))
                run.assert_not_called()
        finally:
            if saved is not None:
                sys.modules["torch"] = saved

    def test_gate_passes_then_missing_bundle_returns_false(self):
        # A correct ABI but no bundle on disk still returns False (not a crash),
        # proving the gate passed and the artifact-presence check caught it.
        with tempfile.TemporaryDirectory() as d:
            missing = pathlib.Path(d) / "does-not-exist"
            with self._good_gate(), \
                 mock.patch.object(self.G, "_prebuilt_bundle_dir",
                                   lambda self: missing), \
                 mock.patch("subprocess.run") as run:
                self.assertFalse(
                    self.g._try_prebuilt_extensions(pathlib.Path(d)))
                run.assert_not_called()


class TestSha256Verification(_Base):
    def test_hash_mismatch_returns_false(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = _make_bundle(pathlib.Path(d), bad_hashes=True)
            paint = pathlib.Path(d) / "paint"
            paint.mkdir()
            with self._good_gate(), \
                 mock.patch.object(self.G, "_prebuilt_bundle_dir",
                                   lambda self: bundle), \
                 mock.patch("subprocess.run") as run:
                self.assertFalse(self.g._try_prebuilt_extensions(paint))
                run.assert_not_called()   # never got to the install step

    def test_corrupted_artifact_returns_false(self):
        # PROVENANCE is correct, but the wheel bytes were tampered after the
        # fact → recomputed sha differs → clean False, no crash.
        with tempfile.TemporaryDirectory() as d:
            bundle = _make_bundle(pathlib.Path(d))
            (bundle / self.G._PREBUILT_WHEEL).write_bytes(b"CORRUPTED-DIFFERENT")
            paint = pathlib.Path(d) / "paint"
            paint.mkdir()
            with self._good_gate(), \
                 mock.patch.object(self.G, "_prebuilt_bundle_dir",
                                   lambda self: bundle), \
                 mock.patch("subprocess.run") as run:
                self.assertFalse(self.g._try_prebuilt_extensions(paint))
                run.assert_not_called()

    def test_provenance_missing_hash_entry_returns_false(self):
        # PROVENANCE parses but lacks an entry for one artifact → False.
        with tempfile.TemporaryDirectory() as d:
            bundle = _make_bundle(pathlib.Path(d))
            (bundle / "PROVENANCE.md").write_text(
                "no checksum lines here\n", encoding="utf-8")
            paint = pathlib.Path(d) / "paint"
            paint.mkdir()
            with self._good_gate(), \
                 mock.patch.object(self.G, "_prebuilt_bundle_dir",
                                   lambda self: bundle), \
                 mock.patch("subprocess.run") as run:
                self.assertFalse(self.g._try_prebuilt_extensions(paint))
                run.assert_not_called()


class TestApplyAndRollback(_Base):
    def test_happy_path_applies_both_and_returns_true(self):
        state = {"kernel_ok": False, "cr_ok": True}
        calls = []

        def fake_run(cmd, **k):
            calls.append(cmd)
            return _FakeCP(0)

        with tempfile.TemporaryDirectory() as d:
            bundle = _make_bundle(pathlib.Path(d))
            paint = pathlib.Path(d) / "paint"
            paint.mkdir()
            with self._good_gate(), \
                 mock.patch.object(self.G, "_prebuilt_bundle_dir",
                                   lambda self: bundle), \
                 mock.patch("importlib.import_module",
                            side_effect=_fake_import_module(state)), \
                 mock.patch("subprocess.run", side_effect=fake_run):
                ok = self.g._try_prebuilt_extensions(paint)

            self.assertTrue(ok)
            # wheel installed via `sys.executable -m pip install <wheel>
            # --no-deps --force-reinstall`, nothing uninstalled
            installs = [c for c in calls if "install" in c]
            self.assertEqual(len(installs), 1)
            self.assertIn("--no-deps", installs[0])
            self.assertIn("--force-reinstall", installs[0])
            self.assertFalse(any("uninstall" in c for c in calls))
            # inpaint pyd landed in DifferentiableRenderer/
            dest = paint / "DifferentiableRenderer" / self.G._PREBUILT_INPAINT
            self.assertTrue(dest.exists())

    def test_skips_install_when_kernel_already_importable(self):
        # If the kernel is already importable, step 2 must NOT pip-install.
        state = {"kernel_ok": True, "cr_ok": True}
        calls = []

        def fake_run(cmd, **k):
            calls.append(cmd)
            return _FakeCP(0)

        with tempfile.TemporaryDirectory() as d:
            bundle = _make_bundle(pathlib.Path(d))
            paint = pathlib.Path(d) / "paint"
            paint.mkdir()
            with self._good_gate(), \
                 mock.patch.object(self.G, "_prebuilt_bundle_dir",
                                   lambda self: bundle), \
                 mock.patch("importlib.import_module",
                            side_effect=_fake_import_module(state)), \
                 mock.patch("subprocess.run", side_effect=fake_run):
                ok = self.g._try_prebuilt_extensions(paint)

            self.assertTrue(ok)
            self.assertFalse(any("install" in c for c in calls))

    def test_rollback_when_inpaint_copy_fails_uninstalls_wheel(self):
        # THE transactionality case: step 2 (wheel install) succeeds, step 3
        # (inpaint copy) fails → the wheel must be uninstalled and the call
        # returns False, leaving pristine state for the source build.
        state = {"kernel_ok": False, "cr_ok": True}
        calls = []

        def fake_run(cmd, **k):
            calls.append(cmd)
            return _FakeCP(0)

        with tempfile.TemporaryDirectory() as d:
            bundle = _make_bundle(pathlib.Path(d))
            paint = pathlib.Path(d) / "paint"
            paint.mkdir()
            with self._good_gate(), \
                 mock.patch.object(self.G, "_prebuilt_bundle_dir",
                                   lambda self: bundle), \
                 mock.patch("importlib.import_module",
                            side_effect=_fake_import_module(state)), \
                 mock.patch("subprocess.run", side_effect=fake_run), \
                 mock.patch("shutil.copyfile",
                            side_effect=OSError("simulated disk failure")):
                ok = self.g._try_prebuilt_extensions(paint)

            self.assertFalse(ok)
            installs = [c for c in calls
                        if "install" in c and "uninstall" not in c]
            uninstalls = [c for c in calls if "uninstall" in c]
            self.assertTrue(installs, "wheel should have been installed")
            self.assertTrue(uninstalls, "rollback should uninstall the wheel")
            self.assertIn("custom_rasterizer", uninstalls[0])
            # no stray pyd left behind
            dest = paint / "DifferentiableRenderer" / self.G._PREBUILT_INPAINT
            self.assertFalse(dest.exists())

    def test_rollback_deletes_copied_pyd_on_late_failure(self):
        # Step 3 copies the pyd, then step 4's verify fails (wrapper import
        # fails) → the copied pyd must be deleted AND the wheel uninstalled.
        state = {"kernel_ok": False, "cr_ok": False}
        calls = []

        def fake_run(cmd, **k):
            calls.append(cmd)
            return _FakeCP(0)

        with tempfile.TemporaryDirectory() as d:
            bundle = _make_bundle(pathlib.Path(d))
            paint = pathlib.Path(d) / "paint"
            paint.mkdir()
            with self._good_gate(), \
                 mock.patch.object(self.G, "_prebuilt_bundle_dir",
                                   lambda self: bundle), \
                 mock.patch("importlib.import_module",
                            side_effect=_fake_import_module(state)), \
                 mock.patch("subprocess.run", side_effect=fake_run):
                ok = self.g._try_prebuilt_extensions(paint)

            self.assertFalse(ok)
            self.assertTrue(any("uninstall" in c for c in calls))
            dest = paint / "DifferentiableRenderer" / self.G._PREBUILT_INPAINT
            self.assertFalse(dest.exists(), "copied pyd must be rolled back")

    def test_rollback_when_wrapper_missing_rasterize(self):
        # A wheel that installs but whose custom_rasterizer lacks `rasterize`
        # (the exact RV4 failure mode) must roll back and return False.
        state = {"kernel_ok": False, "cr_ok": True, "cr_has_rasterize": False}
        calls = []

        def fake_run(cmd, **k):
            calls.append(cmd)
            return _FakeCP(0)

        with tempfile.TemporaryDirectory() as d:
            bundle = _make_bundle(pathlib.Path(d))
            paint = pathlib.Path(d) / "paint"
            paint.mkdir()
            with self._good_gate(), \
                 mock.patch.object(self.G, "_prebuilt_bundle_dir",
                                   lambda self: bundle), \
                 mock.patch("importlib.import_module",
                            side_effect=_fake_import_module(state)), \
                 mock.patch("subprocess.run", side_effect=fake_run):
                ok = self.g._try_prebuilt_extensions(paint)

            self.assertFalse(ok)
            self.assertTrue(any("uninstall" in c for c in calls))

    def test_never_raises_on_pip_failure(self):
        # A non-zero pip install is swallowed (nothing installed → nothing to
        # roll back) → clean False.
        state = {"kernel_ok": False, "cr_ok": True}

        with tempfile.TemporaryDirectory() as d:
            bundle = _make_bundle(pathlib.Path(d))
            paint = pathlib.Path(d) / "paint"
            paint.mkdir()
            with self._good_gate(), \
                 mock.patch.object(self.G, "_prebuilt_bundle_dir",
                                   lambda self: bundle), \
                 mock.patch("importlib.import_module",
                            side_effect=_fake_import_module(state)), \
                 mock.patch("subprocess.run",
                            return_value=_FakeCP(1, stderr="boom")):
                ok = self.g._try_prebuilt_extensions(paint)
            self.assertFalse(ok)


class TestBuildPaintExtensionsUnaffected(_Base):
    def test_source_build_runs_when_prebuilt_false(self):
        # When the prebuilt path declines, _build_paint_extensions must fall
        # through to the existing source build exactly as before.
        state = {"kernel_ok": False}

        def build_cr(cr_dir):
            state["kernel_ok"] = True   # simulate a successful source build
            return _FakeCP(0)

        with tempfile.TemporaryDirectory() as d:
            paint = pathlib.Path(d)
            (paint / "DifferentiableRenderer").mkdir()
            # pre-drop a fake inpaint .pyd so step 2 of the source build skips
            (paint / "DifferentiableRenderer"
                   / "mesh_inpaint_processor.cp311-win_amd64.pyd").write_bytes(b"x")
            with mock.patch.object(self.G, "_try_prebuilt_extensions",
                                   return_value=False) as tpe, \
                 mock.patch.object(self.G, "_patch_custom_rasterizer_sources"), \
                 mock.patch.object(self.G, "_build_custom_rasterizer",
                                   side_effect=build_cr) as bcr, \
                 mock.patch("importlib.import_module",
                            side_effect=_fake_import_module(state)):
                self.g._build_paint_extensions(paint)

            tpe.assert_called_once()
            bcr.assert_called_once()

    def test_try_prebuilt_called_at_top(self):
        import inspect
        src = inspect.getsource(self.G._build_paint_extensions)
        self.assertIn("_try_prebuilt_extensions", src)
        # ...and BEFORE the existing _kernel_ready() / glob checks.
        self.assertLess(src.index("_try_prebuilt_extensions"),
                        src.index("_kernel_ready"))


class TestHelpers(_Base):
    def test_sha256_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "blob.bin"
            data = b"the quick brown fox" * 1000
            p.write_bytes(data)
            self.assertEqual(self.G._sha256(p), hashlib.sha256(data).hexdigest())

    def test_parse_prebuilt_hashes_extracts_mapping(self):
        text = (
            "Prose mentioning custom_rasterizer-0.1-cp311-cp311-win_amd64.whl "
            "with no hash on this line is ignored.\n"
            "## Checksums (sha256)\n\n```\n"
            + "aa" * 32 + "  custom_rasterizer-0.1-cp311-cp311-win_amd64.whl\n"
            + "bb" * 32 + "  mesh_inpaint_processor.cp311-win_amd64.pyd\n"
            + "```\n"
        )
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "PROVENANCE.md"
            p.write_text(text, encoding="utf-8")
            hashes = self.G._parse_prebuilt_hashes(p)
        self.assertEqual(
            hashes,
            {"custom_rasterizer-0.1-cp311-cp311-win_amd64.whl": "aa" * 32,
             "mesh_inpaint_processor.cp311-win_amd64.pyd": "bb" * 32})


class TestCommittedArtifacts(unittest.TestCase):
    """Guards the real, committed bundle — runs against the on-disk artifacts."""
    def setUp(self):
        self.G = _G()
        self.bundle = _REPO / "prebuilt" / self.G._PREBUILT_DIRNAME

    def test_all_three_artifacts_present(self):
        self.assertTrue((self.bundle / self.G._PREBUILT_WHEEL).is_file())
        self.assertTrue((self.bundle / self.G._PREBUILT_INPAINT).is_file())
        self.assertTrue((self.bundle / "PROVENANCE.md").is_file())

    def test_committed_hashes_match_provenance(self):
        expected = self.G._parse_prebuilt_hashes(self.bundle / "PROVENANCE.md")
        for name in (self.G._PREBUILT_WHEEL, self.G._PREBUILT_INPAINT):
            with self.subTest(artifact=name):
                self.assertIn(name, expected)
                self.assertEqual(self.G._sha256(self.bundle / name),
                                 expected[name])


class TestGitignoreNegations(unittest.TestCase):
    def test_pyd_and_whl_negations_present_after_pyd_rule(self):
        text = (_REPO / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("!prebuilt/**/*.pyd", text)
        self.assertIn("!prebuilt/**/*.whl", text)
        # Order matters in .gitignore: the negations must come AFTER `*.pyd`,
        # or the blanket ignore wins and the artifacts silently vanish.
        self.assertLess(text.index("\n*.pyd"), text.index("!prebuilt/**/*.pyd"))
        self.assertLess(text.index("\n*.pyd"), text.index("!prebuilt/**/*.whl"))


if __name__ == "__main__":
    unittest.main()
