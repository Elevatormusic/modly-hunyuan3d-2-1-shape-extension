import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import ckpt_guard

_CORRUPT = ("PytorchStreamReader failed reading zip archive: "
            "failed finding central directory")


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


class _GenCase(unittest.TestCase):
    """Builds a generator against a temp model dir, with the heavy load path stubbed."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.sub = self.root / "hunyuan3d-dit-v2-1"
        self.sub.mkdir()
        (self.sub / "config.yaml").write_text("a: 1")
        self.ckpt = self.sub / "model.fp16.ckpt"

    def _write_valid_ckpt(self):
        with zipfile.ZipFile(self.ckpt, "w") as zf:
            zf.writestr("data.pkl", b"x" * 64)

    def _gen(self):
        _install_services_stub()
        from generator import Hunyuan3DShapeV21Generator as G
        g = G.__new__(G)                 # skip BaseGenerator's app wiring
        g.model_dir = self.root
        g.download_check = None
        g._model = None
        g._shape_cpu_offload = None
        return g

    def _wire(self, g, load_side_effect):
        """Stub everything load() touches except the repair logic under test."""
        g._ensure_hy3dshape = lambda: None
        g._ensure_diso = lambda: None
        self.downloads = []
        g._download_weights = lambda: self.downloads.append(1)
        self.calls = []

        def fake_load_pipeline(model_dir):
            self.calls.append(model_dir)
            return load_side_effect(len(self.calls))

        g._load_pipeline = fake_load_pipeline
        return g


class TestIsDownloaded(_GenCase):
    def test_false_for_truncated_ckpt(self):
        # The exact regression: the file exists, so the old gate said "ready".
        self.ckpt.write_bytes(b"partial")
        self.assertFalse(self._gen().is_downloaded())

    def test_false_when_ckpt_absent(self):
        self.assertFalse(self._gen().is_downloaded())

    def test_true_for_complete_ckpt(self):
        self._write_valid_ckpt()
        with mock.patch.object(ckpt_guard, "MIN_CKPT_BYTES", 1):
            self.assertTrue(self._gen().is_downloaded())

    def test_false_when_config_absent(self):
        self._write_valid_ckpt()
        (self.sub / "config.yaml").unlink()
        with mock.patch.object(ckpt_guard, "MIN_CKPT_BYTES", 1):
            self.assertFalse(self._gen().is_downloaded())


class TestDownloadWeights(_GenCase):
    def _fake_hub(self, on_download=lambda: None):
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = lambda **kw: on_download()
        return hub

    def test_rejects_incomplete_download(self):
        g = self._gen()
        with mock.patch.dict(sys.modules, {"huggingface_hub": self._fake_hub()}):
            with self.assertRaises(RuntimeError) as ctx:
                g._download_weights()
        msg = str(ctx.exception)
        self.assertIn("model.fp16.ckpt", msg)
        self.assertIn(str(self.ckpt), msg)      # names the path to look at

    def _sidecar(self):
        side = self.root / ".cache" / "huggingface" / "download" / "hunyuan3d-dit-v2-1"
        side.mkdir(parents=True)
        meta = side / "model.fp16.ckpt.metadata"
        meta.write_text("etag")
        return meta

    def test_purges_rejected_checkpoint_before_downloading(self):
        # snapshot_download skips any file whose sidecar etag still matches, so
        # a checkpoint truncated after a successful download would otherwise
        # survive the re-fetch and fail identically on every launch.
        self.ckpt.write_bytes(b"partial")
        meta = self._sidecar()
        seen = {}

        def on_download():
            seen["ckpt"] = self.ckpt.exists()
            seen["meta"] = meta.exists()
            self._write_valid_ckpt()

        g = self._gen()
        with mock.patch.dict(sys.modules, {"huggingface_hub": self._fake_hub(on_download)}), \
             mock.patch.object(ckpt_guard, "MIN_CKPT_BYTES", 1):
            g._download_weights()

        self.assertFalse(seen["ckpt"], "damaged checkpoint must be gone before the re-fetch")
        self.assertFalse(seen["meta"], "stale etag would make the re-fetch a no-op")

    def test_healthy_checkpoint_is_not_purged(self):
        # The inverse guard: never discard a good 7.4 GB download.
        self._write_valid_ckpt()
        meta = self._sidecar()
        seen = {}

        def on_download():
            seen["ckpt"] = self.ckpt.exists()
            seen["meta"] = meta.exists()

        g = self._gen()
        with mock.patch.dict(sys.modules, {"huggingface_hub": self._fake_hub(on_download)}), \
             mock.patch.object(ckpt_guard, "MIN_CKPT_BYTES", 1):
            g._download_weights()

        self.assertTrue(seen["ckpt"])
        self.assertTrue(seen["meta"])

    def test_accepts_complete_download(self):
        g = self._gen()
        hub = self._fake_hub(on_download=self._write_valid_ckpt)
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}), \
             mock.patch.object(ckpt_guard, "MIN_CKPT_BYTES", 1):
            g._download_weights()           # must not raise


class TestLoadSelfRepair(_GenCase):
    def setUp(self):
        super().setUp()
        self._write_valid_ckpt()
        # Passes the size gate, so the damage can only surface at load time.
        self._floor = mock.patch.object(ckpt_guard, "MIN_CKPT_BYTES", 1)
        self._floor.start()
        self.addCleanup(self._floor.stop)

    def test_repairs_and_retries_once(self):
        def side_effect(n):
            if n == 1:
                raise RuntimeError(_CORRUPT)
            return "pipeline"

        g = self._wire(self._gen(), side_effect)
        g.load()

        self.assertEqual(g._model, "pipeline")
        self.assertEqual(len(self.calls), 2)        # retried exactly once
        self.assertEqual(len(self.downloads), 1)    # re-fetched in between
        self.assertFalse(self.ckpt.exists())        # bad file purged
        self.assertFalse(g._shape_cpu_offload)      # post-load state still set

    def test_purges_hub_sidecars_so_refetch_is_not_skipped(self):
        side = self.root / ".cache" / "huggingface" / "download" / "hunyuan3d-dit-v2-1"
        side.mkdir(parents=True)
        (side / "model.fp16.ckpt.metadata").write_text("etag")

        g = self._wire(self._gen(), lambda n: "ok" if n > 1 else _raise(_CORRUPT))
        g.load()
        self.assertFalse((side / "model.fp16.ckpt.metadata").exists())

    def test_unrelated_error_is_not_treated_as_corruption(self):
        # A 7 GB re-download must never be triggered by an OOM or a code bug.
        g = self._wire(self._gen(), lambda n: _raise("CUDA out of memory"))
        with self.assertRaises(RuntimeError) as ctx:
            g.load()
        self.assertIn("out of memory", str(ctx.exception))
        self.assertEqual(len(self.calls), 1)        # no retry
        self.assertEqual(self.downloads, [])        # no re-download
        self.assertTrue(self.ckpt.exists())         # nothing deleted

    def test_second_failure_reports_actionably(self):
        g = self._wire(self._gen(), lambda n: _raise(_CORRUPT))
        with self.assertRaises(RuntimeError) as ctx:
            g.load()
        msg = str(ctx.exception)
        self.assertIn("model.fp16.ckpt", msg)
        self.assertIn(str(self.ckpt), msg)
        self.assertNotIn("PytorchStreamReader", msg)   # opaque original replaced
        self.assertIsNotNone(ctx.exception.__cause__)  # but still chained
        self.assertEqual(len(self.calls), 2)           # gave up after one retry


def _raise(message):
    raise RuntimeError(message)


if __name__ == "__main__":
    unittest.main()
