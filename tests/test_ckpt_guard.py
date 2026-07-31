import tempfile
import unittest
import zipfile
from pathlib import Path

import ckpt_guard


def _zip(path, comment=b""):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data.pkl", b"x" * 64)
    if comment:
        with zipfile.ZipFile(path, "a") as zf:
            zf.comment = comment
    return path


class _Tmp(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.d = Path(tmp.name)


class TestCkptOk(_Tmp):
    def test_missing_file(self):
        self.assertFalse(ckpt_guard.ckpt_ok(self.d / "nope.ckpt", min_bytes=1))

    def test_directory_is_not_a_ckpt(self):
        self.assertFalse(ckpt_guard.ckpt_ok(self.d, min_bytes=1))

    def test_lfs_pointer_stub_rejected(self):
        # `git clone` without git-lfs leaves a ~130 byte text pointer in place
        # of the weights, which torch then reports as a zip failure.
        p = self.d / "model.fp16.ckpt"
        p.write_bytes(b"version https://git-lfs.github.com/spec/v1\n" + b"o" * 90)
        self.assertFalse(ckpt_guard.ckpt_ok(p))                # fails the size floor
        self.assertFalse(ckpt_guard.ckpt_ok(p, min_bytes=1))   # and has no zip footer

    def test_truncated_zip_rejected(self):
        p = _zip(self.d / "m.ckpt")
        data = p.read_bytes()
        p.write_bytes(data[: len(data) // 2])   # central directory lives at the end
        self.assertFalse(ckpt_guard.ckpt_ok(p, min_bytes=1))

    def test_empty_file_rejected(self):
        p = self.d / "m.ckpt"
        p.write_bytes(b"")
        self.assertFalse(ckpt_guard.ckpt_ok(p, min_bytes=1))

    def test_complete_zip_accepted(self):
        self.assertTrue(ckpt_guard.ckpt_ok(_zip(self.d / "m.ckpt"), min_bytes=1))

    def test_zip_with_trailing_comment_accepted(self):
        p = _zip(self.d / "m.ckpt", comment=b"c" * 4096)
        self.assertTrue(ckpt_guard.ckpt_ok(p, min_bytes=1))

    def test_size_floor_rejects_short_but_valid_zip(self):
        self.assertFalse(ckpt_guard.ckpt_ok(_zip(self.d / "m.ckpt")))

    def test_default_floor_sits_below_the_published_checkpoint(self):
        # Published tencent/Hunyuan3D-2.1 model.fp16.ckpt is 7,366,389,768 bytes.
        self.assertLess(ckpt_guard.MIN_CKPT_BYTES, 7_366_389_768)
        self.assertGreater(ckpt_guard.MIN_CKPT_BYTES, 1_000_000_000)

    def test_footer_scan_ignores_a_signature_at_the_head(self):
        # Proves the scan window is bounded to the tail: a 7 GB checkpoint must
        # not be read end-to-end just to answer "is this complete?".
        p = self.d / "m.ckpt"
        with open(p, "wb") as fh:
            fh.write(ckpt_guard._EOCD_SIG)                     # decoy at offset 0
            fh.write(b"\0" * (ckpt_guard._EOCD_TAIL + 4096))
        self.assertFalse(ckpt_guard.has_zip_footer(p))


class TestLooksCorrupt(unittest.TestCase):
    def test_matches_the_reported_error(self):
        exc = RuntimeError("PytorchStreamReader failed reading zip archive: "
                           "failed finding central directory")
        self.assertTrue(ckpt_guard.looks_corrupt(exc))

    def test_matches_bad_zip_file(self):
        self.assertTrue(ckpt_guard.looks_corrupt(zipfile.BadZipFile("File is not a zip file")))

    def test_matches_torch_container_enforce_failure(self):
        # Captured on-device from torch 2.7 when the checkpoint was replaced by
        # an unrelated ZIP. Names neither the stream reader nor the directory.
        exc = RuntimeError("[enforce fail at inline_container.cc:176] . file in "
                           "archive is not in a subdirectory: notes.txt")
        self.assertTrue(ckpt_guard.looks_corrupt(exc))

    def test_matches_unpickling_stub(self):
        self.assertTrue(ckpt_guard.looks_corrupt(RuntimeError("invalid load key, '<'.")))

    def test_ignores_out_of_memory(self):
        self.assertFalse(ckpt_guard.looks_corrupt(
            RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")))

    def test_ignores_missing_dependency(self):
        self.assertFalse(ckpt_guard.looks_corrupt(ImportError("No module named 'hy3dshape'")))

    def test_ignores_missing_config_key(self):
        self.assertFalse(ckpt_guard.looks_corrupt(KeyError("scheduler")))


class TestPurge(_Tmp):
    def _seed(self):
        (self.d / "sub").mkdir()
        ckpt = self.d / "sub" / "model.fp16.ckpt"
        ckpt.write_bytes(b"broken")
        side = self.d / ".cache" / "huggingface" / "download" / "sub"
        side.mkdir(parents=True)
        (side / "model.fp16.ckpt.metadata").write_text("etag")
        (side / "model.fp16.ckpt.lock").write_text("")
        return ckpt, side

    def test_removes_file_and_hub_sidecars(self):
        ckpt, side = self._seed()
        removed = ckpt_guard.purge(self.d, "sub/model.fp16.ckpt")
        self.assertFalse(ckpt.exists())
        self.assertFalse((side / "model.fp16.ckpt.metadata").exists())
        self.assertFalse((side / "model.fp16.ckpt.lock").exists())
        self.assertEqual(len(removed), 3)

    def test_leaves_unrelated_files_alone(self):
        _, side = self._seed()
        keep = self.d / "sub" / "config.yaml"
        keep.write_text("a: 1")
        (side / "config.yaml.metadata").write_text("etag")
        ckpt_guard.purge(self.d, "sub/model.fp16.ckpt")
        self.assertTrue(keep.exists())
        self.assertTrue((side / "config.yaml.metadata").exists())

    def test_missing_paths_are_not_an_error(self):
        self.assertEqual(ckpt_guard.purge(self.d, "sub/model.fp16.ckpt"), [])


if __name__ == "__main__":
    unittest.main()
