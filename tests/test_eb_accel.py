import os
import unittest
import eb_accel


class TestResolveSrChunk(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("EB_SR_CHUNK", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["EB_SR_CHUNK"] = self._saved
        else:
            os.environ.pop("EB_SR_CHUNK", None)

    def test_explicit_chunk_wins(self):
        os.environ["EB_SR_CHUNK"] = "1"
        self.assertEqual(eb_accel._resolve_sr_chunk(3), 3)

    def test_env_used_when_none(self):
        os.environ["EB_SR_CHUNK"] = "2"
        self.assertEqual(eb_accel._resolve_sr_chunk(None), 2)

    def test_default_when_env_unset(self):
        self.assertEqual(eb_accel._resolve_sr_chunk(None), 4)

    def test_bad_env_falls_back_to_default(self):
        os.environ["EB_SR_CHUNK"] = "notanint"
        self.assertEqual(eb_accel._resolve_sr_chunk(None), 4)

    def test_floors_at_one(self):
        os.environ["EB_SR_CHUNK"] = "0"
        self.assertEqual(eb_accel._resolve_sr_chunk(None), 1)


if __name__ == "__main__":
    unittest.main()
