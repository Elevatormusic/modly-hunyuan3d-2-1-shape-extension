import os
import unittest
import tempfile
import json
import bpt_runner
from bpt_runner import provision


class TestProvisionReadiness(unittest.TestCase):
    def test_not_ready_without_sentinel(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(provision._is_ready(d, weight_bytes=1_636_512_878))

    def test_ready_with_matching_sentinel_and_weight(self):
        with tempfile.TemporaryDirectory() as d:
            py = provision.venv_python(d)
            os.makedirs(os.path.dirname(py))
            open(py, "wb").close()
            wpath = os.path.join(d, "weights", "bpt-8-16-500m.pt")
            os.makedirs(os.path.dirname(wpath))
            with open(wpath, "wb") as f:
                f.truncate(1_636_512_878)
            with open(os.path.join(d, "provisioned.json"), "w") as fh:
                json.dump({"weight_bytes": 1_636_512_878}, fh)
            self.assertTrue(provision._is_ready(d, weight_bytes=1_636_512_878))

    def test_not_ready_with_short_weight(self):
        with tempfile.TemporaryDirectory() as d:
            py = provision.venv_python(d)
            os.makedirs(os.path.dirname(py))
            open(py, "wb").close()
            wpath = os.path.join(d, "weights", "bpt-8-16-500m.pt")
            os.makedirs(os.path.dirname(wpath))
            with open(wpath, "wb") as f:
                f.truncate(1000)  # half-downloaded
            with open(os.path.join(d, "provisioned.json"), "w") as fh:
                json.dump({"weight_bytes": 1_636_512_878}, fh)
            self.assertFalse(provision._is_ready(d, weight_bytes=1_636_512_878))


class TestEnvSanitize(unittest.TestCase):
    def test_sanitized_env_drops_python_path_vars(self):
        os.environ["PYTHONPATH"] = "C:/poison"
        os.environ["PYTHONHOME"] = "C:/poison"
        try:
            env = bpt_runner._child_env()
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("PYTHONHOME", env)
            self.assertNotIn("PYTHONSTARTUP", env)
        finally:
            os.environ.pop("PYTHONPATH", None)
            os.environ.pop("PYTHONHOME", None)


if __name__ == "__main__":
    unittest.main()
