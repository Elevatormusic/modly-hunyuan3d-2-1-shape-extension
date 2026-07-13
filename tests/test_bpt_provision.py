import os, sys, unittest


class TestProvision(unittest.TestCase):
    def _mod(self):
        from bpt_runner import provision
        return provision

    def test_base_py_derived_from_interpreter(self):
        # Was hard-coded to one developer's absolute path (broke BPT for every
        # other user); now derived from the running interpreter.
        self.assertEqual(self._mod().BASE_PY, sys.executable)

    def test_venv_python_cross_platform(self):
        p = self._mod().venv_python(os.path.join("some", "root"))
        if os.name == "nt":
            self.assertTrue(p.endswith(os.path.join("Scripts", "python.exe")))
        else:
            self.assertTrue(p.endswith(os.path.join("bin", "python")))


if __name__ == "__main__":
    unittest.main()
