import os, tempfile, unittest
from pathlib import Path
from unittest import mock
import trimesh
import retopo


class TestRetopo(unittest.TestCase):
    def test_fallback_when_exe_absent(self):
        # Force the exe absent (it's vendored now) so this genuinely exercises
        # the pymeshlab / quadric fallback and still returns a non-empty mesh.
        m = trimesh.creation.icosphere(subdivisions=4)  # ~5120 faces
        with mock.patch.object(retopo, "_EXE", Path("/no/such/Instant Meshes.exe")):
            out = retopo.retopo_quads(m, 800)
        self.assertGreater(len(out.faces), 0)

    def test_never_raises_all_fail(self):
        with mock.patch.object(retopo, "_instant_meshes", side_effect=RuntimeError), \
             mock.patch.object(retopo, "_pymeshlab_quad", side_effect=RuntimeError):
            out = retopo.retopo_quads(trimesh.creation.box(), 100)
            self.assertGreater(len(out.faces), 0)  # quadric / raw last resort

    def test_instant_meshes_target_and_flags(self):
        captured = {}

        def fake_run(cmd, **kw):
            captured["cmd"] = list(cmd)
            oi = cmd.index("-o")
            import shutil
            shutil.copyfile(cmd[1], cmd[oi + 1])  # emulate IM writing an OBJ
            return mock.Mock(returncode=0)

        with tempfile.TemporaryDirectory() as d:
            fake_exe = os.path.join(d, "Instant Meshes.exe")
            open(fake_exe, "w").close()
            with mock.patch.object(retopo, "_EXE", Path(fake_exe)), \
                 mock.patch.object(retopo.subprocess, "run", side_effect=fake_run):
                retopo.retopo_quads(trimesh.creation.box(), 3000)
        cmd = captured["cmd"]
        # -v is aimed at ~target/8 (IM's soft-target ~8x overshoot); -d deterministic
        self.assertEqual(cmd[cmd.index("-v") + 1], str(round(3000 / 8)))  # 375
        self.assertIn("-d", cmd)


if __name__ == "__main__":
    unittest.main()
