"""BPT neural retopology via an isolated sub-venv subprocess.

`retopo(dense)` returns a clean ~4k-face Trimesh, or None on any failure so the
caller (mesh_cleanup) can fall back to isotropic/quadric.
"""
from __future__ import annotations
import os
import tempfile
import subprocess
import trimesh
from . import provision


def _child_env():
    # Absolute-path launch of the sub-venv python + a sanitized env so Modly's injected
    # PYTHONPATH/HOME can't shadow the sub-venv's packages (VIRTUAL_ENV is ignored anyway).
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env.pop("VIRTUAL_ENV", None)
    return env


def is_provisioned():
    return provision._is_ready()


def ensure_provisioned(progress_cb=None):
    return provision.ensure_provisioned(progress_cb)


def retopo(dense, temperature=0.5, progress_cb=None):
    """dense Trimesh -> clean ~4k-face Trimesh, or None on any failure."""
    if not ensure_provisioned(progress_cb):
        print("[bpt_runner] not provisioned; caller should fall back")
        return None
    py = provision.venv_python()
    infer = os.path.join(os.path.dirname(os.path.abspath(__file__)), "infer.py")
    try:
        with tempfile.TemporaryDirectory() as d:
            in_glb = os.path.join(d, "dense.glb")
            out_obj = os.path.join(d, "retopo.obj")
            dense.export(in_glb)
            r = subprocess.run(
                [py, infer, "--input", in_glb, "--weights", provision.WEIGHTS,
                 "--output", out_obj, "--temperature", str(temperature)],
                capture_output=True, text=True, env=_child_env(),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if r.returncode != 0 or not os.path.exists(out_obj):
                print(f"[bpt_runner] infer failed rc={r.returncode}: {r.stderr[-500:]}")
                return None
            low = trimesh.load(out_obj, force="mesh", process=False)
            comps = low.split(only_watertight=False)
            if len(comps) > 1:
                low = max(comps, key=lambda c: len(c.faces))
            return low
    except Exception as exc:
        print(f"[bpt_runner] retopo error ({exc})")
        return None
