"""Lazily provision the isolated BPT sub-venv + weights (first `bpt` use).

Everything BPT needs (torch cu128 + x-transformers pins that conflict with the paint
stack's numpy 2.x) lives in bpt_runner/venv, invoked as a subprocess. Weights are
downloaded to bpt_runner/weights. Both are gitignored and multi-GB.
"""
from __future__ import annotations
import os
import sys
import json
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(_HERE, "venv")
WEIGHTS = os.path.join(_HERE, "weights", "bpt-8-16-500m.pt")
SENTINEL = os.path.join(_HERE, "provisioned.json")
WEIGHT_BYTES = 1_636_512_878
# Base interpreter used to create the isolated BPT venv. Derived from the running
# extension Python so provisioning works on any machine (was hard-coded to one
# developer's path before, which broke BPT for every other user).
BASE_PY = sys.executable
DEPS = ["trimesh", "pyyaml", "numpy==1.26.4", "tqdm", "einops",
        "x-transformers==1.26.6", "beartype", "omegaconf", "networkx", "scipy",
        "scikit-image", "transformers", "pytorch-custom-utils==0.0.21", "six"]


def venv_python(root=_HERE):
    sub = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    return os.path.join(root, "venv", *sub)


def _is_ready(root=_HERE, weight_bytes=WEIGHT_BYTES):
    py = venv_python(root)
    w = os.path.join(root, "weights", "bpt-8-16-500m.pt")
    s = os.path.join(root, "provisioned.json")
    if not (os.path.exists(py) and os.path.exists(w) and os.path.exists(s)):
        return False
    try:
        with open(s) as fh:
            meta = json.load(fh)
        return os.path.getsize(w) >= weight_bytes and meta.get("weight_bytes") == weight_bytes
    except Exception:
        return False


def ensure_provisioned(progress_cb=None):
    if _is_ready():
        return True

    def _say(msg):
        print(f"[bpt_runner] {msg}")
        if progress_cb:
            try:
                progress_cb(msg)
            except Exception:
                pass

    try:
        if not os.path.exists(BASE_PY):
            raise RuntimeError(f"base python for provisioning not found: {BASE_PY}")
        _say("Creating BPT environment…")
        subprocess.run([BASE_PY, "-m", "venv", VENV_DIR], check=True)
        py = venv_python()
        _say("Installing BPT torch (cu128, ~2.5 GB)…")
        subprocess.run([py, "-m", "pip", "install", "-q", "--index-url",
                        "https://download.pytorch.org/whl/cu128", "torch"], check=True)
        subprocess.run([py, "-m", "pip", "install", "-q", *DEPS, "huggingface_hub"], check=True)
        _say("Downloading BPT weights (1.6 GB)…")
        os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
        subprocess.run([py, "-c",
            "from huggingface_hub import hf_hub_download;"
            f"hf_hub_download('whaohan/bpt','bpt-8-16-500m.pt', local_dir=r'{os.path.dirname(WEIGHTS)}')"],
            check=True)
        # verify CUDA in the sub-venv before declaring ready (fallback triggers on False)
        r = subprocess.run([py, "-c", "import torch;print(torch.cuda.is_available())"],
                           capture_output=True, text=True)
        if "True" not in r.stdout:
            raise RuntimeError(f"sub-venv CUDA not available: {r.stdout} {r.stderr}")
        if os.path.getsize(WEIGHTS) < WEIGHT_BYTES:
            raise RuntimeError("weight download incomplete")
        with open(SENTINEL, "w") as fh:
            json.dump({"weight_bytes": WEIGHT_BYTES}, fh)
        _say("BPT ready.")
        return True
    except Exception as exc:
        print(f"[bpt_runner] provisioning failed ({exc})")
        return False
