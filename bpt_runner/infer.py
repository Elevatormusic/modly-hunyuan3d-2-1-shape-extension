"""BPT inference entry — executed by the sub-venv python. dense mesh -> retopo obj.

Carries the four validated Windows fixes: deepspeed stub (training-ckpt unpickle),
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, bf16 (fp16 overflow->NaN->multinomial assert),
and the vendored utils.sample_pc mesh-or-path patch.
"""
import sys
import os
import types
import importlib.abc
import importlib.machinery
import argparse


# --- deepspeed stub: the released checkpoint references ZeRO classes at unpickle ---
class _AnyDS:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): pass
    def __call__(self, *a, **k): return self
    def __reduce__(self): return (_AnyDS, ())
    def __getattr__(self, n): return _AnyDS()


class _Stub(types.ModuleType):
    def __getattr__(self, n): return _AnyDS


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fn, path, target=None):
        if fn == "deepspeed" or fn.startswith("deepspeed."):
            return importlib.machinery.ModuleSpec(fn, self)
        return None

    def create_module(self, spec): return _Stub(spec.name)

    def exec_module(self, m): pass


sys.meta_path.insert(0, _Finder())
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bpt_src")
sys.path.insert(0, SRC)
# The vendored miche encoder loads its config by a RELATIVE path
# ('miche/shapevae-256.yaml'), so BPT must run with _bpt_src as the working
# directory. Under Modly the cwd is the app's api dir, which broke this with a
# FileNotFoundError. All --input/--output/--weights paths we receive are absolute.
os.chdir(SRC)

import yaml  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
import numpy as np  # noqa: E402
from model.serializaiton import BPT_deserialize  # noqa: E402
from model.model import MeshTransformer  # noqa: E402
from utils import joint_filter, apply_normalize  # noqa: E402
from model.data_utils import to_mesh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--temperature", type=float, default=0.5)
    a = ap.parse_args()

    cfg = yaml.load(open(os.path.join(SRC, "config", "BPT-open-8k-8-16.yaml")),
                    Loader=yaml.FullLoader)
    model = MeshTransformer(
        dim=cfg["dim"], attn_depth=cfg["depth"], max_seq_len=cfg["max_seq_len"],
        dropout=cfg["dropout"], mode=cfg["mode"],
        num_discrete_coors=2 ** int(cfg["quant_bit"]),
        block_size=cfg["block_size"], offset_size=cfg["offset_size"],
        conditioned_on_pc=cfg["conditioned_on_pc"],
        use_special_block=cfg["use_special_block"],
        encoder_name=cfg["encoder_name"], encoder_freeze=cfg["encoder_freeze"])
    model.load(a.weights)
    model = model.eval().bfloat16().cuda()

    base = apply_normalize(trimesh.load(a.input, force="mesh"))
    pts, fidx = base.sample(50000, return_index=True)
    pcn = np.concatenate([pts, base.face_normals[fidx]], axis=-1).astype(np.float32)
    idx = np.random.default_rng(0).choice(pcn.shape[0], 4096, replace=False)
    pc = torch.from_numpy(pcn[idx])[None].cuda().bfloat16()

    with torch.no_grad():
        codes = model.generate(batch_size=1, temperature=a.temperature, pc=pc,
                               filter_logits_fn=joint_filter,
                               filter_kwargs=dict(k=50, p=0.95), return_codes=True)
    code = codes[0]
    code = code[code != model.pad_id].cpu().numpy()
    verts = BPT_deserialize(code, block_size=model.block_size,
                            offset_size=model.offset_size,
                            use_special_block=model.use_special_block)
    faces = torch.arange(1, len(verts) + 1).view(-1, 3)
    mesh = to_mesh(verts, faces, transpose=False, post_process=True)
    mesh.export(a.output)
    print(f"BPT_OK faces={len(mesh.faces)}")


if __name__ == "__main__":
    main()
