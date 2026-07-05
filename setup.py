"""
Hunyuan3D 2.1 Shape (Full) — extension setup script.

Creates an isolated venv and installs all required dependencies for the 2.1
SHAPE pipeline (no texture/paint deps). Called by Modly at install time with:

    python setup.py <json_args>

where json_args contains:
    python_exe   — path to Modly's embedded Python (used to create the venv)
    ext_dir      — absolute path to this extension directory
    torch_flavor — torch flavor (cuda, rocm; defaults to cuda)
    gpu_sm       — GPU compute capability as integer (e.g. 86 for Ampere/RTX 3090; 0 on macOS)
    cuda_version — CUDA major/minor encoded as integer (e.g. 124, 128)
    accelerator  — "mps" | "cuda" | "cpu"
    platform     — Electron's process.platform string ("win32", "darwin", "linux")

Torch-selection and ARM64 wheel logic is kept identical to the reference
Hunyuan3D-2 Mini extension so behaviour matches across the model rail.
"""
import json
import platform
import subprocess
import sys
from pathlib import Path


ARM64_CU124_WHEELS = {
    "cp310": {
        "torch":       "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp310-cp310-linux_aarch64.whl#sha256=d468d0eddc188aa3c1e417ec24ce615c48c0c3f592b0354d9d3b99837ef5faa6",
        "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp310-cp310-linux_aarch64.whl#sha256=38765e53653f93e529e329755992ddbea81091aacedb61ed053f6a14efb289e5",
    },
    "cp311": {
        "torch":       "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp311-cp311-linux_aarch64.whl#sha256=e080353c245b752cd84122e4656261eee6d4323a37cfb7d13e0fffd847bae1a3",
        "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp311-cp311-linux_aarch64.whl#sha256=2c5350a08abe005a16c316ae961207a409d0e35df86240db5f77ec41345c82f3",
    },
    "cp312": {
        "torch":       "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp312-cp312-linux_aarch64.whl#sha256=302041d457ee169fd925b53da283c13365c6de75c6bb3e84130774b10e2fbb39",
        "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp312-cp312-linux_aarch64.whl#sha256=3e3289e53d0cb5d1b7f55b3f5912f46a08293c6791585ba2fc32c12cded9f9af",
    },
    "cp39": {
        "torch":       "https://download-r2.pytorch.org/whl/cu124/torch-2.5.1-cp39-cp39-linux_aarch64.whl#sha256=012887a6190e562cb266d2210052c5deb5113f520a46dc2beaa57d76144a0e9b",
        "torchvision": "https://download-r2.pytorch.org/whl/cu124/torchvision-0.20.1-cp39-cp39-linux_aarch64.whl#sha256=e25b4ac3c9eec3f789f1c5491331dfe236b5f06a1f406ea82fa59fed4fc6f71e",
    },
}


ARM64_CU128_WHEELS = {
    "cp310": {
        "torch":       "https://download-r2.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp310-cp310-manylinux_2_28_aarch64.whl#sha256=b1f0cdd0720ad60536deb5baa427b782fd920dd4fcf72e244d32974caafa3b9e",
        "torchvision": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.0-cp310-cp310-manylinux_2_28_aarch64.whl#sha256=566224d7b4f00bc6366bed1d62f834ca80f8e57fe41e10e4a5636bfa3ffb984e",
    },
    "cp311": {
        "torch":       "https://download-r2.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_aarch64.whl#sha256=47c895bcab508769d129d717a4b916b10225ae3855723aeec8dff8efe5346207",
        "torchvision": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.0-cp311-cp311-manylinux_2_28_aarch64.whl#sha256=6be714bcdd8849549571f6acfaa2dfa9e00676f042bda517432745fb116f7904",
    },
    "cp312": {
        "torch":       "https://download-r2.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=6bba7dca5d9a729f1e8e9befb98055498e551efaf5ed034824c168b560afc1ac",
        "torchvision": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.0-cp312-cp312-manylinux_2_28_aarch64.whl#sha256=6e9752b48c1cdd7f6428bcd30c3d198b30ecea348d16afb651f95035e5252506",
    },
    "cp313": {
        "torch":       "https://download-r2.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp313-cp313-manylinux_2_28_aarch64.whl#sha256=633f35e8b1b1f640ef5f8a98dbd84f19b548222ce7ba8f017fe47ce6badc106a",
        "torchvision": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.0-cp313-cp313-manylinux_2_28_aarch64.whl#sha256=e4d4d5a14225875d9bf8c5221d43d8be97786adc498659493799bdeff52c54cf",
    },
    "cp39": {
        "torch":       "https://download-r2.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp39-cp39-manylinux_2_28_aarch64.whl#sha256=2f155388b1200e08f3e901bb3487ff93ca6d63cde87c29b97bb6762a8f63b373",
        "torchvision": "https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.0-cp39-cp39-manylinux_2_28_aarch64.whl#sha256=7a398fad02f4ac6b7d18bea9a08dc14163ffc5a368618f29ceb0e53dfa91f69e",
    },
}


def pip(venv: Path, *args: str) -> None:
    is_win = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def python_tag(venv: Path) -> str:
    is_win = platform.system() == "Windows"
    python_exe = venv / ("Scripts/python.exe" if is_win else "bin/python")
    return subprocess.check_output(
        [str(python_exe), "-c", "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"],
        text=True,
    ).strip()


def install_arm64_pytorch(venv: Path, wheel_map: dict, label: str, extra_index_url: str) -> None:
    py_tag = python_tag(venv)
    wheel_urls = wheel_map.get(py_tag)
    if wheel_urls is None:
        raise RuntimeError(f"Unsupported Python version for Linux ARM64 PyTorch wheels: {py_tag}")
    print(f"[setup] Installing pinned ARM64 wheels for {label} ({py_tag}) …")
    pip(
        venv, "install", "--retries", "10", "--timeout", "120", "--no-cache-dir",
        "--extra-index-url", extra_index_url,
        wheel_urls["torch"], wheel_urls["torchvision"],
    )


def setup(
    python_exe:    str,
    ext_dir:       Path,
    gpu_sm:        int,
    cuda_version:  int = 0,
    torch_flavor:  str = "cuda",
    accelerator:   str = "",
    platform_name: str = "",
) -> None:
    venv = ext_dir / "venv"
    machine = platform.machine().lower()
    is_win = platform.system() == "Windows"
    is_mac = platform.system() == "Darwin" or platform_name == "darwin"
    is_linux_arm64 = platform.system() == "Linux" and machine in {"aarch64", "arm64"}

    if not accelerator:
        if is_mac:
            accelerator = "mps" if machine == "arm64" else "cpu"
        elif gpu_sm > 0:
            accelerator = "cuda"
        else:
            accelerator = "cpu"

    print(f"[setup] accelerator={accelerator}  gpu_sm={gpu_sm}  cuda_version={cuda_version}")
    print(f"[setup] Creating venv at {venv} …")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # ------------------------------------------------------------------ #
    # PyTorch — choose build based on GPU flavor / architecture
    # ------------------------------------------------------------------ #
    if is_mac:
        label = "Apple Silicon/MPS" if accelerator == "mps" else "Intel/CPU"
        print(f"[setup] macOS ({label}) -> PyTorch from standard PyPI")
        pip(venv, "install", "torch", "torchvision")
    elif torch_flavor == "rocm":
        if is_win:
            print("[setup] WARNING: ROCm is not supported on Windows. Falling back to CPU PyTorch.")
            torch_index = "https://download.pytorch.org/whl/cpu"
            torch_pkgs  = ["torch==2.6.0", "torchvision==0.21.0"]
        else:
            torch_index = "https://download.pytorch.org/whl/rocm7.2"
            torch_pkgs  = ["torch", "torchvision"]
            print("[setup] -> PyTorch + ROCm 7.2")
        pip(venv, "install", *torch_pkgs, "--index-url", torch_index)
    elif is_linux_arm64 and (gpu_sm >= 100 or cuda_version >= 128):
        install_arm64_pytorch(venv, ARM64_CU128_WHEELS, "cu128", "https://download.pytorch.org/whl/cu128")
    elif is_linux_arm64 and gpu_sm >= 70:
        install_arm64_pytorch(venv, ARM64_CU124_WHEELS, "cu124", "https://download.pytorch.org/whl/cu124")
    elif gpu_sm >= 100 or cuda_version >= 128:
        print(f"[setup] GPU SM {gpu_sm}, CUDA {cuda_version} -> PyTorch 2.7 + CUDA 12.8")
        pip(venv, "install", "torch==2.7.0", "torchvision==0.22.0",
            "--index-url", "https://download.pytorch.org/whl/cu128")
    elif gpu_sm >= 70:
        # RTX 3090 (Ampere, SM 86) lands here -> CUDA 12.4 build.
        print(f"[setup] GPU SM {gpu_sm} -> PyTorch 2.6 + CUDA 12.4")
        pip(venv, "install", "torch==2.6.0", "torchvision==0.21.0",
            "--index-url", "https://download.pytorch.org/whl/cu124")
    else:
        print(f"[setup] GPU SM {gpu_sm} (legacy) -> PyTorch 2.5 + CUDA 11.8")
        pip(venv, "install", "torch==2.5.1", "torchvision==0.20.1",
            "--index-url", "https://download.pytorch.org/whl/cu118")

    # ------------------------------------------------------------------ #
    # Core dependencies — Hunyuan3D 2.1 SHAPE inference subset.
    # (Texture/paint deps such as realesrgan, basicsr, xatlas, open3d,
    #  cupy, bpy, deepspeed are intentionally omitted — shape-only.)
    # Versions follow the upstream 2.1 requirements.txt where it matters.
    # ------------------------------------------------------------------ #
    print("[setup] Installing core dependencies …")
    pip(venv, "install",
        "diffusers==0.30.0",
        "transformers==4.46.0",
        "accelerate==1.1.1",
        "pytorch-lightning==1.9.5",
        "torchmetrics==1.6.0",
        "huggingface_hub==0.30.2",
        "safetensors",
        "omegaconf==2.3.0",
        "pyyaml",
        "einops==0.8.0",
        "torchdiffeq",
        "timm",                  # required: denoiser (moe_layers) imports timm blocks
        "peft",                  # lazily referenced by the flow-matching module
        "numpy==1.26.4",
        "scipy",
        "scikit-image",
        "opencv-python-headless",
        "imageio",
        "trimesh==4.4.7",
        "pymeshlab",
        "pygltflib",
        "fast-simplification",   # backs trimesh.simplify_quadric_decimation for the decimate option
        "tqdm",
        "psutil",                # shared-GPU-memory planner reads system RAM (capacity/generator)
    )

    # ------------------------------------------------------------------ #
    # rembg (background removal)
    # ------------------------------------------------------------------ #
    # rembg only runs a ~1 s u2net matte, and generator._remove_background forces
    # the CPU execution provider — so it needs plain CPU onnxruntime, NOT
    # onnxruntime-gpu. rembg[gpu]'s wheels track a newer CUDA (13: cublasLt64_13)
    # than our pinned torch (cu128 / CUDA 12.8), so on the CUDA path it installed a
    # provider that can't load and silently fell back to CPU anyway — while also
    # co-installing onnxruntime + onnxruntime-gpu (a namespace clash). Plain CPU
    # rembg behaves identically on every machine and removes that whole class of
    # CUDA-mismatch breakage for all installers.
    print("[setup] Installing rembg (CPU onnxruntime) …")
    pip(venv, "install", "rembg")
    pip(venv, "install", "onnxruntime")

    # ------------------------------------------------------------------ #
    # Texture (paint) dependencies. Split into two groups so a failure in
    # the fragile packages cannot prevent the build tools from installing.
    # Native modules (custom_rasterizer, mesh_inpaint) are compiled lazily
    # on the first textured run using PyTorch's cpp_extension.
    # ------------------------------------------------------------------ #
    print("[setup] Installing texture build tools … (best-effort)")
    try:
        # ninja speeds torch extension builds; setuptools/wheel are needed to
        # compile. Reliable, and must not be blocked by the fragile deps below.
        pip(venv, "install", "ninja", "setuptools", "wheel")
    except Exception as exc:
        print(f"[setup] WARNING: build tools not installed ({exc}).")

    print("[setup] Installing texture runtime deps … (best-effort)")
    # NB: bpy (Blender) intentionally omitted — it's a huge, version-pinned build
    # that often fails to load; the pipeline is patched to use trimesh instead.
    for pkg in ("xatlas", "open3d", "realesrgan==0.3.0", "basicsr==1.4.2"):
        try:
            pip(venv, "install", pkg)
        except Exception as exc:
            print(f"[setup] WARNING: '{pkg}' not installed ({exc}). "
                  "Shape generation is unaffected; textures may need this later.")

    # basicsr/RealESRGAN import torchvision.transforms.functional_tensor, which
    # newer torchvision removed. Recreate it as a forwarding shim so they load.
    print("[setup] Applying torchvision functional_tensor shim …")
    is_win = platform.system() == "Windows"
    venv_py = venv / ("Scripts/python.exe" if is_win else "bin/python")
    shim_code = (
        "import os, torchvision\n"
        "p = os.path.join(os.path.dirname(torchvision.__file__), 'transforms', 'functional_tensor.py')\n"
        "open(p, 'w').write('from torchvision.transforms.functional import rgb_to_grayscale\\n')\n"
        "print('shim written to', p)\n"
    )
    try:
        subprocess.run([str(venv_py), "-c", shim_code], check=True)
    except Exception as exc:
        print(f"[setup] WARNING: functional_tensor shim not applied ({exc}).")

    print("[setup] Done. Venv ready at:", venv)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(
            python_exe   = sys.argv[1],
            ext_dir      = Path(sys.argv[2]),
            gpu_sm       = int(sys.argv[3]),
            cuda_version = int(sys.argv[4]) if len(sys.argv) >= 5 else 0,
            torch_flavor = sys.argv[5] if len(sys.argv) >= 6 else "cuda",
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe    = args["python_exe"],
            ext_dir       = Path(args["ext_dir"]),
            gpu_sm        = int(args["gpu_sm"]),
            cuda_version  = int(args.get("cuda_version", 0)),
            torch_flavor  = args.get("torch_flavor", "cuda"),
            accelerator   = args.get("accelerator", ""),
            platform_name = args.get("platform", ""),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version] [torch_flavor]")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":124}\'')
        sys.exit(1)
