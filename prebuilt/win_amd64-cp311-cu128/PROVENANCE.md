# Prebuilt native modules — `win_amd64-cp311-cu128`

Prebuilt binaries for the Hunyuan3D-2.1 paint pass's two native modules, so
modern-driver Windows users can run the PBR texture stage with **no C++/CUDA
build toolchain**. The runtime (`generator._try_prebuilt_extensions`) prefers
these over the lazy source build **only** when the ABI matches exactly:

- OS: `win32` (Windows x64)
- Python: CPython **3.11** (`cp311`) — Modly's embedded interpreter
- torch: **2.7.0+cu128**

Anything else (cu124/torch-2.6, cu118, Linux, ROCm, other Python, any ABI
mismatch, any prebuilt failure) falls back to today's automatic source build —
nobody is worse off.

## Artifacts

1. **`custom_rasterizer-0.1-cp311-cp311-win_amd64.whl`** — a `pip wheel` of the
   vendored `custom_rasterizer` source. The wheel carries BOTH the compiled
   `custom_rasterizer_kernel.cp311-win_amd64.pyd` AND the pure-Python wrapper
   package (`custom_rasterizer/__init__.py` + `render.py`) plus its dist-info.
   Shipping only the kernel `.pyd` silently breaks: the source-tree wrapper is an
   empty namespace package, `import custom_rasterizer` succeeds with no
   `rasterize` attribute, the build is skipped, and paint later dies with
   `custom_rasterizer has no attribute 'rasterize'`. Installing the wheel places
   the real wrapper in site-packages where it wins.
2. **`mesh_inpaint_processor.cp311-win_amd64.pyd`** — a single self-contained
   pybind11 module. It is dropped into the paint source's `DifferentiableRenderer/`
   directory, from which `MeshRender` imports it relatively.

## Checksums (sha256)

```
21412851cb03989a27983cdad73cacb3177e564249b0aca7b38c41cdcbd2d815  custom_rasterizer-0.1-cp311-cp311-win_amd64.whl
1849382c687f4b11d7baa5104a0e22c433f9aa2c3a2f40fcbb94e0960f221dfc  mesh_inpaint_processor.cp311-win_amd64.pyd
```

The runtime recomputes these and refuses the prebuilt path on any mismatch
(corruption → clean fallback, never a crash).

## Build inputs (verified, built 2026-07-09 on the dev machine)

- **CPython:** 3.11 (`cp311`).
- **torch:** 2.7.0+cu128 (the wheel/pyd link against this exact torch ABI).
- **MSVC toolset:** 14.44.35207 (VS2022 BuildTools), activated via that
  toolset's `vcvarsall.bat x64 -vcvars_ver=14.44`, with `DISTUTILS_USE_SDK=1`
  and `NVCC_PREPEND_FLAGS=-allow-unsupported-compiler` (the "VS 14.51 is too new
  for the CUDA 12.4 front-end `cudafe++`" workaround).
- **CUDA toolkit:** 12.4.131 (`nvcc -V` → V12.4.131).
- **`TORCH_CUDA_ARCH_LIST`:** `7.5;8.0;8.6;8.9;9.0+PTX` — native SASS for
  Turing (sm_75) → Hopper (sm_90) plus forward-JIT PTX for `sm_90`.
- **Verified embedded code** (via `cuobjdump --list-elf` / `--list-ptx` on the
  kernel): SASS `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_90` + PTX `sm_90`.
- **Source tree:** the vendored Hunyuan3D-2.1 `custom_rasterizer` tree, built
  from the `custom_rasterizer_kernel_for_windows` kernel sources (see modified
  files below), with the `rasterizer_gpu.cu` `z_min` dtype fix applied.

### GPU coverage / limitations

- **Turing → Hopper (sm_75..sm_90):** run on native SASS, no JIT.
- **Blackwell (sm_100 / sm_120):** run via a one-time PTX→SASS JIT of the
  embedded `sm_90` PTX (first-call latency, then cached by the driver). Native
  `sm_100`/`sm_120` SASS is **not** embedded because that requires a CUDA **12.8**
  toolkit; this build box has CUDA 12.4, which caps native codegen at `sm_90`.
  Acceptable for v1 — noted here as a future-rebuild item.
- **Pre-Turing (< sm_75):** can never use the prebuilt (no native SASS, and PTX
  cannot JIT *down* to an older arch). The ABI gate does not check compute
  capability — if such a GPU somehow reaches this path, the final `import` /
  runtime call fails and the transactional rollback restores pristine state so
  the source-build fallback runs exactly as before.

## Modified vendored files (vs pristine upstream)

- `custom_rasterizer/setup.py` — repointed from `lib/custom_rasterizer_kernel/`
  to `lib/custom_rasterizer_kernel_for_windows/` (MSVC-safe kernel sources; the
  Linux sources trip C2398 narrowing and use 32-bit `long` pointers).
- `custom_rasterizer_kernel_for_windows/rasterizer_gpu.cu` — `z_min` is a
  `kInt64` (Long) tensor read with `.data_ptr<uint64_t>()`, which torch rejects
  at run time; changed to `.data_ptr<int64_t>()` (the existing `(uint64_t*)`
  cast still reinterprets the bits, matching the CPU path in the same file).

## Rebuild instructions (e.g. for the next torch bump)

Do this in the extension venv (cp311, with the target torch + a matching CUDA
toolkit and MSVC toolset installed):

1. **Clean first** — remove any stale `build/` under `custom_rasterizer/`
   (old objects, e.g. a single-arch `sm_86` build, will otherwise be reused and
   silently narrow the arch coverage).
2. Apply the two source patches above (the extension does this automatically via
   `generator._patch_custom_rasterizer_sources`).
3. Open the target MSVC toolset shell and set the env:
   ```
   call "<VS>\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.44
   set DISTUTILS_USE_SDK=1
   set NVCC_PREPEND_FLAGS=-allow-unsupported-compiler
   set TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9;9.0+PTX
   ```
   (add native `sm_100`/`sm_120` to the arch list only on a CUDA >= 12.8 box.)
4. Build the wheel (torch is imported by the setup, so no build isolation):
   ```
   <venv>\Scripts\python.exe -m pip wheel . --no-build-isolation -w <out>
   ```
5. Build the inpaint module in `DifferentiableRenderer/` with torch's
   `CppExtension` (`build_ext --inplace`) in the SAME pipeline run so its torch
   provenance matches reality.
6. `cuobjdump --list-elf` / `--list-ptx` the kernel to confirm the embedded arch
   list, then recompute both sha256s and update this file's Checksums block.

--------------------------------------------------------------------------------

# `diso` — Dual Marching Cubes (DMC) surface extraction

A **separately-licensed** prebuilt shipped in the same ABI bundle so DMC surface
extraction (Hunyuan's intended, smoother-contact-line extractor) runs with **no
C++/CUDA build toolchain**. The runtime (`generator._ensure_diso`) installs this
wheel at `load()` — before shape extraction — under the same exact ABI gate
(`win32` + `cp311` + torch `2.7.0+cu128`) and sha256 check; any mismatch/failure
falls back to stock scikit-image `mc`, so nobody is worse off.

> **License notice — this artifact is NOT under the Hunyuan license.** `diso` is
> **CC BY-NC 4.0** (Creative Commons Attribution-**NonCommercial** 4.0
> International), © 2023 Xinyue Wei (UCSD). It is redistributed **unmodified**;
> the verbatim license text is shipped alongside as `diso.LICENSE`. Do not
> relicense. The NonCommercial restriction passes through to any downstream use.

## Artifact

- **`diso-0.1.4-cp311-cp311-win_amd64.whl`** — a stock `pip wheel` of the
  **unmodified** upstream `diso` 0.1.4 source distribution (no source patches).
  It provides `from diso import DiffDMC`, the differentiable Dual Marching Cubes
  kernel the shape pipeline uses when `mc_algo="dmc"`.

- **version:** 0.1.4
- **source:** github.com/SarahWeiii/diso (`diso` 0.1.4 sdist, unmodified)
- **license:** CC BY-NC 4.0 (see `diso.LICENSE`, shipped verbatim)

## Checksum (sha256)

```
e060f36edf5b79fd4be8d6db4c782e7729ebbf838f2ae78f07321424baf093fa  diso-0.1.4-cp311-cp311-win_amd64.whl
```

The runtime recomputes this and refuses the prebuilt path on any mismatch
(corruption → clean fallback to `mc`, never a crash).

## Build inputs (verified, built 2026-07-10 on the dev machine)

- **CPython:** 3.11 (`cp311`).
- **torch:** 2.7.0+cu128 (the wheel links against this exact torch ABI).
- **MSVC toolset:** 14.44 (VS2022 BuildTools) via `vcvarsall.bat x64
  -vcvars_ver=14.44`, `DISTUTILS_USE_SDK=1`, and
  `NVCC_PREPEND_FLAGS=-allow-unsupported-compiler`.
- **CUDA toolkit:** 12.4 (nvcc front-end vs the torch cu128 runtime ABI).
- **`TORCH_CUDA_ARCH_LIST`:** `7.5;8.0;8.6;8.9;9.0+PTX` — Turing (sm_75) →
  Hopper (sm_90) native SASS + forward-JIT PTX for `sm_90` (same coverage /
  Blackwell-JIT note as the paint modules above).
- **Source tree:** unmodified upstream `diso` 0.1.4 sdist (no patches applied).

To rebuild on a torch bump: reinstall the target torch, open the 14.44 toolset
shell with the env above, `pip wheel diso==0.1.4 --no-build-isolation -w <out>`,
then recompute the sha256 and update this Checksum block.
