# modly-hunyuan3d-2-1-shape-extension

Modly extension for the **full** [Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) model — Tencent's high-fidelity image-to-3D model. This wraps the large `hunyuan3d-dit-v2-1` shape checkpoint (3.3B params, per Tencent's Hugging Face model table), not the lightweight Mini variant, with an **optional PBR texture pass**.

**Shape by default; textures optional.** Geometry generates on ~10 GB VRAM. The PBR texture pass is a heavier opt-in (~21 GB VRAM + a build toolchain) — see Textures below.

## What this extension does

- installs an isolated Python venv with the 2.1 shape-inference dependency set
- downloads the `hunyuan3d-dit-v2-1` weights (~7.4 GB) from Hugging Face
- fetches the `hy3dshape` source from the Hunyuan3D-2.1 GitHub repo
- loads the pipeline **offline** from the local `config.yaml` + `model.fp16.ckpt`
- exports a `.glb` mesh (optionally decimated for CAD / 3D-print workflows)
- optionally paints **PBR textures** (shape → paint, run sequentially)

## Textures (optional PBR pass)

Turn on **Generate textures (PBR)**. The extension runs shape first, frees the shape model from VRAM, then runs the `hy3dpaint` paint pipeline on the mesh, and finally reloads the shape model for the next run. Output is a textured `.glb` with albedo/metalness/roughness maps.

**Prerequisites (texture pass only):**
- **~21 GB VRAM** for the paint stage (fits a 24 GB card because stages run sequentially).
- A **C++/CUDA build toolchain**: Visual Studio C++ Build Tools + a CUDA toolkit (`nvcc`) matching your PyTorch CUDA (12.4). The paint pipeline builds two native modules on first use — `custom_rasterizer` (CUDA) and `mesh_inpaint_processor` (C++).
- A large **first-run download**: `hunyuan3d-paintpbr-v2-1` weights, DINOv2-giant, and a RealESRGAN checkpoint.

If the toolchain is missing, texturing fails with a clear error — **shape generation is unaffected**. Textures are **discarded** when converting to a STEP solid or Fusion Form (CAD has no surface textures); use them for renders, game assets, previews.

## How it differs from the Hunyuan3D-2 Mini extension

| | Mini | This (2.1 Full) |
|---|---|---|
| Weights | `model.fp16.safetensors` | `config.yaml` + `model.fp16.ckpt` |
| Code package | `hy3dgen` (Hunyuan3D-2 repo) | `hy3dshape` (Hunyuan3D-2.1 repo) |
| Loader | `from_pretrained(subfolder=…)` | `from_single_file(ckpt, config)` |
| Shape VRAM | ~6 GB | ~10 GB |

## Requirements

- NVIDIA GPU with **≥ 10 GB VRAM** for the shape stage (an RTX 3090 / 24 GB is comfortable)
- ~10 GB free disk for weights + source
- Windows or Linux (CUDA). macOS/MPS falls back to fp32 and is slow/untested for the full model.

## Install

**From GitHub** (recommended — builds the venv automatically):
1. Modly: **Extensions → Install from GitHub**, paste this repo's URL.
2. Click **Download** on the model variant.
3. Select the model, drop in an image, generate.

**From a local folder** (link):
1. Modly: **Extensions → Install from local folder**, pick this folder.
2. Linking does NOT build the Python environment. Click **Repair** on the extension to run `setup.py` and create the `venv` (this pulls PyTorch, Pillow, etc. — a large download).
3. Until the venv exists, the registry may log `No module named 'PIL'` and mark the extension as needing setup — that is the missing-venv symptom, resolved by Repair.
4. Then click **Download** on the model variant (~7.4 GB weights), and generate.

## Parameters

- **Quality** — diffusion steps (30 / 50 / 75)
- **Mesh Resolution** — octree resolution (256 / 384 / 512)
- **Guidance Scale** — how closely the mesh follows the input image
- **Decimate (for CAD/print)** — optional polygon reduction on export
- **Seed** — reproducibility
- **Generate textures (PBR)** — enable the paint pass (see prerequisites above)
- **Texture view resolution** — 512 / 768 per-view render size
- **Texture views** — number of camera views painted/baked (6–9)

## Upstream sources

- Weights: `tencent/Hunyuan3D-2.1` (subfolder `hunyuan3d-dit-v2-1`)
- Source: `Tencent-Hunyuan/Hunyuan3D-2.1` (package `hy3dshape`)

Model weights are under Tencent's `tencent-hunyuan-community` license — review it before commercial use.
