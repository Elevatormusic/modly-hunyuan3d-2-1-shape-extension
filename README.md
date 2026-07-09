# modly-hunyuan3d-2-1-shape-extension

Modly extension for the **full** [Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) model — Tencent's high-fidelity image-to-3D model. This wraps the large `hunyuan3d-dit-v2-1` shape checkpoint (3.3B params, per Tencent's Hugging Face model table), not the lightweight Mini variant, with an **optional PBR texture pass**.

**Shape by default; textures optional.** Geometry generates on ~10 GB VRAM. The PBR texture pass is a heavier opt-in (~21 GB VRAM + a build toolchain) — see Textures below.

## What this extension does

- installs an isolated Python venv with the 2.1 shape-inference dependency set
- downloads the `hunyuan3d-dit-v2-1` weights (~7.4 GB) from Hugging Face
- fetches the `hy3dshape` source from the Hunyuan3D-2.1 GitHub repo
- loads the pipeline **offline** from the local `config.yaml` + `model.fp16.ckpt`
- exports a `.glb` mesh (optionally decimated for CAD / 3D-print workflows)
- optionally paints **PBR textures** — albedo + packed metallic-roughness — with UV-seam reconciliation and an optional QA sheet

## Textures (optional PBR pass)

Turn on **Generate textures (PBR)**. The extension runs shape first, frees the shape model from VRAM, then runs the `hy3dpaint` paint pipeline on the mesh, and finally reloads the shape model for the next run. Output is a textured `.glb` with an albedo map and a packed metallic-roughness map.

**Prerequisites (texture pass only):**
- **~21 GB VRAM** for the paint stage (fits a 24 GB card because stages run sequentially). See **Texture memory** below to tune the ceiling.
- A **C++/CUDA build toolchain**: Visual Studio C++ Build Tools + a CUDA toolkit (`nvcc`) matching your PyTorch CUDA (12.4). The paint pipeline builds two native modules on first use — `custom_rasterizer` (CUDA) and `mesh_inpaint_processor` (C++).
- A large **first-run download**: `hunyuan3d-paintpbr-v2-1` weights, DINOv2-giant, and a RealESRGAN checkpoint.

If the toolchain is missing, texturing fails with a clear error — **shape generation is unaffected**. Textures are **discarded** when converting to a STEP solid or Fusion Form (CAD has no surface textures); use them for renders, game assets, previews.

> **Export GLB to keep the maps** — Modly's OBJ export drops textures.

### Mesh cleanup (texture pass)

The raw shape mesh is a dense marching-cubes surface (~2.6M faces at high resolution). Before painting it's cleaned into a UV-friendly base — you choose how with **Mesh cleanup**:

- **Regular** (default) — quadric decimation; keeps one connected shell with clean UVs. The most reliable choice.
- **Isotropic** — uniform-triangle remesh; even topology, but can shatter Hunyuan's non-watertight surfaces into many UV islands (auto-falls back to Regular).
- **BPT neural** — Tencent's [BPT](https://github.com/Tencent-Hunyuan/bpt) artist-topology retopology (~4k faces, cleanest edge flow). First use downloads ~4 GB into an isolated sub-environment and takes a few minutes per mesh; falls back to Regular if unavailable.

### Texture maps

The paint pass produces a standard glTF PBR set:

- **Albedo** (base color).
- **Metallic-roughness** — the baked metallic and roughness maps, packed into a single glTF-standard image (roughness in the green channel, metallic in blue) and wired into the GLB material.

**Fix texture seams** (on by default) reconciles color jumps across UV-island edges in the baked albedo and metallic-roughness, so chart boundaries don't show as hard color breaks, then dilates colour into the UV gutter so seams stay clean under mip-mapping. Turn it off for the raw bake.

### Normal-map bake (experimental, off by default)

**Bake normal map** transfers dense detail from the full-resolution mesh onto the clean base as a tangent-space normal map, so grooves and panel lines can survive cleanup as shading detail. It is **off by default and experimental**: on detailed / hard-surface meshes the current bake can introduce shading artifacts (a tangent-basis mismatch between the bake and the glTF viewer). A corrected high-quality bake is planned. On smooth subjects it adds little. Only applies when textures are on.

### QA debug sheet

**QA debug sheet** (on by default) writes a `*_qa.png` beside each textured GLB showing the albedo, metallic, roughness and normal maps, the UV layout, and mesh / texture stats. It's diagnostic only and doesn't change the model — useful for checking whether a colour that looks off is the texture itself or just your viewer's material preview.

### Texture memory & shared GPU memory

**Texture memory** caps the paint pass's VRAM so it can't silently spill into system RAM and crawl:

- **Low** — smallest / softest.
- **Balanced** (default) — targets a ~20 GB peak on 24 GB cards.
- **High** — sharpest; wants an otherwise-empty GPU.
- **Max** — 4096 texture; may need shared GPU memory.

The cap is adaptive: on a busy GPU the pass may step down a tier to fit. **Use shared GPU memory** lets High / Max run past your VRAM by paging to system RAM over PCIe — much slower (tens of minutes) and needs a large Windows page file. Leave it off unless you want maximum quality and don't mind the wait.

## How it differs from the Hunyuan3D-2 Mini extension

| | Mini | This (2.1 Full) |
|---|---|---|
| Weights | `model.fp16.safetensors` | `config.yaml` + `model.fp16.ckpt` |
| Code package | `hy3dgen` (Hunyuan3D-2 repo) | `hy3dshape` (Hunyuan3D-2.1 repo) |
| Loader | `from_pretrained(subfolder=…)` | `from_single_file(ckpt, config)` |
| Shape VRAM | ~6 GB | ~10 GB |

## Requirements

- NVIDIA GPU with **≥ 10 GB VRAM** for the shape stage (an RTX 3090 / 24 GB is comfortable; the texture pass wants ~21 GB)
- ~10 GB free disk for weights + source (more for the texture-pass downloads)
- Windows or Linux (CUDA). macOS/MPS falls back to fp32 and is slow / untested for the full model.

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

**Shape**
- **Quality** — diffusion steps (30 / 50 / 75)
- **Mesh Resolution** — octree resolution (256 / 384 / 512)
- **Guidance Scale** — how closely the mesh follows the input image
- **Decimate (for CAD/print)** — optional polygon reduction on export
- **Seed** — reproducibility

**Textures (PBR)**
- **Generate textures (PBR)** — enable the paint pass (see prerequisites above)
- **Texture view resolution** — 512 / 768 per-view render size
- **Texture views** — number of camera views painted / baked (6–9)
- **Texture memory** — VRAM ceiling for the paint pass (Low / Balanced / High / Max)
- **Use shared GPU memory** — allow High / Max to page into system RAM (slow)
- **Mesh cleanup** — Regular (default) / Isotropic / BPT neural
- **Bake normal map** — experimental, off by default (see above)
- **Fix texture seams** — reconcile UV-seam colour jumps (on by default)
- **QA debug sheet** — write a `*_qa.png` diagnostic beside the GLB (on by default)

## Upstream sources

- Weights: `tencent/Hunyuan3D-2.1` (subfolder `hunyuan3d-dit-v2-1`)
- Source: `Tencent-Hunyuan/Hunyuan3D-2.1` (package `hy3dshape`)

Model weights are under Tencent's `tencent-hunyuan-community` license — review it before commercial use.
