<div align="center">

<img src="assets/banner.png" alt="Hunyuan3D 2.1 Full - Modly extension" width="100%">

<p>
  <a href="https://github.com/Elevatormusic/modly-hunyuan3d-2-1-shape-extension"><img src="https://img.shields.io/badge/version-1.7.0-982598" alt="version"></a>
  <img src="https://img.shields.io/badge/Modly-model_extension-15173D" alt="Modly model extension">
  <img src="https://img.shields.io/badge/Windows_%C2%B7_Linux-CUDA-15173D" alt="platform">
  <img src="https://img.shields.io/badge/shape-~10_GB_VRAM-982598" alt="shape vram">
  <img src="https://img.shields.io/badge/textures-~13_GB_reduced-E491C9" alt="textures vram">
</p>

**Turn a single image into a high-fidelity 3D model — with optional PBR textures — right inside [Modly](https://github.com/lightningpixel/modly).**

Wraps the full **Hunyuan3D-2.1** shape checkpoint (Tencent's 3.3B `hunyuan3d-dit-v2-1`, not the lightweight Mini) with an optional paint pass.

</div>

---

## &#10024; Results

<table>
  <tr>
    <th width="50%">Input image</th>
    <th width="50%">Generated 3D (textured)</th>
  </tr>
  <tr>
    <td><img src="assets/results/ramen-in.png" alt="ramen shop input" width="100%"></td>
    <td><img src="assets/results/ramen-out.png" alt="ramen shop 3D result" width="100%"></td>
  </tr>
  <tr>
    <td><img src="assets/results/tree-in.png" alt="tree input" width="100%"></td>
    <td><img src="assets/results/tree-out.png" alt="tree 3D result rendered on the navy backdrop" width="100%"></td>
  </tr>
</table>

<sub>Generated on an RTX 3090 — shape at High quality, textured PBR pass. Modly's model viewport is unlit, so the same assets look richer in any lit renderer or game engine.</sub>

---

## &#9889; Quick start

1. In Modly: **Extensions &#8594; Install from GitHub**, paste this repo's URL. It builds its own isolated Python environment automatically.
2. Click **Download** on the model variant (~7.4 GB weights, one time).
3. Drop in an image and generate. Want color? Switch on **Generate textures (PBR)**.

> **Just want the shape?** It needs only ~10 GB VRAM and no build tools — leave textures off and you're done in under a minute on a modern card.

### &#128444;&#65039; No image? Start from text

Have an idea but no picture? Generate one first: describe your object in **ChatGPT (GPT Image 2.0)** — or any capable text-to-image model — then drop the result into this extension.

A few things make a generated image work well for image&#8594;3D:

- **One object, centered**, on a plain white or neutral background.
- **A 3/4 view** that shows depth — not a dead-on front shot.
- **Even lighting, no harsh shadows.**
- **The whole object in frame** — nothing cropped at the edges.

A prompt like *"product render on a white background, 3/4 view, soft even lighting"* works well. One honest note: busy photo backgrounds can confuse the mesh. The extension removes backgrounds automatically, but a clean source image gives the best geometry.

---

## &#129513; What's inside

- **Full 2.1 shape model** — the 3.3B `hunyuan3d-dit-v2-1` checkpoint, high-fidelity image&#8594;mesh geometry.
- **PBR texture pass** — albedo + packed metallic-roughness, painted from 6–9 camera views.
- **Seam-fix** — reconciles color jumps across UV-island edges so textures don't show hard seams (on by default).
- **Mesh cleanup** — Regular / Isotropic / BPT neural retopology, with automatic fallbacks.
- **CAD / print friendly** — optional decimation produces a mesh that converts cleanly downstream (e.g. Fusion 360's Convert Mesh, or a mesh-to-STEP tool).
- **QA sheet** — an optional one-image diagnostic of every texture map (see the deep dive below).

---

## &#128190; Will it run on my GPU?

**Short answer: textures fit 16 GB cards, and a 12 GB card works too.**

- **Shape** generates on **~10 GB VRAM** with no build tools — comfortable on most modern NVIDIA cards.
- **Textures** paint at full stock quality either way; the only question is how much VRAM they use. **Texture memory** defaults to **Auto**, which measures your free VRAM at generation time and picks:
  - **Standard (full GPU)** — **~20 GB** peak, when it fits.
  - **Reduced VRAM** — **~13 GB** peak at the *same* quality, staging components between CPU and GPU (~5% slower). This is what lets textures **fit a 16 GB card**.
- **12 GB cards** work too — the reduced path peaks ~1 GB over 12 GB, so it spills that ~1 GB to shared system memory and finishes a touch slower.

<sub>Measured on an RTX 3090 (512 view resolution, 6 views): 20.4 GB full-GPU vs 13.0 GB reduced, quality identical. Higher settings cost more — 768 view resolution adds ~14 GB (turn on **Use shared GPU memory**), and each view above 6 adds ~0.7 GB.</sub>

<sub>The texture pass also needs a one-time C++/CUDA build toolchain (Visual Studio C++ Build Tools + a CUDA toolkit). Shape generation needs none, and if the toolchain is missing, texturing fails with a clear message while shape keeps working.</sub>

---

## &#127918; Deep dive

<details>
<summary><b>&#127912; Textures (the PBR paint pass)</b></summary>

<br>

Switch on **Generate textures (PBR)**. The extension runs shape first, frees the shape model from VRAM, paints the mesh with the `hy3dpaint` pipeline, then reloads shape for the next run. Output is a textured `.glb` with an **albedo** map and a **packed metallic-roughness** map (roughness in green, metallic in blue — the glTF standard).

**First run downloads** the `hunyuan3d-paintpbr-v2-1` weights, DINOv2-giant, and a RealESRGAN checkpoint.

> **Export GLB to keep the maps** — Modly's OBJ export drops textures.

</details>

<details>
<summary><b>&#129529; Mesh cleanup</b></summary>

<br>

The raw shape mesh is a dense marching-cubes surface (~2.6M faces at high resolution). Before painting it's cleaned into a UV-friendly base — you pick how with **Mesh cleanup**:

- **Regular** (default) — quadric decimation; one connected shell with clean UVs. The most reliable choice.
- **Isotropic** — uniform-triangle remesh; even topology, but can fragment Hunyuan's non-watertight surfaces into many UV islands (auto-falls back to Regular).
- **BPT neural** — Tencent's [BPT](https://github.com/Tencent-Hunyuan/bpt) artist-topology retopology (~4k faces, cleanest edge flow). First use downloads ~4 GB into an isolated sub-environment; falls back to Regular if unavailable.

</details>

<details>
<summary><b>&#129525; Seam-fix &amp; texture maps</b></summary>

<br>

The paint pass produces a standard glTF PBR set — **albedo** (base color) and a **metallic-roughness** map packed into a single glTF-standard image and wired into the GLB material.

**Fix texture seams** (on by default) reconciles color jumps across UV-island edges in both the albedo and metallic-roughness, so chart boundaries don't show as hard breaks, then dilates color into the UV gutter so seams stay clean under mip-mapping. Turn it off for the raw bake.

</details>

<details>
<summary><b>&#129704; Normal-map bake (experimental, off by default)</b></summary>

<br>

**Bake normal map** transfers dense detail from the full-resolution mesh onto the clean base as a tangent-space normal map, so fine detail can survive cleanup as shading. It's **off by default and experimental**: on detailed / hard-surface meshes the current bake can introduce shading artifacts (a tangent-basis mismatch with the glTF viewer). A corrected high-quality bake is planned. On smooth subjects it adds little. Only applies when textures are on.

</details>

<details>
<summary><b>&#128269; QA debug sheet</b></summary>

<br>

**QA debug sheet** (opt-in) writes a `*_qa.png` beside each textured GLB showing the albedo, metallic, roughness and normal maps, the UV layout, and mesh / texture stats. It's diagnostic only and never changes the model — handy for checking whether a color that looks off is the texture itself or just an unlit viewport.

<div align="center"><img src="assets/qa-sheet-example.png" alt="Example QA debug sheet" width="90%"></div>

</details>

<details>
<summary><b>&#128190; Texture memory &amp; shared GPU memory</b></summary>

<br>

**Texture memory** decides how much VRAM the paint pass may use. Every option paints at the same full stock quality (2048 render / 4096 texture) — the difference is only *where* idle model components wait:

- **Auto** (default) — measures free VRAM at generation time and picks Standard when it fits, else Reduced. You generally never need to touch this.
- **Standard (~20 GB, full GPU)** — everything resident on the GPU; the fastest path when you have the headroom.
- **Reduced VRAM (~13 GB, ~5% slower)** — stages components between CPU and GPU at stage boundaries, so the same run fits a 16 GB card (and a 12 GB card with ~1 GB spill). Same weights, same math, same output.

Measured on an RTX 3090 (512/6 views): 20.4 GB reserved full-GPU vs 13.0 GB reduced, view-space difference 2.24/255 (run-to-run noise).

**Use shared GPU memory** lets a run exceed your VRAM by paging to system RAM over PCIe — needed only for very high settings (e.g. 768 view resolution, which adds ~14 GB) on smaller cards. It's much slower and wants a large Windows page file.

</details>

<details>
<summary><b>&#9881;&#65039; How Reduced VRAM works</b></summary>

<br>

The paint pass runs in distinct stages — encode the condition images, run the multiview diffusion, decode, then upscale and bake — and each stage only needs *some* of the models on the GPU at once.

**Reduced VRAM stages components** between CPU and GPU at those boundaries. The vision encoder (DINO) visits the GPU once for its single forward pass; the VAE only for the encode and decode; the big diffusion model only during the denoising loop. Idle weights wait in ordinary system RAM instead of holding VRAM.

**Why the quality is identical:** the exact same weights run the exact same math — the only thing that changes is where idle components wait. It's verified side-by-side against the full-GPU path (view-space difference within run-to-run noise).

**What it costs:** peak VRAM drops from ~20.4 GB to ~13.0 GB, for roughly 4–5% more time on a PCIe 4 system — a one-time transfer per stage, not per step.

We don't use the standard-library offload for this: the pipeline's custom dual-stream architecture bypasses hook-based offloading, so the extension stages whole components at stage boundaries instead — simpler and more robust for this model.

And because **Auto** measures free VRAM at generation time and picks the full-GPU path when it fits or the reduced path when it doesn't, you generally never have to think about any of this.

</details>

<details>
<summary><b>&#127899; Full parameter list</b></summary>

<br>

**Shape**
- **Quality** — diffusion steps (30 / 50 / 75)
- **Mesh Resolution** — octree resolution (256 / 384 / 512)
- **Guidance Scale** — how closely the mesh follows the input image
- **Decimate (for CAD/print)** — optional polygon reduction on export
- **Seed** — reproducibility

**Textures (PBR)**
- **Generate textures (PBR)** — enable the paint pass
- **Texture view resolution** — 512 / 768 per-view render size
- **Texture views** — camera views painted / baked (6–9)
- **Texture memory** — VRAM path at identical quality: Auto (default) / Standard (~20 GB, full GPU) / Reduced VRAM (~13 GB, ~5% slower)
- **Use shared GPU memory** — let a run page into system RAM when it exceeds VRAM (only needed for very high settings)
- **Mesh cleanup** — Regular (default) / Isotropic / BPT neural
- **Bake normal map** — experimental, off by default
- **Fix texture seams** — reconcile UV-seam color jumps (on by default)
- **QA debug sheet** — write a `*_qa.png` diagnostic beside the GLB

</details>

<details>
<summary><b>&#128421;&#65039; Requirements &amp; how it differs from Mini</b></summary>

<br>

- NVIDIA GPU with **&#8805; 10 GB VRAM** for shape (an RTX 3090 / 24 GB is comfortable; the texture pass runs in ~13 GB on the reduced-VRAM path — fits 16 GB cards — or ~20 GB full GPU).
- ~10 GB free disk for weights + source (more for the texture downloads).
- Windows or Linux (CUDA). macOS/MPS falls back to fp32 and is slow / untested for the full model.

|  | Mini | This (2.1 Full) |
|---|---|---|
| Weights | `model.fp16.safetensors` | `config.yaml` + `model.fp16.ckpt` |
| Code package | `hy3dgen` | `hy3dshape` (Hunyuan3D-2.1) |
| Loader | `from_pretrained(subfolder=…)` | `from_single_file(ckpt, config)` |
| Shape VRAM | ~6 GB | ~10 GB |

</details>

---

## &#128220; Upstream &amp; license

- **Weights:** `tencent/Hunyuan3D-2.1` (subfolder `hunyuan3d-dit-v2-1`)
- **Source:** `Tencent-Hunyuan/Hunyuan3D-2.1` (package `hy3dshape`)

Model weights are under Tencent's `tencent-hunyuan-community` license — review it before commercial use.

### License &amp; attribution

The paint pass builds on the Tencent Hunyuan 3D 2.1 Works, and this repository
redistributes prebuilt binaries compiled from lightly modified copies of them.
The full license text is in [`LICENSE`](LICENSE) and the required attribution,
the list of modified files, and the pass-through use restrictions are in
[`NOTICE`](NOTICE). Use of the model and of the prebuilt binaries in
[`prebuilt/`](prebuilt/) stays subject to the Tencent Hunyuan 3D 2.1 Community
License Agreement and its Acceptable Use Policy — including the territorial
scope (the license does not apply in the EU, UK, or South Korea) and the
prohibited-uses list.

<div align="center"><sub>A community extension for <a href="https://github.com/lightningpixel/modly">Modly</a>. Not affiliated with Tencent or lightningpixel.</sub></div>
