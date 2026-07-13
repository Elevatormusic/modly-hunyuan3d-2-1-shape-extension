# Output Mode Selector — Implementation Plan

> Executed with TDD (unittest, CPU-only), shipped as a single Codex-gated code PR
> after this plan PR merges. Steps use checkbox (`- [ ]`) tracking.

**Goal:** Add an `output_mode` selector (Render – Max / Render – Balanced /
Game-ready / Custom) that drives the existing knobs from one control and, for
Game-ready, produces a clean quad-retopo low-poly asset with baked normal + AO
and transferred PBR.

**Architecture:** A pure preset layer (`output_modes.py`) overlays the existing
knobs for non-Custom modes. A game-ready pipeline runs after paint:
Instant Meshes quad retopo (`retopo.py`) → xatlas UV → a multi-channel bake
(`game_bake.py`, one texel loop over normal/albedo/MR/AO) → assembled low-poly
GLB (`game_ready.py`). Wired through `generator.generate` / `finishing`.

**Tech stack:** Python 3.11, numpy, PIL, trimesh, pymeshlab, xatlas (all present),
optional `embreex` (AO accel). A vendored `Instant Meshes.exe` (BSD-3). No source
builds, no torch in these modules.

## Global Constraints (verbatim, bind every task)

- **No source builds.** Vendor the prebuilt `Instant Meshes.exe` (BSD-3-Clause,
  binary redistribution permitted) under `vendor/instant-meshes/` with upstream
  `LICENSE.txt`; add attribution to the repo `NOTICE`. QuadriFlow is BLOCKED —
  do not use it.
- **Retopo engine order:** Instant Meshes (primary) → pymeshlab
  `meshing_isotropic_explicit_remeshing` + `meshing_tri_to_quad_by_smart_triangle_pairing`
  (fallback) → quadric decimation (last resort). Never-raise between tiers.
- **Instant Meshes CLI:** `"Instant Meshes.exe" <in.obj> -o <out.obj> -v <Q> -d -c 30`
  (pure quads, deterministic, 30° creases). `-v`/`-f` count **quads** — pass
  `Q = round(target_triangles / 2)`.
- **AO:** trimesh hemisphere ray-cast (CPU/headless). Use `embreex` if importable,
  else trimesh's default engine (slower, still correct). NEVER pymeshlab AO.
- **Color space:** albedo is **sRGB**; metallic-roughness is **linear** — never
  apply an sRGB curve to MR. Paint writes MR as `.JPG`; read as-is, treat linear.
- **xatlas:** triangles only (trimesh triangulates the quad OBJ on load; GLB is
  triangles anyway). It adds vertices — rebuild every vertex attribute via the
  returned `vmapping` (`new_attr = old_attr[vmapping]`).
- **Face default:** 30k triangles (`EB_GAMEREADY_FACES`), presets Low/Med/High =
  20k/30k/40k. Knob is in **triangles**; convert to quads for Instant Meshes.
- **Never-raise** at every pipeline boundary; game-ready failure returns the
  dense textured GLB (today's result), never an error.
- **Default `output_mode` = Custom** (backward compatible).
- Commit as `Elevatormusic <22101396+Elevatormusic@users.noreply.github.com>`,
  no co-author trailer. `capacity.py` stays torch-free/untouched. Mirror changed
  source to the extension clone + vendored copy after the code PR merges.

Test command: `"<clone>/venv/Scripts/python.exe" -m unittest discover -s tests -t . -p "test_*.py"`

---

### Task 1: `output_modes.py` — preset layer (pure, torch-free)

**Files:** Create `output_modes.py`; Test `tests/test_output_modes.py`.

**Interface — Produces:** `resolve_params(output_mode: str, raw: dict, schema_ids: set[str]) -> dict`.
- Non-Custom modes overlay the bundle below onto a copy of `raw`. **UI-param
  keys** are written only if present in `schema_ids` (so `saturation` is set only
  when the vibrance knob exists); **internal directive keys** (`_game_ready`,
  `_face_target`) are ALWAYS written and consumed by the generator directly, not
  through the schema. Custom / unknown → return `raw` unchanged.
- `_face_target` (int) is the authoritative face budget for the mode; the
  generator uses it in place of the `EB_FACE_TARGET` env var (see Task 5).

**Bundles** (from the merged spec table):
- `render_max`: octree 512, enable_texture 1, texture_resolution 768, max_num_view 8, mesh_mode regular, bake_normal_map 1, seam_fix 1, saturation subtle*, `_face_target` 100000.
- `render_balanced`: octree 384, enable_texture 1, texture_resolution 512, max_num_view 6, mesh_mode regular, bake_normal_map 1, seam_fix 1, saturation subtle*, `_face_target` 100000.
- `game_ready`: octree 512, enable_texture 1, texture_resolution 768, max_num_view 8, bake_normal_map 1, seam_fix 1, saturation subtle*, `_face_target` 100000, `_game_ready` True.

(`saturation`* = a UI-param key, written only if in `schema_ids`; `_face_target`
and `_game_ready` are internal directive keys, always written.)
- `custom`: passthrough.

**TDD:**
- [ ] Test: each non-Custom mode sets its bundle keys; Custom returns `raw` unchanged (same dict contents).
- [ ] Test: `saturation` is written only when `"saturation" in schema_ids`, omitted otherwise (no unknown key).
- [ ] Test: unknown mode → passthrough (safe default), never raises.
- [ ] Test: `game_ready` sets `_game_ready=True`; others don't.
- [ ] Implement; run; commit.

---

### Task 2: `retopo.py` — quad retopo with tiered fallback

**Files:** Create `retopo.py`; Test `tests/test_retopo.py`.

**Interface — Produces:** `retopo_quads(mesh: trimesh.Trimesh, target_triangles: int) -> trimesh.Trimesh` (returns a triangulated trimesh of the retopo result; never raises — falls through the engine tiers, ultimately quadric decimation of the input).

**Details:**
- Resolve the vendored exe at `vendor/instant-meshes/Instant Meshes.exe` (module-relative). Missing → skip to fallback.
- Write `mesh` to a temp OBJ, invoke
  `subprocess.run(["<exe>", in_obj, "-o", out_obj, "-v", str(round(target_triangles/2)), "-d", "-c", "30"], timeout=..., capture_output=True)`,
  load `out_obj` via trimesh (auto-triangulates quads). Validate non-empty.
- Fallback A (exe missing/error/empty): pymeshlab isotropic remesh to approximately
  the target, then `meshing_tri_to_quad_by_smart_triangle_pairing`, load back.
- Fallback B (pymeshlab error): `mesh_cleanup`/quadric decimation to target.
- Log which tier ran. Never raise.

**TDD:**
- [ ] Test: exe absent (monkeypatch the path) → pymeshlab or quadric fallback returns a mesh with faces ≤ ~target*1.5 and > 0.
- [ ] Test: all engines fail (monkeypatch to raise) → returns a decimated copy of the input, never raises.
- [ ] Test: triangle→quad target conversion passes `round(target/2)` to the exe (assert on the built argv via a patched `subprocess.run`).
- [ ] Implement; run; commit.

---

### Task 3: `game_bake.py` — multi-channel bake (normal + albedo + MR + AO)

**Files:** Create `game_bake.py` (reuse `normal_bake.py`'s UV-rasterization and
dense closest-point/cage helpers; extract them to importable module-level
functions if they aren't already — do not change `normal_bake`'s public bake
behavior). Test `tests/test_game_bake.py`.

**Interface — Produces:** `bake_maps(dense_textured: trimesh.Trimesh, low: trimesh.Trimesh, low_uv, size: int) -> dict` returning `{"albedo": Image, "mr": Image, "normal": Image, "ao": Image}`.

**Algorithm (one texel loop):** rasterize `low_uv` triangles → per-texel 3D
position + interpolated normal; cage-search along the low normal to the dense
surface (reuse normal_bake); at the hit, barycentric-interpolate the dense mesh's
**source UV** and sample the painted atlas for albedo (sRGB) and MR (**linear**,
no curve); tangent-space **normal** as today; **AO** = 1 − hit-fraction of N
cosine-weighted hemisphere rays (trimesh ray, `embreex` if importable). Gutter-
dilate every output atlas (reuse `seam_fix`'s nearest-valid dilation). Clamp
source-atlas sampling to the source triangle/island near seams (no wrap). Never
raise — return whatever channels succeeded.

**TDD (synthetic mesh + known atlas):**
- [ ] Test: albedo transfer — a low texel maps to the correct source color (within tolerance) on a flat 2-triangle case.
- [ ] Test: MR treated as linear (a mid-gray MR sample is not sRGB-shifted).
- [ ] Test: AO ∈ [0,1] for every texel; a fully-open plane bakes ≈1 (unoccluded).
- [ ] Test: gutter dilation writes into padding (no empty texels adjacent to a chart).
- [ ] Test: bad input (empty mesh) → returns {} or partial, never raises.
- [ ] Implement; run; commit.

---

### Task 4: `game_ready.py` — orchestrator

**Files:** Create `game_ready.py`; Test `tests/test_game_ready.py`.

**Interface — Produces:** `to_game_ready(dense_textured_glb: str, target_triangles: int, tex_size: int) -> str` (writes and returns a low-poly GLB path; on any unrecoverable failure returns the input `dense_textured_glb` unchanged).

**Flow:** load dense textured GLB → `retopo.retopo_quads` → `xatlas.parametrize`
(rebuild positions/normals via `vmapping`, apply returned `indices` + `uvs`) →
`game_bake.bake_maps` → assemble a trimesh with the 4 textures
(`baseColorTexture`, `metallicRoughnessTexture`, normalTexture, occlusionTexture)
+ tangents → export low GLB. **xatlas failure → skip game-ready, return the dense
GLB** (never reuse the dense atlas on the retopo mesh — topology differs).

**TDD:**
- [ ] Test (synthetic textured GLB): produces a GLB that loads and carries a baseColor + normal + occlusion texture and fewer faces than the input.
- [ ] Test: xatlas monkeypatched to raise → returns the input dense GLB path unchanged (no crash, no corrupt output).
- [ ] Test: retopo monkeypatched to raise → still returns a valid GLB (quadric fallback path) or the dense GLB.
- [ ] Implement; run; commit.

---

### Task 5: Wire `output_mode` + UI param

**Files:** Modify `generator.py` (resolve params in `generate()`, route game-ready
in `_run_texture`/after finishing), add `output_mode` to the generator schema;
Modify `manifest.json`; Test `tests/test_output_mode_param.py`,
`tests/test_output_mode_wiring.py`.

**Details:**
- In `generate()`: `params = output_modes.resolve_params(params.get("output_mode","custom"), params, {p["id"] for p in self.params_schema()})` before reading the individual knobs. Read `_game_ready = bool(params.get("_game_ready"))`.
- **Face budget:** pass `face_target=params.get("_face_target")` into `_run_texture`
  (new `face_target: int | None = None` param). The texture cleanup then computes
  `_target = face_target or int(os.environ.get("EB_FACE_TARGET", "100000"))` — a
  preset's `_face_target` is authoritative, while `EB_FACE_TARGET` still governs
  Custom mode. This replaces the current direct `os.environ` read at the cleanup site.
- After the textured GLB is finished (post `finishing.finish`), if `_game_ready`,
  route through `game_ready.to_game_ready(out_path, target_triangles, tex_size)`
  and return that path. `target_triangles` from `EB_GAMEREADY_FACES` (default 30000).
- Add `output_mode` select to **both** manifest and generator schema, default
  `custom`, placed at the TOP of the params list; options render_max /
  render_balanced / game_ready / custom.

**TDD:**
- [ ] Test: `output_mode` present in both manifest and generator schema, default `custom`, at index 0.
- [ ] Test: `resolve_params("render_max", …)` applied in `generate()` overrides octree/texture knobs (unit-level on the pure function + a wiring assert).
- [ ] Test: `_run_texture` accepts `face_target`; the cleanup uses it over `EB_FACE_TARGET` (a preset's `_face_target` wins even when the env var is set to a different value; Custom with no `_face_target` falls back to the env/default).
- [ ] Test: game-ready routing calls `game_ready.to_game_ready` (monkeypatched) when `_game_ready`, not otherwise.
- [ ] Implement; run; commit.

---

### Task 6: Vendor exe + license + suite + mirror

**Files:** Add `vendor/instant-meshes/Instant Meshes.exe` + `LICENSE.txt`; update
`NOTICE`; `.gitignore` negation if needed; document optional `embreex` in setup.

- [ ] Vendor the exe + upstream BSD-3 `LICENSE.txt`; add an Instant Meshes stanza to `NOTICE` (BSD-3 notice reproduced verbatim).
- [ ] Add `embreex` as an optional/soft dependency (graceful absence) in the installer notes.
- [ ] Full suite green: `-m unittest discover -s tests -t . -p "test_*.py"`.
- [ ] Mirror `output_modes.py`, `retopo.py`, `game_bake.py`, `game_ready.py`,
  `generator.py`, `manifest.json`, `finishing.py` (if touched) + the vendored exe
  to the extension clone and vendored copy (post-merge).

## Rollout

Single code PR (Approach A) → Codex review → address + resolve threads → merge.
Then mirror + on-device A/B on the game-ready output (retopo quality, bake
registration, face budget), which confirms the 30k default and that Instant
Meshes runs headless on the target machine.

## Research-validated decisions (folded from RV, 2026-07-11)

Instant Meshes vendored BSD-3 exe (headless `-o` batch, `-v` counts quads, `-d -c 30`);
QuadriFlow blocked → pymeshlab isotropic+tri-to-quad fallback; AO via trimesh
hemisphere ray-cast (+embreex soft-accel, never pymeshlab AO); xatlas MIT with
`vmapping` attribute rebuild; PBR transfer with per-channel color space (albedo
sRGB, MR linear), source-seam clamp, destination gutter dilation, all four maps in
one texel loop; default 30k triangles (Low/Med/High 20/30/40k).
