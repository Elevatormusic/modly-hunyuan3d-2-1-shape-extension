# Output Mode Selector — Design Spec

**Date:** 2026-07-11
**Status:** proposed (for review)

## Goal

Add a single **`output_mode`** control to the Hunyuan3D-2.1 node so a user picks
the *kind* of asset they want — a high-detail render, or a **game-ready** asset
(clean quad topology + baked low-poly maps + clean UVs) — instead of hand-tuning
the ~12 low-level knobs the node exposes today.

## Motivation

Two problems, one control:

1. **Knob overload.** The node already has ~12 parameters (Mesh Resolution,
   Decimate, texture view resolution, views, texture memory, mesh cleanup, bake
   normal map, seam fix, saturation, …). Reaching a coherent outcome means
   understanding all of them. A mode selector puts an intent on top.
2. **No game-ready path.** The raw output is dense marching-cubes/DMC geometry
   with auto-UVs — great for renders, wrong for engines. There is no way today
   to get clean quad topology with baked low-poly maps. This feature builds that
   pipeline. (Artist-like edge loops are out of reach until a neural retopo model
   ships weights; the bar here is clean, uniform quad-dominant topology.)

## Scope

**In:**
- `output_mode` select: **Render – Max**, **Render – Balanced**, **Game-ready**,
  **Custom**. Default **Custom** (backward-compatible — existing pinned workflows
  keep their exact knob values).
- A pure preset layer that maps a non-Custom mode to a fixed bundle of the
  existing knobs; Custom passes the raw knobs through untouched.
- A game-ready pipeline: dense textured mesh → **quad retopo** → **clean UV
  unwrap** → **bake** (normal + AO) + **transfer** the painted PBR
  (albedo/metallic/roughness) onto the new UVs → low-poly GLB.
- Extending the existing `normal_bake.py` from a normal-only baker into a
  **multi-channel baker** (normal + PBR transfer + AO).

**Out:**
- Rigging / armature / skin weights (leave to Blender/Mixamo downstream).
- Multiple LODs.
- Artist-authored edge loops (uniform quad-dominant only, per the retopo bar).
- Changes to the shape or paint models.
- Coupling to the `saturation`/vibrance feature beyond a preset default.

## Modes and knob bundles

The generator treats a non-Custom mode as **authoritative**: it overrides the
listed knobs regardless of their UI values. **Custom** changes nothing.

| Knob | Render – Max | Render – Balanced | Game-ready | Custom |
|---|---|---|---|---|
| octree_resolution | 512 | 384 | 512 (need dense detail to bake from) | user |
| enable_texture | on | on | on (need PBR to transfer) | user |
| texture_resolution | 768 | 512 | 768 | user |
| max_num_view | 8 | 6 | 8 | user |
| mesh_mode | regular | regular | (n/a — retopo path) | user |
| bake_normal_map | on | on | on (baked onto retopo) | user |
| seam_fix | on | on | on | user |
| saturation | subtle | subtle | subtle | user |
| face budget | ~100k | ~100k | **~30k quads** (retopo target) | user (`EB_FACE_TARGET`) |
| game-ready pipeline | — | — | **yes** | — |

Exact values are provisional and confirmed on-device; the bundle is data, kept
in one pure module so it is trivially reviewable and testable.

## Architecture (Approach A — all-Python, reuse our baker)

New/modified units, each with one responsibility:

- **`output_modes.py`** (new, pure, torch-free): `resolve_params(output_mode,
  raw_params) -> params`. Non-Custom modes overlay the bundle above; Custom
  returns `raw_params` unchanged. Unit-tested like `capacity.py`.
- **`retopo.py`** (new): wraps **Instant Meshes** — writes the dense mesh to a
  temp OBJ, shell-invokes the binary for a quad-dominant target, reads the result
  back. The binary is provisioned on first use (download + checksum, the existing
  BPT/prebuilt pattern). **Never-raise → falls back** to `mesh_cleanup.clean_mesh`
  quadric decimation if the binary is unavailable or fails.
- **`normal_bake.py`** (extend): generalize the existing high→low closest-point
  baker so the same ray/closest-point query also samples **albedo/metallic/
  roughness** from the painted atlas (barycentric texture transfer) and computes
  **AO** (hemisphere ray-cast against the dense mesh). Emits + dilates all
  channels.
- **`game_ready.py`** (new): orchestrator. `to_game_ready(dense_textured_mesh,
  painted_atlas, *, target_faces, tex_size) -> low_glb`. Runs retopo → xatlas UV
  → multi-channel bake → attaches glTF `baseColor` / `metallicRoughness` /
  `normal` / `occlusion` → exports the low-poly GLB.
- **`generator.py` / `finishing.py`** (wire): `generate()` resolves `output_mode`
  via `output_modes.resolve_params` before running; when the mode is Game-ready,
  the textured result is routed through `game_ready.to_game_ready` and the
  low-poly GLB is returned instead of the dense one.

## Data flow

```
generate(params)
  -> params = output_modes.resolve_params(output_mode, params)
  -> shape (DiT)
  -> if game-ready or textured: paint PBR
  -> finishing (seam_fix, smooth_normals, vibrance, ...)
  -> if game-ready: game_ready.to_game_ready(dense_textured, atlas)  -> low-poly GLB
     else:          dense textured GLB   (today's behavior)
```

## Game-ready pipeline details

1. **Retopo** — Instant Meshes, quad-dominant, target ~30k faces (a knob;
   default confirmed on-device). Fallback: quadric decimation.
2. **UV unwrap** — xatlas (already a dependency) on the retopo mesh → clean UVs.
3. **Bake / transfer** — per low-poly texel, closest point on the dense mesh →
   tangent-space **normal**, transferred **albedo/metallic/roughness** (source-UV
   barycentric sample of the painted atlas), and **AO** (hemisphere ray-cast).
   Gutter-dilate every channel.
4. **Assemble** — attach the four maps to a single atlas (2K/4K) and export the
   low-poly GLB.

## Error handling

Every stage never-raises at its boundary and degrades instead of crashing:
- Instant Meshes missing/fails → quadric decimation (still baked, not crashed).
- xatlas fails → fall back to the existing atlas.
- A bake channel fails → attach the channels that succeeded.
- The whole game-ready pass fails → return the dense textured GLB (today's
  result), never an error to the user. Matches the extension's existing ethos.

## Testing (unittest, CPU-only, no GPU)

- `resolve_params`: table tests per mode (correct bundle; Custom passthrough;
  unknown mode → safe default).
- `retopo`: binary-absent → quadric fallback; output face-count within target.
- multi-channel baker: PBR-transfer texel accuracy on a synthetic mesh+atlas;
  AO ∈ [0,1]; gutter dilation; never-raise.
- `game_ready` end-to-end on a tiny synthetic mesh → a GLB with four textures
  attached; failure path returns the input dense GLB.
- schema: `output_mode` present in **both** manifest and generator schema,
  default **Custom**.

## Open questions — resolved in research-validation before the plan

- **Instant Meshes**: license (must be shippable in a public repo — else fall
  back to QuadriFlow via pymeshlab, which is already a dependency), Windows /
  Python-3.11 subprocess invocation, and quad-mesh output format.
- **AO method**: hemisphere ray-cast vs pymeshlab's built-in AO — quality vs
  cost.
- **xatlas on quad meshes** (it triangulates internally; confirm UV quality).
- **PBR-transfer accuracy** across the UV change (source-atlas resampling).
- **Face-budget default** for game-ready (20k / 30k / 40k).

## Rollout (Codex-gated, staged)

This work ships as a sequence of small PRs, each reviewed by the GitHub Codex
bot, addressed, and merged before the next:

1. **This spec** → PR → address Codex → merge.
2. **Research-validation** (resolves the open questions) → **implementation
   plan** → PR → address Codex → merge.
3. **Code** (TDD, Approach A) → PR → address Codex → merge.

No local reviewer subagents — Codex reviews each PR. After the code merges,
mirror the changed source to the extension clone + vendored copy, and run an
on-device A/B on the game-ready output.
