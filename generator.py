"""
Hunyuan3D 2.1 Shape (Full) — Modly generator.

Reference weights : https://huggingface.co/tencent/Hunyuan3D-2.1  (subfolder hunyuan3d-dit-v2-1)
Reference source   : https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1  (package: hy3dshape)

Notes vs. the Hunyuan3D-2 Mini extension this is modelled on:
  * 2.1 ships the shape model as config.yaml + model.fp16.ckpt (a torch checkpoint),
    NOT a diffusers-style .safetensors folder.
  * 2.1 uses the restructured `hy3dshape` package, loaded via
    `Hunyuan3DDiTFlowMatchingPipeline.from_single_file(ckpt_path, config_path, ...)`.
  * Shape stage runs in-app; an optional PBR paint pass builds native modules on
    first use. The paint path also offers a mesh-cleanup mode (regular quadric /
    isotropic remesh / BPT neural retopo) and a tangent-space normal-map bake that
    transfers the dense mesh's detail onto the clean base.
"""
from __future__ import annotations   # keep annotations as strings so the module
                                     # imports without heavy deps (PIL/torch) present
import io
import random
import sys
import threading
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:                    # type-only import; not loaded at runtime,
    from PIL import Image            # keeping the module import free of heavy deps

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

_HF_REPO_ID = "tencent/Hunyuan3D-2.1"
_SUBFOLDER  = "hunyuan3d-dit-v2-1"
_GITHUB_ZIP = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/archive/refs/heads/main.zip"

# --- Texture (paint) pipeline ------------------------------------------------ #
_PAINT_SUBFOLDER = "hunyuan3d-paintpbr-v2-1"   # weights subfolder in _HF_REPO_ID
_REALESRGAN_URL  = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)

# --- Background removal ------------------------------------------------------ #
# The DiT shape pipeline conditions on the image's ALPHA channel as the object
# mask (see preprocessors.py ImageProcessorV2.recenter: if the array has 4
# channels it takes channel 3 as the mask, else it treats the whole frame as
# foreground). So a fully-opaque alpha — i.e. the background was NOT cut out —
# makes the entire photo the "object", and the model reconstructs the backdrop as
# a flat ground slab. A good matte is therefore what prevents the floor. We run on
# the CPU execution provider on purpose: the venv's onnxruntime-gpu targets a
# different CUDA than torch, so its GPU provider can't load (missing cublasLt64_13
# .dll) — and u2net on CPU is a ~1 s op anyway.
_BG_MODELS = ("u2net", "isnet-general-use")   # primary, then escalate on a suspect matte


def _matte_coverage_ok(fg_fraction: float) -> bool:
    """A healthy matte keeps *some* of the frame but not ~all of it.

    fg_fraction = share of pixels left non-transparent after removal. ~1.0 means
    the background was kept (the floor bug); ~0.0 means the object was erased.
    Either extreme is a failed matte worth escalating to the next model / warning.
    """
    return 0.02 < fg_fraction < 0.92


def _prewarm_hf_symlink_check(repo_id: str) -> None:
    """Work around a Windows race in huggingface_hub's parallel downloader.

    On a machine without symlink privilege (Developer Mode off and not running
    as admin), hf_hub's are_symlinks_supported() is probed concurrently by the
    thread_map download workers. The check sets its cache to True *before*
    running the actual test, so a second worker can read that stale True, take
    the symlink branch, and die with `OSError [WinError 1314]` instead of
    falling back to copying. Running the check once, single-threaded, populates
    the per-repo cache correctly before the parallel download starts: it stays
    True (real symlinks) when Developer Mode is on, and becomes False (copy
    fallback) when it is off. Best-effort — never block generation on it.
    """
    try:
        import os
        from huggingface_hub import file_download
        from huggingface_hub.constants import HF_HUB_CACHE

        repo_folder = os.path.join(HF_HUB_CACHE, "models--" + repo_id.replace("/", "--"))
        os.makedirs(repo_folder, exist_ok=True)
        file_download.are_symlinks_supported(repo_folder)
    except Exception:
        pass


class Hunyuan3DShapeV21Generator(BaseGenerator):
    MODEL_ID     = "hunyuan3d-2-1-shape"
    DISPLAY_NAME = "Hunyuan3D 2.1 Shape (Full)"
    VRAM_GB      = 10  # shape stage only; texture is not run here

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        subfolder = self.download_check if self.download_check else _SUBFOLDER
        model_dir = self.model_dir / subfolder
        return (
            model_dir.exists()
            and (model_dir / "config.yaml").exists()
            and (model_dir / "model.fp16.ckpt").exists()
        )

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._download_weights()

        self._ensure_hy3dshape()

        import torch
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        if sys.platform == "darwin":
            # Apple Silicon: MPS has limited fp16 op coverage, use fp32.
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype  = torch.float32
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype  = torch.float16 if device == "cuda" else torch.float32

        subfolder   = self.download_check if self.download_check else _SUBFOLDER
        model_dir   = self.model_dir / subfolder
        config_path = str(model_dir / "config.yaml")
        ckpt_path   = str(model_dir / "model.fp16.ckpt")

        print(f"[{self.MODEL_ID}] Loading 2.1 shape pipeline from {model_dir} on {device}…")
        # Load directly from explicit local files. This bypasses hy3dshape's
        # smart_load_model() HF/cache resolution so the model loads fully offline.
        self._model = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            ckpt_path,
            config_path,
            device=device,
            dtype=dtype,
            use_safetensors=False,
        )
        print(f"[{self.MODEL_ID}] Loaded on {device}.")

    def unload(self) -> None:
        super().unload()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch

        num_steps      = int(params.get("num_inference_steps", 50))
        octree_res     = int(params.get("octree_resolution", 384))
        guidance_scale = float(params.get("guidance_scale", 5.0))
        target_faces   = int(params.get("target_faces", 0))
        enable_texture = int(params.get("enable_texture", 0)) == 1
        tex_resolution = int(params.get("texture_resolution", 512))
        max_num_view   = int(params.get("max_num_view", 6))
        mesh_mode      = str(params.get("mesh_mode", "regular"))
        bake_normal    = int(params.get("bake_normal_map", 0)) == 1
        texture_memory = str(params.get("texture_memory", "balanced"))
        use_shared_vram = int(params.get("use_shared_vram", 0)) == 1
        seam_fix       = int(params.get("seam_fix", 1)) == 1
        debug_sheet    = int(params.get("debug_sheet", 0)) == 1
        seed           = int(params.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        # VRAM-aware preflight: cap Mesh Resolution so the shape stage fits, and warn
        # if textures are likely to exceed free VRAM (prevents silent OOM / the
        # near-limit "stuck" hang the user hit at ~24/24 GB).
        if torch.cuda.is_available():
            try:
                import capacity
                _free_b, _total_b = torch.cuda.mem_get_info()
                octree_res, _vwarn = capacity.vram_plan(
                    _free_b / 1e9, _total_b / 1e9, enable_texture,
                    octree_res, tex_resolution, max_num_view)
                if _vwarn:
                    print(f"[{self.MODEL_ID}] VRAM: {_vwarn}")
                    self._report(progress_cb, 4, _vwarn)
            except Exception as _exc:
                print(f"[{self.MODEL_ID}] VRAM preflight skipped ({_exc})")

        self._report(progress_cb, 5, "Removing background…")
        image = self._preprocess(image_bytes)
        self._check_cancelled(cancel_event)

        # Leave headroom for the paint stage when texturing is on.
        shape_end = 55 if enable_texture else 82
        self._report(progress_cb, 12, "Generating 3D shape…")
        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 12, shape_end, "Generating 3D shape…", stop_evt),
                daemon=True,
            )
            t.start()

        try:
            with torch.no_grad():
                generator = torch.Generator().manual_seed(seed)
                outputs = self._model(
                    image=image,
                    num_inference_steps=num_steps,
                    octree_resolution=octree_res,
                    guidance_scale=guidance_scale,
                    num_chunks=8000,
                    generator=generator,
                    output_type="trimesh",
                    mc_algo="mc",   # scikit-image marching cubes; avoids the
                                    # compile-heavy `diso` needed by the 'dmc' default
                )
            mesh = outputs[0]
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        # Strip the ground/background plane Hunyuan generates from the photo backdrop.
        # Left in, that flat slab is a separate ~2x2 component that dominates the mesh:
        # isotropic remesh sizes triangles by area, so it eats ~90% of the face budget
        # and the real object comes out starved and fragmented. Removing it up front
        # gives the whole budget (and the normal bake) to the actual object.
        try:
            import mesh_cleanup
            _n0 = len(mesh.faces) if hasattr(mesh, "faces") else 0
            mesh = mesh_cleanup.strip_background(mesh)
            if hasattr(mesh, "faces") and len(mesh.faces) != _n0:
                print(f"[{self.MODEL_ID}] stripped background: {_n0} -> {len(mesh.faces)} faces")
        except Exception as _exc:
            print(f"[{self.MODEL_ID}] background strip skipped ({_exc})")

        # The CAD/print decimation applies to the shape-only export path; when texturing
        # we keep the full-detail mesh so the normal bake has detail to transfer (the
        # texture path does its own mesh_mode cleanup in _run_texture).
        if (target_faces > 0 and not enable_texture
                and hasattr(mesh, "faces") and len(mesh.faces) > target_faces):
            self._report(progress_cb, shape_end - 2, "Decimating mesh…")
            mesh = self._decimate(mesh, target_faces)

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        path = self.outputs_dir / name

        if enable_texture:
            # Shape -> Paint sequence. Free the shape model first so the paint
            # models (~21 GB) fit alongside on a 24 GB card, then restore it.
            self._report(progress_cb, 58, "Freeing VRAM for texture stage…")
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._check_cancelled(cancel_event)
            self._run_texture(
                mesh, image, str(path),
                tex_resolution=tex_resolution, max_num_view=max_num_view,
                progress_cb=progress_cb,
                mesh_mode=mesh_mode, bake_normal_map=bake_normal,
                texture_memory=texture_memory,
                use_shared_vram=use_shared_vram,
                seam_fix=seam_fix,
                debug_sheet=debug_sheet,
            )
            self.load()  # restore shape model for the next run
        else:
            self._report(progress_cb, 96, "Exporting GLB…")
            mesh.export(str(path))

        self._report(progress_cb, 100, "Done")
        return path

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _preprocess(self, image_bytes: bytes) -> "Image.Image":
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        matted, model, fg = self._remove_background(img)
        self._save_matte_debug(matted, model, fg)
        return matted

    def _remove_background(self, img_rgb) -> tuple:
        """Cut the background so the shape pipeline sees only the object.

        Returns (rgba_image, model_name, fg_fraction). Tries each model in
        _BG_MODELS on the CPU provider and returns the first matte that passes
        _matte_coverage_ok. If every model looks wrong it returns the closest-to-
        sane result with a loud warning — a bad matte is what produces the floor.
        """
        import numpy as np
        from rembg import remove, new_session
        best = None
        for model in _BG_MODELS:
            try:
                session = new_session(model, providers=["CPUExecutionProvider"])
                out = remove(img_rgb, session=session,
                             bgcolor=[255, 255, 255, 0]).convert("RGBA")
            except Exception as exc:
                print(f"[{self.MODEL_ID}] bg model {model!r} failed ({exc})")
                continue
            fg = float((np.asarray(out)[..., 3] > 10).mean())
            if _matte_coverage_ok(fg):
                print(f"[{self.MODEL_ID}] background removed via {model} "
                      f"(foreground {fg * 100:.0f}%)")
                return out, model, fg
            print(f"[{self.MODEL_ID}] {model} matte suspect "
                  f"(foreground {fg * 100:.0f}%); escalating")
            if best is None or abs(fg - 0.30) < abs(best[2] - 0.30):
                best = (out, model, fg)
        if best is not None:
            print(f"[{self.MODEL_ID}] WARNING: every matte looked wrong; using best "
                  f"({best[1]}, foreground {best[2] * 100:.0f}%). A backdrop/floor may "
                  f"appear — try a cleaner input image.")
            return best
        # rembg unavailable entirely: don't crash the run. An opaque alpha means the
        # pipeline may add a slab, but the mesh still generates.
        print(f"[{self.MODEL_ID}] WARNING: background removal unavailable; "
              f"passing the raw image (a floor is likely).")
        return img_rgb.convert("RGBA"), "none", 1.0

    def _save_matte_debug(self, matted, model: str, fg: float) -> None:
        """Persist the exact matte the shape model will see, for later diagnosis."""
        try:
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            path = self.outputs_dir / "_last_input_matte.png"
            matted.save(path)
            print(f"[{self.MODEL_ID}] saved input matte -> {path} "
                  f"(model={model}, foreground={fg * 100:.0f}%)")
        except Exception as exc:
            print(f"[{self.MODEL_ID}] matte debug save skipped ({exc})")

    def _decimate(self, mesh, target_faces: int):
        try:
            return mesh.simplify_quadric_decimation(target_faces)
        except Exception as exc:
            print(f"[{self.MODEL_ID}] Decimation skipped: {exc}")
            return mesh

    def _download_weights(self) -> None:
        from huggingface_hub import snapshot_download
        subfolder = self.download_check if self.download_check else _SUBFOLDER
        print(f"[{self.MODEL_ID}] Downloading {_HF_REPO_ID} ({subfolder}, ~7.4 GB)…")
        snapshot_download(
            repo_id=_HF_REPO_ID,
            local_dir=str(self.model_dir),
            allow_patterns=[f"{subfolder}/*"],
        )
        print(f"[{self.MODEL_ID}] Download complete.")

    def _ensure_hy3dshape(self) -> None:
        try:
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
            return
        except ImportError:
            pass

        src_dir = self.model_dir / "_hy3dshape_src"
        if not (src_dir / "hy3dshape").exists():
            self._download_hy3dshape(src_dir)

        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        try:
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"hy3dshape still not importable after extraction to {src_dir}.\n"
                f"Check the folder contents.\n{exc}"
            ) from exc

    def _download_hy3dshape(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[{self.MODEL_ID}] Downloading hy3dshape source from GitHub…")
        with urllib.request.urlopen(_GITHUB_ZIP, timeout=180) as resp:
            data = resp.read()
        print(f"[{self.MODEL_ID}] Extracting hy3dshape…")

        # Zip root is "Hunyuan3D-2.1-main/". We want everything under
        # "Hunyuan3D-2.1-main/hy3dshape/" placed directly under dest, so that
        # dest/hy3dshape/ is the importable inner package.
        prefix = "Hunyuan3D-2.1-main/hy3dshape/"
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if not member.startswith(prefix):
                    continue
                rel = member[len(prefix):]
                if not rel:
                    continue
                target = dest / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))

        print(f"[{self.MODEL_ID}] hy3dshape extracted to {dest}.")

    # ------------------------------------------------------------------ #
    # Texture (paint) stage — Hunyuan3D-2.1 hy3dpaint pipeline
    # ------------------------------------------------------------------ #

    def _run_texture(
        self, mesh, image, out_path: str,
        tex_resolution: int = 512, max_num_view: int = 6, progress_cb=None,
        mesh_mode: str = "isotropic", bake_normal_map: bool = False,
        texture_memory: str = "balanced",
        use_shared_vram: bool = False,
        seam_fix: bool = True,
        debug_sheet: bool = False,
    ) -> None:
        """
        Paint PBR textures onto the shape mesh and write a textured GLB to
        out_path. Runs AFTER the shape model has been freed from VRAM.

        Needs (built/downloaded lazily on first use):
          * hy3dpaint source (from the same GitHub zip as hy3dshape)
          * two compiled modules: custom_rasterizer (CUDA) + mesh_inpaint (C++)
          * paint weights (hunyuan3d-paintpbr-v2-1), DINOv2-giant, RealESRGAN
        """
        import os, shutil, tempfile, torch

        paint_src = self._ensure_hy3dpaint(progress_cb)

        self._report(progress_cb, 66, "Loading paint models…")
        from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

        conf = Hunyuan3DPaintConfig(max_num_view, tex_resolution)
        conf.multiview_cfg_path    = str(paint_src / "cfgs" / "hunyuan-paint-pbr.yaml")
        conf.custom_pipeline       = str(paint_src / "hunyuanpaintpbr")
        conf.realesrgan_ckpt_path  = str(paint_src / "ckpt" / "RealESRGAN_x4plus.pth")
        conf.multiview_pretrained_path = _HF_REPO_ID   # downloads paintpbr subfolder
        # Adaptive texture-memory cap. The shape model is already freed + empty_cache'd
        # (generate() ~line 267), so measure free VRAM HERE and pick render/texture/SR
        # sizes that fit — otherwise render_size stays at the vendored default 2048 and
        # the paint peak spills into system RAM (the 56 GB / 40-min crawl). Paired with
        # the downsample=False patch (see _patch_gpu_accel) so the bake runs at
        # texture_size. On a VRAM read failure, fall back to the user's ceiling tier as a
        # fixed cap (still fixes the render_size leak).
        import capacity
        # Opt-in shared-GPU-memory budget: borrow a safe slice of system RAM so High/Max
        # can run when they exceed free VRAM (pages over PCIe -> slower). Read RAM here
        # (keeps capacity.py pure). Page-file preflight: a small Windows page file turns a
        # shared-RAM overflow into a hard "error 1455" instead of a graceful slowdown.
        _extra_budget = 0.0
        if use_shared_vram:
            try:
                import psutil
                _vm = psutil.virtual_memory()
                _total_gb = _vm.total / 2**30
                _extra_budget = capacity.shared_ram_allowance(_total_gb, _vm.available / 2**30)
                _swap_gb = psutil.swap_memory().total / 2**30
                if _swap_gb < 1.5 * _total_gb:
                    _pf = (f"Shared GPU memory is on but the Windows page file (~{_swap_gb:.0f} GB) "
                           f"is small — enlarge it to >= {1.5 * _total_gb:.0f} GB, or a big run may "
                           f"hard-fail (error 1455) instead of just slowing down.")
                    print(f"[{self.MODEL_ID}] {_pf}")
                    self._report(progress_cb, 61, _pf)
            except Exception as _exc:
                print(f"[{self.MODEL_ID}] shared-RAM read failed ({_exc}); shared budget = 0")
                _extra_budget = 0.0
        try:
            _free_gb = torch.cuda.mem_get_info()[0] / 2**30
            _plan = capacity.plan_texture_memory(
                _free_gb, texture_memory,
                tex_resolution=tex_resolution, max_num_view=max_num_view,
                extra_budget_gb=_extra_budget)
        except Exception as _exc:
            print(f"[{self.MODEL_ID}] free-VRAM read failed ({_exc}); applying {texture_memory} cap")
            _plan = capacity.plan_texture_memory(
                float("inf"), texture_memory, extra_budget_gb=_extra_budget)
        capacity.apply_texture_plan(conf, _plan)
        os.environ["EB_SR_CHUNK"] = str(_plan.sr_chunk)   # read by eb_accel.super_resolve_batch
        if _plan.warning:
            print(f"[{self.MODEL_ID}] VRAM: {_plan.warning}")
            self._report(progress_cb, 62, _plan.warning)
        print(f"[{self.MODEL_ID}] texture_memory tier={_plan.tier} "
              f"render={_plan.render_size} texture={_plan.texture_size} sr_chunk={_plan.sr_chunk} "
              f"shared_budget={_extra_budget:.0f}GB")

        # Constructing the pipeline triggers a parallel hf_hub snapshot_download
        # of the paint weights. Prime the symlink-support check first so that
        # download can't crash with WinError 1314 on non-developer-mode Windows.
        _prewarm_hf_symlink_check(_HF_REPO_ID)

        # Calibration telemetry: reset the CUDA peak counters here so the number
        # logged after the paint call is the TRUE paint-stage peak (model load +
        # inference). Feeds capacity._TEX_PEAK tier-seed calibration (see runbook).
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        paint_pipeline = Hunyuan3DPaintPipeline(conf)

        # The paint pipeline works from files: write the shape mesh + input image.
        tmp_dir = Path(tempfile.mkdtemp())
        in_glb  = tmp_dir / "shape.glb"
        in_png  = tmp_dir / "cond.png"

        # Clean the shape mesh into a paint-friendly base. The raw marching-cubes mesh
        # is enormous (~2.6M faces at 512); xatlas-unwrapping it directly yields a
        # tiny-triangle atlas (~1 texel/triangle) and a speckled texture. mesh_mode
        # picks the strategy (regular quadric / isotropic remesh / bpt neural retopo);
        # each falls back to quadric on error. Keep the full-detail mesh as the source
        # for the normal-map bake so cleanup doesn't cost us fine detail.
        import mesh_cleanup
        dense_for_bake = mesh
        try:
            mesh = mesh_cleanup.clean_mesh(mesh, mesh_mode, 50000)
            print(f"[{self.MODEL_ID}] cleanup mode={mesh_mode} -> {len(mesh.faces)} faces")
        except Exception as exc:
            print(f"[{self.MODEL_ID}] cleanup failed ({exc}); texturing the raw mesh")

        mesh.export(str(in_glb))
        image.save(str(in_png))

        # The paint pipeline follows an OBJ convention: it writes the textured
        # mesh to `output_mesh_path` as OBJ (+ .mtl + texture PNGs) and, with
        # save_glb, converts it to a sibling ".glb" via a literal ".obj"->".glb"
        # string replace. So output_mesh_path MUST end in ".obj" — passing our
        # final ".glb" wrote OBJ into a .glb-named file and made the replace a
        # no-op, so trimesh then mis-read that file as GLB ("incorrect header on
        # GLB file"). Give it a temp ".obj" and move the produced ".glb" to out_path.
        tex_obj = str(tmp_dir / "textured.obj")
        # paint_pipeline() is one long blocking call. Real sub-progress comes from
        # the vendored paint stages calling eb_accel.report() (hook registered just
        # below): per denoise step (74->88) + post-diffusion milestones (88->94).
        self._report(progress_cb, 74, "Painting textures…")
        # Real progress: register a hook the vendored paint stages call. The
        # multiview diffusion reports per denoise step (74->88, via the
        # multiview_utils patch in _patch_gpu_accel) and the post-diffusion stages
        # report milestones (88->94, textureGenPipeline patch). Replaces the old
        # cosmetic creep. eb_accel.report() swallows errors, so a missing patch
        # only leaves the bar static — it can never break the render.
        try:
            import eb_accel
            eb_accel.set_progress_hook(
                (lambda p, l: self._report(progress_cb, p, l)) if progress_cb else None)
        except Exception:
            pass
        try:
            paint_pipeline(
                mesh_path=str(in_glb),
                image_path=str(in_png),
                output_mesh_path=tex_obj,
                use_remesh=False,   # skip Blender-based remesh (bpy removed)
                save_glb=True,
            )
            if torch.cuda.is_available():
                _pk_a = torch.cuda.max_memory_allocated() / 2**30
                _pk_r = torch.cuda.max_memory_reserved() / 2**30
                print(f"[{self.MODEL_ID}] paint peak: allocated {_pk_a:.1f} GB / "
                      f"reserved {_pk_r:.1f} GB (tier={_plan.tier} "
                      f"render={_plan.render_size} texture={_plan.texture_size} "
                      f"tex_res={tex_resolution})")
            produced_glb = tex_obj[:-4] + ".glb"
            if not os.path.exists(produced_glb):
                raise RuntimeError(
                    f"paint pipeline did not produce a GLB at {produced_glb}")
            try:
                os.replace(produced_glb, out_path)
            except OSError:
                shutil.move(produced_glb, out_path)
            # Post-paint finishing pipeline: seam reconcile -> (optional) normal
            # bake -> structural validation -> QA debug sheet. Each stage is
            # non-fatal; finishing.finish() never raises, so the paint ships.
            import finishing
            finishing.finish(
                out_path, tex_obj,
                dense_mesh=dense_for_bake, texture_size=_plan.texture_size,
                mesh_mode=mesh_mode, bake_normal_map=bake_normal_map,
                seam_fix=seam_fix, debug_sheet=debug_sheet,
                input_image_path=str(in_png),
                report=(lambda p, l: self._report(progress_cb, p, l)) if progress_cb else None,
            )
        finally:
            try:
                import eb_accel
                eb_accel.set_progress_hook(None)
            except Exception:
                pass
            del paint_pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Remove the paint scratch dir (shape.glb + cond.png + textured.obj/.mtl
            # + texture maps, ~15-25 MB/gen). It leaked to %TEMP% every run. Runs
            # after finishing.finish() (which reads textured.* for the QA sheet), and
            # is guarded so cleanup can never break the never-raise contract (Fix 10).
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    def _ensure_hy3dpaint(self, progress_cb=None) -> Path:
        """Download hy3dpaint source, build its two native modules, fetch the
        RealESRGAN checkpoint. Returns the paint source directory."""
        paint_src = self.model_dir / "_hy3dpaint_src"

        if not (paint_src / "textureGenPipeline.py").exists():
            self._report(progress_cb, 60, "Downloading paint source…")
            self._download_subtree("hy3dpaint", paint_src)

        if str(paint_src) not in sys.path:
            sys.path.insert(0, str(paint_src))

        # basicsr / RealESRGAN import torchvision.transforms.functional_tensor,
        # a path removed in newer torchvision. Recreate it as a tiny forwarding
        # shim so those (older) libraries keep working. Idempotent.
        self._ensure_functional_tensor_shim()

        # The paint source imports `bpy` (Blender) only for a mesh cleanup step
        # and an OBJ->GLB file conversion. bpy is a full Blender build pinned to
        # one Python version and frequently fails to load; trimesh does both
        # jobs. Patch bpy out so texturing never depends on Blender. Idempotent.
        self._patch_out_bpy(paint_src)

        # Install the extension's pure-torch GPU accelerations (batched SR,
        # push-pull inpaint, RealESRGAN device pin). Each rewired call site falls
        # back to the original CPU path on error, so this only affects speed,
        # never correctness. Idempotent.
        self._patch_gpu_accel(paint_src)

        self._report(progress_cb, 62, "Building paint native modules…")
        self._build_paint_extensions(paint_src)
        self._ensure_realesrgan(paint_src)
        return paint_src

    def _patch_gpu_accel(self, paint_src: Path) -> None:
        """Ship eb_accel.py next to the vendored paint code and rewire three call
        sites to the extension's pure-torch GPU helpers. Idempotent; each patch
        is guarded by a marker so a re-download re-applies cleanly and a running
        install is never double-patched."""
        import shutil

        helper = Path(__file__).resolve().parent / "eb_accel.py"
        if not helper.exists():
            print(f"[{self.MODEL_ID}] eb_accel.py not found next to generator.py; skipping GPU accel")
            return
        try:
            shutil.copyfile(helper, paint_src / "eb_accel.py")
        except OSError as exc:
            print(f"[{self.MODEL_ID}] could not copy eb_accel.py: {exc}")
            return

        # 1. Batched SR — replace the per-image SR loop + the buggy resize loop.
        tgp = paint_src / "textureGenPipeline.py"
        if tgp.exists():
            text = tgp.read_text(encoding="utf-8")
            old = (
                '        for i in range(len(enhance_images["albedo"])):\n'
                '            enhance_images["albedo"][i] = self.models["super_model"](enhance_images["albedo"][i])\n'
                '            enhance_images["mr"][i] = self.models["super_model"](enhance_images["mr"][i])\n'
                '\n'
                '        ###########  Bake  ##########\n'
                '        for i in range(len(enhance_images)):\n'
                '            enhance_images["albedo"][i] = enhance_images["albedo"][i].resize(\n'
                '                (self.config.render_size, self.config.render_size)\n'
                '            )\n'
                '            enhance_images["mr"][i] = enhance_images["mr"][i].resize((self.config.render_size, self.config.render_size))'
            )
            new = (
                '        # GPU: batched RealESRGAN + on-GPU resize (extension patch)\n'
                '        import eb_accel\n'
                '        enhance_images["albedo"] = eb_accel.super_resolve_batch(\n'
                '            self.models["super_model"], enhance_images["albedo"], self.config.render_size)\n'
                '        enhance_images["mr"] = eb_accel.super_resolve_batch(\n'
                '            self.models["super_model"], enhance_images["mr"], self.config.render_size)\n'
                '\n'
                '        ###########  Bake  ##########'
            )
            if "eb_accel.super_resolve_batch" not in text and old in text:
                tgp.write_text(text.replace(old, new), encoding="utf-8")
                print(f"[{self.MODEL_ID}] patched batched SR into textureGenPipeline.py")

            # 1b. Skip the on-save 2x downsample so the bake runs at the output
            # resolution (paired with conf.texture_size=2048) instead of at 2x.
            text = tgp.read_text(encoding="utf-8")
            if "downsample=True" in text:
                tgp.write_text(text.replace("downsample=True", "downsample=False"), encoding="utf-8")
                print(f"[{self.MODEL_ID}] patched save_mesh downsample=False (bake at output res)")

        # 2. Push-pull inpaint — replace only the cv2 NS pass (keep meshVerticeInpaint).
        mr = paint_src / "DifferentiableRenderer" / "MeshRender.py"
        if mr.exists():
            text = mr.read_text(encoding="utf-8")
            old = '            texture_np = cv2.inpaint((texture_np * 255).astype(np.uint8), 255 - mask, 3, cv2.INPAINT_NS)'
            new = (
                '            import eb_accel  # GPU push-pull hole-fill; keeps meshVerticeInpaint above, falls back to cv2\n'
                '            texture_np = eb_accel.inpaint_fill_gpu(texture_np, mask)'
            )
            if "eb_accel.inpaint_fill_gpu" not in text and old in text:
                mr.write_text(text.replace(old, new), encoding="utf-8")
                print(f"[{self.MODEL_ID}] patched push-pull inpaint into MeshRender.py")

        # 3. Pin RealESRGAN to an explicit device (no silent fp16-on-CPU crawl).
        isu = paint_src / "utils" / "image_super_utils.py"
        if isu.exists():
            text = isu.read_text(encoding="utf-8")
            old = (
                '        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)\n'
                '        upsampler = RealESRGANer(\n'
                '            scale=4,\n'
                '            model_path=config.realesrgan_ckpt_path,\n'
                '            dni_weight=None,\n'
                '            model=model,\n'
                '            tile=0,\n'
                '            tile_pad=10,\n'
                '            pre_pad=0,\n'
                '            half=True,\n'
                '            gpu_id=None,\n'
                '        )'
            )
            new = (
                '        import torch\n'
                '        use_cuda = torch.cuda.is_available()\n'
                '        device = torch.device(getattr(config, "device", "cuda") if use_cuda else "cpu")\n'
                '        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)\n'
                '        upsampler = RealESRGANer(\n'
                '            scale=4,\n'
                '            model_path=config.realesrgan_ckpt_path,\n'
                '            dni_weight=None,\n'
                '            model=model,\n'
                '            tile=0,\n'
                '            tile_pad=10,\n'
                '            pre_pad=0,\n'
                '            half=use_cuda,\n'
                '            device=device,\n'
                '            gpu_id=None,\n'
                '        )'
            )
            if "use_cuda = torch.cuda.is_available()" not in text and old in text:
                isu.write_text(text.replace(old, new), encoding="utf-8")
                print(f"[{self.MODEL_ID}] pinned RealESRGAN device in image_super_utils.py")

        # 4. Cleaner UVs — the paint calls a bare xatlas.parametrize (default 1 chart
        # iteration -> ~150 small charts). Tune it: more chart iterations + a moderate
        # cost ceiling (max_cost=2.0 sits at xatlas's area-distortion floor, ~0.20
        # log-std; higher costs trade distortion away for fewer charts, lower costs just
        # add packing waste), plus pack padding so adjacent charts can't bilinear-bleed.
        # Measured 141 charts / 0.208 distortion (vs 128 / 0.236 at the old max_cost=8).
        uvw = paint_src / "utils" / "uvwrap_utils.py"
        if uvw.exists():
            text = uvw.read_text(encoding="utf-8")
            old = "    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)"
            new = (
                "    _atlas = xatlas.Atlas(); _atlas.add_mesh(mesh.vertices, mesh.faces)\n"
                "    _co = xatlas.ChartOptions()\n"
                "    _co.max_iterations = 3\n"
                "    _co.max_cost = 2.0\n"
                "    _co.max_chart_area = 0.0\n"
                "    _co.max_boundary_length = 0.0\n"
                "    _po = xatlas.PackOptions()\n"
                "    _po.padding = 4\n"
                "    _po.bilinear = True\n"
                "    _atlas.generate(chart_options=_co, pack_options=_po)\n"
                "    vmapping, indices, uvs = _atlas.get_mesh(0)"
            )
            if "_atlas.get_mesh" not in text and old in text:
                uvw.write_text(text.replace(old, new), encoding="utf-8")
                print(f"[{self.MODEL_ID}] tuned xatlas for cleaner UVs (fewer charts + padding)")

        # 6. Free the diffusion net before SR/bake (quality-per-VRAM). The 6-view UNet2p5D
        # (+ dual UNet + DINOv2, ~15-18 GB) is unused after the views are produced, but it
        # stays resident through the bake, so a large texture_size stacks on it and blows
        # past 24 GB. Delete it right after the view images are copied out -> the SR + bake
        # get the whole card, so texture_size 4096 is affordable. Single-use pipeline
        # (built fresh + deleted per generation), so nothing calls it again.
        tgp = paint_src / "textureGenPipeline.py"
        if tgp.exists():
            text = tgp.read_text(encoding="utf-8")
            anchor = '        enhance_images["mr"] = copy.deepcopy(multiviews_pbr["mr"])\n'
            if "Free the multiview diffusion net" not in text and anchor in text:
                inject = anchor + (
                    "\n"
                    "        # Free the multiview diffusion net (UNet2p5D + dual UNet + DINOv2)\n"
                    "        # before SR/bake (extension patch): unused after the views are made.\n"
                    "        try:\n"
                    "            if \"multiview_model\" in self.models:\n"
                    "                del self.models[\"multiview_model\"]\n"
                    "                torch.cuda.empty_cache()\n"
                    "        except Exception:\n"
                    "            pass\n"
                )
                tgp.write_text(text.replace(anchor, inject), encoding="utf-8")
                print(f"[{self.MODEL_ID}] patched free-diffusion-before-bake into textureGenPipeline.py")

        # 7. Real progress bar. Wire the paint stages to eb_accel.report():
        # (a) the multiview diffusion fires callback_on_step_end per denoise step
        #     (74->88%); (b) the post-diffusion stages emit milestones (88->94%).
        # _run_texture registers the hook; report() swallows errors, so a missing
        # patch only leaves the bar static — never breaks the render. Anchored on
        # PRISTINE strings; 7b runs after 1 (its anchors need eb_accel.super_resolve_batch).
        mu = paint_src / "utils" / "multiview_utils.py"
        if mu.exists():
            text = mu.read_text(encoding="utf-8")
            old = (
                '        mvd_image = self.pipeline(\n'
                '            input_images[0:1],\n'
                '            num_inference_steps=infer_steps_dict[self.pipeline.scheduler.__class__.__name__],\n'
                '            prompt=prompt,\n'
                '            sync_condition=sync_condition,\n'
                '            guidance_scale=3.0,\n'
                '            **kwargs,\n'
                '        ).images'
            )
            new = (
                '        _eb_steps = infer_steps_dict[self.pipeline.scheduler.__class__.__name__]\n'
                '\n'
                '        def _eb_progress_cb(_pipe, _i, _t, _kw):\n'
                '            try:\n'
                '                import eb_accel\n'
                '                eb_accel.report(74 + int(14 * (_i + 1) / max(_eb_steps, 1)), "Painting textures…")\n'
                '            except Exception:\n'
                '                pass\n'
                '            return _kw\n'
                '\n'
                '        mvd_image = self.pipeline(\n'
                '            input_images[0:1],\n'
                '            num_inference_steps=_eb_steps,\n'
                '            prompt=prompt,\n'
                '            sync_condition=sync_condition,\n'
                '            guidance_scale=3.0,\n'
                '            callback_on_step_end=_eb_progress_cb,\n'
                '            **kwargs,\n'
                '        ).images'
            )
            if "callback_on_step_end=_eb_progress_cb" not in text and old in text:
                mu.write_text(text.replace(old, new), encoding="utf-8")
                print(f"[{self.MODEL_ID}] patched per-step progress callback into multiview_utils.py")

        # 7b. Post-diffusion progress milestones in textureGenPipeline.
        tgp = paint_src / "textureGenPipeline.py"
        if tgp.exists():
            text = tgp.read_text(encoding="utf-8")
            if "eb_accel.report(88" not in text and "eb_accel.super_resolve_batch" in text:
                text = text.replace(
                    '        enhance_images["albedo"] = eb_accel.super_resolve_batch(',
                    '        eb_accel.report(88, "Upscaling views…")\n'
                    '        enhance_images["albedo"] = eb_accel.super_resolve_batch(', 1)
                text = text.replace(
                    '        ###########  Bake  ##########\n'
                    '        texture, mask = self.view_processor.bake_from_multiview(',
                    '        ###########  Bake  ##########\n'
                    '        eb_accel.report(90, "Baking texture…")\n'
                    '        texture, mask = self.view_processor.bake_from_multiview(', 1)
                text = text.replace(
                    '        ##########  inpaint  ###########\n'
                    '        texture = self.view_processor.texture_inpaint(',
                    '        ##########  inpaint  ###########\n'
                    '        eb_accel.report(92, "Inpainting…")\n'
                    '        texture = self.view_processor.texture_inpaint(', 1)
                text = text.replace(
                    '        self.render.save_mesh(output_mesh_path',
                    '        eb_accel.report(94, "Saving mesh…")\n'
                    '        self.render.save_mesh(output_mesh_path', 1)
                tgp.write_text(text, encoding="utf-8")
                print(f"[{self.MODEL_ID}] patched paint-stage progress milestones into textureGenPipeline.py")

    @staticmethod
    def _patch_out_bpy(paint_src: Path) -> None:
        """Remove the hard dependency on `bpy` (Blender) from the paint source.
        bpy is only used for (1) a mesh cleanup 'remesh' and (2) an OBJ->GLB
        conversion. We make the import optional and replace the conversion with
        trimesh. The remesh path is skipped at call time (use_remesh=False)."""
        mu = paint_src / "DifferentiableRenderer" / "mesh_utils.py"
        if not mu.exists():
            return
        text = mu.read_text(encoding="utf-8")
        if "# --- bpy-free patch ---" in text:
            return  # already patched

        # 1) make `import bpy` non-fatal
        text = text.replace(
            "import bpy",
            "try:\n    import bpy\nexcept Exception:\n    bpy = None",
            1,
        )
        # 2) append a trimesh-based convert_obj_to_glb that shadows the bpy one.
        #    Force a non-metallic PBR material with a white base-color factor:
        #    trimesh's OBJ import leaves metallicFactor unset, which glTF treats
        #    as 1.0 (FULLY METALLIC) — that renders gray in viewers without an
        #    environment map even though the baked albedo texture is present.
        #    metallicFactor=0 lets the albedo show as a normal diffuse surface.
        text += (
            "\n\n# --- bpy-free patch ---\n"
            "def convert_obj_to_glb(obj_path, glb_path, *args, **kwargs):  # noqa: F811\n"
            "    \"\"\"OBJ->GLB: wire albedo + baked metallic-roughness (mr_export);\n"
            "    fall back to flat albedo-only PBR on any error.\"\"\"\n"
            "    try:\n"
            "        import mr_export\n"
            "        return mr_export.build_glb_with_mr(obj_path, glb_path)\n"
            "    except Exception:\n"
            "        import trimesh\n"
            "        from trimesh.visual.material import PBRMaterial\n"
            "        scene = trimesh.load(obj_path, process=False)\n"
            "        geoms = scene.geometry.values() if hasattr(scene, 'geometry') else [scene]\n"
            "        for g in geoms:\n"
            "            v = getattr(g, 'visual', None)\n"
            "            mat = getattr(v, 'material', None)\n"
            "            img = (getattr(mat, 'baseColorTexture', None) or getattr(mat, 'image', None)) if mat else None\n"
            "            uv = getattr(v, 'uv', None)\n"
            "            if img is not None and uv is not None:\n"
            "                g.visual = trimesh.visual.TextureVisuals(\n"
            "                    uv=uv, material=PBRMaterial(baseColorTexture=img,\n"
            "                        baseColorFactor=[255, 255, 255, 255], metallicFactor=0.0, roughnessFactor=1.0))\n"
            "        scene.export(glb_path)\n"
            "        return True\n"
        )
        mu.write_text(text, encoding="utf-8")
        print(f"[{__name__}] patched bpy out of {mu}")

    @staticmethod
    def _ensure_functional_tensor_shim() -> None:
        """Restore torchvision.transforms.functional_tensor (removed in newer
        torchvision) as a forwarding stub, so basicsr/RealESRGAN import cleanly."""
        try:
            import importlib.util
            if importlib.util.find_spec("torchvision.transforms.functional_tensor"):
                return  # already importable — nothing to do
        except Exception:
            pass
        try:
            import torchvision
            shim = (Path(torchvision.__file__).parent
                    / "transforms" / "functional_tensor.py")
            if not shim.exists():
                shim.write_text(
                    "# auto-added shim: forwards the one symbol basicsr needs\n"
                    "from torchvision.transforms.functional import rgb_to_grayscale\n",
                    encoding="utf-8",
                )
                print(f"[{__name__}] wrote functional_tensor shim -> {shim}")
        except Exception as exc:
            print(f"[functional_tensor shim] skipped: {exc}")

    @staticmethod
    def _patch_custom_rasterizer_sources(cr_dir: Path) -> None:
        """On Windows, make custom_rasterizer both BUILD and RUN correctly:

        1. Build the `_for_windows` kernel sources, not the Linux ones. The
           Linux sources pass size_t into torch::zeros({...}) braced
           initializers and use 32-bit `long` pointers, which MSVC rejects
           (error C2398 narrowing). Point setup.py at the `_for_windows` variant.
        2. Fix a dtype bug in the _for_windows GPU kernel: `z_min` is a kInt64
           (Long) tensor, but the source reads it with `z_min.data_ptr<uint64_t>()`,
           which torch rejects at RUN time ("expected scalar type UInt64 but
           found Long"). Ask for the pointer as int64_t (matches Long); the
           existing `(uint64_t*)` cast still reinterprets the bits, exactly as
           the CPU path in the same file already does.

        Both patches are idempotent; the whole method is a no-op off Windows."""
        import os
        if os.name != "nt":
            return

        setup_py = cr_dir / "setup.py"
        if setup_py.exists():
            text = setup_py.read_text(encoding="utf-8")
            if "lib/custom_rasterizer_kernel/" in text:
                setup_py.write_text(
                    text.replace("lib/custom_rasterizer_kernel/",
                                 "lib/custom_rasterizer_kernel_for_windows/"),
                    encoding="utf-8")
                print(f"[{__name__}] patched custom_rasterizer to build _for_windows sources")

        gpu = (cr_dir / "lib" / "custom_rasterizer_kernel_for_windows"
               / "rasterizer_gpu.cu")
        if gpu.exists():
            gtext = gpu.read_text(encoding="utf-8")
            if "z_min.data_ptr<uint64_t>()" in gtext:
                gpu.write_text(
                    gtext.replace("z_min.data_ptr<uint64_t>()",
                                  "z_min.data_ptr<int64_t>()"),
                    encoding="utf-8")
                print(f"[{__name__}] patched _for_windows rasterizer_gpu.cu dtype bug")

    @staticmethod
    def _list_msvc_toolsets():
        """[(vcvarsall_path, toolset_version)] for every MSVC toolset found on
        disk, OLDEST toolset first. Uses a filesystem scan rather than vswhere,
        which is unreliable for prerelease / side-by-side VS installs."""
        import os
        found = set()
        for pf in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
            if not pf:
                continue
            root = Path(pf) / "Microsoft Visual Studio"
            if not root.is_dir():
                continue
            # <root>/<ver>/<edition>/VC/Tools/MSVC/<toolset>/bin/<host>/x64/cl.exe
            for cl in root.glob("*/*/VC/Tools/MSVC/*/bin/*/x64/cl.exe"):
                vcvarsall = cl.parents[7] / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
                if vcvarsall.exists():
                    found.add((str(vcvarsall), cl.parents[3].name))
        return sorted(found, key=lambda it: [int(x) for x in it[1].split(".") if x.isdigit()])

    def _build_custom_rasterizer(self, cr_dir: Path):
        """pip-install custom_rasterizer, working around a Visual Studio newer
        than the installed CUDA toolkit supports (which makes nvcc's front-end
        `cudafe++` crash). Try the default compiler first; on Windows, if that
        fails, retry inside each installed MSVC toolset (oldest first, via that
        toolset's vcvars) with -allow-unsupported-compiler until one builds.
        Returns the last CompletedProcess (returncode 0 == success)."""
        import os, shutil, subprocess

        def _clean():
            shutil.rmtree(cr_dir / "build", ignore_errors=True)
            shutil.rmtree(cr_dir / "custom_rasterizer.egg-info", ignore_errors=True)

        # --no-build-isolation: custom_rasterizer's setup.py imports torch, so it
        # must build against the venv's torch rather than an isolated build env.
        _clean()
        r = subprocess.run([sys.executable, "-m", "pip", "install", ".",
                            "--no-build-isolation"],
                           cwd=str(cr_dir), capture_output=True, text=True)
        if r.returncode == 0 or os.name != "nt":
            return r

        for vcvarsall, ver in self._list_msvc_toolsets():
            _clean()
            # Activate the toolset and build in the SAME shell via a temp .bat —
            # capturing/re-injecting the env is fragile (cmd /c quote-stripping).
            bat = cr_dir / "_eb_build_toolset.bat"
            bat.write_text(
                "@echo off\r\n"
                f'call "{vcvarsall}" x64 -vcvars_ver={ver} >nul 2>&1\r\n'
                "if errorlevel 1 exit /b 1\r\n"
                "set DISTUTILS_USE_SDK=1\r\n"
                "set NVCC_PREPEND_FLAGS=-allow-unsupported-compiler\r\n"
                f'"{sys.executable}" -m pip install "{cr_dir}" --no-build-isolation\r\n'
                "exit /b %ERRORLEVEL%\r\n",
                encoding="utf-8")
            print(f"[{self.MODEL_ID}] retrying custom_rasterizer build with MSVC {ver}…")
            try:
                r = subprocess.run(["cmd", "/c", "_eb_build_toolset.bat"],
                                   cwd=str(cr_dir), capture_output=True, text=True)
            finally:
                try:
                    bat.unlink()
                except OSError:
                    pass
            if r.returncode == 0:
                print(f"[{self.MODEL_ID}] custom_rasterizer built with MSVC toolset {ver}")
                return r
        return r

    def _build_paint_extensions(self, paint_src: Path) -> None:
        """Compile custom_rasterizer (CUDA) and mesh_inpaint_processor (C++).
        Requires a matching CUDA toolkit (nvcc) and a C++ compiler (MSVC on
        Windows). Raises a clear error if the toolchain is missing."""
        import importlib, subprocess

        # 1. custom_rasterizer — a CUDA extension. The source tree's OUTER
        #    `custom_rasterizer/` folder has no __init__.py, so with paint_src on
        #    sys.path `import custom_rasterizer` succeeds as an EMPTY namespace
        #    package even when nothing has been compiled. Using that as the
        #    "already built" check silently SKIPS the build and later explodes
        #    with `module 'custom_rasterizer' has no attribute 'rasterize'`.
        #    Probe the compiled kernel itself instead.
        def _kernel_ready() -> bool:
            try:
                importlib.import_module("custom_rasterizer_kernel")
                return True
            except Exception:
                return False

        if not _kernel_ready():
            cr_dir = paint_src / "custom_rasterizer"
            # On Windows, compile the MSVC-safe `_for_windows` kernel sources.
            self._patch_custom_rasterizer_sources(cr_dir)
            print(f"[{self.MODEL_ID}] Building custom_rasterizer (CUDA)…")
            r = self._build_custom_rasterizer(cr_dir)
            if r.returncode != 0:
                raise RuntimeError(
                    "Failed to build 'custom_rasterizer'. Texture generation needs a "
                    "CUDA toolkit (nvcc) and a compatible MSVC C++ toolset. If your "
                    "Visual Studio is newer than your CUDA toolkit supports, install "
                    "the Visual Studio 2022 C++ build tools (or a newer CUDA toolkit "
                    "matching your PyTorch).\n"
                    f"--- build output ---\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
                )
            # A namespace-package stub may already be cached; drop stale entries
            # and caches so the freshly installed build is imported on next use.
            importlib.invalidate_caches()
            for _m in ("custom_rasterizer", "custom_rasterizer_kernel"):
                sys.modules.pop(_m, None)
            if not _kernel_ready():
                raise RuntimeError(
                    "custom_rasterizer reported a successful build but "
                    "'custom_rasterizer_kernel' still can't be imported — see the "
                    "build output above."
                )

        # 2. mesh_inpaint_processor — a pybind11 C++ module compiled in place.
        #    Build it with torch's own cpp_extension: torch ships the pybind11
        #    headers AND locates MSVC on Windows, so this needs no separate
        #    pybind11 package and reuses the exact toolchain custom_rasterizer used.
        dr_dir = paint_src / "DifferentiableRenderer"
        already = list(dr_dir.glob("mesh_inpaint_processor*.so")) + \
                  list(dr_dir.glob("mesh_inpaint_processor*.pyd"))
        if not already:
            print(f"[{self.MODEL_ID}] Building mesh_inpaint_processor (C++)…")
            build_script = dr_dir / "_build_inpaint.py"
            build_script.write_text(
                "from setuptools import setup\n"
                "from torch.utils.cpp_extension import CppExtension, BuildExtension\n"
                "setup(\n"
                "    name='mesh_inpaint_processor',\n"
                "    ext_modules=[CppExtension('mesh_inpaint_processor',\n"
                "                              ['mesh_inpaint_processor.cpp'])],\n"
                "    cmdclass={'build_ext': BuildExtension},\n"
                "    script_args=['build_ext', '--inplace'],\n"
                ")\n",
                encoding="utf-8",
            )
            r = subprocess.run([sys.executable, str(build_script)],
                               cwd=str(dr_dir), capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(
                    "Failed to build 'mesh_inpaint_processor'. This is a C++ "
                    "(pybind11) module compiled via PyTorch. Ensure Visual Studio "
                    "C++ Build Tools are installed. Full build output below.\n"
                    f"--- stdout ---\n{r.stdout[-3000:]}\n--- stderr ---\n{r.stderr[-3000:]}"
                )

    def _ensure_realesrgan(self, paint_src: Path) -> None:
        ckpt = paint_src / "ckpt" / "RealESRGAN_x4plus.pth"
        if ckpt.exists():
            return
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{self.MODEL_ID}] Downloading RealESRGAN_x4plus.pth…")
        urllib.request.urlretrieve(_REALESRGAN_URL, str(ckpt))

    def _download_subtree(self, subdir: str, dest: Path) -> None:
        """Download one top-level folder (e.g. 'hy3dpaint') from the repo zip."""
        dest.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(_GITHUB_ZIP, timeout=300) as resp:
            data = resp.read()
        prefix = f"Hunyuan3D-2.1-main/{subdir}/"
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if not member.startswith(prefix):
                    continue
                rel = member[len(prefix):]
                if not rel:
                    continue
                target = dest / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))
        print(f"[{self.MODEL_ID}] {subdir} extracted to {dest}.")

    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id": "num_inference_steps",
                "label": "Quality",
                "type": "select",
                "default": 50,
                "options": [
                    {"value": 30, "label": "Fast"},
                    {"value": 50, "label": "Balanced"},
                    {"value": 75, "label": "High"},
                ],
                "tooltip": "Number of diffusion steps. More steps = better quality but slower.",
            },
            {
                "id": "octree_resolution",
                "label": "Mesh Resolution",
                "type": "select",
                "default": 384,
                "options": [
                    {"value": 256, "label": "Low"},
                    {"value": 384, "label": "Medium"},
                    {"value": 512, "label": "High"},
                ],
                "tooltip": "Octree resolution for mesh reconstruction. Higher = more detail, more VRAM, slower.",
            },
            {
                "id": "guidance_scale",
                "label": "Guidance Scale",
                "type": "float",
                "default": 5.0,
                "min": 1.0,
                "max": 10.0,
                "step": 0.5,
                "tooltip": "Classifier-free guidance strength. Higher = closer to the input image.",
            },
            {
                "id": "target_faces",
                "label": "Decimate (for CAD/print)",
                "type": "select",
                "default": 0,
                "options": [
                    {"value": 0, "label": "Off (full density)"},
                    {"value": 50000, "label": "50k faces"},
                    {"value": 20000, "label": "20k faces"},
                    {"value": 10000, "label": "10k faces"},
                ],
                "tooltip": "Reduce polygon count on export. Helps mesh-to-solid conversion (e.g. Fusion 360).",
            },
            {
                "id": "seed",
                "label": "Seed",
                "type": "int",
                "default": -1,
                "min": 0,
                "max": 4294967295,
                "tooltip": "Seed for reproducibility. Click shuffle for a random seed.",
            },
            {
                "id": "enable_texture",
                "label": "Generate textures (PBR)",
                "type": "select",
                "default": 0,
                "options": [
                    {"value": 0, "label": "No (shape only)"},
                    {"value": 1, "label": "Yes (paint PBR textures)"},
                ],
                "tooltip": "Paint PBR textures after the shape. Needs ~21 GB VRAM, a big first-run download, and a C++/CUDA build toolchain. Textures are ignored by CAD/STEP export.",
            },
            {
                "id": "texture_resolution",
                "label": "Texture view resolution",
                "type": "select",
                "default": 512,
                "options": [
                    {"value": 512, "label": "512 (faster)"},
                    {"value": 768, "label": "768 (sharper)"},
                ],
                "tooltip": "Per-view render resolution for texture generation. Higher = sharper, more VRAM/time.",
            },
            {
                "id": "max_num_view",
                "label": "Texture views",
                "type": "int",
                "default": 6,
                "min": 6,
                "max": 9,
                "tooltip": "Number of camera views painted and baked. More views = better coverage, slower.",
            },
            {
                "id": "texture_memory",
                "label": "Texture memory",
                "type": "select",
                "default": "balanced",
                "options": [
                    {"value": "low", "label": "Low (smallest, softest)"},
                    {"value": "balanced", "label": "Balanced (recommended)"},
                    {"value": "high", "label": "High (sharpest, needs an empty GPU)"},
                    {"value": "max", "label": "Max (4096 texture; may need shared GPU memory)"},
                ],
                "tooltip": "Caps the texture pass's VRAM so it can't spill into system RAM and crawl. Adaptive to free VRAM; this sets the ceiling — a busy GPU may drop a step lower to fit. Balanced targets a ~20 GB peak for 24 GB cards.",
            },
            {
                "id": "use_shared_vram",
                "label": "Use shared GPU memory",
                "type": "select",
                "default": 0,
                "options": [
                    {"value": 0, "label": "Off"},
                    {"value": 1, "label": "On (borrow system RAM — much slower)"},
                ],
                "tooltip": "Lets High/Max run when they exceed your VRAM by paging to system RAM over PCIe. Much slower (tens of minutes) and needs a large Windows page file. Leave Off unless you want maximum texture quality and don't mind the wait.",
            },
            {
                "id": "mesh_mode",
                "label": "Mesh cleanup",
                "type": "select",
                "default": "regular",
                "options": [
                    {"value": "regular", "label": "Regular (clean connected mesh, default)"},
                    {"value": "isotropic", "label": "Isotropic (uniform triangles; may fragment)"},
                    {"value": "bpt", "label": "BPT neural (~4k, slow, big download)"},
                ],
                "tooltip": "How the shape mesh is cleaned before texturing. Regular = quadric decimation, keeps one connected shell with clean UVs (recommended). Isotropic = uniform triangles but can shatter Hunyuan's non-watertight surfaces into many UV islands (auto-falls back to Regular if it does). BPT = neural artist topology (first use downloads ~4 GB; falls back to Regular if unavailable).",
            },
            {
                "id": "bake_normal_map",
                "label": "Bake normal map",
                "type": "select",
                "default": 0,
                "options": [
                    {"value": 0, "label": "No (recommended)"},
                    {"value": 1, "label": "Yes (experimental)"},
                ],
                "tooltip": "Experimental: bakes dense detail onto the clean base as a tangent-space normal map. Off by default - on detailed meshes the current bake can add shading artifacts (tangent-basis mismatch); a corrected high-quality bake is coming. Only applies when textures are on.",
            },
            {
                "id": "seam_fix",
                "label": "Fix texture seams",
                "type": "select",
                "default": 1,
                "options": [
                    {"value": 1, "label": "Yes (reconcile UV-seam color jumps)"},
                    {"value": 0, "label": "No (raw bake)"},
                ],
                "tooltip": "Reconcile UV-seam color jumps in the baked texture so island edges don't show hard color breaks. On by default; turn off for the raw bake. Only applies when textures are on.",
            },
            {
                "id": "debug_sheet",
                "label": "QA debug sheet",
                "type": "select",
                "default": 0,
                "options": [
                    {"value": 0, "label": "No (default)"},
                    {"value": 1, "label": "Yes (write QA sheet)"},
                ],
                "tooltip": "Optional diagnostic: writes a QA image (*_qa.png) beside each textured GLB - albedo, metallic, roughness, normal, UV layout and mesh/texture stats. Off by default; turn on when checking texture quality. Doesn't change the model. Only applies when textures are on.",
            },
        ]
