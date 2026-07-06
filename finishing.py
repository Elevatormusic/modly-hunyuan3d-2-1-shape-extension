"""Post-paint finishing pipeline: guarded, non-fatal stages run on the final
textured GLB. Order: seam_fix -> bake -> validate -> debug_sheet. finish()
never raises; the paint result survives even if every QA stage fails.

Extension point for Branch 2 (Blender bake replaces the bake stage) and
Branch 3 (export/optimize stages + Khronos validate gate).
"""


def finish(glb_path, obj_path, *, dense_mesh, texture_size, mesh_mode,
           bake_normal_map=False, seam_fix=True, debug_sheet=True,
           input_image_path=None, report=None):
    def _report(pct, label):
        if report:
            try:
                report(pct, label)
            except Exception:
                pass

    rep = {"seam_fix": "off", "bake": "off", "validate": "off", "debug_sheet": "off"}

    # 1. seam reconcile (albedo + MR), in place
    if seam_fix:
        try:
            import seam_fix as _seamfix
            _report(94, "Reconciling texture seams...")
            _seamfix.apply_to_glb(glb_path)
            rep["seam_fix"] = "ok"
        except Exception as exc:
            rep["seam_fix"] = f"skipped ({exc})"
            print(f"[finishing] seam fix skipped ({exc})")

    # 2. normal bake (default OFF), same gate + BPT skip as the old inline tail
    try:
        import capacity
        if capacity.should_bake(bake_normal_map, mesh_mode):
            try:
                import normal_bake
                _report(95, "Baking normal map...")
                normal_bake.bake_normal_map(dense_mesh, glb_path, size=texture_size)
                rep["bake"] = "ok"
            except Exception as exc:
                rep["bake"] = f"skipped ({exc})"
                print(f"[finishing] normal bake skipped ({exc})")
        elif bake_normal_map and mesh_mode == "bpt":
            rep["bake"] = "bpt-skip"
            print("[finishing] BPT mesh: skipping normal bake "
                  "(regenerated surface would misregister the bake)")
    except Exception as exc:
        rep["bake"] = f"skipped ({exc})"
        print(f"[finishing] bake gate skipped ({exc})")

    # 3. structural validation (logged, non-fatal)
    try:
        import glb_validate
        v = glb_validate.validate_glb(glb_path)
        rep["validate"] = "ok" if v.get("ok") else "warn"
        if not v.get("ok"):
            print(f"[finishing] validation warnings: {v.get('warnings')}")
    except Exception as exc:
        rep["validate"] = f"skipped ({exc})"
        print(f"[finishing] validation skipped ({exc})")

    # 4. QA debug sheet (side output)
    if debug_sheet:
        try:
            import debug_sheet as _sheet
            out_png = (glb_path[:-4] if glb_path.lower().endswith(".glb") else glb_path) + "_qa.png"
            _report(96, "Writing QA sheet...")
            res = _sheet.write_debug_sheet(glb_path, obj_path, out_png,
                                           input_image_path=input_image_path)
            rep["debug_sheet"] = "ok" if res else "skipped"
        except Exception as exc:
            rep["debug_sheet"] = f"skipped ({exc})"
            print(f"[finishing] debug sheet skipped ({exc})")

    return rep
