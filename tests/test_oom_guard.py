import unittest

import oom_guard

# Both captured from real failures: the driver-level one is what a 12 GB card
# reported from the VAE decode; the allocator one is torch's own wording.
DRIVER_OOM = "CUDA error: out of memory\nCUDA kernel errors might be asynchronously reported"
ALLOC_OOM = ("CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total "
             "capacity of 11.99 GiB of which 0 bytes is free.")


class TestIsCudaOom(unittest.TestCase):
    def test_matches_driver_level_oom(self):
        self.assertTrue(oom_guard.is_cuda_oom(RuntimeError(DRIVER_OOM)))

    def test_matches_allocator_oom(self):
        self.assertTrue(oom_guard.is_cuda_oom(RuntimeError(ALLOC_OOM)))

    def test_ignores_host_memory_error(self):
        # A host-side OOM needs entirely different advice (RAM, not VRAM).
        self.assertFalse(oom_guard.is_cuda_oom(MemoryError()))
        self.assertFalse(oom_guard.is_cuda_oom(MemoryError("out of memory")))

    def test_ignores_unrelated_cuda_error(self):
        self.assertFalse(oom_guard.is_cuda_oom(
            RuntimeError("CUDA error: no kernel image is available for execution")))

    def test_ignores_ordinary_failure(self):
        self.assertFalse(oom_guard.is_cuda_oom(ValueError("bad mesh")))


class TestContextLost(unittest.TestCase):
    def test_driver_error_poisons_context(self):
        self.assertTrue(oom_guard.context_lost(RuntimeError(DRIVER_OOM)))

    def test_allocator_oom_is_recoverable(self):
        self.assertFalse(oom_guard.context_lost(RuntimeError(ALLOC_OOM)))


class TestAdvice(unittest.TestCase):
    def test_view_resolution_leads_when_raised(self):
        msg = oom_guard.advice(tex_resolution=768, max_num_view=8)
        first = [ln for ln in msg.splitlines() if ln.strip().startswith("1.")][0]
        self.assertIn("View resolution", first)   # the ~14 GB lever comes first
        self.assertIn("768", first)

    def test_views_step_present_and_counted(self):
        msg = oom_guard.advice(tex_resolution=512, max_num_view=9)
        self.assertIn("Reduce Views from 9 to 6", msg)
        self.assertNotIn("View resolution", msg)  # already at 512, nothing to lower

    def test_defaults_omit_inapplicable_steps(self):
        msg = oom_guard.advice(tex_resolution=512, max_num_view=6, shared_on=True,
                               tier="reduced")
        self.assertNotIn("View resolution", msg)
        self.assertNotIn("Reduce Views", msg)
        self.assertNotIn("Use shared GPU memory", msg)   # already on
        self.assertNotIn("Texture memory to Reduced", msg)  # already reduced
        self.assertIn("Close other GPU apps", msg)

    def test_suggests_shared_memory_only_when_off(self):
        self.assertIn("Use shared GPU memory", oom_guard.advice(shared_on=False))
        self.assertNotIn("Use shared GPU memory", oom_guard.advice(shared_on=True))

    def test_suggests_reduced_tier_only_on_standard(self):
        self.assertIn("Texture memory to Reduced", oom_guard.advice(tier="standard"))
        self.assertNotIn("Texture memory to Reduced", oom_guard.advice(tier="reduced"))

    def test_restart_note_only_when_context_lost(self):
        self.assertIn("Restart Modly", oom_guard.advice(lost_context=True))
        self.assertNotIn("Restart Modly", oom_guard.advice(lost_context=False))

    def test_includes_planner_warning(self):
        warn = "Textures need ~30 GB but only ~20 GB is available"
        self.assertIn(warn, oom_guard.advice(planner_warning=warn))

    def test_tolerates_junk_knob_values(self):
        msg = oom_guard.advice(tex_resolution=None, max_num_view="lots")
        self.assertIn("Close other GPU apps", msg)   # falls back, still useful

    def test_steps_are_numbered_contiguously(self):
        msg = oom_guard.advice(tex_resolution=768, max_num_view=8, tier="standard")
        nums = [ln.strip().split(".")[0] for ln in msg.splitlines()
                if ln.strip()[:1].isdigit()]
        self.assertEqual(nums, [str(i) for i in range(1, len(nums) + 1)])


if __name__ == "__main__":
    unittest.main()
