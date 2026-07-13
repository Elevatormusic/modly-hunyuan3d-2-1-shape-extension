import unittest
import output_modes as om

SCHEMA = {"octree_resolution", "enable_texture", "texture_resolution", "max_num_view",
          "mesh_mode", "bake_normal_map", "seam_fix", "saturation"}


class TestResolveParams(unittest.TestCase):
    def test_render_max_overlays_bundle(self):
        out = om.resolve_params("render_max", {"octree_resolution": 256, "foo": 1}, SCHEMA)
        self.assertEqual(out["octree_resolution"], 512)
        self.assertEqual(out["texture_resolution"], 768)
        self.assertEqual(out["max_num_view"], 8)
        self.assertEqual(out["_face_target"], 100000)
        self.assertEqual(out["foo"], 1)  # untouched raw key preserved

    def test_render_balanced_bundle(self):
        out = om.resolve_params("render_balanced", {}, SCHEMA)
        self.assertEqual(out["octree_resolution"], 384)
        self.assertEqual(out["texture_resolution"], 512)
        self.assertEqual(out["max_num_view"], 6)
        self.assertNotIn("_game_ready", out)

    def test_custom_passthrough(self):
        raw = {"octree_resolution": 256, "mesh_mode": "isotropic"}
        self.assertEqual(om.resolve_params("custom", dict(raw), SCHEMA), raw)

    def test_unknown_mode_passthrough(self):
        raw = {"a": 1}
        self.assertEqual(om.resolve_params("wat", dict(raw), SCHEMA), raw)

    def test_saturation_gated_by_schema(self):
        self.assertEqual(om.resolve_params("render_max", {}, SCHEMA)["saturation"], "subtle")
        self.assertNotIn("saturation", om.resolve_params("render_max", {}, SCHEMA - {"saturation"}))

    def test_internal_keys_always_written(self):
        out = om.resolve_params("game_ready", {}, set())  # empty schema
        self.assertEqual(out["_face_target"], 100000)
        self.assertTrue(out["_game_ready"])
        self.assertNotIn("saturation", out)  # UI key gated out

    def test_game_ready_flag_only_on_game_ready(self):
        self.assertTrue(om.resolve_params("game_ready", {}, SCHEMA).get("_game_ready"))
        self.assertNotIn("_game_ready", om.resolve_params("render_max", {}, SCHEMA))

    def test_does_not_mutate_input(self):
        raw = {"octree_resolution": 256}
        om.resolve_params("render_max", raw, SCHEMA)
        self.assertEqual(raw, {"octree_resolution": 256})


if __name__ == "__main__":
    unittest.main()
