import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from scene_reconstruction.blender_entities import (
    decode_index_pass, run, validate_entities, validate_selection, visibility_statistics,
)
from scene_reconstruction.common import sha256


class EntityValidationTests(unittest.TestCase):
    def entities(self):
        return {"source_blend_sha256": "source", "entities": [
            {"id": 7, "name": "jar", "object_names": ["body", "lid"]},
            {"id": 19, "name": "floor", "object_names": ["planks"]}],
            "excluded_object_names": []}

    def test_whole_entities_have_unique_exact_mesh_assignments(self):
        payload = self.entities()
        self.assertEqual(validate_entities(payload, "source"), {"body": 7, "lid": 7, "planks": 19})
        payload["entities"][1]["object_names"].append("lid")
        with self.assertRaisesRegex(ValueError, "multiply"):
            validate_entities(payload)

    def test_foreign_scene_invalid_ids_and_exclusion_conflicts_rejected(self):
        with self.assertRaisesRegex(ValueError, "different"):
            validate_entities(self.entities(), "other")
        for bad in [0, -1, 32768, True, 2.5]:
            payload = self.entities(); payload["entities"][0]["id"] = bad
            with self.assertRaises(ValueError):
                validate_entities(payload)
        payload = self.entities(); payload["excluded_object_names"] = ["body"]
        with self.assertRaisesRegex(ValueError, "Excluded"):
            validate_entities(payload)

    def test_selection_preserves_original_sparse_ids_and_timestamps(self):
        manifest = {"width": 8, "height": 12, "frames": [
            {"index": i, "file": f"{i}.png", "time_seconds": i*.033} for i in range(5)]}
        selection = {"source_manifest_sha256": "hash", "source_frame_count": 5, "width": 8, "height": 12,
                     "frames": [dict(manifest["frames"][i]) for i in [0, 2, 4]]}
        self.assertEqual([f["index"] for f in validate_selection(selection, manifest, "hash")], [0, 2, 4])
        selection["frames"][1]["time_seconds"] += .001
        with self.assertRaisesRegex(ValueError, "timestamps"):
            validate_selection(selection, manifest, "hash")
        with self.assertRaisesRegex(ValueError, "different"):
            validate_selection(selection, manifest, "other")

    def test_integer_labels_not_color_or_interpolated_masks(self):
        actual = decode_index_pass(np.array([[0., 7.], [19., 7.]]), [7, 19])
        self.assertEqual(actual.dtype, np.uint16)
        np.testing.assert_array_equal(actual, [[0, 7], [19, 7]])
        for invalid in [np.array([[7.5]]), np.array([[11.]]), np.array([[np.nan]])]:
            with self.assertRaises(ValueError):
                decode_index_pass(invalid, [7, 19])

    def test_visibility_uses_exclusive_pixel_bounds_and_does_not_invent_absent_ids(self):
        labels = np.array([[0, 7, 7, 0], [19, 0, 7, 0], [19, 0, 0, 0]], dtype=np.uint16)
        self.assertEqual(visibility_statistics(labels), [
            {"id": 7, "area_pixels": 3, "bbox_xyxy": [1, 0, 3, 2]},
            {"id": 19, "area_pixels": 2, "bbox_xyxy": [0, 1, 1, 3]}])


class EntityWrapperTests(unittest.TestCase):
    def test_readonly_launch_exact_provenance_and_selected_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); blend = root/"scene.blend"; blend.write_bytes(b"original scene")
            Image.new("RGB", (8, 12)).save(root/"0.png")
            Image.new("RGB", (8, 12)).save(root/"1.png")
            manifest = {"schema_version": 1, "width": 8, "height": 12, "frames": [
                {"index": i, "file": f"{i}.png", "time_seconds": i*.033} for i in range(2)]}
            mp = root/"manifest.json"; mp.write_text(json.dumps(manifest))
            sp = root/"selection.json"; sp.write_text(json.dumps({"source_manifest_sha256": sha256(mp),
                "source_frame_count": 2, "width": 8, "height": 12, "frames": manifest["frames"]}))
            ep = root/"entities.json"; ep.write_text(json.dumps({"source_blend_sha256": sha256(blend),
                "entities": [{"id": 1, "name": "jar", "object_names": ["body", "lid"]}]}))
            output = root/"result"
            args = SimpleNamespace(blend=blend, manifest=mp, selection=sp, entities=ep, output=output,
                                   samples=8, resolution_scale=1., rgb=True, frames="1", render_device="METAL", blender=None)
            def launch(command, **kwargs):
                self.assertIn("--disable-autoexec", command)
                self.assertIn("--factory-startup", command)
                (output/"index.json").write_text(json.dumps({"status": "complete", "source_blend_sha256": sha256(blend)}))
                return SimpleNamespace(returncode=0)
            with mock.patch("scene_reconstruction.blender_entities.find_blender", return_value="blender"), mock.patch(
                    "scene_reconstruction.blender_entities.subprocess.run", side_effect=launch):
                run(args)
            request = json.loads((output/"request.json").read_text())
            self.assertEqual(request["render_indices"], [1])
            self.assertEqual(request["selection_sha256"], sha256(sp))
            self.assertEqual(request["render_device"], "METAL")
            self.assertEqual(blend.read_bytes(), b"original scene")
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == "__main__":
    unittest.main()
