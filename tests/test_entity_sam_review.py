"""Concept unions stay distinct from physical-instance evaluation."""
import copy
import csv
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from scene_reconstruction.common import sha256, write_json
from scene_reconstruction.entity_sam_review import (
    SCENES, _html, _sprite, concept_inventory, concept_label_image,
    overlap_metrics, run, shared_references, summarize,
)


class ConceptMathTests(unittest.TestCase):
    def test_union_groups_entities_without_assigning_instances(self):
        entities = [{"id": 2, "name": "jar_left", "concept": "jar", "sam_text": "glass jar"},
                    {"id": 8, "name": "jar_right", "concept": "jar", "sam_text": "glass jar"},
                    {"id": 5, "name": "floor", "concept": "floor", "sam_text": "floor"}]
        groups = concept_inventory(entities)
        labels = np.array([[0, 2, 8], [5, 8, 0]], dtype=np.uint16)
        concepts, ordinals = concept_label_image(labels, groups)
        np.testing.assert_array_equal(concepts == ordinals["jar"], [[0, 1, 1], [0, 1, 0]])
        self.assertEqual(groups["jar"]["entities"], ["jar_left", "jar_right"])
        with self.assertRaisesRegex(ValueError, "unknown entity"):
            concept_label_image(np.array([[99]], dtype=np.uint16), groups)

    def test_metrics_use_blender_prediction_sam_reference(self):
        blender = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
        sam = np.array([[0, 1, 1], [0, 0, 1]], dtype=bool)
        result = overlap_metrics(blender, sam)
        self.assertEqual((result["intersection"], result["union"]), (1, 4))
        self.assertEqual(result["iou"], .25)
        self.assertEqual(result["dice"], .4)
        self.assertEqual(result["precision_vs_sam"], .5)
        self.assertAlmostEqual(result["recall_vs_sam"], 1/3)
        with self.assertRaisesRegex(ValueError, "same two-dimensional"):
            overlap_metrics(blender, np.zeros((6,), dtype=bool))

    def test_empty_predictions_are_not_absence_evidence(self):
        blank, full = np.zeros((2, 2), bool), np.ones((2, 2), bool)
        both_empty, sam_empty = overlap_metrics(blank, blank), overlap_metrics(full, blank)
        self.assertIsNone(both_empty["iou"])
        self.assertIsNone(both_empty["dice"])
        self.assertEqual(sam_empty["iou"], 0)
        self.assertEqual(sam_empty["presence"], "sam_empty_unresolved")
        self.assertIsNone(sam_empty["recall_vs_sam"])
        stats = summarize([both_empty, sam_empty])
        self.assertEqual(stats["mean_frame_iou"], 0)
        self.assertEqual(stats["frames_with_nonempty_union"], 1)
        self.assertEqual(stats["both_empty_frames"], 1)
        self.assertIsNone(summarize([both_empty])["pixel_pooled_iou"])

    def test_shared_reference_requires_same_prompt_anchor_and_pixels(self):
        first = {"concepts": {"jar": {"text": "jar", "anchor_frame": 0,
            "entities": ["a"], "frames": [{"index": 7, "sam_pixel_sha256": "same"}]}}}
        second = copy.deepcopy(first)
        second["concepts"]["jar"]["entities"] = ["x", "y"]
        self.assertTrue(shared_references([first, second])["jar"]["reference_comparable"])
        second["concepts"]["jar"]["frames"][0]["sam_pixel_sha256"] = "different"
        self.assertEqual(shared_references([first, second])["jar"]["differing_frame_ids"], [7])
        second = copy.deepcopy(first)
        second["concepts"]["jar"]["anchor_frame"] = 14
        self.assertFalse(shared_references([first, second])["jar"]["reference_comparable"])

    def test_sprite_colors_and_transparency(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sprite.png"
            b = np.array([[0, 0, 1, 1]], dtype=bool)
            s = np.array([[0, 1, 0, 1]], dtype=bool)
            _sprite(b, s, (4, 1), path)
            with Image.open(path) as image:
                pixels = np.asarray(image)
            np.testing.assert_array_equal(pixels[0, 8:],
                [[0, 0, 0, 0], [232, 70, 209, 255], [40, 215, 231, 255], [255, 255, 255, 255]])


class ConceptPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "data/frames").mkdir(parents=True)
        self.source = []
        for index in (0, 7, 14):
            path = self.root / f"data/frames/{index:06d}.png"
            Image.new("RGB", (4, 4), (20, 30, 40)).save(path)
            self.source.append({"index": index, "time_seconds": index/30, "pts_seconds": index/30,
                                "file": path.name})
        for spec in SCENES:
            self.make_scene(*spec)

    def make_scene(self, key, label, config):
        stage = self.root / "outputs/entity_comparison_1711" / key
        (stage / "render/frames").mkdir(parents=True)
        (stage / "render/rgb").mkdir()
        (stage / "sam/masks/jar").mkdir(parents=True)
        (stage / "sam/masks/floor").mkdir(parents=True)
        entities = []
        for identity, name, concept in ((1, "left_jar", "jar"), (3, "right_jar", "jar"), (5, "floor", "floor")):
            entities.append({"id": identity, "name": name, "label": name, "kind": "object",
                             "concept": concept, "sam_text": concept, "object_names": [name],
                             "grouping_status": "name_supported", "notes": "Test entity"})
        entities_path = self.root / "configs" / config
        write_json(entities_path, {"schema_version": 1, "scene_key": key, "entities": entities,
                                  "source_blend_sha256": "a"*64, "source_objects_sha256": "b"*64})
        manifest_path = stage / "sam_inputs/manifest.json"
        selected = [dict(frame, file=os.path.relpath(self.root / "data/frames" / frame["file"], manifest_path.parent))
                    for frame in self.source]
        write_json(manifest_path, {"schema_version": 1, "width": 4, "height": 4, "frames": selected,
                                  "source_manifest_sha256": "c"*64, "selection_sha256": "d"*64})
        frame_rows, records = [], []
        for frame in self.source:
            index = frame["index"]
            labels = np.zeros((4, 4), dtype=np.uint16)
            if index != 14:
                labels[0, :2] = 1
                labels[0, 2:] = 3
                labels[3, :] = 5
            np.savez(stage / f"render/frames/{index:06d}.npz", entity_ids=labels)
            Image.new("RGB", (4, 4), (80, 90, 100)).save(stage / f"render/rgb/{index:06d}.png")
            frame_rows.append({"index": index, "time_seconds": frame["time_seconds"],
                               "pts_seconds": frame["pts_seconds"],
                               "entity_file": f"frames/{index:06d}.npz", "rgb_file": f"rgb/{index:06d}.png"})
            for concept in ("jar", "floor"):
                mask = np.zeros((4, 4), dtype=np.uint8)
                if concept == "jar" and index != 14:
                    mask[0, :2] = 255
                if key == "vggt_guided" and concept == "jar" and index == 7:
                    mask[0, 2:] = 255
                mask_path = f"masks/{concept}/{index:06d}.png"
                Image.fromarray(mask).save(stage / "sam" / mask_path)
                records.append({"index": index, "time_seconds": frame["time_seconds"],
                    "pts_seconds": frame["pts_seconds"], "object": concept, "chunk_start_index": 0,
                    "mask": mask_path, "area_pixels": int(np.count_nonzero(mask)), "instances": []})
        write_json(stage / "render/index.json", {"status": "complete", "width": 4, "height": 4,
            "frame_count": 3, "frames": frame_rows, "source_blend_sha256": "a"*64,
            "source_manifest_sha256": "c"*64, "selection_sha256": "d"*64,
            "entities_sha256": sha256(entities_path)})
        write_json(stage / "sam/metadata.json", {"status": "complete", "manifest_sha256": sha256(manifest_path),
            "width": 4, "height": 4, "frame_count": 3, "objects": [
                {"name": c, "text": c, "frame_index": 0} for c in ("jar", "floor")]})
        (stage / "sam/frames.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def args(self):
        return SimpleNamespace(root=self.root, output=self.root / "review", preview_height=32)

    def test_complete_pair_reports_scope_and_detects_reference_difference(self):
        run(self.args())
        report = json.loads((self.root / "review/review.json").read_text())
        self.assertEqual(report["status"], "complete")
        original = report["scenes"][0]
        self.assertEqual(original["concepts"]["jar"]["summary"]["mean_frame_iou"], .5)
        self.assertEqual(original["concepts"]["jar"]["summary"]["both_empty_frames"], 1)
        self.assertEqual(original["zero_detection_concepts"], ["floor"])
        self.assertEqual(report["shared_references"]["jar"]["differing_frame_ids"], [7])
        self.assertTrue(report["shared_references"]["floor"]["reference_comparable"])
        with (self.root / "review/per_frame.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 12)
        page = (self.root / "review/index.html").read_text()
        self.assertNotIn("__REPORT__", page)
        self.assertIn("Empty SAM masks do not establish", page)
        self.assertIn("await Promise.all(needed.map(ready))", page)
        with self.assertRaises(FileExistsError):
            run(self.args())

    def test_failed_transfer_is_not_interpreted_as_empty_mask(self):
        path = self.root / "outputs/entity_comparison_1711/before_vqa/sam/masks/jar/000007.png"
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing mask"):
            run(self.args())
        self.assertEqual(json.loads((self.root / "review/metadata.json").read_text())["status"], "failed")

    def test_mask_grid_mismatch_fails_instead_of_resampling_for_scores(self):
        path = self.root / "outputs/entity_comparison_1711/before_vqa/sam/masks/jar/000007.png"
        Image.new("L", (2, 2)).save(path)
        with self.assertRaisesRegex(ValueError, "resampling would hide"):
            run(self.args())

    def test_embedded_json_cannot_close_script_element(self):
        page = _html({"text": "</script><script>unsafe()</script>"})
        self.assertNotIn("</script><script>unsafe()", page)
        self.assertIn("\\u003c/script>", page)


if __name__ == "__main__":
    unittest.main()
