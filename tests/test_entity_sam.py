"""Scene-scoped SAM preparation preserves source identity and unresolved entities."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

from scene_reconstruction.common import sha256, write_json
from scene_reconstruction.entity_sam import entity_concepts, prepare, selected_frames, shared_prompts, validate_prepared
from scene_reconstruction.sam3_masks import load_prompts


class EntitySamTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "project"
        self.images = self.root / "data/frames"
        self.images.mkdir(parents=True)
        self.manifest = self.images / "manifest.json"
        rows = []
        for i in range(10):
            name = f"{i:06d}.png"
            Image.new("RGB", (8, 12), (i*10, 0, 0)).save(self.images / name)
            rows.append({"index": i, "time_seconds": i*.033, "pts_seconds": 100+i*.033,
                         "duration_seconds": .033, "file": name})
        write_json(self.manifest, {"schema_version": 1, "width": 8, "height": 12, "frame_count": 10,
                                   "frames": rows})
        self.selection = self.root / "configs/selection.json"
        write_json(self.selection, {"schema_version": 1, "source_manifest_sha256": sha256(self.manifest),
                                    "width": 8, "height": 12, "source_frame_count": 10,
                                    "selection": {"method": "uniform_timestamps", "requested_count": 4},
                                    "frames": [rows[i] for i in (0, 3, 6, 9)]})
        self.entities = self.root / "configs/entities.json"
        objects = [{"id": 1, "name": "jar_01", "label": "Blue jar", "concept": "jars",
                    "sam_text": "storage jar", "object_names": ["Jar_01_body", "Jar_01_lid"],
                    "grouping_status": "needs_review"},
                   {"id": 2, "name": "jar_02", "label": "Red jar", "concept": "jars",
                    "sam_text": "storage jar", "object_names": ["Jar_02_body"]},
                   {"id": 3, "name": "microwave", "concept": "microwave",
                    "sam_text": "microwave oven", "object_names": ["Microwave"]}]
        write_json(self.entities, {"schema_version": 1, "source_blend_sha256": "abc", "entities": objects})
        self.output = self.root / "outputs/sam_input"
        self.args = SimpleNamespace(manifest=self.manifest, selection=self.selection,
                                    entities=self.entities, output=self.output, render_index=None)

    def read(self, name):
        return json.loads((self.output / name).read_text())

    def shared_fixture(self):
        second = self.root / "configs/second_entities.json"
        write_json(second, {"schema_version": 1, "scene_key": "second", "source_blend_sha256": "def",
                            "entities": [{"id": 11, "name": "second_jar", "label": "Second scene jar",
                                          "concept": "jars", "sam_text": "storage jar", "object_names": ["Jar"]},
                                         {"id": 12, "name": "floor", "label": "Floor", "concept": "floor",
                                          "sam_text": "wood floor", "object_names": ["Boards"]}]})
        selected = json.loads(self.selection.read_text())["frames"]
        render_paths = []
        # Scene A's jar visibility peaks at frame 3, scene B's at frame 6.
        # Their combined visibility peaks at 6, so both receive that anchor.
        for name, entities, areas in [("first", self.entities, [0, 65, 20, 0]),
                                      ("second", second, [0, 5, 70, 0])]:
            path = self.root / f"outputs/{name}_render/index.json"
            rows = []
            for frame, area in zip(selected, areas):
                visible = [{"id": 1 if name == "first" else 11, "area_pixels": area,
                            "bbox_xyxy": [0, 0, 8, 12]}] if area else []
                if name == "second": visible.append({"id": 12, "area_pixels": 10})
                rows.append({"index": frame["index"], "time_seconds": frame["time_seconds"],
                             "pts_seconds": frame["pts_seconds"], "entities": visible})
            write_json(path, {"status": "complete", "frame_count": len(rows), "width": 8, "height": 12,
                              "entities_sha256": sha256(entities), "source_blend_sha256": json.loads(entities.read_text())["source_blend_sha256"],
                              "selection_sha256": sha256(self.selection), "source_manifest_sha256": sha256(self.manifest),
                              "frames": rows})
            render_paths.append(path)
        return SimpleNamespace(manifest=self.manifest, selection=self.selection,
                               entities=[self.entities, second], render_indices=render_paths,
                               output=self.root / "configs/shared_prompts.json")

    def test_sparse_ids_timestamps_and_relative_image_paths_survive(self):
        prepare(self.args)
        result = self.read("manifest.json")
        self.assertEqual([f["index"] for f in result["frames"]], [0, 3, 6, 9])
        self.assertEqual([f["time_seconds"] for f in result["frames"]], [0, .099, .198, .29700000000000004])
        self.assertEqual(result["frames"][-1]["pts_seconds"], 100.297)
        for frame in result["frames"]:
            self.assertFalse(Path(frame["file"]).is_absolute())
            self.assertEqual((self.output / frame["file"]).resolve(), (self.images / frame["canonical_file"]).resolve())
        self.assertEqual(result["source_manifest_sha256"], sha256(self.manifest))
        self.assertEqual(validate_prepared(self.output)["selected_frame_count"], 4)

    def test_concepts_deduplicated_without_erasing_repeated_entity_identity(self):
        prepare(self.args)
        prompts = load_prompts(self.output / "prompts.json", {0, 3, 6, 9})
        self.assertEqual([p["name"] for p in prompts], ["jars", "microwave"])
        result = self.read("entity_candidates.json")
        self.assertEqual(result["concepts"][0]["entity_ids"], [1, 2])
        self.assertEqual(len(result["entities"]), 3)
        for entity in result["entities"]:
            self.assertTrue(entity["mapping_required"])
            self.assertFalse(entity["use_concept_union"])
            self.assertEqual(entity["matched_tracks"], [])
            self.assertEqual(entity["status"], "needs_instance_mapping")

    def test_blender_visibility_selects_only_timestamp_not_boxes(self):
        render = self.root / "outputs/render/index.json"
        write_json(render, {"status": "complete", "entities_sha256": sha256(self.entities),
                            "selection_sha256": sha256(self.selection),
                            "frames": [{"index": i, "time_seconds": i*.033,
                                        "entities": [{"id": 1, "area_pixels": area, "bbox_xyxy": [0, 0, 2, 2]}]}
                                       for i, area in [(0, 0), (3, 10), (6, 90), (9, 20)]]})
        self.args.render_index = render
        prepare(self.args)
        prompts = self.read("prompts.json")
        self.assertEqual(prompts["objects"][0], {"name": "jars", "text": "storage jar", "frame_index": 6})
        self.assertEqual(prompts["objects"][1]["frame_index"], 0)
        self.assertEqual(prompts["anchor_provenance"]["microwave"]["source"],
                         "first_selected_frame_no_blender_visibility")
        self.assertNotIn("bbox_xyxy", json.dumps(prompts))

    def test_changed_selection_provenance_or_frame_identity_rejected(self):
        selection = json.loads(self.selection.read_text())
        selection["source_manifest_sha256"] = "wrong"
        write_json(self.selection, selection)
        with self.assertRaisesRegex(ValueError, "hash"):
            selected_frames(self.manifest, self.selection)
        selection["source_manifest_sha256"] = sha256(self.manifest)
        selection["frames"][1]["time_seconds"] += .01
        write_json(self.selection, selection)
        with self.assertRaisesRegex(ValueError, "time/image"):
            selected_frames(self.manifest, self.selection)

    def test_duplicate_ids_and_inconsistent_shared_prompts_fail(self):
        entities = json.loads(self.entities.read_text())
        entities["entities"][1]["id"] = 1
        with self.assertRaisesRegex(ValueError, "IDs"):
            entity_concepts(entities)
        entities["entities"][1]["id"] = 2
        entities["entities"][1]["sam_text"] = "vase"
        with self.assertRaisesRegex(ValueError, "identical"):
            entity_concepts(entities)

    def test_preparation_is_portable_and_tampering_is_detected(self):
        prepare(self.args)
        moved = Path(self.temporary.name) / "remote_project"
        shutil.copytree(self.root, moved)
        prepared = moved / "outputs/sam_input"
        self.assertEqual(validate_prepared(prepared)["entity_count"], 3)
        prompts = json.loads((prepared / "prompts.json").read_text())
        prompts["objects"][0]["frame_index"] = 1
        write_json(prepared / "prompts.json", prompts)
        with self.assertRaisesRegex(ValueError, "prompts changed"):
            validate_prepared(prepared)

    def test_existing_output_cannot_mix_preparations(self):
        prepare(self.args)
        with self.assertRaises(FileExistsError):
            prepare(self.args)

    def test_runner_dry_run_needs_no_cuda_and_keeps_instance_masks(self):
        prepare(self.args)
        script = Path(__file__).resolve().parents[1] / "scripts/run_entity_sam.sh"
        result = subprocess.run(["bash", str(script), str(self.output), "--dry-run"],
                                check=True, capture_output=True, text=True)
        self.assertIn("SAM preparation valid", result.stdout)
        self.assertIn("--chunk-size 0", result.stdout)
        self.assertIn("--save-instances", result.stdout)
        self.assertFalse((self.output / "sam").exists())

    def test_joint_visibility_is_symmetric_and_does_not_supply_geometry_prompts(self):
        args = self.shared_fixture()
        shared_prompts(args)
        result = json.loads(args.output.read_text())
        prompts = {row["name"]: row for row in result["objects"]}
        self.assertEqual(prompts["jars"], {"name": "jars", "text": "storage jar", "frame_index": 6})
        self.assertEqual(prompts["microwave"]["frame_index"], 0)
        self.assertTrue(result["anchor_provenance"]["microwave"]["no_rendered_visibility"])
        self.assertEqual(prompts["floor"]["frame_index"], 0)  # Equal visibility: earliest frame.
        provenance = result["anchor_provenance"]["jars"]
        self.assertEqual(provenance["source"], "maximum_combined_blender_concept_visibility")
        self.assertEqual(provenance["combined_visible_pixels"], 90)
        self.assertEqual(sorted(row["visible_pixels"] for row in provenance["per_scene_visible_pixels"]), [20, 70])
        self.assertTrue(all("render_index_sha256" in row for row in result["inputs"]))
        self.assertNotIn("bbox_xyxy", json.dumps(result))
        reversed_args = SimpleNamespace(**{**vars(args), "entities": list(reversed(args.entities)),
                                            "render_indices": list(reversed(args.render_indices)),
                                            "output": self.root / "configs/shared_reverse.json"})
        shared_prompts(reversed_args)
        reverse = json.loads(reversed_args.output.read_text())
        self.assertEqual(result, reverse)

    def test_both_scene_preparations_share_common_text_and_anchors_portably(self):
        args = self.shared_fixture()
        shared_prompts(args)
        prepared = []
        for i, entities in enumerate(args.entities):
            destination = self.root / f"outputs/scene_{i}_input"
            prepare(SimpleNamespace(manifest=self.manifest, selection=self.selection, entities=entities,
                                    output=destination, render_index=None, anchor_prompts=args.output))
            record = json.loads((destination / "preparation.json").read_text())
            self.assertEqual(record["anchor_prompts_sha256"], sha256(args.output))
            self.assertFalse(Path(record["anchor_prompts"]).is_absolute())
            prepared.append({p["name"]: p for p in load_prompts(destination / "prompts.json", {0, 3, 6, 9})})
        self.assertEqual(prepared[0]["jars"], prepared[1]["jars"])
        self.assertNotIn("floor", prepared[0])
        self.assertNotIn("microwave", prepared[1])
        moved = Path(self.temporary.name) / "remote_project"
        shutil.copytree(self.root, moved)
        self.assertEqual(validate_prepared(moved / "outputs/scene_1_input")["concept_count"], 2)
        with args.output.open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "shared anchor prompts"):
            validate_prepared(self.root / "outputs/scene_0_input")

    def test_shared_anchor_override_rejects_bad_provenance_text_or_selected_id(self):
        args = self.shared_fixture()
        shared_prompts(args)
        original = json.loads(args.output.read_text())
        for edit in ("source_hash", "selection_hash", "text", "frame", "missing_concept"):
            with self.subTest(edit=edit):
                changed = json.loads(json.dumps(original))
                if edit == "source_hash": changed["source_manifest_sha256"] = "wrong"
                if edit == "selection_hash": changed["selection_sha256"] = "wrong"
                jar = next(p for p in changed["objects"] if p["name"] == "jars")
                if edit == "text": jar["text"] = "vase"
                if edit == "frame": jar["frame_index"] = 1
                if edit == "missing_concept": changed["objects"].remove(jar)
                write_json(args.output, changed)
                preparation = SimpleNamespace(**{**vars(self.args), "anchor_prompts": args.output})
                with self.assertRaises(ValueError):
                    prepare(preparation)
                self.assertFalse(self.output.exists())

    def test_shared_visibility_rejects_foreign_grid_incomplete_and_conflicting_inputs(self):
        args = self.shared_fixture()
        original = json.loads(args.render_indices[1].read_text())
        for edit in ("grid", "hash", "incomplete", "time", "duplicate_id", "negative_area"):
            with self.subTest(edit=edit):
                changed = json.loads(json.dumps(original))
                if edit == "grid": changed["width"] = 4
                if edit == "hash": changed["selection_sha256"] = "wrong"
                if edit == "incomplete": changed["frames"].pop()
                if edit == "time": changed["frames"][1]["pts_seconds"] += .01
                if edit == "duplicate_id": changed["frames"][1]["entities"].append(changed["frames"][1]["entities"][0])
                if edit == "negative_area": changed["frames"][1]["entities"][0]["area_pixels"] = -1
                write_json(args.render_indices[1], changed)
                with self.assertRaises(ValueError):
                    shared_prompts(args)
                self.assertFalse(args.output.exists())
        entities = json.loads(args.entities[1].read_text())
        entities["entities"][0]["sam_text"] = "vase"
        write_json(args.entities[1], entities)
        original["entities_sha256"] = sha256(args.entities[1])
        write_json(args.render_indices[1], original)
        with self.assertRaisesRegex(ValueError, "conflicting SAM text"):
            shared_prompts(args)


if __name__ == "__main__":
    unittest.main()
