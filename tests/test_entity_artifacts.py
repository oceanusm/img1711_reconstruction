"""Exact entity-label bridges and honest previews on sparse source timestamps."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scene_reconstruction.common import sha256, write_json
from scene_reconstruction.entity_artifacts import export_masks, load_bundle, load_labels, palette, preview


class EntityArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        images = self.root / "frames"
        images.mkdir()
        canonical = []
        for index in range(15):
            filename = f"{index:06d}.png"
            Image.new("RGB", (8, 12), (10, 20, 30)).save(images / filename)
            canonical.append({"index": index, "file": filename, "time_seconds": index*.033,
                              "pts_seconds": 1+index*.033, "duration_seconds": .033})
        self.manifest = images / "manifest.json"
        write_json(self.manifest, {"schema_version": 1, "width": 8, "height": 12,
                                   "frame_count": 15, "frames": canonical})
        self.ids = [0, 7, 14]
        self.selection = self.root / "selection.json"
        write_json(self.selection, {"schema_version": 1, "source_manifest_sha256": sha256(self.manifest),
                                    "source_frame_count": 15, "width": 8, "height": 12,
                                    "frames": [canonical[i] for i in self.ids]})
        self.subset = self.root / "sam_input/manifest.json"
        chosen = []
        for index in self.ids:
            frame = dict(canonical[index])
            frame["file"] = Path(os.path.relpath(images / frame["file"], self.subset.parent)).as_posix()
            chosen.append(frame)
        write_json(self.subset, {"schema_version": 1, "scope": "selected_canonical_frames",
                                 "width": 8, "height": 12, "frame_count": 3,
                                 "source_manifest_sha256": sha256(self.manifest),
                                 "source_manifest": Path(os.path.relpath(self.manifest, self.subset.parent)).as_posix(),
                                 "selection_sha256": sha256(self.selection), "frames": chosen})
        self.entities = self.root / "entities.json"
        entries = []
        for index, name, members, kind in [(1, "jar", ["JarBody", "JarLid"], "object"),
                                           (7, "floor", ["FloorBoards"], "surface")]:
            entries.append({"id": index, "name": name, "label": name.title(), "kind": kind,
                            "object_names": members, "concept": name, "sam_text": name,
                            "grouping_status": "name_supported", "notes": "Whole semantic entity."})
        write_json(self.entities, {"schema_version": 1, "scene_key": "test_scene",
                                   "source_blend_sha256": "a"*64, "source_objects_sha256": "b"*64,
                                   "entities": entries})
        self.render = self.root / "render/index.json"
        (self.render.parent / "labels").mkdir(parents=True)
        (self.render.parent / "rgb").mkdir()
        self.expected = {}
        rendered = []
        for index in self.ids:
            labels = np.zeros((12, 8), dtype=np.uint16)
            labels[10:, :] = 7
            if index != 7:
                labels[3:6, 2:5] = 1
            self.expected[index] = labels
            entity_file = f"labels/{index:06d}.npz"
            rgb_file = f"rgb/{index:06d}.png"
            np.savez_compressed(self.render.parent / entity_file, entity_ids=labels)
            Image.new("RGB", (8, 12), (90, 100, 110)).save(self.render.parent / rgb_file)
            rendered.append({"index": index, "time_seconds": canonical[index]["time_seconds"],
                             "pts_seconds": canonical[index]["pts_seconds"],
                             "entity_file": entity_file, "rgb_file": rgb_file})
        write_json(self.render, {"status": "complete", "frame_count": 3, "width": 8, "height": 12,
                                 "source_manifest_sha256": sha256(self.manifest),
                                 "entities_sha256": sha256(self.entities), "source_blend_sha256": "a"*64,
                                 "selection_sha256": sha256(self.selection), "frames": rendered,
                                 "label_semantics": "Visible first-hit entity ID; no antialiasing",
                                 "rendering": {"engine": "CYCLES", "samples": 8}})

    def export_args(self, **kwargs):
        values = {"render_index": self.render, "entities": self.entities, "manifest": self.subset,
                  "output": self.root / "masks_out", "objects": None}
        values.update(kwargs)
        return SimpleNamespace(**values)

    def preview_args(self, **kwargs):
        values = {"render_index": self.render, "entities": self.entities, "manifest": self.manifest,
                  "output": self.root / "preview", "max_height": 32,
                  "sam_masks": None, "sam_manifest": None}
        values.update(kwargs)
        return SimpleNamespace(**values)

    def sam_fixture(self, *, assembled=True):
        output = self.root / "assembled"
        (output / "masks/jar").mkdir(parents=True)
        for index in self.ids:
            Image.fromarray(np.full((12, 8), 255, dtype=np.uint8)).save(output / f"masks/jar/{index:06d}.png")
        metadata = {"status": "complete", "manifest_sha256": sha256(self.subset),
                    "objects": ["jar"], "width": 8, "height": 12}
        if assembled:
            metadata["mapping_sha256"] = "c"*64
        write_json(output / "metadata.json", metadata)
        return output / "masks"

    def fake_encoder(self, args):
        """Capture composed pixels; encoding itself is exercised by the real run."""
        captures = {}
        def encode(command, **kwargs):
            concat = Path(command[command.index("-i")+1])
            captures["concat"] = concat.read_text()
            captures["first"] = np.asarray(Image.open(concat.parent / "000000.png")).copy()
            Path(command[-1]).write_bytes(b"encoding stub")
            return SimpleNamespace(returncode=0)
        with patch("scene_reconstruction.entity_artifacts.video_sync_args", return_value=["-fps_mode", "vfr"]), patch(
                "scene_reconstruction.entity_artifacts.subprocess.run", side_effect=encode):
            preview(args)
        return captures

    def test_binary_masks_keep_exact_grid_grouped_id_and_explicit_absence(self):
        args = self.export_args()
        export_masks(args)
        for name, entity_id in [("jar", 1), ("floor", 7)]:
            files = list((args.output / "masks" / name).glob("*.png"))
            self.assertEqual({p.name for p in files}, {f"{i:06d}.png" for i in self.ids})
            for index in self.ids:
                with Image.open(args.output / f"masks/{name}/{index:06d}.png") as image:
                    self.assertEqual(image.size, (8, 12))
                    values = np.asarray(image)
                self.assertEqual(values.dtype, np.uint8)
                np.testing.assert_array_equal(values, (self.expected[index] == entity_id).astype(np.uint8)*255)
        self.assertEqual(np.asarray(Image.open(args.output / "masks/jar/000007.png")).sum(), 0)
        metadata = json.loads((args.output / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["objects"]["jar"], ["JarBody", "JarLid"])
        self.assertEqual(metadata["manifest_sha256"], sha256(self.subset))
        self.assertEqual(metadata["source_manifest_sha256"], sha256(self.manifest))
        self.assertEqual(metadata["render_index_sha256"], sha256(self.render))
        self.assertEqual(metadata["selection_sha256"], sha256(self.selection))

    def test_explicit_entity_subset_does_not_erase_other_ids_before_validation(self):
        args = self.export_args(objects=["jar"])
        export_masks(args)
        self.assertEqual({p.name for p in (args.output / "masks").iterdir()}, {"jar"})
        with self.assertRaisesRegex(ValueError, "absent"):
            export_masks(self.export_args(objects=["unknown"], output=self.root / "unknown"))
        with self.assertRaises(FileExistsError):
            export_masks(args)

    def test_wrong_label_shape_dtype_and_unknown_id_are_rejected(self):
        path = self.root / "bad.npz"
        for values in [np.zeros((6, 4), dtype=np.uint16), np.zeros((12, 8), dtype=np.float32),
                       np.full((12, 8), 9, dtype=np.uint16)]:
            np.savez_compressed(path, entity_ids=values)
            with self.assertRaisesRegex(ValueError, "grid, dtype or entity IDs"):
                load_labels(path, (12, 8), [1, 7])

    def test_missing_label_file_never_becomes_absence(self):
        (self.render.parent / "labels/000007.npz").unlink()
        args = self.export_args()
        with self.assertRaises(FileNotFoundError):
            export_masks(args)
        self.assertFalse(args.output.exists())

    def test_invalid_later_label_marks_partial_mask_export_failed(self):
        np.savez_compressed(self.render.parent / "labels/000007.npz",
                            entity_ids=np.full((12, 8), 99, dtype=np.uint16))
        args = self.export_args()
        with self.assertRaisesRegex(ValueError, "entity IDs"):
            export_masks(args)
        metadata = json.loads((args.output / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "failed")
        self.assertIn("ValueError", metadata["error"])
        self.assertTrue((args.output / "masks/jar/000000.png").is_file())
        self.assertFalse((args.output / "masks/jar/000007.png").exists())

    def test_render_provenance_grid_and_timestamp_checks(self):
        original = json.loads(self.render.read_text())
        changes = [("status", "running"), ("entities_sha256", "f"*64),
                   ("source_blend_sha256", "f"*64), ("source_manifest_sha256", "f"*64),
                   ("width", 4)]
        for key, value in changes:
            with self.subTest(key=key):
                changed = dict(original); changed[key] = value
                write_json(self.render, changed)
                with self.assertRaises(ValueError):
                    load_bundle(self.render, self.entities, self.manifest)
        changed = json.loads(json.dumps(original))
        changed["frames"][1]["time_seconds"] += .01
        write_json(self.render, changed)
        with self.assertRaisesRegex(ValueError, "timestamps"):
            load_bundle(self.render, self.entities, self.manifest)

    def test_sparse_export_requires_exact_sam_selection_and_timing(self):
        original = json.loads(self.subset.read_text())
        for edit in ("hash", "missing_frame", "time", "grid"):
            with self.subTest(edit=edit):
                changed = json.loads(json.dumps(original))
                if edit == "hash": changed["selection_sha256"] = "f"*64
                if edit == "missing_frame": changed["frames"].pop(1)
                if edit == "time": changed["frames"][1]["time_seconds"] += .01
                if edit == "grid": changed["height"] = 6
                write_json(self.subset, changed)
                with self.assertRaises(ValueError):
                    load_bundle(self.render, self.entities, self.subset, sparse=True)

    def test_preview_without_sam_leaves_source_untinted_and_labels_it_honestly(self):
        args = self.preview_args()
        captures = self.fake_encoder(args)
        metadata = json.loads((args.output / "preview.json").read_text())
        self.assertFalse(metadata["sam_present"])
        self.assertEqual(metadata["reviewed_sam_entity_count"], 0)
        self.assertEqual(metadata["frame_indices"], self.ids)
        self.assertEqual(metadata["source_timestamps_seconds"], [i*.033 for i in self.ids])
        self.assertAlmostEqual(metadata["duration_seconds"], .495)
        np.testing.assert_array_equal(captures["first"][64, 11], [10, 20, 30])
        self.assertIn("SAM has not run yet", (args.output / "index.html").read_text())
        self.assertIn("duration 0.231000000", captures["concat"])
        self.assertEqual(captures["concat"].count("file '000002.png'"), 2)

    def test_preview_accepts_only_assembled_sam_and_uses_its_actual_masks(self):
        masks = self.sam_fixture()
        args = self.preview_args(sam_masks=masks, sam_manifest=self.subset)
        captures = self.fake_encoder(args)
        metadata = json.loads((args.output / "preview.json").read_text())
        self.assertTrue(metadata["sam_present"])
        self.assertEqual(metadata["reviewed_sam_entity_count"], 1)
        entities = json.loads(self.entities.read_text())["entities"]
        expected = (.5*np.array([10, 20, 30]) + .5*palette(entities)[1]).astype(np.uint8)
        np.testing.assert_array_equal(captures["first"][64, 11], expected)
        self.assertIn("Reviewed SAM masks", (args.output / "index.html").read_text())

    def test_unassembled_concept_masks_cannot_be_presented_as_entity_sam(self):
        args = self.preview_args(sam_masks=self.sam_fixture(assembled=False), sam_manifest=self.subset)
        with self.assertRaisesRegex(ValueError, "assembled physical-entity"):
            preview(args)
        self.assertFalse(args.output.exists())

    def test_sam_preview_rejects_altered_subset_timing_or_grid(self):
        masks = self.sam_fixture()
        original = json.loads(self.subset.read_text())
        for edit in ("time", "grid"):
            with self.subTest(edit=edit):
                changed = json.loads(json.dumps(original))
                if edit == "time": changed["frames"][1]["time_seconds"] += .01
                if edit == "grid": changed["width"] = 4
                write_json(self.subset, changed)
                metadata = json.loads((masks.parent / "metadata.json").read_text())
                metadata["manifest_sha256"] = sha256(self.subset)
                write_json(masks.parent / "metadata.json", metadata)
                args = self.preview_args(output=self.root / edit, sam_masks=masks, sam_manifest=self.subset)
                with self.assertRaises(ValueError):
                    self.fake_encoder(args)

    def test_missing_sam_frame_and_swapped_source_image_cannot_be_hidden(self):
        masks = self.sam_fixture()
        missing = masks / "jar/000014.png"
        missing.unlink()
        args = self.preview_args(sam_masks=masks, sam_manifest=self.subset)
        with self.assertRaisesRegex(FileNotFoundError, "14"):
            preview(args)
        self.assertFalse(args.output.exists())
        Image.fromarray(np.full((12, 8), 255, dtype=np.uint8)).save(missing)
        subset = json.loads(self.subset.read_text())
        subset["frames"][0]["file"] = subset["frames"][1]["file"]
        write_json(self.subset, subset)
        metadata = json.loads((masks.parent / "metadata.json").read_text())
        metadata["manifest_sha256"] = sha256(self.subset)
        write_json(masks.parent / "metadata.json", metadata)
        with self.assertRaisesRegex(ValueError, "image path"):
            preview(args)
        self.assertFalse(args.output.exists())

    def test_preview_rejects_source_and_render_rgb_grid_mismatches(self):
        for kind in ("source", "render"):
            with self.subTest(kind=kind):
                path = (self.manifest.parent / "000000.png" if kind == "source"
                        else self.render.parent / "rgb/000000.png")
                Image.new("RGB", (4, 12)).save(path)
                args = self.preview_args(output=self.root / f"preview_bad_{kind}")
                with self.assertRaises(ValueError):
                    self.fake_encoder(args)
                Image.new("RGB", (8, 12)).save(path)


if __name__ == "__main__":
    unittest.main()
