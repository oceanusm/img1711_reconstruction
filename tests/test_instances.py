import csv
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scene_reconstruction.common import sha256, write_json
from scene_reconstruction.compare import run as compare
from scene_reconstruction.instances import build_plan, catalog, load_sam_run, run


def fixture(root, frame_indices=None):
    frame_indices = list(range(6)) if frame_indices is None else list(frame_indices)
    frames_dir = root / "frames"
    frames_dir.mkdir()
    frames = []
    for index in frame_indices:
        relative = f"{index:06d}.png"
        Image.new("RGB", (6, 4), (30, 60, 90)).save(frames_dir / relative)
        frames.append({"index": index, "file": relative, "time_seconds": index / 30, "pts_seconds": index / 30 + 1})
    manifest_path = frames_dir / "manifest.json"
    write_json(manifest_path, {"schema_version": 1, "width": 6, "height": 4, "frames": frames})
    sam = root / "sam"
    sam.mkdir()
    write_json(sam / "metadata.json", {
        "status": "complete", "width": 6, "height": 4, "frame_count": 6,
        "manifest_sha256": sha256(manifest_path), "objects": [{"name": "containers"}], "save_instances": True,
    })
    records = []
    for position, frame in enumerate(frames):
        index = frame["index"]
        chunk, obj_id = (frame_indices[0], 1) if position < 3 else (frame_indices[3], 4)
        instances = []
        if position != 2:
            relative = f"instances/containers/chunk_{chunk:06d}/id_{obj_id:06d}/{index:06d}.png"
            path = sam / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            mask = np.zeros((4, 6), dtype=np.uint8)
            mask[1:3, 2:4] = 255
            Image.fromarray(mask).save(path)
            instances.append({"id": obj_id, "score": 0.9, "area_pixels": 4, "mask": relative})
        records.append({"object": "containers", "index": index, "chunk_start_index": chunk,
                        "time_seconds": frame["time_seconds"], "pts_seconds": frame["pts_seconds"],
                        "instances": instances})
    (sam / "frames.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    return manifest_path, sam


def mapping(frame_indices=None):
    frame_indices = list(range(6)) if frame_indices is None else list(frame_indices)
    return {"objects": [{"name": "jar_01", "tracks": [
        {"concept": "containers", "chunk_start_index": frame_indices[0], "id": 1},
        {"concept": "containers", "chunk_start_index": frame_indices[3], "id": 4},
    ], "absent_ranges": [[frame_indices[2], frame_indices[2]]], "missing_detections": "error"}]}


class InstanceCurationTest(unittest.TestCase):
    def test_assembly_joins_chunk_local_ids_and_records_manual_absence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, sam = fixture(root)
            map_path = root / "mapping.json"
            write_json(map_path, mapping())
            out = root / "objects"
            run(types.SimpleNamespace(manifest=manifest, sam_output=sam, mapping=map_path, output=out))
            self.assertEqual(len(list((out / "masks/jar_01").glob("*.png"))), 6)
            areas = [np.count_nonzero(np.asarray(Image.open(out / f"masks/jar_01/{i:06d}.png"))) for i in range(6)]
            self.assertEqual(areas, [4, 4, 0, 4, 4, 4])
            records = [json.loads(line) for line in (out / "frames.jsonl").read_text().splitlines()]
            self.assertEqual(records[2]["disposition"], "manual_absence")
            self.assertEqual(records[3]["tracks"][0]["id"], 4)
            metadata = json.loads((out / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["source_manifest_sha256"], sha256(manifest))
            self.assertEqual(metadata["manifest_sha256"], sha256(manifest))
            self.assertEqual(metadata["frames_with_cross_object_overlap"], 0)
            compare(types.SimpleNamespace(
                manifest=manifest, predicted=out / "masks", reference=out / "masks",
                objects=["jar_01"], output=root / "comparison", overlay_every=0,
            ))
            comparison = json.loads((root / "comparison/summary.json").read_text())
            self.assertEqual(comparison["objects"]["jar_01"]["mean_frame_iou"], 1.0)

    def test_unknown_chunk_is_not_implicitly_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, sam = fixture(Path(temp))
            _, _, tracks, chunks = load_sam_run(manifest, sam)
            payload = mapping()
            payload["objects"][0]["tracks"].pop()
            payload["objects"][0]["missing_detections"] = "reviewed_absent"
            with self.assertRaisesRegex(ValueError, "unreviewed/unmapped"):
                build_plan(payload, 6, tracks, chunks)

    def test_missing_detection_needs_explicit_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, sam = fixture(Path(temp))
            _, _, tracks, chunks = load_sam_run(manifest, sam)
            payload = mapping()
            payload["objects"][0]["absent_ranges"] = []
            with self.assertRaisesRegex(ValueError, "No selected detection"):
                build_plan(payload, 6, tracks, chunks)
            payload["objects"][0]["missing_detections"] = "reviewed_absent"
            self.assertEqual(len(build_plan(payload, 6, tracks, chunks)), 1)

    def test_duplicate_assignment_unknown_tracks_and_out_of_chunk_ranges_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, sam = fixture(Path(temp))
            _, _, tracks, chunks = load_sam_run(manifest, sam)
            payload = mapping()
            duplicate = json.loads(json.dumps(payload["objects"][0]))
            duplicate["name"] = "jar_02"
            payload["objects"].append(duplicate)
            with self.assertRaisesRegex(ValueError, "assigned to both"):
                build_plan(payload, 6, tracks, chunks)
            payload = mapping()
            payload["objects"][0]["tracks"][0]["id"] = 100
            with self.assertRaisesRegex(ValueError, "does not exist"):
                build_plan(payload, 6, tracks, chunks)
            payload = mapping()
            payload["objects"][0]["tracks"][0]["end_frame"] = 3
            with self.assertRaisesRegex(ValueError, "inside its source chunk"):
                build_plan(payload, 6, tracks, chunks)

    def test_hash_and_incomplete_source_coverage_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, sam = fixture(Path(temp))
            metadata = json.loads((sam / "metadata.json").read_text())
            metadata["manifest_sha256"] = "wrong"
            write_json(sam / "metadata.json", metadata)
            with self.assertRaisesRegex(ValueError, "different source"):
                load_sam_run(manifest, sam)
            metadata["manifest_sha256"] = sha256(manifest)
            write_json(sam / "metadata.json", metadata)
            lines = (sam / "frames.jsonl").read_text().splitlines()
            (sam / "frames.jsonl").write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "every concept"):
                load_sam_run(manifest, sam)

    def test_absence_conflict_marks_failed_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, sam = fixture(root)
            payload = mapping()
            payload["objects"][0]["absent_ranges"] = [[1, 2]]
            path = root / "mapping.json"
            write_json(path, payload)
            out = root / "objects"
            with self.assertRaisesRegex(ValueError, "absence conflicts"):
                run(types.SimpleNamespace(manifest=manifest, sam_output=sam, mapping=path, output=out))
            self.assertEqual(json.loads((out / "metadata.json").read_text())["status"], "failed")

    def test_missing_selected_file_is_not_an_empty_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, sam = fixture(root)
            first_mask = next(sam.rglob("*.png"))
            first_mask.unlink()
            path = root / "mapping.json"
            write_json(path, mapping())
            with self.assertRaises(FileNotFoundError):
                run(types.SimpleNamespace(manifest=manifest, sam_output=sam, mapping=path, output=root / "objects"))

    def test_catalog_includes_reviewable_previews_and_frame_intervals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, sam = fixture(root)
            out = root / "catalog"
            catalog(types.SimpleNamespace(manifest=manifest, sam_output=sam, output=out))
            inventory = json.loads((out / "catalog.json").read_text())
            self.assertEqual(len(inventory["tracks"]), 2)
            self.assertEqual(inventory["tracks"][0]["returned_frame_ranges"], [[0, 1]])
            self.assertTrue(all((out / track["preview"]).is_file() for track in inventory["tracks"]))
            self.assertTrue((out / "index.html").is_file())

    def test_sparse_catalog_assembly_and_comparison_preserve_source_ids(self):
        """Chunk-local tracking survives source-ID gaps without renumbering."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ids = [0, 7, 14, 21, 28, 35]
            manifest, sam = fixture(root, ids)
            catalog_out = root / "catalog"
            catalog(types.SimpleNamespace(manifest=manifest, sam_output=sam, output=catalog_out))
            inventory = json.loads((catalog_out / "catalog.json").read_text())
            self.assertEqual(inventory["source_manifest_sha256"], sha256(manifest))
            self.assertEqual([t["chunk_start_index"] for t in inventory["tracks"]], [0, 21])
            self.assertEqual([t["chunk_range"] for t in inventory["tracks"]], [[0, 14], [21, 35]])
            # Do not claim observations on unsampled source frames 1 through 6.
            self.assertEqual(inventory["tracks"][0]["returned_frame_ranges"], [[0, 0], [7, 7]])
            self.assertEqual(inventory["tracks"][1]["returned_frame_ranges"], [[21, 21], [28, 28], [35, 35]])
            self.assertTrue(all((catalog_out / t["preview"]).is_file() for t in inventory["tracks"]))

            map_path = root / "mapping.json"
            write_json(map_path, mapping(ids))
            assembled = root / "assembled"
            run(types.SimpleNamespace(manifest=manifest, sam_output=sam, mapping=map_path, output=assembled))
            expected_names = {f"{i:06d}.png" for i in ids}
            self.assertEqual({p.name for p in (assembled / "masks/jar_01").glob("*.png")}, expected_names)
            records = [json.loads(line) for line in (assembled / "frames.jsonl").read_text().splitlines()]
            self.assertEqual([row["index"] for row in records], ids)
            self.assertEqual([row["time_seconds"] for row in records], [i / 30 for i in ids])
            self.assertEqual(records[2]["disposition"], "manual_absence")
            self.assertEqual(records[3]["tracks"][0]["chunk_start_index"], 21)
            metadata = json.loads((assembled / "metadata.json").read_text())
            self.assertEqual(metadata["manifest_sha256"], sha256(manifest))
            self.assertEqual(metadata["source_manifest_sha256"], sha256(manifest))
            self.assertEqual(metadata["frame_count"], 6)
            self.assertEqual(metadata["mask_count"], 6)

            # Independent reference: one shifted mask gives a known imperfect
            # overlap, exercising the actual source-frame join rather than a
            # self-comparison that could conceal accidental ID renumbering.
            reference = root / "reference"
            (reference / "masks/jar_01").mkdir(parents=True)
            for index in ids:
                mask = np.zeros((4, 6), dtype=np.uint8)
                if index != 14:
                    left = 3 if index == 28 else 2
                    mask[1:3, left:left+2] = 255
                Image.fromarray(mask).save(reference / f"masks/jar_01/{index:06d}.png")
            write_json(reference / "metadata.json", {"status": "complete", "manifest_sha256": sha256(manifest)})
            comparison_out = root / "comparison"
            compare(types.SimpleNamespace(manifest=manifest, predicted=assembled / "masks",
                                          reference=reference / "masks", objects=["jar_01"],
                                          output=comparison_out, overlay_every=1))
            summary = json.loads((comparison_out / "summary.json").read_text())
            self.assertEqual(summary["frame_manifest_sha256"], sha256(manifest))
            self.assertEqual(summary["provenance_verification"],
                             {"predicted": "manifest hash and completion verified",
                              "reference": "manifest hash and completion verified"})
            scores = summary["objects"]["jar_01"]
            self.assertEqual(scores["frames"], 6)
            self.assertEqual(scores["both_empty_frames"], 1)
            self.assertEqual(scores["frames_with_either_object_visible"], 5)
            self.assertAlmostEqual(scores["mean_frame_iou"], 13 / 15)
            self.assertAlmostEqual(scores["pixel_weighted_iou"], 9 / 11)
            with (comparison_out / "jar_01.csv").open(newline="") as handle:
                compared = list(csv.DictReader(handle))
            self.assertEqual([int(row["frame_index"]) for row in compared], ids)
            self.assertAlmostEqual(float(compared[4]["iou"]), 1 / 3)
            self.assertEqual({p.name for p in (comparison_out / "overlays/jar_01").glob("*.png")}, expected_names)

    def test_sparse_chunks_may_skip_unselected_frames_but_not_selected_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ids = [0, 7, 14, 21, 28, 35]
            manifest, sam = fixture(root, ids)
            _, _, _, chunks = load_sam_run(manifest, sam)
            self.assertEqual(chunks[("containers", 0)], {0, 7, 14})
            self.assertEqual(chunks[("containers", 21)], {21, 28, 35})
            path = sam / "frames.jsonl"
            original = path.read_text()
            records = [json.loads(line) for line in original.splitlines()]
            # Every frame still exists, but interleaving another session into
            # the selected sequence must invalidate the remaining first chunk.
            records[1]["chunk_start_index"] = 7
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
            with self.assertRaisesRegex(ValueError, "contiguous interval"):
                load_sam_run(manifest, sam)
            path.write_text("\n".join(line for line in original.splitlines()
                                      if json.loads(line)["index"] != 28) + "\n")
            with self.assertRaisesRegex(ValueError, "every concept"):
                load_sam_run(manifest, sam)

    def test_sparse_absence_ranges_use_selected_source_ids_not_positions(self):
        with tempfile.TemporaryDirectory() as temp:
            ids = [0, 7, 14, 21, 28, 35]
            manifest, sam = fixture(Path(temp), ids)
            _, _, tracks, chunks = load_sam_run(manifest, sam)
            plan = build_plan(mapping(ids), len(ids), tracks, chunks, ids)[0]
            self.assertEqual(set(plan["active"]), set(ids))
            self.assertEqual(plan["absent"], {14})
            wrong = mapping(ids)
            wrong["objects"][0]["absent_ranges"] = [[2, 2]]
            with self.assertRaisesRegex(ValueError, "Absence range"):
                build_plan(wrong, len(ids), tracks, chunks, ids)
            interval = mapping(ids)
            interval["objects"][0]["absent_ranges"] = [[7, 21]]
            plan = build_plan(interval, len(ids), tracks, chunks, ids)[0]
            self.assertEqual(plan["absent"], {7, 14, 21})
            self.assertNotIn(8, plan["active"])

    def test_sparse_missing_selected_mask_is_rejected_before_comparison_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ids = [0, 7, 14, 21, 28, 35]
            manifest, sam = fixture(root, ids)
            map_path = root / "mapping.json"
            write_json(map_path, mapping(ids))
            assembled = root / "assembled"
            run(types.SimpleNamespace(manifest=manifest, sam_output=sam, mapping=map_path, output=assembled))
            (assembled / "masks/jar_01/000028.png").unlink()
            output = root / "comparison"
            with self.assertRaisesRegex(FileNotFoundError, "000028"):
                compare(types.SimpleNamespace(manifest=manifest, predicted=assembled / "masks",
                                              reference=assembled / "masks", objects=["jar_01"],
                                              output=output, overlay_every=0))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
