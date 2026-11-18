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


def fixture(root):
    frames_dir = root / "frames"
    frames_dir.mkdir()
    frames = []
    for index in range(6):
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
    for frame in frames:
        index = frame["index"]
        chunk, obj_id = (0, 1) if index < 3 else (3, 4)
        instances = []
        if index != 2:
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


def mapping():
    return {"objects": [{"name": "jar_01", "tracks": [
        {"concept": "containers", "chunk_start_index": 0, "id": 1},
        {"concept": "containers", "chunk_start_index": 3, "id": 4},
    ], "absent_ranges": [[2, 2]], "missing_detections": "error"}]}


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


if __name__ == "__main__":
    unittest.main()
