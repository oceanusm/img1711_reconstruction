import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scene_reconstruction.common import load_manifest
from scene_reconstruction.common import sha256
from scene_reconstruction.compare import mask_metrics, read_mask, run


class MaskComparisonTests(unittest.TestCase):
    def test_empty_frames_do_not_inflate_iou(self):
        empty = np.zeros((4, 4), dtype=bool)
        present = empty.copy()
        present[1:3, 1:3] = True
        self.assertIsNone(mask_metrics(empty, empty)["iou"])
        self.assertEqual(mask_metrics(present, empty)["iou"], 0)
        self.assertEqual(mask_metrics(present, present)["iou"], 1)

    def test_partial_overlap_and_centroid(self):
        a, b = np.zeros((4, 4), bool), np.zeros((4, 4), bool)
        a[:2, :2], b[1:3, 1:3] = True, True
        metrics = mask_metrics(a, b)
        self.assertAlmostEqual(metrics["iou"], 1 / 7)
        self.assertAlmostEqual(metrics["dice"], 1 / 4)
        self.assertAlmostEqual(metrics["centroid_error_px"], np.sqrt(2))

    def test_rejects_resizing_soft_masks_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.fromarray(np.full((3, 4), 127, np.uint8)).save(path)
            with self.assertRaises(ValueError):
                read_mask(path, (4, 3))
            with self.assertRaises(ValueError):
                read_mask(path, (3, 4))
            with self.assertRaises(FileNotFoundError):
                read_mask(path.with_name("absent.png"), (4, 3))

    def test_complete_comparison_records_timestamps_and_overlay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "000000.jpg"
            Image.new("RGB", (4, 3), "white").save(image)
            manifest = {"schema_version": 1, "width": 4, "height": 3,
                        "frames": [{"index": 0, "file": image.name, "time_seconds": 0.0}]}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            for name in ["sam", "blender"]:
                (root / name / "masks/cup").mkdir(parents=True)
                Image.new("L", (4, 3), 255).save(root / name / "masks/cup/000000.png")
            run(argparse.Namespace(manifest=manifest_path, predicted=root / "sam/masks", reference=root / "blender/masks",
                                   objects=["cup"], output=root / "report", overlay_every=1))
            summary = json.loads((root / "report/summary.json").read_text())
            self.assertEqual(summary["objects"]["cup"]["mean_frame_iou"], 1)
            self.assertTrue((root / "report/overlays/cup/000000.png").is_file())
            metadata_path = root / "sam/metadata.json"
            metadata_path.write_text(json.dumps({"status": "complete", "manifest_sha256": "wrong"}))
            with self.assertRaises(ValueError):
                run(argparse.Namespace(manifest=manifest_path, predicted=root / "sam/masks", reference=root / "blender/masks",
                                       objects=["cup"], output=root / "wrong_source", overlay_every=0))
            metadata_path.write_text(json.dumps({"status": "complete", "manifest_sha256": sha256(manifest_path)}))
            (root / "ignore").mkdir()
            Image.new("L", (4, 3), 255).save(root / "ignore/000000.png")
            run(argparse.Namespace(manifest=manifest_path, predicted=root / "sam/masks", reference=root / "blender/masks",
                                   objects=["cup"], output=root / "ignored", overlay_every=0, ignore_masks=root / "ignore"))
            ignored = json.loads((root / "ignored/summary.json").read_text())
            self.assertIsNone(ignored["objects"]["cup"]["mean_frame_iou"])
            self.assertEqual(ignored["provenance_verification"]["predicted"], "manifest hash and completion verified")
            (root / "blender/masks/cup/000000.png").unlink()
            with self.assertRaises(FileNotFoundError):
                run(argparse.Namespace(manifest=manifest_path, predicted=root / "sam/masks", reference=root / "blender/masks",
                                       objects=["cup"], output=root / "invalid", overlay_every=0))
            self.assertFalse((root / "invalid").exists())

    def test_rejects_reordered_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"schema_version": 1, "frames": [
                {"index": 0, "time_seconds": 1}, {"index": 1, "time_seconds": 0}]}))
            with self.assertRaises(ValueError):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()

