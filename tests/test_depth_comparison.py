from argparse import Namespace
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scene_reconstruction.depth_compare import (
    _grid, _load_depth, coverage, depth_metrics, fit_global_scale, run,
)


class DepthComparisonTests(unittest.TestCase):
    def test_numeric_metrics_and_empty_intersection(self):
        ref = np.array([1, 2, 4, 0, np.nan, 10.])
        pred = np.array([2, 2, 2, 1, 1, np.inf])
        result = depth_metrics(ref, pred)
        self.assertEqual(result["pixels"], 3)
        self.assertAlmostEqual(result["abs_rel"], .5)
        self.assertAlmostEqual(result["mae"], 1)
        self.assertAlmostEqual(result["rmse"], np.sqrt(5 / 3))
        self.assertAlmostEqual(result["log_rmse"], np.sqrt(2 / 3) * np.log(2))
        for key in ("delta1", "delta2", "delta3"):
            self.assertAlmostEqual(result[key], 1 / 3)
        empty = depth_metrics(ref, pred, np.zeros_like(ref, bool))
        self.assertEqual(empty["pixels"], 0)
        self.assertIsNone(empty["rmse"])
        with self.assertRaises(ValueError):
            depth_metrics(np.ones(3), np.ones(4))

    def test_global_scale_is_pooled_not_mean_of_frame_scales(self):
        pairs = [(np.array([2.]), np.array([1.]), np.array([True])),
                 (np.array([12.]), np.array([3.]), np.array([True]))]
        scale, count = fit_global_scale(pairs)
        self.assertAlmostEqual(scale, 3.8)  # (1*2 + 3*12) / (1 + 9)
        self.assertEqual(count, 2)
        scale, count = fit_global_scale([(np.array([4, 900.]), np.array([2, 1.]), np.array([True, False]))])
        self.assertAlmostEqual(scale, 2)
        self.assertEqual(count, 1)
        with self.assertRaisesRegex(ValueError, "no common"):
            fit_global_scale([(np.zeros(3), np.zeros(3), np.ones(3, bool))])

    def test_coverage_accounts_for_missing_geometry(self):
        result = coverage(np.array([[1, 1], [0, 0]], bool), np.array([[1, 0], [1, 0]], bool))
        self.assertEqual(result["common_valid_pixels"], 1)
        self.assertEqual(result["da3_only_pixels"], 1)
        self.assertEqual(result["blender_only_pixels"], 1)
        self.assertEqual(result["neither_valid_pixels"], 1)
        self.assertAlmostEqual(result["common_fraction"], .25)
        self.assertAlmostEqual(result["blender_coverage_of_da3"], .5)

    def test_loader_masks_sky_invalid_confidence_and_reference_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.npz"
            np.savez(path, depth=np.array([[1, 2, 3], [4, 0, 100.]], np.float32),
                     valid_depth=np.array([[1, 1, 1], [0, 1, 1]], bool),
                     sky=np.array([[0, 0, 1], [0, 0, 0]], bool),
                     confidence=np.array([[np.nan, 5, 9], [8, 7, 6.]]))
            _, valid, _, extra = _load_depth(path, (2, 3), max_depth=50)
            np.testing.assert_array_equal(valid, [[False, True, False], [False, False, False]])
            self.assertEqual(extra["source_valid_before_confidence"], 2)
            self.assertEqual(extra["source_valid_after_confidence"], 1)
            np.savez(path, depth=np.array([[1, 2, 3], [4, 5, 6.]], np.float32),
                     confidence=np.array([[0, 1, 2], [3, 4, 5.]]))
            _, valid, _, extra = _load_depth(path, (2, 3), percentile=50)
            self.assertEqual(valid.sum(), 3)
            self.assertEqual(extra["confidence_threshold"], 2.5)

    def test_full_frame_pixel_center_grid_rejects_crop_and_missing_metadata(self):
        index = {"height": 3, "width": 4, "source_height": 6, "source_width": 8,
                 "source_to_export_pixel_affine": [[.5, 0, -.25], [0, .5, -.25], [0, 0, 1]]}
        shape, affine = _grid(index, (6, 8))
        self.assertEqual(shape, (3, 4))
        np.testing.assert_allclose(affine @ [3.5, 2.5, 1], [1.5, 1, 1])
        bad = dict(index, source_to_export_pixel_affine=np.eye(3).tolist())
        with self.assertRaisesRegex(ValueError, "crop/warp"):
            _grid(bad, (6, 8))
        with self.assertRaisesRegex(ValueError, "explicit source"):
            _grid({"height": 3, "width": 4}, (6, 8))

    def _fixture(self, root):
        frames, da3_frames, blender_frames = [], [], []
        k = np.array([[4, 0, 3.5], [0, 4, 2.5], [0, 0, 1.]])
        affine = np.array([[.5, 0, -.25], [0, .5, -.25], [0, 0, 1.]])
        for i, time in enumerate([0., .05, .13]):
            Image.new("RGB", (8, 6), (30 + i * 50, 60, 90)).save(root / f"{i}.jpg")
            frames.append({"index": i, "time_seconds": time, "file": f"{i}.jpg", "duration_seconds": .04})
            da3_frames.append({"index": i, "time_seconds": time, "file": f"da3_{i}.npz",
                               "depth_units": "estimated_meters", "chunk_id": i // 2})
            blender_frames.append({"index": i, "time_seconds": time, "file": f"blender_{i}.npz"})
            np.savez(root / f"da3_{i}.npz", depth=np.full((6, 8), 2 if i < 2 else 3, np.float32),
                     valid_depth=np.ones((6, 8), bool), intrinsics=k, confidence=np.ones((6, 8)))
            valid = np.ones((3, 4), bool)
            valid[0, 0] = False
            # Deliberately unrelated ray_distance verifies only camera-z is used.
            np.savez(root / f"blender_{i}.npz", depth=np.ones((3, 4), np.float32),
                     valid_depth=valid, intrinsics=affine @ k, ray_distance=np.full((3, 4), 200.))
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": 1, "width": 8, "height": 6, "frames": frames}))
        common = {"status": "complete", "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
        da3 = {**common, "depth_convention": "camera-space z depth, not Euclidean ray distance",
               "source_hw": [6, 8], "frames": da3_frames}
        blender = {**common, "depth_convention": "camera_z", "depth_units": "blender_units",
                   "unit_scale_meters": 1, "width": 4, "height": 3, "source_width": 8, "source_height": 6,
                   "source_to_export_pixel_affine": affine.tolist(), "frames": blender_frames}
        (root / "da3.json").write_text(json.dumps(da3))
        (root / "blender.json").write_text(json.dumps(blender))
        args = Namespace(manifest=manifest, depth_index=root / "da3.json", blender_index=root / "blender.json",
                         output=root / "report", blender_unit_meters=None, scale_mode="global",
                         fit_frames="0,1", max_depth=None, confidence_percentile=0, preview_count=2)
        return args, da3, blender

    def test_integration_scale_holdout_and_self_contained_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _, _ = self._fixture(root)
            path = run(args)
            report = json.loads(path.read_text())
            self.assertAlmostEqual(report["global_scale"], 2)
            self.assertEqual(report["fit_frame_indices"], [0, 1])
            self.assertEqual(report["holdout_frame_indices"], [2])
            self.assertEqual(report["fit_pixels"], 22)
            self.assertAlmostEqual(report["metrics"]["fitted"]["fit"]["rmse"], 0)
            self.assertAlmostEqual(report["metrics"]["fitted"]["holdout"]["rmse"], 1)
            self.assertAlmostEqual(report["metrics"]["direct"]["holdout"]["rmse"], 2)
            self.assertEqual(report["metrics"]["direct"]["all"]["da3_only_pixels"], 3)
            self.assertEqual(report["projection_mismatch_frame_indices"], [])
            self.assertEqual(len(report["previews"]), 4)
            text = (args.output / "index.html").read_text()
            self.assertIn("data:image/png;base64,", text)
            self.assertIn("not measured ground truth", text)
            self.assertNotIn("https://", text)
            with (args.output / "per_frame.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[-1]["split"], "holdout")
            with self.assertRaises(FileExistsError):
                run(args)

    def test_provenance_convention_units_and_fit_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, da3, blender = self._fixture(root)
            for field, value, pattern in [("manifest_sha256", "wrong", "SHA256"),
                                          ("depth_convention", "ray_distance", "camera-z"),
                                          ("unit_scale_meters", None, "unit scale")]:
                changed = dict(blender, **{field: value})
                args.blender_index.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, pattern):
                    run(args)
            args.blender_index.write_text(json.dumps(blender))
            args.fit_frames = None
            with self.assertRaisesRegex(ValueError, "explicit --fit-frames"):
                run(args)
            args.scale_mode = "metric"
            da3["frames"][0]["depth_units"] = "relative_model_units"
            args.depth_index.write_text(json.dumps(da3))
            with self.assertRaisesRegex(ValueError, "estimated_meters"):
                run(args)

    def test_metric_units_and_unrendered_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _, blender = self._fixture(root)
            args.scale_mode, args.fit_frames, args.blender_unit_meters = "metric", None, 2
            blender["frames"][2].pop("file")
            args.blender_index.write_text(json.dumps(blender))
            report = json.loads(run(args).read_text())
            self.assertEqual(report["frame_indices"], [0, 1])
            self.assertEqual(report["unrendered_blender_frame_indices"], [2])
            self.assertIsNone(report["global_scale"])
            self.assertAlmostEqual(report["metrics"]["direct"]["all"]["rmse"], 0)
            self.assertEqual(report["blender_unit_scale_source"], "command_line_override")

    def test_all_invalid_produces_null_errors_and_coverage_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, _, _ = self._fixture(root)
            args.scale_mode, args.fit_frames = "metric", None
            for idx in range(3):
                np.savez(root / f"da3_{idx}.npz", depth=np.full((6, 8), np.nan), valid_depth=np.zeros((6, 8), bool))
                np.savez(root / f"blender_{idx}.npz", depth=np.zeros((3, 4)), valid_depth=np.zeros((3, 4), bool))
            report = json.loads(run(args).read_text())
            metrics = report["metrics"]["direct"]["all"]
            self.assertEqual(metrics["pixels"], 0)
            self.assertEqual(metrics["neither_valid_pixels"], 36)
            self.assertIsNone(metrics["rmse"])
            self.assertFalse(report["color_scale"]["depth_range_has_samples"])


if __name__ == "__main__":
    unittest.main()
