"""Analytic alignment checks: gauges, robust fitting and artifact correspondence."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from scene_reconstruction.alignment import (
    fit_similarity, load_vggt_ply, rotation_error_degrees, run,
    transform_cameras, transform_points, validate_transform,
)
from scene_reconstruction.common import write_json
from scene_reconstruction.vggt_omega import _write_ply


def known_transform():
    angle = 0.63
    rotation = [[np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    return validate_transform({"scale": 2.7, "rotation": rotation, "translation": [4, -2, 7]})


def make_bundle(directory, kind, poses, ids, times, intrinsics=None):
    directory.mkdir(parents=True)
    if intrinsics is None:
        intrinsics = np.broadcast_to([[300., 0, 159.5], [0, 310., 239.5], [0, 0, 1]], (len(ids), 3, 3)).copy()
    np.savez_compressed(directory / "cameras.npz", frame_indices=np.asarray(ids, dtype=np.int64),
                        time_seconds=np.asarray(times, dtype=float), camera_to_world=poses,
                        world_to_camera=np.linalg.inv(poses), intrinsics_canonical=intrinsics)
    metadata = {"schema_version": 1, "status": "complete", "files": {"cameras": "cameras.npz"},
                "frames": [{"index": int(frame), "time_seconds": float(t), "camera_array_index": i}
                           for i, (frame, t) in enumerate(zip(ids, times))]}
    metadata["source_manifest_sha256" if kind == "vggt" else "manifest_sha256"] = "a" * 64
    if kind == "vggt":
        metadata["preprocessing"] = {"canonical_size_wh": [320, 480]}
    else:
        metadata.update(canonical_width=320, canonical_height=480, width=160, height=240,
                        source_blend_sha256="b" * 64)
    write_json(directory / "index.json", metadata)


class SimilarityTests(unittest.TestCase):
    def test_known_transform_and_inverse(self):
        source = np.random.default_rng(4).normal(size=(30, 3))
        expected = known_transform()
        target = transform_points(source, expected)
        actual, diagnostics = fit_similarity(source, target, robust=False)
        np.testing.assert_allclose(actual["matrix"], expected["matrix"], atol=1e-12)
        np.testing.assert_allclose(np.asarray(actual["inverse_matrix"]) @ actual["matrix"], np.eye(4), atol=1e-12)
        self.assertEqual(diagnostics["inlier_count"], 30)

    def test_planar_noncollinear_points_suffice(self):
        source = np.array([[0., 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0]])
        expected = known_transform()
        actual, diagnostics = fit_similarity(source, transform_points(source, expected))
        np.testing.assert_allclose(actual["matrix"], expected["matrix"], atol=1e-12)
        self.assertTrue(diagnostics["source_inlier_spread"]["approximately_planar"])

    def test_collinear_near_collinear_and_stationary_paths_fail(self):
        for source in (np.zeros((10, 3)), np.array([[i, 0, 0] for i in range(10)]),
                       np.array([[i, (i % 3) * 1e-5, 0] for i in range(10)])):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "landmarks|extent"):
                fit_similarity(source, source)

    def test_reflection_is_never_returned_as_world_rotation(self):
        source = np.random.default_rng(6).normal(size=(30, 3))
        target = source * [-1, 1, 1]
        transform, diagnostics = fit_similarity(source, target, robust=False)
        self.assertGreater(transform["scale"], 0)
        self.assertAlmostEqual(np.linalg.det(transform["rotation"]), 1)
        self.assertGreater(np.mean(diagnostics["residuals_blender_units"]), 0.1)
        with self.assertRaisesRegex(ValueError, "reflections"):
            validate_transform({"scale": 1, "rotation": np.diag([-1, 1, 1]), "translation": [0, 0, 0]})
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_transform({"scale": -1, "rotation": np.eye(3), "translation": [0, 0, 0]})

    def test_ransac_rejects_outliers_and_is_deterministic(self):
        rng = np.random.default_rng(21)
        source = rng.normal(size=(40, 3))
        target = transform_points(source, known_transform())
        target[-9:] = rng.uniform(40, 70, size=(9, 3))
        actual, diagnostics = fit_similarity(source, target)
        np.testing.assert_allclose(actual["matrix"], known_transform()["matrix"], atol=1e-12)
        self.assertEqual(diagnostics["inlier_count"], 31)
        self.assertFalse(any(diagnostics["inlier_mask"][-9:]))
        self.assertEqual((actual, diagnostics), fit_similarity(source, target))

    def test_no_consensus_fails_instead_of_report_success(self):
        rng = np.random.default_rng(8)
        with self.assertRaisesRegex(ValueError, "majority"):
            fit_similarity(rng.normal(size=(20, 3)), rng.normal(size=(20, 3)), robust_threshold=0.001)

    def test_orientation_is_rotated_but_not_scaled_or_axis_flipped(self):
        poses = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
        poses[:, :3, 3] = [[0, 0, 0], [1, 2, 3], [-1, 0, 2]]
        transform = known_transform()
        actual = transform_cameras(poses, transform)
        expected_rotation = np.asarray(transform["rotation"])
        np.testing.assert_allclose(actual[:, :3, :3], np.broadcast_to(expected_rotation, (3, 3, 3)))
        np.testing.assert_allclose(np.linalg.det(actual[:, :3, :3]), 1)
        np.testing.assert_allclose(actual[:, :3, 3], transform_points(poses[:, :3, 3], transform))
        # A double OpenCV/Blender basis flip is a 180-degree error, not a match.
        flip = np.diag([1, -1, -1])
        np.testing.assert_allclose(rotation_error_degrees(expected_rotation @ flip, expected_rotation), 180)


class AlignmentArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.ids = [0, 10, 20, 30, 40]
        self.times = np.asarray(self.ids) / 30
        self.source = np.broadcast_to(np.eye(4), (5, 4, 4)).copy()
        self.source[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [2, 1, 1]]
        self.target = transform_cameras(self.source, known_transform())
        make_bundle(self.directory / "vggt", "vggt", self.source, self.ids, self.times)
        make_bundle(self.directory / "blender", "blender", self.target, self.ids, self.times)
        self.args = SimpleNamespace(vggt=self.directory / "vggt", blender_export=self.directory / "blender",
                                    output=self.directory / "alignment", fit_frames="0,10,20,30", robust=False,
                                    landmarks=None, transform=None)

    def tearDown(self):
        self.temp.cleanup()

    def mutate_index(self, kind, update):
        path = self.directory / kind / "index.json"
        metadata = json.loads(path.read_text())
        update(metadata)
        write_json(path, metadata)

    def test_holdout_is_evaluated_without_refitting(self):
        self.target[-1, 0, 3] += 0.75
        np.savez_compressed(self.directory / "blender" / "cameras.npz", frame_indices=np.asarray(self.ids),
                            time_seconds=self.times, camera_to_world=self.target,
                            world_to_camera=np.linalg.inv(self.target),
                            intrinsics_canonical=np.broadcast_to(np.eye(3), (5, 3, 3)))
        result_path = run(self.args)
        result = json.loads(result_path.read_text())
        np.testing.assert_allclose(result["transform"]["matrix"], known_transform()["matrix"], atol=1e-12)
        self.assertEqual(result["camera_metrics"]["fit"]["translation_blender_units"]["count"], 4)
        self.assertAlmostEqual(result["camera_metrics"]["holdout"]["translation_blender_units"]["rmse"], 0.75)
        self.assertEqual(len(result["inputs"]["vggt_index_sha256"]), 64)
        with (self.args.output / "camera_errors.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[-1]["role"], "holdout")
        with np.load(self.args.output / "aligned_cameras.npz") as arrays:
            np.testing.assert_allclose(arrays["camera_to_world"], transform_cameras(self.source, known_transform()), atol=1e-12)
            np.testing.assert_allclose(arrays["camera_to_world"] @ arrays["world_to_camera"],
                                       np.broadcast_to(np.eye(4), (5, 4, 4)), atol=1e-12)
        with self.assertRaises(FileExistsError):
            run(self.args)

    def test_manifest_and_timestamp_mismatch_fail(self):
        self.mutate_index("blender", lambda value: value.update(manifest_sha256="b" * 64))
        with self.assertRaisesRegex(ValueError, "different source frame manifests"):
            run(self.args)
        self.mutate_index("blender", lambda value: value.update(manifest_sha256="a" * 64))
        self.mutate_index("blender", lambda value: value["frames"][0].update(time_seconds=0.1))
        with self.assertRaisesRegex(ValueError, "timestamp differs"):
            run(self.args)
        self.assertFalse(self.args.output.exists())

    def test_exact_frame_ids_and_timestamps_not_array_order(self):
        order = [4, 0, 3, 1, 2]
        import shutil
        shutil.rmtree(self.directory / "blender")
        make_bundle(self.directory / "blender", "blender", self.target[order],
                    np.asarray(self.ids)[order], self.times[order])
        result = json.loads(run(self.args).read_text())
        self.assertEqual(result["correspondence"]["matched_frame_indices"], self.ids)
        self.assertLess(result["camera_metrics"]["all"]["translation_blender_units"]["rmse"], 1e-12)

    def test_consistent_but_shifted_timestamps_are_not_nearest_matched(self):
        import shutil
        shutil.rmtree(self.directory / "blender")
        make_bundle(self.directory / "blender", "blender", self.target, self.ids, self.times + 0.002)
        with self.assertRaisesRegex(ValueError, "different timestamps"):
            run(self.args)

    def test_landmark_fit_and_holdout_keep_camera_evaluation_independent(self):
        source = self.source[:, :3, 3]
        target = transform_points(source, known_transform())
        target[-1, 1] += 0.4
        pairs = [{"name": f"corner{i}", "source": a.tolist(), "target": b.tolist(), "use_for_fit": i < 4}
                 for i, (a, b) in enumerate(zip(source, target))]
        self.args.landmarks = self.directory / "landmarks.json"
        write_json(self.args.landmarks, {"pairs": pairs})
        self.args.fit_frames = None
        result = json.loads(run(self.args).read_text())
        self.assertEqual(result["fit"]["method"], "landmarks")
        self.assertAlmostEqual(result["fit"]["landmark_metrics"]["holdout"]["rmse"], 0.4)
        self.assertEqual(result["camera_metrics"]["evaluation"]["translation_blender_units"]["count"], 5)
        self.assertEqual(result["landmarks"][-1]["role"], "holdout")

    def test_manual_transform_bypasses_degenerate_camera_fit(self):
        import shutil
        self.source[:, :3, 3] = [[i, 0, 0] for i in range(5)]
        self.target = transform_cameras(self.source, known_transform())
        for kind, poses in (("vggt", self.source), ("blender", self.target)):
            shutil.rmtree(self.directory / kind)
            make_bundle(self.directory / kind, kind, poses, self.ids, self.times)
        self.args.transform = self.directory / "manual.json"
        self.args.fit_frames = None
        write_json(self.args.transform, known_transform())
        result = json.loads(run(self.args).read_text())
        self.assertEqual(result["fit"]["method"], "supplied_transform")
        self.assertEqual(result["camera_metrics"]["evaluation"]["translation_blender_units"]["count"], 5)

    def test_invalid_requested_fit_frame_fails(self):
        self.args.fit_frames = "0,10,999"
        with self.assertRaisesRegex(ValueError, "not present"):
            run(self.args)


class PlyLoaderTests(unittest.TestCase):
    def test_export_reader_roundtrip_and_sampling(self):
        points = np.arange(60, dtype=np.float32).reshape(20, 3)
        colors = (points % 255).astype(np.uint8)
        confidence = np.linspace(1, 2, 20).astype(np.float32)
        ids = np.arange(20, dtype=np.int32) * 10
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.ply"
            _write_ply(path, [points], [colors], [confidence], [ids])
            values = load_vggt_ply(path, max_points=5)
            selected = np.linspace(0, 19, 5, dtype=np.int64)
            np.testing.assert_array_equal(values["points"], points[selected])
            np.testing.assert_array_equal(values["colors"], colors[selected])
            np.testing.assert_array_equal(values["frame_indices"], ids[selected])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "file length"):
                load_vggt_ply(path)


if __name__ == "__main__":
    unittest.main()
