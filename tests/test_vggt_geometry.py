"""Numerical VGGT adapter checks; no model downloads or torch required."""

from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from scene_reconstruction.vggt_omega import (
    _write_ply,
    _select_cuda_device,
    preprocessing_geometry,
    select_frames,
    unproject_points,
)


class CudaDeviceSelectionTests(unittest.TestCase):
    def test_default_and_explicit_cuda_devices_select_integer_indices(self):
        def device(name, index=None):
            parts = name.split(":")
            return SimpleNamespace(type=parts[0], index=int(parts[1]) if len(parts) == 2 else index)

        for requested, expected in [("cuda", 3), ("cuda:0", 0), ("cuda:2", 2)]:
            cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 3, set_device=Mock())
            resolved = _select_cuda_device(SimpleNamespace(device=device, cuda=cuda), requested)
            self.assertEqual(resolved.type, "cuda")
            self.assertEqual(resolved.index, expected)
            cuda.set_device.assert_called_once_with(expected)


class FrameSelectionTests(unittest.TestCase):
    def test_cap_spans_full_video_instead_of_truncating_start(self):
        frames = [{"index": i} for i in range(698)]
        selected = select_frames(frames, stride=1, max_frames=72)
        ids = [frame["index"] for frame in selected]
        self.assertEqual(len(ids), 72)
        self.assertEqual(ids[0], 0)
        self.assertEqual(ids[-1], 697)
        self.assertEqual(len(set(ids)), len(ids))
        self.assertTrue(all(a < b for a, b in zip(ids, ids[1:])))
        self.assertLessEqual(max(np.diff(ids)) - min(np.diff(ids)), 1)

    def test_stride_always_keeps_final_frame(self):
        frames = [{"index": i} for i in range(22)]
        self.assertEqual([x["index"] for x in select_frames(frames, 10, 0)], [0, 10, 20, 21])

    def test_invalid_or_single_frame_selection_fails(self):
        for count, stride, cap in [(0, 1, 0), (1, 1, 0), (2, 0, 0), (2, 1, 1), (2, 1, -1)]:
            with self.subTest(count=count, stride=stride, cap=cap), self.assertRaises(ValueError):
                select_frames([{"index": i} for i in range(count)], stride, cap)


class PixelGeometryTests(unittest.TestCase):
    def test_portrait_source_has_no_crop_and_exact_half_pixel_mapping(self):
        geometry = preprocessing_geometry(1080, 1920, 288, 512)
        self.assertEqual(geometry["crop_xyxy"], [0, 0, 1080, 1920])
        transform = np.asarray(geometry["canonical_to_model_pixel_affine"])
        inverse = np.asarray(geometry["model_to_canonical_pixel_affine"])
        np.testing.assert_allclose(inverse @ transform, np.eye(3))
        np.testing.assert_allclose(transform @ [-0.5, -0.5, 1], [-0.5, -0.5, 1])
        np.testing.assert_allclose(transform @ [1079.5, 1919.5, 1], [287.5, 511.5, 1])
        intrinsics = np.array([[250.0, 0, 144.0], [0, 250.0, 256.0], [0, 0, 1]])
        ray = np.array([0.2, -0.3, 1.0])
        canonical_intrinsics = inverse @ intrinsics
        np.testing.assert_allclose(transform @ (canonical_intrinsics @ ray), intrinsics @ ray)

    def test_extreme_aspect_crops_preserve_coordinates(self):
        cases = [
            ((100, 400, 256, 512), [0, 100, 100, 300]),
            ((400, 100, 512, 256), [100, 0, 300, 100]),
        ]
        for sizes, crop in cases:
            with self.subTest(sizes=sizes):
                geometry = preprocessing_geometry(*sizes)
                self.assertEqual(geometry["crop_xyxy"], crop)
                transform = np.asarray(geometry["canonical_to_model_pixel_affine"])
                np.testing.assert_allclose(
                    transform @ [crop[0] - 0.5, crop[1] - 0.5, 1], [-0.5, -0.5, 1]
                )

    def test_unprojection_uses_camera_z_and_camera_to_world(self):
        depth = np.full((3, 3), 2.0)
        intrinsics = np.array([[2.0, 0, 1], [0, 2.0, 1], [0, 0, 1]])
        camera_to_world = np.array([
            [0.0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]
        ])
        points = unproject_points(
            depth, intrinsics, camera_to_world, ys=np.array([1, 2]), xs=np.array([1, 2])
        )
        np.testing.assert_allclose(points, [[10, 20, 32], [9, 21, 32]])


class PlyExportTests(unittest.TestCase):
    def test_binary_export_preserves_points_colors_confidence_and_frame_ids(self):
        points = np.array([[1, 2, 3], [-4, 5.5, 6]], dtype=np.float32)
        colors = np.array([[255, 0, 128], [10, 20, 30]], dtype=np.uint8)
        confidence = np.array([1.25, 8.5], dtype=np.float32)
        frames = np.array([0, 697], dtype=np.int32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.ply"
            count = _write_ply(path, [points], [colors], [confidence], [frames])
            header, body = path.read_bytes().split(b"end_header\n", 1)
        self.assertEqual(count, 2)
        self.assertIn(b"format binary_little_endian 1.0", header)
        self.assertIn(b"element vertex 2", header)
        dtype = np.dtype([
            ("position", "<f4", 3), ("rgb", "u1", 3),
            ("confidence", "<f4"), ("frame", "<i4"),
        ])
        records = np.frombuffer(body, dtype=dtype)
        np.testing.assert_array_equal(records["position"], points)
        np.testing.assert_array_equal(records["rgb"], colors)
        np.testing.assert_array_equal(records["confidence"], confidence)
        np.testing.assert_array_equal(records["frame"], frames)


if __name__ == "__main__":
    unittest.main()
