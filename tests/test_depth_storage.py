"""Ensure faster NPZ storage preserves numeric data and atomic replacement."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from scene_reconstruction.depth_anything3 import _write_npz


class DepthStorageTests(unittest.TestCase):
    def test_every_compression_level_has_lossless_numpy_compatible_roundtrip(self):
        arrays = {
            "depth": np.array([[0.0, -0.0, np.nan], [1.75, np.inf, -np.inf]], dtype=np.float32),
            "valid_depth": np.array([[True, False, False], [True, False, False]]),
            "frame_indices": np.array([0, 697], dtype=np.int64),
            "intrinsics": np.asfortranarray(np.eye(3, dtype=np.float32)),
            "scalar": np.array(1.25, dtype=np.float64),
        }
        with tempfile.TemporaryDirectory(prefix="da3-storage-") as directory:
            for level in range(10):
                with self.subTest(level=level):
                    path = Path(directory) / f"level-{level}.npz"
                    _write_npz(path, arrays, level)
                    with np.load(path, allow_pickle=False) as data:
                        self.assertEqual(set(data.files), set(arrays))
                        for key, original in arrays.items():
                            self.assertEqual(data[key].shape, original.shape)
                            self.assertEqual(data[key].dtype, original.dtype)
                            self.assertEqual(data[key].tobytes(), original.tobytes())
                    with zipfile.ZipFile(path) as archive:
                        expected = zipfile.ZIP_STORED if level == 0 else zipfile.ZIP_DEFLATED
                        self.assertTrue(all(item.compress_type == expected for item in archive.infolist()))
                    self.assertFalse(path.with_suffix(".npz.tmp").exists())

    def test_failed_write_preserves_previous_file_and_removes_temporary(self):
        with tempfile.TemporaryDirectory(prefix="da3-storage-") as directory:
            path = Path(directory) / "frame.npz"
            np.savez_compressed(path, depth=np.ones((2, 2), dtype=np.float32))
            original = path.read_bytes()
            with patch("numpy.lib.format.write_array", side_effect=OSError("disk write failed")):
                with self.assertRaisesRegex(OSError, "disk write failed"):
                    _write_npz(path, {"depth": np.zeros((2, 2), dtype=np.float32)})
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(path.with_suffix(".npz.tmp").exists())

    def test_object_arrays_and_invalid_levels_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="da3-storage-") as directory:
            path = Path(directory) / "frame.npz"
            with self.assertRaisesRegex(ValueError, "Object arrays"):
                _write_npz(path, {"depth": np.array([object()], dtype=object)})
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".npz.tmp").exists())
            for level in (-1, 10, 1.5, "1", None):
                with self.subTest(level=level), self.assertRaisesRegex(ValueError, "compression_level"):
                    _write_npz(path, {"depth": np.ones(2)}, level)


if __name__ == "__main__":
    unittest.main()
