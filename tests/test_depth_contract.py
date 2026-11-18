import unittest

import numpy as np

from scene_reconstruction.depth_anything3 import (
    choose_owners, plan_chunks, rescale_intrinsics, resize_depth, summarize_overlap,
)


class DepthContracts(unittest.TestCase):
    def test_chunk_coverage_and_single_owner(self):
        for count in [1, 16, 17, 31, 698]:
            windows = plan_chunks(count, 16, 4)
            owners = choose_owners(count, windows)
            self.assertEqual(len(owners), count)
            for i, owner in enumerate(owners):
                self.assertTrue(windows[owner][0] <= i < windows[owner][1])
            self.assertEqual(windows[-1][1], count)
        with self.assertRaises(ValueError):
            plan_chunks(698, 16, 16)

    def test_intrinsics_restore_pixel_rays(self):
        intrinsic = np.array([[100, 0, 40], [0, 100, 25], [0, 0, 1]], dtype=np.float32)
        restored = rescale_intrinsics(intrinsic, (50, 80), (200, 320))
        native_ray = np.linalg.inv(intrinsic) @ [30, 20, 1]
        source_ray = np.linalg.inv(restored) @ [120, 80, 1]
        np.testing.assert_allclose(native_ray, source_ray)

    def test_invalid_depth_not_spread_by_resize(self):
        depth = np.full((2, 2), 3.0, dtype=np.float32)
        depth[0, 0] = np.nan
        output, valid = resize_depth(depth, (4, 4))
        self.assertFalse(valid[0, 0])
        self.assertTrue(np.isnan(output[0, 0]))
        np.testing.assert_allclose(output[valid], 3)

    def test_overlap_ratio_is_diagnostic_and_does_not_change_arrays(self):
        previous = {"frame_indices": np.array([0, 1]), "depth": np.ones((2, 4, 4))}
        current = {"frame_indices": np.array([1, 2]), "depth": np.full((2, 4, 4), 2.0)}
        diagnostic = summarize_overlap(previous, current)
        self.assertEqual(len(diagnostic), 1)
        self.assertEqual(diagnostic[0]["frame_index"], 1)
        self.assertEqual(diagnostic[0]["median_current_over_previous_depth"], 2)
        np.testing.assert_array_equal(current["depth"], 2)


if __name__ == "__main__":
    unittest.main()

