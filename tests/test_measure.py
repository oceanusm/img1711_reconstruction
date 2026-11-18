import unittest

import numpy as np

from scene_reconstruction.measure import camera_points


class MeasureTests(unittest.TestCase):
    def test_uses_z_depth_not_normalized_ray_distance(self):
        depth = np.full((3, 3), 2.0)
        k = np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]])
        points = camera_points(depth, k, [(1, 1), (2, 1)])
        np.testing.assert_allclose(points, [[0, 0, 2], [1, 0, 2]])
        self.assertEqual(np.linalg.norm(points[1] - points[0]), 1)

    def test_rejects_invalid_samples(self):
        depth = np.full((3, 3), np.nan)
        with self.assertRaises(ValueError):
            camera_points(depth, np.eye(3), [(1, 1)])
        with self.assertRaises(ValueError):
            camera_points(depth, np.eye(3), [(4, 1)])


if __name__ == "__main__":
    unittest.main()

