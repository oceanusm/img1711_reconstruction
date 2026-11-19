import unittest

import numpy as np

from scene_reconstruction.surface_planes import extract_planes


class SurfacePlaneTests(unittest.TestCase):
    def test_noisy_parallel_planes_and_outliers(self):
        rng = np.random.default_rng(12)
        xy = rng.uniform(-1, 1, (1800, 2))
        floor = np.column_stack((xy, rng.normal(0, .001, len(xy))))
        counter = np.column_stack((xy * .8, .7 + rng.normal(0, .001, len(xy))))
        noise = rng.uniform(-1, 1, (300, 3))
        points = np.vstack((floor, counter, noise))
        # Arbitrary source orientation: detector cannot assume z is up.
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        points = np.einsum('ij,kj->ik', points, q)
        result = extract_planes(points, np.arange(len(points)) % 20,
                                voxel_size=.02, distance_threshold=.008, max_planes=3)
        self.assertGreaterEqual(len(result['planes']), 2)
        first = result['planes'][:2]
        self.assertGreater(abs(np.dot(first[0]['normal'], first[1]['normal'])), .999)
        self.assertAlmostEqual(abs(first[0]['offset'] - first[1]['offset']), .7, delta=.003)
        self.assertTrue(all(p['rms'] < .002 for p in first))
        self.assertTrue(all(p['frame_count'] == 20 for p in first))
        np.testing.assert_array_equal(result['points'], points[result['source_indices']])

    def test_deterministic_and_voxel_balanced(self):
        rng = np.random.default_rng(4)
        plane = np.column_stack((rng.uniform(-1, 1, (1500, 2)), np.zeros(1500)))
        # Many duplicates must not become proportionally stronger evidence.
        points = np.vstack((plane, np.repeat(plane[:30], 100, axis=0)))
        a = extract_planes(points, voxel_size=.03, distance_threshold=.004, max_planes=2)
        b = extract_planes(points, voxel_size=.03, distance_threshold=.004, max_planes=2)
        np.testing.assert_array_equal(a['source_indices'], b['source_indices'])
        np.testing.assert_array_equal(a['planes'][0]['indices'], b['planes'][0]['indices'])
        self.assertLessEqual(len(a['points']), len(plane))
        self.assertEqual(len(a['planes']), 1)

    def test_collinear_points_produce_no_plane(self):
        points = np.column_stack((np.linspace(-1, 1, 100), np.zeros((100, 2))))
        self.assertEqual(extract_planes(points)['planes'], [])

    def test_invalid_inputs(self):
        for points in (np.zeros((20, 3)), np.full((20, 3), np.nan), np.zeros((20, 2))):
            with self.assertRaises(ValueError):
                extract_planes(points)
        rng = np.random.default_rng(2)
        with self.assertRaises(ValueError):
            extract_planes(rng.normal(size=(20, 3)), distance_threshold=-1)
        with self.assertRaises(ValueError):
            extract_planes(rng.normal(size=(20, 3)), frame_indices=np.zeros(19))


if __name__ == '__main__':
    unittest.main()
