"""Review reports preserve geometry and cannot turn artifact text into scripts."""
import json
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

from scene_reconstruction.alignment import transform_points
from scene_reconstruction.surface_report import build_payload, write_report


class SurfaceReportTests(unittest.TestCase):
    def setUp(self):
        self.report = {
            "status": "ambiguous",
            "transform": {"scale": 2., "rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                          "translation": [4., 5., 6.]},
            "anchors": [{"name": "floor", "source_plane_index": 2, "point_count": 1,
                         "rms": .1, "median": .05, "p95": .2}],
            "diagnostics": {"condition": np.float64(4.), "not_estimated": np.nan},
            "candidates": [], "warnings": ["Surface match remains ambiguous."],
        }
        self.points = np.arange(300, dtype=float).reshape(-1, 3) / 100
        self.labels = np.full(100, -1, dtype=int)
        self.labels[89] = 0
        self.targets = {"planes": [{"name": "floor", "triangles": [[[0., 0, 0], [1, 0, 0], [0, 1, 0]]]}]}

    def test_transform_and_labels_stay_paired_and_rare_anchor_survives_sampling(self):
        payload = build_payload(self.report, self.points, self.labels, self.targets, max_points=10)
        ids = np.asarray(payload["source_indices"])
        self.assertEqual(len(ids), 10)
        self.assertIn(89, ids)
        np.testing.assert_array_equal(payload["labels"], self.labels[ids])
        np.testing.assert_allclose(payload["points"], transform_points(self.points[ids], self.report["transform"]), atol=1e-6)
        self.assertEqual(payload["source_point_count"], 100)
        self.assertIsNone(payload["report"]["diagnostics"]["not_estimated"])
        self.assertEqual(payload, build_payload(self.report, self.points, self.labels, self.targets, max_points=10))

    def test_target_sampling_does_not_move_or_invent_faces(self):
        faces = np.arange(180., dtype=float).reshape(-1, 3, 3)
        target = {"planes": [{"name": "floor", "triangles": faces}]}
        payload = build_payload(self.report, self.points, self.labels, target, max_triangles=3)
        selected = np.asarray(payload["planes"][0]["triangles"])
        self.assertEqual(selected.shape, (3, 3, 3))
        np.testing.assert_array_equal(selected, faces[[0, 9, 19]])
        self.assertEqual(payload["planes"][0]["total_triangles"], 20)

    def test_profile_and_display_labels_preserve_stable_surface_identities(self):
        self.targets["profile"] = "named_living_dining_structural_planes_v1"
        self.targets["planes"][0].update(name="counter", display_name="Dining tabletop")
        self.report["anchors"][0].update(name="counter", display_name="Dining tabletop")
        payload = build_payload(self.report, self.points, self.labels, self.targets)
        self.assertEqual(payload["report"]["target_profile"], self.targets["profile"])
        self.assertEqual(payload["planes"][0]["display_name"], "Dining tabletop")
        self.assertEqual(payload["planes"][0]["name"], "counter")
        self.assertEqual(payload["report"]["anchors"][0]["display_name"], "Dining tabletop")
        self.assertEqual(payload["report"]["anchors"][0]["name"], "counter")
        # Older artifacts retain their original surface names and need no profile.
        del self.targets["planes"][0]["display_name"]
        del self.targets["profile"]
        legacy = build_payload(self.report, self.points, self.labels, self.targets)
        self.assertEqual(legacy["planes"][0]["display_name"], "counter")
        self.assertIsNone(legacy["report"]["target_profile"])

    def test_script_terminators_and_unicode_are_safe_and_roundtrip_exactly(self):
        hostile = '</script><script>alert("artifact text")</script> & > \u2028\u2029'
        self.report["anchors"][0]["name"] = hostile
        self.report["anchors"][0]["display_name"] = hostile
        self.report["target_profile"] = hostile
        self.report["warnings"] = [hostile]
        with tempfile.TemporaryDirectory() as folder:
            path = write_report(Path(folder) / "report.html", self.report, self.points, self.labels, self.targets)
            html = path.read_text()
        embedded = re.search(r'<script id="surface-data" type="application/json">(.*?)</script>', html, re.S).group(1)
        self.assertNotIn('<', embedded)
        self.assertNotIn('&', embedded)
        self.assertNotIn('\u2028', embedded)
        decoded = json.loads(embedded)
        self.assertEqual(decoded["report"]["warnings"], [hostile])
        self.assertEqual(decoded["report"]["anchors"][0]["name"], hostile)
        self.assertEqual(decoded["report"]["anchors"][0]["display_name"], hostile)
        self.assertEqual(decoded["report"]["target_profile"], hostile)
        self.assertEqual(html.count('<script'), 2)
        self.assertNotIn('src="http', html)
        self.assertIn('href="viewer.html"', html)

    def test_rejects_inconsistent_or_nonfinite_geometry_and_labels(self):
        for labels in (np.full(100, -2), np.zeros(99, dtype=int), np.full(100, 1), np.zeros(100, dtype=float)):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "labels"):
                build_payload(self.report, self.points, labels, self.targets)
        broken = self.points.copy(); broken[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite Nx3"):
            build_payload(self.report, broken, self.labels, self.targets)
        with self.assertRaisesRegex(ValueError, "Mx3x3"):
            build_payload(self.report, self.points, self.labels, {"planes": [{"triangles": [1, 2, 3]}]})
        with self.assertRaisesRegex(ValueError, "max_points"):
            build_payload(self.report, self.points, self.labels, self.targets, max_points=0)

    def test_empty_cloud_and_no_labels_are_supported(self):
        payload = build_payload(self.report, np.empty((0, 3)), None, self.targets)
        self.assertEqual(payload["points"], [])
        self.assertEqual(payload["labels"], [])
        payload = build_payload(self.report, self.points[:2], None, self.targets)
        self.assertEqual(payload["labels"], [-1, -1])


if __name__ == "__main__":
    unittest.main()
