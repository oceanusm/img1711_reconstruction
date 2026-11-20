"""Whole-track catalog aliases must not manufacture favorable per-frame masks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scene_reconstruction.common import write_json
from scene_reconstruction.dense_catalog import _groups, build_catalog


class DenseCatalogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.frames = [{"index": i * 7, "time_seconds": i / 4, "file": f"rgb/{i}.png"}
                       for i in range(4)]
        (self.root / "rgb").mkdir()
        for frame in self.frames:
            Image.new("RGB", (12, 8), (20, 30, 40)).save(self.root / frame["file"])
        write_json(self.root / "manifest.json", {"schema_version": 1, "width": 12, "height": 8,
                                               "frames": self.frames})
        self.raw = {"schema_version": 1, "status": "complete", "width": 12, "height": 8, "tracks": []}
        self.mask = np.zeros((8, 12), dtype=np.uint8)
        self.mask[2:6, 2:6] = 255
        self.empty = np.zeros_like(self.mask)

    def track(self, identity, masks, anchor=0):
        directory = self.root / "tracks" / identity
        directory.mkdir(parents=True)
        rows = []
        for frame, mask in zip(self.frames, masks):
            path = directory / f"{frame['index']:06d}.png"
            Image.fromarray(mask).save(path)
            rows.append({"index": frame["index"], "time_seconds": frame["time_seconds"],
                         "mask": str(path.relative_to(self.root)), "area_pixels": int(np.count_nonzero(mask))})
        index_file = f"tracks/{identity}/index.json"
        write_json(self.root / index_file, {"status": "complete", "id": identity, "frames": rows})
        self.raw["tracks"].append({"id": identity, "anchor_frame": anchor,
                                   "proposal_id": f"proposal_{identity}", "index_file": index_file})
        write_json(self.root / "raw_tracks.json", self.raw)

    def test_duplicate_whole_tracks_preserve_parts_and_raw_files(self):
        self.track("a", [self.mask] * 4)
        self.track("b", [self.mask] * 4, anchor=14)
        part = self.mask.copy()
        part[:, 4:] = 0
        self.track("part", [part] * 4)
        before = (self.root / "tracks/b/index.json").read_bytes()
        catalog = build_catalog(self.root)
        self.assertEqual(len(catalog["tracks"]), 2)
        first = catalog["tracks"][0]
        self.assertEqual(first["representative"], "a")
        self.assertEqual([a["id"] for a in first["aliases"]], ["a", "b"])
        self.assertEqual(first["index_file"], "tracks/a/index.json")
        self.assertEqual(before, (self.root / "tracks/b/index.json").read_bytes())
        with Image.open(self.root / "catalog" / first["aliases"][0]["overlays"][0]) as overlay:
            self.assertEqual(int(np.count_nonzero(np.asarray(overlay)[..., 3])), 16)
        self.assertEqual(catalog["matching"]["min_fraction_above_threshold"], .8)

    def test_single_frame_similarity_is_insufficient(self):
        self.track("a", [self.mask, self.empty, self.empty, self.empty])
        self.track("b", [self.mask, self.empty, self.empty, self.empty])
        catalog = build_catalog(self.root)
        self.assertEqual(len(catalog["tracks"]), 2)
        self.assertEqual(catalog["ambiguous_pairs"][0]["common_visible_frames"], 1)

    def test_presence_disagreement_prevents_merging(self):
        self.track("a", [self.mask] * 4)
        self.track("b", [self.mask, self.mask, self.mask, self.empty])
        catalog = build_catalog(self.root)
        self.assertEqual(len(catalog["tracks"]), 2)
        self.assertEqual(catalog["ambiguous_pairs"][0]["presence_overlap"], .75)

    def test_chain_does_not_merge_without_complete_link(self):
        pairs = {("a", "b"): {"duplicate": True}, ("a", "c"): {"duplicate": False},
                 ("b", "c"): {"duplicate": True}}
        groups, conflicts = _groups(["a", "b", "c"], pairs)
        self.assertEqual(groups, [["a", "b"], ["c"]])
        self.assertEqual(conflicts[0]["track"], "c")

    def test_missing_frames_fail_instead_of_becoming_empty_masks(self):
        self.track("a", [self.mask] * 3)
        with self.assertRaisesRegex(ValueError, "every requested frame"):
            build_catalog(self.root)

    def test_empty_tracks_stay_separate_and_viewer_is_offline(self):
        self.track("a", [self.empty] * 4)
        self.track("b", [self.empty] * 4)
        catalog = build_catalog(self.root)
        self.assertEqual(len(catalog["tracks"]), 2)
        self.assertEqual(catalog["duplicate_pairs"], [])
        self.assertIn("do not establish object absence", catalog["interpretation"])
        html = (self.root / "catalog/index.html").read_text()
        self.assertIn("#loading{height:1.5em", html)
        self.assertNotIn("fetch(", html)
        self.assertIn("const D=", html)
        self.assertIn("absence unverified", html)
        self.assertEqual(json.loads((self.root / "catalog/catalog.json").read_text()), catalog)

    def test_no_discovery_is_a_valid_reviewable_result(self):
        write_json(self.root / "raw_tracks.json", self.raw)
        catalog = build_catalog(self.root)
        self.assertEqual(catalog["tracks"], [])
        self.assertTrue((self.root / "catalog/index.html").is_file())

    def test_seed_thumbnail_and_budget_provenance(self):
        self.track("a", [self.mask] * 4)
        path = self.root / "tracks/a/index.json"
        index = json.loads(path.read_text())
        index.update(seed_mask="tracks/a/000000.png", seed_quality=.97, seed_stability=.99)
        write_json(path, index)
        write_json(self.root / "discovery/000000/index.json", {"status": "complete", "anchor_frame": 0,
                   "proposal_hypotheses": 100, "proposals": [{"proposal_id": "proposal_a"}],
                   "tracking_proposal_ids": ["proposal_a"], "budget_deferred_ids": ["proposal_b"]})
        catalog = build_catalog(self.root)
        alias = catalog["tracks"][0]["aliases"][0]
        self.assertEqual(alias["seed_quality"], .97)
        self.assertTrue((self.root / "catalog" / alias["seed_overlay"]).is_file())
        self.assertEqual(alias["proposal_file"], "discovery/000000/index.json")
        self.assertEqual(catalog["discovery"][0]["budget_deferred"], 1)


if __name__ == "__main__":
    unittest.main()
