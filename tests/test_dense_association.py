import unittest
from unittest import mock

import numpy as np

from scene_reconstruction.dense_association import filter_candidates, match_candidates, _mask_info, _overlap


def rectangle(y0, y1, x0, x1):
    result = np.zeros((24, 24), dtype=bool)
    result[y0:y1, x0:x1] = True
    return result


def proposal(proposal_id, mask, score=0.95, stability=0.98):
    return {"proposal_id": proposal_id, "mask": mask, "score": score,
            "stability": stability, "seed_xy": [5, 5]}


class DenseAssociationTests(unittest.TestCase):
    def test_identical_trials_bypass_pairwise_pixel_overlap(self):
        mask = rectangle(2, 12, 3, 13)
        with mock.patch("scene_reconstruction.dense_association._overlap", side_effect=AssertionError("unneeded overlap")):
            result = filter_candidates([proposal(i, mask.copy()) for i in range(50)])
        self.assertEqual(len(result["accepted"]), 1)
        self.assertTrue(all(d["iou"] == 1 for d in result["decisions"][1:]))

    def test_digest_collisions_do_not_merge_distinct_masks(self):
        first = np.indices((24, 24)).sum(axis=0) % 2 == 0
        second = ~first
        with mock.patch("scene_reconstruction.dense_association._mask_digest", return_value=b"collision"):
            result = filter_candidates([proposal(1, first), proposal(2, second)])
        self.assertEqual(len(result["accepted"]), 2)

    def test_bbox_upper_bound_skips_irrelevant_overlapping_boxes(self):
        left, right = rectangle(2, 12, 1, 11), rectangle(2, 12, 10, 20)
        with mock.patch("scene_reconstruction.dense_association._overlap", side_effect=AssertionError("unneeded overlap")):
            result = filter_candidates([proposal(1, left), proposal(2, right)])
        self.assertEqual(len(result["accepted"]), 2)

    def test_packed_overlap_matches_pixels_with_unaligned_bounds_and_width(self):
        rng = np.random.default_rng(51)
        for width in (17, 24, 31):
            for _ in range(12):
                a, b = rng.random((18, width)) > .4, rng.random((18, width)) > .6
                a[:, :3], b[:, -5:] = False, False
                a[:4], b[-2:] = False, False
                a_info, b_info = _mask_info(a, None), _mask_info(b, None)
                plain = _overlap(a_info, b_info)
                a_info["packed"], b_info["packed"] = np.packbits(a, axis=1), np.packbits(b, axis=1)
                self.assertEqual(_overlap(a_info, b_info), plain)

    def test_acceleration_matches_unpruned_decisions_for_nested_noisy_masks(self):
        rng = np.random.default_rng(73)
        candidates = []
        for i in range(60):
            mask = np.zeros((48, 49), dtype=bool)
            y, x = rng.integers(0, 16, size=2)
            h, w = rng.integers(8, 30, size=2)
            mask[y:y+h, x:x+w] = rng.random((h, w)) > .03
            candidates.append(proposal(i, mask, score=.9 + i / 1000))
            if i % 3 == 0:
                candidates.append(proposal(i + 100, mask.copy()))
        accelerated = filter_candidates(candidates, min_area=1)
        # Disable bbox pruning and exact-hit shortcuts; same exact pairwise
        # definition then serves as an oracle for the optimized control flow.
        with mock.patch("scene_reconstruction.dense_association._could_duplicate_or_contain", return_value=True), \
                mock.patch("scene_reconstruction.dense_association._mask_digest", side_effect=(str(i).encode() for i in range(1000))):
            reference = filter_candidates(candidates, min_area=1)
        self.assertEqual(accelerated["decisions"], reference["decisions"])
        self.assertEqual(accelerated["relations"], reference["relations"])

    def test_three_point_trials_of_one_jar_choose_best_deterministically(self):
        mask = rectangle(2, 12, 3, 13)
        candidates = [proposal("b", mask.copy(), 0.94), proposal("a", mask, 0.96),
                      proposal("c", mask.copy(), 0.96)]
        result = filter_candidates(candidates)
        self.assertEqual([p["proposal_id"] for p in result["accepted"]], ["a"])
        self.assertIs(result["accepted"][0], candidates[1])
        self.assertEqual([d["status"] for d in result["decisions"]], ["duplicate", "accepted", "duplicate"])
        self.assertEqual(result["decisions"][0]["duplicate_of_proposal_id"], "a")
        reversed_result = filter_candidates(list(reversed(candidates)))
        self.assertEqual([p["proposal_id"] for p in reversed_result["accepted"]], ["a"])

    def test_adjacent_objects_remain_separate_and_get_correct_tracks(self):
        left, right = rectangle(2, 12, 2, 10), rectangle(2, 12, 10, 18)
        candidates = [proposal(1, left), proposal(2, right)]
        filtered = filter_candidates(candidates)
        self.assertEqual(len(filtered["accepted"]), 2)
        result = match_candidates(filtered["accepted"], {101: left, 102: right})
        self.assertEqual([(d["status"], d["track_id"]) for d in result["decisions"]],
                         [("matched", 101), ("matched", 102)])

    def test_contained_part_is_not_deduplicated_or_assigned_to_whole(self):
        whole, part = rectangle(1, 22, 1, 22), rectangle(3, 15, 5, 11)
        result = filter_candidates([proposal("whole", whole), proposal("part", part)])
        self.assertEqual(len(result["accepted"]), 2)
        self.assertEqual(result["relations"], [{"child_proposal_id": "part", "parent_proposal_id": "whole",
                                               "containment": 1.0, "area_ratio": 72 / 441}])
        matched = match_candidates([proposal("part", part)], {7: whole})
        self.assertEqual(matched["decisions"][0]["status"], "new")
        self.assertEqual(matched["relations"][0]["relation"], "candidate_inside_track")

    def test_top_two_similar_tracks_are_ambiguous(self):
        mask = rectangle(3, 13, 3, 13)
        shifted = rectangle(3, 13, 4, 14)
        candidate = proposal(1, rectangle(3, 13, 3, 14))
        decision = match_candidates([candidate], {10: mask, 20: shifted})["decisions"][0]
        self.assertEqual(decision["status"], "ambiguous")
        self.assertEqual(decision["reason"], "multiple_similar_tracks")
        self.assertEqual(decision["alternative_track_ids"], [10, 20])

    def test_runner_up_just_below_threshold_still_causes_ambiguity(self):
        mask = rectangle(3, 13, 3, 13)
        result = match_candidates([proposal(1, mask)], {10: mask, 20: rectangle(3, 13, 3, 14)},
                                  match_iou=0.95, match_margin=0.1)
        self.assertEqual(result["decisions"][0]["status"], "ambiguous")

    def test_competing_candidates_do_not_both_get_same_track(self):
        mask = rectangle(3, 13, 3, 13)
        result = match_candidates([proposal(1, mask), proposal(2, mask.copy())], {10: mask})
        self.assertTrue(all(d["status"] == "ambiguous" for d in result["decisions"]))
        self.assertTrue(all(d["reason"] == "competing_proposals_for_track" for d in result["decisions"]))

    def test_clear_competition_winner_and_no_second_best_fallback(self):
        mask = rectangle(3, 13, 3, 13)
        candidates = [proposal(1, mask), proposal(2, rectangle(3, 13, 4, 14))]
        result = match_candidates(candidates, {10: mask}, match_margin=0.1)
        self.assertEqual([d["status"] for d in result["decisions"]], ["matched", "ambiguous"])
        self.assertEqual(result["decisions"][1]["reason"], "track_assigned_to_better_proposal")

    def test_area_ratio_prevents_merge_even_at_low_iou_threshold(self):
        whole, part = rectangle(1, 21, 1, 21), rectangle(1, 11, 1, 21)
        result = filter_candidates([proposal(1, whole), proposal(2, part)],
                                   duplicate_iou=0.4, duplicate_area_ratio=0.8)
        self.assertEqual(len(result["accepted"]), 2)
        self.assertEqual(match_candidates([proposal(2, part)], {10: whole}, match_iou=0.4,
                                         min_area_ratio=0.8)["decisions"][0]["status"], "new")

    def test_empty_masks_are_not_matched_or_new_tracks(self):
        empty, mask = rectangle(0, 0, 0, 0), rectangle(3, 13, 3, 13)
        self.assertEqual(filter_candidates([proposal(0, empty)])["decisions"][0]["reason"], "zero_area")
        result = match_candidates([proposal(0, empty), proposal(1, mask)], {5: empty})
        self.assertEqual(result["ignored_track_ids"], [5])
        self.assertEqual([d["status"] for d in result["decisions"]], ["filtered", "new"])
        self.assertEqual(match_candidates([], {})["decisions"], [])

    def test_filter_reasons_are_retained(self):
        mask = rectangle(2, 12, 2, 12)
        result = filter_candidates([proposal(1, rectangle(2, 3, 2, 3)),
                                    proposal(2, mask, score=0.2), proposal(3, mask, stability=0.2)])
        self.assertEqual([d["reason"] for d in result["decisions"]],
                         ["area_below_minimum", "quality_below_minimum", "stability_below_minimum"])

    def test_invalid_thresholds_shapes_ids_and_values_are_rejected(self):
        candidate = proposal(1, rectangle(2, 12, 2, 12))
        for kwargs in ({"min_area": 0}, {"min_area": 1.5}, {"min_quality": float("nan")},
                       {"min_stability": 1.1}, {"duplicate_iou": 0}, {"duplicate_area_ratio": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                filter_candidates([candidate], **kwargs)
        for kwargs in ({"match_iou": 0}, {"match_iou": float("inf")},
                       {"match_margin": -0.1}, {"min_area_ratio": 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                match_candidates([candidate], {}, **kwargs)
        with self.assertRaisesRegex(ValueError, "duplicate proposal_id"):
            filter_candidates([candidate, candidate])
        with self.assertRaisesRegex(ValueError, "shape"):
            match_candidates([candidate], {2: np.ones((3, 3), dtype=bool)})
        with self.assertRaisesRegex(ValueError, "binary"):
            filter_candidates([proposal(1, np.full((3, 3), 0.5))])
        with self.assertRaisesRegex(ValueError, "finite"):
            filter_candidates([proposal(1, candidate["mask"], score=float("nan"))])


if __name__ == "__main__":
    unittest.main()
