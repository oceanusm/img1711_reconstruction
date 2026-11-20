"""Conservative, frame-local association for independent SAM proposals.

This module never allocates track IDs or infers identity from containment. A
contained knife and a whole rack remain alternatives for later grouping. The
caller retains every raw proposal, including those rejected here.
"""

import hashlib
import math
from numbers import Integral

import numpy as np


_POPCOUNT = np.array([int(i).bit_count() if hasattr(int, "bit_count") else bin(i).count("1")
                     for i in range(256)], dtype=np.uint8)


def _threshold(name, value):
    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between 0 and 1")


def _key(value):
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise ValueError("proposal and track IDs must be strings or integers")
    return (0, int(value)) if isinstance(value, Integral) else (1, value)


def _id(value):
    _key(value)
    return int(value) if isinstance(value, Integral) else value


def _mask_info(mask, shape):
    mask = np.asarray(mask)
    if mask.ndim != 2 or 0 in mask.shape:
        raise ValueError("masks must be nonempty 2D arrays")
    if shape is not None and mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match {shape}")
    if mask.dtype != np.bool_:
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("masks must contain binary values, not probabilities")
        mask = mask.astype(bool)
    area = int(np.count_nonzero(mask))
    if area:
        ys = np.flatnonzero(np.any(mask, axis=1))
        xs = np.flatnonzero(np.any(mask, axis=0))
        bbox = (int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1)
    else:
        bbox = (0, 0, 0, 0)
    return {"mask": mask, "area": area, "bbox": bbox}


def _candidates(candidates):
    result, seen, shape = [], set(), None
    for candidate in candidates:
        proposal_id = _id(candidate["proposal_id"])
        if proposal_id in seen:
            raise ValueError(f"duplicate proposal_id: {proposal_id}")
        seen.add(proposal_id)
        info = _mask_info(candidate["mask"], shape)
        shape = info["mask"].shape
        score, stability = float(candidate["score"]), float(candidate["stability"])
        if not math.isfinite(score) or not math.isfinite(stability):
            raise ValueError("candidate score and stability must be finite")
        info.update(candidate=candidate, proposal_id=proposal_id, score=score, stability=stability)
        result.append(info)
    return result, shape


def _overlap(a, b):
    """Return IoU, smaller/larger area ratio, and smaller-mask containment."""
    if not a["area"] or not b["area"]:
        return 0.0, 0.0, 0.0
    ay0, ay1, ax0, ax1 = a["bbox"]
    by0, by1, bx0, bx1 = b["bbox"]
    y0, y1, x0, x1 = max(ay0, by0), min(ay1, by1), max(ax0, bx0), min(ax1, bx1)
    intersection = 0
    if y1 > y0 and x1 > x0:
        if "packed" in a and "packed" in b:
            # Global row alignment makes edge-byte padding safe: any pixels
            # outside the shared bounding box are zero in at least one mask.
            bits = a["packed"][y0:y1, x0//8:(x1+7)//8] & b["packed"][y0:y1, x0//8:(x1+7)//8]
            counts = np.bitwise_count(bits) if hasattr(np, "bitwise_count") else _POPCOUNT[bits]
            intersection = int(counts.sum(dtype=np.int64))
        else:
            intersection = int(np.count_nonzero(
                a["mask"][y0:y1, x0:x1] & b["mask"][y0:y1, x0:x1]))
    smaller, larger = sorted((a["area"], b["area"]))
    return intersection / (a["area"] + b["area"] - intersection), smaller / larger, intersection / smaller


def _mask_digest(packed):
    return hashlib.blake2b(memoryview(packed), digest_size=16).digest()


def _could_duplicate_or_contain(a, b, duplicate_iou, duplicate_area_ratio):
    """Skip pixel work only when bbox/area upper bounds prove irrelevance."""
    ay0, ay1, ax0, ax1 = a["bbox"]
    by0, by1, bx0, bx1 = b["bbox"]
    bbox_overlap = max(0, min(ay1, by1) - max(ay0, by0)) * max(0, min(ax1, bx1) - max(ax0, bx0))
    smaller, larger = sorted((a["area"], b["area"]))
    upper = min(smaller, bbox_overlap)
    if smaller / larger >= duplicate_area_ratio:
        return upper / (a["area"] + b["area"] - upper) >= duplicate_iou
    return upper / smaller >= 0.95


def filter_candidates(candidates, min_area=64, min_quality=0.85, min_stability=0.9,
                      duplicate_iou=0.9, duplicate_area_ratio=0.85):
    """Filter weak masks and collapse near-equivalent independent point trials.

    Returns ``accepted`` (original dictionaries, strongest first), ``decisions``
    (input order), and ``relations`` between accepted contained parts/wholes.
    Ranking is score, stability, then proposal ID; input order cannot decide a
    duplicate winner. Relations require at least 95% smaller-mask containment.
    Packed masks use about one extra byte per eight input pixels, only for
    accepted proposals plus the current trial. Hash hits are verified exactly;
    bbox pruning uses conservative bounds and never approximates mask overlap.
    """
    if isinstance(min_area, bool) or not isinstance(min_area, Integral) or min_area < 1:
        raise ValueError("min_area must be a positive integer")
    for name, value in (("min_quality", min_quality), ("min_stability", min_stability),
                        ("duplicate_iou", duplicate_iou), ("duplicate_area_ratio", duplicate_area_ratio)):
        _threshold(name, value)
    if duplicate_iou == 0:
        raise ValueError("duplicate_iou must be greater than zero")
    infos, _ = _candidates(candidates)
    accepted, decisions, relations, exact_masks = [], {}, [], {}
    for item in sorted(infos, key=lambda x: (-x["score"], -x["stability"], _key(x["proposal_id"]))):
        decision = {"proposal_id": item["proposal_id"], "area_pixels": item["area"]}
        reason = ("zero_area" if not item["area"] else
                  "area_below_minimum" if item["area"] < min_area else
                  "quality_below_minimum" if item["score"] < min_quality else
                  "stability_below_minimum" if item["stability"] < min_stability else None)
        if reason:
            decision.update(status="filtered", reason=reason)
        else:
            duplicates, contained = [], []
            item["packed"] = np.packbits(item["mask"], axis=1)
            exact_key = (item["area"], item["bbox"], _mask_digest(item["packed"]))
            exact = [other for other in exact_masks.get(exact_key, [])
                     if np.array_equal(item["packed"], other["packed"])]
            if exact:
                # IoU 1 is unbeatable. An exact hit bypasses every other mask;
                # verify bytes so even a hash collision cannot change identity.
                duplicates = [(1.0, min(exact, key=lambda x: _key(x["proposal_id"])), 1.0)]
            for other in (() if exact else accepted):
                if not _could_duplicate_or_contain(item, other, duplicate_iou, duplicate_area_ratio):
                    continue
                iou, ratio, containment = _overlap(item, other)
                if iou >= duplicate_iou and ratio >= duplicate_area_ratio:
                    duplicates.append((iou, other, ratio))
                elif containment >= 0.95 and ratio < duplicate_area_ratio:
                    child, parent = (item, other) if item["area"] < other["area"] else (other, item)
                    contained.append({"child_proposal_id": child["proposal_id"],
                                      "parent_proposal_id": parent["proposal_id"],
                                      "containment": containment, "area_ratio": ratio})
            if duplicates:
                iou, other, ratio = sorted(duplicates, key=lambda x: (-x[0], _key(x[1]["proposal_id"])))[0]
                decision.update(status="duplicate", reason="near_equivalent_mask",
                                duplicate_of_proposal_id=other["proposal_id"], iou=iou, area_ratio=ratio)
                del item["packed"]
            else:
                decision.update(status="accepted", reason="passes_filters")
                accepted.append(item)
                exact_masks.setdefault(exact_key, []).append(item)
                relations.extend(contained)
        decisions[item["proposal_id"]] = decision
    return {"accepted": [item["candidate"] for item in accepted],
            "decisions": [decisions[item["proposal_id"]] for item in infos], "relations": relations}


def match_candidates(accepted, existing, match_iou=0.65, match_margin=0.1, min_area_ratio=0.7):
    """Associate proposals with masks of persistent tracks at the *same* frame.

    A candidate needs a clear best track and must also win competition for that
    track. Similar competing proposals remain ambiguous rather than being
    assigned arbitrarily. No second-best fallback creates a forced match.
    Caller decides how to track ``new`` and ``ambiguous`` proposals provisionally.
    Empty existing masks are ignored; empty candidates are marked filtered.
    """
    for name, value in (("match_iou", match_iou), ("match_margin", match_margin),
                        ("min_area_ratio", min_area_ratio)):
        _threshold(name, value)
    if match_iou == 0:
        raise ValueError("match_iou must be greater than zero")
    infos, shape = _candidates(accepted)
    tracks, ignored = [], []
    for track_id in sorted(existing, key=_key):
        info = _mask_info(existing[track_id], shape)
        shape = info["mask"].shape
        if info["area"]:
            info["track_id"] = _id(track_id)
            tracks.append(info)
        else:
            ignored.append(_id(track_id))
    decisions, relations, eligible = {}, [], []
    for item in infos:
        proposal_id = item["proposal_id"]
        decision = {"proposal_id": proposal_id}
        if not item["area"]:
            decision.update(status="filtered", reason="zero_area")
            decisions[proposal_id] = decision
            continue
        scores = []
        for track in tracks:
            iou, ratio, containment = _overlap(item, track)
            if containment >= 0.95 and ratio < min_area_ratio:
                relations.append({"proposal_id": proposal_id, "track_id": track["track_id"],
                                  "relation": "candidate_inside_track" if item["area"] < track["area"] else "track_inside_candidate",
                                  "containment": containment, "area_ratio": ratio})
            if ratio >= min_area_ratio and iou > 0:
                scores.append((iou, track["track_id"], ratio))
        scores.sort(key=lambda x: (-x[0], _key(x[1])))
        if not scores or scores[0][0] < match_iou:
            decision.update(status="new", reason="no_near_equivalent_track", iou=scores[0][0] if scores else 0.0)
        else:
            best, track_id, ratio = scores[0]
            runner_up = scores[1][0] if len(scores) > 1 else 0.0
            decision.update(track_id=track_id, iou=best, runner_up_iou=runner_up, area_ratio=ratio)
            # Equal scores never have a unique winner, even when margin is zero.
            if len(scores) > 1 and (best == runner_up or best - runner_up < match_margin):
                decision.update(status="ambiguous", reason="multiple_similar_tracks",
                                alternative_track_ids=[score[1] for score in scores if best - score[0] <= match_margin])
            else:
                eligible.append((best, proposal_id, track_id))
        decisions[proposal_id] = decision
    eligible.sort(key=lambda x: (-x[0], _key(x[1]), _key(x[2])))
    by_track = {}
    for edge in eligible:
        by_track.setdefault(edge[2], []).append(edge)
    for edges in by_track.values():
        best = edges[0][0]
        tied = len(edges) > 1 and (best == edges[1][0] or best - edges[1][0] < match_margin)
        for position, (iou, proposal_id, _) in enumerate(edges):
            decision = decisions[proposal_id]
            if tied:
                decision.update(status="ambiguous", reason="competing_proposals_for_track")
            elif position:
                decision.update(status="ambiguous", reason="track_assigned_to_better_proposal")
            else:
                decision.update(status="matched", reason="unique_near_equivalent_track")
    return {"decisions": [decisions[item["proposal_id"]] for item in infos],
            "relations": relations, "ignored_track_ids": ignored}
