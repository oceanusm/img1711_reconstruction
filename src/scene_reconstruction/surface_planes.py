"""Camera-independent, bounded-memory extraction of planes from a point cloud.

Plane equations use unit normals and ``normal @ point = offset``.  The normal
sign is canonicalized for reproducibility, not interpreted as a room direction.
"""
from __future__ import annotations

import numpy as np


def _project(points, normal):
    # Avoid BLAS dispatch for three-coordinate dot products. In particular,
    # some macOS Accelerate builds raise spurious floating flags on matmul.
    return np.einsum("ij,j->i", points, normal)


def _fit_plane(points):
    center = points.mean(axis=0)
    _, values, axes = np.linalg.svd(points - center, full_matrices=False)
    if len(values) < 3 or values[1] <= max(values[0], 1.) * 1e-10:
        raise ValueError("Plane points have insufficient two-dimensional extent")
    normal = axes[-1]
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    return normal, float(normal @ center), center, values


def _plane_candidates(points, rng, count=2048):
    triples = rng.integers(0, len(points), size=(count, 3))
    a, b, c = (points[triples[:, i]] for i in range(3))
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    usable = lengths > 1e-10
    normals = normals[usable] / lengths[usable, None]
    offsets = np.sum(normals * a[usable], axis=1)
    return normals, offsets


def extract_planes(points, frame_indices=None, voxel_size=None,
                   distance_threshold=None, max_planes=18, seed=0):
    """Return spatially balanced points and robust planar patches.

    Only point coordinates and optional provenance frame IDs are read. No
    camera poses, target geometry or target-derived directions enter detection.
    ``indices`` in each returned plane refer to the returned balanced points;
    ``source_indices`` map those points back to this function's input rows.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Points must be finite Nx3 coordinates")
    if len(points) < 12:
        raise ValueError("At least twelve points are required for plane extraction")
    if not isinstance(max_planes, (int, np.integer)) or isinstance(max_planes, bool) or max_planes < 1:
        raise ValueError("max_planes must be a positive integer")
    frames = None if frame_indices is None else np.asarray(frame_indices)
    if frames is not None and (frames.shape != (len(points),) or not np.issubdtype(frames.dtype, np.integer)):
        raise ValueError("frame_indices must contain one integer per input point")
    low, high = np.percentile(points, [1, 99], axis=0)
    extent = float(np.linalg.norm(high - low))
    if not np.isfinite(extent) or extent <= 1e-12:
        raise ValueError("Point cloud has no usable extent")
    voxel = extent / 220 if voxel_size is None else float(voxel_size)
    threshold = extent * .002 if distance_threshold is None else float(distance_threshold)
    if not np.isfinite(voxel) or not np.isfinite(threshold) or min(voxel, threshold) <= 0:
        raise ValueError("voxel_size and distance_threshold must be finite and positive")
    rng = np.random.default_rng(seed)
    # Randomize before choosing one representative per occupied voxel so the
    # earliest video frame does not systematically win every repeated cell.
    order = rng.permutation(len(points))
    grid_float = np.floor((points[order] - low) / voxel)
    if np.max(np.abs(grid_float)) > np.iinfo(np.int64).max // 2:
        raise ValueError("voxel_size is too small for the coordinate range")
    grid = grid_float.astype(np.int64)
    _, representative = np.unique(grid, axis=0, return_index=True)
    selected = np.sort(order[representative])
    # Bound fitting cost even if an unusually large/noisy cloud fills many cells.
    if len(selected) > 60000:
        selected = np.sort(rng.choice(selected, size=60000, replace=False))
    balanced = points[selected]
    balanced_frames = None if frames is None else frames[selected]
    remaining = np.arange(len(balanced))
    minimum = max(30, int(np.ceil(len(balanced) * .003)))
    planes = []
    rejected = 0
    for _ in range(max_planes):
        if len(remaining) < minimum:
            break
        active = balanced[remaining]
        scoring = active if len(active) <= 4096 else active[rng.choice(len(active), 4096, replace=False)]
        normals, offsets = _plane_candidates(active, rng)
        if not len(normals):
            break
        scores = np.empty(len(normals), dtype=np.int32)
        for start in range(0, len(normals), 64):
            residuals = np.abs(np.einsum("ij,kj->ik", scoring, normals[start:start + 64]) - offsets[start:start + 64])
            scores[start:start + 64] = np.sum(residuals <= threshold, axis=0)
        best = None
        for candidate in np.argsort(scores, kind="stable")[-16:]:
            n, d = normals[candidate], offsets[candidate]
            mask = np.abs(_project(active, n) - d) <= threshold
            if mask.sum() < minimum:
                continue
            try:
                for _ in range(5):
                    n, d, center, singular = _fit_plane(active[mask])
                    updated = np.abs(_project(active, n) - d) <= threshold
                    if updated.sum() < minimum or np.array_equal(mask, updated):
                        break
                    mask = updated
                n, d, center, singular = _fit_plane(active[mask])
            except ValueError:
                continue
            # Broad patches are needed for stable normals; reject line-like
            # support, including repeated depth-edge streaks.
            spread = singular[:2] / np.sqrt(mask.sum())
            if spread[1] < voxel * 2 or spread[1] / spread[0] < .04:
                continue
            errors = _project(active[mask], n) - d
            score = (int(mask.sum()), -float(np.mean(errors ** 2)))
            if best is None or score > best[0]:
                best = (score, mask, n, d, center, singular)
        if best is None:
            rejected += 1
            break
        _, mask, n, d, center, singular = best
        ids = remaining[mask]
        patch = balanced[ids]
        _, _, axes = np.linalg.svd(patch - center, full_matrices=False)
        tangent = np.einsum("ij,kj->ik", patch - center, axes[:2])
        widths = np.diff(np.percentile(tangent, [5, 95], axis=0), axis=0)[0]
        planes.append({
            "normal": n.tolist(), "offset": d, "indices": ids,
            "centroid": center.tolist(), "rms": float(np.sqrt(np.mean((_project(patch, n) - d) ** 2))),
            "area": float(np.prod(widths)), "spread": (singular / np.sqrt(len(ids))).tolist(),
            "count": int(len(ids)),
            "frame_count": None if balanced_frames is None else int(len(np.unique(balanced_frames[ids]))),
            "tangent_axes": axes[:2].tolist(), "widths_90_percent": widths.tolist(),
        })
        remaining = remaining[~mask]
    return {"points": balanced, "frame_indices": balanced_frames, "source_indices": selected,
            "planes": planes, "diagnostics": {
                "algorithm": "voxel_balanced_sequential_ransac_orthogonal_refit",
                "input_count": len(points), "balanced_count": len(balanced),
                "unassigned_count": len(remaining), "minimum_plane_support": minimum,
                "robust_extent": extent, "voxel_size": voxel,
                "distance_threshold": threshold, "seed": int(seed),
                "max_planes": int(max_planes), "rejected_searches": rejected,
                "normal_sign": "largest-magnitude component positive; no semantic orientation",
                "uses_camera_poses": False,
            }}
