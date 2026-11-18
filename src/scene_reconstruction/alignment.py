"""One global VGGT-to-Blender similarity, with explicit fit/evaluation sets.

This module needs only NumPy. Both camera inputs must already use OpenCV local
axes; changing camera axes is not a world alignment. No ICP or per-frame fit is
performed, because those can erase the scene discrepancies being measured.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

from .common import sha256, write_json


RANK_TOLERANCE = 1e-3
TIMESTAMP_TOLERANCE = 1e-6


def _points(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite Nx3 coordinates")
    return value


def _spread(points, name):
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if len(singular) < 3 or singular[0] <= np.finfo(float).eps * max(1, np.abs(points).max()) * 100:
        raise ValueError(f"{name} has no usable spatial extent; use distributed landmarks or --transform")
    ratios = singular / singular[0]
    if ratios[1] < RANK_TOLERANCE:
        raise ValueError(
            f"{name} is collinear or nearly collinear (second/first singular value={ratios[1]:.3g}); "
            "camera-center alignment cannot determine a stable rotation. Use distributed landmarks or --transform."
        )
    return {"singular_values": singular.tolist(), "singular_value_ratios": ratios.tolist(),
            "rms_radius": float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))),
            "approximately_planar": bool(ratios[2] < RANK_TOLERANCE),
            "weak_second_axis": bool(ratios[1] < 0.02)}


def _umeyama(source, target):
    """Least-squares proper Sim(3), for row-stored point coordinates."""
    _spread(source, "Source correspondences")
    _spread(target, "Target correspondences")
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    x, y = source - source_mean, target - target_mean
    covariance = y.T @ x / len(x)
    u, values, vt = np.linalg.svd(covariance)
    correction = np.ones(3)
    correction[-1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = (u * correction) @ vt
    scale = float(np.sum(values * correction) / np.mean(np.sum(x * x, axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Correspondences do not admit a positive-scale proper similarity")
    translation = target_mean - scale * rotation @ source_mean
    return validate_transform({"scale": scale, "rotation": rotation, "translation": translation})


def validate_transform(value):
    """Validate and serialize a source-world to Blender-world similarity."""
    if not isinstance(value, dict):
        raise ValueError("Transform must contain scale, rotation (3x3), and translation (3)")
    try:
        scale = float(value["scale"])
        rotation = np.asarray(value["rotation"], dtype=np.float64)
        translation = np.asarray(value["translation"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Transform must contain scale, rotation (3x3), and translation (3)") from error
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Similarity scale must be finite and positive")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("Similarity rotation must be a finite 3x3 matrix")
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError("Similarity rotation must be orthonormal with determinant +1; reflections are forbidden")
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("Similarity translation must be a finite 3-vector")
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return {"scale": scale, "rotation": rotation.tolist(), "translation": translation.tolist(),
            "matrix": matrix.tolist(), "inverse_matrix": np.linalg.inv(matrix).tolist(),
            "equation": "point_blender = scale * rotation @ point_vggt + translation"}


def transform_points(points, transform):
    points = _points(points, "Points")
    return (float(transform["scale"]) * points @ np.asarray(transform["rotation"]).T
            + np.asarray(transform["translation"]))


def transform_cameras(camera_to_world, transform):
    """Apply world rotation to camera axes, and Sim(3) to centers only."""
    poses = np.asarray(camera_to_world, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("Camera poses must be Nx4x4")
    result = poses.copy()
    result[:, :3, :3] = np.asarray(transform["rotation"])[None] @ poses[:, :3, :3]
    result[:, :3, 3] = transform_points(poses[:, :3, 3], transform)
    return result


def rotation_error_degrees(source, target):
    relative = np.asarray(target).swapaxes(-1, -2) @ np.asarray(source)
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1) / 2
    return np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))


def _samples(count, limit=512):
    if math.comb(count, 3) <= limit:
        return list(itertools.combinations(range(count), 3))
    rng = np.random.default_rng(0)
    samples = set()
    while len(samples) < limit:
        samples.add(tuple(sorted(rng.choice(count, 3, replace=False).tolist())))
    return sorted(samples)


def fit_similarity(source, target, robust=True, robust_threshold=None):
    """Fit a positive-scale proper similarity; optional deterministic RANSAC.

    Returns a JSON-compatible transform and diagnostics. A robust fit requires
    a majority consensus and at least three non-collinear inlier pairs. Its
    default threshold is 10% of target median radius, in target-world units.
    """
    source, target = _points(source, "Source"), _points(target, "Target")
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("At least three paired, non-collinear 3D points are required")
    source_spread = _spread(source, "Source correspondences")
    target_spread = _spread(target, "Target correspondences")
    threshold = None
    inliers = np.ones(len(source), dtype=bool)
    if robust and len(source) > 3:
        threshold = (float(robust_threshold) if robust_threshold is not None else
                     0.1 * float(np.median(np.linalg.norm(target - np.median(target, axis=0), axis=1))))
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("Robust threshold must be finite and positive in Blender-world units")
        best = None
        for sample in _samples(len(source)):
            try:
                candidate = _umeyama(source[list(sample)], target[list(sample)])
            except ValueError:
                continue
            errors = np.linalg.norm(transform_points(source, candidate) - target, axis=1)
            mask = errors <= threshold
            score = (int(mask.sum()), -float(np.sum(np.minimum(errors, threshold) ** 2)))
            if best is None or score > best[0]:
                best = (score, mask)
        minimum = max(3, len(source) // 2 + 1)
        if best is None or best[1].sum() < minimum:
            raise ValueError(
                "No non-collinear majority agrees with one global similarity at robust threshold "
                f"{threshold:.6g}. Inspect timing/correspondences, use landmarks, set --robust-threshold, "
                "or use --no-robust to inspect a least-squares fit."
            )
        inliers = best[1]
        # Refine the consensus without allowing endless mask oscillation.
        for _ in range(8):
            candidate = _umeyama(source[inliers], target[inliers])
            updated = np.linalg.norm(transform_points(source, candidate) - target, axis=1) <= threshold
            if np.array_equal(updated, inliers) or updated.sum() < minimum:
                break
            try:
                _spread(source[updated], "Source inliers")
                _spread(target[updated], "Target inliers")
            except ValueError:
                break
            inliers = updated
    elif robust_threshold is not None and (not np.isfinite(robust_threshold) or robust_threshold <= 0):
        raise ValueError("Robust threshold must be finite and positive")
    transform = _umeyama(source[inliers], target[inliers])
    residuals = np.linalg.norm(transform_points(source, transform) - target, axis=1)
    diagnostics = {
        "algorithm": "deterministic_ransac_then_umeyama" if robust and len(source) > 3 else "umeyama",
        "robust_requested": bool(robust), "robust_threshold_blender_units": threshold,
        "correspondence_count": len(source), "inlier_count": int(inliers.sum()),
        "inlier_mask": inliers.tolist(), "residuals_blender_units": residuals.tolist(),
        "source_spread": source_spread, "target_spread": target_spread,
        "source_inlier_spread": _spread(source[inliers], "Source inliers"),
        "target_inlier_spread": _spread(target[inliers], "Target inliers"),
    }
    return transform, diagnostics


def _safe_file(directory, filename):
    path = (directory / filename).resolve()
    try:
        path.relative_to(directory)
    except ValueError as error:
        raise ValueError(f"Artifact path escapes its output directory: {filename}") from error
    return path


def _load_cameras(directory, kind):
    directory = Path(directory).expanduser().resolve()
    index_path = directory / "index.json"
    index = json.loads(index_path.read_text())
    if index.get("schema_version") != 1 or index.get("status") != "complete":
        raise ValueError(f"{kind} input must be a complete schema_version=1 export")
    manifest_hash = index.get("source_manifest_sha256" if kind == "VGGT" else "manifest_sha256")
    if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
        raise ValueError(f"{kind} input lacks a valid source frame manifest hash")
    camera_file = index.get("files", {}).get("cameras", "cameras.npz")
    if not isinstance(camera_file, str):
        raise ValueError(f"{kind} camera file must be a relative path")
    camera_path = _safe_file(directory, camera_file)
    with np.load(camera_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    for key in ("frame_indices", "time_seconds", "camera_to_world", "world_to_camera"):
        if key not in arrays:
            raise ValueError(f"{kind} cameras lack {key}")
    ids = arrays["frame_indices"]
    times = arrays["time_seconds"]
    poses, inverse = arrays["camera_to_world"], arrays["world_to_camera"]
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or not len(ids)
            or (ids < 0).any() or len(set(ids.tolist())) != len(ids)):
        raise ValueError(f"{kind} frame indices must be unique nonnegative integers")
    if times.shape != ids.shape or not np.isfinite(times).all():
        raise ValueError(f"{kind} timestamps must be finite and match frame indices")
    ordered = np.argsort(ids)
    if np.any(np.diff(times[ordered]) <= 0):
        raise ValueError(f"{kind} timestamps must increase with frame index")
    if poses.shape != (len(ids), 4, 4) or inverse.shape != poses.shape:
        raise ValueError(f"{kind} camera matrices must be Nx4x4")
    if not np.isfinite(poses).all() or not np.isfinite(inverse).all():
        raise ValueError(f"{kind} camera matrices must be finite")
    rotations = poses[:, :3, :3]
    if (not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5, rtol=0)
            or not np.allclose(rotations.swapaxes(1, 2) @ rotations, np.eye(3), atol=1e-4, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4, rtol=0)
            or not np.allclose(poses @ inverse, np.eye(4), atol=1e-4, rtol=0)):
        raise ValueError(f"{kind} cameras must contain proper rigid transforms and consistent inverses")
    records = index.get("frames", [])
    by_id = {record["index"]: record for record in records}
    if len(by_id) != len(records) or set(by_id) != set(ids.tolist()):
        raise ValueError(f"{kind} index frame IDs must match cameras.npz exactly")
    for position, (frame_id, timestamp) in enumerate(zip(ids, times)):
        record = by_id[int(frame_id)]
        if not np.isfinite(record["time_seconds"]) or abs(float(record["time_seconds"]) - timestamp) > TIMESTAMP_TOLERANCE:
            raise ValueError(f"{kind} index timestamp differs from camera array at frame {frame_id}")
        if "camera_array_index" in record and record["camera_array_index"] != position:
            raise ValueError(f"{kind} camera_array_index is inconsistent at frame {frame_id}")
    return {"directory": directory, "index": index, "index_path": index_path,
            "camera_path": camera_path, "manifest_sha256": manifest_hash, "arrays": arrays}


def _canonical_intrinsics(bundle, expected_size):
    arrays, index = bundle["arrays"], bundle["index"]
    if "intrinsics_canonical" in arrays:
        intrinsics = arrays["intrinsics_canonical"]
    elif "intrinsics_source" in arrays:
        intrinsics = arrays["intrinsics_source"]
    elif [index.get("width"), index.get("height")] == list(expected_size) and "intrinsics" in arrays:
        intrinsics = arrays["intrinsics"]
    else:
        raise ValueError("Camera input lacks source-grid intrinsics; re-export with intrinsics_canonical")
    if intrinsics.shape != (len(arrays["frame_indices"]), 3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("Canonical intrinsics must be finite Nx3x3 matrices")
    if (np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0)
            or not np.allclose(intrinsics[:, 2], [0, 0, 1], atol=1e-6, rtol=0)):
        raise ValueError("Canonical intrinsics must be valid pinhole calibration matrices")
    return intrinsics


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    return ({"count": 0, "mean": None, "median": None, "rmse": None, "p95": None, "max": None}
            if not values.size else {"count": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "rmse": float(np.sqrt(np.mean(values ** 2))),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())})


def _parse_fit_frames(value, available):
    if value is None or value == "all":
        return set(available)
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except (AttributeError, ValueError) as error:
        raise ValueError("--fit-frames must be comma-separated source frame IDs") from error
    if len(set(values)) != len(values) or not values:
        raise ValueError("--fit-frames must contain unique frame IDs")
    missing = sorted(set(values) - set(available))
    if missing:
        raise ValueError(f"Requested fit frames are not present in both exports: {missing}")
    return set(values)


def _provenance(bundle):
    return {"directory": str(bundle["directory"]), "index_sha256": sha256(bundle["index_path"]),
            "cameras_sha256": sha256(bundle["camera_path"]),
            "manifest_sha256": bundle["manifest_sha256"],
            "source_blend_sha256": bundle["index"].get("source_blend_sha256")}


def _validate_landmark_provenance(document, manifest_hash, inputs):
    """Check identifiers when present; handwritten coordinate pairs need none."""
    if not isinstance(document, dict):
        raise ValueError("Landmarks JSON must be an object containing pairs")
    if "schema_version" in document and document["schema_version"] != 1:
        raise ValueError("Landmarks JSON must use schema_version=1 when specified")
    for key in ("source_manifest_sha256", "manifest_sha256"):
        if key in document and document[key] != manifest_hash:
            raise ValueError(f"Landmark provenance {key} differs from the current source manifest")
    supplied = document.get("inputs", {})
    if not isinstance(supplied, dict):
        raise ValueError("Landmark provenance inputs must be an object of artifact hashes")
    keys = ("vggt_index_sha256", "blender_index_sha256", "vggt_cameras_sha256",
            "blender_cameras_sha256", "blender_geometry_sha256", "vggt_point_cloud_sha256")
    for key in keys:
        if key in supplied and (key not in inputs or supplied[key] != inputs[key]):
            raise ValueError(f"Landmark provenance {key} differs from the current reconstruction/Blender baseline")
    return {key: supplied[key] for key in keys if key in supplied}


def run(args):
    vggt = _load_cameras(args.vggt, "VGGT")
    blender = _load_cameras(args.blender_export, "Blender")
    if vggt["manifest_sha256"] != blender["manifest_sha256"]:
        raise ValueError("VGGT and Blender exports use different source frame manifests")
    source, target = vggt["arrays"], blender["arrays"]
    source_lookup = {int(value): i for i, value in enumerate(source["frame_indices"])}
    target_lookup = {int(value): i for i, value in enumerate(target["frame_indices"])}
    matched = sorted(source_lookup.keys() & target_lookup.keys())
    if not matched:
        raise ValueError("VGGT and Blender exports have no matching source frame IDs")
    si = np.array([source_lookup[value] for value in matched])
    ti = np.array([target_lookup[value] for value in matched])
    if not np.allclose(source["time_seconds"][si], target["time_seconds"][ti], atol=TIMESTAMP_TOLERANCE, rtol=0):
        raise ValueError("Matched frame IDs have different timestamps; no nearest-frame matching is permitted")
    size = vggt["index"].get("preprocessing", {}).get("canonical_size_wh")
    if not isinstance(size, list) or len(size) != 2 or any(int(x) <= 0 for x in size):
        raise ValueError("VGGT index lacks canonical source-image dimensions")
    target_size = [blender["index"].get("canonical_width", blender["index"].get("source_width")),
                   blender["index"].get("canonical_height", blender["index"].get("source_height"))]
    if all(x is not None for x in target_size) and target_size != size:
        raise ValueError("VGGT and Blender canonical source-image dimensions differ")
    source_k, target_k = _canonical_intrinsics(vggt, size), _canonical_intrinsics(blender, size)
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Alignment output must be empty: {output}")
    landmarks_path, transform_path = getattr(args, "landmarks", None), getattr(args, "transform", None)
    fit_frames = getattr(args, "fit_frames", None)
    if landmarks_path and transform_path:
        raise ValueError("Choose --landmarks or --transform, not both")
    if (landmarks_path or transform_path) and fit_frames is not None:
        raise ValueError("--fit-frames applies only to camera-center fitting")
    source_provenance, target_provenance = _provenance(vggt), _provenance(blender)
    inputs = {"vggt": source_provenance, "blender": target_provenance,
              "vggt_index_sha256": source_provenance["index_sha256"],
              "blender_index_sha256": target_provenance["index_sha256"],
              "vggt_cameras_sha256": source_provenance["cameras_sha256"],
              "blender_cameras_sha256": target_provenance["cameras_sha256"]}
    for bundle, file_key, filename, hash_key in (
            (vggt, "point_cloud", "point_cloud.ply", "vggt_point_cloud_sha256"),
            (blender, "geometry", "geometry.npz", "blender_geometry_sha256")):
        path = _safe_file(bundle["directory"], bundle["index"].get("files", {}).get(file_key, filename))
        if path.is_file():
            inputs[hash_key] = sha256(path)
    roles, inlier_by_frame = {}, {}
    landmark_results = None
    fit_input = None
    if transform_path:
        transform_path = Path(transform_path).expanduser().resolve()
        loaded = json.loads(transform_path.read_text())
        transform = validate_transform(loaded.get("transform", loaded))
        fit = {"method": "supplied_transform", "diagnostics": None}
        fit_input = {"path": str(transform_path), "sha256": sha256(transform_path)}
    elif landmarks_path:
        landmarks_path = Path(landmarks_path).expanduser().resolve()
        loaded = json.loads(landmarks_path.read_text())
        checked_hashes = _validate_landmark_provenance(loaded, vggt["manifest_sha256"], inputs)
        pairs = loaded.get("pairs", [])
        if not isinstance(pairs, list) or len(pairs) < 3:
            raise ValueError("Landmarks JSON must contain at least three entries in pairs")
        if any(not isinstance(pair.get("use_for_fit", pair.get("fit", True)), bool) for pair in pairs):
            raise ValueError("Landmark fit flags must be true or false")
        xyz_source = _points([pair.get("source", pair.get("vggt")) for pair in pairs], "VGGT landmarks")
        xyz_target = _points([pair.get("target", pair.get("blender")) for pair in pairs], "Blender landmarks")
        training = np.array([pair.get("use_for_fit", pair.get("fit", True)) for pair in pairs], dtype=bool)
        transform, diagnostics = fit_similarity(xyz_source[training], xyz_target[training],
                                               robust=getattr(args, "robust", True),
                                               robust_threshold=getattr(args, "robust_threshold", None))
        fit = {"method": "landmarks", "diagnostics": diagnostics}
        fit_input = {"path": str(landmarks_path), "sha256": sha256(landmarks_path),
                     "validated_artifact_hashes": checked_hashes}
        errors = np.linalg.norm(transform_points(xyz_source, transform) - xyz_target, axis=1)
        inlier_flags = iter(diagnostics["inlier_mask"])
        landmark_results = [{"name": pair.get("name", str(i)), "role": "fit" if train else "holdout",
                             "fit_inlier": next(inlier_flags) if train else None,
                             "error_blender_units": float(error)}
                            for i, (pair, train, error) in enumerate(zip(pairs, training, errors))]
        fit["landmark_metrics"] = {"all": _stats(errors), "fit": _stats(errors[training]),
                                   "holdout": _stats(errors[~training])}
    else:
        training_ids = _parse_fit_frames(fit_frames, matched)
        train = np.array([value in training_ids for value in matched])
        transform, diagnostics = fit_similarity(source["camera_to_world"][si[train], :3, 3],
                                               target["camera_to_world"][ti[train], :3, 3],
                                               robust=getattr(args, "robust", True),
                                               robust_threshold=getattr(args, "robust_threshold", None))
        fit = {"method": "camera_centers", "frame_indices": [value for value in matched if value in training_ids],
               "diagnostics": diagnostics}
        roles = {value: "fit" if value in training_ids else "holdout" for value in matched}
        inlier_by_frame = dict(zip(fit["frame_indices"], diagnostics["inlier_mask"]))
    aligned = transform_cameras(source["camera_to_world"], transform)
    position_error = np.linalg.norm(aligned[si, :3, 3] - target["camera_to_world"][ti, :3, 3], axis=1)
    angle_error = rotation_error_degrees(aligned[si, :3, :3], target["camera_to_world"][ti, :3, :3])
    rows = []
    for j, (frame_id, source_i, target_i) in enumerate(zip(matched, si, ti)):
        delta = source_k[source_i] - target_k[target_i]
        row = {"frame_index": frame_id, "time_seconds": float(source["time_seconds"][source_i]),
               "role": roles.get(frame_id, "evaluation"), "fit_inlier": inlier_by_frame.get(frame_id, ""),
               "translation_error_blender_units": float(position_error[j]),
               "orientation_error_degrees": float(angle_error[j]),
               "fx_delta_pixels": float(delta[0, 0]), "fy_delta_pixels": float(delta[1, 1]),
               "cx_delta_pixels": float(delta[0, 2]), "cy_delta_pixels": float(delta[1, 2]),
               "fx_relative_error": float(delta[0, 0] / target_k[target_i, 0, 0]),
               "fy_relative_error": float(delta[1, 1] / target_k[target_i, 1, 1]),
               "intrinsics_frobenius_error_pixels": float(np.linalg.norm(delta))}
        for prefix, point in (("vggt_aligned", aligned[source_i, :3, 3]),
                              ("blender", target["camera_to_world"][target_i, :3, 3])):
            row.update({f"{prefix}_{axis}": float(value) for axis, value in zip("xyz", point)})
        rows.append(row)
    metrics = {}
    for role in ("all", "fit", "fit_inliers", "holdout", "evaluation"):
        keep = np.array([role == "all" or row["role"] == role or
                         (role == "fit_inliers" and row["role"] == "fit" and row["fit_inlier"])
                         for row in rows], dtype=bool)
        metrics[role] = {"translation_blender_units": _stats(position_error[keep]),
                         "orientation_degrees": _stats(angle_error[keep])}
    warnings = [
        "One similarity removes global origin, rotation and scale differences; its scale is not independent metric evidence.",
        "Camera-center residuals on fit frames measure agreement conditional on that fit, not independent reconstruction accuracy.",
        "No dense surface or object-center distances are inferred; the point cloud contains visible surfaces and possible outliers.",
    ]
    if fit["method"] == "camera_centers" and not metrics["holdout"]["translation_blender_units"]["count"]:
        warnings.append("No camera frames were held out. Use --fit-frames or independent landmarks for a stronger validation.")
    if fit.get("diagnostics") and fit["diagnostics"]["source_inlier_spread"]["weak_second_axis"]:
        warnings.append("The fitted trajectory/landmarks have little second-axis extent; orientation and scale may be sensitive to noise.")
    result = {
        "schema_version": 1, "stage": "vggt_blender_alignment", "status": "complete",
        "manifest_sha256": vggt["manifest_sha256"], "source_manifest_sha256": vggt["manifest_sha256"],
        "inputs": inputs,
        "fit_input": fit_input, "transform": transform, "fit": fit,
        "coordinates": {"camera": "OpenCV: +x right, +y down, +z forward", "world": "Blender export world",
                        "distance_units": "blender_units", "intrinsics_grid": "canonical upright source pixels",
                        "canonical_size_wh": size},
        "correspondence": {"matched_frame_indices": matched,
                           "vggt_only_frame_indices": sorted(source_lookup.keys() - target_lookup.keys()),
                           "blender_only_frame_indices": sorted(target_lookup.keys() - source_lookup.keys()),
                           "timestamp_tolerance_seconds": TIMESTAMP_TOLERANCE},
        "camera_metrics": metrics, "landmarks": landmark_results, "warnings": warnings,
        "files": {"cameras": "aligned_cameras.npz", "camera_errors": "camera_errors.csv"},
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "aligned_cameras.npz", frame_indices=source["frame_indices"],
                        time_seconds=source["time_seconds"], camera_to_world=aligned,
                        world_to_camera=np.linalg.inv(aligned), intrinsics_canonical=source_k,
                        matched_frame_indices=np.asarray(matched, dtype=np.int64))
    with (output / "camera_errors.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "alignment.json", result)
    print(f"Aligned {len(matched)} matched cameras with one Sim(3); scale={transform['scale']:.6g}; "
          f"translation RMSE={metrics['all']['translation_blender_units']['rmse']:.6g} Blender units; "
          f"orientation median={metrics['all']['orientation_degrees']['median']:.4g} degrees.")
    print(f"Saved {output / 'alignment.json'}")
    return output / "alignment.json"


def load_vggt_ply(path, max_points=100_000):
    """Read only our own binary PLY schema; deterministically subsample for UI."""
    path = Path(path)
    if not isinstance(max_points, int) or isinstance(max_points, bool) or max_points < 1:
        raise ValueError("max_points must be a positive integer")
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                      ("confidence", "<f4"), ("frame_index", "<i4")])
    expected = ["property float x", "property float y", "property float z",
                "property uchar red", "property uchar green", "property uchar blue",
                "property float confidence", "property int frame_index"]
    with path.open("rb") as handle:
        header = []
        for _ in range(100):
            line = handle.readline(512)
            if not line or len(line) >= 512:
                raise ValueError("Invalid VGGT PLY header")
            text = line.decode("ascii").strip()
            if text == "end_header":
                break
            if not text.startswith("comment "):
                header.append(text)
        else:
            raise ValueError("Invalid VGGT PLY header")
        if (len(header) != 11 or header[:2] != ["ply", "format binary_little_endian 1.0"]
                or not header[2].startswith("element vertex ") or header[3:] != expected):
            raise ValueError("Expected the exact binary little-endian PLY schema produced by scene-recon vggt")
        count = int(header[2].split()[-1])
        offset = handle.tell()
    if count < 0 or path.stat().st_size != offset + count * dtype.itemsize:
        raise ValueError("PLY vertex count does not match file length")
    if count:
        records = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=(count,))
        if count > max_points:
            records = records[np.linspace(0, count - 1, max_points, dtype=np.int64)]
        points = np.column_stack([records[key] for key in ("x", "y", "z")]).copy()
        colors = np.column_stack([records[key] for key in ("red", "green", "blue")]).copy()
        confidence = np.asarray(records["confidence"]).copy()
        frame_ids = np.asarray(records["frame_index"]).copy()
        if not np.isfinite(points).all() or not np.isfinite(confidence).all():
            raise ValueError("PLY contains nonfinite point coordinates/confidence")
    else:
        points, colors = np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        confidence, frame_ids = np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)
    return {"points": points, "colors": colors, "confidence": confidence, "frame_indices": frame_ids}
