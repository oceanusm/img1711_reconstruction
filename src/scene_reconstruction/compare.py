"""Compare binary visible masks only when frames and pixels match exactly."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .common import load_manifest, sha256, write_json


def read_mask(path: Path, size):
    if not path.is_file():
        raise FileNotFoundError(f"Missing mask (missing files are not empty masks): {path}")
    with Image.open(path) as image:
        if image.size != tuple(size):
            raise ValueError(f"Mask {path} is {image.size}, expected {size}; resampling would hide alignment errors")
        values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError(f"Expected a single-channel binary PNG, got shape {values.shape}: {path}")
    if not np.isin(values, [0, 1, 255]).all():
        raise ValueError(f"Expected binary values 0/1 or 0/255: {path}")
    return values > 0


def mask_metrics(predicted, reference):
    if predicted.shape != reference.shape:
        raise ValueError("Masks must have identical shapes")
    predicted, reference = predicted.astype(bool), reference.astype(bool)
    intersection = int(np.count_nonzero(predicted & reference))
    union = int(np.count_nonzero(predicted | reference))
    p_area, r_area = int(predicted.sum()), int(reference.sum())
    # Both absent has no overlap evidence; exclude it from mean IoU/Dice.
    distance = None
    if p_area and r_area:
        py, px = np.nonzero(predicted)
        ry, rx = np.nonzero(reference)
        distance = float(np.hypot(px.mean() - rx.mean(), py.mean() - ry.mean()))
    return {"intersection": intersection, "union": union,
            "predicted_area": p_area, "reference_area": r_area,
            "iou": intersection / union if union else None,
            "dice": 2 * intersection / (p_area + r_area) if p_area + r_area else None,
            "centroid_error_px": distance, "both_empty": union == 0}


def _overlay(rgb_path, predicted, reference, destination):
    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = np.zeros_like(rgb)
    color[predicted & reference] = [40, 220, 80]
    color[predicted & ~reference] = [255, 70, 70]
    color[reference & ~predicted] = [60, 130, 255]
    active = predicted | reference
    rgb[active] = 0.5 * rgb[active] + 0.5 * color[active]
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save(destination)


def run(args):
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path, allow_sparse=True)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    if args.overlay_every < 0:
        raise ValueError("--overlay-every must be nonnegative")
    names = args.objects
    if len(set(names)) != len(names) or any(Path(name).name != name or name in {".", ".."} for name in names):
        raise ValueError("Object names must be unique directory names")
    size = (manifest["width"], manifest["height"])
    manifest_hash = sha256(manifest_path)
    provenance = {}
    for label, directory in [("predicted", args.predicted), ("reference", args.reference)]:
        metadata_path = Path(directory).parent / "metadata.json"
        provenance[label] = "no metadata.json; frame identity must be verified externally"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("manifest_sha256") != manifest_hash:
                raise ValueError(f"{metadata_path} does not match this source frame manifest")
            if metadata.get("status") != "complete":
                raise ValueError(f"{metadata_path} is not marked complete")
            provenance[label] = "manifest hash and completion verified"
    ignore_directory = getattr(args, "ignore_masks", None)
    # Validate coverage before writing anything; never silently omit difficult frames.
    for name in names:
        for frame in manifest["frames"]:
            for directory in [args.predicted, args.reference]:
                path = Path(directory) / name / f"{frame['index']:06d}.png"
                if not path.is_file():
                    raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name in names:
        rows = []
        for frame in manifest["frames"]:
            filename = f"{frame['index']:06d}.png"
            predicted = read_mask(Path(args.predicted) / name / filename, size)
            reference = read_mask(Path(args.reference) / name / filename, size)
            ignore = read_mask(Path(ignore_directory) / filename, size) if ignore_directory else np.zeros_like(predicted)
            predicted, reference = predicted & ~ignore, reference & ~ignore
            rows.append({"object": name, "frame_index": frame["index"], "time_seconds": frame["time_seconds"],
                         "ignored_pixels": int(ignore.sum()),
                         **mask_metrics(predicted, reference)})
            if args.overlay_every and frame["index"] % args.overlay_every == 0:
                _overlay(manifest_path.parent / frame["file"], predicted, reference,
                         output / "overlays" / name / filename)
        with (output / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        valid = [r for r in rows if r["iou"] is not None]
        total_union = sum(r["union"] for r in rows)
        summaries[name] = {
            "frames": len(rows), "frames_with_either_object_visible": len(valid),
            "mean_frame_iou": float(np.mean([r["iou"] for r in valid])) if valid else None,
            "mean_frame_dice": float(np.mean([r["dice"] for r in valid])) if valid else None,
            "pixel_weighted_iou": sum(r["intersection"] for r in rows) / total_union if total_union else None,
            "both_empty_frames": len(rows) - len(valid),
        }
    write_json(output / "summary.json", {
        "schema_version": 1, "frame_manifest_sha256": manifest_hash,
        "provenance_verification": provenance,
        "ignore_masks": str(Path(ignore_directory).resolve()) if ignore_directory else None,
        "predicted": str(Path(args.predicted).resolve()), "reference": str(Path(args.reference).resolve()),
        "dimensions": list(size), "objects": summaries,
        "empty_mask_policy": "Both empty excluded from mean; one empty scores zero; missing files fail",
        "overlay_legend": {"green": "intersection", "red": "SAM only", "blue": "Blender only"},
        "interpretation": "Measures visible silhouette agreement against predicted SAM masks, not verified ground truth or complete 3D accuracy",
    })
    print(f"Compared {len(manifest['frames'])} frames for {len(names)} objects: {output / 'summary.json'}")

