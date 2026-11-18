"""Review SAM3 track candidates and assemble explicitly curated physical objects."""
from __future__ import annotations

import html
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .common import load_manifest, sha256, write_json
from .compare import read_mask


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Expected a safe object/concept name: {value!r}")
    return value


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _inside(directory: Path, relative: str):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("SAM mask paths must be relative to its output directory")
    path = (directory / relative).resolve()
    try:
        path.relative_to(directory.resolve())
    except ValueError as error:
        raise ValueError(f"Mask path escapes SAM output directory: {relative}") from error
    return path


def _track_key(reference):
    return (_name(reference["concept"]),
            _integer(reference["chunk_start_index"], "chunk_start_index"),
            _integer(reference["id"], "instance id"))


def _track_dict(key):
    return {"concept": key[0], "chunk_start_index": key[1], "id": key[2]}


def load_sam_run(manifest_path: Path, sam_output: Path):
    """Validate source linkage and complete per-concept frame records.

    Instance presence is indexed separately from frame coverage: a complete
    segmentation result with no detection is still not verified object absence.
    """
    manifest_path, sam_output = Path(manifest_path).resolve(), Path(sam_output).resolve()
    manifest = load_manifest(manifest_path)
    metadata = json.loads((sam_output / "metadata.json").read_text())
    if metadata.get("status") != "complete":
        raise ValueError("SAM run must have metadata status='complete'")
    if metadata.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("SAM run belongs to a different source frame manifest")
    expected_size = (manifest["width"], manifest["height"])
    if (metadata.get("width"), metadata.get("height")) != expected_size:
        raise ValueError("SAM dimensions disagree with the source manifest")
    if metadata.get("frame_count") != len(manifest["frames"]):
        raise ValueError("SAM frame count disagrees with the source manifest")
    concepts = [_name(item["name"]) for item in metadata["objects"]]
    if not concepts or len(set(concepts)) != len(concepts):
        raise ValueError("SAM concept names must be nonempty and unique")
    frames = {item["index"]: item for item in manifest["frames"]}
    records, tracks, chunks = {}, defaultdict(dict), defaultdict(set)
    with (sam_output / "frames.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            concept = record["object"]
            index = _integer(record["index"], "frame index")
            chunk_start = _integer(record["chunk_start_index"], "chunk_start_index")
            if concept not in concepts or index not in frames or chunk_start > index:
                raise ValueError("SAM frame record references an unknown concept/frame/chunk")
            key = (concept, index)
            if key in records:
                raise ValueError(f"Duplicate SAM frame record: {key}")
            for field in ("pts_seconds", "time_seconds"):
                if abs(float(record[field]) - float(frames[index][field])) > 1e-6:
                    raise ValueError(f"SAM timestamp differs from manifest for frame {index}")
            records[key] = record
            chunks[(concept, chunk_start)].add(index)
            seen_ids = set()
            for instance in record["instances"]:
                obj_id = _integer(instance["id"], "instance id")
                if obj_id in seen_ids:
                    raise ValueError(f"Duplicate instance ID in SAM frame {key}")
                seen_ids.add(obj_id)
                tracks[(concept, chunk_start, obj_id)][index] = instance
    expected = {(concept, index) for concept in concepts for index in frames}
    if records.keys() != expected:
        raise ValueError("SAM frames.jsonl does not cover every concept and source frame exactly once")
    for (concept, start), indices in chunks.items():
        if min(indices) != start or sorted(indices) != list(range(start, max(indices) + 1)):
            raise ValueError(f"SAM chunk is not a contiguous interval: {concept}/{start}")
    return manifest, metadata, dict(tracks), dict(chunks)


def _intervals(indices):
    ranges = []
    for index in sorted(indices):
        if ranges and index == ranges[-1][1] + 1:
            ranges[-1][1] = index
        else:
            ranges.append([index, index])
    return ranges


def _new_output(path):
    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty to avoid mixing curation runs: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def catalog(args):
    """Build a visual candidate inventory; no automatic identity claims."""
    manifest_path, sam_output = Path(args.manifest).resolve(), Path(args.sam_output).resolve()
    manifest, metadata, tracks, chunks = load_sam_run(manifest_path, sam_output)
    output = _new_output(args.output)
    items, rows = [], []
    frames = {frame["index"]: frame for frame in manifest["frames"]}
    size = (manifest["width"], manifest["height"])
    for number, (key, observations) in enumerate(sorted(tracks.items())):
        best = max(observations, key=lambda index: observations[index].get("area_pixels", 0))
        item = {
            **_track_dict(key), "chunk_range": [min(chunks[key[:2]]), max(chunks[key[:2]])],
            "returned_frame_ranges": _intervals(observations),
            "nonempty_frame_ranges": _intervals(i for i, obs in observations.items() if obs.get("area_pixels", 0) > 0),
            "best_frame": best, "best_time_seconds": frames[best]["time_seconds"],
            "max_area_pixels": observations[best].get("area_pixels"),
            "best_frame_score": observations[best].get("score"),
            "saved_instance_masks": all("mask" in obs for obs in observations.values()),
        }
        if "mask" in observations[best]:
            mask = read_mask(_inside(sam_output, observations[best]["mask"]), size)
            with Image.open(manifest_path.parent / frames[best]["file"]) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
            rgb[mask] = 0.45 * rgb[mask] + 0.55 * np.array([255, 65, 80], dtype=np.float32)
            preview = Image.fromarray(rgb.astype(np.uint8))
            preview.thumbnail((480, 360))
            relative = f"previews/track_{number:06d}.jpg"
            (output / relative).parent.mkdir(exist_ok=True)
            preview.save(output / relative, quality=90)
            item["preview"] = relative
        items.append(item)
        preview_html = f'<img src="{item["preview"]}" alt="Candidate mask overlay">' if "preview" in item else "No saved instance PNG"
        rows.append(f"<tr><td>{preview_html}</td><td><code>{html.escape(json.dumps(_track_dict(key)))}</code>"
                    f"<p>Source chunk: {item['chunk_range']}; best frame: {best}</p>"
                    f"<p>Returned frames: {html.escape(str(item['returned_frame_ranges']))}</p></td></tr>")
    write_json(output / "catalog.json", {
        "schema_version": 1, "source_manifest_sha256": sha256(manifest_path),
        "sam_output": str(sam_output), "tracks": items,
        "interpretation": "Candidate tracks, not unique physical objects; repeated concepts and chunks may refer to the same item",
    })
    write_json(output / "mapping.template.json", {"schema_version": 1, "objects": []})
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>SAM3 track candidates</title>'
        '<style>body{font:16px system-ui;margin:2rem;background:#171a20;color:#eee}'
        'table{border-collapse:collapse;width:100%}td{padding:1rem;border-bottom:1px solid #555}'
        'img{max-width:480px;width:100%;height:auto}code{overflow-wrap:anywhere}</style>'
        '<h1>SAM3 track candidates</h1><p>Match candidate tracks to physical items before assembling masks. '
        'Tracks may duplicate one item across prompts or chunks. Red overlays show predictions, not verified ground truth.</p>'
        '<p>Copy track references into your mapping file; cover every frame with selected tracks or explicit absence declarations.</p>'
        '<table>' + ''.join(rows) + '</table>\n'
    )
    print(f"Cataloged {len(items)} candidate tracks: {output / 'index.html'}")


def build_plan(mapping, frame_count, tracks, chunks):
    """Validate identity assignment and require an explicit disposition per frame."""
    objects = mapping.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("Mapping must contain a nonempty objects list")
    plans, names, assigned = [], set(), {}
    for obj in objects:
        name = _name(obj["name"])
        if name.casefold() in names:
            raise ValueError(f"Duplicate output object name: {name}")
        names.add(name.casefold())
        policy = obj.get("missing_detections", "error")
        if policy not in {"error", "reviewed_absent"}:
            raise ValueError("missing_detections must be 'error' or 'reviewed_absent'")
        active = [[] for _ in range(frame_count)]
        for reference in obj.get("tracks", []):
            key = _track_key(reference)
            if key not in tracks:
                raise ValueError(f"Referenced track does not exist: {key}")
            if key in assigned and assigned[key] != name:
                raise ValueError(f"Track {key} is assigned to both {assigned[key]} and {name}")
            assigned[key] = name
            chunk = chunks[key[:2]]
            start = _integer(reference.get("start_frame", min(chunk)), "start_frame")
            end = _integer(reference.get("end_frame", max(chunk)), "end_frame")
            if start > end or start not in chunk or end not in chunk:
                raise ValueError(f"Track frame range must stay inside its source chunk: {key}")
            for index in range(start, end + 1):
                if key in active[index]:
                    raise ValueError(f"Overlapping duplicate track selection for {name}, frame {index}: {key}")
                active[index].append(key)
        absent = set()
        for interval in obj.get("absent_ranges", []):
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("absent_ranges entries must be inclusive [start_frame, end_frame] pairs")
            start, end = (_integer(value, "absence range endpoint") for value in interval)
            if start > end or end >= frame_count:
                raise ValueError("Absence range must lie within the source video")
            absent.update(range(start, end + 1))
        uncovered = [i for i in range(frame_count) if not active[i] and i not in absent]
        if uncovered:
            raise ValueError(f"Object {name} has unreviewed/unmapped frames: {_intervals(uncovered)}")
        for index in range(frame_count):
            if index in absent:
                continue
            observed = [key for key in active[index] if index in tracks[key]]
            if not observed and policy == "error":
                raise ValueError(f"No selected detection for {name} on frame {index}; add explicit absence or review the gap")
            for key in observed:
                if "mask" not in tracks[key][index]:
                    raise ValueError(f"Selected track has no saved instance mask: {key}, frame {index}; rerun SAM with instance exports")
        plans.append({"name": name, "active": active, "absent": absent, "missing_detections": policy})
    return plans


def run(args):
    manifest_path, sam_output = Path(args.manifest).resolve(), Path(args.sam_output).resolve()
    mapping_path = Path(args.mapping).resolve()
    manifest, source_metadata, tracks, chunks = load_sam_run(manifest_path, sam_output)
    mapping = json.loads(mapping_path.read_text())
    plans = build_plan(mapping, len(manifest["frames"]), tracks, chunks)
    size = (manifest["width"], manifest["height"])
    # Check all selected references before writing. Missing files cannot mean absence.
    selected = {key for plan in plans for keys in plan["active"] for key in keys}
    for key in selected:
        for observation in tracks[key].values():
            if "mask" not in observation:
                raise ValueError(f"Selected track has no saved instance masks: {key}")
            path = _inside(sam_output, observation["mask"])
            if not path.is_file():
                raise FileNotFoundError(path)
    output = _new_output(args.output)
    metadata = {
        "schema_version": 1, "status": "running", "frame_count": len(manifest["frames"]),
        "width": size[0], "height": size[1], "objects": [plan["name"] for plan in plans],
        "manifest_sha256": sha256(manifest_path),
        "source_manifest_sha256": sha256(manifest_path), "source_sam_output": str(sam_output),
        "source_sam_metadata_sha256": sha256(sam_output / "metadata.json"),
        "source_sam_frames_sha256": sha256(sam_output / "frames.jsonl"),
        "mapping_sha256": sha256(mapping_path), "mapping": mapping,
        "mask_encoding": "Source-resolution grayscale uint8 PNG: background=0, foreground=255",
        "identity_basis": "User-curated track assignment; no automatic physical-object identity inference",
        "overlap_policy": "Independent binary masks; cross-object overlap reported, not removed",
        "absence_policy": "Unknown coverage fails; manual absences and reviewed missing detections are recorded per frame",
    }
    write_json(output / "metadata.json", metadata)
    overlap_rows = []
    try:
        with (output / "frames.jsonl").open("w") as handle:
            for frame in manifest["frames"]:
                index = frame["index"]
                occupancy = np.zeros((size[1], size[0]), dtype=np.uint32)
                for plan in plans:
                    union = np.zeros((size[1], size[0]), dtype=bool)
                    evidence = []
                    for key in plan["active"][index]:
                        observation = tracks[key].get(index)
                        item = _track_dict(key)
                        item["returned_on_frame"] = observation is not None
                        if observation is not None:
                            mask_path = _inside(sam_output, observation["mask"])
                            mask = read_mask(mask_path, size)
                            union |= mask
                            item["source_mask"] = observation["mask"]
                            item["area_pixels"] = int(mask.sum())
                            item["score"] = observation.get("score")
                        evidence.append(item)
                    if index in plan["absent"]:
                        if union.any():
                            raise ValueError(f"Explicit absence conflicts with a selected nonempty mask: {plan['name']}, frame {index}")
                        disposition = "manual_absence"
                    elif any(item["returned_on_frame"] for item in evidence):
                        disposition = "tracked" if union.any() else "model_empty"
                    else:
                        disposition = "reviewed_absent"
                    relative = Path("masks") / plan["name"] / f"{index:06d}.png"
                    (output / relative).parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(union.astype(np.uint8) * 255).save(output / relative)
                    occupancy += union.astype(np.uint32)
                    handle.write(json.dumps({
                        "object": plan["name"], "index": index, "time_seconds": frame["time_seconds"],
                        "pts_seconds": frame["pts_seconds"], "mask": relative.as_posix(),
                        "area_pixels": int(union.sum()), "disposition": disposition, "tracks": evidence,
                    }, allow_nan=False) + "\n")
                overlap_rows.append({"index": index, "overlap_pixels": int(np.count_nonzero(occupancy > 1))})
        write_json(output / "overlaps.json", {"frames": overlap_rows,
            "interpretation": "Pixels assigned to multiple curated objects; inspect for duplicate identity assignments or segmentation errors"})
        metadata["status"] = "complete"
        metadata["mask_count"] = len(plans) * len(manifest["frames"])
        metadata["frames_with_cross_object_overlap"] = sum(row["overlap_pixels"] > 0 for row in overlap_rows)
        write_json(output / "metadata.json", metadata)
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        write_json(output / "metadata.json", metadata)
        raise
    print(f"Assembled {len(plans)} curated objects over {len(manifest['frames'])} frames: {output}")
