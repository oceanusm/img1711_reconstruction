"""Validate and launch timestamp-matched Blender entity-ID rendering.

Object Index labels are hard, non-antialiased visible-surface assignments.
They are not Cryptomatte coverage mattes and do not encode transparency mixtures.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import numpy as np

from .blender_export import find_blender
from .common import load_manifest, sha256, write_json


def validate_entities(payload, source_hash=None):
    if source_hash is not None and payload.get("source_blend_sha256") != source_hash:
        raise ValueError("Entity manifest was generated for a different Blender scene")
    entities = payload.get("entities")
    if not isinstance(entities, list) or not entities:
        raise ValueError("Entity manifest requires a nonempty entities list")
    ids, names, objects = set(), set(), {}
    for entity in entities:
        identifier, name, members = entity.get("id"), entity.get("name"), entity.get("object_names")
        if isinstance(identifier, bool) or not isinstance(identifier, int) or not 1 <= identifier <= 32767:
            raise ValueError("Entity IDs must be positive integers no greater than 32767")
        if identifier in ids or not isinstance(name, str) or not name or name in names:
            raise ValueError("Entity IDs and names must be unique")
        if not isinstance(members, list) or not members:
            raise ValueError(f"Entity {name!r} needs object_names")
        ids.add(identifier); names.add(name)
        for member in members:
            if not isinstance(member, str) or not member or member in objects:
                raise ValueError(f"Invalid or multiply assigned Blender object: {member!r}")
            objects[member] = identifier
    excluded = payload.get("excluded_object_names", [])
    if not isinstance(excluded, list) or any(not isinstance(v, str) or not v for v in excluded):
        raise ValueError("excluded_object_names must be a list of exact object names")
    if len(set(excluded)) != len(excluded) or set(excluded) & set(objects):
        raise ValueError("Excluded objects must be unique and cannot belong to entities")
    return objects


def validate_selection(selection, manifest, manifest_hash):
    if selection.get("source_manifest_sha256") != manifest_hash:
        raise ValueError("Frame selection belongs to a different source manifest")
    if (selection.get("width"), selection.get("height")) != (manifest["width"], manifest["height"]):
        raise ValueError("Frame selection pixel grid differs from the canonical manifest")
    if selection.get("source_frame_count") != len(manifest["frames"]):
        raise ValueError("Frame selection source count differs from the canonical manifest")
    selected = selection.get("frames")
    if not isinstance(selected, list) or not selected:
        raise ValueError("Frame selection is empty")
    canonical = {f["index"]: f for f in manifest["frames"]}
    ids = [f.get("index") for f in selected]
    if any(isinstance(i, bool) or not isinstance(i, int) or i not in canonical for i in ids):
        raise ValueError("Frame selection contains invalid source IDs")
    if ids != sorted(set(ids)):
        raise ValueError("Frame selection source IDs must be unique and ordered")
    for frame in selected:
        original = canonical[frame["index"]]
        if frame.get("file") != original["file"] or not math.isfinite(frame.get("time_seconds", float("nan"))):
            raise ValueError("Frame selection files/times differ from canonical records")
        if abs(frame["time_seconds"] - original["time_seconds"]) > 1e-9:
            raise ValueError("Frame selection timestamps differ from canonical records")
    return [canonical[i] for i in ids]


def decode_index_pass(values, allowed_ids):
    """Reject corrupted/antialiased IDs rather than silently rounding colors."""
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Object Index pass must be a finite 2D image")
    rounded = np.rint(values)
    if np.max(np.abs(values-rounded), initial=0) > 1e-4:
        raise ValueError("Object Index pass contains noninteger labels")
    if not np.isin(rounded, [0, *allowed_ids]).all():
        raise ValueError("Object Index pass contains an unassigned entity ID")
    return rounded.astype(np.uint16)


def visibility_statistics(labels):
    result = []
    for identifier, count in zip(*np.unique(labels, return_counts=True)):
        if identifier == 0:
            continue
        y, x = np.where(labels == identifier)
        result.append({"id": int(identifier), "area_pixels": int(count),
                       "bbox_xyxy": [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]})
    return result


def run(args):
    paths = {name: Path(getattr(args, name)).expanduser().resolve()
             for name in ("blend", "manifest", "selection", "entities", "output")}
    for name in ("blend", "manifest", "selection", "entities"):
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    manifest = load_manifest(paths["manifest"])
    source_hash, manifest_hash = sha256(paths["blend"]), sha256(paths["manifest"])
    entities = json.loads(paths["entities"].read_text())
    validate_entities(entities, source_hash)
    selection = json.loads(paths["selection"].read_text())
    frames = validate_selection(selection, manifest, manifest_hash)
    subset = getattr(args, "frames", None)
    if subset and subset != "all":
        try:
            requested = {int(v) for v in subset.split(",")}
        except ValueError as error:
            raise ValueError("--frames must contain comma-separated selected source IDs") from error
        if not requested or not requested.issubset({f["index"] for f in frames}):
            raise ValueError("--frames must be a subset of the selected source IDs")
        frames = [f for f in frames if f["index"] in requested]
    scale = float(getattr(args, "resolution_scale", 1.0))
    if not math.isfinite(scale) or not 0 < scale <= 1 or min(manifest["width"], manifest["height"])*scale < 4:
        raise ValueError("resolution_scale must be in (0,1] and yield at least four pixels per axis")
    samples = int(getattr(args, "samples", 8))
    if samples < 1:
        raise ValueError("samples must be positive")
    device = getattr(args, "render_device", "CPU").upper()
    if device not in {"CPU", "METAL", "CUDA", "OPTIX"}:
        raise ValueError("render_device must be CPU, METAL, CUDA or OPTIX")
    output = paths["output"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Entity-render output must be new or empty: {output}")
    executable = find_blender(getattr(args, "blender", None))
    script = Path(__file__).resolve().parents[2]/"scripts/blender_entities.py"
    config = {k: str(v) for k, v in paths.items()}
    config.update(source_blend_sha256=source_hash, manifest_sha256=manifest_hash,
                  selection_sha256=sha256(paths["selection"]), entities_sha256=sha256(paths["entities"]),
                  render_indices=[f["index"] for f in frames], resolution_scale=scale,
                  samples=samples, rgb=bool(getattr(args, "rgb", True)), render_device=device)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"request.json", config)
    command = [executable, "--background", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
               "--python", str(script), "--", "--config", str(output/"request.json")]
    log_path = output/"blender.log"
    print(f"Rendering entity IDs and RGB; log: {log_path}", flush=True)
    try:
        with log_path.open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Blender entity render failed; see {log_path}\n" +
                               "\n".join(log_path.read_text(errors="replace").splitlines()[-15:]))
    finally:
        if sha256(paths["blend"]) != source_hash:
            raise RuntimeError("Original Blender scene changed during entity rendering")
    index = json.loads((output/"index.json").read_text())
    if index.get("status") != "complete" or index.get("source_blend_sha256") != source_hash:
        raise RuntimeError("Entity render did not produce a complete matching index")
    print(f"Entity rendering complete: {output/'index.json'}", flush=True)
    return output/"index.json"
