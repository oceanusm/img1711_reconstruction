"""Prepare scene-scoped SAM inputs without turning predictions into entity truth.

The existing SAM video adapter accepts a sparse, ordered set of canonical frame
IDs. This module preserves those IDs and timestamps while deduplicating text
concepts. Every physical Blender entity remains explicitly unmapped until a
SAM instance identity is established; a concept union is never its surrogate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re

from PIL import Image

from .common import load_manifest, sha256, write_json
from .sam3_masks import load_prompts


SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def _read(path):
    return json.loads(Path(path).read_text())


def _relative(path, base):
    return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()


def selected_frames(manifest_path, selection_path):
    """Validate subset provenance, returning canonical rows rather than copies."""
    manifest_path, selection_path = Path(manifest_path).resolve(), Path(selection_path).resolve()
    manifest = load_manifest(manifest_path)
    selection = _read(selection_path)
    if selection.get("schema_version") != 1:
        raise ValueError("Expected a schema_version=1 frame selection")
    if selection.get("source_manifest_sha256") != sha256(manifest_path):
        raise ValueError("Selection source-manifest hash differs from canonical input")
    if any(selection.get(k) != manifest[k] for k in ("width", "height")):
        raise ValueError("Selection pixel grid differs from canonical input")
    if selection.get("source_frame_count") != len(manifest["frames"]):
        raise ValueError("Selection source frame count differs from canonical input")
    chosen = selection.get("frames")
    if not isinstance(chosen, list) or not chosen:
        raise ValueError("Selection must contain a nonempty frames list")
    ids = [row.get("index") for row in chosen if isinstance(row, dict)]
    if (len(ids) != len(chosen) or any(type(i) is not int or i < 0 for i in ids)
            or ids != sorted(set(ids)) or ids[-1] >= len(manifest["frames"])):
        raise ValueError("Selection frame IDs must be unique, ordered canonical integers")
    frames = []
    for row, index in zip(chosen, ids):
        canonical = manifest["frames"][index]
        time = row.get("time_seconds")
        if (not isinstance(time, (int, float)) or not math.isfinite(time)
                or abs(time-canonical["time_seconds"]) > 1e-6
                or row.get("file") != canonical["file"]):
            raise ValueError(f"Selection frame {index} differs from canonical time/image")
        frames.append(dict(canonical))
    return manifest, selection, frames


def entity_concepts(entity_manifest):
    """Return unique concepts and validated entities; duplicate physical IDs fail."""
    rows = entity_manifest.get("entities")
    if entity_manifest.get("schema_version") != 1 or not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty schema_version=1 entity manifest")
    concepts, ids, names, mesh_owners = {}, set(), set(), {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Entity entries must be objects")
        index, name = row.get("id"), row.get("name")
        if type(index) is not int or not 1 <= index <= 65535 or index in ids:
            raise ValueError("Entity IDs must be unique positive uint16 integers")
        if not isinstance(name, str) or not SAFE_NAME.fullmatch(name) or name.lower() in names:
            raise ValueError("Entity names must be unique safe identifiers")
        ids.add(index); names.add(name.lower())
        objects = row.get("object_names")
        if not isinstance(objects, list) or not objects or any(not isinstance(x, str) or not x for x in objects):
            raise ValueError(f"Entity {name} needs exact Blender object_names")
        for object_name in objects:
            if object_name in mesh_owners:
                raise ValueError(f"Blender object {object_name!r} belongs to multiple entity rows")
            mesh_owners[object_name] = name
        concept, text = row.get("concept"), row.get("sam_text")
        if not isinstance(concept, str) or not SAFE_NAME.fullmatch(concept):
            raise ValueError(f"Entity {name} needs a safe SAM concept identifier")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Entity {name} needs nonempty sam_text")
        text = text.strip()
        if concept.lower() in {key.lower() for key in concepts} and concept not in concepts:
            raise ValueError("SAM concept identifiers must be distinct ignoring case")
        group = concepts.setdefault(concept, {"name": concept, "text": text, "entities": []})
        if group["text"] != text:
            raise ValueError(f"Entities sharing concept {concept!r} must use identical sam_text")
        group["entities"].append(row)
    return list(concepts.values())


def _anchors(concepts, frames, render_index, entity_hash, selection_hash):
    """Use optional Blender visibility only to choose a text-prompt timestamp."""
    anchors = {c["name"]: {"frame_index": frames[0]["index"],
                           "source": "first_selected_source_frame", "visible_pixels": None}
               for c in concepts}
    if render_index is None:
        return anchors
    render = _read(render_index)
    if render.get("status") != "complete":
        raise ValueError("Prompt anchoring requires a complete Blender entity render")
    if render.get("entities_sha256") != entity_hash or render.get("selection_sha256") != selection_hash:
        raise ValueError("Render visibility provenance differs from entities/frame selection")
    by_id = {row["index"]: row for row in render.get("frames", [])}
    for frame in frames:
        if frame["index"] not in by_id or abs(by_id[frame["index"]]["time_seconds"]-frame["time_seconds"]) > 1e-6:
            raise ValueError("Render visibility lacks an exact selected timestamp")
    for concept in concepts:
        entity_ids = {row["id"] for row in concept["entities"]}
        areas = []
        for frame in frames:
            visibility = by_id[frame["index"]].get("entities", [])
            area = sum(row["area_pixels"] for row in visibility if row["id"] in entity_ids)
            if not math.isfinite(area) or area < 0:
                raise ValueError("Rendered visibility areas must be finite and nonnegative")
            areas.append((area, frame["index"]))
        best_area, best_frame = max(areas, key=lambda item: (item[0], -item[1]))
        anchors[concept["name"]] = {
            "frame_index": best_frame,
            "source": "maximum_blender_concept_visibility" if best_area else "first_selected_frame_no_blender_visibility",
            "visible_pixels": best_area,
        }
    return anchors


def _shared_anchor_overrides(path, concepts, frames, manifest_hash, selection_hash):
    """Restrict a shared union vocabulary to this scene, validating its contract."""
    shared = _read(path)
    if (shared.get("schema_version") != 1 or shared.get("source_manifest_sha256") != manifest_hash
            or shared.get("selection_sha256") != selection_hash):
        raise ValueError("Shared prompt provenance differs from canonical source/frame selection")
    prompts = {p["name"]: p for p in load_prompts(Path(path), {f["index"] for f in frames})}
    missing = {c["name"] for c in concepts} - prompts.keys()
    if missing:
        raise ValueError(f"Shared prompt file lacks this scene's concepts: {sorted(missing)}")
    provenance = shared.get("anchor_provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Shared prompt anchor_provenance must be an object")
    anchors = {}
    for concept in concepts:
        prompt = prompts[concept["name"]]
        if prompt["text"] != concept["text"]:
            raise ValueError(f"Shared SAM text differs for concept {concept['name']!r}")
        detail = provenance.get(concept["name"], {})
        if not isinstance(detail, dict) or detail.get("frame_index", prompt["frame_index"]) != prompt["frame_index"]:
            raise ValueError("Shared prompt anchor provenance disagrees with its frame ID")
        anchors[concept["name"]] = {**detail, "frame_index": prompt["frame_index"],
                                   "source": detail.get("source", "shared_prompt_file")}
    return anchors


def shared_prompts(args):
    """Choose identical text/anchors from summed visibility across scene inputs.

    This symmetric policy chooses *when* to issue a text prompt. It does not
    provide Blender boxes/masks to SAM, select SAM instances, or score agreement.
    """
    manifest_path, selection_path = Path(args.manifest).resolve(), Path(args.selection).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    canonical, _, frames = selected_frames(manifest_path, selection_path)
    entity_paths, render_paths = list(args.entities), list(args.render_indices)
    if len(entity_paths) < 2 or len(entity_paths) != len(render_paths):
        raise ValueError("Shared prompts need at least two paired entity manifests and render indices")
    manifest_hash, selection_hash = sha256(manifest_path), sha256(selection_path)
    expected_ids = [f["index"] for f in frames]
    vocabulary, scenes = {}, []
    seen_entities = set()
    for entity_path, render_path in zip(entity_paths, render_paths):
        entity_path, render_path = Path(entity_path).resolve(), Path(render_path).resolve()
        entities = _read(entity_path)
        concepts = entity_concepts(entities)
        entity_hash = sha256(entity_path)
        if entity_hash in seen_entities:
            raise ValueError("Each scene must appear only once in shared prompt selection")
        seen_entities.add(entity_hash)
        render = _read(render_path)
        if render.get("status") != "complete":
            raise ValueError("Shared prompts require complete Blender entity renders")
        if (render.get("entities_sha256") != entity_hash or render.get("selection_sha256") != selection_hash
                or render.get("source_manifest_sha256") != manifest_hash
                or render.get("source_blend_sha256") != entities.get("source_blend_sha256")):
            raise ValueError("Shared prompt render provenance differs from paired inputs")
        width, height = canonical["width"], canonical["height"]
        if (render.get("width"), render.get("height")) != (width, height):
            raise ValueError("Shared visibility areas require the same full canonical pixel grid")
        rows = render.get("frames", [])
        if [row.get("index") for row in rows] != expected_ids or render.get("frame_count") != len(frames):
            raise ValueError("Shared visibility must cover the exact ordered selected frame IDs")
        entity_to_concept = {e["id"]: c["name"] for c in concepts for e in c["entities"]}
        areas = {c["name"]: [0] * len(frames) for c in concepts}
        for position, (row, expected) in enumerate(zip(rows, frames)):
            for key in ("time_seconds", "pts_seconds"):
                actual = row.get(key)
                if (not isinstance(actual, (int, float)) or not math.isfinite(actual)
                        or abs(actual-expected[key]) > 1e-6):
                    raise ValueError("Shared visibility timestamps differ from canonical source")
            visible = row.get("entities")
            if not isinstance(visible, list):
                raise ValueError("Shared visibility needs an explicit per-frame entities list")
            seen, total = set(), 0
            for observation in visible:
                index, area = observation.get("id"), observation.get("area_pixels")
                if (type(index) is not int or index not in entity_to_concept or index in seen
                        or type(area) is not int or area < 0):
                    raise ValueError("Rendered entity visibility must use unique known IDs and nonnegative integer areas")
                seen.add(index); total += area
                areas[entity_to_concept[index]][position] += area
            if total > width*height:
                raise ValueError("Exclusive entity visibility exceeds the full frame area")
        for concept in concepts:
            name, text = concept["name"], concept["text"]
            if name.lower() in {key.lower() for key in vocabulary} and name not in vocabulary:
                raise ValueError("Shared concepts must be distinct ignoring case")
            if name in vocabulary and vocabulary[name] != text:
                raise ValueError(f"Shared concept {name!r} has conflicting SAM text across scenes")
            vocabulary[name] = text
        scenes.append({"scene_key": entities.get("scene_key", entity_path.stem),
                       "entities": _relative(entity_path, output.parent), "entities_sha256": entity_hash,
                       "source_blend_sha256": entities.get("source_blend_sha256"),
                       "render_index": _relative(render_path, output.parent), "render_index_sha256": sha256(render_path),
                       "areas": areas})
    scenes.sort(key=lambda row: (row["scene_key"], row["entities_sha256"]))
    objects, provenance = [], {}
    for name in sorted(vocabulary):
        combined = [sum(scene["areas"].get(name, [0]*len(frames))[i] for scene in scenes)
                    for i in range(len(frames))]
        best = max(range(len(frames)), key=lambda i: (combined[i], -frames[i]["index"]))
        objects.append({"name": name, "text": vocabulary[name], "frame_index": frames[best]["index"]})
        provenance[name] = {"source": "maximum_combined_blender_concept_visibility",
                            "frame_index": frames[best]["index"], "combined_visible_pixels": combined[best],
                            "no_rendered_visibility": combined[best] == 0,
                            "tie_policy": "earliest selected source frame",
                            "per_scene_visible_pixels": [{"scene_key": scene["scene_key"],
                                "entities_sha256": scene["entities_sha256"],
                                "render_index_sha256": scene["render_index_sha256"],
                                "concept_present": name in scene["areas"],
                                "visible_pixels": scene["areas"].get(name, [0]*len(frames))[best]}
                                for scene in scenes]}
    result = {"schema_version": 1, "status": "complete", "stage": "shared_entity_prompts",
              "source_manifest_sha256": manifest_hash, "selection_sha256": selection_hash,
              "width": canonical["width"], "height": canonical["height"],
              "objects": objects, "anchor_provenance": provenance,
              "inputs": [{k: v for k, v in scene.items() if k != "areas"} for scene in scenes],
              "policy": "Common concepts use identical text and one anchor selected by the sum of visible pixels across all paired scene renders.",
              "geometry_prompt_policy": "Only text and a source-frame timestamp are sent to SAM; no Blender boxes, masks or overlap scores.",
              "limitations": ["Visibility selects prompt timing and is not a ground-truth segmentation score.",
                              "A missing concept contributes zero visibility; no-visible concepts use the first selected frame.",
                              "Identical text/anchors remove this prompt difference between batches but do not guarantee identical stochastic SAM predictions."]}
    write_json(output, result)
    print(f"Shared SAM prompts: {len(objects)} concepts across {len(scenes)} scenes: {output}", flush=True)
    return output


def prepare(args):
    manifest_path, selection_path = Path(args.manifest).resolve(), Path(args.selection).resolve()
    entities_path, output = Path(args.entities).resolve(), Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"SAM preparation output must be empty: {output}")
    canonical, selection, frames = selected_frames(manifest_path, selection_path)
    entities = _read(entities_path)
    concepts = entity_concepts(entities)
    entity_hash, selection_hash = sha256(entities_path), sha256(selection_path)
    render_index = getattr(args, "render_index", None)
    anchors = _anchors(concepts, frames, render_index, entity_hash, selection_hash)
    anchor_prompts = getattr(args, "anchor_prompts", None)
    if anchor_prompts:
        anchors = _shared_anchor_overrides(anchor_prompts, concepts, frames, sha256(manifest_path), selection_hash)
    # Only image paths change. Original global IDs, PTS and source times survive.
    for frame in frames:
        path = (manifest_path.parent / frame["file"]).resolve()
        with Image.open(path) as image:
            if image.size != (canonical["width"], canonical["height"]):
                raise ValueError(f"Canonical image grid mismatch: {path}")
        frame["canonical_file"] = frame["file"]
        frame["file"] = _relative(path, output)
    subset = {"schema_version": 1, "scope": "selected_canonical_frames",
              "width": canonical["width"], "height": canonical["height"], "frame_count": len(frames),
              "source_manifest": _relative(manifest_path, output),
              "source_manifest_sha256": sha256(manifest_path),
              "source_frame_count": len(canonical["frames"]),
              "selection_sha256": selection_hash,
              "selection": selection.get("selection", {}),
              "pixel_coordinates": canonical.get("pixel_coordinates"), "frames": frames}
    prompts = {"schema_version": 1, "source_entities_sha256": entity_hash,
               "description": "Scene-scoped text concepts; each can detect multiple real-video instances.",
               "objects": [{"name": group["name"], "text": group["text"],
                             "frame_index": anchors[group["name"]]["frame_index"]} for group in concepts],
               "anchor_provenance": anchors,
               "geometry_prompt_policy": "Text only: no Blender boxes, points, masks or overlap scores are supplied to SAM."}
    candidates = {"schema_version": 1, "status": "awaiting_sam_instances",
                  "entities_sha256": entity_hash, "selection_sha256": selection_hash,
                  "source_blend_sha256": entities.get("source_blend_sha256"),
                  "mapping_policy": "Associate SAM instance tracks to physical entities before entity scoring; never use concept unions as per-entity masks.",
                  "selection_policy": "No maximum-overlap selection against Blender masks; that would favor the reconstruction being evaluated.",
                  "track_identity": "(concept, chunk_start_index, instance_id); names alone do not establish physical identity",
                  "concepts": [{"name": c["name"], "text": c["text"],
                                "entity_ids": [e["id"] for e in c["entities"]],
                                "entity_count": len(c["entities"]),
                                "union_semantics": "All detected concept instances, potentially including real items absent from Blender"}
                               for c in concepts],
                  "entities": [{"entity_id": e["id"], "name": e["name"], "label": e.get("label", e["name"]),
                                "concept": e["concept"], "grouping_status": e.get("grouping_status", "unreviewed"),
                                "status": "needs_instance_mapping", "mapping_required": True,
                                "matched_tracks": [], "use_concept_union": False}
                               for e in entities["entities"]]}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", subset)
    write_json(output / "prompts.json", prompts)
    write_json(output / "entity_candidates.json", candidates)
    preparation = {"schema_version": 1, "status": "complete", "stage": "entity_sam_preparation",
                   "canonical_manifest": _relative(manifest_path, output),
                   "canonical_manifest_sha256": sha256(manifest_path),
                   "selection": _relative(selection_path, output), "selection_sha256": selection_hash,
                   "entities": _relative(entities_path, output), "entities_sha256": entity_hash,
                   "manifest_sha256": sha256(output / "manifest.json"),
                   "prompts_sha256": sha256(output / "prompts.json"),
                   "entity_candidates_sha256": sha256(output / "entity_candidates.json"),
                   "render_index_sha256": sha256(Path(render_index)) if render_index else None,
                   "anchor_prompts": _relative(anchor_prompts, output) if anchor_prompts else None,
                   "anchor_prompts_sha256": sha256(Path(anchor_prompts)) if anchor_prompts else None,
                   "selected_frame_count": len(frames), "entity_count": len(entities["entities"]),
                   "concept_count": len(concepts),
                   "recommended_sam_arguments": {"chunk_size": 0, "save_instances": True, "threshold": .5},
                   "limitations": ["SAM inference is not performed by preparation; run the CUDA stage on the A6000.",
                                   "Tracking receives the sparse selected frames, not the full-rate source video; larger temporal gaps may harm tracking.",
                                   "A text session can detect multiple instances or none; every entity still needs identity association.",
                                   "Group semantics and segmentation errors require review before treating masks as evaluation references."]}
    write_json(output / "preparation.json", preparation)
    validate_prepared(output)
    print(f"Prepared SAM input: {len(frames)} source frames, {len(concepts)} concepts, "
          f"{len(entities['entities'])} entities awaiting instance mapping: {output}", flush=True)
    return output / "preparation.json"


def validate_prepared(directory):
    """Check a transferred preparation without importing Torch or SAM."""
    directory = Path(directory).resolve()
    preparation = _read(directory / "preparation.json")
    if preparation.get("status") != "complete":
        raise ValueError("SAM preparation is incomplete")
    for name in ("manifest", "prompts", "entity_candidates"):
        if sha256(directory / f"{name}.json") != preparation[f"{name}_sha256"]:
            raise ValueError(f"Prepared {name} changed after preparation")
    for name, key in (("canonical_manifest", "canonical_manifest_sha256"),
                      ("selection", "selection_sha256"), ("entities", "entities_sha256")):
        if sha256(directory / preparation[name]) != preparation[key]:
            raise ValueError(f"Transferred {name} differs from preparation")
    subset = _read(directory / "manifest.json")
    canonical, _, frames = selected_frames(directory / preparation["canonical_manifest"],
                                            directory / preparation["selection"])
    if len(frames) != len(subset["frames"]):
        raise ValueError("Prepared subset count differs from selection")
    for actual, expected in zip(subset["frames"], frames):
        if (actual["index"] != expected["index"] or actual["time_seconds"] != expected["time_seconds"]
                or actual["pts_seconds"] != expected["pts_seconds"]):
            raise ValueError("Prepared subset changed canonical identity or timing")
        path = (directory / actual["file"]).resolve()
        canonical_path = (directory / preparation["canonical_manifest"]).resolve().parent / expected["file"]
        if path != canonical_path.resolve():
            raise ValueError("Prepared source image path differs from canonical input")
        with Image.open(path) as image:
            if image.size != (canonical["width"], canonical["height"]):
                raise ValueError(f"Transferred source image grid mismatch: {path}")
    actual_prompts = load_prompts(directory / "prompts.json", {row["index"] for row in frames})
    concepts = entity_concepts(_read(directory / preparation["entities"]))
    if preparation.get("anchor_prompts"):
        shared_path = directory / preparation["anchor_prompts"]
        if sha256(shared_path) != preparation.get("anchor_prompts_sha256"):
            raise ValueError("Transferred shared anchor prompts differ from preparation")
        anchors = _shared_anchor_overrides(shared_path, concepts, frames,
                                           preparation["canonical_manifest_sha256"], preparation["selection_sha256"])
        expected = {c["name"]: {"name": c["name"], "text": c["text"],
                                "frame_index": anchors[c["name"]]["frame_index"]} for c in concepts}
        if {p["name"]: p for p in actual_prompts} != expected:
            raise ValueError("Prepared scene prompts differ from the shared text/anchor contract")
    return preparation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = validate_prepared(args.directory)
    print(f"SAM preparation valid: {result['selected_frame_count']} frames, {result['concept_count']} concepts")


if __name__ == "__main__":
    main()
