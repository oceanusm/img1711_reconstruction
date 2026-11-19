"""Validate explicit whole-entity groupings for Blender/SAM comparisons.

The checked-in IMG_1711 configurations are reviewed name-based proposals. This
module validates coverage and identity contracts; it does not infer semantics
from mesh names or turn SAM concept detections into physical correspondences.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from .common import sha256


_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def validate_entity_manifest(payload, *, source_objects=None):
    """Validate IDs, exhaustive mesh ownership and shared concept vocabulary.

    ``source_objects`` may be the ``objects`` list from the Blender geometry
    export. Without it, internal consistency is checked but coverage cannot be
    independently established. The input is returned unchanged.
    """
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Expected entity manifest schema_version=1")
    entities = payload.get("entities")
    if not isinstance(entities, list) or not entities:
        raise ValueError("Entity manifest requires a nonempty entities list")
    for key in ("source_blend_sha256", "source_objects_sha256"):
        if not isinstance(payload.get(key), str) or not _SHA256.fullmatch(payload[key]):
            raise ValueError(f"Entity manifest needs a valid {key}")
    if payload.get("entity_id_background", 0) != 0:
        raise ValueError("Entity ID 0 is reserved for background")
    ids, names, objects, vocabulary, concept_ids = set(), set(), {}, {}, {}
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError("Each entity must be an object")
        identity = entity.get("id")
        if isinstance(identity, bool) or not isinstance(identity, int) or not 1 <= identity <= 32767:
            raise ValueError("Entity IDs must be integers in 1..32767 for Blender IndexOB")
        if identity in ids:
            raise ValueError(f"Duplicate entity ID: {identity}")
        ids.add(identity)
        name = entity.get("name")
        if not isinstance(name, str) or not _SLUG.fullmatch(name):
            raise ValueError("Entity names must be safe nonempty slugs")
        if name.lower() in names:
            raise ValueError(f"Duplicate entity name: {name}")
        names.add(name.lower())
        if entity.get("kind") not in {"object", "surface"}:
            raise ValueError(f"Entity {name} needs object/surface kind")
        if entity.get("grouping_status") not in {"name_supported", "needs_review"}:
            raise ValueError(f"Entity {name} has unsupported grouping status")
        if not isinstance(entity.get("label"), str) or not entity["label"].strip():
            raise ValueError(f"Entity {name} needs a display label")
        if not isinstance(entity.get("notes"), str) or not entity["notes"].strip():
            raise ValueError(f"Entity {name} needs grouping rationale")
        members = entity.get("object_names")
        if not isinstance(members, list) or not members:
            raise ValueError(f"Entity {name} must own at least one Blender object")
        for member in members:
            if not isinstance(member, str) or not member:
                raise ValueError(f"Entity {name} contains an invalid Blender object name")
            if member in objects:
                raise ValueError(f"Blender object {member!r} belongs to multiple entities")
            objects[member] = identity
        concept, text = entity.get("concept"), entity.get("sam_text")
        if not isinstance(concept, str) or not _SLUG.fullmatch(concept):
            raise ValueError(f"Entity {name} needs a safe SAM concept slug")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Entity {name} needs a SAM text proposal")
        if concept in vocabulary and vocabulary[concept] != text:
            raise ValueError(f"Concept {concept} has inconsistent SAM text")
        vocabulary[concept] = text
        concept_ids.setdefault(concept, []).append(identity)
    excluded = payload.get("excluded_object_names", [])
    if (not isinstance(excluded, list) or any(not isinstance(x, str) or not x for x in excluded)
            or len(excluded) != len(set(excluded))):
        raise ValueError("Excluded object names must be a unique string list")
    if set(excluded) & set(objects):
        raise ValueError("An object cannot be both assigned and excluded")
    if source_objects is not None:
        source_names = [obj["name"] for obj in source_objects]
        if len(source_names) != len(set(source_names)):
            raise ValueError("Repeated evaluated object names need explicit instance handling before entity rendering")
        expected, actual = set(source_names), set(objects) | set(excluded)
        if expected != actual:
            raise ValueError(f"Entity mesh coverage differs: missing={sorted(expected-actual)}, unknown={sorted(actual-expected)}")
        source_by_id = {obj["id"]: obj["name"] for obj in source_objects}
        for entity in entities:
            if "source_object_ids" in entity:
                try:
                    referenced = [source_by_id[i] for i in entity["source_object_ids"]]
                except (KeyError, TypeError) as error:
                    raise ValueError("Invalid source object ID in entity manifest") from error
                if referenced != entity["object_names"]:
                    raise ValueError(f"Source object IDs/names disagree for {entity['name']}")
    if "concepts" in payload:
        records = payload["concepts"]
        if not isinstance(records, list) or len(records) != len(vocabulary):
            raise ValueError("Concept summary does not cover the entity vocabulary")
        seen = set()
        for record in records:
            name = record.get("name")
            if name in seen or name not in vocabulary or record.get("text") != vocabulary[name]:
                raise ValueError("Concept summary has duplicate, missing or conflicting definitions")
            seen.add(name)
            if record.get("entity_ids") != concept_ids[name]:
                raise ValueError(f"Concept summary entity IDs disagree for {name}")
            expected = "instance_association_required" if len(concept_ids[name]) > 1 else "single_entity_requires_semantic_review"
            if record.get("comparison_policy") != expected:
                raise ValueError(f"Concept {name} cannot assert an unreviewed one-to-one mask identity")
    coverage = payload.get("coverage", {})
    expected_counts = {"assigned_object_count": len(objects), "entity_count": len(entities),
                       "concept_count": len(vocabulary),
                       "needs_review_entity_count": sum(e["grouping_status"] == "needs_review" for e in entities)}
    if source_objects is not None:
        expected_counts["mesh_object_count"] = len(source_objects)
    for key, count in expected_counts.items():
        if key in coverage and coverage[key] != count:
            raise ValueError(f"Entity coverage count {key} disagrees with actual assignments")
    return payload


def load_entity_manifest(path, *, blend_path=None, objects_path=None):
    """Load a manifest, optionally verifying exact source file hashes/coverage."""
    payload = json.loads(Path(path).read_text())
    source_objects = None
    if objects_path is not None:
        objects_path = Path(objects_path)
        if sha256(objects_path) != payload.get("source_objects_sha256"):
            raise ValueError("Blender object export hash differs from entity manifest")
        source_objects = json.loads(objects_path.read_text())["objects"]
    if blend_path is not None and sha256(Path(blend_path)) != payload.get("source_blend_sha256"):
        raise ValueError("Blender scene hash differs from entity manifest")
    return validate_entity_manifest(payload, source_objects=source_objects)


def entity_object_lookup(payload):
    """Map exact Blender object names to their scene-local positive entity ID."""
    validate_entity_manifest(payload)
    return {name: entity["id"] for entity in payload["entities"] for name in entity["object_names"]}
