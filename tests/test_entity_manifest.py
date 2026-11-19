"""Entity grouping contracts prevent partial coverage and false SAM identity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from scene_reconstruction.common import sha256
from scene_reconstruction.entity_manifest import entity_object_lookup, load_entity_manifest, validate_entity_manifest


def fixture():
    entities = []
    for i, members in enumerate((["Jar body", "Jar lid"], ["Second jar"]), 1):
        entities.append({"id": i, "name": f"jar_{i}", "label": f"Jar {i}", "kind": "object",
                         "object_names": members, "concept": "jar", "sam_text": "jar",
                         "grouping_status": "name_supported", "notes": "Body and lid grouped."})
    return {"schema_version": 1, "source_blend_sha256": "a"*64, "source_objects_sha256": "b"*64,
            "entities": entities, "excluded_object_names": [],
            "concepts": [{"name": "jar", "text": "jar", "entity_ids": [1, 2],
                          "comparison_policy": "instance_association_required"}]}


class EntityManifestTests(unittest.TestCase):
    def test_whole_objects_keep_separate_instance_ids_and_shared_text(self):
        value = fixture()
        self.assertEqual(entity_object_lookup(value), {"Jar body": 1, "Jar lid": 1, "Second jar": 2})
        self.assertEqual(value["concepts"][0]["comparison_policy"], "instance_association_required")

    def test_missing_or_extra_mesh_names_rejected(self):
        source = [{"id": i, "name": name} for i, name in enumerate(["Jar body", "Jar lid", "Second jar"])]
        validate_entity_manifest(fixture(), source_objects=source)
        for extra in (source[:-1], source + [{"id": 3, "name": "Forgotten spoon"}]):
            with self.assertRaisesRegex(ValueError, "coverage"):
                validate_entity_manifest(fixture(), source_objects=extra)

    def test_duplicate_mesh_ownership_and_reserved_ids_rejected(self):
        value = fixture(); value["entities"][1]["object_names"].append("Jar lid")
        with self.assertRaisesRegex(ValueError, "multiple entities"):
            validate_entity_manifest(value)
        for invalid in [0, -1, 32768, True, 1.5]:
            value = fixture(); value["entities"][0]["id"] = invalid
            with self.assertRaisesRegex(ValueError, "IDs"):
                validate_entity_manifest(value)

    def test_conflicting_concept_text_and_union_identity_rejected(self):
        value = fixture(); value["entities"][1]["sam_text"] = "bottle"
        with self.assertRaisesRegex(ValueError, "inconsistent SAM text"):
            validate_entity_manifest(value)
        value = fixture(); value["concepts"][0]["comparison_policy"] = "single_entity"
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            validate_entity_manifest(value)

    def test_hashes_and_source_id_name_pairs_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary); objects = p/"objects.json"; blend = p/"scene.blend"; manifest = p/"entities.json"
            objects.write_text(json.dumps({"objects": [{"id": 0, "name": "Jar body"},
                                                        {"id": 1, "name": "Jar lid"},
                                                        {"id": 2, "name": "Second jar"}]}))
            blend.write_bytes(b"fixture")
            value = fixture(); value["source_objects_sha256"] = sha256(objects); value["source_blend_sha256"] = sha256(blend)
            value["entities"][0]["source_object_ids"] = [0, 1]
            manifest.write_text(json.dumps(value))
            load_entity_manifest(manifest, blend_path=blend, objects_path=objects)
            value["entities"][0]["source_object_ids"] = [1, 0]; manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "IDs/names"):
                load_entity_manifest(manifest, objects_path=objects)
            blend.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "scene hash"):
                load_entity_manifest(manifest, blend_path=blend)

    def test_checked_in_scene_profiles_are_complete_and_share_vocabulary(self):
        root = Path(__file__).resolve().parents[1]
        vocabulary = {}
        for name in ("before_vqa", "vggt_guided"):
            p = root/"configs"/f"entities_1711_{name}.json"
            value = load_entity_manifest(p)
            self.assertGreater(len(value["entities"]), 100)
            self.assertFalse(value["excluded_object_names"])
            for entity in value["entities"]:
                self.assertEqual(vocabulary.setdefault(entity["concept"], entity["sam_text"]), entity["sam_text"])
            lookup = entity_object_lookup(value)
            self.assertEqual(len(lookup), value["coverage"]["mesh_object_count"])
            self.assertGreater(sum(e["concept"] == "knife" for e in value["entities"]), 10)
            source = root/value["source_objects"]
            if source.exists():
                load_entity_manifest(p, objects_path=source)


if __name__ == "__main__":
    unittest.main()
