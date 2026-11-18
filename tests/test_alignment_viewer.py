"""Offline viewer payload integrity, without depending on a browser or Blender."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from scene_reconstruction import alignment, alignment_viewer
from scene_reconstruction.common import sha256, write_json
from scene_reconstruction.vggt_omega import _write_ply


class ViewerPayloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.frames = self.root / "frames"
        self.frames.mkdir()
        self.times = [0., 0.031, 0.067]
        for i in range(3):
            Image.new("RGB", (8, 12), (i * 40, 100, 180)).save(self.frames / f"{i}.png")
        self.manifest = self.frames / "manifest.json"
        write_json(self.manifest, {"schema_version": 1, "width": 8, "height": 12,
                                  "frames": [{"index": i, "time_seconds": time, "file": f"{i}.png"}
                                             for i, time in enumerate(self.times)]})
        self.blender = self.root / "blender"
        self.blender.mkdir()
        self.transform = alignment.validate_transform({"scale": 2.5,
            "rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]], "translation": [4, 5, 6]})
        self.source_poses = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
        self.source_poses[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        self.target_poses = alignment.transform_cameras(self.source_poses, self.transform)
        self.canonical_k = np.broadcast_to([[6., 0, 3.5], [0, 6., 5.5], [0, 0, 1]], (3, 3, 3)).copy()
        self.render_k = np.broadcast_to([[3., 0, 1.5], [0, 3., 2.5], [0, 0, 1]], (3, 3, 3)).copy()
        self.vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float32)
        self.triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        self.save_geometry()
        self.save_cameras(self.blender, self.target_poses)
        records = []
        for i, time in enumerate(self.times):
            image_file = f"render{i}.png"
            Image.new("RGB", (4, 6), (100, i * 40, 180)).save(self.blender / image_file)
            records.append({"index": i, "time_seconds": time, "camera_array_index": i, "rgb_file": image_file})
        write_json(self.blender / "index.json", {"schema_version": 1, "status": "complete",
                   "manifest_sha256": sha256(self.manifest), "canonical_width": 8, "canonical_height": 12,
                   "width": 4, "height": 6, "frames": records})
        self.args = SimpleNamespace(manifest=self.manifest, blender_export=self.blender,
                    vggt=None, alignment=None, max_points=100, max_triangles=100, max_images=0,
                    output=self.root / "viewer.html")

    def tearDown(self):
        self.temporary.cleanup()

    def save_cameras(self, folder, poses, ids=None):
        ids = np.array([0, 1, 2] if ids is None else ids, dtype=np.int64)
        np.savez_compressed(folder / "cameras.npz", frame_indices=ids, time_seconds=np.asarray(self.times)[ids],
                            camera_to_world=poses, world_to_camera=np.linalg.inv(poses),
                            intrinsics=self.render_k[ids], intrinsics_canonical=self.canonical_k[ids])

    def save_geometry(self, names=None):
        np.savez_compressed(self.blender / "geometry.npz", vertices=self.vertices, triangles=self.triangles,
                            triangle_object_ids=np.array([1, 0]),
                            object_names=np.array(names or ["counter", "jar"]),
                            face_colors=np.array([[255, 0, 0], [0, 128, 255]], dtype=np.uint8))

    def add_model(self):
        self.vggt = self.root / "vggt"
        self.vggt.mkdir()
        self.save_cameras(self.vggt, self.source_poses)
        write_json(self.vggt / "index.json", {"schema_version": 1, "status": "complete",
                   "source_manifest_sha256": sha256(self.manifest),
                   "preprocessing": {"canonical_size_wh": [8, 12]},
                   "frames": [{"index": i, "time_seconds": time, "camera_array_index": i}
                              for i, time in enumerate(self.times)]})
        self.cloud = np.array([[1.12345678, 2, 3], [0, 1, 1]], dtype=np.float32)
        _write_ply(self.vggt / "point_cloud.ply", [self.cloud],
                   [np.array([[255, 0, 100], [20, 40, 255]], dtype=np.uint8)],
                   [np.array([1, 2], dtype=np.float32)], [np.array([0, 2], dtype=np.int32)])
        with contextlib.redirect_stdout(io.StringIO()):
            result = alignment.run(SimpleNamespace(vggt=self.vggt, blender_export=self.blender,
                                      output=self.root / "alignment", robust=False))
        self.args.vggt = self.vggt
        self.args.alignment = result

    def mutate_json(self, path, change):
        value = json.loads(path.read_text())
        change(value)
        write_json(path, value)

    def test_baseline_only_has_render_grid_cameras_and_no_implied_model(self):
        payload = alignment_viewer.build_payload(self.args)
        self.assertIn("baseline only", payload["mode"])
        self.assertIn("pending", payload["mode"])
        self.assertIsNone(payload["alignment"])
        self.assertEqual(payload["points"], [])
        self.assertEqual(payload["vggtCameras"], [])
        self.assertEqual(payload["manifestHash"], sha256(self.manifest))
        self.assertEqual((payload["blenderWidth"], payload["blenderHeight"]), (4, 6))
        self.assertEqual((payload["sourceWidth"], payload["sourceHeight"]), (8, 12))
        np.testing.assert_array_equal(payload["blenderCameras"][0]["K"], self.render_k[0])
        np.testing.assert_array_equal(payload["blenderCameras"][0]["pose"], self.target_poses[0])
        np.testing.assert_array_equal(np.asarray(payload["vertices"])[payload["triangles"]],
                                      self.vertices[self.triangles])
        self.assertEqual(payload["objectIds"], [1, 0])
        self.assertFalse(any("image" in frame or "render" in frame for frame in payload["frames"]))

    def test_preview_budget_caps_both_source_and_render_images(self):
        self.args.max_images = 2
        payload = alignment_viewer.build_payload(self.args)
        self.assertEqual([frame["index"] for frame in payload["frames"] if "image" in frame], [0, 2])
        self.assertEqual([frame["index"] for frame in payload["frames"] if "render" in frame], [0, 2])
        self.assertTrue(payload["frames"][0]["render"].startswith("data:image/jpeg;base64,"))

    def test_source_previews_prioritize_retained_render_timestamps(self):
        def remove_end_renders(value):
            for record in value["frames"]:
                if record["index"] != 1:
                    record.pop("rgb_file")
        self.mutate_json(self.blender / "index.json", remove_end_renders)
        self.args.max_images = 2
        payload = alignment_viewer.build_payload(self.args)
        source_ids = {frame["index"] for frame in payload["frames"] if "image" in frame}
        render_ids = {frame["index"] for frame in payload["frames"] if "render" in frame}
        self.assertEqual(render_ids, {1})
        self.assertTrue(render_ids <= source_ids)
        self.assertEqual(len(source_ids), 2)

    def test_camera_paths_sorted_by_frame_even_if_archive_is_not(self):
        order = [2, 0, 1]
        self.save_cameras(self.blender, self.target_poses[order], ids=order)
        def reorder(value):
            value["frames"] = [value["frames"][i] for i in order]
            for i, frame in enumerate(value["frames"]):
                frame["camera_array_index"] = i
        self.mutate_json(self.blender / "index.json", reorder)
        payload = alignment_viewer.build_payload(self.args)
        self.assertEqual([camera["index"] for camera in payload["blenderCameras"]], [0, 1, 2])
        np.testing.assert_array_equal([camera["pose"] for camera in payload["blenderCameras"]], self.target_poses)

    def test_alignment_applies_once_and_preserves_raw_landmark_coordinates(self):
        self.add_model()
        payload = alignment_viewer.build_payload(self.args)
        np.testing.assert_allclose(payload["points"], alignment.transform_points(self.cloud, self.transform), atol=1e-6)
        np.testing.assert_array_equal(payload["originalPoints"], self.cloud)
        self.assertEqual(payload["pointFrames"], [0, 2])
        np.testing.assert_allclose([camera["pose"] for camera in payload["vggtCameras"]], self.target_poses, atol=1e-12)
        # VGGT must use canonical intrinsics even if an unrelated 'intrinsics' key exists.
        np.testing.assert_array_equal(payload["vggtCameras"][0]["K"], self.canonical_k[0])
        self.assertAlmostEqual(np.linalg.det(np.asarray(payload["vggtCameras"][0]["pose"])[:3, :3]), 1)

    def test_model_and_alignment_are_required_together(self):
        for key in ("vggt", "alignment"):
            with self.subTest(key=key):
                setattr(self.args, key, self.root / "missing")
                with self.assertRaisesRegex(ValueError, "together"):
                    alignment_viewer.build_payload(self.args)
                setattr(self.args, key, None)

    def test_stale_alignment_rejected_after_camera_or_index_change(self):
        self.add_model()
        path = self.vggt / "cameras.npz"
        original = path.read_bytes()
        self.save_cameras(self.vggt, self.source_poses + np.eye(4)[None] * 0.01)
        with self.assertRaisesRegex(ValueError, "provenance"):
            alignment_viewer.build_payload(self.args)
        path.write_bytes(original)
        self.mutate_json(self.blender / "index.json", lambda value: value.update(revised=True))
        with self.assertRaisesRegex(ValueError, "provenance"):
            alignment_viewer.build_payload(self.args)

    def test_changed_cloud_or_mesh_rejected_when_alignment_records_asset_hashes(self):
        self.add_model()
        for path in (self.vggt / "point_cloud.ply", self.blender / "geometry.npz"):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                if path.suffix == ".npz":
                    self.save_geometry(["changed counter", "jar"])
                else:
                    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
                with self.assertRaisesRegex(ValueError, "provenance"):
                    alignment_viewer.build_payload(self.args)
                path.write_bytes(original)

    def test_downloaded_landmark_provenance_roundtrip_and_wrong_baseline_rejection(self):
        self.add_model()
        payload = alignment_viewer.build_payload(self.args)
        document = dict(payload["landmarkProvenance"], pairs=[
            {"source": source.tolist(), "target": target.tolist(), "use_for_fit": True}
            for source, target in zip(self.source_poses[:, :3, 3], self.target_poses[:, :3, 3])])
        identifiers = document["inputs"]
        self.assertEqual(len(identifiers), 6)
        self.assertEqual(identifiers["blender_geometry_sha256"], sha256(self.blender / "geometry.npz"))
        self.assertEqual(identifiers["vggt_point_cloud_sha256"], sha256(self.vggt / "point_cloud.ply"))
        self.assertIn("JSON.stringify({...D.landmarkProvenance,pairs}", alignment_viewer.HTML)
        landmarks = self.root / "downloaded_landmarks.json"
        write_json(landmarks, document)
        arguments = SimpleNamespace(vggt=self.vggt, blender_export=self.blender, landmarks=landmarks,
                                    output=self.root / "landmark_refit", robust=False)
        with contextlib.redirect_stdout(io.StringIO()):
            result = json.loads(alignment.run(arguments).read_text())
        self.assertEqual(result["fit_input"]["validated_artifact_hashes"], identifiers)
        arguments.output = self.root / "invalid_landmark_refit"
        for key in identifiers:
            with self.subTest(identifier=key):
                bad = dict(document, inputs=dict(identifiers, **{key: "0" * 64}))
                write_json(landmarks, bad)
                with self.assertRaisesRegex(ValueError, key):
                    alignment.run(arguments)
        write_json(landmarks, dict(document, source_manifest_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "source_manifest_sha256"):
            alignment.run(arguments)
        write_json(landmarks, document)
        # Same source video does not make landmarks portable between Blender baselines.
        self.mutate_json(self.blender / "index.json", lambda value: value.update(source_blend_sha256="f" * 64))
        with self.assertRaisesRegex(ValueError, "blender_index_sha256"):
            alignment.run(arguments)
        self.assertFalse(arguments.output.exists())

    def test_incomplete_wrong_manifest_or_reflected_alignment_rejected(self):
        self.add_model()
        original = json.loads(self.args.alignment.read_text())
        changes = [({"status": "running"}, "complete"),
                   ({"source_manifest_sha256": "0" * 64}, "manifest"),
                   ({"transform": {"scale": 1, "rotation": np.diag([-1, 1, 1]).tolist(),
                                    "translation": [0, 0, 0]}}, "reflections")]
        for changeset, expected in changes:
            with self.subTest(changeset=changeset):
                write_json(self.args.alignment, dict(original, **changeset))
                with self.assertRaisesRegex(ValueError, expected):
                    alignment_viewer.build_payload(self.args)

    def test_baseline_index_timestamp_and_source_dimensions_are_checked(self):
        path = self.blender / "index.json"
        original = path.read_bytes()
        self.mutate_json(path, lambda value: value["frames"][0].update(time_seconds=.1))
        with self.assertRaisesRegex(ValueError, "timestamp"):
            alignment_viewer.build_payload(self.args)
        path.write_bytes(original)
        self.mutate_json(path, lambda value: value.update(canonical_width=12))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            alignment_viewer.build_payload(self.args)

    def test_html_is_standalone_with_safe_embedded_json_and_no_overwrite(self):
        name = "</script><script>unexpected()</script>"
        self.save_geometry(["counter", name])
        with contextlib.redirect_stdout(io.StringIO()):
            output = alignment_viewer.run(self.args)
        html = output.read_text()
        encoded = html.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
        payload = json.loads(encoded)
        self.assertEqual(payload["objectNames"][1], name)
        self.assertNotIn(name, html)
        self.assertNotIn('<script src=', html)
        with self.assertRaisesRegex(ValueError, "new .html"):
            alignment_viewer.run(self.args)


class ViewerMeshTests(unittest.TestCase):
    def test_sampling_remaps_indices_and_retains_object_labels(self):
        vertices = np.arange(30).reshape(10, 3)
        triangles = np.array([[0, 2, 4], [1, 3, 5], [7, 8, 9], [0, 8, 9]])
        object_ids = np.array([4, 3, 2, 1])
        sampled, faces, objects, chosen = alignment_viewer.sample_mesh(vertices, triangles, object_ids, 2)
        np.testing.assert_array_equal(sampled[faces], vertices[triangles[chosen]])
        np.testing.assert_array_equal(objects, object_ids[chosen])
        self.assertEqual(chosen.tolist(), [0, 3])

    def test_invalid_triangle_and_object_indices_fail(self):
        vertices = np.zeros((3, 3))
        with self.assertRaisesRegex(ValueError, "bounds"):
            alignment_viewer.sample_mesh(vertices, np.array([[0, 1, 3]]), [0], 10)
        with self.assertRaisesRegex(ValueError, "object IDs"):
            alignment_viewer.sample_mesh(vertices, np.array([[0, 1, 2]]), [], 10)


if __name__ == "__main__":
    unittest.main()
