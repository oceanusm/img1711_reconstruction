import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scene_reconstruction.surface_targets import extract_target_planes


def box(lower, upper):
    lower, upper = np.asarray(lower), np.asarray(upper)
    vertices = np.array([[x, y, z] for x in (lower[0], upper[0])
                         for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])
    quads = [(0, 4, 6, 2), (1, 3, 7, 5), (0, 1, 5, 4),
             (2, 6, 7, 3), (0, 2, 3, 1), (4, 5, 7, 6)]
    return vertices, np.array([[a, b, c] for a, b, c, d in quads]
                             + [[a, c, d] for a, b, c, d in quads])


def fixture(path, rotation=None, translation=None, scale=1, reverse=False, parallel_walls=False):
    specs = [
        ("Oak flooring 1", [0, -3, -.03], [4, 0, 0]),
        ("Left of range stone counter", [0, -.7, .86], [3.4, 0, .9]),
        ("Range wall stone backsplash", [0, -.01, .9], [4, 0, 1.8]),
        ("Return wall stone below window", [0, -2.4, .9], [.01, 0, 1.5]),
        ("Island quartz top", [2.7, -2.2, .88], [3.7, -1.4, .92]),
        ("Black countertop carafe", [1, -.3, .9], [1.2, -.1, 1.6]),
    ]
    if parallel_walls:
        specs[3] = (specs[3][0], [0, -.02, .9], [3, -.01, 1.7])
    vertices, triangles, ids = [], [], []
    for i, (name, lower, upper) in enumerate(specs):
        v, t = box(lower, upper)
        triangles.extend(t + len(vertices)); vertices.extend(v); ids.extend([i] * len(t))
    # A tiny horizontal sliver above the counter must not replace its broad top.
    triangles.append(np.arange(3) + len(vertices))
    vertices.extend([[0, 0, 1.4], [.01, 0, 1.4], [0, -.01, 1.4]]); ids.append(1)
    # A bevel-like face with an offset above the slab, but a sloping normal.
    triangles.append(np.arange(3) + len(vertices))
    vertices.extend([[0, -.7, .9], [3.4, -.7, .9], [0, -.69, .91]]); ids.append(1)
    vertices, triangles = np.asarray(vertices), np.asarray(triangles)
    if rotation is not None:
        vertices = vertices @ np.asarray(rotation).T
    vertices = scale * vertices
    if translation is not None:
        vertices += translation
    if reverse:
        triangles = triangles[:, ::-1]
    np.savez(path / "geometry.npz", vertices=vertices, triangles=triangles,
             triangle_object_ids=np.asarray(ids), object_names=np.array([s[0] for s in specs]))


def living_fixture(path, rotation=None, translation=None, scale=1, reverse=False,
                   parallel_walls=False, zero_height=False):
    specs = [
        ("Oak_plank_00_00", [0, 0, -.02], [4, 6, 0]),
        ("Dining_oak_slab_top", [2, 3, .75], [3, 5, .8]),
        ("Art_and_dining_wall", [4, 0, 0], [4.15, 6, 3]),
        ("Window_wall_pier_0.00", [0, -.15, 0], [.5, 0, 3]),
        ("Window_wall_pier_3.50", [3.5, -.15, 0], [4, 0, 3]),
        ("Window_wall_lintel", [0, -.15, 2.5], [4, 0, 3]),
        ("Window_wall_base", [0, -.15, 0], [4, 0, .2]),
        ("Coffee_table_rounded_oak_top", [1, 1, .36], [2, 2, .4]),
        ("Console_long_oak_top", [3.4, .2, .77], [3.9, 2.8, .81]),
        # All of these have broad planes but must remain excluded from anchors.
        ("Floor_substrate", [0, 0, -.15], [4, 6, -.03]),
        ("Window_1_roller_shade", [.5, .06, 1], [3.5, .076, 2.5]),
        ("Window_1_glass", [.5, -.08, .2], [3.5, -.07, 2.5]),
        ("Dining_oak_slab_top_decoration", [2.1, 3.1, .8], [2.9, 4.9, 1.2]),
    ]
    if parallel_walls:
        specs[2] = (specs[2][0], [0, -.15, 0], [4, 0, 3])
    if zero_height:
        specs[1] = (specs[1][0], [2, 3, -.02], [3, 5, 0])
    vertices, triangles, ids = [], [], []
    for i, (name, lower, upper) in enumerate(specs):
        v, t = box(lower, upper)
        triangles.extend(t + len(vertices)); vertices.extend(v); ids.extend([i] * len(t))
    vertices, triangles = np.asarray(vertices), np.asarray(triangles)
    if rotation is not None:
        vertices = vertices @ np.asarray(rotation).T
    vertices = scale * vertices
    if translation is not None:
        vertices += translation
    if reverse:
        triangles = triangles[:, ::-1]
    np.savez(path / "geometry.npz", vertices=vertices, triangles=triangles,
             triangle_object_ids=np.asarray(ids), object_names=np.array([s[0] for s in specs]))


class SurfaceTargetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_actual_top_and_room_faces_exclude_thickness_bevel_and_clutter(self):
        fixture(self.path)
        result = extract_target_planes(self.path)
        self.assertEqual([p["name"] for p in result["planes"]], ["floor", "counter", "back", "side"])
        np.testing.assert_allclose([p["normal"] for p in result["planes"]],
                                   [[0, 0, 1], [0, 0, 1], [0, -1, 0], [1, 0, 0]], atol=1e-12)
        np.testing.assert_allclose([p["offset"] for p in result["planes"]], [0, .9, .01, .01], atol=1e-12)
        self.assertEqual(result["planes"][1]["object_ids"], [1])
        self.assertEqual(result["planes"][1]["triangle_count"], 2)
        self.assertEqual(result["holdout_planes"][0]["object_ids"], [4])
        self.assertAlmostEqual(result["holdout_planes"][0]["offset"], .92)
        self.assertFalse(result["diagnostics"]["uses_camera_poses"])
        self.assertEqual(result["profile"], "named_kitchen_structural_planes_v1")
        self.assertEqual(result["diagnostics"]["profile_selection"]["selected"], "kitchen")
        self.assertEqual(result["diagnostics"]["offset_system_rank"], 4)
        json.dumps(result, allow_nan=False)

    def test_reversed_winding_does_not_select_back_or_bottom_faces(self):
        fixture(self.path, reverse=True)
        result = extract_target_planes(self.path)
        np.testing.assert_allclose([p["offset"] for p in result["planes"]], [0, .9, .01, .01], atol=1e-12)

    def test_named_sink_return_counter_is_included_but_island_remains_held_out(self):
        fixture(self.path)
        with np.load(self.path / "geometry.npz") as archive:
            data = dict(archive)
        vertices, triangles = box([0, -2.4, .86], [.7, -.7, .9])
        data["triangles"] = np.concatenate((data["triangles"], triangles + len(data["vertices"])))
        data["vertices"] = np.concatenate((data["vertices"], vertices))
        object_id = len(data["object_names"])
        data["object_names"] = np.append(data["object_names"], "Window return stone slab with sink opening")
        data["triangle_object_ids"] = np.append(data["triangle_object_ids"], [object_id] * len(triangles))
        np.savez(self.path / "geometry.npz", **data)
        result = extract_target_planes(self.path)
        counter = result["planes"][1]
        self.assertEqual(counter["object_ids"], [1, object_id])
        self.assertEqual(counter["triangle_count"], 4)
        self.assertAlmostEqual(counter["area"], 3.4 * .7 + .7 * 1.7)
        self.assertEqual(result["holdout_planes"][0]["object_ids"], [4])

    def test_vggt_guided_kitchen_aliases_select_actual_faces_and_exclude_similar_names(self):
        fixture(self.path)
        with np.load(self.path / "geometry.npz") as archive:
            data = dict(archive)
        names = data["object_names"].tolist()
        names[:4] = ["Pale oak plank flooring", "L-shaped sink and preparation worktop",
                     "Main stone backsplash", "Sink wall backsplash"]
        additions = [
            ("Spice counter slab", [3.5, -.7, .86], [4, 0, .9]),
            # Broad, offset surfaces would corrupt the anchors if names were
            # matched by a loose substring or prefix instead of exact aliases.
            ("Pale oak plank flooring sample", [0, -3, .15], [4, 0, .2]),
            ("L-shaped sink and preparation worktop decoration", [0, -.7, 1.1], [3.4, 0, 1.2]),
            ("Spice counter slab bottle", [3.5, -.7, 1.1], [4, 0, 1.2]),
            ("Main stone backsplash receptacle", [0, -.2, .9], [4, -.1, 1.8]),
            ("Sink wall backsplash trim", [.1, -2.4, .9], [.2, 0, 1.5]),
        ]
        for name, lower, upper in additions:
            vertices, triangles = box(lower, upper)
            data["triangles"] = np.concatenate((data["triangles"], triangles + len(data["vertices"])))
            data["vertices"] = np.concatenate((data["vertices"], vertices))
            data["triangle_object_ids"] = np.append(data["triangle_object_ids"], [len(names)] * len(triangles))
            names.append(name)
        data["object_names"] = np.asarray(names)
        np.savez(self.path / "geometry.npz", **data)

        result = extract_target_planes(self.path)
        self.assertEqual(result["diagnostics"]["profile_selection"]["selected"], "kitchen")
        np.testing.assert_allclose([p["normal"] for p in result["planes"]],
                                   [[0, 0, 1], [0, 0, 1], [0, -1, 0], [1, 0, 0]], atol=1e-12)
        np.testing.assert_allclose([p["offset"] for p in result["planes"]], [0, .9, .01, .01], atol=1e-12)
        self.assertEqual([p["object_ids"] for p in result["planes"]], [[0], [1, 6], [2], [3]])
        self.assertEqual([p["triangle_count"] for p in result["planes"]], [2, 4, 2, 2])
        self.assertAlmostEqual(result["planes"][1]["area"], (3.4 + .5) * .7)
        self.assertEqual(result["holdout_planes"][0]["object_ids"], [4])

        # A similarly named object cannot fill a missing required anchor.
        data["object_names"][0] = "Unlabeled floor"
        np.savez(self.path / "geometry.npz", **data)
        with self.assertRaisesRegex(ValueError, "kitchen missing floor"):
            extract_target_planes(self.path)

    def test_rotation_translation_and_scale_are_equivariant_without_cameras(self):
        a, b = .4, -.7
        rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
        rotation, translation, scale = rz @ rx, np.array([2., -3, .7]), 2.7
        fixture(self.path, rotation=rotation, translation=translation, scale=scale)
        result = extract_target_planes(self.path)
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, -1, 0], [1, 0, 0]]) @ rotation.T
        expected_offsets = scale * np.array([0, .9, .01, .01]) + normals @ translation
        np.testing.assert_allclose([p["normal"] for p in result["planes"]], normals, atol=1e-12)
        np.testing.assert_allclose([p["offset"] for p in result["planes"]], expected_offsets, atol=1e-12)
        self.assertAlmostEqual(result["diagnostics"]["floor_counter_separation"], .9 * scale)

    def test_unknown_object_names_fail_instead_of_guessing_or_using_cameras(self):
        fixture(self.path)
        with np.load(self.path / "geometry.npz") as archive:
            data = dict(archive)
        data["object_names"] = np.array(["Unknown"] * 6)
        np.savez(self.path / "geometry.npz", **data)
        with self.assertRaisesRegex(ValueError, "No complete named surface profile"):
            extract_target_planes(self.path)
        with self.assertRaisesRegex(ValueError, "Missing named floor"):
            extract_target_planes(self.path, profile="kitchen")

    def test_parallel_backsplashes_reject_unconstrained_fit(self):
        fixture(self.path, parallel_walls=True)
        with self.assertRaisesRegex(ValueError, "span three dimensions"):
            extract_target_planes(self.path)

    def test_bad_object_references_fail(self):
        fixture(self.path)
        with np.load(self.path / "geometry.npz") as archive:
            data = dict(archive)
        data["triangle_object_ids"][0] = 999
        np.savez(self.path / "geometry.npz", **data)
        with self.assertRaisesRegex(ValueError, "object IDs"):
            extract_target_planes(self.path)

    def test_living_dining_profile_uses_opaque_finite_faces_and_optional_checks(self):
        living_fixture(self.path)
        result = extract_target_planes(self.path)
        self.assertEqual(result["profile"], "named_living_dining_structural_planes_v1")
        self.assertEqual(result["diagnostics"]["profile_selection"],
                         {"requested": "auto", "selected": "living_dining"})
        self.assertEqual([p["name"] for p in result["planes"]], ["floor", "counter", "back", "side"])
        self.assertEqual([p["display_name"] for p in result["planes"]],
                         ["Floor", "Dining tabletop", "Art/dining wall", "Window wall"])
        np.testing.assert_allclose([p["normal"] for p in result["planes"]],
                                   [[0, 0, 1], [0, 0, 1], [-1, 0, 0], [0, 1, 0]], atol=1e-12)
        np.testing.assert_allclose([p["offset"] for p in result["planes"]], [0, .8, -4, 0], atol=1e-12)
        self.assertEqual([p["object_ids"] for p in result["planes"]], [[0], [1], [2], [3, 4, 5, 6]])
        self.assertAlmostEqual(result["planes"][0]["area"], 24)
        self.assertAlmostEqual(result["planes"][1]["area"], 2)
        wall_triangles = np.asarray(result["planes"][3]["triangles"])
        self.assertEqual(len(wall_triangles), 8)
        # The window's central opening must not be filled by a wall bounding box.
        point = np.array([2, 0, 1])
        inside_bounds = np.all((point >= wall_triangles.min(axis=1) - 1e-10)
                               & (point <= wall_triangles.max(axis=1) + 1e-10), axis=1)
        self.assertFalse(inside_bounds.any())
        self.assertEqual([p["name"] for p in result["holdout_planes"]], ["coffee_table_top", "console_top"])
        np.testing.assert_allclose([p["offset"] for p in result["holdout_planes"]], [.4, .81], atol=1e-12)
        self.assertEqual(result["diagnostics"]["normal_rank"], 3)
        self.assertEqual(result["diagnostics"]["offset_system_rank"], 4)
        self.assertAlmostEqual(result["diagnostics"]["floor_raised_surface_separation"], .8)
        self.assertAlmostEqual(result["diagnostics"]["floor_counter_separation"], .8)
        json.dumps(result, allow_nan=False)

    def test_living_dining_rotated_scaled_and_reversed_faces(self):
        a, b = .6, -.3
        rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
        rotation, translation, scale = rz @ rx, np.array([-2., .3, 4]), 1.8
        living_fixture(self.path, rotation=rotation, translation=translation, scale=scale, reverse=True)
        result = extract_target_planes(self.path, profile="living_dining")
        normals = np.array([[0, 0, 1], [0, 0, 1], [-1, 0, 0], [0, 1, 0]]) @ rotation.T
        expected_offsets = scale * np.array([0, .8, -4, 0]) + normals @ translation
        np.testing.assert_allclose([p["normal"] for p in result["planes"]], normals, atol=1e-12)
        np.testing.assert_allclose([p["offset"] for p in result["planes"]], expected_offsets, atol=1e-12)
        self.assertAlmostEqual(result["diagnostics"]["floor_raised_surface_separation"], .8 * scale)

    def test_auto_requires_complete_unambiguous_name_coverage(self):
        living_fixture(self.path)
        with np.load(self.path / "geometry.npz") as archive:
            living = dict(archive)
        fixture(self.path)
        with np.load(self.path / "geometry.npz") as archive:
            kitchen = dict(archive)
        combined = {
            "vertices": np.concatenate((living["vertices"], kitchen["vertices"])),
            "triangles": np.concatenate((living["triangles"], kitchen["triangles"] + len(living["vertices"]))),
            "triangle_object_ids": np.concatenate((living["triangle_object_ids"],
                                                    kitchen["triangle_object_ids"] + len(living["object_names"]))),
            "object_names": np.concatenate((living["object_names"], kitchen["object_names"])),
        }
        np.savez(self.path / "geometry.npz", **combined)
        with self.assertRaisesRegex(ValueError, "Ambiguous automatic surface profile"):
            extract_target_planes(self.path)
        self.assertEqual(extract_target_planes(self.path, "living_dining")["profile"],
                         "named_living_dining_structural_planes_v1")
        self.assertEqual(extract_target_planes(self.path, "kitchen")["profile"],
                         "named_kitchen_structural_planes_v1")
        living["object_names"][1] = "Unknown tabletop"
        np.savez(self.path / "geometry.npz", **living)
        with self.assertRaisesRegex(ValueError, "living_dining missing counter"):
            extract_target_planes(self.path)

    def test_living_dining_rejects_rank_deficient_anchors(self):
        living_fixture(self.path, parallel_walls=True)
        with self.assertRaisesRegex(ValueError, "span three dimensions"):
            extract_target_planes(self.path)
        living_fixture(self.path, zero_height=True)
        with self.assertRaisesRegex(ValueError, "separated vertical direction"):
            extract_target_planes(self.path)

    def test_holdout_surfaces_are_optional_and_unknown_profiles_fail(self):
        living_fixture(self.path)
        with np.load(self.path / "geometry.npz") as archive:
            data = dict(archive)
        data["object_names"][7:9] = "Unnamed object"
        np.savez(self.path / "geometry.npz", **data)
        self.assertEqual(extract_target_planes(self.path)["holdout_planes"], [])
        with self.assertRaisesRegex(ValueError, "Unknown surface profile"):
            extract_target_planes(self.path, profile="guess")


if __name__ == "__main__":
    unittest.main()
