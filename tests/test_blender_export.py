import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from scene_reconstruction.blender_export import (
    blender_frame, find_blender, intrinsics_from_projection, normalize_depth,
    pixel_resize_affine, ray_norms, run, select_frame_indices,
)
from scene_reconstruction.common import sha256


class BlenderMathTests(unittest.TestCase):
    def test_frame_selection_is_global_and_validated(self):
        self.assertEqual(select_frame_indices("4,0,4,2", 5), [0, 2, 4])
        self.assertEqual(select_frame_indices("all", 3), [0, 1, 2])
        for value in ["", "-1", "5", "x"]:
            with self.assertRaises(ValueError):
                select_frame_indices(value, 5)

    def test_timestamp_mapping_keeps_fractional_frames(self):
        frame, fraction = blender_frame(12.668333, 1, 30, 1)
        self.assertEqual(frame, 381)
        self.assertAlmostEqual(fraction, 0.04999)
        self.assertEqual(blender_frame(2, 10, 24, .5), (34, 0))

    def test_projection_to_intrinsics_includes_shift_and_pixel_centers(self):
        projection = np.array([[2, 0, .1, 0], [0, 3, -.2, 0], [0, 0, -1, -.2], [0, 0, -1, 0]])
        k = intrinsics_from_projection(projection, 100, 80)
        np.testing.assert_allclose(k, [[100, 0, 44.5], [0, 120, 31.5], [0, 0, 1]])
        point = np.array([.3, -.4, -2, 1])
        clip = projection @ point
        pixel = np.array([(clip[0] / clip[3] + 1) * 50 - .5, (1 - clip[1] / clip[3]) * 40 - .5])
        projected = k @ (np.diag([1, -1, -1]) @ point[:3])
        np.testing.assert_allclose(pixel, projected[:2] / projected[2])

    def test_resized_intrinsics_preserve_source_rays(self):
        affine = pixel_resize_affine((1080, 1920), (270, 480))
        k = np.array([[1600, 0, 539.5], [0, 1600, 959.5], [0, 0, 1]])
        native = affine @ k
        np.testing.assert_allclose(native, [[400, 0, 134.5], [0, 400, 239.5], [0, 0, 1]])
        source_pixel = np.array([120.0, 400.0, 1])
        np.testing.assert_allclose(np.linalg.inv(native) @ affine @ source_pixel, np.linalg.inv(k) @ source_pixel)

    def test_verified_camera_z_is_not_divided_by_ray_norm(self):
        raw = np.full((5, 7), 5.0, np.float32)
        intrinsic = np.array([[4, 0, 3], [0, 4, 2], [0, 0, 1]])
        z, valid, radial = normalize_depth(raw, intrinsic, "camera_z", 20)
        np.testing.assert_array_equal(z, raw)
        self.assertTrue(valid.all())
        self.assertGreater(radial[0, 0], 5)
        recovered, _, _ = normalize_depth(radial, intrinsic, "ray_distance", 20)
        np.testing.assert_allclose(recovered, raw, atol=1e-6)

    def test_no_hit_and_far_clip_are_invalid(self):
        raw = np.array([[0, np.inf], [1e10, 2]], np.float32)
        z, valid, _ = normalize_depth(raw, np.eye(3), "camera_z", 30)
        np.testing.assert_array_equal(valid, [[False, False], [False, True]])
        self.assertTrue(np.isnan(z[~valid]).all())
        with self.assertRaises(ValueError):
            normalize_depth(raw, np.eye(3), "ambiguous", 30)


class BlenderWrapperTests(unittest.TestCase):
    def test_launch_disables_embedded_scripts_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "scene.blend"
            source.write_bytes(b"unchanged source fixture")
            Image.new("RGB", (8, 12)).save(root / "000000.png")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "width": 8, "height": 12,
                                           "frames": [{"index": 0, "file": "000000.png", "time_seconds": 0}]}))
            output = root / "out"
            args = SimpleNamespace(blend=source, manifest=manifest, output=output, blender="blender",
                                   frames="all", depth=True, rgb=False, resolution_scale=1,
                                   time_scale=1, frame_start=None)
            def fake_run(command, **kwargs):
                self.assertIn("--disable-autoexec", command)
                self.assertIn("--factory-startup", command)
                self.assertIn("--python-exit-code", command)
                (output / "index.json").write_text(json.dumps({"status": "complete", "source_blend_sha256": sha256(source)}))
                return SimpleNamespace(returncode=0)
            with mock.patch("scene_reconstruction.blender_export.find_blender", return_value="blender"), mock.patch(
                "scene_reconstruction.blender_export.subprocess.run", side_effect=fake_run):
                run(args)
            self.assertEqual(source.read_bytes(), b"unchanged source fixture")
            request = json.loads((output / "request.json").read_text())
            self.assertEqual(request["render_indices"], [0])
            self.assertTrue((output / "blender.log").is_file())
            with self.assertRaises(FileExistsError):
                run(args)

    @unittest.skipUnless(os.environ.get("RUN_BLENDER_TESTS") == "1", "Opt-in actual Blender renderer test")
    def test_actual_blender_depth_pass_against_analytic_planes(self):
        executable = find_blender(os.environ.get("BLENDER_BIN"))
        script = Path(__file__).resolve().parents[1] / "scripts/blender_export.py"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "validation.json"
            subprocess.run([executable, "--background", "--factory-startup", "--disable-autoexec",
                            "--python-exit-code", "1", "--python", str(script), "--",
                            "--validate-depth-only", "--output", str(output)], check=True,
                           stdout=subprocess.DEVNULL)
            validation = json.loads(output.read_text())
            self.assertTrue(validation["performed"])
            self.assertEqual(len(validation["tests"]), 2)
            self.assertLess(max(t["max_absolute_z_error"] for t in validation["tests"]), .001)


if __name__ == "__main__":
    unittest.main()
