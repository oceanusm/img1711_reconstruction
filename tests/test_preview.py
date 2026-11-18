import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from scene_reconstruction.preview import colorize, color_limits, frame_durations, run
from scene_reconstruction.ffmpeg_compat import video_sync_args
from scene_reconstruction.frames import extract


class FFmpegCompatibilityTests(unittest.TestCase):
    def test_prefers_modern_option_and_supports_both_legacy_modes(self):
        with patch("scene_reconstruction.ffmpeg_compat.subprocess.check_output", return_value=(
                "  -vsync parameter  set video sync\n-fps_mode[:stream] mode  set frame mode\n")):
            self.assertEqual(video_sync_args("ffmpeg", "vfr"), ["-fps_mode", "vfr"])
            self.assertEqual(video_sync_args("ffmpeg", "passthrough"), ["-fps_mode", "passthrough"])
        with patch("scene_reconstruction.ffmpeg_compat.subprocess.check_output", return_value=(
                "FFmpeg build; no -fps_mode support\n  -vsync parameter  set video sync\n")):
            self.assertEqual(video_sync_args("ffmpeg", "vfr"), ["-vsync", "2"])
            self.assertEqual(video_sync_args("ffmpeg", "passthrough"), ["-vsync", "0"])

    def test_rejects_unrecognized_capabilities_and_failed_probe(self):
        with patch("scene_reconstruction.ffmpeg_compat.subprocess.check_output", return_value="no options"):
            with self.assertRaisesRegex(RuntimeError, "neither"):
                video_sync_args("ffmpeg", "vfr")
        with patch("scene_reconstruction.ffmpeg_compat.subprocess.check_output",
                   side_effect=subprocess.CalledProcessError(1, "ffmpeg", output="broken binary")):
            with self.assertRaisesRegex(RuntimeError, "broken binary"):
                video_sync_args("ffmpeg", "vfr")


class PreviewTests(unittest.TestCase):
    def test_fixed_scale_and_invalid_pixels(self):
        low, high = color_limits([np.array([1, 2, 3]), np.array([7, 8, 9])])
        self.assertLess(low, 2)
        self.assertGreater(high, 8)
        a = colorize(np.array([[2.0, np.nan]]), np.array([[True, False]]), low, high)
        b = colorize(np.array([[2.0, 8.0]]), np.ones((1, 2), bool), low, high)
        np.testing.assert_array_equal(a[0, 0], b[0, 0])
        np.testing.assert_array_equal(a[0, 1], [0, 0, 0])
        with self.assertRaises(ValueError):
            frame_durations([{"time_seconds": 0}])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires FFmpeg")
    def test_vfr_export_and_hash_guard(self):
        self._check_vfr_export("ffmpeg", video_sync_args("ffmpeg", "vfr"))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires FFmpeg")
    def test_legacy_option_path_preserves_preview_and_extraction_timestamps(self):
        # Simulate the older CLI surface, but perform real encoding/decoding.
        with tempfile.TemporaryDirectory(prefix="legacy-ffmpeg-test-") as directory:
            executable = Path(directory) / "ffmpeg-old-options"
            executable.write_text(
                f"#!{sys.executable}\nimport os, sys\n"
                "args = sys.argv[1:]\n"
                "if args == ['-hide_banner', '-h', 'full']:\n"
                "    print('-vsync parameter  set video sync method')\n"
                "    sys.exit(0)\n"
                "if '-fps_mode' in args:\n"
                "    sys.exit(2)\n"
                f"os.execv({shutil.which('ffmpeg')!r}, [{shutil.which('ffmpeg')!r}, *args])\n"
            )
            executable.chmod(0o755)
            self._check_vfr_export(str(executable), ["-vsync", "2"], check_extraction=True)

    def _check_vfr_export(self, ffmpeg, expected_sync_args, check_extraction=False):
        with tempfile.TemporaryDirectory(prefix="depth-preview-test-") as directory:
            root = Path(directory)
            frames, predictions = [], []
            for i, (time, duration) in enumerate(zip([0, 0.033367, 0.091234], [0.033367, 0.057867, 0.031667])):
                Image.new("RGB", (80, 60), (i * 80, 40, 80)).save(root / f"{i:06d}.jpg")
                depth = np.full((60, 80), i * 3 + 1, np.float32)
                np.savez_compressed(root / f"{i:06d}.npz", depth=depth, valid_depth=np.ones_like(depth, bool))
                frames.append({"index": i, "file": f"{i:06d}.jpg", "time_seconds": time, "duration_seconds": duration})
                predictions.append({"index": i, "file": f"{i:06d}.npz", "time_seconds": time, "chunk_id": 0, "depth_units": "estimated_meters"})
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "width": 80, "height": 60, "frames": frames}))
            index = root / "index.json"
            data = {"status": "complete", "frames": predictions, "manifest_sha256": "wrong"}
            index.write_text(json.dumps(data))
            args = Namespace(manifest=manifest, depth_index=index, output=root / "preview.mp4", max_height=240, ffmpeg=ffmpeg)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                run(args)
            data["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            index.write_text(json.dumps(data))
            with patch("scene_reconstruction.preview.video_sync_args", side_effect=RuntimeError("unsupported FFmpeg")), \
                    patch("scene_reconstruction.preview._load_depth") as read_depth:
                with self.assertRaisesRegex(RuntimeError, "unsupported FFmpeg"):
                    run(args)
                read_depth.assert_not_called()
            run(args)
            metadata = json.loads((root / "preview.scale.json").read_text())
            self.assertEqual(metadata["source_frame_count"], 3)
            self.assertAlmostEqual(metadata["duration_seconds"], 0.122901)
            self.assertEqual(metadata["ffmpeg_video_sync_args"], expected_sync_args)
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                "-show_entries", "frame=pts_time,duration_time", "-of", "json", str(args.output)
            ]))
            timestamps = [float(frame["pts_time"]) for frame in probe["frames"]]
            np.testing.assert_allclose(timestamps, [0, 0.033367, 0.091234, 0.122901], atol=1e-6)
            with self.assertRaises(FileExistsError):
                run(args)
            if check_extraction:
                extracted = root / "extracted"
                extract(Namespace(video=args.output, output=extracted, ffmpeg=ffmpeg, color="sdr"))
                decoded = json.loads((extracted / "manifest.json").read_text())
                self.assertEqual(decoded["frame_count"], 4)
                self.assertEqual(decoded["extraction"]["ffmpeg_video_sync_args"], ["-vsync", "0"])
                np.testing.assert_allclose([f["time_seconds"] for f in decoded["frames"]], timestamps, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
