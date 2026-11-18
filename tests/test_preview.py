import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from argparse import Namespace

import numpy as np
from PIL import Image

from scene_reconstruction.preview import colorize, color_limits, frame_durations, run


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
        with tempfile.TemporaryDirectory(prefix="depth-preview-test-") as directory:
            root = Path(directory)
            frames, predictions = [], []
            for i, (time, duration) in enumerate(zip([0, 0.05, 0.13], [0.05, 0.08, 0.04])):
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
            args = Namespace(manifest=manifest, depth_index=index, output=root / "preview.mp4", max_height=240, ffmpeg="ffmpeg")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                run(args)
            data["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            index.write_text(json.dumps(data))
            run(args)
            metadata = json.loads((root / "preview.scale.json").read_text())
            self.assertEqual(metadata["source_frame_count"], 3)
            self.assertAlmostEqual(metadata["duration_seconds"], 0.17)
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                "-show_entries", "frame=pts_time,duration_time", "-of", "json", str(args.output)
            ]))
            timestamps = [float(frame["pts_time"]) for frame in probe["frames"]]
            np.testing.assert_allclose(timestamps, [0, 0.05, 0.13, 0.17], atol=1e-6)
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == "__main__":
    unittest.main()
