"""Exercise adapter I/O and resume without importing PyTorch or downloading weights."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from scene_reconstruction.depth_anything3 import run


class DepthIntegrationTests(unittest.TestCase):
    def test_full_frame_export_resume_and_configuration_guard(self):
        class FakeModel:
            calls = 0
            _commit_hash = "synthetic-checkpoint"

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                return cls()

            def to(self, _device):
                return self

            def eval(self):
                return self

            def inference(self, image, **_kwargs):
                type(self).calls += 1
                count = len(image)
                depth = np.stack([
                    np.full((4, 8), int(Path(path).stem) + 1, dtype=np.float32)
                    for path in image
                ])
                intrinsic = np.array([[4, 0, 4], [0, 4, 2], [0, 0, 1]], dtype=np.float32)
                extrinsic = np.repeat(np.eye(4, dtype=np.float32)[None, :3], count, axis=0)
                extrinsic[:, 0, 3] = -np.arange(count)
                sky = np.zeros_like(depth, dtype=bool)
                sky[:, 0, 0] = True
                return Namespace(
                    depth=depth, conf=np.ones_like(depth), sky=sky,
                    intrinsics=np.repeat(intrinsic[None], count, axis=0),
                    extrinsics=extrinsic, is_metric=True, scale_factor=1.0,
                )

        api = types.ModuleType("depth_anything_3.api")
        api.DepthAnything3 = FakeModel
        modules = {"depth_anything_3.api": api, "torch": types.ModuleType("torch")}
        with tempfile.TemporaryDirectory(prefix="da3-integration-") as directory, patch.dict(sys.modules, modules):
            root = Path(directory)
            frames = []
            for index in range(21):
                name = f"{index:06d}.jpg"
                Image.new("RGB", (16, 8)).save(root / name)
                frames.append({"index": index, "file": name, "time_seconds": index / 30,
                               "pts_seconds": index / 30, "duration_seconds": 1 / 30})
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "width": 16, "height": 8,
                                            "source": {}, "frames": frames}))
            args = Namespace(manifest=manifest, output=root / "output", model="synthetic",
                             chunk_size=16, overlap=4, process_res=504, device="cpu")
            index_path = run(args)
            result = json.loads(index_path.read_text())
            self.assertEqual(result["status"], "complete")
            self.assertEqual([item["index"] for item in result["frames"]], list(range(21)))
            self.assertEqual(FakeModel.calls, 2)
            self.assertEqual(len(result["chunks"]), 2)
            for frame in result["frames"]:
                self.assertEqual(frame["depth_units"], "estimated_meters")
                with np.load(args.output / frame["file"], allow_pickle=False) as data:
                    self.assertEqual(data["depth"].shape, (8, 16))
                    np.testing.assert_allclose(data["depth"], frame["index"] + 1)
                    self.assertFalse(data["valid_depth"][0, 0])
                    self.assertEqual(data["world_to_camera"].shape, (4, 4))
                    self.assertEqual(data["world_to_camera"][0, 3], -frame["local_index"])
                    self.assertEqual(data["camera_to_world"][0, 3], frame["local_index"])
                    np.testing.assert_allclose(data["intrinsics"], [[8, 0, 8], [0, 8, 4], [0, 0, 1]])
            diagnostics = json.loads((args.output / "overlap_diagnostics.json").read_text())
            self.assertEqual([item["frame_index"] for item in diagnostics["windows"][0]["frames"]], [12, 13, 14, 15])
            for item in diagnostics["windows"][0]["frames"]:
                self.assertEqual(item["median_current_over_previous_depth"], 1)

            with patch.object(FakeModel, "from_pretrained", side_effect=AssertionError("Resume must use caches")):
                run(args)
            self.assertEqual(FakeModel.calls, 2)
            resumed = json.loads(index_path.read_text())
            self.assertEqual(resumed, result)

            args.chunk_size = 8
            with self.assertRaisesRegex(ValueError, "different run"):
                run(args)


if __name__ == "__main__":
    unittest.main()
