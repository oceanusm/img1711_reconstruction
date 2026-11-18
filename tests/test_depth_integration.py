"""Exercise adapter I/O and resume without importing PyTorch or downloading weights."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import zipfile
from argparse import Namespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from scene_reconstruction.depth_anything3 import run


class DepthIntegrationTests(unittest.TestCase):
    def test_profile_window_limit_and_full_resume_share_provenance(self):
        model = Namespace()
        model.to = lambda _device: model
        model.eval = lambda: model
        model.inference = Mock(side_effect=lambda image, **kwargs: Namespace(
            depth=np.ones((len(image), 4, 8), dtype=np.float32)))
        api = types.ModuleType("depth_anything_3.api")
        api.DepthAnything3 = Namespace(from_pretrained=Mock(return_value=model))
        with tempfile.TemporaryDirectory(prefix="da3-profile-") as directory, patch.dict(
            sys.modules, {"depth_anything_3.api": api, "torch": types.ModuleType("torch")}
        ):
            root = Path(directory)
            Image.new("RGB", (16, 8)).save(root / "image.jpg")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"width": 16, "height": 8, "frames": [
                {"index": i, "file": "image.jpg", "time_seconds": i / 30} for i in range(3)
            ]}))
            args = Namespace(manifest=manifest, output=root / "output", model="synthetic",
                             chunk_size=2, overlap=0, process_res=504, device="cpu",
                             stop_after_windows=1)
            index_path = run(args)
            initial = json.loads(index_path.read_text())
            timing_path = args.output / "timings.json"
            timing = json.loads(timing_path.read_text())
            self.assertEqual(initial["status"], "incomplete")
            self.assertEqual([frame["index"] for frame in initial["frames"]], [0, 1])
            self.assertEqual(timing["status"], "incomplete")
            self.assertEqual(timing["stop_after_windows"], 1)
            self.assertEqual(timing["totals"]["completed_windows"], 1)
            self.assertEqual(model.inference.call_count, 1)
            config_before = (args.output / "run.json").read_bytes()
            self.assertNotIn("stop_after_windows", json.loads(config_before))
            cache_path = args.output / initial["chunks"][0]["file"]
            cache_before = cache_path.read_bytes()

            # A cached window also counts toward the per-invocation limit.
            run(args)
            self.assertEqual(model.inference.call_count, 1)
            limited_resume = json.loads(timing_path.read_text())
            self.assertEqual(limited_resume["status"], "incomplete")
            self.assertEqual(limited_resume["totals"]["cache_hits"], 1)
            self.assertEqual(limited_resume["totals"]["completed_windows"], 1)

            args.stop_after_windows = None
            run(args)
            self.assertEqual(model.inference.call_count, 2)
            complete = json.loads(index_path.read_text())
            self.assertEqual(complete["status"], "complete")
            self.assertEqual([frame["index"] for frame in complete["frames"]], [0, 1, 2])
            resumed_timing = json.loads(timing_path.read_text())
            self.assertEqual(resumed_timing["status"], "complete")
            self.assertEqual(resumed_timing["totals"]["cache_hits"], 1)
            self.assertEqual(resumed_timing["totals"]["inferred_windows"], 1)
            self.assertEqual((args.output / "run.json").read_bytes(), config_before)
            self.assertEqual(cache_path.read_bytes(), cache_before)

            args.stop_after_windows = 2
            run(args)
            self.assertEqual(json.loads(index_path.read_text())["status"], "complete")
            self.assertEqual(json.loads(timing_path.read_text())["status"], "complete")

    def test_profile_window_limit_requires_positive_integer(self):
        for limit in (0, -1, 1.5, "1"):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "positive integer"):
                run(Namespace(stop_after_windows=limit))

    def test_cuda_peak_measurements_are_per_inferred_window(self):
        cuda = Namespace(
            is_available=lambda: True,
            reset_peak_memory_stats=Mock(),
            max_memory_allocated=Mock(side_effect=[1000, 2000]),
            max_memory_reserved=Mock(side_effect=[3000, 4000]),
            empty_cache=Mock(),
        )
        model = Namespace()
        model.to = lambda _device: model
        model.eval = lambda: model
        model.inference = lambda image, **kwargs: Namespace(
            depth=np.ones((len(image), 4, 8), dtype=np.float32))
        api = types.ModuleType("depth_anything_3.api")
        api.DepthAnything3 = Namespace(from_pretrained=Mock(return_value=model))
        torch = types.ModuleType("torch")
        torch.cuda = cuda
        with tempfile.TemporaryDirectory(prefix="da3-memory-") as directory, patch.dict(
            sys.modules, {"depth_anything_3.api": api, "torch": torch}
        ):
            root = Path(directory)
            Image.new("RGB", (16, 8)).save(root / "image.jpg")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"width": 16, "height": 8, "frames": [
                {"index": i, "file": "image.jpg", "time_seconds": i / 30} for i in range(3)
            ]}))
            run(Namespace(manifest=manifest, output=root / "output", model="synthetic",
                          chunk_size=2, overlap=0, process_res=504, device="cuda:0"))
            timing = json.loads((root / "output" / "timings.json").read_text())
            self.assertEqual(cuda.reset_peak_memory_stats.call_count, 2)
            cuda.reset_peak_memory_stats.assert_called_with("cuda:0")
            self.assertEqual([item["cuda_peak_memory"]["allocated_bytes"] for item in timing["windows"]],
                             [1000, 2000])
            self.assertEqual(timing["totals"]["cuda_peak_memory"],
                             {"allocated_bytes": 2000, "reserved_bytes": 4000})

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

            timing = json.loads((args.output / "timings.json").read_text())
            self.assertEqual(timing["status"], "complete")
            self.assertEqual(timing["requested_compression_level"], 1)
            self.assertEqual(timing["totals"]["inferred_windows"], 2)
            self.assertEqual(timing["totals"]["cache_hits"], 0)
            self.assertEqual(timing["totals"]["frames_written"], 21)
            self.assertEqual(timing["totals"]["completed_windows"], 2)
            self.assertGreater(timing["totals"]["bytes_written"]["native_cache"], 0)
            self.assertIsNone(timing["totals"]["cuda_peak_memory"])
            for window in timing["windows"]:
                stages = window["seconds"]
                self.assertTrue(all(value >= 0 for value in stages.values()))
                self.assertAlmostEqual(stages["total"], sum(value for key, value in stages.items()
                                                          if key != "total"))
                self.assertGreater(stages["inference"], 0)
                self.assertGreater(stages["native_cache_write"], 0)
            total_stages = timing["totals"]["seconds"]
            self.assertAlmostEqual(total_stages["total"], sum(value for key, value in total_stages.items()
                                                            if key != "total"))

            # Old interrupted runs used NumPy's default compression and no storage
            # setting in run.json. A new storage preference must reuse those caches.
            config_before = (args.output / "run.json").read_bytes()
            chunk_bytes = {}
            for chunk in result["chunks"]:
                path = args.output / chunk["file"]
                with np.load(path, allow_pickle=False) as data:
                    legacy_arrays = {key: data[key] for key in data.files}
                np.savez_compressed(path, **legacy_arrays)
                chunk_bytes[path] = path.read_bytes()
            (args.output / result["frames"][-1]["file"]).unlink()
            args.compression_level = 0
            with patch.object(FakeModel, "from_pretrained", side_effect=AssertionError("Resume must use caches")):
                run(args)
            self.assertEqual(FakeModel.calls, 2)
            resumed = json.loads(index_path.read_text())
            self.assertEqual(resumed, result)
            self.assertEqual((args.output / "run.json").read_bytes(), config_before)
            for path, content in chunk_bytes.items():
                self.assertEqual(path.read_bytes(), content)
            resumed_timing = json.loads((args.output / "timings.json").read_text())
            self.assertEqual(resumed_timing["requested_compression_level"], 0)
            totals = resumed_timing["totals"]
            self.assertEqual(totals["inferred_windows"], 0)
            self.assertEqual(totals["cache_hits"], 2)
            self.assertEqual(totals["frames_written"], 21)
            self.assertEqual(totals["native_cache_bytes_read"], sum(len(data) for data in chunk_bytes.values()))
            self.assertEqual(totals["bytes_written"]["native_cache"], 0)
            self.assertEqual(totals["seconds"]["model_setup"], 0)
            self.assertEqual(totals["seconds"]["inference"], 0)
            self.assertGreater(totals["seconds"]["cache_read"], 0)
            for frame in resumed["frames"]:
                with zipfile.ZipFile(args.output / frame["file"]) as archive:
                    self.assertTrue(all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist()))

            args.chunk_size = 8
            with self.assertRaisesRegex(ValueError, "different run"):
                run(args)


if __name__ == "__main__":
    unittest.main()
