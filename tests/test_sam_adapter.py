"""SAM3 file/coordinate/session contracts, without importing model dependencies."""

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from scene_reconstruction.common import sha256

from scene_reconstruction.sam3_masks import (
    _save_frame,
    iter_chunks,
    load_prompts,
    normalize_output,
    propagation_direction,
    run,
)


def model_output(empty=False):
    masks = np.zeros((0 if empty else 2, 4, 6), dtype=bool)
    if not empty:
        masks[0, 1:3, 1:3] = True
        masks[1, 0, 0] = True
    return {
        "out_binary_masks": masks,
        "out_obj_ids": np.array([] if empty else [3, 8], dtype=np.int64),
        "out_probs": np.array([] if empty else [0.8, 0.3], dtype=np.float32),
        "out_boxes_xywh": np.ones((len(masks), 4), dtype=np.float32) * 0.1,
    }


class SamContractsTest(unittest.TestCase):
    def test_prompts_validate_names_and_global_anchors(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prompts.json"
            path.write_text(json.dumps({"objects": [{"name": "Cup_blue", "text": " blue cup "}]}))
            self.assertEqual(load_prompts(path, {10, 20}), [
                {"name": "Cup_blue", "text": "blue cup", "frame_index": 10}
            ])
            for entries in [
                [],
                [{"name": "../cup", "text": "cup"}],
                [{"name": "cup", "text": ""}],
                [{"name": "cup", "text": "cup", "frame_index": 0}],
                [{"name": "cup", "text": "cup", "frame_index": True}],
                [{"name": "cup", "text": "cup"}, {"name": "Cup", "text": "cup"}],
            ]:
                with self.subTest(entries=entries):
                    path.write_text(json.dumps({"objects": entries}))
                    with self.assertRaises(ValueError):
                        load_prompts(path, {10, 20})

    def test_chunks_cover_tail_and_zero_means_continuous(self):
        frames = [{"index": i} for i in range(7, 12)]
        self.assertEqual([len(chunk) for chunk in iter_chunks(frames, 2)], [2, 2, 1])
        self.assertEqual([frame for chunk in iter_chunks(frames, 2) for frame in chunk], frames)
        self.assertEqual(list(iter_chunks(frames, 0)), [frames])
        with self.assertRaises(ValueError):
            list(iter_chunks(frames, -1))

    def test_boundary_anchor_does_not_trigger_cached_second_pass(self):
        self.assertEqual(propagation_direction(0, 1), "forward")
        self.assertEqual(propagation_direction(0, 4), "forward")
        self.assertEqual(propagation_direction(2, 4), "both")
        self.assertEqual(propagation_direction(3, 4), "backward")
        with self.assertRaises(ValueError):
            propagation_direction(4, 4)

    def test_output_threshold_and_empty_frames(self):
        ids, masks, scores, boxes = normalize_output(model_output(), 4, 6, 0.5)
        self.assertEqual(ids.tolist(), [3])
        self.assertEqual(int(masks.sum()), 4)
        self.assertEqual(boxes.shape, (1, 4))
        self.assertGreater(scores[0], 0.5)
        ids, masks, scores, boxes = normalize_output(model_output(empty=True), 4, 6, 0.5)
        self.assertEqual(masks.shape, (0, 4, 6))
        self.assertEqual(ids.size, 0)
        self.assertEqual(boxes.shape, (0, 4))

    def test_output_rejects_wrong_grid_and_malformed_values(self):
        for key, value in [
            ("out_binary_masks", np.ones((2, 6, 4), dtype=bool)),
            ("out_binary_masks", np.ones((2, 4, 6), dtype=float) * 0.5),
            ("out_obj_ids", np.array([3, 3])),
            ("out_probs", np.array([float("nan"), 0.3])),
            ("out_probs", np.array([0.8])),
            ("out_boxes_xywh", np.ones((2, 5))),
        ]:
            with self.subTest(key=key, shape=value.shape):
                output = model_output()
                output[key] = value
                with self.assertRaises(ValueError):
                    normalize_output(output, 4, 6, 0.5)

    def test_source_grid_union_and_instance_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frame = {"index": 17, "time_seconds": 0.2, "pts_seconds": 0.8}
            record = _save_frame(root, {"name": "cup"}, 15, frame, model_output(), 4, 6, 0.0, True)
            mask = np.asarray(Image.open(root / record["mask"]))
            self.assertEqual(mask.shape, (4, 6))
            self.assertEqual(set(np.unique(mask)), {0, 255})
            self.assertEqual(np.count_nonzero(mask), 5)
            self.assertEqual(record["mask"], "masks/cup/000017.png")
            instances = [np.asarray(Image.open(root / item["mask"])) > 0 for item in record["instances"]]
            np.testing.assert_array_equal(mask > 0, np.logical_or.reduce(instances))
            self.assertTrue(all("chunk_000015" in item["mask"] for item in record["instances"]))
            self.assertEqual(record["index"], 17)
            empty = _save_frame(root, {"name": "hand"}, 15, frame, model_output(True), 4, 6, 0.5, True)
            self.assertEqual(empty["instances"], [])
            self.assertEqual(np.asarray(Image.open(root / empty["mask"])).sum(), 0)


class FakeDevice:
    def __init__(self, kind, index=None):
        self.type = kind.split(":")[0]
        self.index = index

    def __str__(self):
        return f"{self.type}:{self.index}"


class FakePredictor:
    """Simulate the official output grid and backward pass excluding its anchor."""

    def __init__(self, omit_index=None):
        self.omit_index = omit_index
        self.frame_count = 0
        self.requests = []
        self.closed_sessions = 0
        self.shutdown_called = False

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "start_session":
            self.frame_count = len(list(Path(request["resource_path"]).iterdir()))
            return {"session_id": "test"}
        if request["type"] == "add_prompt":
            return {"frame_index": request["frame_index"], "outputs": model_output()}
        if request["type"] == "close_session":
            self.closed_sessions += 1
            return {}
        raise AssertionError(request)

    def handle_stream_request(self, request):
        self.requests.append(request)
        anchor = request["start_frame_index"]
        direction = request["propagation_direction"]
        # Upstream switches to cached predictions after a boundary-started pass;
        # a last-frame anchor must never request a redundant forward pass.
        if anchor == self.frame_count - 1 and anchor > 0:
            if direction != "backward":
                raise AssertionError("last-frame anchor must propagate backward only")
        order = []
        if direction in ("forward", "both"):
            order.extend(range(anchor, self.frame_count))
        if direction in ("backward", "both"):
            order.extend(range(anchor - 1, -1, -1))
        for i in order:
            if i != self.omit_index:
                yield {"frame_index": i, "outputs": model_output(empty=(i == 0))}

    def shutdown(self):
        self.shutdown_called = True


class SamRunTest(unittest.TestCase):
    def run_fake(self, root, predictor, anchor=11):
        frames = []
        for i in range(10, 15):
            name = f"{i}.png"
            Image.new("RGB", (6, 4)).save(root / name)
            frames.append({"index": i, "file": name, "time_seconds": (i - 10) / 30, "pts_seconds": i / 30})
        (root / "manifest.json").write_text(json.dumps({"width": 6, "height": 4, "frames": frames}))
        (root / "prompts.json").write_text(json.dumps({"objects": [
            {"name": "cup", "text": "cup", "frame_index": anchor}
        ]}))
        (root / "checkpoint").write_bytes(b"fake checkpoint")
        sam = types.ModuleType("sam3")
        sam.__file__ = str(root / "sam3" / "__init__.py")
        builder = types.ModuleType("sam3.model_builder")
        builder.build_sam3_video_predictor = lambda **kwargs: predictor
        torch = types.ModuleType("torch")
        torch.device = FakeDevice
        torch.__version__ = "fake"
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: True, current_device=lambda: 0, set_device=lambda index: None,
        )
        args = types.SimpleNamespace(
            manifest=root / "manifest.json", prompts=root / "prompts.json", output=root / "output",
            device="cuda", chunk_size=2, checkpoint=root / "checkpoint", threshold=0.5, save_instances=True,
        )
        with patch.dict("sys.modules", {"torch": torch, "sam3": sam, "sam3.model_builder": builder}):
            run(args)

    def test_run_preserves_global_indices_and_all_frames_with_offloading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            predictor = FakePredictor()
            self.run_fake(root, predictor)
            metadata = json.loads((root / "output/metadata.json").read_text())
            records = [json.loads(line) for line in (root / "output/frames.jsonl").read_text().splitlines()]
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["mask_count"], 5)
            self.assertEqual(len(metadata["manifest_sha256"]), 64)
            self.assertEqual(metadata["checkpoint_sha256"], sha256(root / "checkpoint"))
            self.assertEqual([record["index"] for record in records], list(range(10, 15)))
            self.assertEqual([record["chunk_start_index"] for record in records], [10, 10, 12, 12, 14])
            self.assertEqual(len(list((root / "output/masks").rglob("*.png"))), 5)
            self.assertEqual(predictor.closed_sessions, 3)
            self.assertTrue(predictor.shutdown_called)
            sessions = [request for request in predictor.requests if request["type"] == "start_session"]
            self.assertTrue(all(r["offload_state_to_cpu"] and r["offload_video_to_cpu"] for r in sessions))

    def test_missing_propagation_frame_fails_and_closes_session(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            predictor = FakePredictor(omit_index=1)
            with self.assertRaisesRegex(RuntimeError, "omitted frames"):
                self.run_fake(root, predictor, anchor=10)
            metadata = json.loads((root / "output/metadata.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(predictor.closed_sessions, 1)
            self.assertTrue(predictor.shutdown_called)
            self.assertFalse((root / "output/masks/cup/000011.png").exists())


if __name__ == "__main__":
    unittest.main()
