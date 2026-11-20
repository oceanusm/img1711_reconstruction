"""Native SAM3 dense adapter contracts; these tests do not run a GPU model."""

import subprocess
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scene_reconstruction.sam3_dense_backend import (
    UPSTREAM_COMMIT, Sam3DenseBackend, discover_native, track_native, verify_native_revision,
)


class FakeImageProcessor:
    def __init__(self):
        self.images = []

    def set_image(self, image):
        assert isinstance(image, Image.Image)
        self.images.append(image)
        return {"width": image.width, "height": image.height}


class FakeImageModel:
    def __init__(self):
        self.calls = []

    def predict_inst(self, state, **kwargs):
        self.calls.append(kwargs)
        points = kwargs["point_coords"]
        batch = len(points)
        logits = np.full((batch, 3, state["height"], state["width"]), -2.0)
        for index, point in enumerate(points[:, 0]):
            x, y = map(int, point)
            logits[index, :, y, x] = 2.0
            # One near-threshold pixel reduces alternative 1's stability.
            logits[index, 1, 0, 0] = 0.5
        quality = np.tile(np.array([0.9, 0.8, 0.7]), (batch, 1))
        if batch == 1:
            logits, quality = logits[0], quality[0]
        return logits, quality, None


class FakeVideoPredictor:
    def __init__(self, frame_count=5, omit_frame=None, missing_cache=False, bad_grid=False):
        self.frame_count = frame_count
        self.omit_frame = omit_frame
        self.missing_cache = missing_cache
        self.bad_grid = bad_grid
        self.requests = []
        self.stream_requests = []
        self.closed = False
        self.seeds = {}
        self.state = {"cached_frame_outputs": {}, "tracker_metadata": {
            "obj_id_to_tracker_score_frame_wise": {},
        }}

    def _get_session(self, session_id):
        assert session_id == "session"
        return {"state": self.state}

    def _cache(self, index):
        masks = {key: value[None].copy() for key, value in self.seeds.items()}
        if self.bad_grid:
            masks = {key: np.zeros((1, 2, 2), dtype=bool) for key in masks}
        self.state["cached_frame_outputs"][index] = masks
        self.state["tracker_metadata"]["obj_id_to_tracker_score_frame_wise"][index] = {
            key: 0.75 for key in masks
        }

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": "session"}
        if request["type"] == "close_session":
            self.closed = True
            return {}
        assert request["type"] == "add_mask"
        assert "object_id" not in request
        self.seeds[request["obj_id"]] = request["mask"]
        self._cache(request["frame_index"])
        if self.missing_cache:
            self.state["cached_frame_outputs"].clear()
        return {"frame_index": request["frame_index"], "outputs": {}}

    def handle_stream_request(self, request):
        self.stream_requests.append(request)
        anchor = request["start_frame_index"]
        distance = request["max_frame_num_to_track"]
        if request["propagation_direction"] == "forward":
            indices = range(anchor, min(self.frame_count, anchor + distance + 1))
        else:
            # Native backwards may omit the anchor; adapter must still retain it.
            indices = range(anchor - 1, max(-1, anchor - distance - 1), -1)
        for index in indices:
            if index == self.omit_frame:
                continue
            assert not self.state["cached_frame_outputs"], "Adapter should evict full-resolution masks after copying"
            self._cache(index)
            # Public postprocessing clips overlaps; fake output is intentionally
            # empty so only reading the independent cache can pass the test.
            yield {"frame_index": index, "outputs": {"out_binary_masks": []}}


def seeds(anchor=2):
    outer = np.zeros((4, 6), dtype=bool)
    inner = np.zeros_like(outer)
    outer[1:4, 1:5] = True
    inner[2, 2] = True
    return [
        {"object_id": 7, "frame_index": anchor, "mask": outer},
        {"object_id": 18, "frame_index": anchor, "mask": inner},
    ]


class DiscoveryContracts(unittest.TestCase):
    def test_independent_prompt_batch_and_reused_embedding(self):
        processor, model = FakeImageProcessor(), FakeImageModel()
        points = np.array([[1.5, 2.5], [4.2, 3.1], [2.2, 1.2]])
        proposals = list(discover_native(model, processor, np.zeros((4, 6, 3), np.uint8), points, batch_size=2))
        self.assertEqual(len(proposals), 9)
        self.assertEqual(len(processor.images), 1)
        self.assertEqual(processor.images[0].size, (6, 4))
        self.assertEqual([call["point_coords"].shape for call in model.calls], [(2, 1, 2), (1, 1, 2)])
        np.testing.assert_allclose(model.calls[0]["point_coords"][:, 0], points[:2])
        for call in model.calls:
            np.testing.assert_array_equal(call["point_labels"], 1)
            self.assertTrue(call["normalize_coords"])
            self.assertTrue(call["return_logits"])
        self.assertEqual([item["point_index"] for item in proposals], [0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertEqual([item["alternative_index"] for item in proposals], [0, 1, 2] * 3)
        self.assertEqual(proposals[0]["mask"].shape, (4, 6))
        self.assertEqual(proposals[0]["mask"].dtype, np.bool_)
        self.assertEqual(proposals[0]["stability_score"], 1.0)
        self.assertEqual(proposals[1]["stability_score"], 0.5)
        self.assertAlmostEqual(proposals[1]["predicted_iou"], 0.8)
        np.testing.assert_allclose(proposals[0]["seed_xy"], points[0])

    def test_coordinates_must_use_supplied_crop_grid(self):
        for point in [[6, 1], [1, 4], [-1, 1], [float("nan"), 1]]:
            with self.subTest(point=point), self.assertRaises(ValueError):
                list(discover_native(FakeImageModel(), FakeImageProcessor(), np.zeros((4, 6, 3), np.uint8), [point]))

    def test_empty_points_skip_embedding_and_bad_image_rejected(self):
        processor = FakeImageProcessor()
        self.assertEqual(list(discover_native(FakeImageModel(), processor, np.zeros((4, 6, 3), np.uint8), np.empty((0, 2)))), [])
        self.assertEqual(processor.images, [])
        with self.assertRaises(ValueError):
            list(discover_native(FakeImageModel(), processor, np.zeros((4, 6, 3)), [[1, 1]]))

    def test_wrong_mask_grid_fails_instead_of_resizing_silently(self):
        class BadModel:
            def predict_inst(self, *args, **kwargs):
                return np.zeros((3, 6, 4)), np.ones(3), None
        with self.assertRaisesRegex(RuntimeError, "wrong mask grid"):
            list(discover_native(BadModel(), FakeImageProcessor(), np.zeros((4, 6, 3), np.uint8), [[1, 1]]))


class TrackingContracts(unittest.TestCase):
    def test_full_sequence_keeps_overlapping_tracks_and_exact_seed(self):
        predictor = FakeVideoPredictor()
        output = list(track_native(predictor, "/tmp/frames", 5, seeds(), 4, 6))
        self.assertEqual([row["frame_index"] for row in output], [2, 3, 4, 1, 0])
        for row in output:
            self.assertTrue(row["masks"][7][2, 2])
            self.assertTrue(row["masks"][18][2, 2])
        for seed in seeds():
            np.testing.assert_array_equal(output[0]["masks"][seed["object_id"]], seed["mask"])
        self.assertEqual(output[0]["scores"], {7: None, 18: None})
        self.assertEqual(output[1]["scores"], {7: 0.75, 18: 0.75})
        self.assertEqual([call["propagation_direction"] for call in predictor.stream_requests], ["forward", "backward"])
        self.assertTrue(all(call["force_tracker_propagation"] for call in predictor.stream_requests))
        self.assertTrue(all(not call["evict_cached_frame_outputs"] for call in predictor.stream_requests))
        self.assertTrue(predictor.requests[0]["offload_video_to_cpu"])
        self.assertTrue(predictor.requests[0]["offload_state_to_cpu"])
        self.assertTrue(predictor.closed)

    def test_boundary_anchors_and_single_frame(self):
        for anchor, count, directions in [(0, 5, ["forward"]), (4, 5, ["backward"]), (0, 1, [])]:
            with self.subTest(anchor=anchor, count=count):
                predictor = FakeVideoPredictor(frame_count=count)
                output = list(track_native(predictor, "/tmp/frames", count, seeds(anchor), 4, 6))
                self.assertEqual({row["frame_index"] for row in output}, set(range(count)))
                self.assertEqual([call["propagation_direction"] for call in predictor.stream_requests], directions)
                self.assertTrue(predictor.closed)

    def test_generator_cancellation_closes_native_session(self):
        predictor = FakeVideoPredictor()
        stream = track_native(predictor, "/tmp/frames", 5, seeds(), 4, 6)
        next(stream)
        stream.close()
        self.assertTrue(predictor.closed)

    def test_missing_frame_fails_and_closes_session(self):
        predictor = FakeVideoPredictor(omit_frame=3)
        with self.assertRaisesRegex(RuntimeError, "omitted selected frames"):
            list(track_native(predictor, "/tmp/frames", 5, seeds(), 4, 6))
        self.assertTrue(predictor.closed)

    def test_missing_private_cache_never_falls_back_to_clipped_masks(self):
        predictor = FakeVideoPredictor(missing_cache=True)
        with self.assertRaisesRegex(RuntimeError, "cache contract"):
            list(track_native(predictor, "/tmp/frames", 5, seeds(), 4, 6))
        self.assertTrue(predictor.closed)

    def test_wrong_output_grid_fails_and_closes_session(self):
        predictor = FakeVideoPredictor(bad_grid=True)
        with self.assertRaisesRegex(RuntimeError, "invalid mask"):
            list(track_native(predictor, "/tmp/frames", 5, seeds(), 4, 6))
        self.assertTrue(predictor.closed)

    def test_missing_object_is_error_not_fabricated_empty_mask(self):
        class MissingObjectPredictor(FakeVideoPredictor):
            def handle_stream_request(self, request):
                for response in super().handle_stream_request(request):
                    self.state["cached_frame_outputs"][response["frame_index"]].pop(18)
                    yield response
        predictor = MissingObjectPredictor()
        with self.assertRaisesRegex(RuntimeError, "every seeded object"):
            list(track_native(predictor, "/tmp/frames", 5, seeds(), 4, 6))
        self.assertTrue(predictor.closed)

    def test_seed_ids_anchor_and_grid_validated_before_starting_session(self):
        cases = []
        mixed = seeds()
        mixed[1]["frame_index"] = 3
        cases.append(mixed)
        duplicate = seeds()
        duplicate[1]["object_id"] = 7
        cases.append(duplicate)
        for field, value in [("object_id", True), ("frame_index", 5), ("mask", np.ones((6, 4), bool)), ("mask", np.ones((4, 6), float))]:
            malformed = seeds()
            malformed[0][field] = value
            cases.append(malformed)
        for case in cases:
            predictor = FakeVideoPredictor()
            with self.assertRaises(ValueError):
                list(track_native(predictor, "/tmp/frames", 5, case, 4, 6))
            self.assertEqual(predictor.requests, [])

    def test_revision_is_explicitly_pinned(self):
        with patch("scene_reconstruction.sam3_dense_backend.subprocess.run", return_value=subprocess.CompletedProcess([], 0, UPSTREAM_COMMIT + "\n", "")):
            verify_native_revision("/tmp/vendor/sam3/sam3/__init__.py")
        for returncode, stdout in [(0, "wrong"), (128, "")]:
            with patch("scene_reconstruction.sam3_dense_backend.subprocess.run", return_value=subprocess.CompletedProcess([], returncode, stdout, "")):
                with self.assertRaisesRegex(RuntimeError, "requires native commit"):
                    verify_native_revision("/tmp/vendor/sam3/sam3/__init__.py")


class ModelConstructionContracts(unittest.TestCase):
    def test_selected_gpu_native_builder_literal_and_separate_lifetimes(self):
        current_devices, resets, image_builds, video_builds, shutdowns = [], [], [], [], []
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 4,
            set_device=current_devices.append,
            reset_peak_memory_stats=resets.append,
            empty_cache=lambda: None,
            get_device_properties=lambda index: types.SimpleNamespace(total_memory=48 * 1024 ** 3),
            get_device_name=lambda index: "Fake A6000",
            max_memory_allocated=lambda index: 1234,
            max_memory_reserved=lambda index: 5678,
        )
        torch = types.SimpleNamespace(cuda=cuda, __version__="2.7.1", version=types.SimpleNamespace(cuda="12.6"))

        def build_image(**kwargs):
            image_builds.append(kwargs)
            self.assertEqual(current_devices[-1], 2)
            # Deliberately mimic pinned _setup_device_and_mode exactly.
            return types.SimpleNamespace(device=f"cuda:{current_devices[-1]}" if kwargs["device"] == "cuda" else "cpu")

        def processor(model, device):
            self.assertEqual(model.device, device, "Pinned native builder silently ignores cuda:N")
            return types.SimpleNamespace(model=model, device=device)

        def build_video(**kwargs):
            video_builds.append(kwargs)
            return types.SimpleNamespace(shutdown=lambda: shutdowns.append(True))

        modules = {
            "sam3": types.SimpleNamespace(__file__="/tmp/vendor/sam3/sam3/__init__.py"),
            "torch": torch,
            "sam3.model_builder": types.SimpleNamespace(build_sam3_image_model=build_image, build_sam3_video_predictor=build_video),
            "sam3.model.sam3_image_processor": types.SimpleNamespace(Sam3Processor=processor),
        }
        with patch.dict("sys.modules", modules), patch("scene_reconstruction.sam3_dense_backend.verify_native_revision") as verify:
            backend = Sam3DenseBackend(checkpoint="/tmp/checkpoint.pt", gpu_index=2)
            backend.load_discovery()
            backend.load_discovery()
            self.assertEqual(len(image_builds), 1)
            self.assertEqual(image_builds[0], {
                "checkpoint_path": "/tmp/checkpoint.pt", "device": "cuda",
                "enable_inst_interactivity": True, "compile": False,
            })
            backend.load_tracking()
            self.assertIsNone(backend._image_model)
            self.assertIsNone(backend._image_processor)
            self.assertEqual(video_builds, [{
                "checkpoint_path": "/tmp/checkpoint.pt", "gpus_to_use": [2],
                "compile": False, "async_loading_frames": False,
            }])
            backend.close()
            self.assertEqual(shutdowns, [True])
            self.assertEqual(current_devices, [2])
            self.assertEqual(resets, [2])
            verify.assert_called_once()
            runtime = backend.runtime_info()
            self.assertEqual(runtime["peak_allocated_bytes"], 1234)
            self.assertEqual(runtime["peak_reserved_bytes"], 5678)
            self.assertEqual(runtime["visible_gpu_index"], 2)


if __name__ == "__main__":
    unittest.main()
