"""Pinned native SAM3 discovery/tracking adapter for the dense-mask pilot.

Discovery points are independent one-positive-click trials, batched on one image
embedding. Tracking batches share an anchor but keep independent, overlapping
masklets. SAM3's public video output makes masks mutually exclusive, so this
adapter deliberately copies its *pre-postprocess* cache at the pinned revision.
This private contract is checked and must not silently fall back to clipped masks.
Heavy dependencies are imported only when a model is loaded.
"""

from __future__ import annotations

import gc
import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


UPSTREAM_COMMIT = "31cb7e624e4fe95301b1be1d4e809111593d2e45"


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def discover_native(
    model, processor, image_rgb, points_xy, batch_size=16,
    mask_threshold=0.0, stability_offset=1.0,
) -> Iterator[dict]:
    """Yield all three alternatives per click, without deduplication or filtering.

    ``points_xy`` contains pixel coordinates in the supplied RGB image, including
    when that image is a crop. The caller owns any offset back to the full image.
    ``predicted_iou`` is a model quality estimate; stability is the intersection /
    union of masks thresholded at ``mask_threshold +/- stability_offset``.
    """
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Discovery image must be HxWx3 uint8 RGB")
    height, width = image.shape[:2]
    if not height or not width:
        raise ValueError("Discovery image must not be empty")
    batch_size = _positive_integer(batch_size, "batch_size")
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("points_xy must be a finite Nx2 array of pixel XY coordinates")
    if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
        raise ValueError("Discovery points must be inside the supplied image")
    if not np.isfinite(mask_threshold) or not np.isfinite(stability_offset) or stability_offset <= 0:
        raise ValueError("Mask threshold must be finite and stability_offset positive")
    if not len(points):
        return

    # Passing HWC numpy directly to this pinned processor misidentifies H/W;
    # its PIL branch explicitly reads image.size and is the safe public path.
    state = processor.set_image(Image.fromarray(image))
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        logits, quality, _ = model.predict_inst(
            state,
            point_coords=batch[:, None, :],
            point_labels=np.ones((len(batch), 1), dtype=np.int32),
            normalize_coords=True,
            multimask_output=True,
            return_logits=True,
        )
        logits, quality = _numpy(logits), _numpy(quality)
        # The pinned predictor squeezes batch dimension only when its size is 1.
        if len(batch) == 1:
            if logits.ndim == 3:
                logits = logits[None]
            if quality.ndim == 1:
                quality = quality[None]
        if logits.ndim != 4 or logits.shape[0] != len(batch) or logits.shape[2:] != (height, width):
            raise RuntimeError(f"Native discovery returned wrong mask grid: {logits.shape}, expected BxCx{height}x{width}")
        if quality.shape != logits.shape[:2] or not np.isfinite(quality).all():
            raise RuntimeError("Native discovery returned invalid quality scores")
        for batch_index, point in enumerate(batch):
            for alternative, logit in enumerate(logits[batch_index]):
                if not np.isfinite(logit).all():
                    raise RuntimeError("Native discovery returned non-finite mask logits")
                high_area = np.count_nonzero(logit > mask_threshold + stability_offset)
                low_area = np.count_nonzero(logit > mask_threshold - stability_offset)
                yield {
                    "point_index": start + batch_index,
                    "alternative_index": alternative,
                    "seed_xy": [float(point[0]), float(point[1])],
                    "mask": logit > mask_threshold,
                    "predicted_iou": float(quality[batch_index, alternative]),
                    "stability_score": float(high_area / low_area) if low_area else 0.0,
                }


def _copy_cached_masks(predictor, session_id, frame_index, object_ids, height, width):
    """Read independent masks BEFORE public cross-object non-overlap clipping."""
    try:
        state = predictor._get_session(session_id)["state"]
        cache = state["cached_frame_outputs"]
        cached = cache[frame_index]
        metadata = state["tracker_metadata"]
    except (AttributeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "Pinned SAM3 pre-postprocess cache contract is unavailable; refusing "
            "to substitute clipped public masks. Re-run setup_a6000.sh sam."
        ) from exc
    if not isinstance(cached, dict) or set(cached) != set(object_ids):
        raise RuntimeError("Native tracking cache must contain every seeded object ID, including explicit empty masks")
    scores_at_frame = metadata.get("obj_id_to_tracker_score_frame_wise", {}).get(frame_index, {})
    masks, scores = {}, {}
    for object_id in object_ids:
        mask = _numpy(cached[object_id])
        if mask.shape == (1, height, width):
            mask = mask[0]
        if mask.shape != (height, width) or mask.dtype != np.bool_:
            raise RuntimeError(f"Native tracking returned invalid mask for {object_id}: {mask.shape}/{mask.dtype}")
        masks[object_id] = mask.copy()
        score = scores_at_frame.get(object_id)
        if score is not None:
            score = float(_numpy(score).item())
            if not np.isfinite(score):
                raise RuntimeError("Native tracking returned non-finite tracker score")
        scores[object_id] = score
    # Public base-predictor eviction runs BEFORE it yields to us. Evict here,
    # after the CPU copy instead, to bound the full-resolution GPU mask cache.
    cache.pop(frame_index, None)
    return {"frame_index": frame_index, "masks": masks, "scores": scores}


def track_native(predictor, resource_path, frame_count, seeds, height, width) -> Iterator[dict]:
    """Track a same-anchor batch across the complete selected-frame sequence.

    Each seed has ``object_id``, local ``frame_index``, and bool ``mask`` HxW.
    Caller IDs are retained. Output order is seed, forward, backward; every frame
    occurs once. Native empty masks are retained; missing IDs are errors rather
    than fabricated absence. Scores are raw tracker values,
    not calibrated quality, and are None at the exact supplied seed frame.
    """
    frame_count = _positive_integer(frame_count, "frame_count")
    height, width = _positive_integer(height, "height"), _positive_integer(width, "width")
    seeds = list(seeds)
    if not seeds:
        raise ValueError("Tracking requires at least one seed")
    checked = []
    for seed in seeds:
        object_id = _positive_integer(seed["object_id"], "object_id")
        frame_index = seed["frame_index"]
        if isinstance(frame_index, (bool, np.bool_)) or not isinstance(frame_index, (int, np.integer)) or not 0 <= frame_index < frame_count:
            raise ValueError("Seed frame_index must be a local integer inside the sequence")
        mask = np.asarray(seed["mask"])
        if mask.dtype != np.bool_ or mask.shape != (height, width) or not mask.any():
            raise ValueError("Seed mask must be a non-empty bool array in the original frame grid")
        checked.append({"object_id": object_id, "frame_index": int(frame_index), "mask": mask.copy()})
    object_ids = [seed["object_id"] for seed in checked]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("Object IDs must be unique within a tracking batch")
    anchor = checked[0]["frame_index"]
    if any(seed["frame_index"] != anchor for seed in checked):
        raise ValueError("A tracking batch must have a single shared anchor frame")
    if not callable(getattr(predictor, "_get_session", None)):
        raise RuntimeError("Pinned SAM3 predictor must expose its pre-postprocess session cache")

    response = predictor.handle_request({
        "type": "start_session", "resource_path": str(resource_path),
        "offload_video_to_cpu": True, "offload_state_to_cpu": True,
    })
    session_id = response["session_id"]
    try:
        for seed in checked:
            predictor.handle_request({
                "type": "add_mask", "session_id": session_id,
                "obj_id": seed["object_id"], "frame_index": seed["frame_index"],
                "mask": seed["mask"],
            })
        # Check the private cache contract even for a one-frame video. Keep the
        # authoritative supplied seed masks rather than SAM's resized round trip.
        _copy_cached_masks(predictor, session_id, anchor, object_ids, height, width)
        seen = {anchor}
        yield {
            "frame_index": anchor,
            "masks": {seed["object_id"]: seed["mask"].copy() for seed in checked},
            "scores": {object_id: None for object_id in object_ids},
        }
        directions = []
        if anchor < frame_count - 1:
            directions.append(("forward", frame_count - 1 - anchor))
        if anchor > 0:
            directions.append(("backward", anchor))
        for direction, distance in directions:
            stream = predictor.handle_stream_request({
                "type": "propagate_in_video", "session_id": session_id,
                "propagation_direction": direction, "start_frame_index": anchor,
                "max_frame_num_to_track": distance,
                "force_tracker_propagation": True,
                "evict_cached_frame_outputs": False,
            })
            try:
                for response in stream:
                    frame_index = response["frame_index"]
                    if isinstance(frame_index, (bool, np.bool_)) or not isinstance(frame_index, (int, np.integer)) or not 0 <= frame_index < frame_count:
                        raise RuntimeError("Native tracker emitted an invalid frame index")
                    frame_index = int(frame_index)
                    result = _copy_cached_masks(predictor, session_id, frame_index, object_ids, height, width)
                    if frame_index == anchor:
                        continue
                    if frame_index in seen:
                        raise RuntimeError(f"Native tracker emitted duplicate frame {frame_index}")
                    if (direction == "forward" and frame_index < anchor) or (direction == "backward" and frame_index > anchor):
                        raise RuntimeError("Native tracker emitted a frame outside the requested direction")
                    seen.add(frame_index)
                    yield result
            finally:
                if callable(getattr(stream, "close", None)):
                    stream.close()
        missing = set(range(frame_count)) - seen
        if missing:
            raise RuntimeError(f"Native tracker omitted selected frames: {sorted(missing)}")
    finally:
        predictor.handle_request({"type": "close_session", "session_id": session_id})


def verify_native_revision(package_file):
    """Private cache semantics require the editable native checkout we pinned."""
    result = subprocess.run(
        ["git", "-C", str(Path(package_file).resolve().parent), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode or result.stdout.strip() != UPSTREAM_COMMIT:
        actual = result.stdout.strip() or "unknown (not an editable git checkout)"
        raise RuntimeError(
            f"Dense SAM3 backend requires native commit {UPSTREAM_COMMIT}; found {actual}. "
            "Use the repository's setup_a6000.sh sam environment."
        )


class Sam3DenseBackend:
    """Load discovery and tracking models in separate VRAM lifetimes."""

    def __init__(self, checkpoint=None, gpu_index=0):
        if isinstance(gpu_index, bool) or not isinstance(gpu_index, int) or gpu_index < 0:
            raise ValueError("gpu_index must be a non-negative visible CUDA index")
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.gpu_index = gpu_index
        self._torch = None
        self._image_model = None
        self._image_processor = None
        self._video_predictor = None

    def _initialize(self):
        if self._torch is not None:
            return
        try:
            import sam3
            import torch
        except ImportError as exc:
            raise RuntimeError("Run this command in the native .venv-sam environment") from exc
        verify_native_revision(sam3.__file__)
        if not torch.cuda.is_available() or self.gpu_index >= torch.cuda.device_count():
            raise RuntimeError(f"Visible CUDA device {self.gpu_index} is unavailable")
        torch.cuda.set_device(self.gpu_index)
        torch.cuda.reset_peak_memory_stats(self.gpu_index)
        self._torch = torch

    def load_discovery(self):
        if self._image_model is not None:
            return self
        self._initialize()
        self.release_tracking()
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        device = f"cuda:{self.gpu_index}"
        # At the pinned revision, _setup_device_and_mode only moves a model
        # for the literal "cuda"; "cuda:N" silently leaves it on CPU. The
        # current device was selected in _initialize before building.
        self._image_model = build_sam3_image_model(
            checkpoint_path=self.checkpoint, device="cuda",
            enable_inst_interactivity=True, compile=False,
        )
        self._image_processor = Sam3Processor(self._image_model, device=device)
        return self

    def discover(self, image_rgb, points_xy, batch_size=16):
        self.load_discovery()
        with self._torch.inference_mode(), self._torch.autocast(device_type="cuda", dtype=self._torch.bfloat16):
            yield from discover_native(self._image_model, self._image_processor, image_rgb, points_xy, batch_size)

    def release_discovery(self):
        self._image_processor = None
        self._image_model = None
        self._empty_cache()

    def load_tracking(self):
        if self._video_predictor is not None:
            return self
        self._initialize()
        self.release_discovery()
        from sam3.model_builder import build_sam3_video_predictor
        self._video_predictor = build_sam3_video_predictor(
            checkpoint_path=self.checkpoint, gpus_to_use=[self.gpu_index],
            compile=False, async_loading_frames=False,
        )
        return self

    def track(self, resource_path, frame_count, seeds, height, width):
        self.load_tracking()
        with self._torch.inference_mode(), self._torch.autocast(device_type="cuda", dtype=self._torch.bfloat16):
            yield from track_native(self._video_predictor, resource_path, frame_count, seeds, height, width)

    def release_tracking(self):
        if self._video_predictor is not None:
            try:
                self._video_predictor.shutdown()
            finally:
                self._video_predictor = None
        self._empty_cache()

    def _empty_cache(self):
        gc.collect()
        if self._torch is not None:
            self._torch.cuda.empty_cache()

    def close(self):
        try:
            self.release_tracking()
        finally:
            self.release_discovery()

    def runtime_info(self):
        """Record measured runtime provenance without initializing a new model."""
        info = {
            "upstream_commit": UPSTREAM_COMMIT,
            "checkpoint": self.checkpoint,
            "visible_gpu_index": self.gpu_index,
            "overlap_policy": "Independent pre-postprocess native cached masks; copied to CPU then evicted",
            "score_semantics": "Raw native tracker scores; supplied seed scores are null",
        }
        if self._torch is not None:
            properties = self._torch.cuda.get_device_properties(self.gpu_index)
            info.update({
                "torch_version": self._torch.__version__,
                "torch_cuda_version": self._torch.version.cuda,
                "device_name": self._torch.cuda.get_device_name(self.gpu_index),
                "device_total_memory_bytes": int(properties.total_memory),
                "peak_allocated_bytes": int(self._torch.cuda.max_memory_allocated(self.gpu_index)),
                "peak_reserved_bytes": int(self._torch.cuda.max_memory_reserved(self.gpu_index)),
            })
        return info
