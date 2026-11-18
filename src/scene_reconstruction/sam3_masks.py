"""Stream SAM 3 concept masks onto the canonical source-frame pixel grid.

Heavy imports stay in ``run`` so frame extraction/comparison need no CUDA stack.
The adapter intentionally runs each text concept in a separate video session:
SAM 3's text prompt is session-wide, not a per-object class label.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .common import sha256


UPSTREAM_COMMIT = "31cb7e624e4fe95301b1be1d4e809111593d2e45"
UPSTREAM_URL = "https://github.com/facebookresearch/sam3"


def load_prompts(path: Path, frame_indices: set[int]) -> list[dict]:
    """Validate stable output names and global-frame text prompt anchors."""
    payload = json.loads(Path(path).read_text())
    objects = payload.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("Prompts JSON must contain a nonempty 'objects' list")
    names: set[str] = set()
    result = []
    for item in objects:
        if not isinstance(item, dict):
            raise ValueError("Each prompt must be an object with name and text")
        name, text = item.get("name"), item.get("text")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError("Prompt names must contain only letters, digits, '_' or '-'")
        if name.lower() in names:
            raise ValueError(f"Duplicate prompt name: {name}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Prompt {name!r} needs nonempty text")
        index = item.get("frame_index", min(frame_indices))
        if isinstance(index, bool) or not isinstance(index, int) or index not in frame_indices:
            raise ValueError(f"Prompt {name!r} frame_index is absent from the frame manifest")
        names.add(name.lower())
        result.append({"name": name, "text": text.strip(), "frame_index": index})
    return result


def iter_chunks(frames: list[dict], chunk_size: int):
    """Non-overlapping chunks; zero requests one continuous tracking session."""
    if chunk_size < 0:
        raise ValueError("chunk_size must be nonnegative")
    size = chunk_size or len(frames)
    if size:
        for start in range(0, len(frames), size):
            yield frames[start : start + size]


def propagation_direction(anchor: int, frame_count: int) -> str:
    """Avoid upstream's fetch-only second pass after a boundary-started pass.

    At the last frame, backward propagation omits the anchor itself. Its
    add_prompt result is saved separately by the caller.
    """
    if not 0 <= anchor < frame_count:
        raise ValueError("The prompt anchor must lie within the chunk")
    if anchor == 0:
        return "forward"
    if anchor == frame_count - 1:
        return "backward"
    return "both"


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_output(outputs: dict, height: int, width: int, threshold: float):
    """Validate the upstream contract and apply a reproducible score filter.

    Masks from the official video API are already binary, original-resolution
    arrays. Reject a different grid rather than silently stretching geometry.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    masks = _numpy(outputs["out_binary_masks"])
    ids = _numpy(outputs["out_obj_ids"]).reshape(-1)
    scores = _numpy(outputs["out_probs"]).reshape(-1)
    boxes = _numpy(outputs["out_boxes_xywh"])
    n = len(ids)
    if masks.shape != (n, height, width):
        raise ValueError(f"SAM3 mask shape {masks.shape} != {(n, height, width)}")
    if scores.shape != (n,) or boxes.shape != (n, 4):
        raise ValueError("SAM3 IDs, scores, and boxes disagree in length")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(boxes)):
        raise ValueError("SAM3 returned non-finite scores or boxes")
    if len(set(ids.tolist())) != n:
        raise ValueError("SAM3 returned duplicate instance IDs within a frame")
    if masks.dtype != np.bool_ and not np.all((masks == 0) | (masks == 1)):
        raise ValueError("Expected binary masks from the SAM3 video API")
    keep = scores >= threshold
    masks = masks[keep].astype(bool, copy=False)
    return ids[keep].astype(np.int64), masks, scores[keep], boxes[keep]


def _write_json(path: Path, payload: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _upstream_revision(module) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(module.__file__).resolve().parent), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _save_frame(output: Path, prompt: dict, chunk_start: int, frame: dict,
                outputs: dict, height: int, width: int, threshold: float,
                save_instances: bool) -> dict:
    ids, masks, scores, boxes = normalize_output(outputs, height, width, threshold)
    union = np.any(masks, axis=0) if len(masks) else np.zeros((height, width), dtype=bool)
    mask_path = Path("masks") / prompt["name"] / f"{frame['index']:06d}.png"
    (output / mask_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(union.astype(np.uint8) * 255).save(output / mask_path)
    instances = []
    for obj_id, mask, score, box in zip(ids, masks, scores, boxes):
        instance = {
            "id": int(obj_id), "score": float(score),
            "box_xywh_normalized": box.tolist(), "area_pixels": int(mask.sum()),
        }
        if save_instances:
            path = (Path("instances") / prompt["name"] / f"chunk_{chunk_start:06d}"
                    / f"id_{int(obj_id):06d}" / f"{frame['index']:06d}.png")
            (output / path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(output / path)
            instance["mask"] = path.as_posix()
        instances.append(instance)
    return {
        "index": frame["index"], "time_seconds": frame["time_seconds"],
        "pts_seconds": frame["pts_seconds"], "object": prompt["name"],
        "chunk_start_index": chunk_start, "mask": mask_path.as_posix(),
        "area_pixels": int(union.sum()), "instances": instances,
    }


def run(args) -> None:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    frames = manifest["frames"]
    if not frames:
        raise ValueError("The frame manifest is empty")
    width, height = int(manifest["width"]), int(manifest["height"])
    indices = [frame["index"] for frame in frames]
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices):
        raise ValueError("Frame indices must be nonnegative integers")
    if len(set(indices)) != len(indices) or indices != sorted(indices):
        raise ValueError("Frame indices must be unique and ordered")
    chunk_size = int(args.chunk_size)
    if chunk_size < 0:
        raise ValueError("chunk_size must be nonnegative")
    threshold = float(getattr(args, "threshold", 0.5))
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    save_instances = bool(getattr(args, "save_instances", True))
    prompts = load_prompts(Path(args.prompts), set(indices))
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint is not None:
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint does not exist: {checkpoint}")
    paths = {}
    for frame in frames:
        path = (manifest_path.parent / frame["file"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            if image.size != (width, height):
                raise ValueError(f"Frame {path} has size {image.size}, expected {(width, height)}")
        paths[frame["index"]] = path
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty to avoid mixing runs: {output}")

    try:
        import torch
        import sam3
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as error:
        raise RuntimeError("Install the SAM3 environment first; see docs/sam3.md") from error
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The official SAM3 video predictor requires CUDA; run this stage on the A6000")
    gpu_index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(gpu_index)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1, "status": "running", "model": "facebook/sam3",
        "upstream_url": UPSTREAM_URL, "verified_upstream_commit": UPSTREAM_COMMIT,
        "installed_upstream_commit": _upstream_revision(sam3),
        "checkpoint": str(checkpoint) if checkpoint else "facebook/sam3:sam3.pt (Hugging Face cache)",
        "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
        "manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "width": width, "height": height, "frame_count": len(frames),
        "objects": prompts, "chunk_size": chunk_size, "score_threshold": threshold,
        "save_instances": save_instances, "precision": "bfloat16 (upstream video predictor)",
        "offload_requests": {"video_to_cpu": True, "state_to_cpu": True},
        "memory_note": "Upstream may copy the full preprocessed image batch back to CUDA; chunking bounds sequence length",
        "device": str(torch.device("cuda", gpu_index)), "torch_version": torch.__version__,
        "pixel_grid": "Source-frame pixels, top-left origin; no adapter resize/crop/rotation",
        "mask_encoding": "Grayscale uint8 PNG: background=0, foreground=255",
        "mask_semantics": "Union of all detected instances of each text concept; concepts may overlap",
        "instance_identity": "IDs are local to (object name, chunk_start_index); no cross-chunk identity association",
        "anchor_policy": "Use configured global frame_index in its chunk; use first frame in every other chunk",
        "completed_sessions": 0,
    }
    _write_json(output / "metadata.json", metadata)
    predictor = None
    try:
        predictor = build_sam3_video_predictor(
            checkpoint_path=str(checkpoint) if checkpoint else None,
            gpus_to_use=[gpu_index], compile=False, async_loading_frames=False,
        )
        with (output / "frames.jsonl").open("w") as index_file:
            for chunk in iter_chunks(frames, chunk_size):
                chunk_start = chunk[0]["index"]
                with tempfile.TemporaryDirectory(prefix="sam3_frames_") as temporary:
                    # Numeric names preserve manifest order even for arbitrary source filenames.
                    for local_index, frame in enumerate(chunk):
                        source = paths[frame["index"]]
                        link = Path(temporary) / f"{local_index:06d}{source.suffix.lower()}"
                        link.symlink_to(source)
                    for prompt in prompts:
                        print(f"SAM3 {prompt['name']}: frames {chunk_start}..{chunk[-1]['index']}", flush=True)
                        anchor = next((i for i, frame in enumerate(chunk)
                                       if frame["index"] == prompt["frame_index"]), 0)
                        session_id = None
                        records = {}
                        try:
                            session = predictor.handle_request(request={
                                "type": "start_session", "resource_path": temporary,
                                "offload_video_to_cpu": True, "offload_state_to_cpu": True,
                            })
                            session_id = session["session_id"]
                            seed = predictor.handle_request(request={
                                "type": "add_prompt", "session_id": session_id,
                                "frame_index": anchor, "text": prompt["text"],
                                "output_prob_thresh": threshold,
                            })
                            for response in predictor.handle_stream_request(request={
                                "type": "propagate_in_video", "session_id": session_id,
                                "propagation_direction": propagation_direction(anchor, len(chunk)),
                                "start_frame_index": anchor, "output_prob_thresh": threshold,
                            }):
                                local_index = int(response["frame_index"])
                                if not 0 <= local_index < len(chunk):
                                    raise RuntimeError(f"SAM3 returned invalid frame index {local_index}")
                                # Both directions may emit their common anchor. Keep forward result.
                                if local_index in records:
                                    continue
                                records[local_index] = _save_frame(
                                    output, prompt, chunk_start, chunk[local_index],
                                    response["outputs"], height, width, threshold, save_instances,
                                )
                            if anchor not in records:
                                records[anchor] = _save_frame(
                                    output, prompt, chunk_start, chunk[anchor], seed["outputs"],
                                    height, width, threshold, save_instances,
                                )
                            missing = sorted(set(range(len(chunk))) - records.keys())
                            if missing:
                                raise RuntimeError(f"SAM3 omitted frames {[chunk[i]['index'] for i in missing]}")
                            for i in range(len(chunk)):
                                index_file.write(json.dumps(records[i], allow_nan=False) + "\n")
                            index_file.flush()
                            metadata["completed_sessions"] += 1
                            _write_json(output / "metadata.json", metadata)
                        finally:
                            if session_id is not None:
                                predictor.handle_request(request={"type": "close_session", "session_id": session_id})
        metadata["status"] = "complete"
        metadata["mask_count"] = len(frames) * len(prompts)
        _write_json(output / "metadata.json", metadata)
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        _write_json(output / "metadata.json", metadata)
        raise
    finally:
        if predictor is not None:
            predictor.shutdown()
