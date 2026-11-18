"""Joint VGGT-Ω reconstruction with frame and pixel correspondence preserved.

The official model is imported only by :func:`run`, inside its own environment.
No upstream inference output is treated as a metric measurement.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
from PIL import Image


UPSTREAM_URL = "https://github.com/facebookresearch/vggt-omega"
TESTED_REVISION = "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6"


def select_frames(frames: list[dict], stride: int, max_frames: int) -> list[dict]:
    """Sample the full duration; a cap must not truncate the end of the video."""
    if stride < 1:
        raise ValueError("stride must be positive")
    if max_frames < 0 or max_frames == 1:
        raise ValueError("max_frames must be 0 (unlimited) or at least 2")
    if len(frames) < 2:
        raise ValueError("VGGT reconstruction needs at least two video frames")
    selected = frames[::stride]
    if selected[-1]["index"] != frames[-1]["index"]:
        selected.append(frames[-1])
    if max_frames and len(selected) > max_frames:
        positions = np.rint(np.linspace(0, len(selected) - 1, max_frames)).astype(int)
        selected = [selected[i] for i in positions]
    return selected


def preprocessing_geometry(width: int, height: int, model_width: int, model_height: int) -> dict:
    """Describe upstream aspect-ratio crop + PIL resize for equal-size frames.

    Integer coordinates designate pixel centers. PIL's half-pixel resize mapping
    is used, including the +/-0.5 offsets, so the inverse maps model pixels back
    into the canonical decoded image. Upstream never pads equal-size frames.
    """
    left = top = 0
    crop_width, crop_height = width, height
    if height / width < 0.5:
        crop_width = min(width, max(1, round(height / 0.5)))
        left = max((width - crop_width) // 2, 0)
    elif height / width > 2.0:
        crop_height = min(height, max(1, round(width * 2.0)))
        top = max((height - crop_height) // 2, 0)
    sx, sy = model_width / crop_width, model_height / crop_height
    affine = np.array([
        [sx, 0.0, sx * (0.5 - left) - 0.5],
        [0.0, sy, sy * (0.5 - top) - 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return {
        "canonical_size_wh": [width, height],
        "crop_xyxy": [left, top, left + crop_width, top + crop_height],
        "model_size_wh": [model_width, model_height],
        "canonical_to_model_pixel_affine": affine.tolist(),
        "model_to_canonical_pixel_affine": np.linalg.inv(affine).tolist(),
        "pixel_coordinates": "Integer coordinates are pixel centers; PIL half-pixel resize convention.",
        "interpolation": "PIL bicubic",
        "padding": [0, 0, 0, 0],
    }


def unproject_points(depth: np.ndarray, intrinsics: np.ndarray, camera_to_world: np.ndarray,
                     ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Unproject selected optical-axis depths into the common predicted world."""
    pixels = np.stack((xs, ys, np.ones_like(xs)), axis=-1)
    rays = pixels @ np.linalg.inv(intrinsics).T
    camera_points = rays * depth[ys, xs, None]
    return camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _select_cuda_device(torch_module, requested):
    """Resolve bare 'cuda' before set_device, which requires a concrete index."""
    device = torch_module.device(requested)
    if device.type != "cuda":
        raise ValueError("This official VGGT-Ω adapter requires a CUDA GPU")
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this command in the A6000 environment")
    index = device.index if device.index is not None else torch_module.cuda.current_device()
    torch_module.cuda.set_device(index)
    return torch_module.device("cuda", index)


def _write_ply(path: Path, point_chunks: list[np.ndarray], color_chunks: list[np.ndarray],
               confidence_chunks: list[np.ndarray], frame_chunks: list[np.ndarray]) -> int:
    count = sum(len(chunk) for chunk in point_chunks)
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("confidence", "<f4"), ("frame_index", "<i4"),
    ])
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment VGGT-Omega predicted common world; scale is arbitrary\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float confidence\nproperty int frame_index\nend_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        for points, colors, confidence, frame_indices in zip(
            point_chunks, color_chunks, confidence_chunks, frame_chunks
        ):
            records = np.empty(len(points), dtype=dtype)
            for i, axis in enumerate(("x", "y", "z")):
                records[axis] = points[:, i]
            for i, channel in enumerate(("red", "green", "blue")):
                records[channel] = colors[:, i]
            records["confidence"] = confidence
            records["frame_index"] = frame_indices
            stream.write(records.tobytes())
    return count


def run(args) -> Path:
    """Run the official model once over a sampled, full-duration frame set."""
    manifest_path = Path(args.manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve()
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required: obtain vggt_omega_1b_512.pt from facebook/VGGT-Omega")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VGGT-Ω checkpoint does not exist: {checkpoint}")
    if not (repo / "vggt_omega" / "models").is_dir():
        raise FileNotFoundError(f"--repo must point to the official VGGT-Ω source checkout: {repo}")
    image_resolution = int(getattr(args, "image_resolution", 512))
    mode = getattr(args, "preprocess_mode", "max_size")
    point_stride = int(getattr(args, "point_stride", 4))
    percentile = float(getattr(args, "confidence_percentile", 50.0))
    max_points = int(getattr(args, "max_points", 1_000_000))
    if image_resolution < 16 or image_resolution % 16:
        raise ValueError("image_resolution must be a positive multiple of 16")
    if mode not in {"balanced", "max_size"}:
        raise ValueError("preprocess_mode must be balanced or max_size")
    if point_stride < 1 or not 0 <= percentile <= 100 or max_points < 1:
        raise ValueError("point_stride and max_points must be positive; confidence percentile must be 0..100")
    if not str(args.device).startswith("cuda"):
        raise ValueError("This official VGGT-Ω adapter requires a CUDA GPU")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty to prevent mixing reconstructions: {output}")
    manifest = json.loads(manifest_path.read_text())
    selected = select_frames(manifest["frames"], int(args.stride), int(args.max_frames))
    width, height = int(manifest["width"]), int(manifest["height"])
    image_paths = [manifest_path.parent / frame["file"] for frame in selected]
    for path in image_paths:
        with Image.open(path) as image:
            if image.size != (width, height):
                raise ValueError(f"Mixed image sizes would alter pixel geometry: {path}: {image.size}")
    revision = _revision(repo)
    if revision != TESTED_REVISION:
        warnings.warn(f"VGGT-Ω source revision {revision!r} differs from tested {TESTED_REVISION}", stacklevel=2)

    sys.path.insert(0, str(repo))
    try:
        import torch
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera
    except ImportError as error:
        raise RuntimeError("Install the VGGT-Ω environment first; see docs/vggt_omega.md") from error
    device = _select_cuda_device(torch, args.device)
    precision = getattr(args, "precision", "auto")
    if precision == "auto":
        precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("precision must be auto, bf16, fp16, or fp32")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA GPU does not support bf16; use --precision fp16")

    print(f"VGGT-Ω: {len(selected)} frames spanning {selected[0]['time_seconds']:.3f}–"
          f"{selected[-1]['time_seconds']:.3f}s; {mode} {image_resolution}; {precision}", flush=True)
    # Disable the upstream auto choice so the requested aggregator precision is
    # respected. Its camera and depth heads explicitly retain FP32 computation.
    model = VGGTOmega(autocast=False).eval()
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model = model.to(device)
    images = load_and_preprocess_images(
        [str(path) for path in image_paths], mode=mode, image_resolution=image_resolution
    ).to(device)
    model_height, model_width = map(int, images.shape[-2:])
    geometry = preprocessing_geometry(width, height, model_width, model_height)
    amp = contextlib.nullcontext() if precision == "fp32" else torch.autocast(
        device_type="cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16
    )
    try:
        with torch.inference_mode(), amp:
            predictions = model(images)
        with torch.inference_mode():
            extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"].float(), (model_height, model_width))
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError(
            "VGGT-Ω exceeded GPU memory. Reduce --max-frames, or use --preprocess-mode max_size. "
            "Keep one joint reconstruction; independently normalized chunks cannot simply be concatenated."
        ) from error
    to_numpy = lambda value: value.detach().float().cpu().numpy()
    world_to_camera = np.broadcast_to(np.eye(4, dtype=np.float32), (len(selected), 4, 4)).copy()
    world_to_camera[:, :3, :] = to_numpy(extrinsics)[0]
    camera_to_world = np.linalg.inv(world_to_camera)
    intrinsics_model = to_numpy(intrinsics)[0]
    affine = np.asarray(geometry["canonical_to_model_pixel_affine"])
    intrinsics_canonical = np.linalg.inv(affine)[None] @ intrinsics_model
    depth = to_numpy(predictions["depth"])[0, ..., 0]
    confidence = to_numpy(predictions["depth_conf"])[0]
    rgb = (to_numpy(images).transpose(0, 2, 3, 1) * 255).round().clip(0, 255).astype(np.uint8)
    pose_enc = to_numpy(predictions["pose_enc"])[0]
    del predictions, extrinsics, intrinsics, images, model
    torch.cuda.empty_cache()
    if depth.shape != (len(selected), model_height, model_width) or confidence.shape != depth.shape:
        raise RuntimeError(f"Unexpected upstream depth/confidence shapes: {depth.shape}, {confidence.shape}")
    if not np.isfinite(world_to_camera).all() or not np.isfinite(intrinsics_model).all():
        raise RuntimeError("Model produced nonfinite camera parameters")

    output.mkdir(parents=True, exist_ok=True)
    (output / "depth").mkdir()
    (output / "rgb_model").mkdir()
    frame_indices = np.asarray([frame["index"] for frame in selected], dtype=np.int64)
    np.savez_compressed(
        output / "cameras.npz",
        frame_indices=frame_indices,
        time_seconds=np.asarray([frame["time_seconds"] for frame in selected], dtype=np.float64),
        pts_seconds=np.asarray([frame["pts_seconds"] for frame in selected], dtype=np.float64),
        world_to_camera=world_to_camera,
        camera_to_world=camera_to_world,
        camera_to_world_blender_camera_basis=camera_to_world @ np.diag([1.0, -1.0, -1.0, 1.0]),
        intrinsics_model=intrinsics_model,
        intrinsics_canonical=intrinsics_canonical,
        canonical_to_model_pixel_affine=affine,
        pose_enc=pose_enc,
    )
    frame_outputs = []
    points, colors, confidences, point_frames = [], [], [], []
    point_budget, extra_points = divmod(max_points, len(selected))
    yy, xx = np.mgrid[0:model_height:point_stride, 0:model_width:point_stride]
    for position, frame in enumerate(selected):
        per_frame_limit = point_budget + int(position < extra_points)
        name = f"{frame['index']:08d}"
        depth_file = f"depth/{name}.npz"
        np.savez_compressed(output / depth_file, depth=depth[position], confidence=confidence[position])
        rgb_file = f"rgb_model/{name}.png"
        Image.fromarray(rgb[position]).save(output / rgb_file)
        finite = np.isfinite(depth[position]) & (depth[position] > 0) & np.isfinite(confidence[position])
        threshold = float(np.percentile(confidence[position][finite], percentile)) if finite.any() else None
        valid = finite[yy, xx]
        if threshold is not None:
            valid &= confidence[position][yy, xx] >= threshold
        ys, xs = yy[valid], xx[valid]
        if len(ys) > per_frame_limit:
            indices = np.linspace(0, len(ys) - 1, per_frame_limit).astype(int)
            ys, xs = ys[indices], xs[indices]
        xyz = unproject_points(depth[position], intrinsics_model[position], camera_to_world[position], ys, xs)
        keep = np.isfinite(xyz).all(axis=1)
        points.append(xyz[keep].astype(np.float32))
        colors.append(rgb[position][ys, xs][keep])
        confidences.append(confidence[position][ys, xs][keep])
        point_frames.append(np.full(int(keep.sum()), frame["index"], dtype=np.int32))
        frame_outputs.append({
            "index": frame["index"], "time_seconds": frame["time_seconds"],
            "pts_seconds": frame["pts_seconds"], "source_file": frame["file"],
            "camera_array_index": position, "depth_file": depth_file, "rgb_model_file": rgb_file,
            "point_cloud_confidence_threshold": threshold,
        })
    point_count = _write_ply(output / "point_cloud.ply", points, colors, confidences, point_frames)
    index = {
        "schema_version": 1, "stage": "vggt_omega", "status": "complete",
        "source_manifest": str(manifest_path), "source_manifest_sha256": _sha256(manifest_path),
        "upstream": {"url": UPSTREAM_URL, "revision": revision, "tested_revision": TESTED_REVISION},
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "device": torch.cuda.get_device_name(device), "precision": precision},
        "selection": {"stride": args.stride, "max_frames": args.max_frames,
                      "strategy": "stride candidates plus final frame, evenly capped over full duration",
                      "input_frame_count": len(manifest["frames"]), "selected_frame_count": len(selected)},
        "preprocessing": {"mode": mode, "image_resolution": image_resolution, **geometry},
        "coordinates": {
            "world": "Common predicted VGGT-Ω world; no axis realignment applied.",
            "camera": "OpenCV: +X right, +Y down, +Z forward.",
            "extrinsics": "world_to_camera maps world homogeneous column vectors to camera coordinates.",
            "intrinsics": "Pixel coordinates; intrinsics_canonical applies to the original oriented image.",
            "blender": "camera_to_world_blender_camera_basis flips local Y/Z only; world alignment remains necessary.",
            "depth": "Optical-axis camera Z depth, not Euclidean ray distance.",
            "scale": "Arbitrary learned scale shared by this joint inference; not meters. Align by a similarity transform.",
            "confidence": "Uncalibrated upstream confidence score, not a probability or error bound.",
        },
        "files": {"cameras": "cameras.npz", "point_cloud": "point_cloud.ply"},
        "point_cloud": {"vertices": point_count, "point_stride": point_stride, "max_points": max_points,
                        "confidence_percentile_per_frame": percentile, "fused": False,
                        "note": "Depth-unprojected colored points; duplicates and dynamic objects are retained."},
        "frames": frame_outputs,
    }
    (output / "index.json").write_text(json.dumps(index, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(selected)} cameras and {point_count:,} points to {output}", flush=True)
    return output / "index.json"
