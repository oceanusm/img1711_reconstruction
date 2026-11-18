"""Bounded-window DA3 inference with explicit frame and coordinate provenance.

This adapter deliberately keeps each window's world coordinate system separate.
The source-resolution depth images are resamples of the model output, not new
high-resolution estimates. Model imports are deferred until inference is needed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter
from typing import Any
import zipfile

import numpy as np
from PIL import Image

UPSTREAM_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DEFAULT_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DEFAULT_MODEL_REVISION = "b2359bdf726fb44ef62acca04d629dcf158053e7"
_TIMED_STAGES = (
    "model_setup", "inference", "cache_read", "prediction_conversion",
    "native_cache_write", "diagnostics", "resample", "frame_write", "metadata_write",
)


def plan_chunks(count: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Return [start, stop) windows covering every frame, including the tail."""
    if count < 1 or chunk_size < 1 or not 0 <= overlap < chunk_size:
        raise ValueError("Need frames > 0, chunk_size > 0, and 0 <= overlap < chunk_size")
    windows = []
    start = 0
    while start < count:
        stop = min(start + chunk_size, count)
        windows.append((start, stop))
        if stop == count:
            break
        start = stop - overlap
    return windows


def choose_owners(count: int, windows: list[tuple[int, int]]) -> list[int]:
    """Select the window with the most context on both sides; ties go earlier."""
    owners = [-1] * count
    best = [-1] * count
    for chunk_id, (start, stop) in enumerate(windows):
        for position in range(start, stop):
            score = min(position - start, stop - 1 - position)
            if score > best[position]:
                owners[position] = chunk_id
                best[position] = score
    if -1 in owners:
        raise ValueError("Chunk plan does not cover all frames")
    return owners


def rescale_intrinsics(intrinsics: np.ndarray, native_hw: tuple[int, int],
                       source_hw: tuple[int, int]) -> np.ndarray:
    """Undo DA3's resize using the upstream intrinsic-matrix convention."""
    native_h, native_w = native_hw
    source_h, source_w = source_hw
    transform = np.diag([source_w / native_w, source_h / native_h, 1.0])
    return (transform @ np.asarray(intrinsics)).astype(np.float32)


def _resize(array: np.ndarray, source_hw: tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    height, width = source_hw
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return np.asarray(Image.fromarray(np.asarray(array, dtype=np.float32)).resize(
        (width, height), resample=mode), dtype=np.float32)


def resize_depth(depth: np.ndarray, source_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear resampling that cannot spread NaNs across neighboring pixels."""
    valid = np.isfinite(depth) & (depth > 0)
    numerator = _resize(np.where(valid, depth, 0), source_hw)
    denominator = _resize(valid.astype(np.float32), source_hw)
    output = np.full(source_hw, np.nan, dtype=np.float32)
    good = (denominator > 1e-6) & (_resize(valid, source_hw, nearest=True) > 0.5)
    np.divide(numerator, denominator, out=output, where=good)
    return output, good


def summarize_overlap(previous: dict[str, np.ndarray], current: dict[str, np.ndarray]) -> list[dict]:
    """Measure repeated-frame disagreement without altering either prediction."""
    prev_lookup = {int(index): i for i, index in enumerate(previous["frame_indices"])}
    reports = []
    for j, index in enumerate(current["frame_indices"]):
        i = prev_lookup.get(int(index))
        if i is None:
            continue
        a, b = previous["depth"][i], current["depth"][j]
        if a.shape != b.shape:
            raise ValueError("Overlapping DA3 windows returned different native resolutions")
        valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        if "sky" in previous:
            valid &= ~previous["sky"][i].astype(bool)
        if "sky" in current:
            valid &= ~current["sky"][j].astype(bool)
        # An evenly spaced sample bounds diagnostic work independently of resolution.
        av, bv = a[::4, ::4], b[::4, ::4]
        valid = valid[::4, ::4]
        item: dict[str, Any] = {"frame_index": int(index), "sampled_valid_pixels": int(valid.sum())}
        if valid.any():
            av, bv = av[valid], bv[valid]
            ratio = float(np.median(bv / av))
            item.update({
                "median_current_over_previous_depth": ratio,
                "median_absolute_relative_difference": float(np.median(np.abs(bv - av) / av)),
                "median_relative_difference_after_diagnostic_scale_fit": float(
                    np.median(np.abs(bv / ratio - av) / av)),
            })
        reports.append(item)
    return reports


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _validate_compression_level(level: int) -> int:
    if not isinstance(level, (int, np.integer)) or not 0 <= level <= 9:
        raise ValueError("DA3 compression_level must be an integer from 0 through 9")
    return int(level)


def _write_npz(path: Path, arrays: dict[str, np.ndarray], compression_level: int = 1) -> None:
    """Atomically write a standard lossless NPZ, trading compression time for size."""
    level = _validate_compression_level(compression_level)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary, mode="w",
            compression=zipfile.ZIP_STORED if level == 0 else zipfile.ZIP_DEFLATED,
            compresslevel=None if level == 0 else level,
            allowZip64=True,
        ) as archive:
            for key, array in arrays.items():
                with archive.open(f"{key}.npy", mode="w", force_zip64=True) as entry:
                    np.lib.format.write_array(entry, np.asanyarray(array), allow_pickle=False)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _save_timings(output: Path, report: dict, started: float,
                  outside_seconds: dict[str, float], outside_metadata_bytes: int) -> None:
    """Snapshot this invocation; writing the snapshot itself is excluded from bytes."""
    windows = report["windows"]
    seconds = dict(outside_seconds)
    for stage in _TIMED_STAGES:
        seconds[stage] = seconds.get(stage, 0.0) + sum(item["seconds"][stage] for item in windows)
    total_seconds = perf_counter() - started
    seconds["other"] = max(0.0, total_seconds - sum(seconds.values()))
    seconds["total"] = total_seconds
    byte_counts = {kind: sum(item["bytes_written"][kind] for item in windows)
                   for kind in ("native_cache", "frames", "metadata")}
    byte_counts["metadata"] += outside_metadata_bytes
    byte_counts["total"] = sum(byte_counts.values())
    peaks = [item["cuda_peak_memory"] for item in windows if item["cuda_peak_memory"] is not None]
    report["totals"] = {
        "completed_windows": len(windows),
        "cache_hits": sum(item["cache_hit"] for item in windows),
        "inferred_windows": sum(not item["cache_hit"] for item in windows),
        "frames_written": sum(item["frames_written"] for item in windows),
        "native_cache_bytes_read": sum(item["native_cache_bytes_read"] for item in windows),
        "bytes_written": byte_counts,
        "seconds": seconds,
        "cuda_peak_memory": ({key: max(item[key] for item in peaks) for key in peaks[0]}
                             if peaks else None),
    }
    _write_json(output / "timings.json", report)


def _homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] not in {(3, 4), (4, 4)}:
        raise ValueError(f"Unexpected DA3 extrinsics shape {extrinsics.shape}")
    if extrinsics.shape[1:] == (4, 4):
        return extrinsics.copy()
    result = np.broadcast_to(np.eye(4, dtype=np.float32), (len(extrinsics), 4, 4)).copy()
    result[:, :3] = extrinsics
    return result


def _prediction_arrays(prediction: Any, indices: list[int], source_hw: tuple[int, int]) -> dict:
    depth = np.asarray(prediction.depth, dtype=np.float32)
    if depth.ndim != 3 or depth.shape[0] != len(indices):
        raise ValueError(f"Unexpected DA3 depth shape {depth.shape}")
    native_hw = tuple(depth.shape[1:])
    result = {"frame_indices": np.asarray(indices, dtype=np.int64), "depth": depth}
    for attribute, key in (("conf", "confidence"), ("sky", "sky")):
        value = getattr(prediction, attribute, None)
        if value is not None:
            value = np.asarray(value)
            if value.shape != depth.shape:
                raise ValueError(f"Unexpected DA3 {attribute} shape {value.shape}")
            result[key] = value.astype(bool if key == "sky" else np.float32)
    intrinsics = getattr(prediction, "intrinsics", None)
    if intrinsics is not None:
        intrinsics = np.asarray(intrinsics, dtype=np.float32)
        if intrinsics.shape != (len(indices), 3, 3):
            raise ValueError(f"Unexpected DA3 intrinsics shape {intrinsics.shape}")
        result["intrinsics_native"] = intrinsics
        result["intrinsics_source"] = rescale_intrinsics(intrinsics, native_hw, source_hw)
    extrinsics = getattr(prediction, "extrinsics", None)
    if extrinsics is not None:
        w2c = _homogeneous(extrinsics)
        if len(w2c) != len(indices) or not np.isfinite(w2c).all():
            raise ValueError("DA3 returned invalid camera transforms")
        result["world_to_camera"] = w2c
        result["camera_to_world"] = np.linalg.inv(w2c).astype(np.float32)
    return result


def run(args: Any) -> Path:
    """Run all frames and return the final output index. Identical runs resume."""
    started = perf_counter()
    compression_level = _validate_compression_level(getattr(args, "compression_level", 1))
    stop_after_windows = getattr(args, "stop_after_windows", None)
    if stop_after_windows is not None and (
        not isinstance(stop_after_windows, (int, np.integer)) or stop_after_windows < 1
    ):
        raise ValueError("DA3 stop_after_windows must be a positive integer")
    if stop_after_windows is not None:
        stop_after_windows = int(stop_after_windows)
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    frames = manifest["frames"]
    source_hw = (int(manifest["height"]), int(manifest["width"]))
    if min(source_hw) < 1:
        raise ValueError("Manifest resolution must be positive")
    indices = [int(frame["index"]) for frame in frames]
    if len(set(indices)) != len(indices) or indices != sorted(indices):
        raise ValueError("Manifest frame indices must be unique and ordered")
    paths = [(manifest_path.parent / frame["file"]).resolve() for frame in frames]
    image_stats = []
    for path in paths:
        stat = path.stat()
        with Image.open(path) as image:
            if image.size != source_hw[::-1]:
                raise ValueError(f"Unexpected dimensions for {path}; all canonical frames must match")
        image_stats.append([stat.st_size, stat.st_mtime_ns])
    chunk_size, overlap = int(args.chunk_size), int(args.overlap)
    process_res = int(args.process_res)
    if process_res < 28 or process_res % 14:
        raise ValueError("DA3 process_res must be a multiple of 14 and at least 28")
    windows = plan_chunks(len(frames), chunk_size, overlap)
    owners = choose_owners(len(frames), windows)
    output = Path(args.output).expanduser().resolve()
    model_name = str(args.model)
    model_revision = getattr(args, "model_revision", None)
    if model_revision is None and model_name == DEFAULT_MODEL:
        model_revision = DEFAULT_MODEL_REVISION
    settings = {
        "schema_version": 1,
        "adapter_version": 1,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "input_file_stat_sha256": hashlib.sha256(json.dumps(image_stats).encode()).hexdigest(),
        "model": model_name,
        "model_revision": model_revision,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "process_res": process_res,
        "process_res_method": "upper_bound_resize",
        "device": str(args.device),
        "source_hw": list(source_hw),
        "reviewed_upstream_commit": UPSTREAM_COMMIT,
    }
    outside_seconds = {"input_validation": perf_counter() - started, "metadata_write": 0.0}
    outside_metadata_bytes = 0
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != settings:
            raise ValueError("DA3 output contains a different run. Use a new --output directory.")
    else:
        if any(output.iterdir()):
            raise ValueError("DA3 output directory is nonempty without run.json; choose a new directory")
        stage_started = perf_counter()
        _write_json(config_path, settings)
        outside_seconds["metadata_write"] += perf_counter() - stage_started
        outside_metadata_bytes += config_path.stat().st_size
    (output / "chunks").mkdir(exist_ok=True)
    (output / "frames").mkdir(exist_ok=True)
    index: dict[str, Any] = {
        **settings,
        "status": "incomplete",
        "source_manifest": str(manifest_path),
        "frame_count": len(frames),
        "camera_convention": "OpenCV: x right, y down, z forward; extrinsics are world-to-camera",
        "depth_convention": "camera-space z depth, not Euclidean ray distance",
        "coordinate_scope": "separate local world coordinate system per chunk; no global pose stitching",
        "depth_scale": "per-chunk model estimate; see is_metric and depth_units on each chunk",
        "confidence_semantics": "unmodified model score, not a calibrated probability",
        "source_resolution_note": "bilinear upsampling of native prediction, no added geometric detail",
        "ownership": "window maximizing distance to its nearest boundary; ties prefer earlier window",
        "chunks": [],
        "frames": [],
    }
    frame_records: dict[int, dict] = {}
    overlap_reports = []
    previous = None
    model = None
    torch = None
    timing_report = {
        "schema_version": 1,
        "status": "incomplete",
        "scope": "latest invocation; window records describe completed windows only",
        "clock": "perf_counter wall seconds; inference includes the complete model.inference API call",
        "requested_compression_level": compression_level,
        "stop_after_windows": stop_after_windows,
        "compression_note": "lossless NPZ; 0 stores arrays uncompressed, 1 through 9 use DEFLATE",
        "resume_note": "native caches retain their existing compression; source-resolution frames are re-exported",
        "bytes_written_note": "counts each completed write, including overwritten metadata; excludes timings.json",
        "elapsed_note": "elapsed time is measured before saving each timing snapshot; earlier snapshot writes are in other",
        "cuda_memory_note": "per-window PyTorch allocator peaks after model setup; not total device memory usage",
        "windows": [],
    }
    _save_timings(output, timing_report, started, outside_seconds, outside_metadata_bytes)
    for chunk_id, (start, stop) in enumerate(windows):
        window_started = perf_counter()
        timing = {
            "chunk_id": chunk_id,
            "frame_indices": indices[start:stop],
            "cache_hit": False,
            "frames_written": 0,
            "native_cache_bytes_read": 0,
            "bytes_written": {"native_cache": 0, "frames": 0, "metadata": 0},
            "seconds": dict.fromkeys(_TIMED_STAGES, 0.0),
            "cuda_peak_memory": None,
        }
        seconds = timing["seconds"]
        print(f"DA3 window {chunk_id + 1}/{len(windows)}: source frames {indices[start]}–{indices[stop - 1]}", flush=True)
        chunk_path = output / "chunks" / f"{chunk_id:06d}.npz"
        meta_path = chunk_path.with_suffix(".json")
        if chunk_path.exists() and meta_path.exists():
            stage_started = perf_counter()
            metadata = json.loads(meta_path.read_text())
            with np.load(chunk_path, allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
            if arrays["frame_indices"].tolist() != indices[start:stop]:
                raise ValueError(f"Cached DA3 chunk mapping is corrupt: {chunk_path}")
            seconds["cache_read"] = perf_counter() - stage_started
            timing["cache_hit"] = True
            timing["native_cache_bytes_read"] = chunk_path.stat().st_size
        else:
            if model is None:
                stage_started = perf_counter()
                try:
                    import torch as torch_module
                    from depth_anything_3.api import DepthAnything3
                except ImportError as exc:
                    raise RuntimeError("DA3 dependencies are missing. Use the dedicated DA3 environment; see docs/depth_anything3.md") from exc
                torch = torch_module
                if str(args.device).startswith("cuda") and not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable. Run DA3 in its environment on the A6000 host.")
                kwargs = {"revision": model_revision} if model_revision else {}
                model = DepthAnything3.from_pretrained(model_name, **kwargs).to(args.device).eval()
                seconds["model_setup"] = perf_counter() - stage_started
            cuda = getattr(torch, "cuda", None) if str(args.device).startswith("cuda") else None
            measure_cuda = cuda is not None and all(hasattr(cuda, method) for method in (
                "reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"))
            if measure_cuda:
                cuda.reset_peak_memory_stats(args.device)
            stage_started = perf_counter()
            try:
                prediction = model.inference(
                    image=[str(path) for path in paths[start:stop]],
                    process_res=process_res,
                    process_res_method="upper_bound_resize",
                    infer_gs=False,
                    export_dir=None,
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError("DA3 exhausted GPU memory. Stop other GPU jobs and retry with a smaller --chunk-size and/or --process-res in a new output directory.") from exc
                raise
            seconds["inference"] = perf_counter() - stage_started
            if measure_cuda:
                timing["cuda_peak_memory"] = {
                    "allocated_bytes": int(cuda.max_memory_allocated(args.device)),
                    "reserved_bytes": int(cuda.max_memory_reserved(args.device)),
                }
            stage_started = perf_counter()
            arrays = _prediction_arrays(prediction, indices[start:stop], source_hw)
            is_metric = bool(getattr(prediction, "is_metric", False))
            scale_factor = getattr(prediction, "scale_factor", None)
            native_h, native_w = arrays["depth"].shape[1:]
            scale_x, scale_y = native_w / source_hw[1], native_h / source_hw[0]
            metadata = {
                "chunk_id": chunk_id,
                "file": str(chunk_path.relative_to(output)),
                "frame_indices": indices[start:stop],
                "native_hw": [native_h, native_w],
                "source_hw": list(source_hw),
                "is_metric": is_metric,
                "depth_units": "estimated_meters" if is_metric else "relative_model_units",
                "metric_scale_factor": None if scale_factor is None else float(scale_factor),
                "model_commit_hash": getattr(model, "_commit_hash", None),
                "source_to_native_intrinsics_transform": [[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]],
                "source_to_native_pixel_center_transform": [[scale_x, 0, (scale_x - 1) / 2], [0, scale_y, (scale_y - 1) / 2], [0, 0, 1]],
                "crop": None,
                "world_frame": f"da3_chunk_{chunk_id:06d}",
            }
            try:
                metadata["depth_anything_3_package_version"] = importlib.metadata.version("depth-anything-3")
            except importlib.metadata.PackageNotFoundError:
                metadata["depth_anything_3_package_version"] = "unpackaged"
            seconds["prediction_conversion"] = perf_counter() - stage_started
            stage_started = perf_counter()
            _write_npz(chunk_path, arrays, compression_level)
            seconds["native_cache_write"] = perf_counter() - stage_started
            timing["bytes_written"]["native_cache"] = chunk_path.stat().st_size
            stage_started = perf_counter()
            _write_json(meta_path, metadata)
            seconds["metadata_write"] += perf_counter() - stage_started
            timing["bytes_written"]["metadata"] += meta_path.stat().st_size
            del prediction
            if torch is not None and str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
        stage_started = perf_counter()
        if previous is not None:
            overlap_reports.append({
                "previous_chunk_id": chunk_id - 1,
                "current_chunk_id": chunk_id,
                "frames": summarize_overlap(previous, arrays),
            })
        seconds["diagnostics"] = perf_counter() - stage_started
        for local_index, position in enumerate(range(start, stop)):
            if owners[position] != chunk_id:
                continue
            stage_started = perf_counter()
            depth, valid = resize_depth(arrays["depth"][local_index], source_hw)
            payload = {"depth": depth, "valid_depth": valid}
            if "sky" in arrays:
                sky = _resize(arrays["sky"][local_index], source_hw, nearest=True) > 0.5
                payload["sky"] = sky
                payload["valid_depth"] &= ~sky
            if "confidence" in arrays:
                payload["confidence"] = _resize(arrays["confidence"][local_index], source_hw)
            for native_key, frame_key in (("intrinsics_source", "intrinsics"),
                                           ("world_to_camera", "world_to_camera"),
                                           ("camera_to_world", "camera_to_world")):
                if native_key in arrays:
                    payload[frame_key] = arrays[native_key][local_index]
            seconds["resample"] += perf_counter() - stage_started
            frame_path = output / "frames" / f"{indices[position]:06d}.npz"
            stage_started = perf_counter()
            _write_npz(frame_path, payload, compression_level)
            seconds["frame_write"] += perf_counter() - stage_started
            timing["frames_written"] += 1
            timing["bytes_written"]["frames"] += frame_path.stat().st_size
            frame_records[position] = {
                "index": indices[position],
                "time_seconds": frames[position]["time_seconds"],
                "pts_seconds": frames[position].get("pts_seconds"),
                "file": str(frame_path.relative_to(output)),
                "chunk_id": chunk_id,
                "local_index": local_index,
                "world_frame": metadata["world_frame"],
                "depth_units": metadata["depth_units"],
                "valid_fraction": float(payload["valid_depth"].mean()),
            }
        index["chunks"].append(metadata)
        index["frames"] = [frame_records[position] for position in sorted(frame_records)]
        stage_started = perf_counter()
        _write_json(output / "index.json", index)
        _write_json(output / "overlap_diagnostics.json", {
            "note": "Duplicate-frame diagnostics only; no scale or pose alignment has been applied",
            "windows": overlap_reports,
        })
        seconds["metadata_write"] += perf_counter() - stage_started
        timing["bytes_written"]["metadata"] += sum((output / name).stat().st_size for name in (
            "index.json", "overlap_diagnostics.json"))
        previous = arrays
        total_seconds = perf_counter() - window_started
        seconds["other"] = max(0.0, total_seconds - sum(seconds.values()))
        seconds["total"] = total_seconds
        timing["bytes_written"]["total"] = sum(timing["bytes_written"].values())
        timing_report["windows"].append(timing)
        _save_timings(output, timing_report, started, outside_seconds, outside_metadata_bytes)
        detail = " | ".join(f"{stage.replace('_', ' ')} {seconds[stage]:.2f}s" for stage in _TIMED_STAGES
                            if seconds[stage] > 0 or stage in {"inference", "resample", "frame_write"})
        memory = timing["cuda_peak_memory"]
        memory_note = (f" | CUDA peak {memory['allocated_bytes'] / 2**30:.2f} GiB allocated, "
                       f"{memory['reserved_bytes'] / 2**30:.2f} GiB reserved" if memory else "")
        print(f"DA3 window {chunk_id + 1} wall {total_seconds:.2f}s "
              f"({'cache hit' if timing['cache_hit'] else 'inferred'}): {detail} | "
              f"other {seconds['other']:.2f}s | wrote {timing['frames_written']} frames, "
              f"{timing['bytes_written']['total'] / 2**20:.1f} MiB{memory_note}", flush=True)
        if (stop_after_windows is not None and len(timing_report["windows"]) >= stop_after_windows
                and chunk_id + 1 < len(windows)):
            print(f"DA3 stopped after {len(timing_report['windows'])} "
                  f"window{'s' if len(timing_report['windows']) != 1 else ''}: incomplete profile run; "
                  "rerun without --stop-after-windows to finish. "
                  f"Breakdown: {output / 'timings.json'}", flush=True)
            return output / "index.json"
    if len(frame_records) != len(frames):
        raise RuntimeError("DA3 output does not cover every canonical frame")
    index["status"] = "complete"
    stage_started = perf_counter()
    _write_json(output / "index.json", index)
    outside_seconds["metadata_write"] += perf_counter() - stage_started
    outside_metadata_bytes += (output / "index.json").stat().st_size
    timing_report["status"] = "complete"
    _save_timings(output, timing_report, started, outside_seconds, outside_metadata_bytes)
    totals = timing_report["totals"]
    print(f"DA3 wall total {totals['seconds']['total']:.2f}s: "
          f"model setup {totals['seconds']['model_setup']:.2f}s | "
          f"inference {totals['seconds']['inference']:.2f}s | "
          f"resample {totals['seconds']['resample']:.2f}s | "
          f"frame write {totals['seconds']['frame_write']:.2f}s | "
          f"cache hits {totals['cache_hits']}/{len(windows)} | "
          f"wrote {totals['bytes_written']['total'] / 2**30:.2f} GiB; "
          f"breakdown: {output / 'timings.json'}", flush=True)
    print(f"DA3 complete: {len(frames)} frames; {output / 'index.json'}", flush=True)
    return output / "index.json"
