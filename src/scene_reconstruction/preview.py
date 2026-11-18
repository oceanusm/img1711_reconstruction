"""Qualitative RGB/depth preview with one color range and source video timing."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import load_manifest, sha256, write_json


def frame_durations(frames: list[dict]) -> list[float]:
    times = [float(frame["time_seconds"]) for frame in frames]
    if not times or not all(math.isfinite(value) for value in times):
        raise ValueError("Preview requires finite frame timestamps")
    durations = [b - a for a, b in zip(times, times[1:])]
    if any(value <= 0 for value in durations):
        raise ValueError("Preview timestamps must increase strictly")
    last = float(frames[-1].get("duration_seconds", 0))
    if not math.isfinite(last) or last <= 0:
        if not durations:
            raise ValueError("A one-frame preview needs a positive duration_seconds")
        last = float(np.median(durations))
    return durations + [last]


def color_limits(samples: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate(samples) if samples else np.empty(0)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        raise ValueError("No valid positive depth values to visualize")
    low, high = np.percentile(values, [2, 98])
    if high <= low:
        high = low + max(float(low) * 0.01, 1e-6)
    return float(low), float(high)


def colorize(depth: np.ndarray, valid: np.ndarray, low: float, high: float) -> np.ndarray:
    """A fixed blue/purple/orange/yellow sequential map; invalid pixels black."""
    if high <= low:
        raise ValueError("Color range must be increasing")
    palette = np.asarray([[10, 15, 70], [65, 40, 140], [155, 45, 110],
                          [235, 105, 45], [255, 235, 120]], dtype=np.float32)
    good = np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > 0)
    normalized = np.clip((np.where(good, depth, low) - low) / (high - low), 0, 1)
    positions = np.linspace(0, 1, len(palette))
    colors = np.stack([np.interp(normalized, positions, palette[:, i]) for i in range(3)], axis=-1)
    colors[~good] = 0
    return colors.astype(np.uint8)


def _load_depth(path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        depth = np.asarray(archive["depth"], dtype=np.float32)
        if depth.shape != shape:
            raise ValueError(f"Depth resolution differs from the canonical video: {path}")
        valid = np.isfinite(depth) & (depth > 0)
        if "valid_depth" in archive:
            if archive["valid_depth"].shape != shape:
                raise ValueError(f"Invalid validity-mask dimensions: {path}")
            valid &= archive["valid_depth"].astype(bool)
        if "sky" in archive:
            valid &= ~archive["sky"].astype(bool)
    return depth, valid


def run(args) -> Path:
    manifest_path = Path(args.manifest).expanduser().resolve()
    depth_index_path = Path(args.depth_index).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    sidecar = output.with_suffix(".scale.json")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Preview output must be an .mp4 file")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Preview output or color scale already exists: {output}")
    ffmpeg = str(args.ffmpeg)
    if not shutil.which(ffmpeg):
        raise RuntimeError(f"FFmpeg is unavailable: {ffmpeg}")
    max_height = int(args.max_height)
    if max_height < 240 or max_height > 2160:
        raise ValueError("Preview max_height must be between 240 and 2160")
    manifest = load_manifest(manifest_path)
    depth_index = json.loads(depth_index_path.read_text())
    if depth_index.get("status") != "complete":
        raise ValueError("Depth preview requires a complete DA3 index")
    manifest_hash = sha256(manifest_path)
    if depth_index.get("manifest_sha256") not in {None, manifest_hash}:
        raise ValueError("DA3 index and canonical frame manifest have different SHA256 hashes")
    frames = manifest["frames"]
    predictions = depth_index.get("frames", [])
    if [frame["index"] for frame in predictions] != [frame["index"] for frame in frames]:
        raise ValueError("Depth index must cover every canonical frame exactly once in order")
    for frame, prediction in zip(frames, predictions):
        prediction_time = float(prediction["time_seconds"])
        if not math.isfinite(prediction_time) or abs(float(frame["time_seconds"]) - prediction_time) > 1e-6:
            raise ValueError("Depth index and canonical manifest timestamps differ")
    durations = frame_durations(frames)
    source_hw = (int(manifest["height"]), int(manifest["width"]))
    paths = [depth_index_path.parent / prediction["file"] for prediction in predictions]
    # Around 4096 spatial samples per frame, with equal sampling across frames.
    stride = max(1, math.ceil(math.sqrt(source_hw[0] * source_hw[1] / 4096)))
    samples = []
    for position, path in enumerate(paths):
        depth, valid = _load_depth(path, source_hw)
        selected = depth[::stride, ::stride][valid[::stride, ::stride]]
        samples.append(selected)
        if position % 100 == 0:
            print(f"Depth preview range: {position + 1}/{len(paths)} frames", flush=True)
    low, high = color_limits(samples)
    del samples
    content_height = min(source_hw[0], max_height - 104)
    content_height -= content_height % 2
    image_width = max(2, round(source_hw[1] * content_height / source_hw[0]))
    image_width += image_width % 2
    panel_width = max(300, image_width)
    canvas_size = (panel_width * 2, content_height + 104)
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:  # Pillow 10.0 compatibility
        font = ImageFont.load_default()
    units = sorted({prediction.get("depth_units", "model_units") for prediction in predictions})
    if len(units) != 1:
        raise ValueError("A shared preview color scale requires identical depth units across frames")
    unit_label = {"estimated_meters": "estimated m", "relative_model_units": "relative units"}.get(units[0], units[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".depth-preview-", dir=output.parent) as temporary:
        temp = Path(temporary)
        concat_lines = ["ffconcat version 1.0"]
        for position, (frame, prediction, path, duration) in enumerate(zip(frames, predictions, paths, durations)):
            depth, valid = _load_depth(path, source_hw)
            colors = colorize(depth, valid, low, high)
            with Image.open(manifest_path.parent / frame["file"]) as source:
                if source.size != source_hw[::-1]:
                    raise ValueError("Canonical RGB dimensions differ from manifest")
                rgb = source.convert("RGB").resize((image_width, content_height), Image.Resampling.LANCZOS)
            heatmap = Image.fromarray(colors).resize((image_width, content_height), Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", canvas_size, "#161820")
            inset = (panel_width - image_width) // 2
            canvas.paste(rgb, (inset, 48))
            canvas.paste(heatmap, (panel_width + inset, 48))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 6), f"Frame {frame['index']:06d}  |  {frame['time_seconds']:.3f} s", font=font, fill="white")
            draw.text((8, 26), "RGB", font=font, fill="white")
            draw.text((panel_width + 8, 26), f"DA3 depth | window {prediction.get('chunk_id', '?')}", font=font, fill="white")
            footer_y = 48 + content_height
            gradient = np.repeat(np.linspace(low, high, canvas_size[0] - 16, dtype=np.float32)[None], 12, axis=0)
            canvas.paste(Image.fromarray(colorize(gradient, np.ones_like(gradient, dtype=bool), low, high)), (8, footer_y + 4))
            draw.text((8, footer_y + 20), f"Near {low:.3g}  ->  Far {high:.3g} {unit_label} | black = invalid", font=font, fill="white")
            draw.text((8, footer_y + 38), "Approximate depth; scale can vary between windows", font=font, fill="#f4cc80")
            name = f"{position:06d}.jpg"
            canvas.save(temp / name, quality=90)
            concat_lines.extend([f"file '{name}'", "option framerate 1000000", f"duration {duration:.9f}"])
            if position % 100 == 0:
                print(f"Depth preview rendering: {position + 1}/{len(paths)} frames", flush=True)
        # Endpoint duplicate tells FFmpeg to hold the last source frame for its duration.
        concat_lines.extend([f"file '{len(frames) - 1:06d}.jpg'", "option framerate 1000000"])
        concat_path = temp / "frames.ffconcat"
        concat_path.write_text("\n".join(concat_lines) + "\n")
        video_path = temp / "preview.mp4"
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                   "-f", "concat", "-safe", "0", "-i", str(concat_path),
                   "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-bf", "0",
                   "-pix_fmt", "yuv420p", "-fps_mode", "vfr", "-enc_time_base", "1:1000000",
                   "-video_track_timescale", "1000000", "-movflags", "+faststart", str(video_path)]
        subprocess.run(command, check=True)
        metadata = {
            "schema_version": 1,
            "purpose": "qualitative preview; source NPZ files are authoritative",
            "manifest_sha256": manifest_hash,
            "depth_index_sha256": sha256(depth_index_path),
            "depth_units": units,
            "color_range": [low, high],
            "percentiles": [2, 98],
            "sampling": f"every frame, spatial stride {stride}",
            "normalization": "one fixed linear color scale across the whole video; clipped at range endpoints",
            "warning": "Depth is approximate; depth scale and local coordinates may vary between windows",
            "invalid_color": "black",
            "source_frame_count": len(frames),
            "source_start_time_seconds": float(frames[0]["time_seconds"]),
            "duration_seconds": sum(durations),
            "terminal_duplicate": "one repeated final image at the endpoint, lasting 1 microsecond, preserves the final hold",
            "video_size": list(canvas_size),
        }
        write_json(temp / "scale.json", metadata)
        if output.exists() or sidecar.exists():
            raise FileExistsError("Preview destination appeared during rendering; refusing to replace it")
        video_path.replace(output)
        (temp / "scale.json").replace(sidecar)
    print(f"Depth preview: {output}; fixed color scale: {sidecar}", flush=True)
    return output
