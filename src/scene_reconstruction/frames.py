"""Decode once, preserving presentation order and variable-rate timestamps."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from .common import load_manifest, sha256, write_json


HLG_FILTER = (
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
    "tonemap=tonemap=mobius:desat=0,zscale=t=bt709:m=bt709:r=full,format=yuvj444p"
)


def probe_video(video: Path):
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe is required (install FFmpeg)")
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_streams", "-show_frames", "-show_entries",
               "stream:frame=best_effort_timestamp_time,pkt_duration_time,duration_time",
               "-of", "json", str(video)]
    result = json.loads(subprocess.check_output(command))
    if not result.get("streams") or not result.get("frames"):
        raise ValueError("No decodable video frames")
    return result


def extract(args):
    video = Path(args.video).expanduser().resolve()
    output = Path(args.output).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}; choose a new directory")
    ffmpeg = getattr(args, "ffmpeg", "ffmpeg")
    if not shutil.which(ffmpeg):
        raise RuntimeError(f"ffmpeg is required: {ffmpeg}")
    probe = probe_video(video)
    stream = probe["streams"][0]
    raw_frames = probe["frames"]
    pts = [float(frame["best_effort_timestamp_time"]) for frame in raw_frames]
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("Duplicate or nonmonotonic presentation timestamps; inspect this video before extracting")
    hdr = stream.get("color_transfer") in {"arib-std-b67", "smpte2084"}
    tone_map = args.color == "tonemap" or (args.color == "auto" and hdr)
    if tone_map:
        filters = subprocess.check_output([ffmpeg, "-hide_banner", "-filters"], text=True, stderr=subprocess.DEVNULL)
        if not all(name in filters for name in ("zscale", "tonemap")):
            raise RuntimeError("HDR extraction needs FFmpeg with zscale and tonemap (e.g. Ubuntu apt ffmpeg). Use --ffmpeg /path/to/full-build; do not silently treat HDR as SDR.")
    pixel_format = "yuvj444p"
    vf = HLG_FILTER if tone_map else f"format={pixel_format}"
    output.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
               "-i", str(video), "-map", "0:v:0", "-an", "-sn", "-dn",
               "-vf", vf, "-fps_mode", "passthrough", "-q:v", "2",
               "-start_number", "0", str(output / "%06d.jpg")]
    subprocess.run(command, check=True)
    paths = sorted(output.glob("*.jpg"))
    if len(paths) != len(pts):
        raise RuntimeError(f"Decoded {len(paths)} images but probed {len(pts)} frames; no manifest written")
    with Image.open(paths[0]) as first:
        width, height = first.size
    frames = []
    for i, path in enumerate(paths):
        duration = (pts[i + 1] - pts[i]) if i + 1 < len(pts) else float(
            raw_frames[i].get("duration_time", raw_frames[i].get("pkt_duration_time", 0)))
        if i + 1 == len(pts) and stream.get("duration") is not None:
            remaining = float(stream.get("start_time", pts[0])) + float(stream["duration"]) - pts[i]
            if remaining > 0:
                duration = min(duration, remaining) if duration > 0 else remaining
        frames.append({"index": i, "file": path.name, "pts_seconds": pts[i],
                       "time_seconds": pts[i] - pts[0], "duration_seconds": duration})
    manifest = {
        "schema_version": 1,
        "source": {"filename": video.name, "sha256": sha256(video),
                   "coded_width": stream["width"], "coded_height": stream["height"],
                   "codec": stream.get("codec_name"), "pixel_format": stream.get("pix_fmt"),
                   "color_transfer": stream.get("color_transfer"),
                   "color_primaries": stream.get("color_primaries"),
                   "display_rotation_degrees": next((x["rotation"] for x in stream.get("side_data_list", []) if "rotation" in x), 0),
                   "average_frame_rate": stream.get("avg_frame_rate"),
                   "stream_duration_seconds": float(stream["duration"]) if stream.get("duration") is not None else None,
                   "time_base": stream.get("time_base")},
        "width": width, "height": height, "frame_count": len(frames),
        "pixel_coordinates": "origin top-left; x right; y down; unmirrored upright display orientation",
        "extraction": {"autorotate": True, "resampled_fps": False,
                       "tone_mapped_to_bt709": tone_map, "filter": vf,
                       "image_format": "JPEG quality=2 4:4:4; lossy RGB inputs, masks remain lossless PNG",
                       "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0]},
        "frames": frames,
    }
    write_json(output / "manifest.json", manifest)
    make_contact_sheet(output / "manifest.json", output / "contact_sheet.jpg")
    print(f"Extracted {len(frames)} frames at {width}x{height}: {output / 'manifest.json'}")


def make_contact_sheet(manifest_path: Path, destination: Path, count=12):
    manifest = load_manifest(manifest_path)
    frames = manifest["frames"]
    count = min(count, len(frames))
    indices = [round(i * (len(frames) - 1) / max(count - 1, 1)) for i in range(count)]
    cols = min(6, count)
    cell_w, cell_h = 180, 350
    sheet = Image.new("RGB", (cols * cell_w, math.ceil(count / cols) * cell_h), "#202020")
    draw = ImageDraw.Draw(sheet)
    for n, i in enumerate(indices):
        frame = frames[i]
        with Image.open(manifest_path.parent / frame["file"]) as im:
            im.thumbnail((cell_w - 8, cell_h - 32))
            x, y = (n % cols) * cell_w, (n // cols) * cell_h
            sheet.paste(im, (x + (cell_w - im.width) // 2, y))
        draw.text((x + 4, y + cell_h - 25), f"{i:06d} | {frame['time_seconds']:.3f}s", fill="white")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=90)

