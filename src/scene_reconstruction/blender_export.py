"""Launch a read-only Blender export and share its camera/depth math with tests."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .common import load_manifest, sha256, write_json


def select_frame_indices(spec: str, count: int) -> list[int]:
    if spec == "all":
        return list(range(count))
    try:
        result = sorted({int(value.strip()) for value in spec.split(",")})
    except ValueError as error:
        raise ValueError("--frames must be 'all' or comma-separated source indices") from error
    if not result or result[0] < 0 or result[-1] >= count:
        raise ValueError("Requested render frame lies outside the source manifest")
    return result


def pixel_resize_affine(source_wh, export_wh):
    sx, sy = export_wh[0] / source_wh[0], export_wh[1] / source_wh[1]
    return np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], dtype=np.float64)


def intrinsics_from_projection(projection, width: int, height: int):
    """Blender perspective clip matrix -> top-left integer pixel-center K."""
    p = np.asarray(projection, dtype=np.float64)
    if p.shape != (4, 4) or not np.isfinite(p).all():
        raise ValueError("Expected a finite 4x4 perspective projection matrix")
    if not np.allclose(p[3], [0, 0, -1, 0], atol=1e-6):
        raise ValueError("Only perspective Blender cameras are supported")
    return np.array([[p[0, 0] * width / 2, 0, (1 - p[0, 2]) * width / 2 - .5],
                     [0, p[1, 1] * height / 2, (1 + p[1, 2]) * height / 2 - .5],
                     [0, 0, 1]], dtype=np.float64)


def ray_norms(intrinsics, shape):
    y, x = np.mgrid[:shape[0], :shape[1]]
    rays = np.stack([x, y, np.ones_like(x)], axis=-1) @ np.linalg.inv(intrinsics).T
    rays /= rays[..., 2:3]
    return np.linalg.norm(rays, axis=-1)


def normalize_depth(raw_depth, intrinsics, convention: str, clip_end: float):
    raw = np.asarray(raw_depth, dtype=np.float32)
    norms = ray_norms(intrinsics, raw.shape)
    if convention == "camera_z":
        depth = raw.copy()
    elif convention == "ray_distance":
        depth = (raw / norms).astype(np.float32)
    else:
        raise ValueError(f"Unknown renderer depth convention: {convention}")
    valid = np.isfinite(depth) & (depth > 0) & (depth < clip_end)
    depth[~valid] = np.nan
    radial = (depth * norms).astype(np.float32)
    return depth, valid, radial


def blender_frame(time_seconds: float, frame_start: float, fps: float, time_scale: float):
    value = frame_start + time_seconds * fps * time_scale
    if not math.isfinite(value):
        raise ValueError("Invalid Blender frame mapping")
    return math.floor(value), value - math.floor(value)


def find_blender(requested=None):
    if requested:
        result = shutil.which(str(requested))
        if result:
            return result
        raise FileNotFoundError(f"Blender executable not found: {requested}")
    candidate = shutil.which("blender")
    if candidate:
        return candidate
    mac = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    if mac.is_file():
        return str(mac)
    raise FileNotFoundError("Install Blender 4.2+ or pass --blender /path/to/blender")


def run(args):
    source = Path(args.blend).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_manifest(manifest_path)
    scale = float(args.resolution_scale)
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("--resolution-scale must be in (0,1]")
    if min(round(manifest["width"] * scale), round(manifest["height"] * scale)) < 4:
        raise ValueError("Export resolution must be at least four pixels on each axis")
    if not math.isfinite(float(args.time_scale)) or args.time_scale <= 0:
        raise ValueError("--time-scale must be finite and positive")
    if args.frame_start is not None and not math.isfinite(float(args.frame_start)):
        raise ValueError("--frame-start must be finite")
    samples = int(getattr(args, "samples", 16))
    if samples < 1:
        raise ValueError("--samples must be positive")
    selected = select_frame_indices(args.frames, len(manifest["frames"]))
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Export output must be empty: {output}")
    executable = find_blender(getattr(args, "blender", None))
    script = Path(__file__).resolve().parents[2] / "scripts/blender_export.py"
    if not script.is_file():
        raise FileNotFoundError("Run from the editable project checkout; scripts/blender_export.py is required")
    source_hash = sha256(source)
    config = {
        "blend": str(source), "source_blend_sha256": source_hash,
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "output": str(output), "render_indices": selected,
        "depth": bool(args.depth), "rgb": bool(args.rgb), "resolution_scale": scale,
        "frame_start": args.frame_start, "time_scale": float(args.time_scale),
        "samples": samples, "camera": getattr(args, "camera", None),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "request.json", config)
    command = [executable, "--background", "--factory-startup", "--disable-autoexec",
               "--python-exit-code", "1", "--python", str(script), "--",
               "--config", str(output / "request.json")]
    log_path = output / "blender.log"
    print(f"Blender is exporting cameras/geometry and requested rasters; log: {log_path}", flush=True)
    try:
        with log_path.open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
            raise RuntimeError(f"Blender exited with code {result.returncode}; see {log_path}\n{tail}")
    finally:
        if sha256(source) != source_hash:
            raise RuntimeError("Source .blend changed during export; investigate before comparison")
    result = json.loads((output / "index.json").read_text())
    if result.get("status") != "complete" or result.get("source_blend_sha256") != source_hash:
        raise RuntimeError("Blender did not produce a complete matching export")
    print(f"Blender export complete: {output / 'index.json'}")
    return output / "index.json"

