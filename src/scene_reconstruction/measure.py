"""Read camera-space surface distances from a single DA3 frame."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import write_json


def camera_points(depth, intrinsics, pixels):
    height, width = depth.shape
    result = []
    for x, y in pixels:
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"Pixel {(x, y)} is outside {width}x{height}")
        z = float(depth[y, x])
        if not np.isfinite(z) or z <= 0:
            raise ValueError(f"Invalid depth at {(x, y)}")
        ray = np.linalg.solve(intrinsics, [x, y, 1.0])
        result.append(ray * z / ray[2])
    return np.asarray(result)


def run(args):
    index_path = Path(args.depth_index).resolve()
    index = json.loads(index_path.read_text())
    record = next((frame for frame in index["frames"] if frame["index"] == args.frame), None)
    if record is None:
        raise ValueError(f"No depth for frame {args.frame}")
    pixels = []
    for value in args.pixels:
        pieces = value.split(",")
        if len(pieces) != 2:
            raise ValueError("Pixels must be integer x,y pairs")
        pixels.append(tuple(map(int, pieces)))
    with np.load(index_path.parent / record["file"], allow_pickle=False) as arrays:
        if "intrinsics" not in arrays:
            raise ValueError("This depth model did not output intrinsics; cannot unproject")
        depth = arrays["depth"].copy()
        if "valid_depth" in arrays:
            depth[~arrays["valid_depth"]] = np.nan
        points = camera_points(depth, arrays["intrinsics"], pixels)
        samples = [{"pixel_xy": list(pixel), "camera_xyz": point.tolist(),
                    "camera_z": float(point[2]), "camera_ray_distance": float(np.linalg.norm(point))}
                   for pixel, point in zip(pixels, points)]
        if "confidence" in arrays:
            for sample, (x, y) in zip(samples, pixels):
                score = float(arrays["confidence"][y, x])
                sample["model_confidence"] = score if np.isfinite(score) else None
    result = {"frame_index": args.frame, "time_seconds": record["time_seconds"],
              "units": record["depth_units"], "world_frame": record["world_frame"],
              "samples": samples,
              "pair_distances": [{"sample_a": i, "sample_b": j,
                                  "distance": float(np.linalg.norm(points[i] - points[j]))}
                                 for i in range(len(points)) for j in range(i + 1, len(points))],
              "interpretation": "Estimated distance between sampled visible surfaces, not object centers. Source depth is upsampled; inspect boundaries/confidence and validate scale against a known measurement."}
    if args.output:
        write_json(Path(args.output), result)
    print(json.dumps(result, indent=2, allow_nan=False))

