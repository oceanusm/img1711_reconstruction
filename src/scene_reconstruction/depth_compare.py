"""Compare timestamp-matched DA3 estimates and Blender camera-z renders.

No pose registration, image warping, or frame-wise scale adjustment is applied.
The comparison therefore measures the combined scene/camera/depth disagreement.
"""
from __future__ import annotations

import base64
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import load_manifest, sha256, write_json
from .depth_anything3 import resize_depth
from .preview import color_limits, colorize

METRICS = ("abs_rel", "rmse", "mae", "log_rmse", "delta1", "delta2", "delta3")
CAMERA_Z = {"camera_z", "camera-space z depth, not Euclidean ray distance"}


def depth_metrics(reference: np.ndarray, prediction: np.ndarray,
                  valid: np.ndarray | None = None) -> dict:
    """Metrics on common finite positive pixels; reference is an estimate, not GT."""
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    if reference.shape != prediction.shape:
        raise ValueError("Depth metric inputs must have identical shapes")
    good = np.isfinite(reference) & np.isfinite(prediction) & (reference > 0) & (prediction > 0)
    if valid is not None:
        if np.shape(valid) != reference.shape:
            raise ValueError("Metric validity mask has the wrong shape")
        good &= np.asarray(valid, bool)
    count = int(good.sum())
    if not count:
        return {"pixels": 0, **{key: None for key in METRICS}}
    ref, pred = reference[good].astype(np.float64), prediction[good].astype(np.float64)
    difference = pred - ref
    ratio = np.maximum(pred / ref, ref / pred)
    return {
        "pixels": count,
        "abs_rel": float(np.mean(np.abs(difference) / ref)),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "mae": float(np.mean(np.abs(difference))),
        "log_rmse": float(np.sqrt(np.mean((np.log(pred) - np.log(ref)) ** 2))),
        **{f"delta{i}": float(np.mean(ratio < 1.25 ** i)) for i in (1, 2, 3)},
    }


def fit_global_scale(pairs) -> tuple[float, int]:
    """One least-squares scalar: minimize sum((scale * Blender - DA3)^2)."""
    numerator, denominator, count = 0.0, 0.0, 0
    for reference, prediction, valid in pairs:
        if np.shape(reference) != np.shape(prediction) or np.shape(valid) != np.shape(reference):
            raise ValueError("Global scale inputs must have identical shapes")
        good = (np.asarray(valid, bool) & np.isfinite(reference) & np.isfinite(prediction)
                & (reference > 0) & (prediction > 0))
        ref, pred = np.asarray(reference)[good].astype(np.float64), np.asarray(prediction)[good].astype(np.float64)
        numerator += float(np.dot(ref, pred))
        denominator += float(np.dot(pred, pred))
        count += len(ref)
    if not count or denominator <= 0:
        raise ValueError("Scale-fit frames have no common valid positive depth pixels")
    scale = numerator / denominator
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Global scale fit is not finite and positive")
    return scale, count


def coverage(reference_valid: np.ndarray, blender_valid: np.ndarray) -> dict:
    if np.shape(reference_valid) != np.shape(blender_valid):
        raise ValueError("Validity masks must have identical shapes")
    ref, pred = np.asarray(reference_valid, bool), np.asarray(blender_valid, bool)
    total, common = int(ref.size), int((ref & pred).sum())
    return {
        "grid_pixels": total, "da3_valid_pixels": int(ref.sum()),
        "blender_valid_pixels": int(pred.sum()), "common_valid_pixels": common,
        "da3_only_pixels": int((ref & ~pred).sum()),
        "blender_only_pixels": int((~ref & pred).sum()), "neither_valid_pixels": int((~ref & ~pred).sum()),
        "common_fraction": common / total,
        "blender_coverage_of_da3": common / int(ref.sum()) if ref.any() else None,
    }


def _aggregate(rows: list[dict], prefix: str) -> dict:
    count = sum(row["common_valid_pixels"] for row in rows)
    result = {"frames": len(rows), "pixels": count}
    for key in METRICS:
        power = 2 if key in {"rmse", "log_rmse"} else 1
        value = sum(row["common_valid_pixels"] * row[f"{prefix}_{key}"] ** power
                    for row in rows if row["common_valid_pixels"])
        result[key] = (value / count) ** (1 / power) if count else None
    for key in ("grid_pixels", "da3_valid_pixels", "blender_valid_pixels", "common_valid_pixels",
                "da3_only_pixels", "blender_only_pixels", "neither_valid_pixels"):
        result[key] = sum(row[key] for row in rows)
    result["common_fraction"] = count / result["grid_pixels"] if result["grid_pixels"] else None
    result["blender_coverage_of_da3"] = count / result["da3_valid_pixels"] if result["da3_valid_pixels"] else None
    return result


def _records(index: dict, manifest: dict, label: str) -> dict[int, dict]:
    records = index.get("frames", [])
    ids = [int(record["index"]) for record in records]
    if not ids or ids != sorted(set(ids)):
        raise ValueError(f"{label} frame IDs must be nonempty, unique, and ordered")
    lookup = {int(record["index"]): record for record in manifest["frames"]}
    for record in records:
        idx, timestamp = int(record["index"]), float(record["time_seconds"])
        if idx not in lookup or not math.isfinite(timestamp) or abs(timestamp - lookup[idx]["time_seconds"]) > 1e-6:
            raise ValueError(f"{label} frame {idx} does not match canonical frame IDs/timestamps")
    return dict(zip(ids, records))


def _grid(index: dict, source_hw: tuple[int, int]) -> tuple[tuple[int, int], np.ndarray]:
    shape = (int(index["height"]), int(index["width"]))
    if min(shape) < 1:
        raise ValueError("Blender export dimensions must be positive")
    source = (int(index.get("source_height", source_hw[0])), int(index.get("source_width", source_hw[1])))
    if source != source_hw:
        raise ValueError("Blender export source dimensions differ from the canonical manifest")
    sy, sx = shape[0] / source_hw[0], shape[1] / source_hw[1]
    affine = np.asarray([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], float)
    if shape != source_hw:
        if "source_width" not in index or "source_height" not in index or "source_to_export_pixel_affine" not in index:
            raise ValueError("Resized Blender depth needs explicit source dimensions and full-frame pixel affine")
        # Integer render rounding can change the aspect by up to one output pixel.
        if abs(shape[1] - shape[0] * source_hw[1] / source_hw[0]) > 1.01:
            raise ValueError("Blender export aspect ratio differs from the full canonical frame")
        if sx > 1 or sy > 1:
            raise ValueError("Depth comparison supports native or downsampled Blender grids, not upsampling DA3")
    supplied = np.asarray(index.get("source_to_export_pixel_affine", affine), float)
    if supplied.shape != (3, 3) or not np.allclose(supplied, affine, atol=1e-6, rtol=1e-6):
        raise ValueError("Unsupported Blender crop/warp: expected a full-frame pixel-center resize affine")
    if index.get("crop") not in (None, False):
        raise ValueError("Cropped Blender exports are unsupported")
    return shape, affine


def _load_depth(path: Path, shape: tuple[int, int], percentile: float = 0,
                max_depth: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    with np.load(path, allow_pickle=False) as archive:
        depth = np.asarray(archive["depth"], dtype=np.float32)
        if depth.shape != shape:
            raise ValueError(f"Depth dimensions differ from indexed grid: {path}")
        valid = np.isfinite(depth) & (depth > 0)
        for key in ("valid_depth", "sky", "confidence"):
            if key in archive and archive[key].shape != shape:
                raise ValueError(f"{key} has incorrect dimensions: {path}")
        if "valid_depth" in archive:
            valid &= archive["valid_depth"].astype(bool)
        if "sky" in archive:
            valid &= ~archive["sky"].astype(bool)
        if max_depth is not None:
            valid &= depth <= max_depth
        before_confidence = int(valid.sum())
        threshold = None
        if "confidence" in archive:
            confidence = np.asarray(archive["confidence"], float)
            valid &= np.isfinite(confidence)
            if percentile and valid.any():
                threshold = float(np.percentile(confidence[valid], percentile))
                valid &= confidence >= threshold
        elif percentile:
            raise ValueError(f"A confidence percentile was requested but confidence is absent: {path}")
        intrinsics = np.asarray(archive["intrinsics"], float) if "intrinsics" in archive else None
        if intrinsics is not None and (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
                                      or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0):
            raise ValueError(f"Invalid camera intrinsics: {path}")
    return depth, valid, intrinsics, {"source_valid_before_confidence": before_confidence,
                                    "source_valid_after_confidence": int(valid.sum()),
                                    "confidence_threshold": threshold}


def _pair(context: dict, frame: int):
    ref_record, blender_record = context["da3"][frame], context["blender"][frame]
    ref, ref_valid, ref_k, extra = _load_depth(
        context["depth_root"] / ref_record["file"], context["source_hw"], context["percentile"], context["max_depth"])
    if context["shape"] != context["source_hw"]:
        ref, ref_valid = resize_depth(np.where(ref_valid, ref, np.nan), context["shape"])
    pred, pred_valid, pred_k, _ = _load_depth(context["blender_root"] / blender_record["file"], context["shape"])
    pred = pred.astype(np.float64) * context["unit_scale"]
    diagnostic = {"intrinsics_available": ref_k is not None and pred_k is not None}
    if diagnostic["intrinsics_available"]:
        ref_k = context["affine"] @ ref_k
        diagnostic.update({
            "focal_x_relative_difference": float((pred_k[0, 0] - ref_k[0, 0]) / ref_k[0, 0]),
            "focal_y_relative_difference": float((pred_k[1, 1] - ref_k[1, 1]) / ref_k[1, 1]),
            "principal_point_difference_pixels": float(np.linalg.norm(pred_k[:2, 2] - ref_k[:2, 2])),
            "da3_intrinsics_grid": ref_k.tolist(), "blender_intrinsics_grid": pred_k.tolist(),
        })
    return ref, pred, ref_valid, pred_valid, extra, diagnostic


def _fit_indices(value: str | None, ids: list[int], mode: str) -> set[int]:
    if mode == "metric":
        if value:
            raise ValueError("--fit-frames requires --scale-mode global")
        return set()
    if not value:
        raise ValueError("Global scale fitting requires explicit --fit-frames IDs (or 'all'); remaining frames are held out")
    if value == "all":
        return set(ids)
    try:
        parsed = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("--fit-frames must be comma-separated canonical frame IDs or 'all'") from exc
    if len(parsed) != len(set(parsed)) or not set(parsed).issubset(ids):
        raise ValueError("Fit frame IDs must be unique and have both DA3 and Blender depth")
    return set(parsed)


def _error_colors(error: np.ndarray, valid: np.ndarray, high: float) -> np.ndarray:
    palette = np.asarray([[20, 25, 35], [190, 50, 65], [255, 210, 100]], float)
    x = np.clip(np.where(valid, error, 0) / high, 0, 1)
    colors = np.stack([np.interp(x, [0, .5, 1], palette[:, c]) for c in range(3)], axis=-1)
    colors[~valid] = 0
    return colors.astype(np.uint8)


def _montage(context: dict, frame: int, scale: float, label: str, limits: dict, destination: Path):
    ref, pred, rv, pv, _, _ = _pair(context, frame)
    pred *= scale
    common = rv & pv
    low, high, error_high = limits["near"], limits["far"], limits["absolute_error_max"]
    size = (max(1, round(context["shape"][1] * min(1, 420 / context["shape"][0]))),
            min(context["shape"][0], 420))
    width = max(215, size[0])
    canvas = Image.new("RGB", (4 * width, size[1] + 136), "#151923")
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    record = context["manifest_frames"][frame]
    draw.text((8, 7), f"Frame {frame:06d} | {record['time_seconds']:.3f} s | {label} | common valid {common.mean():.1%}", font=font, fill="white")
    with Image.open(context["manifest_root"] / record["file"]) as image:
        rgb = image.convert("RGB")
    panels = [rgb, Image.fromarray(colorize(ref, rv, low, high)),
              Image.fromarray(colorize(pred, pv, low, high)),
              Image.fromarray(_error_colors(np.abs(pred - ref), common, error_high))]
    labels = ["Original video", "DA3 reference estimate", "Blender camera-z depth", "Absolute depth difference"]
    for i, (panel, title) in enumerate(zip(panels, labels)):
        draw.text((i * width + 8, 29), title, font=font, fill="white")
        canvas.paste(panel.resize(size, Image.Resampling.NEAREST if i else Image.Resampling.LANCZOS),
                     (i * width + (width - size[0]) // 2, 50))
    footer = size[1] + 58
    gradient = np.repeat(np.linspace(low, high, 2 * width - 16)[None], 12, axis=0)
    canvas.paste(Image.fromarray(colorize(gradient, np.ones_like(gradient, bool), low, high)), (8, footer))
    err_gradient = np.repeat(np.linspace(0, error_high, 2 * width - 16)[None], 12, axis=0)
    canvas.paste(Image.fromarray(_error_colors(err_gradient, np.ones_like(err_gradient, bool), error_high)), (2 * width + 8, footer))
    draw.text((8, footer + 17), f"Depth {low:.3g} to {high:.3g} {limits['units']}", font=font, fill="white")
    draw.text((2 * width + 8, footer + 17), f"Absolute error 0 to {error_high:.3g} {limits['units']}", font=font, fill="white")
    draw.text((8, footer + 39), "Black = invalid / no common data. Fixed ranges across all frames and modes; endpoints clip.", font=font, fill="#f4cc80")
    draw.text((8, footer + 57), "Depth estimates and camera mismatch can both cause disagreement. Source NPZ files are authoritative.", font=font, fill="#b8c0cf")
    canvas.save(destination)


def _html(report: dict, rows: list[dict], output: Path):
    def fmt(value):
        return "n/a" if value is None else f"{value:.5g}" if isinstance(value, (float, int)) else html.escape(str(value))
    warnings = "".join(f"<li>{html.escape(item)}</li>" for item in report["interpretation"])
    summaries = []
    for mode, splits in report["metrics"].items():
        for split, values in splits.items():
            if values["frames"]:
                summaries.append("<tr>" + "".join(f"<td>{fmt(value)}</td>" for value in
                    [mode, split, values["frames"], values["pixels"], values["common_fraction"],
                     *[values[key] for key in METRICS]]) + "</tr>")
    previews = []
    for entry in report["previews"]:
        data = base64.b64encode((output / entry["file"]).read_bytes()).decode("ascii")
        previews.append(f'<figure data-frame="{entry["frame_index"]}"><figcaption>{html.escape(entry["label"])}</figcaption>'
                        f'<img alt="Depth comparison frame {entry["frame_index"]}" src="data:image/png;base64,{data}"></figure>')
    options = ''.join(f'<option value="{i}">Frame {i} at {report["frame_times"][str(i)]:.3f} s</option>'
                      for i in sorted({entry["frame_index"] for entry in report["previews"]}))
    row_html = []
    for row in rows:
        row_html.append("<tr>" + "".join(f"<td>{fmt(row[key])}</td>" for key in
            ["frame_index", "time_seconds", "split", "chunk_id", "common_fraction", "blender_coverage_of_da3",
             "direct_abs_rel", "direct_rmse", "direct_mae", "direct_log_rmse", "direct_delta1"]
            + (["fitted_abs_rel", "fitted_rmse"] if report["scale_mode"] == "global" else [])) + "</tr>")
    extra_headers = "<th>Fitted AbsRel</th><th>Fitted RMSE</th>" if report["scale_mode"] == "global" else ""
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DA3 / Blender depth comparison</title><style>
body{{font:16px system-ui,sans-serif;background:#111620;color:#e4e9f1;margin:28px auto;max-width:1440px;padding:0 20px;line-height:1.5}}
h1,h2{{line-height:1.2}}a{{color:#94c4ff}}table{{border-collapse:collapse;font-size:13px;width:100%}}th,td{{border-bottom:1px solid #394352;padding:7px;text-align:right;white-space:nowrap}}
th{{background:#222d3c}}figure{{margin:20px 0}}img{{max-width:100%;height:auto}}.scroll{{overflow:auto}}li{{margin:5px 0}}code,select{{background:#222d3c;color:#e4e9f1;padding:4px}}.muted{{color:#bbc6d6}}
</style><h1>DA3 / Blender depth comparison</h1>
<p>{len(rows)} matched frames. DA3 is a reference estimate, not measured ground truth. Both arrays use camera-space z depth.</p>
<p>Blender unit conversion: {report['blender_unit_meters']:.6g} m/unit. Mode: <strong>{report['scale_mode']}</strong>.
Additional fitted scalar: {fmt(report.get('global_scale'))}. Comparison grid: {report['grid_hw'][1]} × {report['grid_hw'][0]}.</p>
<ul>{warnings}</ul><p><a href="comparison.json">Full provenance and diagnostics</a> · <a href="per_frame.csv">Per-frame CSV</a></p>
<h2>Pixel-weighted metrics</h2><p class="muted">AbsRel = mean |Blender − DA3| / DA3; RMSE/MAE use {html.escape(report['comparison_units'])}; logRMSE uses natural logarithms.
δ1/2/3 count ratios below 1.25, 1.25², 1.25³. Smaller errors and larger δ are better. Empty intersections have no score.</p>
<div class="scroll"><table><thead><tr><th>Mode</th><th>Split</th><th>Frames</th><th>Pixels</th><th>Common fraction</th>{''.join('<th>'+x+'</th>' for x in METRICS)}</tr></thead><tbody>{''.join(summaries)}</tbody></table></div>
<h2>Fixed-scale previews</h2><p>DA3 and Blender share one depth range. All differences share one error range. Colors clip at the displayed endpoints.</p>
<label>Preview frame <select id="frame"><option value="all">All preview frames</option>{options}</select></label>{''.join(previews)}
<h2>Per-frame direct comparison</h2><div class="scroll"><table><thead><tr><th>Frame</th><th>Time (s)</th><th>Split</th><th>DA3 chunk</th><th>Common fraction</th><th>Blender / DA3 coverage</th><th>AbsRel</th><th>RMSE</th><th>MAE</th><th>logRMSE</th><th>δ1</th>{extra_headers}</tr></thead><tbody>{''.join(row_html)}</tbody></table></div>
<script>document.getElementById('frame').addEventListener('change',e=>{{document.querySelectorAll('figure').forEach(x=>x.hidden=e.target.value!=='all'&&x.dataset.frame!==e.target.value)}});</script></html>'''
    (output / "index.html").write_text(content)


def run(args: Any) -> Path:
    manifest_path = Path(args.manifest).expanduser().resolve()
    depth_path = Path(args.depth_index).expanduser().resolve()
    blender_path = Path(args.blender_index).expanduser().resolve()
    if blender_path.is_dir():
        blender_path /= "index.json"
    if depth_path.is_dir():
        depth_path /= "index.json"
    output = Path(args.output).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Choose a new or empty depth-comparison directory: {output}")
    manifest = load_manifest(manifest_path)
    manifest_hash = sha256(manifest_path)
    da3_index, blender_index = json.loads(depth_path.read_text()), json.loads(blender_path.read_text())
    for label, index in (("DA3", da3_index), ("Blender", blender_index)):
        if index.get("status") != "complete":
            raise ValueError(f"{label} index must be complete")
        if index.get("manifest_sha256") != manifest_hash:
            raise ValueError(f"{label} index and canonical manifest SHA256 must match exactly")
        if index.get("depth_convention") not in CAMERA_Z:
            raise ValueError(f"{label} depth convention must explicitly be camera-z; ray distance cannot be compared directly")
    mode = str(args.scale_mode)
    if mode not in {"metric", "global"}:
        raise ValueError("Scale mode must be metric or global")
    percentile = float(args.confidence_percentile)
    if not math.isfinite(percentile) or not 0 <= percentile < 100:
        raise ValueError("Confidence percentile must be finite and in [0,100)")
    max_depth = None if args.max_depth is None else float(args.max_depth)
    if max_depth is not None and (not math.isfinite(max_depth) or max_depth <= 0):
        raise ValueError("Maximum reference depth must be finite and positive")
    preview_count = int(args.preview_count)
    if preview_count < 1:
        raise ValueError("Preview count must be positive")
    unit_scale = args.blender_unit_meters
    unit_source = "command_line_override" if unit_scale is not None else "blender_export.unit_scale_meters"
    if unit_scale is None:
        unit_scale = blender_index.get("unit_scale_meters")
    if unit_scale is None or not math.isfinite(float(unit_scale)) or float(unit_scale) <= 0:
        raise ValueError("Blender unit scale is unknown; provide positive --blender-unit-meters explicitly")
    if blender_index.get("depth_units") != "blender_units":
        raise ValueError("Expected Blender depth in blender_units before physical unit conversion")
    da3, blender = _records(da3_index, manifest, "DA3"), _records(blender_index, manifest, "Blender")
    if list(da3) != [frame["index"] for frame in manifest["frames"]]:
        raise ValueError("The complete DA3 index must cover every canonical frame")
    ids = [idx for idx, frame in blender.items() if frame.get("file")]
    if not ids:
        raise ValueError("Blender index contains no rendered depth frames; export with --depth")
    units = {da3[idx].get("depth_units") for idx in ids}
    if units != {"estimated_meters"}:
        raise ValueError("DA3 comparison requires estimated_meters on every selected frame; relative depth needs explicit external metric calibration")
    source_hw = (int(manifest["height"]), int(manifest["width"]))
    if "source_hw" in da3_index and tuple(da3_index["source_hw"]) != source_hw:
        raise ValueError("DA3 source dimensions disagree with manifest")
    shape, affine = _grid(blender_index, source_hw)
    fit_ids = _fit_indices(args.fit_frames, ids, mode)
    context = {"da3": da3, "blender": blender, "depth_root": depth_path.parent,
               "blender_root": blender_path.parent, "source_hw": source_hw, "shape": shape,
               "affine": affine, "unit_scale": float(unit_scale), "percentile": percentile,
               "max_depth": max_depth, "manifest_frames": {f["index"]: f for f in manifest["frames"]},
               "manifest_root": manifest_path.parent}
    scale, fit_pixels = None, 0
    if mode == "global":
        def pairs():
            for idx in sorted(fit_ids):
                ref, pred, rv, pv, _, _ = _pair(context, idx)
                yield ref, pred, rv & pv
        scale, fit_pixels = fit_global_scale(pairs())
    rows, diagnostics, samples, error_samples = [], [], [], []
    stride = max(1, math.ceil(math.sqrt(np.prod(shape) / 4096)))
    for position, idx in enumerate(ids):
        ref, pred, rv, pv, extra, diagnostic = _pair(context, idx)
        valid = rv & pv
        row = {"frame_index": idx, "time_seconds": da3[idx]["time_seconds"],
               "split": ("fit" if idx in fit_ids else "holdout") if mode == "global" else "unfitted",
               "chunk_id": da3[idx].get("chunk_id"), **coverage(rv, pv), **extra}
        for label, factor in (("direct", 1), *(([("fitted", scale)] if scale is not None else []))):
            metrics = depth_metrics(ref, pred * factor, valid)
            row.update({f"{label}_{key}": metrics[key] for key in METRICS})
            sampled_pred = pred[::stride, ::stride] * factor
            samples.append(sampled_pred[pv[::stride, ::stride]])
            error_samples.append(np.abs(sampled_pred - ref[::stride, ::stride])[valid[::stride, ::stride]])
        samples.append(ref[::stride, ::stride][rv[::stride, ::stride]])
        rows.append(row)
        diagnostics.append({"frame_index": idx, **diagnostic})
        if position % 100 == 0:
            print(f"Depth comparison: {position + 1}/{len(ids)} frames", flush=True)
    if any(len(sample) for sample in samples):
        low, high = color_limits(samples)
    else:
        # A fully invalid pair still has useful coverage information and must not
        # become a fabricated zero-error score or fail to produce a report.
        low, high = 0.0, 1.0
    errors = np.concatenate(error_samples)
    error_high = max(float(np.percentile(errors, 98)), 1e-6) if len(errors) else 1.0
    limits = {"near": low, "far": high, "absolute_error_max": error_high, "units": "estimated m",
              "percentiles": {"depth": [2, 98], "absolute_error": 98}, "spatial_stride": stride,
              "sampling": "all compared frames; both direct and globally fitted modes when enabled",
              "invalid_color": "black", "range_policy": "fixed across all previews; clipped endpoints",
              "depth_range_has_samples": any(len(sample) for sample in samples),
              "error_range_has_samples": bool(len(errors))}
    warnings = [
        "DA3 depth is inferred, not measured ground truth; reflective surfaces, motion, and occlusions can be wrong.",
        "This pairs corresponding source pixels without image warping. Camera pose, lens/FOV, timing, and geometry differences all affect the errors.",
        "DA3 chunks can have scale disagreement. No per-frame or per-chunk scale correction is applied.",
        "Metrics exclude pixels invalid in either source. Inspect validity coverage: missing Blender geometry must not be judged only by errors on surviving pixels.",
    ]
    mismatch = [d["frame_index"] for d in diagnostics if d["intrinsics_available"] and (
        abs(d["focal_x_relative_difference"]) > .01 or abs(d["focal_y_relative_difference"]) > .01
        or d["principal_point_difference_pixels"] > 1)]
    if mismatch:
        warnings.append(f"{len(mismatch)} frame(s) have >1% focal-length or >1 pixel principal-point disagreement on the comparison grid; depth errors include this projection mismatch.")
    if any(not d["intrinsics_available"] for d in diagnostics):
        warnings.append("Some frames lack intrinsics; their projection mismatch could not be diagnosed.")
    if shape != source_hw:
        warnings.append("DA3 is resampled to the smaller Blender grid with validity-aware bilinear interpolation and nearest-pixel validity. This smooths depth edges and does not add geometric detail.")
    if mode == "global":
        warnings.append("A single scalar is fitted to Blender depth after scene-unit conversion. The fitted score removes one scale error; the direct score preserves it. This is not a 3D scene registration.")
        if fit_ids == set(ids):
            warnings.append("All compared frames were used for scale fitting; there is no held-out validation.")
    report = {
        "schema_version": 1, "status": "complete", "stage": "da3_blender_depth_comparison",
        "source_manifest": str(manifest_path), "manifest_sha256": manifest_hash,
        "da3_index": str(depth_path), "da3_index_sha256": sha256(depth_path),
        "blender_index": str(blender_path), "blender_index_sha256": sha256(blender_path),
        "source_blend_sha256": blender_index.get("source_blend_sha256"),
        "depth_convention": "camera_z", "comparison_units": "estimated meters",
        "blender_unit_meters": float(unit_scale), "blender_unit_scale_source": unit_source,
        "scale_mode": mode, "global_scale": scale, "fit_frame_indices": sorted(fit_ids),
        "holdout_frame_indices": [i for i in ids if i not in fit_ids] if mode == "global" else [],
        "fit_pixels": fit_pixels, "scale_fit_objective": "one positive scalar minimizing pooled squared depth error on fit frames" if scale else None,
        "frame_indices": ids, "frame_times": {str(i): da3[i]["time_seconds"] for i in ids},
        "unrendered_blender_frame_indices": [i for i in blender if i not in ids],
        "source_hw": list(source_hw), "grid_hw": list(shape), "source_to_grid_pixel_affine": affine.tolist(),
        "resampling": "validity-aware bilinear DA3 depth; nearest validity; full-frame pixel-center affine; no crop or pose warp",
        "confidence_percentile_per_frame": percentile, "confidence_note": "raw uncalibrated DA3 score; nonfinite scores excluded when present",
        "max_da3_reference_depth": max_depth, "max_depth_note": "applied only to DA3 reference before resampling; Blender residuals are not clipped",
        "aggregation": "pixel-weighted across each split; invalid intersections have null errors",
        "metrics": {label: {"all": _aggregate(rows, label),
                    **({split: _aggregate([row for row in rows if row["split"] == split], label)
                        for split in ("fit", "holdout")} if mode == "global" else {})}
                    for label in (["direct", "fitted"] if scale else ["direct"])},
        "intrinsics_diagnostics": diagnostics, "projection_mismatch_frame_indices": mismatch,
        "color_scale": limits, "interpretation": warnings, "previews": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "previews").mkdir(exist_ok=True)
    with (output / "per_frame.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for position in sorted(set(np.linspace(0, len(ids) - 1, min(preview_count, len(ids))).round().astype(int))):
        idx = ids[position]
        for label, factor in (("direct", 1), *(([("fitted", scale)] if scale is not None else []))):
            path = f"previews/{idx:06d}_{label}.png"
            description = f"{label}: frame {idx}, {da3[idx]['time_seconds']:.3f} s"
            _montage(context, idx, factor, description, limits, output / path)
            report["previews"].append({"frame_index": idx, "label": description, "file": path})
    write_json(output / "comparison.json", report)
    _html(report, rows, output)
    print(f"Depth comparison report: {output / 'index.html'}", flush=True)
    return output / "comparison.json"
