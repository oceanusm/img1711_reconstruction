# Compare DA3 estimates with Blender depth

`scene-recon compare-depth` compares depth from the original video's DA3 inference with the depth rendered from a Blender scene's own animated camera at the same video timestamps. It produces per-frame errors, validity coverage, projection diagnostics, and an offline HTML report. DA3 is a reference **estimate**, not measured ground truth.

The comparison does not use the VGGT similarity transform. Each image already describes distance from its own camera. Blender scene units are converted to meters, then depths are compared at corresponding source pixels. Camera pose, focal length, scene geometry, occlusion, and timing differences can all contribute to disagreement. VGGT alignment is a separate check of the 3D geometry and trajectory.

## Run the direct comparison first

First run DA3 over the full video and export Blender depth with `scene-recon blender-export --depth`; see [Blender export](blender_export.md). A Blender export can contain depth for a selected subset of timestamps while still exporting all camera poses. Only records with depth files are scored.

```bash
scene-recon compare-depth \
  --manifest data/frames/manifest.json \
  --depth-index outputs/depth/index.json \
  --blender-index outputs/blender/vqa_informed/index.json \
  --output outputs/comparison/vqa_informed_depth

scene-recon compare-depth \
  --manifest data/frames/manifest.json \
  --depth-index outputs/depth/index.json \
  --blender-index outputs/blender/before_vqa/index.json \
  --output outputs/comparison/before_vqa_depth
```

Open each output's `index.html` in a browser. It embeds its preview images and needs no network access. The PNG previews are also saved separately under `previews/`; `per_frame.csv` contains every compared timestamp, and `comparison.json` records provenance, choices, aggregates, and camera-intrinsic diagnostics. The NPZ arrays remain the authoritative numerical data. Output directories must be new or empty.

The default `--scale-mode metric` leaves Blender's physical scale intact. Meters per Blender unit come from the export's `unit_scale_meters`, based on the scene's declared units. Both supplied scenes declare one meter per Blender unit. A declaration is not proof that their modeled dimensions are physically correct. Use `--blender-unit-meters VALUE` only when you have a justified conversion; it is recorded as an override. Unknown scene units require an explicit value. DA3 must identify the selected depths as `estimated_meters`; uncalibrated relative-model units are rejected.

## Separate scale mismatch from shape mismatch

A second report can fit **one global positive scalar** to Blender depths after the declared unit conversion:

```bash
scene-recon compare-depth \
  --depth-index outputs/depth/index.json \
  --blender-index outputs/blender/vqa_informed/index.json \
  --output outputs/comparison/vqa_informed_depth_global \
  --scale-mode global --fit-frames 0,127
```

Choose frame IDs that exist in the Blender depth export. All remaining compared frames are held out. If only the four preview timestamps `0,127,380,697` were exported, this example fits on the first two and evaluates the last two separately. For a broader evaluation, deliberately select fitting timestamps that span the clip and leave independent timestamps for validation.

The scalar minimizes pooled squared error `sum((scale * Blender_meters - DA3_estimated_meters)^2)` over common valid pixels in the fitting frames. It is not a robust fit; large depths and large errors can dominate it. Every fitting pixel has equal weight. There are no independent frame, chunk, or object scale fits, and no depth offsets. The report retains the **direct** comparison alongside the **fitted** comparison, with separate fitting and held-out aggregates. `--fit-frames all` explicitly allows fitting on everything but yields no held-out validation. Global fitting requires `--fit-frames`; it never silently chooses the validation data.

A strong improvement after global scaling suggests a scale disagreement but does not establish which source has the correct physical scale. DA3 windows can also disagree in their estimated scale. The per-frame CSV records DA3 chunk IDs to make discontinuities easier to inspect.

## Pixels, depth conventions, and missing data

Both inputs must explicitly identify camera-space **z depth**. Ray distance is different away from the principal ray: `ray_distance = z * norm(inv(K) @ [u, v, 1])` when the unnormalized camera ray has z component 1. A ray-distance index is rejected rather than silently treated as z. The exporter validates its rendering convention; consult its metadata and [Blender export documentation](blender_export.md), rather than assuming every renderer's “Z” output has the same meaning.

Manifest SHA256 hashes, frame IDs, and timestamps must agree. DA3 must cover the complete canonical frame manifest. No nearest-time substitution is used. Blender source dimensions must match the canonical video. A smaller Blender render is supported only with explicit source dimensions and a full-frame pixel-center resize transform:

```text
x_export = sx * x_source + (sx - 1) / 2
y_export = sy * y_source + (sy - 1) / 2
```

DA3 is downsampled to that grid using validity-aware bilinear interpolation and nearest-pixel validity. Crops, unknown image warps, mismatched aspect ratios beyond integer render rounding, and upsampled Blender grids are rejected. Resampling smooths boundaries; run a higher-resolution comparison to investigate edge discrepancies. DA3 source-resolution maps themselves are resamples of a smaller model prediction, so higher output resolution does not create additional depth evidence.

Finite positive depths are required. Saved validity masks, sky masks, and nonfinite confidence values are excluded. Blender no-hit pixels are excluded by its validity mask. Optional `--confidence-percentile 50` keeps DA3 scores at or above each frame's median among otherwise valid source pixels. These scores are not calibrated probabilities; thresholds and retained coverage are reported. A percentile request fails if confidence is absent. `--max-depth 15` limits **DA3 reference** depths before resampling to 15 estimated meters; it does not clip large Blender errors or let them disappear from scoring.

Scores use the intersection of valid pixels. Each frame also reports DA3-valid, Blender-valid, common-valid, DA3-only, Blender-only, and neither-valid counts, plus the fraction of DA3-valid pixels covered by Blender. Missing modeled geometry can reduce this coverage while leaving apparently good errors on the surviving pixels. Empty intersections have null errors, never zero error.

## Read the metrics and images

For reference estimate `r` and Blender prediction `p`, the CSV contains:

| Metric | Definition | Better |
| --- | --- | --- |
| AbsRel | mean `abs(p-r)/r` | Lower |
| RMSE | square root of mean `(p-r)^2`, in estimated meters | Lower |
| MAE | mean `abs(p-r)`, in estimated meters | Lower |
| logRMSE | square root of mean `(ln(p)-ln(r))^2` | Lower |
| δ1, δ2, δ3 | Fraction with `max(p/r,r/p) < 1.25^n` | Higher |

Aggregates weight every valid pixel equally, including when combining frames. Their RMSE is computed from pooled squared errors, not an average of frame RMSEs. Repeated observations and neighboring video frames are correlated; a large pixel count is not a count of independent measurements.

Previews show original RGB, DA3 depth, Blender depth, and absolute difference. Their depth color limits are shared across both inputs, all sampled frames, and both comparison modes. Error colors also share one range. Limits use deterministic spatial samples from every compared frame, with depth 2nd–98th percentiles and absolute-error 98th percentile; colors clip at those endpoints. Black means invalid/no common data. These visual limits never clip the numeric metrics. `--preview-count 12` controls the number of evenly spaced preview timestamps.

When intrinsics are available, DA3's source-grid matrix is mapped to the Blender export grid and compared with the evaluated Blender camera's matrix. The report flags focal-length disagreement over 1% or principal-point disagreement over one comparison-grid pixel. These are diagnostic thresholds, not calibrated acceptance criteria. No reprojection is performed to conceal such a mismatch.

Run the same timestamps and filtering settings for both Blender baselines. Investigate regions with disagreement using the RGB image, depth validity, camera trajectory/FOV, and aligned VGGT geometry. Low depth error alone cannot prove a complete scene reconstruction, and high error alone cannot identify which estimate is wrong.
