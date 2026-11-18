# Aligning VGGT-Ω to either Blender baseline

The alignment tool fits one global similarity from VGGT's arbitrary world into a Blender export:

```text
p_blender = scale * rotation @ p_vggt + translation
```

The scale is positive and the rotation has determinant +1. Reflections are never accepted. The same transform applies to every point and camera center. Camera orientation receives only the rotation; scaling the camera's axes would corrupt its calibration. Both adapters already express camera axes as OpenCV x-right, y-down, z-forward, so no extra Blender camera-axis flip belongs in this step.

Run alignment separately for each baseline after the real `outputs/vggt` result is available:

```bash
scene-recon align-vggt \
  --vggt outputs/vggt \
  --blender-export outputs/blender_preview/vqa_informed \
  --output outputs/alignment/vqa_informed

scene-recon align-vggt \
  --vggt outputs/vggt \
  --blender-export outputs/blender_preview/before_vqa \
  --output outputs/alignment/before_vqa
```

These paths assume the preview exports created for this repository. All 698 timestamped Blender cameras are included even though preview RGB/depth was rendered only for selected frames. A low-resolution preview still carries full-source-grid intrinsics, so it is suitable for camera alignment.

## Fit frames and validation

The default fits corresponding camera centers on all exact common source frame IDs. The source-manifest SHA-256 must agree, frame IDs must be unique, and timestamps must agree within one microsecond. The tool does not guess correspondences by array position or choose the nearest timestamp. It records unmatched IDs. Intrinsics are compared in upright canonical source pixels, including when Blender images were rendered smaller.

Use a subset of the **actual VGGT-selected global frame IDs** to leave other frames for validation:

```bash
scene-recon align-vggt \
  --vggt outputs/vggt \
  --blender-export outputs/blender_preview/vqa_informed \
  --fit-frames 0,100,200,300,400,500,600,697 \
  --output outputs/alignment/vqa_informed_holdout
```

Choose frames distributed throughout the trajectory. A fit requires at least three non-collinear camera centers. A planar but non-collinear path is sufficient; a stationary, collinear, or nearly collinear path fails with an explanation instead of returning an arbitrary world rotation. The minimum second/first singular-value ratio is 0.001, and ratios below 0.02 produce a conditioning warning. These checks identify geometric degeneracy, not a guarantee that the motion is large compared with model error.

Robust fitting uses deterministic RANSAC followed by a least-squares Umeyama fit on its consensus. It requires a strict majority of the fit correspondences and at least three non-collinear inliers. By default, the inlier threshold is 10% of the target median distance from its coordinate-wise median; the actual value is saved in Blender-world units. `--robust-threshold 0.1` sets an explicit threshold in those units. `--no-robust` fits all selected correspondences and can help inspect a disagreement that has no majority consensus. Three-point fits cannot detect outliers.

Training residuals are conditional on the alignment fit, and robust outliers remain visible in the complete error table. An alignment with low training error is not independent proof that the scene matches. Held-out centers, camera orientations, intrinsics, static surfaces, and depth are complementary checks. Scale removed by the fit cannot establish physical distances. Compare the saved scales across baselines and validate physical dimensions separately.

## Static landmarks or a supplied transform

For weak camera motion, or to make the camera comparison independent of the fitting data, use corresponding **static visible surface points**. Object origins and mesh bounding-box centers generally do not correspond to VGGT surface samples. Prefer distributed cabinet/counter corners or other well-resolved features, avoid moving containers/hands and reflective surfaces, and reserve additional landmarks for validation.

The viewer's landmark download uses this format:

```json
{
  "schema_version": 1,
  "pairs": [
    {"name": "corner_a", "source": [0, 0, 0], "target": [4, 1, 2], "use_for_fit": true},
    {"name": "corner_b", "source": [1, 0, 0], "target": [6, 1, 2], "use_for_fit": true},
    {"name": "corner_c", "source": [0, 1, 0], "target": [4, 3, 2], "use_for_fit": true},
    {"name": "validation_corner", "source": [1, 1, 1], "target": [6, 3, 4], "use_for_fit": false}
  ]
}
```

Coordinates above are an illustrative synthetic example. `source` is the original unaligned VGGT world; `target` is Blender world. `use_for_fit` defaults to true. Aliases `vggt`, `blender`, and `fit` are also accepted. At least three non-collinear training pairs are required. With landmark fitting, all cameras are labeled `evaluation` and landmark holdouts are reported separately.

Viewer downloads also contain `source_manifest_sha256` and an `inputs` object with SHA-256 identifiers for both indexes, both camera archives, the Blender geometry, and the VGGT point cloud. Alignment validates every supplied identifier before fitting, so points from a different baseline or a changed reconstruction fail explicitly. Preserve these identifiers when editing downloaded correspondences. Handwritten files containing only `pairs` remain supported; optional provenance is checked whenever present.

```bash
scene-recon align-vggt \
  --vggt outputs/vggt \
  --blender-export outputs/blender_preview/vqa_informed \
  --landmarks landmarks_vqa_informed.json \
  --output outputs/alignment/vqa_informed_landmarks
```

Alternatively, `--transform manual_transform.json` accepts `{"scale": 1, "rotation": [[1,0,0],[0,1,0],[0,0,1]], "translation": [0,0,0]}` or an existing `alignment.json` containing a `transform` object. The identity shown is only a format example, not a valid assumed alignment. This option validates a supplied transform and evaluates it without refitting. Its source-file hash is recorded. It can support manual alignment when centers are degenerate. Do not combine `--fit-frames` with `--landmarks` or `--transform`.

## Output contract

The output directory must be new or empty to prevent stale mixed results.

- `alignment.json`: schema 1, completion status, source-manifest hash, hashes and locations of both input indexes and camera archives, fit method and inlier diagnostics, one transform and its inverse, all/fit/inlier/holdout/evaluation camera metrics, landmark errors, and interpretation warnings. Distances use Blender world units, not an inferred meter conversion. Positive signed intrinsic deltas mean VGGT estimates exceed Blender's values.
- `aligned_cameras.npz`: all VGGT `frame_indices`, `time_seconds`, rigid OpenCV-basis `camera_to_world` and `world_to_camera` in Blender world, unchanged `intrinsics_canonical`, and `matched_frame_indices`. `transform.matrix` is a point Sim(3) with scale; the pose rotations in this archive remain unit length.
- `camera_errors.csv`: one row per exact matched frame, fit/holdout/evaluation role, fit-inlier flag, both camera positions, center distance, orientation geodesic error in degrees, source-grid focal-length/principal-point differences and focal relative errors. A holdout is never used to refit the transform.

The viewer checks input hashes before overlaying the cloud and the Blender geometry. The helper `load_vggt_ply(path, max_points)` reads the exact binary little-endian schema produced by this repository and uses deterministic subsampling. It returns `points`, RGB `colors`, `confidence`, and `frame_indices`; the viewer should retain the unaligned points for landmark downloads.

No dense nearest-surface score is included at this stage: the VGGT cloud has duplicate observations, uncertain surfaces and moving content, and the exported Blender geometry is a timestamped snapshot. A single unfiltered nearest-neighbor number can hide missing geometry or reward unrelated nearby surfaces. Use the overlay, held-out landmarks and pixel-matched depth comparison first. Neither model disagreement nor agreement alone proves which reconstruction is correct.
