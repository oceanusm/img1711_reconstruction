# Automatic surface alignment

`align-surfaces` fits one global positive-scale similarity from VGGT into each Blender baseline using structural surfaces. It identifies candidate planes in the point cloud, matches them to named Blender mesh surfaces, and applies the resulting rotation, translation and uniform scale to the entire cloud and camera trajectory.

**Camera correspondences do not enter detection, initialization, hypothesis ranking or refinement.** Camera bundles are validated for provenance and timestamp matching, then used to evaluate the fitted transform. VGGT's cloud itself was reconstructed using its predicted cameras, so this is independence from the alignment objective, not statistical independence of camera and geometry predictions.

The current implementation has automatic named profiles for the two original IMG_1711 kitchen scenes, the [VGGT-guided IMG_1711 scene](vggt_guided_1711.md), and the IMG_1696 living/dining scene in this repository. It is not a general semantic scene recognizer. It requires no manual surface labels for these scenes, but an unfamiliar scene or object naming scheme may need a different target profile or explicit correspondences.

## Run it

From the repository root, with the existing CPU NumPy/Pillow environment:

```bash
PYTHONPATH=src python3 -m scene_reconstruction align-surfaces \
  --vggt outputs/vggt_quality \
  --blender-export outputs/blender_preview/before_vqa \
  --output outputs/comparison_surfaces/before_vqa/automatic

PYTHONPATH=src python3 -m scene_reconstruction align-surfaces \
  --vggt outputs/vggt_quality \
  --blender-export outputs/blender_preview/vqa_informed \
  --output outputs/comparison_surfaces/vqa_informed/automatic
```

These are the paths used for the generated local results. **The output directory must be new or empty**; choose a fresh output name to rerun without mixing artifacts. Original `.blend` files, raw VGGT files, and earlier camera-based alignments are preserved.

No GPU or new model inference is needed. The existing point cloud, camera/index bundles, Blender geometry, and canonical frame manifest are sufficient. VGGT's missing `depth/` and `rgb_model/` files do not block this workflow. The viewer uses canonical source images for its video thumbnails; completing the transfer is still useful for depth comparisons and pixel-based inspection.

Defaults:

| Option | Default | Purpose |
|---|---:|---|
| `--surface-profile` | `auto` | Select exactly one complete named profile; explicit alternatives are `kitchen` and `living_dining` |
| `--sample-points` | 300,000 | Deterministic PLY sample before spatial balancing and plane detection |
| `--voxel-size` | Automatic | One source-point representative per spatial cell; units are VGGT units |
| `--plane-threshold` | Automatic | Source-plane RANSAC distance tolerance, in VGGT units |
| `--max-planes` | 18 | Maximum source planes before fragment consolidation |
| `--seed` | 0 | Reproducible voxel representatives and RANSAC sampling |
| `--stability-runs` | 2 | Total detector-seed runs, including the initial run; allowed range 1–8 |
| `--viewer-points` | 5,000,000 | Point budget for the separate full 3D viewer |

Use `--viewer-points 0` to skip full-viewer generation while retaining fitting results and the diagnostic report. Changing the display budget does not change the fitted transform. The fit sample reduces duplicate observations and computation; it does not change VGGT's learned geometry or remove points from the original PLY.

`--stability-runs 1` skips the independent detector-seed repeat. Its result is **not stability-validated**, even if the single-run geometric checks produce `automatic_candidate`; the report records the unchecked status and warning. The default two runs use `--seed` and `--seed + 1` on the same sampled cloud.

## The four target surfaces

The target extractor reads actual indexed triangles and object names from `geometry.npz`; it does not substitute object bounds for face planes.

| Anchor | Before-VQA objects | VQA-informed objects |
|---|---|---|
| Floor | `Oak flooring …` | `Floor plank …` |
| Main countertop | `Left/Right of range stone counter`; `Window return stone slab with sink opening` | `L-shaped countertop …`; `Countertop \| right of range` |
| Back surface | `Range wall stone backsplash` | `Stove backsplash …`; `Range backsplash …` |
| Side surface | `Return wall backsplash …`; `Return wall stone below window` | `Sink backsplash …` |

The VGGT-guided IMG_1711 scene adds exact kitchen aliases: `Pale oak plank flooring`, `L-shaped sink and preparation worktop`, `Spice counter slab`, `Main stone backsplash`, and `Sink wall backsplash`. Actual coplanar face selection remains unchanged; its spice slab is slightly lower than the main worktop and is not included in that final anchor. [Prepared overlay and results](vggt_guided_1711.md).

For IMG_1696, `living_dining` uses `Oak_plank_…` floor tops, `Dining_oak_slab_top`, `Art_and_dining_wall`, and the opaque `Window_wall_pier_…`, `Window_wall_lintel`, and `Window_wall_base` faces. It excludes floor substrate, glass, blinds and window openings. The dining-table-to-floor separation is about 0.7855 Blender units. Coffee-table and console tops are exported as additional inspection surfaces, not used as anchors. The legacy internal role `counter` means the raised horizontal anchor; reports show its scene-specific name.

`auto` examines object names and requires exactly one profile with all four anchor groups. Missing or ambiguous names cause an error; no camera-derived fallback is used. Actual face directions and finite extents still come from mesh triangles.

Dominant triangle normals establish surface directions. The counter's position relative to the floor establishes upward orientation. The floor footprint establishes the room-facing side of vertical surfaces. Triangle winding signs are ignored. Substantial outer coplanar faces are selected, excluding slab backs, thickness, bevels and tiny slivers. The island top is excluded from these anchor identities and exported separately as a potential inspection surface.

A floor alone leaves scale, yaw and two horizontal translations unconstrained. A parallel countertop at a different height establishes scale; two nonparallel vertical surfaces establish heading and the remaining position. The actual target floor-to-counter separation is about 0.9100 Blender units before VQA and 0.9144 units in the informed scene. Those dimensions are calibration inputs, not independent physical measurements.

## What the automatic fit does

1. **Balance and detect source planes.** Sample the PLY, select one representative per occupied voxel, and run deterministic sequential RANSAC with orthogonal least-squares plane refits. Normal signs are canonicalized only for reproducibility. Narrow or degenerate patches are rejected. Automatic voxel size is the robust cloud extent divided by 220; the initial plane-distance threshold is 0.002 times that extent. Detection is capped at 60,000 balanced points.
2. **Consolidate fragments.** Sequential RANSAC can split one physical surface into neighboring patches. Merge candidates must have normal angles below 3°, a nearby member within three plane thresholds, and no member beyond six thresholds. The merged patch must retain RMS residual below two thresholds and 95th-percentile residual below four thresholds. This prevents unrestricted chains from merging distinct layers.
3. **Enumerate geometric assignments.** Consider source parallel-plane pairs as floor/counter candidates and nearly perpendicular planes as back/side candidates. Enumerate normal signs, require a proper rotation and positive scale, align normals, and solve plane-offset equations for scale and translation. No previous camera-derived transform supplies an initial guess.
4. **Rank and refine candidates.** Rank hypotheses using supported source patches on target faces, plane residuals and the fraction of cloud on the room side of the structural planes. Diversify up to 32 finalists by their resulting spatial displacement. Plane identities propose the initial placement, but final refinement pools points from nearby parallel detections rather than retaining one arbitrary RANSAC fragment. Candidate planes must agree with the transformed target normal within 8°; their points must lie near the target plane and within its actual finite faces, with a small edge tolerance. Refine using a balanced Huber point-to-plane objective, re-selecting pooled regions five times. Each anchor contributes equal total weight; optimization uses at most 2,000 points per anchor.
5. **Check and repeat.** Test rank, conditioning, spatial/temporal support, residuals and competing assignments. By default, repeat the entire detection, merging, matching and refinement procedure with the next detector seed. Compare its scale, rotation and transformed-cloud displacement against the initial solution. An unstable repeat or a repeat that fails geometric checks marks the result `needs_review`; it does not silently average the two transforms.
6. **Evaluate cameras.** Freeze the initial surface-derived transform, apply it to all points and camera poses, and evaluate all exact timestamp-matched cameras. For IMG_1711, all 350 camera pairs have role `evaluation`; for IMG_1696, all 450 pairs do.

For an anchor with Blender plane equation `n · x = d`, the fitted residual is:

```text
error = n · (scale * rotation @ source_point + translation) - d
```

Support is one-sided: observed source points must be supported by the corresponding target face, but unseen Blender floor/wall extensions are not counted as missing reconstruction. A source plane can span coplanar regions outside a named target face; those regions are excluded from that anchor's refinement rather than forcing every point onto the limited face.

With `h` denoting the target floor-to-counter height, the pooling distance gate is the larger of `0.05 h` and six times the source plane threshold after applying the candidate scale. The finite-face edge tolerance is `0.04 h`. Pooling reduces sensitivity to how RANSAC divides one noisy surface; it does not prove that nearby parallel points share the intended semantic identity. `pooled_source_plane_indices` and `raw_source_plane_indices` preserve which detections contributed.

## Acceptance and limitations

`surface_fit.json` distinguishes `automatic_candidate` from `needs_review`. The former means the geometric hypothesis passed the implemented checks; it is not a calibrated confidence score or confirmation of semantic correctness. A rank-deficient system or lack of any usable candidate raises an error rather than silently falling back to camera alignment.

The main checks are expressed relative to target floor-to-counter height `h`:

| Check | Current criterion |
|---|---|
| Similarity observability | Seven-parameter Jacobian has rank 7; normalized condition above 100 flags review |
| Room-side support | At least 90% of sampled cloud inside each floor/back/side half-space |
| Finite patch evidence | At least 30 points to refine each anchor; finite-support score at least 0.8 |
| Patch shape | Both 5th–95th percentile PCA tangent widths at least `0.05 h`; their product at least `0.01 h²` |
| Temporal support | At least three distinct source-frame IDs where provenance is available |
| Anchor residual | RMS at most `0.025 h` and 95th percentile at most `0.05 h` |
| Ambiguity | A materially different eligible finalist, displaced by more than `0.08 h`, must trail by at least 0.12 score units |
| Detector-seed stability | Every repeat is `automatic_candidate`, scale changes by at most 1%, rotation by at most 1°, and transformed-cloud RMS displacement by at most `0.03 h` |

The finite-support score combines supported sample count and PCA footprint area; it is a heuristic ranking quantity, not a probability. These thresholds are engineering heuristics, not validated accuracy guarantees. `distinct_runner_up_margin: null` means no materially different eligible competitor remained among the searched finalists, not proof of a globally unique solution. The selected source patches and competing candidates remain inspectable.

Detector-seed stability checks sensitivity to stochastic extraction and voxel representatives on the same model result. It is not a separate video reconstruction, a confidence interval, or evidence against shared model bias. Repeat transforms, differences, thresholds and pass/fail outcomes are saved under `diagnostics.stability` in `surface_fit.json`.

`transform_stable` records the numerical scale/rotation/displacement checks separately from `repeat_quality_checks_passed`. Overall `passed` still requires both. Thus a consistently recovered alignment can remain `needs_review` because the same anchor residual fails on each repeat, as in the [IMG_1696 result](img1696.md#automatic-surface-alignment). Repeat records also identify the initial proposal and final pooled source-plane families; proposal IDs need not equal the final support after reassociation.

Anchor residuals measure fit quality on calibration surfaces. Floor-to-counter height cannot then be presented as independent validation. Other cloud points influence the room-interior hypothesis test, so even objects excluded from the point-to-plane objective are not strictly held-out evidence for the entire selection procedure. Camera-center correspondences are fully excluded from that procedure. A good surface fit or disagreement with a camera-derived fit does not establish which model is physically correct.

All distances are **Blender units**. Neither the uniform scale fit nor Blender's declared meter setting establishes physical dimensions. Reflections, separate object transforms, per-frame transforms and nonrigid deformation are not permitted.

## Artifacts and inspection

| Artifact | Contents |
|---|---|
| `index.html` | Offline diagnostic report, colored anchor-patch projections, source/target geometry overlays and fit/camera summaries |
| `viewer.html` | Full offline interactive cloud/mesh/camera viewer, when requested; default displays all five million PLY points |
| `surface_fit.json` | Selected transform, status, anchor identities, pooled detections, residuals, support, thresholds, competing hypotheses and detector-seed stability results |
| `target_planes.json` | Actual target triangles, names, offsets, normals and selected-profile limitations |
| `source_planes.json` | Detected/consolidated plane metadata and sampling diagnostics |
| `surface_points.npz` | Balanced source points, anchor labels, source frame IDs and per-plane point indices |
| `alignment.json` | Common alignment artifact with provenance hashes, `automatic_surfaces` fit method and camera evaluation metrics |
| `aligned_cameras.npz`, `camera_errors.csv` | Transformed rigid camera poses and timestamp-matched position/orientation/intrinsics errors |

The full viewer applies the same similarity to raw cloud coordinates during display. It does not refit the transform or rewrite the source PLY. The diagnostic report uses a smaller sample for responsiveness; its sampling does not imply that the reconstructed cloud contains only those displayed points.

The final local runs under `outputs/comparison_surfaces/{before_vqa,vqa_informed}/automatic/` detected 18 source planes from 26,681 spatially balanced points in the initial seed and consolidated them into 14 planes. Both selected the same structural families: floor, pooled countertop fragments, back-surface fragments, and side-surface fragments. Both passed the current geometric gates and detector-seed repeat, and remain labeled `automatic_candidate`. Earlier experimental directories are preserved separately.

| Baseline | Fitted scale | Camera-position RMSE, Blender units | Median camera-orientation error |
|---|---:|---:|---:|
| Before VQA | 3.77355 | 0.15501 | 5.96° |
| VQA-informed | 3.79474 | 0.16285 | 7.82° |

| Baseline | Scale change with detector seed 1 | Rotation change | Transformed-cloud RMS displacement, Blender units |
|---|---:|---:|---:|
| Before VQA | 0.312% | 0.0457° | 0.00604 |
| VQA-informed | 0.296% | 0.0690° | 0.00581 |

These are errors across all 350 camera pairs after surface-based fitting. They are not the earlier camera-fitted residuals and should not be interpreted as measured physical reconstruction error. Generated outputs are local, Git-ignored artifacts; preserve their input manifests and hashes when transferring them.
