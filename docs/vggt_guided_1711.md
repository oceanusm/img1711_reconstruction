# VGGT-guided IMG_1711 Blender comparison

The model was instructed to fit the Blender reconstruction roughly to the VGGT point cloud. Both guided-scene viewers state this construction instruction separately from the overlay alignment method. The automatic comparison uses structural-surface alignment, with camera correspondences reserved for evaluation. Agreement here measures adherence to the construction reference; it is not independent validation of the physical scene or VGGT accuracy.

Open the [full 3D overlay](../outputs/comparison_surfaces/vggt_guided_1711/automatic/viewer.html) or [alignment report](../outputs/comparison_surfaces/vggt_guided_1711/automatic/index.html). The overlay includes **all 5,000,000 exported points and all 235,496 exported mesh triangles**, plus 350 VGGT cameras and 698 Blender cameras. The Blender geometry is a snapshot at the first video timestamp; moving objects do not animate in this viewer. No Blender RGB or depth renders were requested for this comparison.

## Preserved inputs

The original `/Users/liumartin/Downloads/img_1711/scene.blend` was copied to `blender_scenes/vggt_guided_1711/scene.blend`. Both retain SHA256 `7584e629d457d6fd1b06c7238b6cbc94dcf4cc453fb585b2b4cf9411adff1ad9`; neither was saved over during export. The resulting `outputs/blender_preview/vggt_guided_1711/` contains 290 mesh objects/instances, 126,152 vertices, and all 698 timestamped cameras.

The three supplied VGGT files in `/Users/liumartin/Desktop/scene_reconstruction/src/1711_vggt/` are byte-identical to the existing `outputs/vggt_quality/` bundle, which this comparison reuses:

| File | SHA256 |
|---|---|
| `index.json` | `771d823ca887e9174154aa0c695d11dfba9c30f0a0848d4462115a65c2dabd57` |
| `cameras.npz` | `f34511f73aac8c91654f3f58b0431600783b5eefe5ce8a687cd034970f515b8b` |
| `point_cloud.ply` | `148ca976a35688139b93df1bb368bfe36769a017f2a52247d1a1c43cff826388` |

The canonical `data/frames/manifest.json` hash is `22532e3764f694f04a501c903b35b830e35a5d10f7d2ace544081bfc9e44cd23`. All 350 selected source IDs and timestamps match exactly, spanning frames 0–697 and 0–23.235 seconds. Blender evaluates `1 + source_time_seconds × 30`; the final timestamp maps to frame 698.05, recorded without stretching the animation.

## Reproduce the comparison

These outputs already exist locally. Choose fresh output directories and a new HTML filename before rerunning; the tools reject mixed or overwritten results.

```bash
PYTHONPATH=src python3 -m scene_reconstruction blender-export \
  --blend blender_scenes/vggt_guided_1711/scene.blend \
  --manifest data/frames/manifest.json --frames 0,127,380,697 \
  --output outputs/blender_preview/vggt_guided_1711

PYTHONPATH=src python3 -m scene_reconstruction align-surfaces \
  --vggt outputs/vggt_quality \
  --blender-export outputs/blender_preview/vggt_guided_1711 \
  --manifest data/frames/manifest.json --surface-profile auto \
  --stability-runs 2 --viewer-points 0 \
  --output outputs/comparison_surfaces/vggt_guided_1711/automatic

PYTHONPATH=src python3 -m scene_reconstruction view-alignment \
  --vggt outputs/vggt_quality \
  --blender-export outputs/blender_preview/vggt_guided_1711 \
  --alignment outputs/comparison_surfaces/vggt_guided_1711/automatic/alignment.json \
  --manifest data/frames/manifest.json \
  --max-points 5000000 --max-triangles 300000 --max-images 36 \
  --output outputs/comparison_surfaces/vggt_guided_1711/automatic/viewer.html
```

The separate viewer command raises the mesh budget above this scene's triangle count. It applies the saved transform without refitting or changing the source cloud. Surface fitting uses the default 300,000-point sample, spatially balanced to 26,681 points; this fitting sample is independent of the full display budget.

## Actual automatic result

The result is `automatic_candidate`: the implemented geometric checks and two detector-seed runs pass. The fit uses one positive uniform scale, rotation and translation, with no camera-derived initialization or fallback. `alignment.json` records `automatic_surfaces`; all 350 camera rows have role `evaluation`.

| Measurement | Result |
|---|---:|
| Global scale | 3.752607555 |
| Camera-position RMSE, Blender units | 0.069562801 |
| Median camera-orientation difference | 0.635735° |
| Scale change with detector seed 1 | 0.214273% |
| Rotation change with detector seed 1 | 0.019978° |
| Transformed-cloud RMS displacement, Blender units | 0.003733134 |

The kitchen profile matches the floor, main worktop and two backsplashes by their exact scene names, then selects actual finite mesh faces. The floor-to-worktop separation is 0.907000057 Blender units. Although `Spice counter slab` is recognized, its top is 0.003 units below the main worktop and is excluded from the final coplanar anchor. The island top remains a separate inspection surface.

Anchor RMS residuals are 0.004027 (floor), 0.017531 (worktop), 0.009793 (main backsplash), and 0.013755 (sink backsplash), in Blender units. Those surfaces determined the fit. Passing the checks is a geometric hypothesis, not verified semantic matching or a physical accuracy guarantee; the model-guided construction further limits any claim of independent validation. See [automatic surface alignment](surface_alignment.md) for the algorithm and limitations.

## Alignment recovered from the inspection Blender scene

The later supplied `/Users/liumartin/Downloads/img_1711 (2).zip` contains `scene_with_vggt.blend` and its original `vggt_alignment.json`. The saved placement is now available as [reference_alignment.json](../blender_scenes/vggt_guided_1711/reference_alignment.json), with the [unaltered source alignment](../blender_scenes/vggt_guided_1711/source_vggt_alignment.json) and [verification results](../blender_scenes/vggt_guided_1711/reference_alignment_verification.json). Open the [saved-reference overlay](../outputs/comparison_reference/vggt_guided_1711/viewer.html) to reproduce that placement. The [automatic overlay](../outputs/comparison_surfaces/vggt_guided_1711/automatic/viewer.html) remains separately preserved.

This is an extracted transform, with no new fit. The native cloud object's local XYZ, RGB, confidence and source-frame IDs match **all five million original PLY records bit for bit**. Its `matrix_world` agrees with the archived transform within Blender's numeric precision: maximum matrix-element difference is `3.6244e-7`, and maximum transformed-point difference is `6.6937e-7` Blender units. The matrix is unchanged at Blender frames 1, 349 and 698; the object has no parent or constraints, and its only modifier converts vertices to display points. The source scene remains unchanged.

The saved transform maps raw VGGT coordinates into Blender world as `p_blender = scale * rotation @ p_vggt + translation`, with scale **3.840510865**. Compared with the automatic surface fit, that is **2.34246% larger scale** and a **0.636004° rotation difference**. It retains the original assumed island-height calibration; its physical scale is not independently measured.

The inspection file's blue preview uses centroids of occupied 18 mm voxels and 4 mm-radius markers. The new viewer instead displays every original colored point under the same saved alignment, together with all 235,496 authored mesh triangles. This display difference can affect the visual impression without changing placement.

Reuse the existing clean Blender export and apply the saved transform:

```bash
PYTHONPATH=src python3 -m scene_reconstruction align-vggt \
  --vggt outputs/vggt_quality \
  --blender-export outputs/blender_preview/vggt_guided_1711 \
  --transform blender_scenes/vggt_guided_1711/reference_alignment.json \
  --output outputs/comparison_reference/vggt_guided_1711

PYTHONPATH=src python3 -m scene_reconstruction view-alignment \
  --vggt outputs/vggt_quality \
  --blender-export outputs/blender_preview/vggt_guided_1711 \
  --alignment outputs/comparison_reference/vggt_guided_1711/alignment.json \
  --manifest data/frames/manifest.json \
  --max-points 5000000 --max-triangles 300000 --max-images 36 \
  --output outputs/comparison_reference/vggt_guided_1711/viewer.html
```

These outputs already exist; use fresh destinations when rerunning. The result records `supplied_transform`, with camera-position RMSE **0.000449992 Blender units** and median orientation difference rounding to zero. The scene's anchor cameras were constructed from VGGT, so this agreement checks reproduction of the saved placement rather than independent camera accuracy. Small remaining residuals are compatible with evaluating integer-frame camera keys at the video's exact subframe timestamps.
