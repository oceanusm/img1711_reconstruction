# Blender camera, mesh, and depth export

`scene-recon blender-export` reads one original `.blend` in a separate background Blender process. It never saves that file. It exports the evaluated camera at every source-video timestamp, a mesh snapshot, and optional RGB/depth rasters on selected frames. The host checks the original file's SHA256 before and after execution. Embedded script auto-execution is disabled.

## Actual baselines inspected

Both supplied files opened successfully in **Blender 4.2.23 LTS**. Both have a 1080×1920 render grid, frames 1–698 at 30 fps, metric scene units with scale length 1, no linked Blender libraries, and packed file textures.

| File | Active camera | Camera animation | Source scene |
|---|---|---|---|
| `blender_scenes/before_vqa/scene.blend` | `Tracked capture camera` | Pose and lens animated | 455 scene objects, including 448 meshes |
| `blender_scenes/vqa_informed/scene.blend` | `Source camera` | Pose animated, fixed lens | 1,001 scene objects, including 911 meshes |

The original renderer is EEVEE in both baselines. This export uses **Cycles on CPU** for a consistent ray-traced geometric depth pass, with motion blur and depth of field disabled. Original compositing is replaced with direct render outputs in the temporary process. RGB is therefore a geometry-review render, not a promise to reproduce every original post-processing effect.

## Commands

Install Blender 4.2 or later. The wrapper finds `blender` on PATH or `/Applications/Blender.app/Contents/MacOS/Blender`; use `--blender /path/to/blender` to override.

Export all camera poses and one geometry snapshot without rendering:

```bash
scene-recon blender-export \
  --blend blender_scenes/before_vqa/scene.blend \
  --output outputs/blender/before_vqa
```

Export cameras/geometry plus sampled depth and RGB:

```bash
scene-recon blender-export \
  --blend blender_scenes/vqa_informed/scene.blend \
  --output outputs/blender/vqa_informed \
  --frames 0,127,380,697 --depth --rgb \
  --resolution-scale 0.25 --samples 64
```

For depth on all 698 frames at the full source resolution, choose a new output directory and use `--frames all --depth --resolution-scale 1`. Omitting `--rgb` uses one Cycles sample, sufficient for the single-sample Z pass. RGB uses `--samples` (default 16); higher values reduce RGB grain but do not add antialiasing to depth edges. Full-resolution/full-video export may take much longer than the sampled checks.

`--frames` controls only optional raster rendering. **The camera archive always contains all source timestamps**, so an initial four-frame depth export still supports complete trajectory comparison. `--camera NAME` optionally overrides the scene's active camera.

The default time mapping is:

```text
Blender frame = scene.frame_start + source_time_seconds × scene_fps × time_scale
```

Subframes are preserved. `--frame-start` overrides the offset and `--time-scale` explicitly changes the timeline rate. The default is 1 + source seconds × 30 for these scenes. The last video PTS maps to Blender frame 698.05, slightly beyond the declared last frame 698; this is recorded in metadata rather than silently stretching the trajectory. Blender evaluates the animation's own extrapolation there. Confirm the intended timing before interpreting small temporal discrepancies.

## Depth convention: verified rather than assumed

The Cycles Z/Depth pass in the tested Blender version is already **camera-space z**, not Euclidean distance along a ray. The exporter verifies the installed renderer before each depth run using two actual 32×24 renders: a frontoparallel plane at z=5 and a tilted plane with known analytical intersections.

On Blender 4.2.23, the frontoparallel plane yielded z=5 across the whole image with zero error. The tilted-plane maximum camera-z error was approximately **4.94×10⁻⁵ Blender units**. Dividing this Z pass by the ray norm would be an error. This behavior is also explicit in [Cycles' depth-pass implementation](https://raw.githubusercontent.com/blender/blender/v4.2.23/intern/cycles/kernel/film/data_passes.h): it writes `camera_z_depth` for depth, while the Mist pass uses a different distance calculation.

The runtime test accepts either a verified camera-z pass or a verified radial-distance pass and converts only the latter. It fails if neither interpretation matches both planes within 0.001 Blender units. The measured convention and residuals are stored in `index.json`. It does not infer the convention from a pass name or generic documentation wording.

The exporter routes Render Layers `Depth` through a float32 OpenEXR, reads it as non-color numeric data, and flips Blender's bottom-up pixel buffer into the source's top-left image coordinates. It saves float32 `depth`, boolean `valid_depth`, and derived `ray_distance`. No-hit, nonfinite, zero, and beyond-far-clip values are invalid; their depth is NaN. Glass, transparent materials, and alpha-threshold decisions retain renderer semantics and need review for comparison. The depth pass describes visible surface intersections, not reflected objects or volumetric appearance. See [Blender's render-pass documentation](https://docs.blender.org/manual/en/4.4/render/layers/passes.html).

For a perspective camera, `ray_distance = depth × ||K⁻¹ [x,y,1]||`, with rays normalized to z=1. DA3 and VGGT use camera-z, so compare their values to `depth`, not `ray_distance`.

## Output contract

```text
index.json       Source hashes, timing, camera/depth conventions, rendering/calibration information
request.json     Exact requested export settings
blender.log      Full Blender process output
cameras.npz      Cameras at every source frame
geometry.npz     One evaluated world-space mesh snapshot
objects.json     Geometry snapshot object IDs/names/bounds
frames/000000.npz  Optional numeric depth at a selected frame
rgb/000000.png     Optional RGB at a selected frame
```

`index.json` is complete only when `status == "complete"`. Every `frames` row has `index`, `time_seconds`, `pts_seconds`, `camera_array_index`, and the evaluated `blender_frame`. Only rendered rows have `file` and/or `rgb_file`. Dimensions in `width/height` are the export grid; `source_width/source_height` and `canonical_width/canonical_height` describe the source video. `source_to_export_pixel_affine` records the full-frame pixel-center resize. Rounded raster sizes are compensated using render pixel aspect so scaling does not change the camera's physical frustum. No crop is applied.

`cameras.npz` keys:

- `frame_indices`, `time_seconds`, `lens_mm`.
- `camera_to_world`, `world_to_camera`: homogeneous 4×4 matrices in **OpenCV camera axes** (right/down/forward), in the original Blender world.
- `camera_to_world_blender_basis`: Blender's right/up/back camera basis.
- `intrinsics`: K on the exported raster grid. `intrinsics_canonical` and `intrinsics_source` both provide K on the upright source-video grid.

K comes from Blender's evaluated projection matrix, including animated focal length, sensor fit, shifts, and pixel aspect. Integer coordinates identify pixel centers; for an unshifted image, the principal point is `((width−1)/2, (height−1)/2)`. Orthographic/panoramic cameras and sheared/reflected camera transforms fail explicitly.

`geometry.npz` contains `vertices`, `triangles`, `triangle_object_ids`, approximate material `face_colors`, and Unicode `object_names`. Geometry is a single evaluated viewport-depsgraph snapshot at the first source timestamp, excludes objects with `hide_render`, and includes evaluated curve/text meshes and instances. Render-only modifier settings and moving objects can differ from this snapshot; the actual depth renders evaluate the scene at each selected time. Treat approximate mesh colors as visualization aids; texture appearance comes from RGB renders.

Depth is in Blender scene-coordinate units. `unit_scale_meters` records the scene's configured scale length for metric/imperial scenes, or is null for unspecified units. This is a modeling convention, not independently verified physical calibration. Both present baselines declare scale 1 meter per scene unit.

## Verification performed here

Both baselines have complete local preview exports at `outputs/blender_preview/{before_vqa,vqa_informed}`: all 698 cameras, one mesh snapshot, and source frames 0, 127, 380, and 697 rendered at 270×480, with 8 RGB samples. Mesh arrays and camera matrices were checked finite, source hashes were unchanged, and RGB orientation/framing was inspected. These are actual Blender results; they are not VGGT/DA3 predictions.

Pure CPU tests cover projection/shift conventions, resize geometry, timestamps, no-hit validity, radial versus z conversion, and protected source handling. To rerun the actual analytic renderer test:

```bash
RUN_BLENDER_TESTS=1 python3 -m unittest discover -s tests -p test_blender_export.py -v
```

On this Mac, Blender's Metal initialization crashed inside the restricted execution sandbox but succeeded outside it. This is separate from the earlier Documents-folder access issue; the source files used here are already in the repository. Remote Linux execution does not use that macOS Metal backend.
