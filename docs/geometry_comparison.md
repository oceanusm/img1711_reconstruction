# Comparing VGGT-Ω with the reconstructed Blender scene

The repository exports Blender cameras, geometry and ray-traced depth, aligns VGGT to either scene, builds an offline interactive viewer, and compares DA3 depth with Blender depth. SAM3 and Cryptomatte are independent of this workflow.

For focal length, sensor fit, iPhone stabilization and residual distortion, see [camera calibration](camera_calibration.md). Keep the original-camera comparison separate from any later rerender using a shared video-derived calibration; matching a nominal focal length alone does not correct edge warps.

The unchanged originals are `blender_scenes/before_vqa/scene.blend` and `blender_scenes/vqa_informed/scene.blend`; [baselines.json](../blender_scenes/baselines.json) records their hashes and conditions. Actual preview exports exist locally under `outputs/blender_preview/`: **698 timestamped cameras per scene**, one mesh snapshot, and depth/RGB at source frames **0, 127, 380 and 697**, rendered at 270×480. These generated outputs are Git-ignored. VGGT and DA3 comparisons await the real inference results.

## Open the prepared viewers

On this Mac:

```bash
open outputs/blender_preview/before_vqa/viewer.html
open outputs/blender_preview/vqa_informed/viewer.html
```

These currently show Blender geometry and camera paths alongside sampled video/RGB renders. Once VGGT is available, the same viewer adds its aligned colored cloud, trajectory, camera frusta and camera residuals. The mesh is a static snapshot at the first source timestamp; the camera slider does not animate that mesh. Actual depth renders evaluate the scene separately at each timestamp.

## Prepare an export on another machine

Comparison tools need NumPy/Pillow, but no GPU or model imports. Use either existing model environment, or create a small local environment:

```bash
python3 -m venv .venv-tools
.venv-tools/bin/python -m pip install -e .
source .venv-tools/bin/activate
```

Install Blender 4.2+ separately if exporting there. Alternatively, transfer `outputs/blender_preview/` from this Mac; the export contains portable numeric arrays and relative raster paths. Preserve the exact `data/frames/manifest.json` used for inference.

To recreate the previews in **new output directories**:

```bash
for scene in before_vqa vqa_informed; do
  scene-recon blender-export \
    --blend "blender_scenes/$scene/scene.blend" \
    --output "outputs/blender_preview/$scene" \
    --frames 0,127,380,697 --depth --rgb --resolution-scale 0.25 --samples 16
  scene-recon view-alignment \
    --blender-export "outputs/blender_preview/$scene" \
    --output "outputs/blender_preview/$scene/viewer.html"
done
```

Do not rerun this into existing previews. Export commands reject nonempty directories. See [Blender export](blender_export.md) for timing, render conventions and camera calibration details.

## Plug in VGGT and DA3 results

Place the complete inference directories at `outputs/vggt` and `outputs/depth`, retaining the exact source manifest/frames. From the repository root:

```bash
bash scripts/compare_geometry.sh align   # VGGT alignment and interactive overlay, both scenes
bash scripts/compare_geometry.sh depth   # DA3 versus available Blender depth, both scenes
```

Or run `bash scripts/compare_geometry.sh all` once both models are ready. The helper chooses `.venv-tools`, `.venv-depth`, or `.venv-vggt` if available, falling back to `python3`. `PYTHON_BIN` overrides that choice. It uses the existing preview exports by default. Each comparison output must be new or empty.

Results are written under `outputs/comparison/{before_vqa,vqa_informed}/`:

- `alignment/alignment.json` and `alignment/camera_errors.csv`: one global transform, camera position/orientation/intrinsics errors and provenance.
- `alignment/viewer.html`: standalone interactive cloud/mesh/camera overlay, object selection and landmark-pair download.
- `depth/index.html`, `depth/comparison.json` and `depth/per_frame.csv`: shared-color depth previews, numeric errors, validity coverage and projection diagnostics.

Default camera-center fitting uses all matched frames. To evaluate independent evidence, use `scene-recon align-vggt --fit-frames …` with held-out frames or fit static surface landmarks. The viewer downloads corresponding VGGT points and Blender vertices for a landmark fit. [Alignment options and interpretation](alignment.md).

The depth report compares camera-z on matching timestamped image grids. It compares Blender's scene camera with the real-video estimate directly; VGGT world alignment is unnecessary for this image-space comparison. The default preserves declared physical units. A separate optional global scale fit can diagnose scale mismatch, with explicit fitting and held-out frames. [Depth comparison options and metrics](depth_comparison.md).

## Expand depth rendering to all 698 frames

The four preview frames provide an initial check. To produce full-video, full-resolution Blender depth, render into a new directory. RGB is omitted to avoid its additional sampling cost:

```bash
for scene in before_vqa vqa_informed; do
  scene-recon blender-export \
    --blend "blender_scenes/$scene/scene.blend" \
    --output "outputs/blender_full/$scene" \
    --frames all --depth --resolution-scale 1
done

BLENDER_EXPORT_ROOT=outputs/blender_full \
COMPARISON_OUTPUT=outputs/comparison_full \
bash scripts/compare_geometry.sh depth
```

The exporter currently uses CPU Cycles and validates its depth convention with analytic plane renders. Full-video rendering is a separate substantial job; it has not been run locally. In the tested Blender 4.2.23, the Z pass is already camera-z and must not be divided by ray length. Missing surfaces, transparency and reflections need inspection alongside validity coverage.

Compare both scenes with identical timestamps and filtering settings. The uninformed scene animates its camera lens as well as its pose; the informed scene keeps a fixed lens. Intrinsics/FOV therefore matter alongside position and orientation. There is no dense automatic object-position score yet; use corresponding surface landmarks and the overlay first.

## Preserve inputs

Keep the `.blend` file (with its animated camera and any moved objects), the exact `data/frames/manifest.json`, and the complete `outputs/vggt` directory. Also retain DA3 outputs for independent depth checks. The scene's animation timing must map to the video's timestamps; matching integer Blender frame numbers to video frame indices can be wrong for variable-rate video.

For a camera-only exchange, export per-source-frame `time_seconds`, evaluated camera-to-world matrices, projection type, effective render dimensions/pixel aspect, and lens/sensor/shift settings (or a calibrated pixel intrinsic matrix). Evaluate constraints and parent transforms. For object comparison, keep evaluated meshes or actual surface landmarks in Blender world coordinates; arbitrary object origins are not geometric centers.

## Align coordinate systems once

VGGT returns a common world for its selected frames, but its origin, orientation, and scale do not automatically match Blender. Fit **one similarity transform**:

```text
point_blender = scale * rotation @ point_vggt + translation
```

Use stationary, distributed scene correspondences such as cabinet/counter corners. At least three non-collinear paired 3D points are needed for a nondegenerate similarity fit; use additional distributed correspondences for robustness and reserve some for validation. Alternatively, corresponding camera centers can support trajectory alignment if their motion provides sufficient spatial extent. Near-pure rotation or a very limited trajectory constrains scale poorly.

Apply the same scale/rotation/translation to the whole cloud and camera centers. Apply only rotation to camera orientation bases. Do not independently align each camera or each object: that can hide the very errors being measured. If alignment is fitted to camera centers, their post-fit residual is a conditional trajectory check rather than independent proof of scene geometry; inspect held-out surfaces too.

`cameras.npz` includes `camera_to_world_blender_camera_basis`. It converts local camera axes from OpenCV (x right, y down, z forward) to Blender (x right, y up, viewing along -z). It does **not** align the world or resolve scale. The supplied `intrinsics_canonical` uses upright source-frame pixels.

## Compare complementary evidence

1. **Camera motion:** at matched timestamps, compare aligned camera centers, orientation error in degrees, and how the trajectory changes over time. A common global offset should be handled by alignment; changing residuals can reveal trajectory mismatch or model drift.
2. **Projection:** compare the estimated intrinsics/FOV with Blender's camera settings. Render representative Blender frames and inspect the same static landmarks. Good 3D placement with a wrong focal length can still produce poor image agreement.
3. **Static surfaces:** overlay the colored point cloud on the Blender meshes. Check countertop heights and widths, cabinet/wall planes, appliance boundaries, and spacing. Use robust point-to-surface distances or matched landmarks for numerical evaluation; the VGGT cloud contains duplicate observations, outliers, and uncertain edges.
4. **Depth:** compare Blender depth with VGGT/DA3 depth on corresponding pixels using the correct intrinsics, scale, and validity masks. Both adapters and this repository's Blender exporter save camera-z depth. Convert only an independently verified ray-distance output. DA3's default model estimates meters, while each window has independent local camera coordinates and can have scale disagreement. DA3 depth is an estimate from the original video, not measured ground-truth depth.

VGGT does not label individual objects or directly produce their object-center positions. Object-level positions require selecting corresponding surface regions/landmarks, manually at first or through masks later. Static structure should establish the initial alignment because this clip includes a hand moving containers. Moved objects need matching animation or separate time-specific analysis.

A discrepancy is evidence to investigate, not automatic proof that Blender is wrong: reconstruction models can make depth/pose errors, particularly on reflective appliances, plain surfaces, occlusion boundaries, and dynamic regions. Agreement across reprojection, trajectory, static surfaces, and independent depth estimates is stronger than agreement on any one measure.

