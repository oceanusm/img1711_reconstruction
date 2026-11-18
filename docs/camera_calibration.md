# Camera calibration for IMG_1711

Use the iPhone's nominal focal length as a starting prior, then check the **effective camera of the recorded video**. Matching a nominal 1× lens alone does not remove edge distortion or establish the correct Blender projection. Preserve the original model estimates and both original Blender scenes for comparison.

## What is known about this recording

Read-only `ffprobe` inspection of `data/IMG_1711.MOV` identifies an **iPhone 15 Pro Max**. The coded raster is 1920×1080 with a −90° display rotation; the canonical frames are upright **1080×1920**, with 698 source timestamps. The exposed container/stream tags do **not** establish the selected physical camera, actual focal length, zoom factor, stabilization setting, or a calibrated intrinsic matrix. Timed metadata tracks exist but have not been decoded for camera calibration. In particular, the phone model alone does not prove that this clip was recorded at 1×.

Apple lists the main camera as 24 mm. This is a nominal equivalent-focal-length description, not a measurement of the physical lens for this MOV. Apple's camera API describes nominal 35 mm equivalent focal length as a diagonal field-of-view measure associated with the photo format. The crop and aspect ratio of a particular video still matter. [iPhone 15 Pro Max specifications](https://support.apple.com/en-us/111828), [Apple's nominal focal-length definition](https://developer.apple.com/documentation/avfoundation/avcapturedevice/nominalfocallengthin35mmfilm).

Apple documents that Enhanced Stabilization slightly zooms video and that Lock Camera prevents switching between cameras during recording. Neither setting has been established for this file. [Apple video recording settings](https://support.apple.com/en-az/guide/iphone/iphc1827d32f/ios).

## Focal length and distortion are different parameters

The useful calibration for comparing the canonical frames is the pixel matrix

```text
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

Here `fx` and `fy` control field of view, and `cx` and `cy` locate the principal point. Radial/tangential distortion is a separate mapping. Changing one focal-length value cannot straighten curved image lines. Edge stretching can also arise from ordinary wide-angle perspective; it is not by itself evidence of uncorrected lens distortion. [OpenCV camera model and calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).

Apple's capture system supports geometric distortion correction and exposes the field of view after that correction. This means processed video **may already include correction**; it does not establish the correction applied to this clip. Do not apply guessed raw-lens coefficients to it. [Apple distortion correction](https://developer.apple.com/documentation/avfoundation/avcapturedevice/isgeometricdistortioncorrectionenabled), [corrected video field of view](https://developer.apple.com/documentation/avfoundation/avcapturedevice/format/geometricdistortioncorrectedvideofieldofview).

Residual stabilization warps, rolling shutter, motion blur, inaccurate geometry, and inaccurate pose can also produce mismatch. These are hypotheses to check against temporal/spatial residuals, not effects a fixed 24 mm setting resolves.

## The two Blender cameras already illustrate the issue

The existing local preview exports contain evaluated cameras at all 698 source timestamps. The following values come from `outputs/blender_preview/<variant>/{index.json,cameras.npz}`, inspected on 2026-11-18. They describe the **generated scenes**, not a measured calibration of the video.

| Baseline | Blender sensor fit | Lens setting | Canonical `fx ≈ fy` | Horizontal FOV | Vertical FOV |
| --- | --- | --- | --- | --- | --- |
| `before_vqa` | Horizontal; 36 mm width | Animated, 54.25–56.29 mm | 1627–1689 px | 35.47–36.71° | 59.24–61.07° |
| `vqa_informed` | Vertical; 36 mm height | Fixed, 29.079 mm | 1550.905 px | 38.39° | 63.51° |

Both have square pixels and canonical principal point `(539.5, 959.5)`. FOV values use the centered pinhole relation `2 atan(image_dimension / (2 focal_pixels))`. Despite the different millimeter settings, their actual projections are much closer than those numbers suggest. Sensor fit, sensor dimension, portrait aspect ratio, pixel aspect and camera shifts determine the final projection together. Compare exported `intrinsics_canonical`, not `lens_mm` alone. The exporter derives K from Blender's evaluated projection matrix; see [Blender export conventions](blender_export.md).

## Recommended comparison procedure

1. **Run the existing geometry pipeline on the unchanged canonical frames.** Keep VGGT's estimated intrinsics, poses and point cloud together. Its current adapter estimates intrinsics rather than accepting a fixed iPhone focal input. Do not substitute another K into the saved cameras while leaving the points/poses untouched. DA3 window estimates provide an additional diagnostic; they are not one globally aligned camera trajectory.
2. **Inspect the camera evidence before changing the scenes.** Compare VGGT and DA3 canonical K/FOV over time with the two exported Blender cameras. Inspect static cabinet/counter edges and tracked landmarks across both central and outer image regions. Exclude moving hands/containers and unreliable reflections from calibration evidence. Estimated focal jitter alone does not prove that the real phone zoomed.
3. **If refinement is needed, estimate it from the video independently of either Blender baseline.** Start with a shared pinhole calibration for a stable lens/zoom segment, using the nominal main-camera FOV only as an initialization if that lens is supported by evidence. A feature-based reconstruction/bundle adjustment can jointly refine intrinsics, poses and static points. Add a simple residual radial model only if supported by spatial residuals and validation. COLMAP recommends the simplest sufficient camera model and sharing intrinsics where appropriate; unconstrained complex calibration per frame can fail. [COLMAP camera models](https://colmap.github.io/cameras.html).
4. **Validate before adopting it.** Reserve timestamps and landmarks from fitting; report held-out reprojection errors separately for image center and edges. Check parameter stability across subsets. Start with fixed intrinsics; introduce segments or regularized temporal changes only when the video supports them. A calibration target filmed with the same lens, resolution, crop and stabilization mode can provide stronger evidence, but it cannot retroactively recover unknown settings of this clip.
5. **Use one documented video calibration across both baseline comparisons.** If residual distortion is significant, choose one common image domain: either rectify the source once and use that new K/grid for every model and comparison, or apply the same calibrated forward distortion to all Blender projections. Preserve originals, record the mapping and validity borders, and keep RGB/depth/masks synchronized. A rectified inference experiment needs a separate manifest/output run; do not mix its pixels with the current outputs. Do not fit a separate video warp to each scene.

Keep two evaluations distinct. **Original-camera evaluation** uses each baseline's authored camera and geometry and therefore measures their combined agreement with the video; the current comparison tools support this. **Evaluation with a shared estimated camera** would render each scene from the same independently estimated video trajectory/intrinsics, after one documented global similarity alignment per scene, to examine geometry with less camera variation. This second experiment is proposed work, not an implemented camera override. Keep its results separate because replacing a scene's camera changes what is being evaluated. Use the same fitting/held-out timestamps and correspondence rules for both variants.

These estimates can support substantial reconstruction improvements even if pixel-perfect overlap remains limited by capture effects or uncertain boundaries. No calibration/refinement has been fitted or applied by this document.
