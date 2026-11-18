# Comparing VGGT-Ω with the reconstructed Blender scene

The first comparison can use camera poses, a point cloud, and static scene surfaces. It does not depend on deciding the SAM3/Cryptomatte workflow. The current code produces the geometry evidence; automated Blender import/alignment/scoring is not implemented yet, and no `.blend` file has been provided in this folder.

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
4. **Depth:** compare Blender depth with VGGT/DA3 depth on corresponding pixels using the correct intrinsics, scale, and validity masks. Both adapters save camera-z depth. Convert Blender ray-distance depth to z before comparing. DA3's default model estimates meters, while each window has independent local camera coordinates and can have scale disagreement.

VGGT does not label individual objects or directly produce their object-center positions. Object-level positions require selecting corresponding surface regions/landmarks, manually at first or through masks later. Static structure should establish the initial alignment because this clip includes a hand moving containers. Moved objects need matching animation or separate time-specific analysis.

A discrepancy is evidence to investigate, not automatic proof that Blender is wrong: reconstruction models can make depth/pose errors, particularly on reflective appliances, plain surfaces, occlusion boundaries, and dynamic regions. Agreement across reprojection, trajectory, static surfaces, and independent depth estimates is stronger than agreement on any one measure.

