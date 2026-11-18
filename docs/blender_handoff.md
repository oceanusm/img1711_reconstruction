# Blender mask handoff

Use the extracted frames as the common reference. Do not reconstruct a frame mapping by multiplying timestamps by 30.

## Required alignment

- Canonical dimensions: **1080 width × 1920 height**. Already rotated upright. No further rotation, mirroring, crop, letterboxing, or vertical flip.
- Frame identity: zero-based indices `0..697`, filenames `000000.png` through `000697.png`.
- Timing: render at each `frames[i].time_seconds` in `data/frames/manifest.json`. Blender's first animation frame is often 1; map it explicitly to canonical frame 0. If an animation assumes a constant rate, sample it at `scene_start + time_seconds * animation_fps`, with subframes, rather than simply matching integer indices.
- Pixel origin: upper left, x right, y down. Blender's internal buffers may need vertical flipping on export; verify against a rendered RGB image, not an assumption.
- Masks: visible surfaces only, with correct depth occlusion. Hidden/amodal full-object masks are a different comparison.
- Motion blur, lens distortion, rolling shutter, electronic stabilization, focus/zoom, and anti-aliasing can affect the observed silhouette. Record the render settings. Do not claim a residual is geometry error until image formation differences are considered.

## Object definitions

Agree on stable names and whole-object grouping. A single physical jar normally includes its lid, body, and label; a microwave includes its handle and door. Multiple Blender meshes may therefore need to be combined into one target mask.

Keep separate bottles separate. A `bottles` union from SAM must be compared with the union of all corresponding Blender bottles. The `instances` outputs provide separate detected tracks, but SAM IDs reset between processing chunks and may drift. Use the assembly mapping before calling these stable physical identities.

Define surfaces explicitly (floor, wall, countertop top, backsplash, shelf, etc.). Avoid mixing an island's top surface with its full outer silhouette without marking that grouping choice. SAM candidates can overlap; they are not automatically an exclusive label image.

## Files

```text
data/blender/
  metadata.json
  masks/
    microwave/000000.png
    microwave/000001.png
    ...
    jar_01/000000.png
    ...
```

Each mask is a **single-channel uint8 binary PNG**, background 0, foreground 255 (0/1 is also accepted). Export an all-zero PNG when the object is absent/fully occluded. A missing file is an error and must not be used to mean absence. Provide every frame for every compared target. If rendering antialiased mattes, retain those separately and use a documented threshold (e.g. 0.5) for these binary comparisons.

The metadata shape is:

```json
{
  "schema_version": 1,
  "status": "complete",
  "manifest_sha256": "SHA256_OF_THE_EXACT_SHARED_MANIFEST_FILE",
  "width": 1080,
  "height": 1920,
  "frame_count": 698,
  "objects": {
    "microwave": ["MicrowaveBody", "MicrowaveDoor", "MicrowaveHandle"],
    "jar_01": ["JarBody", "JarLid"]
  },
  "mask_semantics": "visible object silhouettes; binary threshold 0.5"
}
```

Compute the manifest hash using `shasum -a 256 data/frames/manifest.json` or Python's `hashlib.sha256`. File paths can change when the project is copied; the manifest bytes should not. The comparison validates the metadata if supplied; filenames alone cannot prove correct timing.

## Cameras and depth

Both model adapters label their camera conventions explicitly. World-to-camera extrinsics mean `X_camera = R @ X_world + t`. Their camera axes are OpenCV: x right, y down, z forward. Blender cameras use local x right, y up, viewing along -z. If using a model camera-to-world matrix, convert its **camera basis** with `C_blender = C_opencv @ diag(1,-1,-1,1)`, then separately apply the chosen world alignment. The VGGT file already exports this camera-basis conversion; it does not align the scene's world axes or units.

Depth outputs are camera-space **z**, not Euclidean ray length. If exporting Blender's ray-distance Z pass, convert with the camera ray geometry before numerical comparison. Use source-grid intrinsics, not intrinsics for the model's downsampled input. Preserve the numeric float depth (EXR/NPY/NPZ), validity mask, units, and camera calibration; colorized PNGs are insufficient.

VGGT scale is arbitrary. Align a single common similarity transform (rotation, translation, positive scale) to known scene correspondences. The default DA3 model supplies estimated meters and window-local camera poses; validate the scale and keep chunk scope explicit. Avoid independently fitting every frame or every object when scoring a reconstruction, because doing so can conceal inconsistent geometry or camera motion.

## Occlusion and dynamics

The clip contains a hand moving containers. A static Blender reconstruction cannot reproduce those trajectories. Either animate the affected items and hand, compare only agreed static intervals, or explicitly exclude known dynamic/occluding pixels and report that separately. SAM masks are model predictions, so review bad masks before using them as evaluation references.
