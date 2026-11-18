# VGGT-Ω reconstruction

This adapter targets the official [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega) source at commit `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6`, inspected on 2026-11-18. It uses the actual Ω architecture, not the older VGGT package. The adapter performs one joint inference over a frame selection spanning the complete clip. It exports a colored point cloud, camera poses, camera intrinsics, depth, confidence, and exact frame correspondence.

## Environment and checkpoint

Use the project's isolated geometry environment: Python 3.10–3.12, PyTorch 2.6.0, torchvision 0.21.0 and CUDA 12.4 wheels on the reported Ubuntu 22.04/A6000 host. Official minimums are Python 3.10, PyTorch 2.6, and torchvision 0.21. The upstream package requires `numpy<2`, Pillow, einops, safetensors, and opencv-python. The setup script pins a matched CUDA/PyTorch profile; see [remote setup and migration](remote_geometry.md) before reusing an earlier CUDA 12.6 environment.

```bash
CUDA_FLAVOR=cu124 PYTHON_BIN=python3 bash scripts/setup_a6000.sh vggt
```

Download the checkpoint from the public [wincentIsMe/VGGT-Omega mirror](https://huggingface.co/wincentIsMe/VGGT-Omega). Its model card describes an unmodified copy of `vggt_omega_1b_512.pt`, distributed under the FAIR Noncommercial Research License. The mirror does not require the official repository's per-account access approval. Run on the A6000 host:

```bash
.venv-vggt/bin/hf download wincentIsMe/VGGT-Omega vggt_omega_1b_512.pt --local-dir checkpoints/vggt-omega
```

Model code still comes from the pinned official GitHub source. The 512 checkpoint is the upstream recommendation for real-world reconstruction. The 416 reproduction checkpoint is a different release intended for benchmark comparisons. This adapter does not support the separate text-alignment checkpoint. The adapter reads the downloaded local file; it does not automatically fetch weights. Checkpoint contents are loaded with `weights_only=True` and a strict state-dict check, and the checkpoint SHA256 is recorded with the results.

## Running

After canonical frames have been extracted, run the quality preset. Use a new output directory to retain the earlier 71-frame reconstruction:

```bash
scene-recon vggt \
  --manifest data/frames/manifest.json \
  --repo vendor/vggt-omega \
  --checkpoint checkpoints/vggt-omega/vggt_omega_1b_512.pt \
  --preset quality \
  --output outputs/vggt_quality
```

`quality` is the default preset. Explicit options override the preset, for example `--preset quality --max-frames 250`. See `scene-recon vggt --help` for all options. `--max-frames 0` removes the cap; it does not change the stride. `--stride` creates candidates across the video, always including its final frame. If there are too many candidates, the cap samples them evenly over the full duration. At least two frames are required. The output directory must be empty, so incomplete or differently configured runs cannot be mistaken for one reconstruction.

## Quality, resolution and memory

| Preset | Frame selection | Views in this clip | Model image, width×height | Cloud export |
|---|---|---|---|---|
| `quality` (default) | Stride 2, cap 400 | 350 | `balanced 512`: 384×688 | Point stride 1, at most 5,000,000 points |
| `quick` | Stride 10, cap 72 | 71 | `max_size 512`: 288×512 | Point stride 4, at most 1,000,000 points |

Both presets use BF16 and a per-frame 50th-percentile confidence threshold for cloud export. The larger cloud budget affects exported sampling only; all model-resolution depth/confidence values are saved regardless of cloud filtering.

The upstream `balanced` loader targets roughly 512² pixels while preserving aspect ratio and rounding to 16-pixel patches. For this portrait clip it retains about **1.79 times as many pixels per view** as `max_size 512`; the number 512 is not its longest-side limit. [Pinned loader](https://github.com/facebookresearch/vggt-omega/blob/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6/vggt_omega/utils/load_fn.py).

The quality preset's peak A6000 memory has **not** been measured here. The official [A100 measurements](https://github.com/facebookresearch/vggt-omega#runtime-and-gpu-memory) report 28.26 GB for 300 views and 35.71 GB for 400 at 624×416, a comparable pixel count. These guide the choice but do not guarantee the same allocation on an A6000 or another software stack. The adapter prints actual model dimensions and CUDA memory readings and records runtime statistics in `index.json`. Use one otherwise available GPU and inspect the measured peak before enlarging a run. No automatic fallback silently lowers the requested settings.

BF16 follows upstream's usual inference choice on supported GPUs; the camera and depth heads retain FP32 computation. `--precision auto` selects BF16 when supported and FP16 otherwise. FP32 is available at higher memory cost, without a guarantee of better geometry. [Pinned model implementation](https://github.com/facebookresearch/vggt-omega/blob/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6/vggt_omega/models/vggt_omega.py). Only CUDA inference is supported by this adapter.

For all **698** frames, explicitly add `--stride 1 --max-frames 0`; balanced inference at that size is memory-heavy and is not promised to fit a 48 GB card. Increasing `--image-resolution` beyond the checkpoint's 512 setting is an experiment, not a guaranteed quality improvement. More redundant, blurry or moving-object views can also fail to improve reconstruction. Validate geometry with the independent Blender/depth comparisons rather than treating view count as an accuracy score.

After an OOM, choose a smaller explicit `--max-frames` and a new output directory. Keep one joint reconstruction: independently normalized chunks cannot simply be concatenated. Every selected pose and depth in one output belongs to the same joint inference.

To compare this run with both Blender baselines:

```bash
VGGT_OUTPUT=outputs/vggt_quality \
COMPARISON_OUTPUT=outputs/comparison_quality \
bash scripts/compare_geometry.sh align
```

## Output contract

`index.json` is the completion marker and contains frame IDs, timestamps, input/checkpoint hashes, upstream revision, environment versions, resolved settings, actual preprocessing geometry, CUDA memory statistics and per-frame output filenames. Treat a directory without this marker as incomplete.

`cameras.npz` contains:

| Array | Shape | Meaning |
| --- | --- | --- |
| `frame_indices` | N | Original canonical frame indices |
| `time_seconds`, `pts_seconds` | N | Timeline correspondence from frame extraction |
| `world_to_camera` | N × 4 × 4 | Homogeneous OpenCV camera-from-world transforms |
| `camera_to_world` | N × 4 × 4 | Inverses of the above |
| `camera_to_world_blender_camera_basis` | N × 4 × 4 | Camera-to-world with local Y/Z flipped for Blender cameras |
| `intrinsics_model` | N × 3 × 3 | Intrinsics in processed-image pixels |
| `intrinsics_canonical` | N × 3 × 3 | Intrinsics mapped back into canonical video pixels |
| `canonical_to_model_pixel_affine` | 3 × 3 | Crop/resize mapping A for homogeneous pixels |
| `pose_enc` | N × 9 | Raw upstream camera encoding |

`depth/<frame_index>.npz` contains float32 `depth` and `confidence` at **model resolution**. Depth is camera Z, not radial distance. `rgb_model/<frame_index>.png` gives the exact RGB raster paired with that depth. Numeric frame filenames are eight digits and are linked explicitly from `index.json`.

The official loader center-crops aspect ratios outside [0.5, 2.0], then resizes to patch-aligned dimensions. Equal-size canonical frames require no padding. The manifest stores crop bounds, actual model size, and both affine directions. Integer image coordinates refer to pixel centers; the PIL resize transform is `u_model = sx*(u_original - crop_left + 0.5) - 0.5`, and similarly for Y. Thus `K_original = inverse(A) @ K_model`. Cropped-away pixels have no prediction. Use these transforms when sampling SAM masks or comparing original-resolution Blender rasters; do not assume simple whole-image stretching.

`point_cloud.ply` is binary little-endian PLY with XYZ, RGB, raw confidence, and original frame index. It contains filtered, spatially subsampled depth-unprojected points in the model's common world. `--confidence-percentile` filters independently per frame; `--point-stride` controls raster subsampling; `--max-points` bounds the exported total by distributing the budget across frames. The cloud is not a mesh or a fused surface: duplicate observations, depth discontinuity artifacts, and dynamic objects can remain. Changing these point-cloud options does not change saved depth or cameras.

## Interpreting distance and Blender alignment

Camera axes are OpenCV: X right, Y down, Z forward. Point-cloud axes are the model's unmodified predicted world axes. Camera translation, camera-Z depth, and point positions share one learned scale within this joint run. They are not automatically meters. A real measured distance or a known Blender correspondence is needed to establish physical scale; multiple reliable correspondences are preferable to one.

Fit a similarity transform between the predicted world and the Blender world before comparing 3D quantities. The supplied Blender camera-basis array changes only the camera's local axis convention; it does not resolve world rotation, translation, or scale. Its rotation can be mapped by the fitted world rotation, and its camera center by the full similarity. Avoid applying scene scale to a camera's rotation basis.

Confidence is an uncalibrated score, not a probability. Cross-view consistency, occlusions, segmentation, static-scene assumptions, image stabilization, and lens distortion all affect comparisons. Use the saved raw outputs to inspect disagreements; don't interpret point-cloud agreement alone as proof that the Blender geometry is correct.

In `IMG_1711.MOV`, a hand moves jars/containers. Those moving objects and the hand can produce inconsistent points, duplicated surfaces, and biased camera estimates in a joint reconstruction. Establish the camera/world alignment from stationary scene structure first. Use SAM masks to identify dynamic regions during analysis; this adapter currently retains them in model inference and the exported point cloud. High confidence does not guarantee that a moving point is compatible with the static scene.

Occlusion also changes what can be compared: a hand can hide an object's surface, and rotation can cause the object to hide parts of itself. SAM outputs visible silhouettes. Blender's corresponding comparison mask must respect the same occluders and object motion. Comparing a full unobstructed Blender object mask with a partly occluded video mask would penalize valid geometry. Object motion and camera motion must be separated when interpreting mask, pose, or depth discrepancies.
