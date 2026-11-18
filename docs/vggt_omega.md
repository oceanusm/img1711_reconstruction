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

After canonical frames have been extracted, invoke the project's `vggt` command with:

```bash
scene-recon vggt \
  --manifest data/frames/manifest.json \
  --repo vendor/vggt-omega \
  --checkpoint checkpoints/vggt-omega/vggt_omega_1b_512.pt \
  --output outputs/vggt \
  --stride 10 --max-frames 72 \
  --image-resolution 512 --preprocess-mode max_size \
  --precision bf16
```

See `scene-recon vggt --help` for options and defaults. `--max-frames 0` removes the cap. `--stride` creates candidates across the video, always including its final frame. If there are too many candidates, the cap samples them evenly over the full duration. A cap never silently limits reconstruction to the first part of the clip. At least two frames are required. The output directory must be empty, so incomplete or differently configured runs cannot be mistaken for one reconstruction.

The 72-frame, longest-side-512 setting is a conservative starting point for a 48 GB A6000; its GPU inference has not been benchmarked here. Upstream's [memory table](https://github.com/facebookresearch/vggt-omega#runtime-and-gpu-memory) is measured on an A100 with a different image shape and preprocessing mode. Increase frame count after observing actual memory use. Reduce `--max-frames` after a CUDA OOM. The adapter deliberately does not concatenate separately normalized reconstructions. Every selected pose and depth in one output belongs to the same joint inference.

`--precision auto` selects BF16 on supported hardware and FP16 otherwise. FP32 is available at higher memory cost. The camera and depth heads retain upstream FP32 computation. Only CUDA inference is supported by this adapter.

## Output contract

`index.json` is the completion marker and contains frame IDs, timestamps, input/checkpoint hashes, upstream revision, environment versions, per-frame output filenames, and preprocessing geometry. Treat a directory without this marker as incomplete.

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
