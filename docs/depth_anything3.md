# Depth Anything 3

For the reported Ubuntu 22.04 / driver 550.54.14 host, use `CUDA_FLAVOR=cu124 PYTHON_BIN=python3 bash scripts/setup_a6000.sh depth`. This selects PyTorch 2.6.0 / torchvision 0.21.0 / xformers 0.0.29.post3 with CUDA 12.4 wheels and supports Python 3.10–3.12. See [remote setup](remote_geometry.md) for migrating an earlier CUDA 12.6 environment without deleting its files.

This adapter runs **every canonical video frame** through overlapping multiview windows. It saves both native predictions and depth resampled to the canonical frame dimensions, with a frame/timestamp mapping for later Blender comparisons.

The default checkpoint is [`depth-anything/DA3NESTED-GIANT-LARGE-1.1`](https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1). The official project recommends the `-1.1` models after a training fix; the original nested checkpoint is deprecated. The nested model combines multiview geometry with a metric-depth model. It estimates metric scale; it does not provide calibrated physical measurements. The checkpoint is distributed under CC BY-NC 4.0. [Official model table](https://github.com/ByteDance-Seed/Depth-Anything-3#-model-cards)

## Installation and provenance

The adapter was checked against upstream source commit:

```text
ByteDance-Seed/Depth-Anything-3
3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
```

Use `bash scripts/setup_a6000.sh depth` on the Linux GPU host. This creates `.venv-depth`. Keep it separate from SAM3: DA3 currently requires NumPy below 2 and MoviePy 1.0.3. Upstream lists `torch>=2`, torchvision, xformers, OpenCV, Open3D, pycolmap, Pillow, Hugging Face Hub, safetensors, einops, OmegaConf, evo, e3nn and other export dependencies. The setup script adds `addict` explicitly: model source imports it although upstream's package dependency list omits it. The API eagerly imports several exporters even when inference has no export requested. No Gaussian-splatting extension is required for this adapter. [Pinned package definition](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/pyproject.toml)

The source installation sequence, after activating the isolated Python environment and installing compatible CUDA PyTorch/torchvision/xformers wheels, is:

```bash
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git vendor/Depth-Anything-3
git -C vendor/Depth-Anything-3 checkout 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
python -m pip install -e vendor/Depth-Anything-3
python -m pip install addict
python -m pip install -e .
python -c 'from depth_anything_3.api import DepthAnything3; print("DA3 import OK")'
```

Model weights download from Hugging Face on first inference. The default checkpoint is pinned to `b2359bdf726fb44ef62acca04d629dcf158053e7` (verified from the public model repository). `--model` can also be a local snapshot directory. `--model-revision` overrides the default pin or fixes the revision of an alternative model; use a commit SHA instead of a moving branch, especially when resuming a run. The adapter records the requested revision and the loaded model's commit hash when upstream exposes one. A local snapshot should be retained with its original Hugging Face revision metadata.

## Running

From the repository root:

```bash
.venv-depth/bin/scene-recon depth
```

Relevant options:

| Option | Default | Meaning |
|---|---|---|
| `--manifest` | `data/frames/manifest.json` | Canonical extracted-frame manifest |
| `--output` | `outputs/depth` | Output directory for this configuration |
| `--model` | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | Hugging Face model or local snapshot |
| `--model-revision` | Default model uses the pinned SHA above | Override Hugging Face revision |
| `--chunk-size` | 16 | Maximum images in one multiview inference |
| `--overlap` | 4 | Repeated images between adjacent windows |
| `--process-res` | 504 | Longest-side inference size, a multiple of 14 |
| `--device` | `cuda` | PyTorch device |

Sixteen frames at 504 is a conservative starting configuration for the intended 48 GB A6000, not a measured peak-memory guarantee. Run one GPU model at a time. If memory is exhausted, reduce the window to 8 with overlap 2, or reduce resolution to 392, and use a different output directory. A change in window size changes the evidence given to the model and can change the estimates.

The adapter uses the official `DepthAnything3.from_pretrained(...).to(device).eval()` and `model.inference(image=..., process_res_method="upper_bound_resize", infer_gs=False)` APIs. The upstream model manages mixed precision automatically. [Pinned API source](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/api.py)

## Output contract

```text
run.json                     Exact settings and input fingerprint
index.json                   Completion status, frame mapping, coordinate/scale notes
overlap_diagnostics.json      Same-frame disagreement between neighboring windows
chunks/000000.npz             Native model outputs for a whole window
chunks/000000.json            Window scale, dimensions and preprocessing transforms
frames/000000.npz             Selected prediction at canonical frame dimensions
```

The numeric filename is the original canonical **zero-based frame index**, not a timestamp or a chunk-local index. `index.json` includes `time_seconds` and `pts_seconds`. Read its `status` before treating the run as complete. An interrupted run writes an incomplete index and retains completed windows. Re-running the exact same command/input resumes from atomic window caches; changed configuration or input file sizes/modification times require a new output directory. Original images must not be modified underneath a run.

Per-frame arrays:

| Key | Meaning |
|---|---|
| `depth` | Float32 camera-z depth, resampled to canonical H×W |
| `valid_depth` | Finite positive depth, excluding sky when the model exposes it |
| `confidence` | Model confidence resampled to H×W; optional, not calibrated probability |
| `sky` | Nearest-neighbor sky mask; optional |
| `intrinsics` | 3×3 K for canonical frame dimensions; optional |
| `world_to_camera` | Homogeneous 4×4 OpenCV extrinsics; optional |
| `camera_to_world` | Inverse of `world_to_camera`; optional |

Native window files contain `depth`, optional `confidence` and `sky`, `frame_indices`, optional `intrinsics_native`, `intrinsics_source`, `world_to_camera`, and `camera_to_world`. They retain predictions for overlapping frames, including predictions not selected for the final per-frame sequence. Load with `numpy.load(path, allow_pickle=False)`.

`depth_units="estimated_meters"` is only recorded if the model returns `is_metric`. Other models are recorded as `relative_model_units`. Use the nested checkpoint for this task; a standalone `DA3METRIC-LARGE` prediction requires focal-length scaling and is not a substitute for the nested pipeline. [Metric and nested implementation](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/model/da3.py)

## Coordinates, resizing and window boundaries

DA3 cameras use OpenCV axes: x right, y down, z forward, with world-to-camera extrinsics. Depth is z along that camera's forward axis. A point is reconstructed as `X_camera = depth * inverse(K) @ [u, v, 1]`, then transformed with `camera_to_world`. For 3D Blender comparison, convert camera axes and align each reconstruction's world frame explicitly. This repository's [Blender exporter](blender_export.md) provides validated camera-z depth for [direct pixel comparison](depth_comparison.md); only an independently verified ray-distance output needs conversion. [Official output convention](https://github.com/ByteDance-Seed/Depth-Anything-3#-basic-usage)

The z-depth interpretation is also confirmed by the upstream point-cloud exporter, which multiplies depth by `inverse(K) @ [u,v,1]` without normalizing that ray. [Pinned unprojection implementation](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/utils/export/glb.py)

All canonical input images must have the same dimensions. The adapter chooses the resize mode that keeps the full image: first the longest side is resized, then dimensions are rounded to multiples of 14. No crop is applied. It saves both the upstream intrinsic scaling transform and the half-pixel-center sampling transform. K is returned to canonical dimensions with `diag(source_width/native_width, source_height/native_height, 1) @ K_native`, matching upstream's K resize convention. Depth is bilinearly resampled; this restores pixel correspondence, not lost detail or sharper boundaries. [Pinned preprocessing implementation](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/utils/io/input_processor.py)

Each window has an independent local world frame, named `da3_chunk_XXXXXX`. Its camera translations and depth have the same scale, but **poses from different windows must not be concatenated as one global trajectory**. Metric scaling can also vary between windows. The selected per-frame output uses the prediction with the most context on both sides of the frame; ties prefer the earlier window. It does not blend or silently rescale predictions. Boundaries can have visible jumps.

`overlap_diagnostics.json` reports median depth ratios and relative differences for the exact same images predicted in adjacent windows, excluding sky and invalid depths. It also reports residual disagreement after a diagnostic scalar fit; that fit is never applied to the saved data. Large discrepancies are a reason to inspect geometry or rerun with more context. Treating this output as a single smooth global trajectory would require a separate robust pose/scale alignment and loop-closure step. VGGT-Ω provides the separate reconstruction intended for the first global camera comparison.

Moving hands or objects can violate the static multiview scene assumption. They may produce inconsistent depth or pull camera estimates away from the static background. Inspect these regions separately with the SAM3 masks; confidence alone is not proof that they are geometrically correct.

## Depth preview

After inference finishes, run:

```bash
.venv-depth/bin/scene-recon preview-depth
```

This writes `outputs/depth_preview.mp4` and `outputs/depth_preview.scale.json`. The video displays RGB next to depth, with source frame IDs, timestamps, and DA3 window IDs. One linear color range is used across the entire clip: the 2nd and 98th percentiles of a spatial sample from every frame. Colors saturate outside that range, and invalid/sky pixels are black. The sidecar records the color range, units, input hashes, and duration. `--max-height` controls the rendered preview size; it does not alter any saved prediction.

The preview requires a complete DA3 index, matching manifest hash and timestamps, and exactly one depth record per canonical frame. Existing output files are protected from replacement; pass a new `--output` path to rerun. Variable frame timing is preserved at microsecond resolution. A duplicate of the final image at the endpoint, lasting one microsecond, ensures FFmpeg holds the last source frame for its recorded duration. Use the source manifest and NPZ files, rather than decoded preview frame counts or colors, for quantitative comparisons.

## Verification scope

Local checks cover chunk coverage, source-to-native pixel mapping, matrix conversion, overlap diagnostics, file export and resume behavior with synthetic predictions. The actual model and peak GPU memory must still be checked on the A6000; no full model inference is claimed from a machine without that GPU and downloaded weights.
