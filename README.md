# IMG_1711 reconstruction measurements

Prepare three independent sources of evidence for matching a video with a Blender scene:

| Stage | Result | Scope |
|---|---|---|
| VGGT-Ω | Joint camera estimates, intrinsics, depth/confidence, colored point cloud | Frames sampled across the entire video; arbitrary scale |
| Depth Anything 3 | Depth for every frame, confidence and camera estimates | Overlapping windows; metric estimates with the default nested model; local camera coordinates per window |
| SAM3 | Separate detected instance masks and optional concept unions | Every frame, prompted objects **and surfaces**; tracking IDs are local to each processing chunk |

The source video has been copied to `data/IMG_1711.MOV`. All **698 frames** have been extracted into `data/frames` at **1080×1920**, upright, with HDR tone mapping and original presentation timestamps. See [the contact sheet](data/frames/contact_sheet.jpg). The clip lasts about 23.27 seconds and includes a hand moving containers.

**Status:** CPU tests, real-video extraction, and real Blender camera/geometry/depth exports have run locally. Both unchanged Blender baselines have all 698 camera poses and four depth/RGB preview frames under `outputs/blender_preview/`. VGGT alignment, an offline 3D viewer, and DA3-versus-Blender depth comparison are implemented. CUDA inference, model accuracy, runtime, and peak A6000 memory have **not** been tested here. No model predictions have been generated locally yet.

**Current first run: geometry and depth only.** Follow [the remote A6000 commands](docs/remote_geometry.md) for clone/pull, input verification, `setup_a6000.sh geometry`, and `run_geometry.sh`. The video, canonical frames and two Blender originals are now tracked in this checkout; generated outputs and weights remain Git-ignored. SAM3 and Cryptomatte can be handled later. See [VGGT-to-Blender alignment and depth comparison](docs/geometry_comparison.md) for the complete next stage.

## Install on the A6000 host

The geometry setup targets **Ubuntu 22.04 / CUDA 12.4**, using Python 3.10–3.12 with venv support, Git, and FFmpeg/ffprobe with `zscale` and `tonemap`. The reported remote's Python 3.12.11 and NVIDIA driver 550.54.14 fit this profile. The installer uses CUDA-enabled PyTorch wheels; it does not install or change the system toolkit, driver, or OS packages. Allocate tens of GB for environments, weights, and outputs.

```bash
CUDA_FLAVOR=cu124 PYTHON_BIN=python3 bash scripts/setup_a6000.sh geometry
```

This creates `.venv-vggt` and `.venv-depth` with PyTorch 2.6.0, torchvision 0.21.0 and, for DA3, xformers 0.0.29.post3 from the CUDA 12.4 wheel index. It pins the upstream sources, checks imports and small CUDA kernels, and saves resolved versions under `outputs/environment`. Transitive dependencies are resolved at installation rather than claimed to be a fully locked environment. Individual geometry stages can be installed with `vggt` or `depth`.

If you ran the earlier CUDA 12.6 installer, use the migration command in [the remote guide](docs/remote_geometry.md): `--recreate-envs` archives existing project environments and builds fresh ones. Inputs, checkpoints and results remain in place. SAM3 has a separate explicit `cu126`/Python 3.12 profile; the CUDA 12.4 geometry setup does not install it.

Download the **512 in-the-wild checkpoint** from the public [wincentIsMe/VGGT-Omega mirror](https://huggingface.co/wincentIsMe/VGGT-Omega):

```bash
.venv-vggt/bin/hf download wincentIsMe/VGGT-Omega vggt_omega_1b_512.pt --local-dir checkpoints/vggt-omega
```

The mirror's model card describes an unmodified copy of the 512 checkpoint; it does not require the official repository's per-account access approval. Model code still comes from the pinned official GitHub source. The 416 reproduction checkpoint is for published benchmark comparisons and is not the selected model. DA3 downloads the public `depth-anything/DA3NESTED-GIANT-LARGE-1.1` model automatically at the pinned revision recorded by the adapter. See [SAM3 setup](docs/sam3.md) when that stage is needed. Model licenses remain those of their publishers.

## Run

All commands below run from this folder. Run models sequentially so they do not compete for GPU memory.

The prepared frames are portable. If extracting a different source or regenerating them, use a **new empty** output directory:

```bash
.venv-vggt/bin/scene-recon extract --video data/IMG_1711.MOV --output data/frames_new
```

Do not force a nominal FPS: each model reads the common frame manifest. The original iPhone stream is variable-rate; its portrait orientation and HDR transfer function need to be respected. On macOS, an FFmpeg build without `zscale` must be replaced or selected using `--ffmpeg /path/to/ffmpeg`.

Start with a small joint VGGT run, then a denser one:

```bash
.venv-vggt/bin/scene-recon vggt --checkpoint checkpoints/vggt-omega/vggt_omega_1b_512.pt --max-frames 12 --output outputs/vggt_smoke
.venv-vggt/bin/scene-recon vggt --checkpoint checkpoints/vggt-omega/vggt_omega_1b_512.pt
.venv-depth/bin/scene-recon depth
.venv-sam/bin/scene-recon sam --checkpoint checkpoints/sam3/sam3.pt
```

Or, after reviewing the SAM prompt inventory, use `bash scripts/run_all.sh`. It extracts only if the manifest is absent, then launches each model, catalogs the detected instances, and builds a depth preview. Existing VGGT/SAM output directories are protected against accidental mixing; use a new `--output` to rerun. DA3 resumes matching runs from completed window caches and rejects changed settings in an existing output directory.

Defaults:

- VGGT: every tenth frame plus the last frame, maximum 72; one joint reconstruction, max image side 512, BF16. For this video that selects 71 frames. Increase sampling density once the initial result works.
- DA3: 16-frame windows, overlap 4, processing resolution 504. Every frame gets depth. The native maps are upsampled back to 1080×1920; upsampling does not add geometric detail.
- SAM3: 120-frame chunks, one text concept per session, separate instance PNGs enabled. This bounds sequence memory but resets instance IDs at chunk boundaries. `--chunk-size 0` requests continuous tracking and may require substantially more memory. The number of concepts and detected objects affects cost.

If you hit CUDA OOM, lower VGGT `--max-frames`, DA3 `--chunk-size` (keep `--overlap` smaller), or SAM `--chunk-size`, using a new output directory. Full model runs require checkpoint access and the GPU; local CPU tests do not validate their accuracy.

## Segmentation targets

Your intended granularity is **distinct whole items plus surfaces**, with lids/handles/labels normally belonging to their parent item. Read [segmentation strategy and modes](docs/segmentation_strategy.md) before editing [the target inventory](configs/objects.json).

SAM3 finds instances of a prompted concept; it does not guarantee that every scene item is discovered automatically. This project's current SAM CLI uses **text prompts**, stores every accepted instance, and also writes category unions. Visual-example and interactive point prompting are upstream SAM3 capabilities; this repository does not yet provide an interactive annotation UI.

The broad inventory is a first pass. Review missed items, duplicates across concepts, surface boundaries, and over-segmented parts. A mask for `bottles` is a category union; use individual instance tracks when comparing a single Blender bottle. Chunk-local IDs must be explicitly associated before treating them as one object across the whole clip. See [instance assembly](docs/instances.md).

```bash
.venv-sam/bin/scene-recon catalog-instances
# Review the catalog and write a track-to-physical-object mapping, then:
.venv-sam/bin/scene-recon assemble-instances --mapping configs/reviewed_instances.json
```

The mapping file is created after reviewing actual model outputs; it cannot be filled accurately before inference. Assembly requires explicit frame coverage and explicit reviewed absence, rather than treating unreviewed missing detections as absent objects.

## Inspect depth and geometry

```bash
.venv-depth/bin/scene-recon preview-depth
.venv-depth/bin/scene-recon measure --frame 0 --pixels 400,800 700,1100
```

The first command creates an RGB/depth preview using one fixed color scale across the clip. The second reports two sampled visible 3D surface points and their separation; coordinates are source-image pixels. Choose actual points of interest in the extracted frame. Raw `.npz` arrays are the quantitative data; preview colors are for inspection.

`outputs/vggt/point_cloud.ply` can be opened in a point-cloud viewer or imported into Blender. Its scale and world orientation need alignment before comparison. Do not interpret arbitrary VGGT units as meters. DA3's default model estimates meters, but those values need validation against a known measurement in the scene. Each DA3 window has its own world coordinates; the adapter deliberately does not concatenate their camera trajectories into a falsely global path. Repeated-frame depth disagreements are saved in `overlap_diagnostics.json`.

Use the iPhone's nominal focal length only as a calibration prior. It does not correct lens distortion or stabilization crops. Compare effective pixel intrinsics/FOV, preserving the model's original predictions and both Blender baselines. See [camera calibration and fair comparison](docs/camera_calibration.md).

## Blender handoff and comparison

Both supplied scenes are preserved in `blender_scenes/`, with source hashes in [baselines.json](blender_scenes/baselines.json). Open the local [uninformed viewer](outputs/blender_preview/before_vqa/viewer.html) or [VQA-informed viewer](outputs/blender_preview/vqa_informed/viewer.html). These initially show Blender geometry/cameras; an aligned VGGT cloud is added after inference.

With real results in `outputs/vggt` and `outputs/depth`, run from this repository:

```bash
bash scripts/compare_geometry.sh align
bash scripts/compare_geometry.sh depth
```

These compare both baselines using the existing Blender previews. Outputs go under `outputs/comparison/{before_vqa,vqa_informed}/`: `alignment/viewer.html`, camera metrics, and `depth/index.html` with depth/error images and CSV metrics. [The full workflow](docs/geometry_comparison.md) includes recreating portable Blender exports, rendering all 698 depth frames, landmark alignment and held-out validation. The default depth comparison retains declared physical units; DA3 remains an estimate, not measured ground truth.

For masks later, give the other chat [the Blender mask contract](docs/blender_handoff.md), `data/frames/manifest.json`, and the agreed object inventory. Cryptomatte and render-mask generation remain in that workflow.

After receiving named binary visible masks:

```bash
.venv-sam/bin/scene-recon compare-masks \
  --predicted outputs/assembled/masks \
  --reference data/blender/masks \
  --objects microwave toaster_oven jar_01
```

For a deliberate category-union comparison, use `--predicted outputs/sam/masks` with the same category grouping on the Blender side. Metrics include per-frame IoU, Dice, centroid displacement, and area, plus RGB overlays (green intersection, red SAM only, blue Blender only). Missing masks and wrong image dimensions fail. Both-empty frames are excluded from mean IoU rather than increasing the score. A paired `metadata.json`, if present, must match the source-manifest hash and be complete.

Optional `--ignore-masks outputs/sam/masks/hand` excludes hand pixels from both masks and records the exclusion. Report this separately from the unmodified comparison: it changes what the score measures. A moving jar requires an animated Blender counterpart for an exact frame-by-frame match. High silhouette overlap is evidence of visible projection agreement, not proof of correct hidden geometry or absolute scale.

## Outputs and implementation details

- [VGGT-Ω adapter and conventions](docs/vggt_omega.md)
- [Depth Anything 3 windows, scales, and arrays](docs/depth_anything3.md)
- [SAM3 prompts, tracking, and mask export](docs/sam3.md)
- [Blender frame/mask contract](docs/blender_handoff.md)
- [Blender camera/mesh/depth export](docs/blender_export.md)
- [VGGT alignment and validation](docs/alignment.md)
- [DA3-versus-Blender depth metrics](docs/depth_comparison.md)

Base extraction/comparison/testing tools need only NumPy, Pillow, and FFmpeg; model imports are lazy:

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
scene-recon doctor
```

Source APIs were checked against the official [VGGT-Ω](https://github.com/facebookresearch/vggt-omega), [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), and [SAM3](https://github.com/facebookresearch/sam3) repositories. Exact reviewed commits are recorded in the model documentation and installer.

