# IMG_1711 reconstruction measurements

Prepare three independent sources of evidence for matching a video with a Blender scene:

| Stage | Result | Scope |
|---|---|---|
| VGGT-Ω | Joint camera estimates, intrinsics, depth/confidence, colored point cloud | Frames sampled across the entire video; arbitrary scale |
| Depth Anything 3 | Depth for every frame, confidence and camera estimates | Overlapping windows; metric estimates with the default nested model; local camera coordinates per window |
| SAM3 | Separate detected instance masks and optional concept unions | Every frame, prompted objects **and surfaces**; tracking IDs are local to each processing chunk |

The source video has been copied to `data/IMG_1711.MOV`. All **698 frames** have been extracted into `data/frames` at **1080×1920**, upright, with HDR tone mapping and original presentation timestamps. See [the contact sheet](data/frames/contact_sheet.jpg). The clip lasts about 23.27 seconds and includes a hand moving containers.

**Status:** CPU tests and real-video extraction have run locally. CUDA inference, model accuracy, runtime, and peak A6000 memory have **not** been tested here. No model predictions have been generated yet. The starting settings target a single 48 GB A6000; use the smoke runs below before full inference.

**Current first run: geometry and depth only.** Follow [the remote A6000 commands](docs/remote_geometry.md) for GitHub clone/pull, separate data transfer, `setup_a6000.sh geometry`, and `run_geometry.sh`. `data/` is Git-ignored, so it will not arrive through GitHub. SAM3 and Cryptomatte can be handled later. The next geometry step is [VGGT-to-Blender alignment and comparison](docs/geometry_comparison.md).

## Install on the A6000 host

Copy this entire folder, including `data`, to a Linux x86_64 machine. Install Python 3.12 with venv support, Git, FFmpeg/ffprobe with `zscale` and `tonemap`, and an NVIDIA driver supporting CUDA 12.6. The installer uses CUDA-enabled PyTorch wheels; it does not install a driver or OS packages. Allocate tens of GB for environments, weights, and outputs.

```bash
bash scripts/setup_a6000.sh all
```

This creates `.venv-vggt`, `.venv-depth`, and `.venv-sam`, pins each upstream source revision and core numerical packages, checks dependencies/CUDA, and saves resolved package versions under `outputs/environment`. Transitive dependencies are resolved during installation rather than claimed to be a fully locked environment. Use `geometry` in place of `all` for only VGGT/DA3. Individual stages can be installed with `vggt`, `depth`, or `sam`.

Request model access with your Hugging Face account for [VGGT-Ω](https://huggingface.co/facebook/VGGT-Omega) and [SAM3](https://huggingface.co/facebook/sam3), then authenticate and download:

```bash
.venv-vggt/bin/hf auth login
.venv-vggt/bin/hf download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir checkpoints/vggt-omega
.venv-sam/bin/hf download facebook/sam3 sam3.pt --local-dir checkpoints/sam3
```

The login uses the normal shared Hugging Face cache. Do not put tokens in source files. DA3 downloads the public `depth-anything/DA3NESTED-GIANT-LARGE-1.1` model automatically at the pinned revision recorded by the adapter. Model licenses remain those of their publishers.

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

## Blender handoff and comparison

Give the other chat [the Blender mask contract](docs/blender_handoff.md), `data/frames/manifest.json`, and the agreed object inventory. The actual scene and render-mask generation remain in that workflow.

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

Base extraction/comparison/testing tools need only NumPy, Pillow, and FFmpeg; model imports are lazy:

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
scene-recon doctor
```

Source APIs were checked against the official [VGGT-Ω](https://github.com/facebookresearch/vggt-omega), [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), and [SAM3](https://github.com/facebookresearch/sam3) repositories. Exact reviewed commits are recorded in the model documentation and installer.

