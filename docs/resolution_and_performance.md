# Choose resolution and measure runtime

Increasing model input resolution can preserve smaller image features, but its effect on reconstruction accuracy, GPU memory and runtime must be measured. The current VGGT quality run uses 350 views at 384×688. DA3 processes all 698 frames in overlapping windows and saves depth back on the 1080×1920 source grid. Resampling a smaller prediction to that grid does not recover detail the model never saw.

The commands below run from the remote repository root after pushing these changes and pulling them there. They select physical GPU 7 using `CUDA_VISIBLE_DEVICES=7`; confirm that card is available for the job. Changing code or settings does not change an inference process already running. Run successive experiments sequentially on that GPU, preserving existing outputs.

## VGGT: resolution, attention work and memory

For this portrait video, the pinned loader produces the following grids. Token counts exclude the 17 camera/register tokens added per view. The work column squares the patch-token ratio while keeping the **same frame count**:

| Setting | Model image, width×height | Patch tokens per view | Relative tokens | Relative patch-pair attention work |
|---|---:|---:|---:|---:|
| `balanced 512` | 384×688 | 1,032 | 1× | 1× |
| `balanced 768` | 576×1024 | 2,304 | 2.23× | 4.98× |
| `balanced 1024` | 768×1360 | 4,080 | 3.95× | 15.63× |
| `max_size 1920` | 1088×1920 | 8,160 | 7.91× | 62.52× |

The last row is close to source resolution: 1080 is rounded to 1088 to fit 16-pixel patches. It is not an unchanged 1080×1920 input. [Pinned VGGT loader](https://github.com/facebookresearch/vggt-omega/blob/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6/vggt_omega/utils/load_fn.py).

**The attention-work ratios are not measured wall-time multipliers.** Including the special tokens makes the last ratio about 60.76×. Projection, MLP, depth-head, loading and export work scale differently, and efficient attention kernels need not allocate a full quadratic attention matrix. The adapter's timings are needed to determine the actual slowdown. [Pinned attention implementation](https://github.com/facebookresearch/vggt-omega/blob/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6/vggt_omega/models/layers/attention.py).

**350 views at 1088×1920 cannot fit the current implementation on a 48 GiB GPU.** The aggregator retains four feature tensors. Even assuming two bytes per feature, their minimum storage is:

```text
4 layers × 350 views × (8,160 + 17) tokens × 2,048 features × 2 bytes
= 43.67 GiB
```

The retained float32 RGB input adds 8.17 GiB, giving **51.84 GiB before model weights, live attention tensors or depth heads**. This is a lower bound derived from the implementation, not an extrapolated benchmark. Supporting that run would require changing how inference stores/computes features or using more suitable hardware. Reducing views changes the experiment and its temporal evidence. [Pinned aggregator and cached layers](https://github.com/facebookresearch/vggt-omega/blob/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6/vggt_omega/models/aggregator.py).

Larger input grids are accepted by the loader, but the released checkpoint and recommended examples specify resolution 512. Higher-resolution inference is an experiment; the inspected documentation does not establish better geometry at 768, 1024 or 1920. More pixels cannot repair blur, occlusion or inconsistent moving objects. [Official checkpoint and usage](https://github.com/facebookresearch/vggt-omega/blob/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6/README.md).

## Benchmark the same frames before scaling up

For one A6000 with approximately 47 GiB free, the only configuration confirmed by the completed project run is **350 views at 384×688**. Its older `index.json` does not contain a measured memory peak, so it does not establish the maximum capacity. Reasonable first higher-resolution trials with the current unmodified model are:

| Mode / resolution | Actual width×height | Initial view cap | Total patch + special tokens versus the completed run |
|---|---:|---:|---:|
| `balanced 768` | 576×1024 | 128 | 81% |
| `balanced 1024` | 768×1360 | 64 | 71% |
| `max_size 1920` | 1088×1920 | 32 | 71% |

These are conservative starting experiments, **not measured capacity limits or guarantees against OOM**. Reducing the total tokens leaves room for larger per-image encoder/depth-head operations, but those operations and attention work do not scale solely with total tokens. Each cap samples the full clip, including both endpoints. Fewer views can miss brief observations, so retain the 350-view baseline and compare geometry as well as visual density. A direct middle-resolution trial is:

```bash
CUDA_VISIBLE_DEVICES=7 .venv-vggt/bin/scene-recon vggt \
  --checkpoint checkpoints/vggt-omega/vggt_omega_1b_512.pt \
  --preset quality --stride 1 --max-frames 64 \
  --preprocess-mode balanced --image-resolution 1024 \
  --output outputs/vggt_1024_64
```

The command uses the same confidence/export settings as the baseline to avoid attributing a filtering change to resolution. Read its recorded GPU peaks before increasing the view cap. To isolate resolution effects from view-count effects, use the matched-frame benchmark below.

Run a matched comparison at 512, 768 and 1024:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/benchmark_vggt_resolution.sh
```

The helper defaults to **24 frames** selected across the whole clip using stride 1 and a 24-frame cap. Each resolution gets exactly the same source frame IDs. It writes logs and separate result directories under `outputs/vggt_resolution_benchmark`; each successful run's `index.json` records actual dimensions, timings and CUDA memory peaks. It runs experiments sequentially. These small runs measure the resolution tradeoff at 24 views, not the runtime of a 350-view reconstruction.

To also try the near-source-resolution grid:

```bash
CUDA_VISIBLE_DEVICES=7 \
OUTPUT_ROOT=outputs/vggt_resolution_benchmark_full \
bash scripts/benchmark_vggt_resolution.sh --include-full
```

`--include-full` adds `max_size 1920`; it does not request all 698 frames. `FRAMES` changes the shared frame count, and `OUTPUT_ROOT` selects a different result root. For example, a separate 71-view comparison is:

```bash
CUDA_VISIBLE_DEVICES=7 FRAMES=71 \
OUTPUT_ROOT=outputs/vggt_resolution_benchmark_71 \
bash scripts/benchmark_vggt_resolution.sh
```

Choose a fresh output root for an independent repetition. Keep the same frame count when attributing timing or geometric differences to resolution. A 24-view timing cannot be scaled linearly to 350 views because global attention also couples the views. Compare the resulting camera/landmark/depth agreement, then increase views at a useful resolution while checking measured memory. Retain the 350-view/512 run as a separate baseline.

## DA3: increase input detail without dropping video frames

The current `upper_bound_resize` path scales the longest side, then rounds dimensions to multiples of 14:

| `--process-res` | Model input, width×height | Pixels relative to 504 |
|---|---:|---:|
| 504 | 280×504 | 1× |
| 756 | 420×756 | 2.25× |
| 1008 | 574×1008 | 4.10× |
| 1918 (near source size) | 1078×1918 | 14.65× |

At 1008, the intermediate width is 567, and the nearest-multiple helper rounds that tie upward to 574. These are image-pixel ratios, not predicted runtime or memory multipliers. [Pinned DA3 preprocessing](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/3d835ec1a5802d64a8b8b15f817a1ab54809bfe4/src/depth_anything_3/utils/io/input_processor.py).

First profile two windows at the higher resolution while retaining 16-frame windows and four-frame overlap:

```bash
CUDA_VISIBLE_DEVICES=7 .venv-depth/bin/scene-recon depth \
  --process-res 1008 --chunk-size 16 --overlap 4 \
  --compression-level 1 --stop-after-windows 2 --output outputs/depth_1008
```

The profile retains an incomplete index. Rerun the same command **without `--stop-after-windows 2`** to reuse those caches and finish all 698 frames. The limit counts processed windows, including cache hits, so use a fresh directory when measuring new inference. Use a new output directory when changing model resolution or window settings. A 16-frame window at 1008 has not been benchmarked on this A6000 here. Saved source-grid depth is still 1080×1920, but now comes from a larger native prediction. Overlap supplies context and disagreement diagnostics; this adapter does not average predictions, align window scales or stitch local camera worlds.

`--process-res 1918` is a near-native-resolution experiment (the nearest allowed multiple of 14 to 1920). It has about 14.65 times the input pixels of 504, and may require smaller windows. It is not a measured memory or accuracy guarantee. Evaluate 1008 first, then use the same profiling procedure with another fresh output directory if more detail is needed.

## Separate inference time from CPU export time

DA3 now prints a timing breakdown for each window and saves `timings.json`. It separates model setup, inference, native-cache loading/writing, source-grid resampling, frame writing, diagnostics and metadata work, and records written bytes and available CUDA peak measurements. The report describes the current invocation, including whether windows were inferred or reused from cache. A cached run is not a fresh inference benchmark.

`--compression-level` controls **lossless NPZ storage**, independently of model settings:

| Level | Behavior |
|---|---|
| `1` (new default) | Light compression to reduce CPU cost while retaining smaller files |
| `6` | The earlier NumPy compression level |
| `0` | Store arrays without compression; less compression work, larger writes |
| `2`–`9` | Other lossless compression levels; measure the tradeoff on the actual host |

Array values, dtypes and model predictions are preserved at every level. A compression-only change can reuse completed native inference caches in the same output directory when all model/input settings match. It does not require another GPU inference for those cached windows. New native chunks use the requested compression; existing cache files can remain at their original level. A process already running continues with its original settings.

A local **synthetic-array export benchmark**, not real DA3 inference or the remote filesystem, measured:

| Storage | Write time per frame | File size |
|---|---:|---:|
| Deflate level 6 | 0.459 s | 13.36 MB |
| Deflate level 1 | 0.237 s | 13.40 MB |
| Uncompressed level 0 | 0.0034 s | 20.74 MB |

The fixture used source-sized synthetic depth, confidence and mask arrays with standard NPZ writes. Timings used local buffered writes without forcing durable disk synchronization. Compression depends on the real arrays and storage system; these figures show a possible CPU saving, **not a promise of twice-as-fast end-to-end DA3**. Inspect the remote report to determine whether inference, compression, resampling or storage is dominant. Uncompressed writes can be slower overall when disk or network throughput is the limiting resource.

For all 698 source frames, uncompressed float32 depth and confidence plus a boolean validity mask require about **13.03 GB** of raw array data; an additional boolean sky mask raises this to **14.47 GB**. These decimal-GB figures exclude native caches, archive overhead, metadata and other outputs. Reserve extra space for those files and for keeping multiple resolution experiments.
