# Independent dense SAM pilot

This experiment builds one reusable set of video mask candidates for IMG_1711,
without Blender geometry or text categories. It is the first step toward comparing
the before-VQA and VGGT-guided scenes using persistent object masks. It does **not**
yet assign candidates to Blender entities or compute entity scores.

Blender-guided grouping remains a useful next experiment. Ten knife/handle/block
tracks can belong to one Blender knife-rack entity without being duplicates of
one another. We keep their masks so this choice can be made later.

## Run on the A6000

Push/pull the updated repository first. Use the existing working `.venv-sam`;
there is no new CUDA installation. From the remote repository directory, test the
native point-prompt and mask-tracking APIs with a small run:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run_dense_sam.sh --smoke
```

This uses the first eight selected frames, one 8×8 discovery grid, and at most two
tracked seeds. It writes `outputs/sam_dense_1711_smoke/`. Check the final
`metadata.json`: `status` should be `complete`, and `raw_track_count` should be
positive. Zero tracks means discovery accepted nothing; it does not test tracking.

Then run the pilot spanning the complete video:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run_dense_sam.sh
```

The output is `outputs/sam_dense_1711_pilot/`. Defaults:

| Setting | Pilot value |
|---|---|
| Video frames | The same 100 timestamps as both Blender comparisons |
| Discovery frames | 8, distributed over the video |
| Point trials | 32×32 independent positive-point prompts per discovery frame |
| Point batch size | 16 independent prompts sharing the image embedding |
| Mask alternatives | Native multimask alternatives, filtered individually |
| Quality / stability | At least 0.85 / 0.90 |
| Minimum area | 64 pixels on the original 1080×1920 grid |
| Tracking budget | Up to 32 proposals per anchor, after within-frame deduplication |
| Tracking batch size | 8 seed masks, propagated forward and backward |
| Crop layers | 0; optional `--crop-layers 1` adds four overlapping crops |

This allows up to 256 raw seed tracks, before reconciling repeated discoveries.
Tracking the same object from several anchors is intentional: agreement provides
evidence for deduplication. Output masks use the original image grid; the model
still performs its own internal resizing. Saved pixel dimensions do not imply
native-resolution detail or perfect boundaries.

The limits make this a pilot, not an exhaustive segmentation. Quality filtering,
the seed budget, and discovery spacing can miss small items. Candidates are ranked
by predicted quality for the tracking budget, so the pilot does not guarantee
spatially balanced coverage. Every quality-passing, non-crop-truncated discovery
mask is saved **before** deduplication and the budget, including deferred seeds.
Use the review results to decide whether more anchors, crops, or a larger budget
would help; changing settings requires a new output directory.

If interrupted, repeat the same command with `--resume`:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run_dense_sam.sh --resume
# For the smoke run, preserve that profile too:
CUDA_VISIBLE_DEVICES=7 bash scripts/run_dense_sam.sh --smoke --resume
```

Completed discovery anchors and tracking batches are reused. The interrupted
unit is rerun. Input images, selection, checkpoint choice and settings must match.
No output directory needs to be cleared. `SAM_CHECKPOINT=/path/to/sam3.pt` selects
an explicit checkpoint; otherwise the native builder uses its existing HF cache.

For planning without GPU/model loading:

```bash
PYTHON_BIN=python3 bash scripts/run_dense_sam.sh --dry-run
```

`--stage discover`, `--stage track`, and `--stage catalog` can separate the stages.
Later stages require the earlier artifacts and `--resume` in the same output.

## What happens to the masks

1. Each grid point is a separate trial, with multiple possible masks. Points in
   the same model batch are **not** combined into one object prompt.
2. Within each anchor, nearly identical masks are deduplicated at IoU ≥0.95 with
   similar area. Containment alone does not merge a handle with a whole knife.
3. Accepted seed masks are tracked through the 100 selected frames. Each seed
   initially has its own ID; SAM is not assumed to recognize a later independent
   discovery as the same object automatically.
4. Whole tracks are reconciled afterward. Two tracks must have median IoU ≥0.90,
   at least three measurable common frames, at least 80% of common frames above
   that IoU threshold, and at least 80% overlap of their nonempty-frame sets.
   Matching uses a 270×480 grid to bound CPU cost. Tiny differences can disappear
   at this resolution, so raw full-resolution masks remain authoritative.
5. Every member of a duplicate group must agree with every other member. The
   canonical result references one whole track, while preserving all raw aliases.
   It never switches to whichever track happens to fit a frame best. Uncertain
   matches and differently sized parts remain separate.

Empty tracker output is recorded as an empty prediction, not proof of absence.
Missing object/frame output is an error rather than silently becoming an empty
mask. Identity switches and granularity errors remain possible and need review.

The backend uses the existing pinned native SAM3 commit
`31cb7e624e4fe95301b1be1d4e809111593d2e45`. The native public video output makes
overlapping object masks mutually exclusive. That would interfere with retaining
both a rack and its knives, so this adapter reads the independent masks from the
pinned internal cache before that operation, then evicts the copied cache entry.
It fails on an incompatible revision. See the pinned
[image predictor](https://github.com/facebookresearch/sam3/blob/31cb7e624e4fe95301b1be1d4e809111593d2e45/sam3/model/sam1_task_predictor.py)
and [video inference](https://github.com/facebookresearch/sam3/blob/31cb7e624e4fe95301b1be1d4e809111593d2e45/sam3/model/sam3_video_inference.py).

Discovery and video models use separate GPU lifetimes. This does not make the
operation free: repeated tracking, PNG writes, and CPU deduplication can dominate
wall time. Runtime metadata records GPU memory peaks when native inference runs.
Actual A6000 timing and segmentation quality require the smoke/pilot runs; local
tests use fake model outputs and verify data flow and failure handling only.

## Pull and inspect

From the local repository directory:

```bash
rsync -avh --progress \
  oceanusm@pluslab02-a6000.cs.ucla.edu:/local1/oceanusm/img1711_reconstruction/outputs/sam_dense_1711_pilot/ \
  outputs/sam_dense_1711_pilot/
open outputs/sam_dense_1711_pilot/catalog/index.html
```

For the smoke result, replace `sam_dense_1711_pilot` with
`sam_dense_1711_smoke` in both paths. The viewer contains its own RGB previews and
mask overlays and works offline. Select canonical tracks, inspect their raw
aliases and seed masks, and scrub the video. No Blender score is shown yet.

Files to retain:

- `discovery/`: all retained seed masks, quality estimates, dedup decisions and
  budget-deferred candidates.
- `tracks/` and `raw_tracks.json`: original full-resolution temporal masks and
  their source frame IDs/timestamps.
- `catalog/`: conservative aliases, pairwise agreement diagnostics and viewer.
- `plan.json`, `manifest.json`, `metadata.json`: reproducibility and run status.

## Follow-up: Blender scaffolding

For each scene independently, a later mapping stage can assign **sets of tracks**
to an entity. For example, a fixed set of ten tracks could map to the rack when
their pixels mostly fall inside its rendered mask across many views. The group
mask in each frame would be the union of those tracks, allowing the same grouping
to be compared throughout the video. A mask that fits competing Blender entities
should remain ambiguous rather than being assigned to both by default.

Containment and IoU answer different questions here: a tiny knife part can have
excellent containment in a rack but low IoU with the whole rack. Containment can
help choose group members; union IoU evaluates the assembled group. Repeated
discoveries can be aliases, while different physical parts can be group members.

Initially this grouping should use the same independently generated candidates
for both scenes. If that proves inadequate, Blender can also help resolve track
fragmentation or choose additional discovery seeds. Those are separate refinements
and are not implemented in this first pilot.
