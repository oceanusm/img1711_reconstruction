# SAM 3 video masks

This stage uses Meta's original SAM 3 concept detector and tracker, with a separate
session per text concept. It exports a binary union mask for every concept on
every decoded source frame. It also saves instance masks by default. A text
prompt such as `chair` can select several chairs; the union is not a unique
Blender-object identity.

The implementation was checked against official source commit
`31cb7e624e4fe95301b1be1d4e809111593d2e45`. It calls the
[official video predictor](https://github.com/facebookresearch/sam3/blob/31cb7e624e4fe95301b1be1d4e809111593d2e45/sam3/model/sam3_video_predictor.py)
and its
[request API](https://github.com/facebookresearch/sam3/blob/31cb7e624e4fe95301b1be1d4e809111593d2e45/sam3/model/sam3_base_predictor.py).
It does not silently substitute SAM 2 or SAM 3.1.

## Environment and checkpoint

Use the repository's setup script for the ordinary installation. For a manual
installation in a separate Linux environment, run from this project root:

```bash
python3.12 -m venv .venvs/sam3
source .venvs/sam3/bin/activate
python -m pip install --upgrade pip
python -m pip install 'setuptools<81'
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
mkdir -p third_party
git clone https://github.com/facebookresearch/sam3.git third_party/sam3
git -C third_party/sam3 checkout 31cb7e624e4fe95301b1be1d4e809111593d2e45
python -m pip install -e third_party/sam3
python -m pip install 'numpy==1.26.4' pycocotools scipy psutil einops 'opencv-python-headless==4.11.0.86'
python -m pip install -e .
hf auth login
```

Meta documents Python 3.12+, PyTorch 2.7+, and CUDA 12.6+. These pins meet those
requirements and match this project's other environments. Meta's current README
example uses PyTorch 2.10 with CUDA 12.8 wheels; the older pinned environment here
has not yet been exercised on the target GPU. The GPU host needs a compatible
NVIDIA driver. The supplemental packages above cover imports used by the video code
that are not all declared in the base package dependencies. See the
[official installation instructions](https://github.com/facebookresearch/sam3#installation).

Request access and accept the checkpoint terms at
[facebook/sam3](https://huggingface.co/facebook/sam3) using your own account, then
authenticate on the GPU host with `hf auth login`. The first run downloads and
caches `sam3.pt`. Alternatively, supply `--checkpoint /path/to/sam3.pt` if you
already have the authorized checkpoint. Do not place an access token in configs
or shell scripts checked into the repository.

The RTX A6000 is an Ampere GPU with compute capability 8.6 according to
[NVIDIA](https://developer.nvidia.com/cuda/gpus). This adapter uses one selected
CUDA device, the predictor's BF16 inference, and `compile=False`. It uses the
original SAM3 builder's default PyTorch attention path; FlashAttention 3 and the
SAM 3.1 multiplex model are unnecessary. GPU memory use still depends on the
number of detected instances. The code has not been benchmarked on your A6000.

## Prompts and chunk boundaries

The prompts file is ordinary JSON:

```json
{
  "objects": [
    {"name": "chair", "text": "chair", "frame_index": 0},
    {"name": "table", "text": "table", "frame_index": 0}
  ]
}
```

`name` is the stable output-directory name and should match your future Blender
mask label. It accepts letters, digits, underscores and hyphens. `text` is the
SAM3 concept description. `frame_index` is a global frame index from the frame
manifest, not a timestamp. If omitted, the first manifest frame is used.

The adapter processes consecutive chunks and requests CPU offload for frames and
tracking state. The pinned upstream code still copies its preprocessed image
batch to CUDA during initialization; these flags do not guarantee that all video
data stays in host memory. Chunking is the main limit on sequence memory. Within
the chunk containing the requested anchor, propagation covers both
sides of that frame. A boundary anchor uses only the applicable direction;
the last-frame anchor itself uses its initial prompt result. Every other chunk re-prompts the concept
on its first frame. For continuous instance tracking, use `--chunk-size 0`,
which processes the complete video as one session per concept; this can require
considerably more memory. Start with the default bounded chunks on the A6000.
Reduce chunk size if a run exhausts GPU or host memory.

Tracking resets at chunk boundaries. Therefore an instance is identified by the
triple `(name, chunk_start_index, id)`, not by `id` alone. Instances must be
matched manually or with a later association step to establish physical-object
identity across chunks. Concept masks can also change slightly at a boundary.
The adapter keeps all frames; chunking does not subsample the video.

`--threshold` filters detections by the returned SAM3 instance score (default
0.5); it does not change the upstream binary-mask threshold. The metadata keeps
the threshold and prompt definitions. If an inference result has no detections,
its union mask is a valid all-zero image. If propagation omits a frame, the stage
fails instead of writing a fabricated empty mask.

## Output contract

```text
outputs/sam/
  metadata.json
  frames.jsonl
  masks/<name>/<global_frame_index:06d>.png
  instances/<name>/chunk_<first_global_index:06d>/id_<instance_id:06d>/<global_frame_index:06d>.png
```

`metadata.json` records source manifest hash, model code revision if discoverable,
checkpoint setting, prompts, dimensions, parameters and completion status.
A supplied local checkpoint also has its SHA256 recorded; the default
Hugging Face-managed checkpoint is identified by model/file name without a
checkpoint digest in this adapter.
`frames.jsonl` contains one record per frame and concept, including its time,
union area, instance IDs/scores, normalized XYWH boxes, and relative mask paths.
Paths are relative to the stage output directory. Records are grouped by chunk
and concept; use `index` to join them to the source manifest.

All PNGs are grayscale, with 0 for background and 255 for foreground. Dimensions
are exactly those of the canonical orientation-correct source frames. The
upstream model internally resizes images, then returns masks on that original
grid. The adapter rejects mismatched output dimensions and applies no additional
resize, crop, rotation or smoothing. It stages numeric symlinks to the decoded
images, so there is no additional JPEG encoding.

Concept unions may overlap each other because they are inferred independently.
Per-instance PNGs exist only when the instance is returned on that frame;
absence in `frames.jsonl` means no visible predicted instance mask for that ID.
Use `--no-save-instances` to save disk space while retaining union masks and
instance metadata.

For comparison, render Blender masks on this same pixel grid and at manifest
timestamps, using the same visible-object definition. Distinct Blender chairs
should be unioned for comparison to the `chair` concept mask, or paired with the
saved SAM3 instance masks after checking identity. Inspect masks before treating
low overlap as a geometry or camera error: segmentation mistakes, occlusion,
motion blur and chunk resets also affect the score.

The source video includes moving objects. An object leaving a concept mask does
not necessarily indicate a camera error, and a mask from a moving hand or moved
container should not be interpreted as a constraint on a static reconstruction.

The output directory must be empty at the start of a run. A failed run preserves
partial files with `status: failed`; choose a new output directory for the retry.
Only a `status: complete` run promises every frame and concept is present.
