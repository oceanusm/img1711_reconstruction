#!/usr/bin/env bash
# Joint reconstruction of IMG_1696, independently of the IMG_1711 runs.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    -h|--help)
      cat <<'HELP'
Usage: bash scripts/run_img1696_vggt.sh [--dry-run]

Runs up to 450 views spread across IMG_1696 with balanced 512 preprocessing,
BF16, and a single joint reconstruction. A successful fit in GPU memory is
not guaranteed; settings are never reduced automatically after a failure.

Overrides:
  MAX_FRAMES        View cap, at least 2 (default 450; no unlimited setting).
  IMAGE_RESOLUTION  Positive multiple of 16 (default 512).
  PREPROCESS_MODE   balanced or max_size (default balanced).
  VGGT_OUTPUT       Default: outputs/img1696/vggt_<cap>_<mode><resolution>.
  FRAMES_MANIFEST   Default: data/img1696/frames/manifest.json.
  CHECKPOINT        Default: checkpoints/vggt-omega/vggt_omega_1b_512.pt.
  SCENE_RECON_BIN   Default: .venv-vggt/bin/scene-recon.

Set CUDA_VISIBLE_DEVICES=7 to select physical card 7. The wrapper preserves
that environment variable and does not select a GPU itself. Inputs must
already be extracted. Use a fresh output directory for each reconstruction;
the model runner refuses a nonempty directory, including an interrupted run.
--dry-run prints the command without running the model or writing files.
HELP
      exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

MAX_FRAMES=${MAX_FRAMES:-450}
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-512}
PREPROCESS_MODE=${PREPROCESS_MODE:-balanced}
# Bound the number of digits before shell arithmetic to avoid integer overflow.
if [[ ! "$MAX_FRAMES" =~ ^[0-9]{1,9}$ ]] || ((10#$MAX_FRAMES < 2)); then
  echo 'MAX_FRAMES must be an integer of at least 2 (at most 9 digits); unlimited selection is disabled.' >&2
  exit 2
fi
if [[ ! "$IMAGE_RESOLUTION" =~ ^[0-9]{1,9}$ ]] || ((10#$IMAGE_RESOLUTION < 16 || 10#$IMAGE_RESOLUTION % 16 != 0)); then
  echo 'IMAGE_RESOLUTION must be a positive multiple of 16 (at most 9 digits).' >&2
  exit 2
fi
case "$PREPROCESS_MODE" in
  balanced|max_size) ;;
  *) echo 'PREPROCESS_MODE must be balanced or max_size.' >&2; exit 2 ;;
esac
# Normalize leading zeroes so equivalent settings resolve to the same directory.
MAX_FRAMES=$((10#$MAX_FRAMES))
IMAGE_RESOLUTION=$((10#$IMAGE_RESOLUTION))
VGGT_OUTPUT=${VGGT_OUTPUT:-outputs/img1696/vggt_${MAX_FRAMES}_${PREPROCESS_MODE}${IMAGE_RESOLUTION}}
FRAMES_MANIFEST=${FRAMES_MANIFEST:-data/img1696/frames/manifest.json}
CHECKPOINT=${CHECKPOINT:-checkpoints/vggt-omega/vggt_omega_1b_512.pt}
SCENE_RECON_BIN=${SCENE_RECON_BIN:-.venv-vggt/bin/scene-recon}

command=("$SCENE_RECON_BIN" vggt --checkpoint "$CHECKPOINT" --manifest "$FRAMES_MANIFEST"
  --preset quality --stride 1 --max-frames "$MAX_FRAMES"
  --preprocess-mode "$PREPROCESS_MODE" --image-resolution "$IMAGE_RESOLUTION"
  --precision bf16 --point-stride 1 --max-points 5000000 --confidence-percentile 50
  --output "$VGGT_OUTPUT")
if $DRY_RUN; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
if [[ ! -f "$FRAMES_MANIFEST" ]]; then
  echo "Missing frames manifest: $FRAMES_MANIFEST" >&2
  echo 'Extract IMG_1696 first, or set FRAMES_MANIFEST to its existing manifest.' >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing VGGT-Omega checkpoint: $CHECKPOINT" >&2
  echo 'Install the VGGT model assets first, or set CHECKPOINT to the downloaded weights.' >&2
  exit 1
fi
if ! command -v "$SCENE_RECON_BIN" >/dev/null 2>&1; then
  echo "Missing scene-recon executable: $SCENE_RECON_BIN" >&2
  echo 'Install the VGGT environment first, or set SCENE_RECON_BIN to its executable.' >&2
  exit 1
fi
echo "IMG_1696: up to $MAX_FRAMES views across the clip, $PREPROCESS_MODE $IMAGE_RESOLUTION, BF16."
echo "Manifest: $FRAMES_MANIFEST"
echo "Output: $VGGT_OUTPUT"
exec "${command[@]}"
