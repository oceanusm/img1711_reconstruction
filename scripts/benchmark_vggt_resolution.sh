#!/usr/bin/env bash
# Same views at every resolution; run sequentially on the selected GPU.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
INCLUDE_FULL=false
DRY_RUN=false
for arg in "$@"; do
  case "$arg" in
    --include-full) INCLUDE_FULL=true ;;
    --dry-run) DRY_RUN=true ;;
    -h|--help)
      echo 'Usage: bash scripts/benchmark_vggt_resolution.sh [--include-full] [--dry-run]'
      echo 'Runs balanced 512/768/1024 with the same 24 frames spread across the entire clip.'
      echo '--include-full adds max_size 1920 (1088x1920 for this video).'
      echo 'Overrides: FRAMES (default 24), OUTPUT_ROOT, SCENE_RECON_BIN, MANIFEST, CHECKPOINT.'
      echo 'Set CUDA_VISIBLE_DEVICES=7 to select card 7. Each run needs a fresh output directory.'
      echo 'Forward times/peak memory are recorded in each index.json; log files capture failures.'
      exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done
FRAMES=${FRAMES:-24}
if [[ ! "$FRAMES" =~ ^[0-9]+$ ]] || ((10#$FRAMES < 2)); then
  echo 'FRAMES must be an integer of at least 2; this benchmark does not accept an unlimited frame cap.' >&2
  exit 2
fi
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/vggt_resolution_benchmark}
SCENE_RECON_BIN=${SCENE_RECON_BIN:-.venv-vggt/bin/scene-recon}
MANIFEST=${MANIFEST:-data/frames/manifest.json}
CHECKPOINT=${CHECKPOINT:-checkpoints/vggt-omega/vggt_omega_1b_512.pt}
FAILED=false
run_case() {
  local name=$1 mode=$2 resolution=$3
  local command=("$SCENE_RECON_BIN" vggt --checkpoint "$CHECKPOINT" --manifest "$MANIFEST"
    --preset quality --stride 1 --max-frames "$FRAMES" --preprocess-mode "$mode"
    --image-resolution "$resolution" --precision bf16 --point-stride 1 --max-points 100000
    --output "$OUTPUT_ROOT/$name")
  if $DRY_RUN; then
    printf '%q ' "${command[@]}"
    printf '\n'
    return
  fi
  if [[ -e "$OUTPUT_ROOT/$name" || -e "$OUTPUT_ROOT/$name.log" ]]; then
    echo "Refusing to reuse benchmark case $OUTPUT_ROOT/$name; choose a fresh OUTPUT_ROOT." >&2
    FAILED=true
    return
  fi
  mkdir -p "$OUTPUT_ROOT"
  if "${command[@]}" 2>&1 | tee "$OUTPUT_ROOT/$name.log"; then
    echo "Completed $name. Read runtime.timings_seconds and GPU peaks in its index.json."
  else
    echo "Failed $name; retained log and continued without reducing the requested settings." >&2
    FAILED=true
  fi
}
run_case balanced_512 balanced 512
run_case balanced_768 balanced 768
run_case balanced_1024 balanced 1024
if $INCLUDE_FULL; then run_case full_1920 max_size 1920; fi
if $FAILED; then exit 1; fi
