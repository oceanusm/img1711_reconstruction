#!/usr/bin/env bash
# Run one scene's prepared concepts over the common selected source frames.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
DRY_RUN=0
POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      echo 'Usage: bash scripts/run_entity_sam.sh PREPARED_DIR [OUTPUT_DIR] [--dry-run]'
      echo 'PREPARED_DIR contains manifest.json, prompts.json, entity_candidates.json and preparation.json.'
      echo 'Default output: PREPARED_DIR/sam; output must be new or empty.'
      echo 'Default: one continuous 100-selected-frame session per concept; retain instance masks.'
      echo 'Overrides: SCENE_RECON_BIN, PYTHON_BIN, SAM_CHECKPOINT, SAM_CHUNK_SIZE, SAM_THRESHOLD.'
      echo 'Select a card with CUDA_VISIBLE_DEVICES. Instance identities still require association before scoring.'
      exit 0 ;;
    --*) echo "Unknown option: $arg" >&2; exit 2 ;;
    *) POSITIONAL+=("$arg") ;;
  esac
done
if [[ ${#POSITIONAL[@]} -lt 1 || ${#POSITIONAL[@]} -gt 2 ]]; then
  echo 'Usage: bash scripts/run_entity_sam.sh PREPARED_DIR [OUTPUT_DIR] [--dry-run]' >&2
  exit 2
fi
if [[ -n "${CUDA_VSIIBLE_DEVICES:-}" ]]; then
  echo 'CUDA_VSIIBLE_DEVICES is misspelled; use CUDA_VISIBLE_DEVICES.' >&2
  exit 2
fi
PREPARED=${POSITIONAL[0]}
OUTPUT=${POSITIONAL[1]:-"$PREPARED/sam"}
SCENE_RECON_BIN=${SCENE_RECON_BIN:-.venv-sam/bin/scene-recon}
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x .venv-sam/bin/python ]]; then PYTHON_BIN=.venv-sam/bin/python; else PYTHON_BIN=python3; fi
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m scene_reconstruction.entity_sam validate "$PREPARED"
COMMAND=("$SCENE_RECON_BIN" sam --manifest "$PREPARED/manifest.json" --prompts "$PREPARED/prompts.json"
         --chunk-size "${SAM_CHUNK_SIZE:-0}" --threshold "${SAM_THRESHOLD:-0.5}" --save-instances --output "$OUTPUT")
if [[ -n "${SAM_CHECKPOINT:-}" ]]; then COMMAND+=(--checkpoint "$SAM_CHECKPOINT"); fi
if [[ "$DRY_RUN" == 1 ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
"${COMMAND[@]}"
