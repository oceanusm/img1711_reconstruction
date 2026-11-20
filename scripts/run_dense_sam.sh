#!/usr/bin/env bash
# Native SAM3 grid proposals and persistent-mask pilot; no Blender inputs.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
if [[ -n "${CUDA_VSIIBLE_DEVICES:-}" ]]; then
  echo 'CUDA_VSIIBLE_DEVICES is misspelled; use CUDA_VISIBLE_DEVICES.' >&2
  exit 2
fi
PYTHON_BIN=${PYTHON_BIN:-.venv-sam/bin/python}
if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null; then
  echo "Python not found: $PYTHON_BIN. Use the existing SAM environment on the A6000." >&2
  exit 1
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
ARGS=()
SMOKE=0
GPU_OPTION=0
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --gpus|--gpus=*|--gpu-index|--gpu-index=*) GPU_OPTION=1; ARGS+=("$arg") ;;
    *) ARGS+=("$arg") ;;
  esac
done
if [[ "${CUDA_VISIBLE_DEVICES+x}" == x && -z "$CUDA_VISIBLE_DEVICES" ]]; then
  echo 'CUDA_VISIBLE_DEVICES is empty; dense SAM needs at least one visible GPU.' >&2
  exit 2
fi
if [[ "$GPU_OPTION" == 0 ]]; then
  # Preserve an explicit visibility restriction, including the previous single
  # GPU invocation. Otherwise use the three cards selected for this experiment.
  ARGS=(--gpus "${CUDA_VISIBLE_DEVICES-2,5,7}" ${ARGS[@]+"${ARGS[@]}"})
fi
if [[ "$SMOKE" == 1 ]]; then
  ARGS=(--output outputs/sam_dense_1711_parallel_smoke --frame-limit 8 --anchor-count 3
        --grid-side 8 --max-per-anchor 2 --track-batch-size 2 ${ARGS[@]+"${ARGS[@]}"})
fi
if [[ -n "${SAM_CHECKPOINT:-}" ]]; then ARGS+=(--checkpoint "$SAM_CHECKPOINT"); fi
"$PYTHON_BIN" -m scene_reconstruction sam-dense "${ARGS[@]}"
