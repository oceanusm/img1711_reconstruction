#!/usr/bin/env bash
# Compare both supplied Blender baselines with completed model outputs.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
STAGE=${1:-all}
case "$STAGE" in
  -h|--help)
    echo 'Usage: bash scripts/compare_geometry.sh [all|align|depth]'
    echo 'Requires completed Blender exports and the requested model outputs.'
    echo 'Overrides: PYTHON_BIN, BLENDER_EXPORT_ROOT, VGGT_OUTPUT, DA3_OUTPUT, COMPARISON_OUTPUT.'
    echo 'Defaults: outputs/blender_preview, outputs/vggt, outputs/depth, outputs/comparison.'
    exit 0 ;;
  all|align|depth) ;;
  *) echo 'Usage: bash scripts/compare_geometry.sh [all|align|depth]' >&2; exit 2 ;;
esac
if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN=python3
  for candidate in .venv-tools/bin/python .venv-depth/bin/python .venv-vggt/bin/python; do
    if [[ -x "$candidate" ]]; then PYTHON_BIN=$candidate; break; fi
  done
fi
BLENDER_EXPORT_ROOT=${BLENDER_EXPORT_ROOT:-outputs/blender_preview}
VGGT_OUTPUT=${VGGT_OUTPUT:-outputs/vggt}
DA3_OUTPUT=${DA3_OUTPUT:-outputs/depth}
COMPARISON_OUTPUT=${COMPARISON_OUTPUT:-outputs/comparison}
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Fail before creating reports if a required stage has not been transferred.
for scene in before_vqa vqa_informed; do
  if [[ ! -f "$BLENDER_EXPORT_ROOT/$scene/index.json" ]]; then
    echo "Missing Blender export: $BLENDER_EXPORT_ROOT/$scene/index.json; see docs/geometry_comparison.md." >&2
    exit 1
  fi
done
if [[ "$STAGE" != depth && ! -f "$VGGT_OUTPUT/index.json" ]]; then
  echo "Missing VGGT result: $VGGT_OUTPUT/index.json" >&2; exit 1
fi
if [[ "$STAGE" != align && ! -f "$DA3_OUTPUT/index.json" ]]; then
  echo "Missing DA3 result: $DA3_OUTPUT/index.json" >&2; exit 1
fi

for scene in before_vqa vqa_informed; do
  if [[ "$STAGE" == all || "$STAGE" == align ]]; then
    "$PYTHON_BIN" -m scene_reconstruction align-vggt \
      --vggt "$VGGT_OUTPUT" --blender-export "$BLENDER_EXPORT_ROOT/$scene" \
      --output "$COMPARISON_OUTPUT/$scene/alignment"
    "$PYTHON_BIN" -m scene_reconstruction view-alignment \
      --vggt "$VGGT_OUTPUT" --blender-export "$BLENDER_EXPORT_ROOT/$scene" \
      --alignment "$COMPARISON_OUTPUT/$scene/alignment/alignment.json" \
      --output "$COMPARISON_OUTPUT/$scene/alignment/viewer.html"
  fi
  if [[ "$STAGE" == all || "$STAGE" == depth ]]; then
    "$PYTHON_BIN" -m scene_reconstruction compare-depth \
      --depth-index "$DA3_OUTPUT/index.json" \
      --blender-index "$BLENDER_EXPORT_ROOT/$scene/index.json" \
      --output "$COMPARISON_OUTPUT/$scene/depth"
  fi
done
