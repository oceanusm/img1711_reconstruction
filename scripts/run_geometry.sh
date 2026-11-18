#!/usr/bin/env bash
# Geometry-only stages, sequentially, in their dedicated environments.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
STAGE=${1:-all}
case "$STAGE" in
  -h|--help)
    echo 'Usage: bash scripts/run_geometry.sh [all|vggt|depth|preview]'
    echo 'Defaults: all = VGGT-Omega, full-video DA3, then depth preview.'
    echo 'Requires setup_a6000.sh geometry and the VGGT checkpoint for vggt/all.'
    echo "Existing outputs follow each adapter's protection/resume rules; see docs/remote_geometry.md."
    exit 0 ;;
  all|vggt|depth|preview) ;;
  *) echo 'Usage: bash scripts/run_geometry.sh [all|vggt|depth|preview]' >&2; exit 2 ;;
esac
if [[ ! -f data/frames/manifest.json ]]; then
  if [[ "$STAGE" == preview ]]; then
    echo 'Missing data/frames/manifest.json; transfer the canonical frames used for inference.' >&2
    exit 1
  fi
  EXTRACTOR=.venv-depth/bin/scene-recon
  if [[ "$STAGE" == vggt ]]; then EXTRACTOR=.venv-vggt/bin/scene-recon; fi
  "$EXTRACTOR" extract --video data/IMG_1711.MOV --output data/frames
fi
if [[ "$STAGE" == all || "$STAGE" == vggt ]]; then
  .venv-vggt/bin/scene-recon vggt \
    --checkpoint checkpoints/vggt-omega/vggt_omega_1b_512.pt
fi
if [[ "$STAGE" == all || "$STAGE" == depth ]]; then
  .venv-depth/bin/scene-recon depth
fi
if [[ "$STAGE" == all || "$STAGE" == preview ]]; then
  .venv-depth/bin/scene-recon preview-depth
fi

