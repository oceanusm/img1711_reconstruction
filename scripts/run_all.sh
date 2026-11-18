#!/usr/bin/env bash
# Separate processes release VRAM between the three models.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
if [[ ! -f data/frames/manifest.json ]]; then
  .venv-vggt/bin/scene-recon extract
fi
.venv-vggt/bin/scene-recon vggt --checkpoint checkpoints/vggt-omega/vggt_omega_1b_512.pt
.venv-depth/bin/scene-recon depth
.venv-sam/bin/scene-recon sam --checkpoint checkpoints/sam3/sam3.pt
.venv-sam/bin/scene-recon catalog-instances
.venv-depth/bin/scene-recon preview-depth

