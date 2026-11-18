#!/usr/bin/env bash
# Run on a Linux x86_64 A6000 host, from anywhere. Does not install OS packages.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
TARGET=${1:-all}
PYTHON_BIN=${PYTHON_BIN:-python3.12}
case "$TARGET" in all|geometry|vggt|depth|sam) ;; *) echo 'Usage: bash scripts/setup_a6000.sh [all|geometry|vggt|depth|sam]' >&2; exit 2;; esac
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'This installer targets Linux x86_64 with NVIDIA CUDA. Core tools also run on macOS.' >&2
  exit 1
fi
command -v "$PYTHON_BIN" >/dev/null
command -v git >/dev/null
command -v ffmpeg >/dev/null
command -v ffprobe >/dev/null
command -v nvidia-smi >/dev/null
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12 for these pinned environments"'
mkdir -p vendor checkpoints outputs/environment

checkout() {
  local url=$1 directory=$2 revision=$3
  if [[ ! -d "$directory" ]]; then
    git clone "$url" "$directory"
    git -C "$directory" checkout --detach "$revision"
  elif [[ ! -d "$directory/.git" || "$(git -C "$directory" rev-parse HEAD)" != "$revision" ]]; then
    echo "Existing $directory is not the pinned source. Move it aside or inspect it; no automatic reset." >&2
    exit 1
  fi
  if [[ -n "$(git -C "$directory" status --porcelain)" ]]; then
    echo "Uncommitted upstream edits in $directory; refusing an ambiguous installation." >&2
    exit 1
  fi
}

install_env() {
  local name=$1 directory=$2
  local env_dir="$ROOT/.venv-$name"
  "$PYTHON_BIN" -m venv "$env_dir"
  "$env_dir/bin/python" -m pip install --upgrade pip wheel 'setuptools<81'
  "$env_dir/bin/python" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
  if [[ "$name" == depth ]]; then
    "$env_dir/bin/python" -m pip install xformers==0.0.31.post1 --index-url https://download.pytorch.org/whl/cu126
  fi
  "$env_dir/bin/python" -m pip install -c configs/constraints.txt -e "$directory" -e "$ROOT" huggingface_hub
  if [[ "$name" == depth ]]; then
    "$env_dir/bin/python" -m pip install -c configs/constraints.txt addict
  elif [[ "$name" == sam ]]; then
    "$env_dir/bin/python" -m pip install -c configs/constraints.txt pycocotools scipy psutil einops opencv-python
  fi
  "$env_dir/bin/python" -m pip check
  case "$name" in
    vggt) "$env_dir/bin/python" -c 'from vggt_omega.models import VGGTOmega; print("VGGT-Omega import OK")' ;;
    depth) "$env_dir/bin/python" -c 'from depth_anything_3.api import DepthAnything3; print("DA3 import OK")' ;;
    sam) "$env_dir/bin/python" -c 'from sam3.model_builder import build_sam3_video_predictor; print("SAM3 import OK")' ;;
  esac
  "$env_dir/bin/python" -m pip freeze > "outputs/environment/$name-freeze.txt"
  "$env_dir/bin/scene-recon" doctor --gpu > "outputs/environment/$name-doctor.json"
}

if [[ "$TARGET" == all || "$TARGET" == geometry || "$TARGET" == vggt ]]; then
  checkout https://github.com/facebookresearch/vggt-omega.git vendor/vggt-omega 48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6
  install_env vggt vendor/vggt-omega
fi
if [[ "$TARGET" == all || "$TARGET" == geometry || "$TARGET" == depth ]]; then
  checkout https://github.com/ByteDance-Seed/Depth-Anything-3.git vendor/Depth-Anything-3 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
  install_env depth vendor/Depth-Anything-3
fi
if [[ "$TARGET" == all || "$TARGET" == sam ]]; then
  checkout https://github.com/facebookresearch/sam3.git vendor/sam3 31cb7e624e4fe95301b1be1d4e809111593d2e45
  install_env sam vendor/sam3
fi
echo 'Environments installed. See README.md for checkpoint access and run commands.'

