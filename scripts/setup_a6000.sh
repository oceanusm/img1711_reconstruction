#!/usr/bin/env bash
# Linux x86_64 only. Installs project venvs, never OS packages/drivers/toolkits.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

usage() {
  cat <<'USAGE'
Usage: bash scripts/setup_a6000.sh [geometry|vggt|depth|sam|all] [--recreate-envs]
Defaults: geometry, CUDA_FLAVOR=cu124, PYTHON_BIN=python3 (Python 3.10-3.12).
SAM/all requires CUDA_FLAVOR=cu126 and Python 3.12 (PYTHON_BIN=python3.12).
--recreate-envs archives selected existing project venvs before a fresh install.
USAGE
}

TARGET=geometry
TARGET_SET=0
RECREATE_ENVS=0
for argument in "$@"; do
  case "$argument" in
    --help|-h) usage; exit 0 ;;
    --recreate-envs) RECREATE_ENVS=1 ;;
    geometry|vggt|depth|sam|all)
      if (( TARGET_SET )); then usage >&2; exit 2; fi
      TARGET=$argument
      TARGET_SET=1
      ;;
    *) usage >&2; exit 2 ;;
  esac
done

CUDA_FLAVOR=${CUDA_FLAVOR:-cu124}
PYTHON_BIN=${PYTHON_BIN:-python3}
case "$CUDA_FLAVOR" in
  cu124) TORCH_VERSION=2.6.0; VISION_VERSION=0.21.0; XFORMERS_VERSION=0.0.29.post3; CUDA_VERSION=12.4 ;;
  cu126) TORCH_VERSION=2.7.1; VISION_VERSION=0.22.1; XFORMERS_VERSION=0.0.31.post1; CUDA_VERSION=12.6 ;;
  *) echo 'CUDA_FLAVOR must be cu124 or cu126.' >&2; exit 2 ;;
esac
if [[ "$TARGET" == sam || "$TARGET" == all ]]; then
  if [[ "$CUDA_FLAVOR" != cu126 ]]; then
    echo 'SAM3 requires the explicit cu126 profile and Python 3.12. Use geometry for CUDA 12.4.' >&2
    exit 2
  fi
fi

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'This installer targets Linux x86_64 with NVIDIA CUDA. Core tools also run on macOS.' >&2
  exit 1
fi
for executable in "$PYTHON_BIN" git ffmpeg ffprobe nvidia-smi; do
  if ! command -v "$executable" >/dev/null; then
    echo "Required executable not found: $executable" >&2
    exit 1
  fi
done
PYTHON_VERSION=$("$PYTHON_BIN" -c '
import sys
version = sys.version_info[:2]
if not (3, 10) <= version <= (3, 12):
    raise SystemExit("Use Python 3.10-3.12 for the pinned geometry wheel stack")
if sys.argv[1] in ("sam", "all") and version != (3, 12):
    raise SystemExit("SAM3 requires Python 3.12; set PYTHON_BIN=python3.12")
print(".".join(map(str, version)))
' "$TARGET")
CONSTRAINTS="configs/constraints-$CUDA_FLAVOR.txt"
CONSTRAINTS_SHA256=$("$PYTHON_BIN" -c '
import hashlib, pathlib, sys
print(hashlib.sha256(b"".join(pathlib.Path(p).read_bytes() for p in sys.argv[1:])).hexdigest())
' "$CONSTRAINTS" configs/constraints.txt)

NAMES=()
DIRECTORIES=()
URLS=()
REVISIONS=()
if [[ "$TARGET" == all || "$TARGET" == geometry || "$TARGET" == vggt ]]; then
  NAMES+=(vggt)
  DIRECTORIES+=(vendor/vggt-omega)
  URLS+=(https://github.com/facebookresearch/vggt-omega.git)
  REVISIONS+=(48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6)
fi
if [[ "$TARGET" == all || "$TARGET" == geometry || "$TARGET" == depth ]]; then
  NAMES+=(depth)
  DIRECTORIES+=(vendor/Depth-Anything-3)
  URLS+=(https://github.com/ByteDance-Seed/Depth-Anything-3.git)
  REVISIONS+=(3d835ec1a5802d64a8b8b15f817a1ab54809bfe4)
fi
if [[ "$TARGET" == all || "$TARGET" == sam ]]; then
  NAMES+=(sam)
  DIRECTORIES+=(vendor/sam3)
  URLS+=(https://github.com/facebookresearch/sam3.git)
  REVISIONS+=(31cb7e624e4fe95301b1be1d4e809111593d2e45)
fi

profile_metadata() {
  printf '%s\n' "schema=1" "root=$ROOT" "model=$1" "source_revision=$2" \
    "cuda=$CUDA_FLAVOR" "python=$PYTHON_VERSION" "torch=$TORCH_VERSION" \
    "torchvision=$VISION_VERSION" "xformers=$XFORMERS_VERSION" \
    "constraints_sha256=$CONSTRAINTS_SHA256"
}

verify_stack() {
  "$ROOT/.venv-$1/bin/python" -c '
import sys, torch, torchvision
from importlib.metadata import version
expected_python, expected_torch, expected_vision, expected_cuda, model, expected_xformers = sys.argv[1:]
assert ".".join(map(str, sys.version_info[:2])) == expected_python, "Python version changed"
assert torch.__version__.split("+")[0] == expected_torch, "torch version differs from profile"
assert torchvision.__version__.split("+")[0] == expected_vision, "torchvision version differs from profile"
assert torch.version.cuda == expected_cuda, "torch CUDA build differs from profile"
if model == "depth":
    assert version("xformers").split("+")[0] == expected_xformers, "xformers version differs from profile"
' "$PYTHON_VERSION" "$TORCH_VERSION" "$VISION_VERSION" "$CUDA_VERSION" "$1" "$XFORMERS_VERSION"
}

# Check every selected environment before cloning, moving, or installing anything.
# Legacy/partial environments have no completion marker and require explicit recreation.
for i in "${!NAMES[@]}"; do
  name=${NAMES[$i]}
  env_dir="$ROOT/.venv-$name"
  if [[ -L "$env_dir" || ( -e "$env_dir" && ! -d "$env_dir" ) ]]; then
    echo "Refusing non-directory or symlink environment: $env_dir. Move it aside manually." >&2
    exit 1
  fi
  if [[ -d "$env_dir" && "$RECREATE_ENVS" == 0 ]]; then
    if [[ ! -f "$env_dir/.scene-recon-profile" ]] || \
       [[ "$(cat "$env_dir/.scene-recon-profile")" != "$(profile_metadata "$name" "${REVISIONS[$i]}")" ]]; then
      echo "$env_dir has a different, incomplete, or unrecorded installation. Rerun with --recreate-envs to archive it and install $CUDA_FLAVOR." >&2
      exit 1
    fi
    if ! verify_stack "$name"; then
      echo "$env_dir no longer matches its profile. Rerun with --recreate-envs." >&2
      exit 1
    fi
  fi
  directory=${DIRECTORIES[$i]}
  if [[ -e "$directory" || -L "$directory" ]]; then
    if [[ ! -d "$directory/.git" || "$(git -C "$directory" rev-parse HEAD)" != "${REVISIONS[$i]}" ]]; then
      echo "Existing $directory is not the pinned source. Move it aside or inspect it; no automatic reset." >&2
      exit 1
    fi
    if [[ -n "$(git -C "$directory" status --porcelain)" ]]; then
      echo "Uncommitted upstream edits in $directory; refusing an ambiguous installation." >&2
      exit 1
    fi
  fi
done

mkdir -p vendor checkpoints outputs/environment
for i in "${!NAMES[@]}"; do
  if [[ ! -d "${DIRECTORIES[$i]}" ]]; then
    git clone "${URLS[$i]}" "${DIRECTORIES[$i]}"
    git -C "${DIRECTORIES[$i]}" checkout --detach "${REVISIONS[$i]}"
  fi
done

for i in "${!NAMES[@]}"; do
  name=${NAMES[$i]}
  directory=${DIRECTORIES[$i]}
  env_dir="$ROOT/.venv-$name"
  if [[ -d "$env_dir" && "$RECREATE_ENVS" == 1 ]]; then
    backup_dir=$(mktemp -d "$ROOT/.venv-$name.backup-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
    mv -- "$env_dir" "$backup_dir/environment"
    echo "Archived $name environment at $backup_dir/environment (venvs are not runnable at an archived path)."
  fi
  if [[ ! -d "$env_dir" ]]; then
    "$PYTHON_BIN" -m venv "$env_dir"
    "$env_dir/bin/python" -m pip install --upgrade pip wheel 'setuptools<81'
    "$env_dir/bin/python" -m pip install -c "$CONSTRAINTS" --only-binary=:all: \
      "torch==$TORCH_VERSION" "torchvision==$VISION_VERSION" --index-url "https://download.pytorch.org/whl/$CUDA_FLAVOR"
    if [[ "$name" == depth ]]; then
      "$env_dir/bin/python" -m pip install -c "$CONSTRAINTS" --only-binary=:all: \
        "xformers==$XFORMERS_VERSION" --index-url "https://download.pytorch.org/whl/$CUDA_FLAVOR"
    fi
    "$env_dir/bin/python" -m pip install -c "$CONSTRAINTS" -e "$directory" -e "$ROOT" huggingface_hub
    if [[ "$name" == depth ]]; then
      "$env_dir/bin/python" -m pip install -c "$CONSTRAINTS" addict
    elif [[ "$name" == sam ]]; then
      "$env_dir/bin/python" -m pip install -c "$CONSTRAINTS" pycocotools scipy psutil einops opencv-python
    fi
  else
    echo "Reusing $name environment ($CUDA_FLAVOR, Python $PYTHON_VERSION)."
  fi
  verify_stack "$name"
  "$env_dir/bin/python" -m pip check
  case "$name" in
    vggt) "$env_dir/bin/python" -c 'from vggt_omega.models import VGGTOmega; print("VGGT-Omega import OK")' ;;
    depth) "$env_dir/bin/python" -c 'from depth_anything_3.api import DepthAnything3; print("DA3 import OK")' ;;
    sam) "$env_dir/bin/python" -c 'from sam3.model_builder import build_sam3_video_predictor; print("SAM3 import OK")' ;;
  esac
  "$env_dir/bin/python" -m pip freeze > "outputs/environment/$name-freeze.txt"
  "$env_dir/bin/scene-recon" doctor --gpu > "outputs/environment/$name-doctor.json"
  # Publish a completion marker only after package, import, and GPU checks pass.
  profile_metadata "$name" "${REVISIONS[$i]}" > "$env_dir/.scene-recon-profile"
  cp "$env_dir/.scene-recon-profile" "outputs/environment/$name-profile.txt"
done
echo 'Environments ready. See README.md for checkpoint access and run commands.'
