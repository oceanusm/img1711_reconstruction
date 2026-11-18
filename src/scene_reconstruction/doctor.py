from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

from .common import load_manifest


def command_report(command):
    if not shutil.which(command[0]):
        return {"available": False}
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        return {"available": True, "returncode": result.returncode,
                "output": result.stdout.strip(), "error": result.stderr.strip() or None}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": True, "error": str(error)}


def gpu_smoke_check(torch):
    """Launch tiny kernels on the current logical device, not every shared GPU."""
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    with torch.inference_mode():
        matrix = torch.ones((16, 16), device=device)
        product = matrix @ matrix
        if product[0, 0].item() != 16.0:
            raise RuntimeError("CUDA matrix multiplication smoke check returned an incorrect value")
        tensor = torch.ones((1, 1, 8, 16), device=device, dtype=dtype)
        attention = torch.nn.functional.scaled_dot_product_attention(tensor, tensor, tensor)
        if not torch.isfinite(attention).all().item() or not torch.allclose(attention, tensor):
            raise RuntimeError("CUDA scaled-dot-product attention smoke check failed")
        torch.cuda.synchronize(device_index)
    return {"logical_device_index": device_index, "matrix_multiplication": "passed",
            "scaled_dot_product_attention": "passed", "attention_dtype": str(dtype),
            "scope": "small kernels only; not model inference or peak-memory validation"}


def run(args):
    checks = {"python": platform.python_version(), "platform": platform.platform(),
              "ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe")}
    path = Path(args.manifest)
    if path.is_file():
        manifest = load_manifest(path)
        checks["frames"] = {"count": len(manifest["frames"]), "width": manifest["width"], "height": manifest["height"]}
    else:
        checks["frames"] = "Not extracted yet"
    for package in ["scene-reconstruction", "torch", "torchvision", "vggt-omega", "sam3", "depth-anything-3"]:
        try:
            checks[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            checks[package] = "not installed in this environment"
    if args.gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available in this environment; model inference requires the A6000 host")
        checks["gpu"] = [{"name": torch.cuda.get_device_name(i),
                          "memory_GiB": round(torch.cuda.get_device_properties(i).total_memory / 1024 ** 3, 2),
                          "capability": list(torch.cuda.get_device_capability(i))}
                         for i in range(torch.cuda.device_count())]
        checks["torch_cuda"] = torch.version.cuda
        checks["bf16_supported"] = torch.cuda.is_bf16_supported()
        checks["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        checks["nvidia_driver"] = command_report([
            "nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"])
        checks["cuda_toolkit_nvcc"] = command_report(["nvcc", "--version"])
        checks["cuda_version_note"] = (
            "torch_cuda is the PyTorch runtime build. nvcc reports an optional system toolkit. "
            "nvidia-smi's CUDA version reports driver support. These are different values."
        )
        checks["gpu_smoke_check"] = gpu_smoke_check(torch)
    print(json.dumps(checks, indent=2))

