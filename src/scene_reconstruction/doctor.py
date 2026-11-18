from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

from .common import load_manifest


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
    print(json.dumps(checks, indent=2))

