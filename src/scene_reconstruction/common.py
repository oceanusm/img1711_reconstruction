from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path, *, allow_sparse: bool = False):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("frames"):
        raise ValueError("Expected nonempty schema_version=1 frame manifest")
    ids = [frame["index"] for frame in manifest["frames"]]
    if allow_sparse:
        if any(type(index) is not int or index < 0 for index in ids) or ids != sorted(set(ids)):
            raise ValueError("Manifest frame indices must be unique, ordered, nonnegative integers")
    elif ids != list(range(len(ids))):
        raise ValueError("Manifest frame indices must be contiguous and zero-based")
    times = [frame["time_seconds"] for frame in manifest["frames"]]
    if not all(math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Frame timestamps must be strictly increasing")
    for frame in manifest["frames"]:
        if not (path.parent / frame["file"]).is_file():
            raise FileNotFoundError(path.parent / frame["file"])
    return manifest

