"""Preserve video timing with both older and newer FFmpeg command-line APIs."""
from __future__ import annotations

import re
import subprocess


def video_sync_args(ffmpeg: str, mode: str) -> list[str]:
    """Inspect supported options instead of assuming a particular release number."""
    legacy_values = {"passthrough": "0", "vfr": "2"}
    if mode not in legacy_values:
        raise ValueError(f"Unsupported video synchronization mode: {mode}")
    try:
        help_text = subprocess.check_output(
            [ffmpeg, "-hide_banner", "-h", "full"],
            stderr=subprocess.STDOUT, text=True, timeout=30,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Could not inspect FFmpeg options: {ffmpeg}\n{error.output[-2000:]}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Timed out inspecting FFmpeg options: {ffmpeg}") from error
    options = set(re.findall(r"(?m)^\s*(-[A-Za-z0-9_]+)(?=\s|\[|$)", help_text))
    if "-fps_mode" in options:
        return ["-fps_mode", mode]
    if "-vsync" in options:
        return ["-vsync", legacy_values[mode]]
    raise RuntimeError(
        f"FFmpeg exposes neither -fps_mode nor -vsync: {ffmpeg}. "
        "Select a supported FFmpeg binary with --ffmpeg /path/to/ffmpeg."
    )
