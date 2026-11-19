"""Shared sparse evaluation timestamps without renumbering source frames."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import load_manifest, sha256, write_json


def select_frames(manifest, count):
    frames = manifest['frames']
    if isinstance(count, bool) or not isinstance(count, int) or not 2 <= count <= len(frames):
        raise ValueError('Frame count must be between two and the source frame count')
    times = np.asarray([f['time_seconds'] for f in frames], dtype=float)
    targets = np.linspace(times[0], times[-1], count)
    # Constrain each nearest choice to leave enough later frames; endpoints retained.
    selected = [0]
    for position, time in enumerate(targets[1:-1], 1):
        first = selected[-1] + 1
        last = len(frames) - (count - position)
        selected.append(first + int(np.argmin(np.abs(times[first:last + 1] - time))))
    selected.append(len(frames)-1)
    return [dict(frames[i]) for i in selected]


def load_selection(path, manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    selection = json.loads(Path(path).read_text())
    if selection.get('schema_version') != 1 or selection.get('source_manifest_sha256') != sha256(manifest_path):
        raise ValueError('Selection belongs to a different canonical manifest')
    for key in ('width', 'height'):
        if selection.get(key) != manifest[key]:
            raise ValueError('Selection pixel dimensions differ from canonical source')
    frames = selection.get('frames', [])
    ids = [f.get('index') for f in frames]
    if not ids or any(type(i) is not int for i in ids) or ids != sorted(set(ids)):
        raise ValueError('Selection frame IDs must be nonempty, unique, ordered integers')
    canonical = {f['index']: f for f in manifest['frames']}
    for frame in frames:
        if frame['index'] not in canonical or frame != canonical[frame['index']]:
            raise ValueError('Selected frame metadata must exactly match canonical source records')
    return selection, manifest


def run(args):
    path, output = Path(args.manifest).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = load_manifest(path)
    frames = select_frames(manifest, args.count)
    write_json(output, {'schema_version': 1, 'source_manifest_sha256': sha256(path),
        'source_frame_count': len(manifest['frames']), 'width': manifest['width'],
        'height': manifest['height'], 'frames': frames,
        'selection': {'method': 'uniform_timestamps_nearest_unique_source_frames',
                      'requested_count': args.count, 'preserves_original_frame_ids': True,
                      'first_and_last_included': True}})
    print(f'Selected {len(frames)} shared timestamps: {output}')
    return output
