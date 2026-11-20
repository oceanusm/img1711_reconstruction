"""Render the clean Blender scene to numbered, display-oriented JPEGs.

Run with Blender's Python. Set the Blender timeline and camera in the scene build
first, then run the separate source timestamp encoder on the resulting directory.
"""
from pathlib import Path
import argparse
import itertools
import sys
import time

import bpy


def complete_jpeg(path):
    if not path.is_file() or path.stat().st_size < 32:
        return False
    with path.open('rb') as handle:
        start = handle.read(2)
        handle.seek(-2, 2)
        return start == b'\xff\xd8' and handle.read(2) == b'\xff\xd9'


def render_frames(folder, frame_numbers=None, samples=4, resume=False):
    """Use the active scene, its lights, materials, textures and camera animation."""
    folder = Path(folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    if scene.camera is None:
        raise ValueError('The scene has no final animation camera')
    if bpy.data.collections.get('VGGT_REFERENCE') is not None:
        raise ValueError('Open the clean scene.blend for the final render')
    first, last, current = scene.frame_start, scene.frame_end, scene.frame_current
    numbers = sorted(set(range(first, last + 1) if frame_numbers is None else frame_numbers))
    if not numbers or numbers[0] < first or numbers[-1] > last:
        raise ValueError(f'Expected Blender frame numbers in [{first}, {last}]')
    if resume:
        numbers = [i for i in numbers if not complete_jpeg(folder / f'frame_{i:04d}.jpg')]
    render = scene.render
    old_values = (render.engine, render.filepath, render.image_settings.file_format,
                  render.resolution_percentage, render.use_file_extension, render.image_settings.quality,
                  scene.eevee.taa_render_samples)
    try:
        render.engine = 'BLENDER_EEVEE_NEXT'
        scene.eevee.taa_render_samples = samples
        render.resolution_percentage = 100
        render.use_file_extension = True
        render.image_settings.file_format = 'JPEG'
        render.image_settings.quality = 95
        render.filepath = str(folder / 'frame_')
        started = time.perf_counter()
        for _, indexed in itertools.groupby(enumerate(numbers), lambda p: p[1]-p[0]):
            group = [number for index, number in indexed]
            scene.frame_start, scene.frame_end = group[0], group[-1]
            bpy.ops.render.render(animation=True)
            missing = [i for i in group if not complete_jpeg(folder / f'frame_{i:04d}.jpg')]
            if missing:
                raise OSError(f'Blender returned with {len(missing)} incomplete frame files')
        return {'frames_rendered': len(numbers), 'engine': render.engine,
                'seconds': time.perf_counter()-started, 'samples': samples,
                'width': render.resolution_x, 'height': render.resolution_y}
    finally:
        scene.frame_start, scene.frame_end = first, last
        scene.frame_set(current)
        (render.engine, render.filepath, render.image_settings.file_format,
         render.resolution_percentage, render.use_file_extension, render.image_settings.quality,
                  scene.eevee.taa_render_samples) = old_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frames_dir', type=Path)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--frames', type=int, nargs='+')
    settings = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    print(render_frames(settings.frames_dir, settings.frames, settings.samples, settings.resume))
