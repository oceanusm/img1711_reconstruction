#!/usr/bin/env python3
"""Build the local four-video IMG_1696 comparison, preserving source frame times."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def inspect(path):
    result = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,time_base,duration_ts,nb_frames:stream_side_data=rotation:packet=pts',
        '-of', 'json', str(path)]))
    stream = result['streams'][0]
    base = Fraction(stream['time_base'])
    ticks = sorted(int(packet['pts']) for packet in result['packets'])
    return stream, ticks, [tick * base for tick in ticks]


def ffmpeg(*args):
    subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
                    *map(str, args)], check=True)


def copy_at_source_times(source, output, target_stream, target_ticks):
    """Retimestamp H.264 packets without decoding or re-encoding their pictures."""
    stream, ticks, _ = inspect(source)
    if len(ticks) != len(target_ticks):
        raise ValueError(f'{source}: frame count differs from the source video')
    target_base = Fraction(target_stream['time_base'])
    converted = [round(tick * Fraction(stream['time_base']) / target_base) for tick in ticks]
    offsets = [target - actual for target, actual in zip(target_ticks, converted)]
    events = [(converted[i], offsets[i] - offsets[i - 1])
              for i in range(1, len(ticks)) if offsets[i] != offsets[i - 1]]
    def expression(field):
        return field + (f'{offsets[0]:+d}' if offsets[0] else '') + ''.join(
            f'{delta:+d}*gte({field}\\,{at})' for at, delta in events)
    options = f'setts=time_base={target_base}:prescale=1:pts={expression("PTS")}:dts={expression("DTS")}'
    ffmpeg('-i', source, '-map', '0:v:0', '-c:v', 'copy', '-bsf:v', options,
           '-movflags', '+faststart', '-movie_timescale', target_base.denominator,
           '-video_track_timescale', target_base.denominator, output)


def build(args):
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Choose a new or empty output directory: {output}')
    source = (ROOT / 'data/img1696/IMG_1696.MOV').resolve()
    standard, guided = args.standard_render.resolve(), args.guided_render.resolve()
    smoothed = args.smoothed_render.resolve()
    manifest_path = ROOT / 'data/img1696/frames/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source_stream, source_ticks, source_times = inspect(source)
    expected = [row['time_seconds'] for row in manifest['frames']]
    if len(source_times) != len(expected) or any(abs(float(a) - b) > 1e-6 for a, b in zip(source_times, expected)):
        raise ValueError('Source video timestamps differ from the prepared frame manifest')
    if digest(source) != manifest['source']['sha256']:
        raise ValueError('Source video hash differs from the prepared frame manifest')
    for path in (standard, guided, smoothed):
        if len(inspect(path)[1]) != len(source_times):
            raise ValueError(f'{path}: not one rendered frame per source frame')
    if inspect(guided)[2] != source_times:
        raise ValueError('Guided render timestamps differ from the source video')
    if inspect(smoothed)[2] != source_times:
        raise ValueError('Smoothed-camera render timestamps differ from the source video')

    media = output / 'media'
    media.mkdir(parents=True)
    print('Preparing full-resolution source video from the existing tone-mapped frames...', flush=True)
    # These are the existing upright, full-resolution Mobius-tone-mapped
    # source frames. Reuse their established color treatment and exact PTS.
    # Frame timing corrections are ticks in the source 1/600 time base.
    if Fraction(source_stream['time_base']) != Fraction(1, 600):
        raise ValueError('This IMG_1696 builder expects the source 1/600 time base')
    offsets = [tick - i * 20 for i, tick in enumerate(source_ticks)]
    events = [(i, offsets[i] - offsets[i - 1]) for i in range(1, len(offsets))
              if offsets[i] != offsets[i - 1]]
    expression = 'N*20' + (f'{offsets[0]:+d}' if offsets[0] else '') + ''.join(
        f'{delta:+d}*gte(N\\,{i})' for i, delta in events)
    filters = ('settb=1/600,setpts=' + expression +
               ',scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p')
    ffmpeg('-framerate', '30', '-start_number', '0', '-i', manifest_path.parent / '%06d.jpg',
           '-i', source, '-map', '0:v:0', '-map', '1:a:0?', '-vf', filters,
           '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-bf', '0',
           '-fps_mode', 'passthrough', '-enc_time_base', '1:600',
           '-video_track_timescale', '600', '-movie_timescale', '600',
           '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
           '-color_range', 'tv', '-c:a', 'copy', '-movflags', '+faststart',
           media / 'original.mp4')
    print('Copying renders and matching original-render timestamps...', flush=True)
    copy_at_source_times(standard, media / 'standard.mp4', source_stream, source_ticks)
    shutil.copy2(guided, media / 'vggt.mp4')
    shutil.copy2(smoothed, media / 'vggt-smooth.mp4')

    provenance = {'scene': 'IMG_1696', 'source_manifest_sha256': digest(manifest_path),
                  'frame_count': len(source_times), 'source_duration_seconds': float(
                      source_stream['duration_ts'] * Fraction(source_stream['time_base'])),
                  'videos': {}}
    for name, original in [('original', source), ('standard', standard), ('vggt', guided),
                           ('vggt-smooth', smoothed)]:
        path = media / f'{name}.mp4'
        stream, _, timestamps = inspect(path)
        if timestamps != source_times:
            raise ValueError(f'{name}: browser video timestamps do not match the source')
        rotation = next((s['rotation'] for s in stream.get('side_data_list', []) if 'rotation' in s), 0)
        size = [stream['width'], stream['height']]
        if abs(rotation) == 90:
            size.reverse()
        if size != [manifest['width'], manifest['height']]:
            raise ValueError(f'{name}: displayed dimensions differ from source')
        ffmpeg('-i', path, '-frames:v', '1', '-vf', 'scale=360:640', '-q:v', '3', media / f'{name}.jpg')
        provenance['videos'][name] = {
            'input_path': str(original), 'input_sha256': digest(original),
            'output_sha256': digest(path), 'source_timestamps_match': True,
            'display_size': size, 'stream': stream,
            'processing': {'original': 'Encoded existing full-resolution upright SDR source frames; original audio',
                           'standard': 'Compressed video copied; packet timestamps matched to source',
                           'vggt': 'Byte-identical copy of supplied render',
                           'vggt-smooth': 'Byte-identical copy of supplied VGGT-guided render after camera smoothing'}[name]}
    for name in ('index.html', 'styles.css', 'app.js', 'sync-controller.js'):
        shutil.copy2(ROOT / 'web/video-comparison' / name, output / name)
    config = {'scene': 'IMG_1696', 'title': 'IMG_1696 · Video comparison',
              'duration': provenance['source_duration_seconds'], 'pts': [float(t) for t in source_times]}
    (output / 'config.js').write_text('window.COMPARISON = ' + json.dumps(config) + ';\n')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'All {len(source_times)} timestamps match in all four videos.\nOpen {output / "index.html"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--standard-render', type=Path, default=ROOT / 'blender_scenes/img1696_blender/render.mp4')
    parser.add_argument('--guided-render', type=Path, default=ROOT / 'img_1696/render.mp4')
    parser.add_argument('--smoothed-render', type=Path, default=ROOT / 'img_1696/render_camera_smoothed.mp4')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/comparison_img1696/video')
    build(parser.parse_args())
