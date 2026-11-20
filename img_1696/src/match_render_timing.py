"""Losslessly transfer the source display timestamps to a nominal-frame-rate MP4 render.

Usage: python match_render_timing.py source.mov cfr_render.mp4 render.mp4
Requires ffmpeg with the setts bitstream filter's prescale option and ffprobe.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import struct
import subprocess
import tempfile


def inspect(path):
    which = 'stream=time_base,r_frame_rate,avg_frame_rate,duration_ts,width,height:packet=pts'
    data = json.loads(subprocess.check_output(['ffprobe', '-v','error','-select_streams','v:0',
                  '-show_streams','-show_packets','-show_entries',which,'-of','json',str(path)]))
    return data['streams'][0], sorted(int(p['pts']) for p in data['packets'])


def presentation_duration(path, duration):
    """Match QuickTime's end edit. Container time scales are set to source ticks."""
    with path.open('r+b') as f:
        def update_atoms(start, end):
            pos = start
            while pos+8 <= end:
                f.seek(pos)
                size, tag = struct.unpack('>I4s',f.read(8))
                head = 8
                if size == 1:
                    size = struct.unpack('>Q', f.read(8))[0]
                    head = 16
                if size == 0:
                    size = end-pos
                if size < head or pos+size > end:
                    raise ValueError('Invalid MP4 atom')
                body = pos+head
                if tag in (b'moov', b'trak', b'edts'):
                    update_atoms(body, pos+size)
                elif tag in (b'mvhd',b'tkhd',b'elst'):
                    f.seek(body)
                    version = f.read(1)[0]
                    if version not in (0,1):
                        raise ValueError('Unsupported MP4 atom version')
                    if tag == b'elst':
                        f.seek(body+4)
                        if struct.unpack('>I',f.read(4))[0] != 1:
                            raise ValueError('Expected one video presentation edit')
                        at = body+8
                    elif tag == b'tkhd':
                        at = body + (28 if version else 20)
                    else:
                        at = body + (24 if version else 16)
                    f.seek(at)
                    f.write(struct.pack('>Q' if version else '>I',duration))
                pos += size
        update_atoms(0,path.stat().st_size)


def retime(source, rendered, out):
    src, pts = inspect(source)
    video, recorded = inspect(rendered)
    if not len(pts) == len(recorded):
        raise ValueError('Render and source have different frame counts')
    if pts[0] != 0 or recorded[0] != 0:
        raise ValueError('Expected zero-based presentation timestamps')
    scale = Fraction(src['time_base'])
    if scale.numerator != 1:
        raise ValueError('Expected reciprocal integer source time base')
    original_base = Fraction(video['time_base'])
    converted = [round(p*original_base/scale) for p in recorded]
    if not all(b>a for a,b in zip(converted, converted[1:])):
        raise ValueError('Source time base cannot distinguish the input frames')
    offsets = [p-q for p,q in zip(pts,converted)]
    events = [(q,d-prior) for q,d,prior in zip(converted[1:],offsets[1:],offsets) if d != prior]
    def expression(field):
        terms = [f'{change:+d}*gte({field}\\,{at})' for at,change in events]
        return field + ''.join(terms)
    opts = f'setts=time_base={scale}:prescale=1:pts={expression("PTS")}:dts={expression("DTS")}'
    out.parent.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        rewritten = Path(temp)/'timed.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(rendered),
            '-map','0:v:0','-c:v','copy','-map_metadata','-1','-bsf:v',opts,'-movflags','+faststart',
            '-movie_timescale',str(scale.denominator),'-video_track_timescale',str(scale.denominator),
            str(rewritten)],check=True)
        # iPhone MOV edit duration may truncate part of the last sample.
        actual, times = inspect(rewritten)
        if times != pts:
            raise ValueError('Retimed sample timestamps did not match')
        if int(actual['duration_ts']) != int(src['duration_ts']):
            presentation_duration(rewritten, int(src['duration_ts']))
        actual, times = inspect(rewritten)
        if times != pts or int(actual['duration_ts']) != int(src['duration_ts']):
            raise ValueError('Failed to match presentation duration')
        rewritten.replace(out)
    return dict(frame_count=len(pts),source_time_base=str(scale),source_avg_fps=src['avg_frame_rate'],
                final_avg_fps=actual['avg_frame_rate'],duration_ticks=src['duration_ts'],
                pts_match=True,compressed_frames_copied=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ('source','rendered','output'):
        ap.add_argument(arg,type=Path)
    args=ap.parse_args()
    print(json.dumps(retime(args.source.expanduser(),args.rendered,args.output),indent=2))
