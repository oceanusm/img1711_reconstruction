"""Encode Blender frame JPEGs and validate the source's stored and displayed timing/layout."""
from pathlib import Path
import argparse
import json
import subprocess
import tempfile

from match_render_timing import retime


def stream_info(path, count=False):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0']
    if count:
        command += ['-count_frames']
    command += ['-show_entries',
                'stream=width,height,nb_frames,nb_read_frames,r_frame_rate,avg_frame_rate,time_base,duration_ts:stream_side_data=rotation',
                '-of', 'json', str(path)]
    stream = json.loads(subprocess.check_output(command))['streams'][0]
    stream['rotation'] = next((row['rotation'] for row in stream.get('side_data_list', [])
                               if 'rotation' in row), 0)
    return stream


def encode(source, frame_dir, out):
    source, frame_dir, out = (Path(p).expanduser().resolve() for p in (source,frame_dir,out))
    s = stream_info(source)
    rotation = int(s['rotation'])
    n = int(s['nb_frames'])
    frames = [frame_dir/f'frame_{i:04d}.jpg' for i in range(1,n+1)]
    missing = [p.name for p in frames if not p.is_file()]
    if missing:
        raise ValueError(f'Missing {len(missing)} rendered images, starting at {missing[0]}')
    restore_sensor = {0:[], -90:['transpose=cclock'], 90:['transpose=clock'],
                      180:['hflip','vflip'], -180:['hflip','vflip']}
    if rotation not in restore_sensor:
        raise ValueError(f'Unsupported source display rotation {rotation}')
    filters = restore_sensor[rotation] + ['scale=in_range=full:out_range=tv:out_color_matrix=bt709','format=yuv420p']
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as t:
        coded = Path(t)/'coded.mp4'
        tagged = Path(t)/'tagged.mp4'
        checked = Path(t)/'checked.mp4'
        subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-framerate',s['r_frame_rate'],
                    '-start_number','1','-i',str(frame_dir/'frame_%04d.jpg'),'-frames:v',str(n),
                    '-vf',','.join(filters),'-c:v','libx264','-crf','17','-preset','veryfast',
                    '-pix_fmt','yuv420p','-color_range','tv','-color_primaries','bt709',
                    '-colorspace','bt709','-color_trc','iec61966-2-1','-an',str(coded)],check=True)
        subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-display_rotation:v:0',str(rotation),
                    '-i',str(coded),'-map','0:v:0','-c:v','copy',str(tagged)],check=True)
        result = retime(source,tagged,checked)
        r = stream_info(checked,count=True)
        for k in ['width','height','rotation','r_frame_rate','avg_frame_rate','duration_ts','time_base']:
            if r[k] != s[k]:
                raise ValueError(f'Source and render disagree on {k}: {s[k]} and {r[k]}')
        if int(r['nb_read_frames']) != n:
            raise ValueError(f'Decoder counted {r["nb_read_frames"]} frames, expected {n}')
        checked.replace(out)
    result.update(stored_dimensions=[r['width'],r['height']], display_rotation_degrees=rotation,
                  decoded_frames=int(r['nb_read_frames']), image_source='Blender authored geometry')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path)
    ap.add_argument('frame_dir',type=Path)
    ap.add_argument('output',type=Path)
    args = ap.parse_args()
    print(json.dumps(encode(args.source,args.frame_dir,args.output),indent=2))
