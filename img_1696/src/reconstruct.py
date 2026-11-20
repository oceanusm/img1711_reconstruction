"""Rebuild the authored scene, its reference audit, and its movie from one video path.

Requires Blender, ffmpeg/ffprobe, a C compiler, NumPy, OpenCV and Pillow.
"""
import argparse,json,shutil,subprocess,sys,tempfile
from pathlib import Path
from vg_reference import reference_for

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video',type=Path)
    args=parser.parse_args();video=args.video.expanduser().resolve()
    matched=reference_for(video)
    print(f'Source {video}\nMatching reference {matched}',flush=True)
    profile=Path(__file__).resolve().parent
    out=video.parent.parent/'output'/video.stem
    executable=shutil.which('blender')
    if not executable and Path('/Applications/Blender.app/Contents/MacOS/Blender').exists():
        executable='/Applications/Blender.app/Contents/MacOS/Blender'
    if video.stem.lower()!='img_1696':
        from survey import describe
        from prepare_camera import prepare
        inspection=out/'inspection';describe(video,inspection)
        track=out/'src'/'camera_track.npz';prepare(video,track,None,None)
        if executable:
            with tempfile.TemporaryDirectory(prefix='vggt-survey-') as t:
                script=Path(t)/'survey_stage.py'
                script.write_text('import sys\n'+f'sys.path.insert(0,{str(profile)!r})\n'+'from survey import save_native\n'+f'save_native({str(video)!r},{str(inspection)!r})\n')
                subprocess.run([executable,'--background','--factory-startup','--python-exit-code','1','--python',str(script)],check=True)
        print(f'Prepared the matched reference metadata and camera track at {out}. This recording now needs a scene-specific object model and scale reference.')
        return
    if not executable:raise SystemExit('Blender was not found on PATH.')
    out.mkdir(parents=True,exist_ok=True)
    src=out/'src'
    if src.resolve()!=profile:shutil.copytree(profile,src,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    published_notes=profile.parent/'NOTES.md'
    if published_notes.exists() and published_notes.resolve()!=(out/'NOTES.md').resolve():
        shutil.copy2(published_notes,out/'NOTES.md')
    from render_inspections import render
    from encode_render import encode
    with tempfile.TemporaryDirectory(prefix=video.stem+'-rebuild-') as t:
        work=Path(t);cache=work/'evaluated.npz';frames=work/'frames'
        job=('import sys,bpy\nfrom pathlib import Path\n'
             +f'sys.path.insert(0,{str(src)!r})\n'
             +'from build import build\nfrom render_native import render_frames\nfrom verify_scenes import verify_both\n'
             +f'build({str(video)!r},{str(out)!r},{str(cache)!r})\n'
             +f'verify_both({str(video)!r},{str(out)!r})\n'
             +f'render_frames({str(frames)!r},samples=8)\n')
        script=work/'native_stage.py';script.write_text(job)
        subprocess.run([executable,'--background','--factory-startup','--python-exit-code','1','--python',str(script)],check=True)
        render(video,out,cache)
        report=encode(video,frames,out/'render.mp4');print(report)
        checked_path=out/'inspection/09_scene_validation.json'
        checked=json.loads(checked_path.read_text());checked['encoded_video']=report
        checked_path.write_text(json.dumps(checked,indent=2)+'\n')
        subprocess.run([sys.executable,str(src/'compare_views.py'),str(video),str(out/'render.mp4'),str(out/'inspection/07_video_comparison.jpg')],check=True)
        subprocess.run([sys.executable,str(src/'validate_view_motion.py'),str(video),str(out/'render.mp4'),str(out/'inspection/08_image_motion.json')],check=True)
    print(out)
if __name__=='__main__':main()
