"""Compare optical motion in the decoded source and geometry render across time.

The metric is a 2D image-motion diagnostic. It is sensitive to scene depth,
feature distributions and rendering differences. It does not calibrate metric scale.
Usage: python validate_view_motion.py source.mov render.mp4 output.json
Dependencies: OpenCV, NumPy, FFmpeg/ffprobe.
"""
import argparse
from pathlib import Path
import subprocess
import json

import cv2
import numpy as np


WIDTH, HEIGHT = 270,480
GRID_ROWS, GRID_COLS = 6,4


def read_frame(pipe):
    required = WIDTH*HEIGHT
    data=pipe.read(required)
    if not data:return None
    if len(data)!=required:raise ValueError('FFmpeg returned an incomplete grayscale image')
    return np.frombuffer(data,np.uint8).reshape(HEIGHT,WIDTH)


def measure_frames(path,raw_input=False):
    command=['ffmpeg','-v','error','-threads','2','-i',str(path),'-map','0:v:0',
        '-vf',f'scale={WIDTH}:{HEIGHT}:flags=area,format=gray','-fps_mode','passthrough',
        '-f','rawvideo','-threads','1','pipe:1']
    result=[]
    if raw_input:
        raw=np.memmap(path,mode='r',dtype='u1').reshape(-1,640,360)
        stream=(cv2.resize(frame,(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA) for frame in raw)
        process=None
    else:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,bufsize=1048576)
        def frames():
            while True:
                im=read_frame(process.stdout)
                if im is None:return
                yield im
        stream=frames()
    it=iter(stream)
    first=next(it)
    for second in it:
        grid=np.full((GRID_ROWS,GRID_COLS,2),np.nan)
        xy=cv2.goodFeaturesToTrack(first,350,.012,5,blockSize=5)
        if xy is not None:
            xy=xy.reshape(-1,2)
            settings=dict(winSize=(17,17),maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,18,.035))
            hit,a,e=cv2.calcOpticalFlowPyrLK(first,second,xy,None,**settings)
            back,b,_=cv2.calcOpticalFlowPyrLK(second,first,hit,None,**settings)
            valid=a.ravel().astype(bool)&b.ravel().astype(bool)
            valid&=(np.linalg.norm(xy-back,axis=1)<.75)&(e.ravel()<25)
            uv=xy[valid]; delta=(hit[valid]-uv)
            col=np.clip((uv[:,0]/WIDTH*GRID_COLS).astype(int),0,GRID_COLS-1)
            row=np.clip((uv[:,1]/HEIGHT*GRID_ROWS).astype(int),0,GRID_ROWS-1)
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    cell=delta[(row==r)&(col==c)]
                    if len(cell)>=3:
                        grid[r,c]=np.median(cell,axis=0)
        result.append(grid)
        first=second
    if process and process.wait()!=0:raise OSError('FFmpeg decode error')
    return np.array(result)


def compare(source,render,width,display_height):
    if source.shape!=render.shape:
        raise ValueError('Decoded render and source lengths differ')
    def summary(data):
        return dict(median=round(float(np.median(data)),3),
                    p90=round(float(np.percentile(data,90)),3),n=len(data))
    errors=[]; magnitude=[];per_frame=[];counts=[]
    for a,b in zip(source,render):
        valid=np.isfinite(a[:,:,0])&np.isfinite(b[:,:,0])
        if valid.sum()<4:
            per_frame.append(None)
            continue
        tensor=(a-b)[valid]*[width/WIDTH,display_height/HEIGHT]
        e=np.linalg.norm(tensor,axis=1)
        mag=np.linalg.norm(a[valid]*[width/WIDTH,display_height/HEIGHT],axis=1)
        errors.extend(e);magnitude.extend(mag);per_frame.append(float(np.median(e)))
        counts.append(int(valid.sum()))
    good=[v for v in per_frame if v is not None]
    return {'total_adjacent_frame_pairs':len(source), 'tested_adjacent_frame_pairs':len(good),
            'source_pixel_motion_magnitude_per_cell':summary(magnitude),
            'source_render_motion_discrepancy_per_cell_px':summary(errors),
            'median_cell_discrepancy_per_frame_pair_px':summary(good),
            'common_cells_per_tested_frame_pair':summary(counts),
            'coarsely_localized_highest_discrepancies':[{'source_frame':i,'median_motion_difference_px':round(per_frame[i],3)}
            for i in sorted([i for i,v in enumerate(per_frame) if v is not None],key=lambda i:-per_frame[i])[:10]],
            'method':'Decode both streams in display orientation. Track corners at 270 x 480 between adjacent frames in each stream independently with bidirectional Lucas-Kanade. Compare median displacement vectors in corresponding cells of a 4 x 6 grid with at least three valid tracks per image per cell and at least four common cells per frame pair. Distances use the full display pixel grid.',
            'scope':'Measures local apparent image motion, jointly affected by camera path, geometry, texture and within-cell feature composition. It does not measure a static 3D or point-correspondence error.'}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source',type=Path);ap.add_argument('render',type=Path);ap.add_argument('result',type=Path)
    args=ap.parse_args();cv2.setNumThreads(2)
    src=measure_frames(args.source)
    ren=measure_frames(args.render)
    cap=cv2.VideoCapture(str(args.source));ok,frame=cap.read()
    if not ok:raise OSError('Source display dimensions unavailable')
    h,w=frame.shape[:2];cap.release()
    args.result.write_text(json.dumps(compare(src,ren,w,h),indent=2)+'\n')
