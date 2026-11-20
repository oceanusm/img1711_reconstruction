"""Rectify the pigment field of the observed dining-room artwork for img_1696."""
from pathlib import Path
from io import BytesIO
import argparse
import subprocess
import numpy as np
from PIL import Image


def prepare(source_video, output):
    if source_video.stem.lower() != 'img_1696':
        raise ValueError('The quadrilateral below belongs to img_1696 only.')
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-i',str(source_video),
         '-vf','select=eq(n\\,539)','-frames:v','1','-f','image2pipe','-vcodec','png','-']
    img=Image.open(BytesIO(subprocess.check_output(cmd))).convert('RGB')
    if img.size != (1080,1920):
        raise ValueError('These artwork observations use the oriented portrait frame.')
    source=np.array([(605,486),(794,461),(782,686),(607,681)],float)
    dest=np.array([(0,0),(480,0),(480,304),(0,304)],float)
    A=[]; b=[]
    for (x,y),(u,v) in zip(dest,source):
        A.extend([[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]])
        b.extend([u,v])
    coeff=np.linalg.solve(A,b)
    tex=img.transform((480,304),Image.Transform.PERSPECTIVE,coeff,Image.Resampling.BICUBIC)
    rgb=np.asarray(tex).astype(float)
    daylight_gain=218/np.quantile(rgb.mean(2),.87)
    out=Image.fromarray((rgb*daylight_gain).clip(0,255).astype('uint8'))
    output.parent.mkdir(parents=True,exist_ok=True)
    out.save(output,quality=94)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('source_video',type=Path)
    cli.add_argument('--out',type=Path,default=Path(__file__).parent/'assets/dining_print.jpg')
    args=cli.parse_args()
    prepare(args.source_video.expanduser(),args.out)
