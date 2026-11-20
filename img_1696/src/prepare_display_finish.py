"""Recreate the television's architectural finish without the onscreen characters.

Requires the source MOV, fitted camera_track.npz, vggt_alignment.json, ffmpeg,
NumPy and Pillow. Only positively identified architectural fragments of frames
425 and 449 are admitted. Other pixels are authored gradients, skyline blocks,
water, masonry, unoccupied glazing, doors, and road.
"""
from pathlib import Path
from io import BytesIO
import math
import json
import subprocess
import argparse
import numpy as np
from PIL import Image,ImageDraw,ImageFilter

S=Path(__file__).resolve().parent
W,H=960,540


def rectify(video,f,matrix,track):
    encoded=subprocess.check_output(['ffmpeg','-hide_banner','-loglevel','error','-i',str(video),
            '-vf',f'select=eq(n\\,{f})','-frames:v','1','-f','image2pipe','-vcodec','mjpeg','-q:v','2','-'])
    im=Image.open(BytesIO(encoded)).convert('RGB')
    y=np.array([1,-1,-1,1])*.8305+5.49
    z=np.array([1,1,-1,-1])*.465+1.542
    x=4.5834-.0374*(y-5.665)-.095
    xyz=np.column_stack((x,y,z));raw=(xyz-matrix[:3,3])@np.linalg.inv(matrix[:3,:3]).T
    e=np.linalg.inv(track['camera_to_world_cv'][f]);k=track['intrinsics'][f]
    uvw=(raw@e[:3,:3].T+e[:3,3])@k.T
    observed=uvw[:,:2]/uvw[:,2,None];a=[];b=[]
    for (x,y),(u,v) in zip([(0,0),(W,0),(W,H),(0,H)],observed):
        a.extend([[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]]);b.extend([u,v])
    coeff=np.linalg.solve(a,b)
    return im.transform((W,H),Image.Transform.PERSPECTIVE,coeff,Image.Resampling.BICUBIC)


def generate(video,out):
    if video.stem.lower()!='img_1696':raise ValueError('TV material annotations are specific to img_1696.')
    alignment=json.loads((S.parent/'vggt_alignment.json').read_text())
    m=np.array(alignment['matrix']);track=np.load(S/'camera_track.npz')
    left=rectify(video,425,m,track);right=rectify(video,449,m,track)
    yy,xx=np.mgrid[0:H,0:W]
    stop=[0,42,95,137,146,158,199,267,473,480,514,539]
    colors=np.array([[38,39,128],[94,40,163],[200,76,218],[158,77,202],[92,56,164],[194,94,211],
                     [107,49,158],[84,35,137],[74,33,109],[27,24,62],[47,31,102],[68,37,135]])
    RGB=np.stack([np.interp(yy,stop,colors[:,j]) for j in range(3)],axis=2)
    for cx,cy,rx,ry,power in [(130,32,64,48,31),(270,46,70,54,39),(468,36,50,42,78),(735,29,57,53,87),(932,114,115,35,95)]:
        halo=power*np.exp(-((xx-cx)/rx)**2-((yy-cy)/ry)**2)
        RGB+=halo[:,:,None]*[1,.77,.69]
    im=Image.fromarray(RGB.clip(0,255).astype('uint8'));d=ImageDraw.Draw(im)
    rng=np.random.default_rng(1696)
    for layer in range(2):
        x=-9
        while x<W:
            w=int(rng.integers(10,33));top=int(rng.integers(68,127))+(16 if layer else 0)
            fill=(93,57,153) if not layer else (61,40,124)
            d.rectangle((x,top,x+w,145),fill)
            d.rectangle((x,top+2,x+3,144),tuple(min(255,c+11) for c in fill))
            if w>22:d.rectangle((x+4,top-4,x+w-5,top),fill)
            if w>20 and not layer:
                for ry in range(top+5,143,9):
                    for rx in range(x+6,x+w-2,6):
                        if rng.random()<.42:d.rectangle((rx,ry,rx+2,ry+2),(148,87,172))
            x+=w+int(rng.integers(0,9))
    for _ in range(110):
        y=int(rng.integers(151,230));x=int(rng.integers(-10,W));v=int(rng.integers(75,155))
        d.line((x,y,x+int(rng.integers(8,57)),y),fill=(v,51+v//5,min(212,69+v//2)))
    # The newsroom, weather inset and rooftop figures are replaced with windows and masonry.
    d.polygon([(309,472),(309,250),(336,250),(340,201),(481,201),(491,238),(517,238),(517,471)],fill=(104,60,153))
    for a,b in [(203,209),(213,216),(239,247),(251,255),(348,355),(363,368),(466,473)]:
        d.rectangle((315,a,517,b),(134,77,177));d.line((315,b+1,517,b+1),fill=(52,32,103),width=3)
    for x in [317,341,476,498]:
        d.rectangle((x,251,x+9,464),(95,54,145))
        d.line((x+1,251,x+1,463),fill=(180,111,196),width=2)
    for c in [(352,220,469,235),(350,266,469,342),(350,379,469,460)]:
        x0,y0,x1,y1=c;d.rectangle(c,(58,42,106),outline=(146,86,177),width=3)
        for y in range(y0+4,y1-2):
            t=(y-y0)/(y1-y0)
            d.line((x0+4,y,x1-4,y),fill=(int(82+49*t),int(54+15*t),int(143+30*t)))
        for x in range(x0+18,x1-3,23):d.rectangle((x,y0+2,x+3,y1-2),(58,41,105))
        if y1-y0>20:d.line((x0+3,(y0+y1)//2,x1-4,(y0+y1)//2),fill=(62,40,108),width=3)
    def retain(source,polygon,shift=(0,0),flipped=False):
        mask=Image.new('L',(W,H));ImageDraw.Draw(mask).polygon(polygon,fill=255)
        layer=source
        if flipped:
            mask=mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT);layer=layer.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        im.paste(layer,shift,mask)
    # Clock pavilion and two symmetrical wings, all sourced from the stone façade.
    wing=[(79,472),(79,304),(87,298),(96,248),(136,248),(144,300),(164,302),(164,472)]
    retain(left,wing,(-75,0))
    retain(left,wing,(35,0))
    pavilion=[(154,475),(154,281),(143,278),(143,268),(163,263),(185,188),(243,188),(260,261),
              (278,264),(278,272),(266,278),(269,474)]
    retain(left,pavilion,(-72,0))
    retain(left,wing,(116,0))
    # Fill the west edge with continuations of the same unoccupied window bays.
    retain(left,[(83,319),(111,319),(111,466),(83,466)],(-83,0))
    # Granite hall and the courthouse pediment/columns, clipping off all roof skyline.
    retain(right,[(514,469),(514,191),(520,184),(690,184),(694,209),(692,466)])
    retain(right,[(689,474),(690,234),(702,229),(708,220),(862,161),(940,193),(940,474)])
    # Extend the cutoff courthouse facade to the right edge using its verified outer bay.
    retain(right,[(905,252),(936,252),(936,468),(905,468)],(32,0))
    d=ImageDraw.Draw(im)
    # All unresolved silhouettes behind the central hall glazing are removed.
    for x in [566,590,611]:
        d.rectangle((x,244,x+13,367),(105,56,157))
        for y in range(247,367):
            t=(y-247)/120
            d.line((x+2,y,x+11,y),fill=(int(127+56*t),int(65+32*t),int(163+30*t)))
        d.line((x,290,x+13,290),fill=(59,38,108),width=4)
    d.rectangle((559,257,627,348),(76,46,130))
    for x in (564,585,606):
        for y0,y1 in [(263,295),(302,340)]:
            for y in range(y0,y1):
                t=(y-y0)/(y1-y0)
                d.line((x,y,x+13,y),fill=(int(100+46*t),int(63+22*t),int(155+25*t)))
    # No vehicle or contents are retained through the open central portal.
    d.pieslice((547,397,637,476),180,360,fill=(68,43,110),outline=(83,53,131),width=3)
    d.rectangle((548,438,635,471),(68,43,110))
    for y in range(434,471,5):d.line((551,y,632,y),fill=(105,63,147),width=2)
    # Foreground is deliberately empty, including all vehicles, pavement silhouettes and displays.
    d.rectangle((0,474,W,480),(43,29,74))
    for y in [482,491,512,521]:d.line((0,y,W,y),fill=(75,43,122),width=2)
    # Recreate the temperature graphic with linework, omitting the source portrait.
    d.rectangle((349,267,472,336),fill=(54,29,111),outline=(110,75,167),width=2)
    d.line([(363,289),(379,289),(368,310)],fill=(235,199,244),width=3)
    d.line([(384,294),(388,289),(396,289),(400,293),(398,298),(385,309),(401,309)],fill=(235,199,244),width=3)
    d.ellipse((402,284,408,290),outline=(235,199,244),width=2)
    d.ellipse((428,292,444,308),outline=(239,199,238),width=2)
    for i in range(8):
        a=i*np.pi/4
        d.line((436+11*np.cos(a),300+11*np.sin(a),436+15*np.cos(a),300+15*np.sin(a)),fill=(239,199,238),width=2)
    im=im.filter(ImageFilter.GaussianBlur(.42))
    out.parent.mkdir(parents=True,exist_ok=True);im.save(out,'JPEG',quality=95,subsampling=0)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('source',type=Path);cli.add_argument('--out',type=Path,default=S/'assets/display-clean.jpg')
    args=cli.parse_args();generate(args.source.expanduser(),args.out)
