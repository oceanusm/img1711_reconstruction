"""Make a source/geometry comparison at inspected, human-free views.

Usage: python compare_views.py source.mov [preview_directory | render.mp4] sheet.jpg
Requires OpenCV, NumPy and Pillow. The 12 sampled indices and redactions below
belong to the img_1696 scene profile.
"""
import argparse
import hashlib
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

SELECTED = [0,130,218,306,365,435,491,525,605,740,840,965]
CAPTIONS = ['Dining room','Dining and sectional','Window seating','Coffee table and throw',
            'Armchair and street','Television and fireplace','Feature wall and bar',
            'Kitchen and dining','Sectional and console','Decorative console objects',
            'Window and exterior','End table and window']


def font(s,bold=False):
    for path in (('/System/Library/Fonts/Supplemental/Arial Bold.ttf' if bold else '/System/Library/Fonts/Supplemental/Arial.ttf'),
                 ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')):
        if Path(path).exists():
            return ImageFont.truetype(path,s)
    return ImageFont.load_default()


def grab(cap, i):
    cap.set(cv2.CAP_PROP_POS_FRAMES,i)
    ok, bgr=cap.read()
    if not ok:
        raise ValueError(f'Failed to decode frame {i}')
    return Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))


def source_view(cap, frame):
    img = grab(cap,frame)
    # Face-free source inspection. Cover the television content and mirror reflection
    # in frames where reproduced media or reflected content would be ambiguous.
    if frame in (435,491):
        w,h=img.size
        overlay=ImageDraw.Draw(img)
        if frame==435:
            pix=[(0,.036),(1,.123),(1,.589),(0,.626)]
            overlay.polygon([(round(x*w),round(y*h)) for x,y in pix],fill=(73,75,81))
        if frame==491:
            overlay.ellipse([(.729*w,.267*h),(1.142*w,.505*h)],fill=(110,115,120))
    return img


def make_sheet(video,render,out):
    if hashlib.sha256(video.read_bytes()).hexdigest()!='078681b5795fc3141e95cd53e9385c2263286a48e4b8e520f03f5e1a2a8c3b10':
        raise ValueError('Compare views requires the img_1696 inspected source')
    source=cv2.VideoCapture(str(video))
    src_render=cv2.VideoCapture(str(render)) if render.is_file() else None
    tw,th=184,327
    pad,gap,per=24,18,3
    pair=2*tw+4
    w=2*pad+per*pair+(per-1)*gap
    sheet=Image.new('RGB',(w,119+4*(th+67)+24),(245,245,241))
    d=ImageDraw.Draw(sheet)
    d.text((pad,19),'Camera-aligned reconstruction review',fill='#182321',font=font(28,True))
    d.text((pad,58),'Source at left  /  Authored Blender view at right     •     Frame numbers start at 0',fill='#394744',font=font(17))
    d.text((pad,83),'TV content and mirror reflection in the source are masked in the indicated views.',fill='#56625e',font=font(14))
    for j,(f,caption) in enumerate(zip(SELECTED,CAPTIONS)):
        sr=source_view(source,f)
        rr=grab(src_render,f) if src_render else Image.open(render/f'source_{f:04d}.jpg')
        x=pad+(j%per)*(pair+gap)
        y=116+(j//per)*(th+67)
        d.text((x,y),f'F{f:03}   {caption}',fill='#1b3530',font=font(17,True))
        title_y=y+26
        for k,im in enumerate((sr,rr)):
            if im.width>=im.height:
                raise ValueError('Expected automatically oriented portrait images')
            sheet.paste(im.convert('RGB').resize((tw,th),Image.Resampling.LANCZOS),
                   (x+k*(tw+4),title_y))
        label='Source'
        if f in (435,491):label+='  ·  masked'
        d.text((x,title_y+th+4),label,font=font(13),fill='#4e5d58')
        d.text((x+tw+4,title_y+th+4),'Blender',font=font(13),fill='#4e5d58')
    out.parent.mkdir(exist_ok=True,parents=True)
    sheet.save(out,quality=92,optimize=True)
    source.release()
    if src_render:src_render.release()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','render','sheet'):ap.add_argument(name,type=Path)
    a=ap.parse_args()
    make_sheet(a.source.expanduser(),a.render,a.sheet)
