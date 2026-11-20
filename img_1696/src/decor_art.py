"""Painted finishes traced from the two abstract canvases in source frame 80.
The fitted artwork is specific to img_1696. It contains no source-video pixels.
"""
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSETS=Path(__file__).resolve().with_name('assets')
ASSETS.mkdir(exist_ok=True)
rng=np.random.default_rng(1696)

# Boundaries sampled in the rectified 270 by 216 reference image.
panels=[
([(0,64),(5,50),(11,48),(14,24),(18,21),(23,25),(30,46),(34,51),(40,49),(47,51),(54,58),(60,63),(66,64),(70,71),(81,75),(94,88),(109,95),(122,99),(133,98),(142,100),(157,102),(173,105),(182,106),(196,101),(207,94),(216,94),(229,95),(242,91),(254,82),(266,67),(270,64)],[(0,162),(5,166),(11,180),(19,183),(25,190),(29,187),(36,167),(40,157),(53,157),(61,160),(70,158),(87,160),(102,155),(114,148),(132,144),(143,138),(151,139),(163,130),(176,126),(186,123),(205,122),(215,127),(227,131),(240,140),(246,144),(246,168),(256,172),(269,181)],[(0,3),(40,3),(75,7),(123,3),(145,11),(165,4),(204,5),(239,4),(270,1)],[(0,6),(25,11),(42,5),(61,29),(78,10),(105,8),(127,4),(147,11),(173,4),(212,3),(240,4),(270,3)]),
([(0,94),(11,92),(19,88),(32,77),(43,73),(52,65),(58,67),(66,72),(78,74),(91,75),(102,73),(115,71),(125,69),(133,73),(142,75),(153,72),(165,71),(184,68),(191,65),(198,59),(213,58),(225,55),(232,49),(239,32),(245,27),(257,21),(268,6),(270,4)],[(0,115),(12,117),(24,115),(39,111),(50,107),(62,101),(75,99),(88,99),(100,96),(116,106),(131,112),(149,120),(161,123),(175,122),(185,124),(197,132),(211,141),(224,148),(239,155),(253,166),(267,174),(270,174)],[(0,5),(23,12),(31,34),(47,35),(55,34),(65,19),(80,8),(99,7),(127,4),(147,7),(167,15),(190,8),(230,4),(270,2)],[(0,5),(32,9),(60,8),(85,8),(110,7),(146,15),(176,8),(196,9),(220,6),(258,4),(270,4)])]
def interpolate(points):
    a,b=np.array(points).T
    return np.interp(np.linspace(0,270,768),a,b)
for index,(upper,lower,top_rim,bottom_rim) in enumerate(panels,1):
    cols=np.arange(768)
    noise=np.interp(cols,np.arange(0,774,6),rng.normal(0,.65,129))
    up=interpolate(upper)+noise
    lo=interpolate(lower)+noise[::-1]
    row=np.linspace(0,216,576)[:,None]
    color=np.zeros((576,768,3),float)+np.array([204,183,188])
    white=(row>=up-interpolate(top_rim)) & (row<lo+interpolate(bottom_rim))
    dark=(row>=up) & (row<=lo)
    color[white]=[241,237,230];color[dark]=[28,28,34]
    ridge=(row>up-interpolate(top_rim)-2.4) & (row<up-interpolate(top_rim))
    color[ridge]=[185,160,164]
    color+=rng.normal(0,1.5,(576,768,1))
    Image.fromarray(color.clip(0,255).astype('uint8')).save(ASSETS/f'diptych_{index}.jpg',quality=93)

def font(size):
    for p in ['/System/Library/Fonts/Supplemental/Georgia Bold.ttf','/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf']:
        if Path(p).is_file(): return ImageFont.truetype(p,size)
    return ImageFont.load_default()
def centered(d,text,x,y,size,color):
    d.text((x,y),text,font=font(size),fill=color,anchor='mm')
for i in [0,1]:
    W,H=512,420
    sky=([29,54,95],[145,164,181]) if i==0 else ([178,47,28],[248,190,96])
    grad=np.linspace(0,1,H)[:,None,None]
    pix=np.broadcast_to(np.array(sky[0])+(np.array(sky[1])-sky[0])*grad,(H,W,3)).copy();im=Image.fromarray(pix.astype('uint8'));d=ImageDraw.Draw(im)
    colors=[(83,89,123),(69,93,138),(71,111,158),(47,77,120)] if i==0 else [(124,91,66),(156,121,54),(111,126,55),(60,93,46)]
    for j,c in enumerate(colors):
        xs=np.arange(-16,W+16,13)
        ys=H*(.40+j*.12)+np.sin(xs/(41+8*j)+j)*14+np.cos(xs/(65+j*3))*18+rng.uniform(-4,4,len(xs))
        d.polygon([(int(x),int(y)) for x,y in zip(xs,ys)]+[(W,H),(0,H)],c)
    centered(d,'CATAN',W//2,70,76,'#f6d86a')
    if not i: centered(d,'EXPLORERS & PIRATES',W//2,133,20,'#f3de99')
    im.save(ASSETS/f'game_{i}_top.jpg',quality=93)
    im=Image.new('RGB',(512,108),('#3b4c79','#596032')[i]);d=ImageDraw.Draw(im)
    centered(d,'CATAN',256,43,56,'#f5d96c')
    if not i: centered(d,'EXPLORERS & PIRATES',256,85,17,'#eed47e')
    im.save(ASSETS/f'game_{i}_spine.jpg',quality=93)
import json
(ASSETS/'diptych_contours.json').write_text(json.dumps(panels,separators=(',',':')))
