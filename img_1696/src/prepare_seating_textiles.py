"""Rebuild the two measured blanket surfaces and isolated textile swatches.
Run with Python 3, NumPy, Pillow and ffmpeg. The supplied annotations belong
to img_1696; they are intentionally not reused for other footage.
"""
import argparse,json,pathlib,subprocess,tempfile
import numpy as np
from PIL import Image,ImageDraw
from vg_reference import reference_for, read_vertices

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('video',type=pathlib.Path)
parser.add_argument('--alignment',type=pathlib.Path)
parser.add_argument('--out',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parent/'assets'/'seating')
args=parser.parse_args()
video=args.video.expanduser().resolve();ref=reference_for(video)
if video.stem.lower()!='img_1696':raise ValueError('The textile outlines and observations belong to img_1696 only.')
out=args.out;out.mkdir(parents=True,exist_ok=True)
align=args.alignment or video.parent.parent/'output'/video.stem/'vggt_alignment.json'
M=np.asarray(json.loads(align.read_text())['matrix'])
cams=np.load(ref/'cameras.npz');inds=cams['frame_indices']
points=read_vertices(ref/'point_cloud.ply')
def world(xyz):
 return np.einsum('...i,ji->...j',xyz,M[:3,:3])+M[:3,3]
class View:
 def __init__(self,i):
  index=int(np.abs(inds-i).argmin());v=points[points['frame_index']==int(inds[index])]
  self.xyz=np.stack([v[k] for k in ('x','y','z')],1).astype('f8')
  E=cams['world_to_camera'][index];K=cams['intrinsics_canonical'][index]
  cam=np.einsum('ni,ji->nj',self.xyz,E[:3,:3])+E[:3,3]
  pixel=np.einsum('ni,ji->nj',cam,K);self.uv=pixel[:,:2]/pixel[:,2:]/2
temp=tempfile.TemporaryDirectory();frames=pathlib.Path(temp.name)
for i in (249,320,614):
 subprocess.run(['ffmpeg','-v','error','-i',str(video),'-vf',f'select=eq(n\\,{i}),scale=540:960','-frames:v','1','-q:v','2','-y',str(frames/f'f{i}.jpg')],check=True)
# Trace the visible throws separately. A two-sided measured shell fills the hidden backing.
brown=[(81,414),(87,389),(103,368),(113,344),(126,322),(143,300),(151,284),(171,270),(199,258),(242,231),(253,225),(282,233),(306,249),(328,248),(354,260),(365,275),(390,285),(401,306),(418,308),(429,318),(424,336),(437,348),(440,365),(447,377),(445,392),(456,407),(461,435),(474,453),(472,476),(477,510),(474,536),(463,562),(451,571),(422,579),(418,598),(418,621),(427,638),(416,661),(392,670),(384,652),(381,630),(378,601),(374,575),(350,567),(335,546),(314,531),(295,517),(272,521),(252,521),(240,511),(213,505),(187,495),(164,484),(143,468),(126,453),(105,453),(91,441)]
cream=[(146,311),(155,293),(173,278),(190,270),(204,257),(219,250),(228,237),(242,230),(252,226),(265,228),(279,233),(293,241),(305,248),(323,247),(336,251),(349,255),(357,263),(362,275),(376,278),(389,285),(396,294),(398,305),(394,317),(387,326),(374,331),(362,340),(349,350),(332,352),(318,357),(299,361),(282,366),(265,376),(250,387),(237,406),(215,418),(206,410),(196,395),(181,383),(166,367),(156,350),(148,333)]
# Pull color only from the isolated object. UV coordinates are retained on the curved surface.
def prepare_cloth(i,poly,key):
 img=Image.open(frames/f'f{i}.jpg').convert('RGB'); mask=Image.new('L',img.size);ImageDraw.Draw(mask).polygon(poly,fill=255)
 xmin,ymin,xmax,ymax=mask.getbbox();crop=img.crop((xmin,ymin,xmax,ymax))
 block=Image.new('RGB',crop.size,(163,153,141) if key=='throw_taupe' else (218,211,199));block.paste(crop,(0,0),mask.crop((xmin,ymin,xmax,ymax)));block.save(out/f'{key}.jpg',quality=96)
 vv=View(i); xy=vv.uv; xyz=world(vv.xyz); xp=xy[:,0].astype(int).clip(0,539);yp=xy[:,1].astype(int).clip(0,959)
 # Reject mixed nearest neighbors on the couch, wall, and floor.
 good=(np.asarray(mask)[yp,xp]>0)&(xyz[:,0]>1.8)&(xyz[:,0]<2.67)&(xyz[:,1]>6.86)&(xyz[:,1]<7.75)&(xyz[:,2]>.02)&(xyz[:,2]<.88)
 if key=='throw_cream':good&=xyz[:,2]>.50
 xy=xy[good]; xyz=xyz[good]; E=cams['camera_to_world'][np.abs(inds-i).argmin()]; K=cams['intrinsics_canonical'][np.abs(inds-i).argmin()]
 eye=world(E[:3,3]); camd=np.linalg.norm(xyz-eye,axis=1);Kinv=np.linalg.inv(K)
 def point(x,y):
  dsq=((xy-[x,y])**2).sum(axis=1);ix=np.argsort(dsq)[:12]; dd=np.median(camd[ix]);wt=np.exp(-(dsq[ix]-dsq[ix].min())/150.)*(np.abs(camd[ix]-dd)<.13)
  depth=np.average(camd[ix],weights=wt) if wt.sum()>.01 else dd
  ray=world(E[:3,3]+E[:3,:3]@Kinv@np.array([2*x,2*y,1]))-eye
  return eye+ray*depth/np.linalg.norm(ray)
 verts=[];uv=[];faces=[];lookup={};pix=np.asarray(mask);st=4;col=[]
 for y in range(ymin,ymax,st):
  for x in range(xmin,xmax,st):
   if pix[y,x]==0:continue
   ix=len(verts);lookup[x,y]=ix;v=point(x,y)
   verts.append(v);uv.append(((x-xmin)/(xmax-xmin),1-(y-ymin)/(ymax-ymin)));col.append(img.getpixel((x,y)))
   a=lookup.get((x-st,y-st));b=lookup.get((x,y-st));c=lookup.get((x-st,y))
   if a is not None and b is not None and c is not None:faces.append((a,c,ix,b))
 # Smooth the sampled contour for inspection from oblique viewpoints.
 edges={}
 for face in faces:
  for a,b in zip(face,face[1:]+face[:1]):
   edge=(min(a,b),max(a,b));edges[edge]=edges.get(edge,0)+1
 neighbors={}
 for (a,b),count in edges.items():
  if count==1:
   neighbors.setdefault(a,[]).append(b);neighbors.setdefault(b,[]).append(a)
 if key=='throw_taupe':
  # Clear the separate ivory layer from the hidden taupe surface texture.
  hidden=Image.new('L',img.size);ImageDraw.Draw(hidden).polygon(cream,fill=255)
  patch=Image.new('RGB',block.size,(155,143,132))
  block.paste(patch,(0,0),hidden.crop((xmin,ymin,xmax,ymax)))
  block=block.point(lambda value:round(.93*value))
  block.save(out/f'{key}.jpg',quality=96)
  m2=Image.new('L',(540,960));ImageDraw.Draw(m2).polygon(cream,fill=255);ma=np.asarray(m2)
  for (x,y),ii in lookup.items():
   if ma[y,x]:verts[ii][2]-=.060
   if y>620 and 379<x<425:
    # The hanging tip is additionally constrained by its observed rug contact.
    v=verts[ii];t=min(1.,(y-620)/46);z=v[2]*(1-t)+.022*t
    ray=v-eye
    if abs(ray[2])>.01:verts[ii]=eye+ray*(z-eye[2])/ray[2]
 verts=np.array(verts);uv=np.array(uv)
 for _ in range(5):
  prev=verts.copy();prev_uv=uv.copy()
  for idx,near in neighbors.items():
   if len(near)==2:
    verts[idx]=.46*prev[idx]+.27*(prev[near[0]]+prev[near[1]])
    uv[idx]=.46*prev_uv[idx]+.27*(prev_uv[near[0]]+prev_uv[near[1]])
 faces=np.array(faces,dtype='i4')
 used,flat=np.unique(faces,return_inverse=True)
 np.savez_compressed(out/f'{key}.npz',verts=verts[used].astype('f4'),uv=uv[used].astype('f4'),quads=flat.reshape(faces.shape).astype('i4'),rgb=np.array(col,dtype='u1')[used])
for poly,k in [(brown,'throw_taupe'),(cream,'throw_cream')]:prepare_cloth(320,poly,k)
img=Image.open(frames/'f614.jpg')
img.transform((420,420),Image.Transform.QUAD,(174,672,150,926,423,952,434,678),Image.Resampling.BICUBIC).save(out/'faux_fur.jpg',quality=95)
# An unobstructed rug region is inverse-projected to an orthographic textile sample.
img=Image.open(frames/'f249.jpg'); idx=np.abs(inds-249).argmin();E=cams['world_to_camera'][idx];K=cams['intrinsics_canonical'][idx]; invM=np.linalg.inv(M)
def proj(x,y,z):
 xyz=(invM @ [x,y,z,1])[:3];p=K@(E[:3,:3]@xyz+E[:3,3]);return tuple((p[:2]/p[2]/2).tolist())
# Sample grid fits inside lower-right visible rug area, mirrored across joins.
xy=[proj(x,y,.007) for x,y in [(2.19,5.40),(2.19,5.04),(2.50,5.04),(2.50,5.40)]]

sample=img.transform((256,256),Image.Transform.QUAD,tuple(c for pt in xy for c in pt),Image.Resampling.BICUBIC)
from PIL import ImageOps
rug=Image.new('RGB',(512,512));rug.paste(sample,(0,0));rug.paste(ImageOps.mirror(sample),(256,0));rug.paste(ImageOps.flip(sample),(0,256));rug.paste(ImageOps.flip(ImageOps.mirror(sample)),(256,256));rug.save(out/'rug_pile.jpg',quality=95)
