"""Refresh the source-specific botanical print and synthetic finish assets.
Run with Python, OpenCV, and NumPy, passing the source video as the sole argument.
"""
import cv2,numpy as np,sys,json
from pathlib import Path
from vg_reference import reference_for
src=Path(sys.argv[1]).expanduser().resolve()
if src.stem!='img_1696':
    raise ValueError('These artwork correspondences belong to img_1696')
ref=reference_for(src)
directory=Path(__file__).resolve().parent
M=np.array(json.loads((directory.parent/'vggt_alignment.json').read_text())['matrix'])
cam=np.load(ref/'cameras.npz'); K=cam['intrinsics_canonical']
out=directory/'assets';out.mkdir(exist_ok=True)
# Orthographically rectify only the inset painting face. Earlier full-panel views fill above the cropped close-up.
W,H=400,755
x,z=np.meshgrid(np.linspace(1.504,1.988,W),np.linspace(2.114,1.229,H))
p=np.stack((x,np.full_like(x,8.085),z,np.ones_like(x)),-1)
raw=p@np.linalg.inv(M).T
cap=cv2.VideoCapture(str(src))
img=np.full((H,W,3),186,np.uint8)
for frame in (140,158,196,218,251,861,874,877,880,882):
 j=int(np.argmin(abs(cam['frame_indices']-frame)));f=int(cam['frame_indices'][j])
 cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read()
 if not ok:continue
 camera=raw@cam['world_to_camera'][j].T
 uv=camera[:,:,:3]@K[j].T
 uv=uv[:,:,:2]/uv[:,:,2:]
 mask=(uv[:,:,0]>6)&(uv[:,:,0]<1074)&(uv[:,:,1]>6)&(uv[:,:,1]<1914)
 mapped=cv2.remap(im,uv[:,:,0].astype('f'),uv[:,:,1].astype('f'),cv2.INTER_LINEAR)
 img[mask]=mapped[mask]
cv2.imwrite(str(out/'botanical-etching.jpg'),img,[cv2.IMWRITE_JPEG_QUALITY,94])
# Long-grain oak albedo, used with an actual staggered plank mesh.
h,w=1024,192
y,x=np.mgrid[:h,:w];r=np.random.default_rng(24)
slow=cv2.resize(r.normal(0,1,(14,14)).astype('f'),(w,h),interpolation=cv2.INTER_CUBIC)
grain=cv2.resize(r.normal(0,1,(15,124)).astype('f'),(w,h),interpolation=cv2.INTER_CUBIC)
bands=np.sin((x+7*np.sin(y/195+x/56)+3*slow)*1.4)
f=slow*4+grain*5.8+bands*1.3+r.normal(0,1.8,(h,w))
wood=np.uint8(np.clip(np.array([168,191,205])[None,None,:]+f[:,:,None],0,255))
cv2.imwrite(str(out/'oak-grain.jpg'),wood)
