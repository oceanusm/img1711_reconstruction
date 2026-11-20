"""Measure local point-to-plane discrepancy for the authored img_1696 decor.

Run with a completed output folder: python3 src/check_decor.py img_1696.mov
Prints JSON for the measured windows. Bounds are explicit to make the effect
of screening nearby clutter auditable. Statistics describe selected surfaces.
"""
from pathlib import Path
import argparse, json
import numpy as np
from vg_reference import reference_for, read_vertices

REGIONS=[
    ('Left canvas',580,727,5,[(-.036,.056),(4.87,5.84),(1.10,1.70)],0,.008),
    ('Right canvas',726,792,4,[(-.036,.056),(6.30,7.22),(1.08,1.70)],0,.003),
    ('Exposed console top',713,765,3,[(.29,.415),(5.990,6.105),(.68,.836)],2,.764),
    ('Rattan shelf margin',632,678,3,[(.025,.37),(4.920,4.960),(.68,.830)],2,.756),
    ('Cart lower shelf',489,513,2,[(3.45,4.04),(3.20,3.30),(.11,.175)],2,.153),
    ('Cart upper shelf',489,513,2,[(3.39,4.08),(3.21,3.36),(.78,.87)],2,.838)]
def check(video,alignment):
    if video.stem.lower()!='img_1696':
        raise ValueError('These decor planes have been fitted specifically to img_1696.')
    root=reference_for(video)
    src=read_vertices(root/'point_cloud.ply')
    transform=np.array(json.loads(alignment.read_text())['matrix'])
    cams=np.load(root/'cameras.npz')
    available=cams['frame_indices']
    def samples(frame):
        f=available[np.argmin(abs(available-frame))]
        group=src[src['frame_index']==f]
        points=np.column_stack([group[k] for k in 'xyz'])
        return np.einsum('ij,nj->ni',transform[:3,:3],points)+transform[:3,3]
    result={}
    for name,lo,hi,step,bounds,axis,plane in REGIONS:
        differences=[]
        frames=list(range(lo,hi,step))
        for frame in frames:
            pts=samples(frame);inside=np.ones(len(pts),bool)
            for a,(xmin,xmax) in enumerate(bounds):
                inside&=(pts[:,a]>xmin)&(pts[:,a]<xmax)
            differences.extend(abs(pts[inside,axis]-plane).tolist())
        result[name]={'number_of_points_in_selection':len(differences),
            'source_frames_nearest_to':frames,'xyz_selection_bounds_m':bounds,
            'plane_axis':axis,'nominal_plane_m':plane,
            'median_and_p90_absolute_axis_distance_m':np.quantile(differences,[.5,.9]).tolist()}
    return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('video',type=Path);parser.add_argument('--alignment',type=Path,default=Path(__file__).resolve().parents[1]/'vggt_alignment.json')
    a=parser.parse_args();print(json.dumps(check(a.video.expanduser(),a.alignment),indent=2))
