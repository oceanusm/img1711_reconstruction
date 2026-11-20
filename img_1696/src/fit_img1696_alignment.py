"""Reproduce img_1696's scene-specific global similarity and reference provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from vg_reference import reference_for,read_vertices

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()

def estimate(video,output):
    video=Path(video).expanduser().resolve()
    if video.stem.lower()!='img_1696':
        raise ValueError('The manually identified floor, wall and tabletop correspondences belong to img_1696. Identify scene-specific anchors before fitting another video.')
    folder=reference_for(video)
    index=json.loads((folder/'index.json').read_text())
    src=read_vertices(folder/'point_cloud.ply')
    coords=lambda a:np.column_stack([a[k] for k in ('x','y','z')]).astype('f8')
    points=coords(src[::31])
    def plane(normal,offset):
        for halfwidth in (.006,.004,.003,.0025,.002):
            sample=points[np.abs(np.einsum('ij,j->i',points,normal)-offset)<halfwidth]
            center=sample.mean(0)
            _,_,vt=np.linalg.svd(sample-center,full_matrices=False)
            new=vt[-1]
            normal=new if new@normal>0 else -new
            offset=float(center@normal)
        return normal,offset
    down,floor=plane(np.array([.00797,.95782,.28725]),.28158)
    wall_normal,wall_offset=plane(np.array([.84473,.14724,-.51454]),-.44712)
    X=wall_normal-down*(wall_normal@down)
    X/=np.linalg.norm(X);Z=-down;Y=np.cross(Z,X)
    rotation=np.stack((X,Y,Z))
    wall_members=points[np.abs(np.einsum('ij,j->i',points,wall_normal)-wall_offset)<.002]
    x0=np.median(np.einsum('ij,j->i',wall_members,X))
    clicks=[(85,422),(250,447),(330,390),(360,353),(178,390)]
    with np.load(folder/'cameras.npz') as cams:
        j=int(np.where(cams['frame_indices']==0)[0][0])
        y0=Y@cams['camera_to_world'][j,:3,3]
        first=coords(src[src['frame_index']==0])
        cam=cams['world_to_camera'][j]
        cp=np.einsum('ij,nj->ni',cam[:3,:3],first)+cam[:3,3]
        K=cams['intrinsics_canonical'][j]
        uv=np.stack((cp[:,0]/cp[:,2]*K[0,0]+K[0,2],cp[:,1]/cp[:,2]*K[1,1]+K[1,2]),1)*.4
        surface=[np.median(first[np.argsort(np.linalg.norm(uv-q,axis=1))[:8]],0) for q in clicks]
        camera_arrays={n:{'shape':list(cams[n].shape),'dtype':str(cams[n].dtype)} for n in cams.files}
    drops=np.einsum('ij,j->i',surface,down)
    scale=.75/(floor-np.median(drops))
    translation=-scale*np.array([x0,y0,-floor])
    matrix=np.eye(4);matrix[:3,:3]=scale*rotation;matrix[:3,3]=translation
    data={
        'source_video':str(video),'video_sha256':digest(video),
        'reference_paths':{f:str(folder/f) for f in ('index.json','cameras.npz','point_cloud.ply')},
        'reference_sha256':{f:digest(folder/f) for f in ('index.json','cameras.npz','point_cloud.ply')},
        'source_coordinate_conventions':index['coordinates'],'preprocessing':index['preprocessing'],'selection':index['selection'],
        'source_point_cloud':index['point_cloud'],'ply_vertex_properties':[[n,str(src.dtype[n])] for n in src.dtype.names],
        'transform_direction':'world_m_homogeneous_column = matrix @ VGGT_raw_homogeneous_column',
        'matrix':matrix.tolist(),'rotation':rotation.tolist(),'translation':translation.tolist(),'scale':float(scale),
        'world_coordinate_conventions':{'up':'+Z','across_room':'+X away from artwork wall','along_room':'+Y toward end windows','units':'metre',
            'origin':'Intersection of the leveled artwork-wall datum and the floor, in the transverse plane through the first VGGT camera center.'},
        'floor_plane_raw':{'down_unit_normal':down.tolist(),'dot_normal_point':floor},
        'long_wall_raw':{'inward_unit_normal':wall_normal.tolist(),'dot_normal_point':wall_offset},
        'fitting':'Global major horizontal and artwork-wall planes initialized by RANSAC, refined by successively trimmed least squares using every 31st PLY vertex. Axes are orthonormal and right-handed.',
        'metric_scale_calibrated':False,'metric_scale_anchor':{
            'physical_reference':'Dining table upper surface, assumed to be 0.75 m above the finished floor.',
            'reason':'No survey, tape measure, known model dimensions, or independently calibrated metric reference was supplied.',
            'source_frame_0_clicks_xy_on_432_by_768_portrait':clicks,
            'method':'Median 3D position among eight nearest reprojected source-frame-0 points for each click. Use median floor clearance across clicks.',
            'floor_to_top_raw_units':(floor-drops).tolist(),
            'illustrative_sensitivity':'A table actually 0.72–0.78 m high changes all reported lengths by −4% to +4%. This is a sensitivity example, not a measured confidence interval.'},
        'npz_arrays':camera_arrays,
        'reference_frame_mapping':[{k:row[k] for k in ('index','camera_array_index','time_seconds','pts_seconds')} for row in index['frames']],
        'coverage':[
           'Dense support on the visible floor strips, dining top, art wall, canvas faces, window-wall pier, television, coffee-table areas and sofa upholstery facing the lens.',
           'Large unsampled regions run through the open center of the room, opaque window shades and glazing, much of the ceiling, and the space behind the sectional.',
           'Undersides, hidden furniture supports, floor concealed by furnishings, and thin cords/rails generally have poor direct observations.',
           'Chrome, glass, TV content, and the lit fireplace are liable to inconsistent depths. The hearth opening is largely missing from the VGGT interior returns.',
           'Individual depth pixels from different camera locations can duplicate static surfaces with centimeter offsets. Corners, small objects and grazing surfaces are less well constrained. No local warping was applied.'],
        'reference_storage':'All native coordinates, byte RGB values, per-point confidence and source frame IDs are retained in a separate hidden-render collection. Its mesh object transform stores the similarity. A deterministic inspection sample displays alongside authored geometry.',
        'algorithm':index['stage'],'source_revision':index['upstream']['revision']}
    # Cross-check only. The actual plate model, dimensions and maker are unknown.
    plate_corners = {
        344: (np.array([(287,95),(453,89),(454,324),(290,329)])/3+[450,710]),
        347: (np.array([(15,177),(186,166),(190,402),(26,412)])/3+[440,705])}
    plate_results=[]
    with np.load(folder/'cameras.npz') as cams:
        for fr,corner in plate_corners.items():
            j=int(np.flatnonzero(cams['frame_indices']==fr)[0]);pose=cams['camera_to_world'][j]
            center=matrix[:3,:3]@pose[:3,3]+translation
            rays=matrix[:3,:3]@pose[:3,:3]@np.linalg.inv(cams['intrinsics_canonical'][j])@np.column_stack((corner,np.ones(4))).T
            hit=(center[:,None]+rays*((8.095-center[1])/rays[1])).T
            lengths=np.linalg.norm(hit-np.roll(hit,-1,axis=0),axis=1)
            plate_results.append({'frame':fr,'canonical_uv_clockwise_from_top_left':corner.tolist(),
                                  'widths_m':lengths[[0,2]].tolist(),'heights_m':lengths[[1,3]].tolist(),
                                  'center_m':hit.mean(0).tolist()})
    data['secondary_scale_check']={
        'feature':'Outer corners of the duplex wall plate near the black side table',
        'method':'Intersect independent ray picks from VGGT frames 344 and 347 with the wall-plate plane at Y=8.095 m in the estimated metric scene.',
        'observations':plate_results,
        'comparison':'A commercial standard-size North American duplex wallplate (Leviton 88003) is 0.06985 m wide and 0.11430 m high. The installed model is unidentified. Agreement within a few percent is a plausibility check; this measurement did not set the scale.',
        'comparison_specification_url':'https://www.mainelectricsupply.com/customer/docs/skudocs/idw-document-573c0facf2c1fea263ae33fb312f02aa493636fe62b2d856989b139f7ac20e7e.pdf'}
    path=Path(output)/'vggt_alignment.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get('authored_surface_validation'):
            data['authored_surface_validation']=previous['authored_surface_validation']
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n')
    return data

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('video',type=Path);ap.add_argument('output',type=Path)
    opts=ap.parse_args();estimate(opts.video,opts.output)
