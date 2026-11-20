"""Source-specific provenance and a native Blender reference-only scene for a new shot."""
import hashlib,json
from pathlib import Path
import numpy as np
from vg_reference import read_vertices,reference_for

def hash_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def describe(video,output):
    video=Path(video).expanduser().resolve();ref=reference_for(video)
    index=json.loads((ref/'index.json').read_text())
    pts=read_vertices(ref/'point_cloud.ply')
    with np.load(ref/'cameras.npz',allow_pickle=False) as cams:
        schema={k:{'shape':list(cams[k].shape),'dtype':str(cams[k].dtype)} for k in cams.files}
    source_rows=index.get('frames',[])
    data={'source':str(video),'video_sha256':hash_file(video),'reference':str(ref),
          'reference_sha256':{n:hash_file(ref/n) for n in ('cameras.npz','point_cloud.ply','index.json')},
          'point_count':len(pts),'point_properties':list(pts.dtype.names or []),'cameras_npz_arrays':schema,
          'coordinates':index.get('coordinates',{}),'selection':index.get('selection',{}),'preprocessing':index.get('preprocessing',{}),
          'reference_to_source_frame_map':[{k:r.get(k) for k in ('index','camera_array_index','pts_seconds')} for r in source_rows],
          'scale_status':'Source-native VGGT units. No scene-specific scale or axis alignment has been established.',
          'stage':'Reference and camera preparation. This is an inspection dataset for authoring a new scene.'}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'source_reference.json').write_text(json.dumps(data,indent=2)+'\n')
    return data

def save_native(video,output):
    import bpy
    from vg_reference import import_reference
    for o in list(bpy.data.objects):bpy.data.objects.remove(o,do_unlink=True)
    scene=bpy.context.scene;scene.name='Source-native geometric reference';scene.unit_settings.system='NONE'
    ref=import_reference(video,np.eye(4).tolist());ref['alignment_status']='Raw VGGT world; units and axes uncalibrated.'
    scene['purpose']='New-source inspection only. No authored scene geometry or metric measurements.'
    bpy.context.preferences.filepaths.save_version=0
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(output)/'source_reference.blend'),compress=True)
