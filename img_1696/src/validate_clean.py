"""Independently check evaluated scene geometry and calibration after reopening scene.blend."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def verify(folder):
    folder = Path(folder).expanduser().resolve()
    bpy.ops.wm.open_mainfile(filepath=str(folder/'scene.blend'))
    scene = bpy.context.scene
    camera = json.loads((folder/'camera.json').read_text())
    manifest = json.loads((folder/'manifest.json').read_text())['objects']
    frames = camera['frames']
    assert scene.render.resolution_percentage == 100
    assert scene.frame_start == 1 and scene.frame_end == len(frames)
    assert scene.unit_settings.system == 'METRIC' and scene.unit_settings.scale_length == 1
    assert 'VGGT_REFERENCE' not in bpy.data.collections
    assert not any(o.name.startswith('VGGT |') for o in scene.objects)
    geom_types = {'MESH','FONT','CURVE','SURFACE'}
    objects = {o.name:o for o in scene.objects if o.type in geom_types}
    assert set(objects) == {item['blender_object_name'] for item in manifest}
    file_images = [im for im in bpy.data.images if im.type=='IMAGE' and im.filepath]
    assert all(im.packed_file for im in file_images)
    schedule = defaultdict(list)
    for item in manifest:
        schedule[0].append((item, 'bbox_m'))
        if 'bbox_at_first_observation_m' in item:
            schedule[item['first_observed_frame']].append((item, 'bbox_at_first_observation_m'))
    scene.frame_set(1)
    scene.view_layers[0].update()
    baseline = {name:obj.matrix_world.copy() for name,obj in objects.items()}
    graph = bpy.context.evaluated_depsgraph_get()
    w,h = camera['resolution_wh']
    assert (w,h) == (scene.render.resolution_x,scene.render.resolution_y)
    pose_error = intrinsic_error = bbox_error = 0.
    checks = 0
    for i,row in enumerate(frames):
        assert row['source_frame'] == i and row['blender_frame'] == i+1
        scene.frame_set(i+1)
        ev = scene.camera.evaluated_get(graph)
        pose_error = max(pose_error,np.max(np.abs(np.asarray(ev.matrix_world)-row['camera_to_world'])))
        p = ev.calc_matrix_camera(graph,x=w,y=h,scale_x=1.,scale_y=1.)
        k = [[w*p[0][0]/2,0,w*(1-p[0][2])/2-.5],
             [0,h*p[1][1]/2,h*(1+p[1][2])/2-.5],[0,0,1]]
        intrinsic_error = max(intrinsic_error,np.max(np.abs(np.asarray(k)-row['K_px'])))
        if any(obj.matrix_world != baseline[name] for name,obj in objects.items()):
            raise ValueError(f'A rigid authored object moved at source frame {i}')
        for item,key in schedule[i]:
            obj = objects[item['blender_object_name']].evaluated_get(graph)
            mesh = obj.to_mesh()
            try:
                # An independent per-vertex mathutils transformation checks the
                # vectorized calculation used when writing the manifest.
                xyz = np.asarray([tuple(obj.matrix_world @ v.co) for v in mesh.vertices],float)
                bb = item[key]
                bbox_error=max(bbox_error,np.max(np.abs(xyz.min(0)-bb['min'])),
                               np.max(np.abs(xyz.max(0)-bb['max'])))
                checks+=1
            finally:
                obj.to_mesh_clear()
    assert pose_error<1e-6 and intrinsic_error<1e-4 and bbox_error<2e-5
    scene.frame_set(1)
    return {'blender':bpy.app.version_string,'frame_count':len(frames),'geometry_objects':len(objects),
            'static_transforms_checked':len(objects)*len(frames),'metric_unit_length':scene.unit_settings.scale_length,
            'bounds_checks':checks,'max_bbox_coordinate_disagreement_m':float(bbox_error),
            'max_camera_matrix_element_disagreement':float(pose_error),
            'max_intrinsics_disagreement_px':float(intrinsic_error),
            'packed_textures':len(file_images),'reference_collection_present':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_folder',type=Path)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    print(json.dumps(verify(args.output_folder),indent=2))
