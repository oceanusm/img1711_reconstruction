"""Build the animated Blender camera and export its evaluated calibration."""
from fractions import Fraction
import json
from pathlib import Path

import bpy
from mathutils import Matrix
import numpy as np


def _populate_keys(owner, field, samples, action_name):
    """Bulk populate directly keyed F-curves in Blender 4.2."""
    if not owner.animation_data:
        owner.animation_data_create()
    if not owner.animation_data.action:
        owner.animation_data.action = bpy.data.actions.new(action_name)
    action = owner.animation_data.action
    for axis, values in enumerate(np.asarray(samples, float).T):
        if hasattr(action, 'fcurves'):
            fc = action.fcurves.new(data_path=field, index=axis)
        else:
            fc = action.fcurve_ensure_for_datablock(owner, field, index=axis)
        coordinates = np.column_stack([np.arange(1,len(values)+1), values]).astype('f')
        fc.keyframe_points.add(len(values))
        fc.keyframe_points.foreach_set('co', coordinates.ravel())
        for k in fc.keyframe_points:
            k.interpolation = 'LINEAR'
        fc.update()


def add_camera(scene, track_path, similarity, output_path):
    """Raw-world to metric Blender scene via one rotation, translation and scale."""
    track = np.load(str(track_path), allow_pickle=False)
    similarity = np.asarray(similarity, dtype=float)
    linear = similarity[:3,:3]
    scale = np.linalg.det(linear)**(1/3)
    rotate = linear/scale
    if not np.allclose(rotate @ rotate.T, np.eye(3), atol=1.e-5):
        raise ValueError('Expected a uniform, orientation-preserving world similarity')
    w, h = map(int, track['size_wh'])
    raw = track['camera_to_world_cv']
    prescribed = track['intrinsics']
    camera_basis = np.diag([1.,-1.,-1.])
    locations = raw[:,:3,3] @ linear.T + similarity[:3,3]
    rotations = rotate @ raw[:,:3,:3] @ camera_basis
    quats = []
    for R in rotations:
        q = Matrix(R.tolist()).to_quaternion().normalized()
        if quats and np.dot(np.array(q),quats[-1]) < 0:
            q.negate()
        quats.append(list(q))
    camera_data = bpy.data.cameras.new('Tracked camera intrinsics')
    cam = bpy.data.objects.new('Tracked video camera', camera_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.rotation_mode = 'QUATERNION'
    camera_data.sensor_fit = 'HORIZONTAL'
    camera_data.sensor_width = 36.0
    camera_data.clip_start = .025
    camera_data.clip_end = 100.
    f = prescribed[:,0,0]
    shx = ((w-1)*.5 - prescribed[:,0,2])/w
    shy = (prescribed[:,1,2] - (h-1)*.5)/w
    _populate_keys(cam, 'location', locations, 'Tracked camera pose')
    _populate_keys(cam, 'rotation_quaternion', quats, 'Tracked camera pose')
    _populate_keys(camera_data, 'lens', (f*camera_data.sensor_width/w)[:,None], 'Tracked lens')
    for field, curve in [('shift_x', shx), ('shift_y', shy)]:
        if np.ptp(curve) < 1e-12:
            setattr(camera_data, field, float(curve[0]))
        else:
            _populate_keys(camera_data, field, curve[:,None], 'Tracked lens')
    source = json.loads(str(track['stream_info']))
    nominal = Fraction(source['r_frame_rate'])
    scene.render.resolution_x, scene.render.resolution_y = w,h
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.
    scene.render.fps = int(round(float(nominal)))
    scene.render.fps_base = scene.render.fps / float(nominal)
    scene.frame_start, scene.frame_end = 1, len(raw)
    mode_names = ['VGGT_anchor', 'time_interpolated', 'optical_flow_refined']
    dep = bpy.context.evaluated_depsgraph_get()
    samples = []
    for i in range(len(raw)):
        scene.frame_set(i+1)
        ev = cam.evaluated_get(dep)
        projection = np.array(ev.calc_matrix_camera(dep, x=w, y=h, scale_x=1., scale_y=1.))
        # OpenGL NDC to oriented image coordinates, integer-valued pixel centers.
        K = np.array([[w/2*projection[0,0], 0., w/2*(1-projection[0,2])-.5],
                      [0., h/2*projection[1,1], h/2*(1+projection[1,2])-.5],
                      [0., 0., 1.]])
        c2w = np.array(ev.matrix_world)
        if (np.max(np.abs(K-prescribed[i])) > .01 or
                np.max(np.abs(c2w[:3,3]-locations[i])) > .00005 or
                np.max(np.abs(c2w[:3,:3]-rotations[i])) > .00005):
            raise ValueError(f'Evaluated camera did not match specification at frame {i}')
        samples.append(dict(source_frame=i, blender_frame=i+1, pts=int(track['pts'][i]),
                     time_seconds=round(float(track['pts'][i]*track['time_base']),9),
                     camera_to_world=c2w.round(9).tolist(), K_px=K.round(6).tolist(),
                     pose_source=mode_names[int(track['mode'][i])]))
    scene.frame_set(1)
    output = dict(source=str(track['source_name']), source_sha256=str(track['source_sha256']),
              cameras_npz_sha256=str(track['reference_sha256']), resolution_wh=[w,h],
              coded_source_wh=[source['width'],source['height']],
              source_display_rotation_deg=next((item['rotation'] for item in source.get('side_data_list',[]) if 'rotation' in item),0),
              matrix_convention='Row-major 4x4 matrices multiply homogeneous column vectors. Camera axes are +X right, +Y up, -Z forward. World axes are the metric Blender scene axes; +Z is up.',
              projection_convention='For a world point use inverse(camera_to_world), then multiply camera y and z by -1. K_px supplies the pinhole calibration for top-left image coordinates, with integer pixel centers. Distortion is uncalibrated.',
              focal_basis='Pinhole square pixels. The geometric mean of the reference focal lengths fx and fy is used at each reference timestamp.',
              fps_nominal=source['r_frame_rate'], fps_average_source=source['avg_frame_rate'],
              source_time_base=source['time_base'], frame_count=len(raw),
              timing='Source PTS are exported frame by frame. Blender plays the nominal frame rate. Source display frames correspond one-for-one with Blender frames.',
              tracking=json.loads(str(track['stats'])), frames=samples)
    Path(output_path).write_text(json.dumps(output,separators=(',',':'),allow_nan=False)+'\n')
    return cam
