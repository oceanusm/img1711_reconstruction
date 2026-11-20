"""Reproject marked tabletop corners using the evaluated final Blender camera.

Run after opening scene.blend:
    blender -b scene.blend -P src/validate_dining_edges.py -- result.json

The annotated points are source frame indices, captured on a half-width display.
Two obscured corners of the return viewpoints are deliberately unscored.
"""
import json
from pathlib import Path
import sys
import numpy as np
import bpy
from bpy_extras.object_utils import world_to_camera_view


OBSERVATIONS={
    0: [(69,520.5),(342,577.5),(528,434.5),(345,413.5)],
    66: [(18.5,542),(312,610),(529,442.5),(340,418.5)],
    530: [None,(187.5,469.5),(296.5,530),(472.5,517)],
    539: [None,(42,456.5),(170,512.5),(336.5,495)]
}


def measure():
    scene=bpy.context.scene
    if (scene.render.resolution_x,scene.render.resolution_y)!=(1080,1920):
        raise ValueError('These observations belong to the portrait img_1696 reconstruction.')
    obj=bpy.data.objects['Dining table · oak top']
    # Corners on the unchamfered reference rectangle. The visible roundover is 4 mm.
    corners=[v.co.copy() for v in obj.data.vertices if v.co.z>0]
    if len(corners)!=4:
        raise ValueError('Rectangular top design mesh expected before the small bevel.')
    previous=scene.frame_current; rows=[]
    try:
        for src_frame,picks in OBSERVATIONS.items():
            scene.frame_set(src_frame+1)
            dep=bpy.context.evaluated_depsgraph_get()
            camera=scene.camera.evaluated_get(dep)
            top=obj.evaluated_get(dep)
            for idx,(v,pick) in enumerate(zip(corners,picks)):
                if pick is None:continue
                uv=world_to_camera_view(scene,camera,top.matrix_world@v)
                predicted=np.array([uv.x*1080,(1-uv.y)*1920])
                observed=np.array(pick)*2
                rows.append(dict(source_frame=src_frame,corner=idx,
                                 observed_px=observed.tolist(),
                                 predicted_px=predicted.round(3).tolist(),
                                 euclidean_error_px=float(np.linalg.norm(predicted-observed))))
    finally:
        scene.frame_set(previous)
    e=[r['euclidean_error_px'] for r in rows]
    return dict(method='Manually marked tabletop corners, reconstructed camera and tabletop evaluated in Blender. Coordinates in displayed 1080 × 1920 pixels.',
                existing_constraints='The corner observations helped fit the tabletop and are not independent accuracy checks. Two of 16 possible corners were obscured. Rounded corners and motion blur affect the picks.',
                count=len(rows),median_pixel_error=float(np.median(e)),
                p90_pixel_error=float(np.quantile(e,.9)),max_pixel_error=max(e),correspondences=rows)


if __name__=='__main__':
    report=measure()
    if '--' in sys.argv and len(sys.argv)>sys.argv.index('--')+1:
        Path(sys.argv[sys.argv.index('--')+1]).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='correspondences'},indent=2))
