"""Reopen both delivered scenes and verify alignment, editability, cameras and contacts."""
import json,sys
from pathlib import Path
import bpy
import numpy as np
from vg_reference import read_vertices, reference_for
from validate_clean import verify
from validate_dining_edges import measure

def reference_check(video,out):
    bpy.ops.wm.open_mainfile(filepath=str(out/'scene_with_vggt.blend'))
    ref=bpy.data.collections['VGGT_REFERENCE']
    full=ref.objects['VGGT | full-resolution source']
    sample=ref.objects['VGGT | visible inspection sample']
    rows=read_vertices(reference_for(video)/'point_cloud.ply')
    T=json.loads((out/'vggt_alignment.json').read_text())['matrix']
    assert ref.hide_render and full.hide_render and sample.hide_render and sample.visible_get()
    cols=full.data.color_attributes['SourceRGB']
    assert len(full.data.vertices)==len(cols.data)==len(rows)
    indices=np.linspace(0,len(rows)-1,min(113,len(rows)),dtype=int)
    xyz=np.array([tuple(full.data.vertices[i].co) for i in indices])
    rgb=np.array([tuple(cols.data[i].color_srgb)[:3] for i in indices])
    xyz_error=float(np.abs(xyz-np.column_stack([rows[k][indices] for k in ('x','y','z')])).max())
    rgb_error=float(np.abs(rgb-np.column_stack([rows[k][indices] for k in ('red','green','blue')])/255).max())
    matrix_error=float(np.abs(np.asarray(full.matrix_world)-T).max())
    assert xyz_error<1e-7 and rgb_error<2e-6 and matrix_error<2e-6
    return len(rows),{'global_matrix_max_abs_difference':matrix_error,'raw_xyz_max_abs_difference':xyz_error,
                     'raw_srgb_max_abs_difference':rgb_error,'source_vertices_checked':len(indices),'visible_inspection_proxy_vertices':len(sample.data.vertices)}

def contacts(out):
    rows={o['name']:o for o in json.loads((out/'manifest.json').read_text())['objects']}
    pairs=[('Beverage | opened citrus can','Coffee table | rounded oak top'),
           ('Laptop | aluminum base','Coffee table | rounded oak top'),
           ('Controller | powder blue','Coffee table | rounded oak top'),
           ('Catan base game box','Catan Explorers and Pirates box'),
           ('Catan Explorers and Pirates box','Pale oak oval sofa console top'),
           ('White supplement bottle','Pale oak oval sofa console top'),
           ('Table lamp | ceramic base','Black end table | turned body'),
           ('Kitchen · cobalt bottle','Kitchen · peninsula quartz slab')]
    return {'method':'Compare the minimum world z of the named object and the maximum world z of its visibly flat support at frame 1. This test is restricted to these eight pairs.',
            'unit':'metre','object_base_minus_support_top':{f'{a} / {b}':round(rows[a]['bbox_m']['min'][2]-rows[b]['bbox_m']['max'][2],7) for a,b in pairs}}

def verify_both(video,out):
    out=Path(out).resolve();count,evidence=reference_check(Path(video).expanduser(),out)
    report=verify(out)
    report.update(inspection_vertices=count,reference_data_check=evidence,inspection_render_excluded=True,
                  dining_edges=measure(),selected_contacts=contacts(out))
    path=out/'inspection/09_scene_validation.json';path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')
    return report
