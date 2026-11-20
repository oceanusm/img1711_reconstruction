"""Compute metric extents from evaluated vertices, including bevel, curve and cloth geometry."""
import bpy, numpy as np, json
from collections import defaultdict
from pathlib import Path

def bounds(obj,graph):
    ev=obj.evaluated_get(graph)
    geom=ev.to_mesh()
    try:
        if geom is None or not len(geom.vertices):raise ValueError('Empty mesh for '+obj.name)
        xyz=np.empty(len(geom.vertices)*3,dtype='f4');geom.vertices.foreach_get('co',xyz)
        xyz=xyz.reshape(-1,3).astype(float)
        M=np.array(ev.matrix_world);world=np.einsum('ij,nj->ni',M[:3,:3],xyz)+M[:3,3]
        dims=(xyz.max(0)-xyz.min(0))*np.sqrt(np.sum(M[:3,:3]**2,axis=0))
        return {'min':world.min(0).round(6).tolist(),'max':world.max(0).round(6).tolist()},dims.round(6).tolist()
    finally:ev.to_mesh_clear()

def write_manifest(output,video):
    scene=bpy.context.scene;scene.frame_set(1);graph=bpy.context.evaluated_depsgraph_get()
    geom=[o for o in scene.objects if o.type in {'MESH','FONT','CURVE','SURFACE'} and not o.hide_render and not any(c.name=='VGGT_REFERENCE' for c in o.users_collection)]
    geom.sort(key=lambda o:o.name)
    ids={o.name:'obj_%04d'%(i+1) for i,o in enumerate(geom)}
    by_frame=defaultdict(list);rows=[]
    for o in geom:
        name=o.name;bb,dim=bounds(o,graph)
        parent=o.parent; first=int(o.get('first_observed_frame',0))
        if parent and 'first_observed_frame' not in o:first=int(parent.get('first_observed_frame',0))
        row=dict(id=ids[name],name=name,description=o.get('description',name),scene_path=(o.users_collection[0].name if o.users_collection else 'Scene Collection')+'/'+name,
                 blender_object_name=name,parent=ids.get(parent.name) if parent else None,type=o.get('classification','architectural or furnishing component'),
                 material=[m.material.name for m in o.material_slots if m.material],bbox_m=bb,dimensions_m=dim,measurement_confidence=o.get('measurement_confidence','low'),
                 measurement_basis=o.get('measurement_basis','Visual proportion estimate in the source video.'),first_observed_frame=first,geometry_frame=0)
        rel=json.loads(o.get('relationships_json','{}'))
        if isinstance(rel,dict):rel={k:ids.get(v,v) if isinstance(v,str) else v for k,v in rel.items()}
        if isinstance(rel,dict) and parent:rel.setdefault('attached_to',ids.get(parent.name,parent.name))
        if rel:row['relationships']=rel
        if first:by_frame[first].append((row,o))
        anim=[]
        for slot in o.material_slots:
            tree=getattr(slot.material,'node_tree',None)
            action=tree and tree.animation_data and tree.animation_data.action
            if action:
                f0,f1=list(action.frame_range)
                anim.append(dict(kind='material illumination',action=action.name,source_frame_range=[int(f0-1),int(f1-1)],rigid_transform='static'))
        if anim:row['animation']=anim
        rows.append(row)
    for f,pairs in sorted(by_frame.items()):
        scene.frame_set(f+1)
        for row,o in pairs:row['bbox_at_first_observation_m']=bounds(o,graph)[0]
    scene.frame_set(1)
    camera=json.loads((output/'camera.json').read_text())
    meta=dict(video_filename=Path(video).name,blender_version=bpy.app.version_string,units='metres',coordinate_axes='+Z up, +X away from long artwork wall, +Y toward end windows',
              coordinate_origin='Intersection of the leveled artwork-wall datum and floor in the transverse plane through the first tracked camera center.',
              frame_count=len(camera['frames']),frame_rate_nominal=camera['fps_nominal'],frame_rate_average=camera['fps_average_source'],frame_mapping='Ascending, zero-based source frames. Blender frame = source frame + 1.',
              scale_calibrated=False,scale_assumption='Dining tabletop is assumed 0.75 m above the floor. All reported length estimates scale uniformly.',
              dimension_convention='XYZ extents along each object’s modeled local axes, with the world scale of those axes applied. World-authored geometry uses room axes. Values describe the named component. World-axis-aligned bbox_m comes from evaluated vertices on source frame 0. Compound extents include descendant components in assembly_bbox_m. Numeric storage precision does not imply measurement accuracy.',
              visibility_convention='first_observed_frame is the estimated first identifiable source view, often assigned by component group. Occluded construction inherits its parent observation. Some camera-frustum appearances precede identification.',
              object_count=len(rows),motion='No rigid furniture or prop motion identified. Camera animated throughout. Ember illumination varies procedurally.')
    children=defaultdict(list)
    for i,r in enumerate(rows):
        if r['parent']:children[r['parent']].append(r)
    def union(row):
        bb=row['bbox_m'];lo=np.array(bb['min']);hi=np.array(bb['max'])
        for child in children.get(row['id'],[]):
            a,b=union(child);lo=np.minimum(lo,a);hi=np.maximum(hi,b)
        if children.get(row['id']):
            row['assembly_bbox_m']={'min':lo.tolist(),'max':hi.tolist()}
            row['assembly_world_dimensions_m']=(hi-lo).round(6).tolist()
        return lo,hi
    for r in rows:
        if not r['parent']:union(r)
    (output/'manifest.json').write_text(json.dumps(dict(scene=meta,objects=rows),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
