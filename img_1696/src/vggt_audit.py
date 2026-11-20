"""Compare evaluated, authored meshes with source-frame-tagged VGGT points in Blender."""
import json
from pathlib import Path
import numpy as np
from vg_reference import read_vertices, reference_for

def capture(video, output, cache, scene_frame=1):
    """Save a triangle cache for overlays and insert BVH discrepancy statistics in the alignment JSON."""
    import bpy
    from mathutils.bvhtree import BVHTree
    output,cache=Path(output),Path(cache)
    cache.parent.mkdir(parents=True,exist_ok=True)
    align_path=output/'vggt_alignment.json'
    align=json.loads(align_path.read_text())
    transform=np.array(align['matrix'],dtype='f8')
    scene=bpy.context.scene
    scene.frame_set(scene_frame)
    graph=bpy.context.evaluated_depsgraph_get()
    points=[];faces=[];labels=[];face_colors=[];names=[];offset=0
    for actual in scene.objects:
        if actual.hide_render or actual.type not in {'MESH','CURVE','FONT','SURFACE','META'}:
            continue
        if any(c.name=='VGGT_REFERENCE' or c.hide_render for c in actual.users_collection):
            continue
        eva=actual.evaluated_get(graph)
        mesh=eva.to_mesh()
        if not mesh:continue
        try:
            if not mesh.polygons:continue
            mesh.calc_loop_triangles()
            p=np.empty(len(mesh.vertices)*3,dtype='f4');mesh.vertices.foreach_get('co',p)
            matrix=np.array(eva.matrix_world)
            p=(np.einsum('nj,ij->ni',p.reshape(-1,3),matrix[:3,:3])+matrix[:3,3]).astype('f4')
            tri=np.empty(len(mesh.loop_triangles)*3,dtype='i4'); mesh.loop_triangles.foreach_get('vertices',tri)
            idx=np.empty(len(mesh.loop_triangles),dtype='i4');mesh.loop_triangles.foreach_get('material_index',idx)
            colors=[tuple(slot.material.diffuse_color[:3]) if slot.material else (.6,.6,.6) for slot in eva.material_slots] or [(.6,.6,.6)]
            color=np.array(colors)[np.minimum(idx,len(colors)-1)]
            color=(np.where(color<=.0031308,12.92*color,1.055*np.maximum(color,0)**(1/2.4)-.055)*255).clip(0,255).astype('u1')
            names.append(actual.name)
            points.append(p);faces.append(tri.reshape(-1,3)+offset);labels.append(np.full(len(idx),len(names)-1,dtype='i4'));face_colors.append(color)
            offset+=len(p)
        finally:
            eva.to_mesh_clear()
    if not faces:
        raise ValueError('No final authored geometry was found')
    vertices=np.concatenate(points);triangles=np.concatenate(faces);labels=np.concatenate(labels)
    if not np.isfinite(vertices).all():
        raise ValueError('One or more evaluated mesh vertices have nonfinite coordinates')
    tree=BVHTree.FromPolygons(vertices.tolist(),triangles.tolist(),all_triangles=True)
    rows=read_vertices(reference_for(video)/'point_cloud.ply')[::83]
    coords=np.column_stack([rows[k] for k in ('x','y','z')])
    xyz=np.einsum('nj,ij->ni',coords,transform[:3,:3])+transform[:3,3]
    distances=np.fromiter((tree.find_nearest(q)[3] for q in xyz),float,count=len(xyz))
    if not np.isfinite(distances).all():
        raise ValueError('BVH comparison returned a nonfinite distance')
    np.savez_compressed(cache,vertices=vertices,triangles=triangles,labels=labels,colors=np.concatenate(face_colors),
                        names=np.array(names),audit_points=xyz.astype('f4'),audit_distances=distances.astype('f4'),source_indices=rows['frame_index'])
    x,y,z=xyz.T
    rgb=np.column_stack([rows[k] for k in ('red','green','blue')]).astype('i2')
    def roi(x_lo,x_hi,y_lo,y_hi,z_lo,z_hi):
        return (x>x_lo)&(x<x_hi)&(y>y_lo)&(y<y_hi)&(z>z_lo)&(z<z_hi)
    tests={
        'all_sampled_vertices':np.ones(len(x),dtype=bool),
        'exposed_floor_walkway':roi(2.0,3.8,3.55,4.0,-.10,.13),
        'plain_artwork_wall_between_pictures':roi(-.14,.12,2.9,4.55,.95,1.92),
        'central_window_wall_pier_including_picture':roi(1.48,2.14,7.93,8.25,1.04,2.13),
        'oak_dining_top_and_runner':roi(.81,1.60,1.69,3.15,.68,.86),
        'coffee_table_top_and_contents':roi(1.90,2.34,5.67,6.65,.29,.68),
        'media_display_purple_pixels':roi(4.1,4.85,4.53,6.5,.90,2.15)&(rgb[:,2]>rgb[:,0]+15)&(rgb[:,2]>rgb[:,1]+45),
        'media_wall_above_hearth':roi(4.40,4.82,4.0,6.9,.83,1.02),
        'sofa_chaise_seating':roi(.83,1.89,4.52,5.30,.29,.67)}
    def describe(flag):
        result={'number_of_reference_samples':int(flag.sum()),'source_views':int(len(np.unique(rows['frame_index'][flag])))}
        if flag.any():
            quant=np.percentile(distances[flag],[50,75,90,95])
            result.update(dict(zip(('median_m','p75_m','p90_m','p95_m'),np.round(quant,5).tolist())))
            result['fraction_within_0_05_m']=round(float(np.mean(distances[flag]<=.05)),4)
        return result
    align['authored_surface_validation']={
        'metric':'Unsigned Euclidean distance in meters from each sampled reference point to the nearest evaluated authored triangle, at Blender frame '+str(scene_frame)+'.',
        'sampling':'Deterministic every 83rd original vertex. Region selectors use world-space boxes and a color predicate for the display. The all-sampled statistic includes occluders, duplicate observations and outliers. Nearest surfaces can belong to adjacent objects.',
        'evaluation_frame_blender':scene_frame,'authored_evaluated_objects':len(names),'authored_triangles':len(triangles),
        'regions':{n:describe(s) for n,s in tests.items()}}
    # Search populated cells to find spatially coherent discrepancies rather than isolated depth outliers.
    cell_size=.30
    voxels=np.floor(xyz/cell_size).astype('i4')
    unique,bin_id,count=np.unique(voxels,axis=0,return_inverse=True,return_counts=True)
    entries=[]
    for i in np.flatnonzero(count>=30):
        flag=bin_id==i
        if len(np.unique(rows['frame_index'][flag]))<7:continue
        entries.append({'cell_min_m':np.round(unique[i]*cell_size,2).tolist(),
                        'sample_centroid_m':np.round(np.median(xyz[flag],axis=0),3).tolist(),**describe(flag)})
    entries.sort(key=lambda x:x['median_m'],reverse=True)
    align['authored_surface_validation']['largest_populated_cell_discrepancies']={
        'cell_edge_m':cell_size,'criterion':'Largest median nearest-mesh distance among populated 30 cm cells with at least 30 samples from seven source frames.',
        'cells':entries[:12]}
    align_path.write_text(json.dumps(align,indent=2)+'\n')
    print('VGGT audit:',json.dumps(align['authored_surface_validation']['regions']),flush=True)
    return align['authored_surface_validation']
