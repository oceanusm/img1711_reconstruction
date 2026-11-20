"""Editable geometry primitives. Colors are authored in display sRGB."""
import bpy, math, json
from mathutils import Vector
from functools import lru_cache
COLLECTION=None

def link(name,data):
    obj=bpy.data.objects.new(name,data)
    (COLLECTION if COLLECTION is not None else bpy.context.scene.collection).objects.link(obj)
    return obj

def linear(v):
    return v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4

def mat(name,srgb,rough=.55,metal=0):
    old=bpy.data.materials.get(name)
    if old: return old
    m=bpy.data.materials.new(name)
    if isinstance(srgb,str):
        s=srgb.lstrip('#'); srgb=[int(s[i:i+2],16)/255 for i in (0,2,4)]
    rgb=tuple(linear(v if v<=1 else v/255) for v in srgb[:3])
    m.diffuse_color=(*rgb,1)
    m.use_nodes=True
    bs=m.node_tree.nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value=(*rgb,1)
    bs.inputs['Roughness'].default_value=rough
    bs.inputs['Metallic'].default_value=metal
    return m

def mesh(name,verts,faces,mat=None):
    data=bpy.data.meshes.new(name)
    data.from_pydata([tuple(p) for p in verts],[],[tuple(f) for f in faces])
    data.update()
    if mat: data.materials.append(mat)
    return link(name,data)

def box(name,center,size,mat,bevel=0):
    x,y,z=[float(v)/2 for v in size]
    o=mesh(name,[(-x,-y,-z),(x,-y,-z),(x,y,-z),(-x,y,-z),(-x,-y,z),(x,-y,z),(x,y,z),(-x,y,z)],[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)],mat)
    o.location=center
    if bevel:
        m=o.modifiers.new('Softened edges','BEVEL'); m.width=min(bevel,.499*min(size));m.segments=4
        n=o.modifiers.new('Face normals','WEIGHTED_NORMAL')
    return o

def cyl(name,center,radius,depth,mat,vertices=48):
    verts=[(radius*math.cos(2*math.pi*i/vertices),radius*math.sin(2*math.pi*i/vertices),h) for h in [-depth/2,depth/2] for i in range(vertices)]
    N=vertices; faces=[tuple(range(N-1,-1,-1)),tuple(range(N,2*N))]+[(i,(i+1)%N,(i+1)%N+N,i+N) for i in range(N)]
    o=mesh(name,verts,faces,mat); o.location=center
    for f in list(o.data.polygons)[2:]: f.use_smooth=True
    return o

def uv(name,center,dimensions,mat):
    n=32;rows=17
    verts=[(0,0,-dimensions[2]/2)]
    for j in range(1,rows):
        a=-math.pi/2+math.pi*j/rows
        for i in range(n):
            t=2*math.pi*i/n
            verts.append((dimensions[0]*.5*math.cos(a)*math.cos(t),dimensions[1]*.5*math.cos(a)*math.sin(t),dimensions[2]*.5*math.sin(a)))
    verts.append((0,0,dimensions[2]/2))
    faces=[(0,1+(i+1)%n,1+i) for i in range(n)]
    for j in range(rows-2):
        for i in range(n):
            a=1+j*n+i;b=1+j*n+(i+1)%n;faces.append((a,b,b+n,a+n))
    faces.extend((1+(rows-2)*n+i,1+(rows-2)*n+(i+1)%n,len(verts)-1) for i in range(n))
    o=mesh(name,verts,faces,mat);o.location=center
    for p in o.data.polygons:p.use_smooth=True
    return o

def rod(name,a,b,radius,mat):
    a,b=Vector(a),Vector(b);d=b-a
    o=cyl(name,(a+b)/2,radius,d.length,mat,16)
    if d.length:o.rotation_mode='QUATERNION';o.rotation_quaternion=d.to_track_quat('Z','Y')
    return o

def tube(name,points,radius,mat):
    curve=bpy.data.curves.new(name,'CURVE');curve.dimensions='3D';curve.resolution_u=8
    curve.bevel_depth=radius;curve.bevel_resolution=2;curve.use_fill_caps=True
    s=curve.splines.new('POLY');s.points.add(len(points)-1)
    for a,v in zip(s.points,points):a.co=(*v,1)
    if mat:curve.materials.append(mat)
    return link(name,curve)

def tag(obj,kind,description,basis,confidence='low',parent=None,relations=None):
    if isinstance(obj,str):obj=bpy.data.objects[obj]
    obj['classification']=kind;obj['description']=description;obj['measurement_basis']=basis;obj['measurement_confidence']=confidence
    if parent:
        p=bpy.data.objects.get(parent) if isinstance(parent,str) else parent
        if p:
            # Collections keep components organized. This relationship is also an editable transform hierarchy.
            p.update_tag()
            from mathutils import Matrix
            def world(o):
                return world(o.parent) @ o.matrix_parent_inverse @ o.matrix_basis if o.parent else o.matrix_basis.copy()
            old=world(obj); pw=world(p);obj.parent=p;obj.matrix_parent_inverse=pw.inverted_safe();obj.matrix_basis=old
    if relations:obj['relationships_json']=json.dumps(relations)
    return obj
