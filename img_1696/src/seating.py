"""Sectional, floor covering, low table and the objects resting on them.
Locations use the single VGGT-to-room metric transform shared by the scene.
The measured draped surfaces are stored beside this module in assets/seating.
"""
import math
from pathlib import Path
import bpy
import numpy as np
from common import box, cyl, mat, mesh, rod, tag, tube, uv, linear
from mathutils import Vector

ASSETS=Path(__file__).resolve().parent/'assets'/'seating'
SOFT='Frame-indexed VGGT surface samples, cushion boundaries, and their silhouettes in the source video. The undersides are inferred.'
SMALL='Estimated from multi-view video proportions and the known supporting surface; small parts have limited VGGT depth support.'
TABLE='Planar intersection of tabletop perimeter rays in frames 249 and 285 with the VGGT-supported horizontal top surface.'
FLOOR='Estimated floor-plane intersections of rug edges in source frames 179, 340 and 375.'
def note(o,k,d,basis=SMALL,confidence='low',parent=None,on=None,first=None):
    tag(o,k,d,basis,confidence,parent,{'on':on} if on else None)
    if first is not None: o['first_observed_frame']=first
    return o

def packed_texture(material,path):
    nodes=material.node_tree.nodes; t=nodes.new('ShaderNodeTexImage')
    t.image=bpy.data.images.load(str(path),check_existing=True)
    if not t.image.packed_file: t.image.pack()
    material.node_tree.links.new(t.outputs['Color'], nodes.get('Principled BSDF').inputs['Base Color'])
    nodes.active=t
    return t

def texture_uv(obj,coord):
    layer=obj.data.uv_layers.new(name='MeasuredSurface')
    for loop in obj.data.loops: layer.data[loop.index].uv=coord[loop.vertex_index]

def weave(material):
    nd=material.node_tree.nodes;ln=material.node_tree.links
    bs=nd.get('Principled BSDF')
    tex=nd.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=280;tex.inputs['Detail'].default_value=1.4
    crd=nd.new('ShaderNodeTexCoord');ln.new(crd.outputs['Object'],tex.inputs['Vector'])
    bump=nd.new('ShaderNodeBump');bump.inputs['Distance'].default_value=.00034;bump.inputs['Strength'].default_value=.22
    ln.new(tex.outputs['Fac'],bump.inputs['Height']);ln.new(bump.outputs['Normal'],bs.inputs['Normal'])

def rounded_soft(name,center,size,m,r,bulge=.01,rot=(0,0,0),textured=False):
    # A welded six-face control grid, rounded through each edge, with a padded face profile.
    dims=np.array(size); vs=[];faces=[];lookup={};uvs=[];n=12
    for axis,(u,v) in enumerate([(1,2),(2,0),(0,1)]):
        for sign in (-1,1):
            ids={}
            for j in range(n+1):
                for i in range(n+1):
                    key=[0,0,0];key[axis]=n*sign;key[u]=-n+2*i;key[v]=-n+2*j;key=tuple(key)
                    if key not in lookup:
                        unit=np.array(key,dtype=float)/n;p=unit*dims*.5
                        rr=min(r,.43*min(size));inner=np.maximum(dims*.5-rr,0);q=p.clip(-inner,inner)
                        delta=p-q;normal=delta/np.linalg.norm(delta);p=q+rr*normal
                        # A low-amplitude irregularity avoids hard machined edges.
                        s=1.
                        for a in (u,v):s*=max(0,1-unit[a]**2)
                        p+=normal*(bulge*s+.0014*math.sin(4*unit[u]+.8)*math.sin(7*unit[v])*s)
                        lookup[key]=len(vs);vs.append(p)
                        uvs.append(((unit[1]+1)/2,(unit[2]+1)/2))
                    ids[i,j]=lookup[key]
            for j in range(n):
                for i in range(n):
                    f=(ids[i,j],ids[i+1,j],ids[i+1,j+1],ids[i,j+1]);faces.append(f if sign>0 else tuple(reversed(f)))
    o=mesh(name,vs,faces,m);o.location=center;o.rotation_euler=rot
    for f in o.data.polygons: f.use_smooth=True
    if textured:texture_uv(o,uvs)
    return o

def sweep(name,spine,radii,m,sides=16):
    verts=[];faces=[]
    for j,pt in enumerate(spine):
        p=Vector(pt);d=(Vector(spine[min(j+1,len(spine)-1)])-Vector(spine[max(0,j-1)])).normalized()
        c=d.cross(Vector((0,0,1)))
        if c.length<.01: c=d.cross(Vector((0,1,0)))
        c.normalize(); b=c.cross(d).normalized();r,rz=radii[j]
        for k in range(sides):
            a=k*math.tau/sides;verts.append(tuple(p+c*(r*math.cos(a))+b*(rz*math.sin(a))))
    for j in range(len(spine)-1):
        for k in range(sides):
            a=j*sides+k;b=j*sides+(k+1)%sides
            faces.append((a,b,b+sides,a+sides))
    faces.extend([tuple(reversed(range(sides))),tuple(range(len(verts)-sides,len(verts)))])
    o=mesh(name,verts,faces,m)
    for p in o.data.polygons:p.use_smooth=True
    return o

def slabs(name,cx,cy,sx,sy,z0,z1,r,m):
    ring=[]
    for s,t,d in [(1,1,0),(-1,1,90),(-1,-1,180),(1,-1,270)]:
        for i in range(13):
            a=math.radians(d+90*i/12)
            ring.append((s*(sx/2-r)+r*math.cos(a),t*(sy/2-r)+r*math.sin(a)))
    n=len(ring);depth=z1-z0
    verts=[(x,y,z) for z in (-depth/2,depth/2) for x,y in ring]
    faces=[tuple(reversed(range(n))),tuple(range(n,2*n))]+[(k,(k+1)%n,(k+1)%n+n,k+n) for k in range(n)]
    o=mesh(name,verts,faces,m);o.location=(cx,cy,(z1+z0)/2)
    for p in list(o.data.polygons)[2:]:p.use_smooth=True
    return o

def shell(key,name,color):
    ar=np.load(ASSETS/(key+'.npz'));m=mat(name+' | pile',color,.97);packed_texture(m,ASSETS/(key+'.jpg'))
    o=mesh(name,ar['verts'],ar['quads'],m);texture_uv(o,ar['uv'])
    for f in o.data.polygons:f.use_smooth=True
    solid=o.modifiers.new('Plush fabric thickness','SOLIDIFY');solid.thickness=.009;solid.offset=-1
    return o

def controller(name,center,theta,shell_mat,dark,silver):
    # Joined molded shell, contoured grips, center pad, thumbsticks and grouped buttons.
    allparts=[];acc=mat('Controller | graphite insets','#2D3036')
    def transformed(x,y,z):
        c,s=math.cos(theta),math.sin(theta);return(center[0]+c*x-s*y,center[1]+s*x+c*y,center[2]+z)
    def part(n,xyz,dim,m,angle=0):
        o=uv(n,transformed(*xyz),dim,m);o.rotation_euler[2]=theta+angle;allparts.append(o)
    part(name,(0,.010,.018),(.138,.069,.036),shell_mat)
    for sg in [-1,1]:
        part(name+' Grip '+str(sg),(.057*sg,-.013,.016),(.041,.089,.035),shell_mat,.34*sg)
        part(name+' Shoulder '+str(sg),(.046*sg,.039,.019),(.043,.019,.021),dark)
    part(name+' Center',(0,-.006,.032),(.074,.052,.014),dark)
    part(name+' Touchpad',(0,.018,.035),(.055,.028,.010),shell_mat)
    for sg in [-1,1]:
        part(name+' Thumbstick '+str(sg),(.022*sg,-.017,.039),(.021,.021,.011),acc)
    for k in range(4):
        a=k*math.pi/2
        for sign in [-1,1]:
            part(name+' Control '+str(sign)+' '+str(k),(.049*sign+.009*math.cos(a),.013+.009*math.sin(a),.036),(.0065,.0065,.004),dark)
    return joinparts(allparts,name)

def joinparts(parts,name):
    # Retain one selectable object for the non-separable keys/handles on each control.
    bpy.context.view_layer.objects.active=parts[0]
    bpy.ops.object.select_all(action='DESELECT')
    for x in parts:x.select_set(True)
    bpy.ops.object.join();ob=parts[0];ob.name=name;ob.select_set(False)
    return ob

def remote(name,x,y,z,length,angle,dark,gray):
    o=rounded_soft(name,(x,y,z+.009),(.042,length,.018),dark,.009,.0008,rot=(0,0,angle))
    parts=[o];sa,ca=math.sin(angle),math.cos(angle)
    for row in range(6):
        for col in range(3):
            dx=(col-1)*.009;dy=(row-2.5)*length/8
            b=uv('Remote keypad',(x+ca*dx-sa*dy,y+sa*dx+ca*dy,z+.018),(.006,.006,.0026),gray);parts.append(b)
    return joinparts(parts,name)

def build():
    existing_objects=set(bpy.context.scene.objects)
    fabric=mat('Sectional | oatmeal linen','#EBE7DF',.97);weave(fabric)
    trim=mat('Sectional | shadow welt','#CCC6BB',.96);plinth=mat('Sectional | foot and inset base','#383029',.63)
    furry=mat('Soft pile | warm ivory','#EDE9DE',.96);weave(furry)
    dark=mat('Accessories | black polymer','#25262B',.48);charcoal=mat('Accessories | inset gray','#646570',.56)
    ivory=mat('Accessories | porcelain ivory','#EBE6DD',.51);blue=mat('Controller | cornflower shell','#8DB1DA',.53)
    alu=mat('Laptop and can | silver alloy','#B3B1B5',.36,.52)
    rugmat=mat('Rug | distressed dove and taupe','#BAB4AF',.99);packed_texture(rugmat,ASSETS/'rug_pile.jpg')
    rug=box('Rug | rectangular distressed weave',(2.805,5.688,.007),(2.39,2.95,.012),rugmat)
    texture_uv(rug,[(v.co.x/.62,v.co.y/.72) for v in rug.data.vertices])
    note(rug,'area rug','Short-pile mottled gray, taupe and cream rectangular area rug.',FLOOR,'medium')
    edge=mat('Rug | bound edge','#BDB8B3',.96)
    path=[(1.61,4.213,.014),(4.0,4.213,.014),(4.0,7.163,.014),(1.61,7.163,.014),(1.61,4.213,.014)]
    note(tube('Rug | stitched perimeter',path,.003,edge),'binding','Narrow bound textile perimeter.',FLOOR,'low',rug)
    # A chaise, center modules, and return share one continuous L-shaped seating arrangement.
    bases=[('chaise',.548,1.988,4.438,5.423),('middle',.597,1.420,5.426,6.245),
           ('inner',.575,1.424,6.249,6.994),('corner',.526,1.432,6.997,7.806),
           ('window return',1.435,2.550,6.995,7.803)]
    modules=[]
    for label,x0,x1,y0,y1 in bases:
        x=(x0+x1)/2;y=(y0+y1)/2;dx=x1-x0;dy=y1-y0
        base=note(rounded_soft('Sectional | '+label+' frame',(x,y,.157),(dx,dy,.225),fabric,.031,.002),
                  'sofa module','Upholstered structural base of the '+label+' section.',SOFT,'medium')
        modules.append(base)
        note(box('Sectional | '+label+' inset foot',(x,y,.032),(dx-.105,dy-.095,.057),plinth,.019),
             'furniture support','Recessed dark support below the upholstered body.',SOFT,'low',base)
        x0b=x0+(.039 if label in ['chaise','middle','inner','corner'] else .006)
        y0b=y0+(.139 if label=='chaise' else .001)
        seat=rounded_soft('Sectional | '+label+' seat',((x0b+x1)/2,(y0b+y1)/2,.352),
                          (x1-x0b-.008,y1-y0b-.008,.179),fabric,.036,.009)
        if label=='chaise':
            # The outer half of the pad extends past the end of the arm.
            # At X>1.46 the reference puts the front face near Y 4.45.
            for vertex in seat.data.vertices:
                wx=vertex.co.x+seat.location.x
                t=max(0.,min(1.,(wx-1.35)/.11))
                near=1-(vertex.co.y/(y1-y0b-.008)+.5)
                vertex.co.y-=.139*t*near
            seat.data.update()
        note(seat,'seat cushion','Removable broad, padded seat with rounded corners.',SOFT,'medium',base)
    arm=rounded_soft('Sectional | chaise end arm',(.999,4.506,.354),(.87,.159,.590),fabric,.052,.006)
    note(arm,'armrest','Low padded end panel beside the chaise and charging cable.',SOFT,'medium',modules[0])
    back_specs=[
      ((.735,4.979,.673),(.198,.863,.474),(0,-.17,.018),0),
      ((.704,5.832,.671),(.194,.809,.464),(.012,-.19,-.018),1),
      ((.699,6.632,.667),(.185,.791,.461),(-.021,-.19,.001),2),
      ((.66,7.382,.672),(.196,.728,.452),(.02,-.12,-.025),3),
      ((1.048,7.666,.668),(.754,.186,.470),(-.46,.018,0),3),
      ((1.730,7.716,.675),(.594,.180,.470),(-.46,.011,.012),4)]
    # Corner back surfaces overlap in plan at the join as in the footage.
    for i,(p,s,r,mod) in enumerate(back_specs):
        note(rounded_soft(f'Sectional | back cushion {i+1}',p,s,fabric,.056,.027,r),
             'back cushion','Soft square back cushion leaning toward the outside frame.',SOFT,'medium',modules[mod])
    spotted=mat('Cushion | spotted gray faux fur','#A39E9C',.97);packed_texture(spotted,ASSETS/'faux_fur.jpg')
    note(rounded_soft('Pillow | gray spotted fur',(.965,5.275,.648),(.119,.475,.402),spotted,.04,.022,(.075,-.36,-.09),True),
         'throw pillow','Square gray pillow with two dark spotted bands and a central gray stripe.',SOFT,'medium',on=modules[0].name)
    note(rounded_soft('Pillow | loose ivory center',(.947,5.980,.632),(.147,.426,.436),fabric,.052,.032,(.07,-.41,.025)),
         'throw pillow','Loose ivory pillow propped at an angle on the long seat.',SOFT,'medium',on=modules[1].name)
    p=rounded_soft('Pillow | diagonal corner',(.877,7.286,.659),(.140,.506,.434),fabric,.054,.026,(.13,-.28,-.64))
    note(p,'throw pillow','Pale square cushion angled across the interior corner.',SOFT,'medium',on=modules[3].name)
    # Boucle horseshoe, with two enclosed tapered tips and one continuous stuffed bend.
    path=[];rads=[]
    for i in range(15):
        t=i/14;path.append((1.592-.048*t,5.228-.498*t,.485+.012*math.sin(math.pi*t)));rads.append((.005+.056*min(1,5*t),.004+.046*min(1,5*t)))
    for i in range(1,20):
        a=math.pi+math.pi*i/19;path.append((1.690+.146*math.cos(a),4.730+.111*math.sin(a),.492));rads.append((.064,.050))
    for i in range(1,15):
        t=i/14;path.append((1.836-.007*t,4.730+.481*t,.488+.007*math.sin(math.pi*t)));rads.append((.006+.061*min(1,5*(1-t)),.004+.044*min(1,5*(1-t))))
    note(sweep('Pillow | ivory horseshoe bolster',path,rads,furry),'bolster',
         'Long U-shaped boucle bolster with tapered tips on the front chaise.',SOFT,'medium',on=modules[0].name)
    note(shell('throw_taupe','Throw | taupe draped blanket','#A69D92'),'blanket',
         'Heavy taupe pile throw folded on the window seat and hanging over its end.',
         'Source frame 320 silhouette and metric VGGT depth sampled on the draped surface. Back and interior folds are occluded.','medium',on=modules[4].name)
    note(shell('throw_cream','Throw | ivory upper layer','#E2DACC'),'blanket',
         'Pale plush layer folded over the taupe throw; their textile connection is uncertain.',
         'Layer outline and same-frame VGGT depth in source frame 320. Exposed faces have measured relief.','medium',on='Throw | taupe draped blanket')
    p=controller('Controller | white and black', (1.302,4.930,.453),-.90,ivory,dark,alu)
    note(p,'game controller','White and black twin-stick gamepad on front chaise.',on=modules[0].name)
    unit=note(box('Power pack | black square',(1.438,4.941,.467),(.072,.096,.029),dark,.009),'electronics',
              'Small black rectangular device connected to a looping black lead.',on=modules[0].name)
    # Cable traced in frame 589. The free connector rests near the inside of the low arm.
    cord=[(1.435,4.995,.464),(1.389,5.017,.457),(1.323,5.086,.455),(1.239,5.100,.457),
          (1.144,5.090,.456),(1.051,4.991,.457),(.995,4.788,.463),(1.032,4.717,.471),(1.13,4.653,.495)]
    note(tube('Power pack | charging lead',cord,.0033,dark),'cable','Flexible black cable resting on the seat and rising toward the arm.',SOFT,'medium',unit)
    note(rod('Power pack | lead plug',(1.131,4.653,.495),(1.164,4.640,.495),.005,dark),'plug','Loose tip of the charging lead.',parent=unit)
    coffee(modules,rug,alu,dark,charcoal,ivory,blue,furry,existing_objects)

def coffee(modules,rug,alu,dark,charcoal,ivory,blue,furry,existing_objects):
    wood=mat('Coffee table | pale straight grain','#C2B38F',.54)
    nodes=wood.node_tree.nodes;links=wood.node_tree.links
    tc=nodes.new('ShaderNodeTexCoord');scale=nodes.new('ShaderNodeVectorMath');scale.operation='MULTIPLY'
    scale.inputs[1].default_value=(230,4.5,18);links.new(tc.outputs['Object'],scale.inputs[0])
    n=nodes.new('ShaderNodeTexNoise');n.inputs['Scale'].default_value=1.;n.inputs['Detail'].default_value=1.8
    links.new(scale.outputs[0],n.inputs['Vector']);r=nodes.new('ShaderNodeValToRGB')
    r.color_ramp.elements[0].color=(.31,.243,.160,1);r.color_ramp.elements[1].color=(.59,.505,.359,1)
    links.new(n.outputs['Fac'],r.inputs[0]);links.new(r.outputs['Color'],nodes.get('Principled BSDF').inputs['Base Color'])
    top=slabs('Coffee table | rounded oak top',2.019,6.131,.712,1.303,.396,.427,.236,wood)
    note(top,'coffee table','Light oak oblong low table, nearly straight sides with broad curved corners.',TABLE,'medium',on=rug.name)
    pm=mat('Coffee table | pedestal wood','#8E7E65',.64)
    for i,cy in enumerate([5.803,6.474]):
        note(slabs('Coffee table | slab support '+str(i+1),2.026,cy,.474,.176,.016,.398,.041,pm),'table support',
             'Broad inset pedestal supporting the table. Rear underside is substantially occluded.',
             'Front profile and floor contact visible in the source; hidden depth and far support inferred.','low',top)
    # The closed laptop has a physically separate keyboard enclosure and display lid.
    laptop=note(slabs('Laptop | aluminum base',2.041,5.876,.345,.235,.428,.439,.011,dark),'laptop',
                'Closed silver laptop with black lower edge, lying on the near half of the oak table.',
                'Corner rays intersect the table plane in frames 249 and 285.','medium',on=top.name)
    laptop.rotation_euler[2]=-.043
    lid=slabs('Laptop | silver display lid',2.041,5.876,.346,.236,.439,.444,.011,alu);lid.rotation_euler[2]=-.043
    note(lid,'hinged lid','Aluminum display lid, observed closed. Hinge runs along the long rear edge.',parent=laptop)
    # Cut-out silhouette of the small dark logo on the closed lid.
    logo=[(-.013,.008),(-.012,.013),(-.008,.016),(-.003,.015),(0,.013),(.006,.016),(.011,.013),(.013,.009),(.009,.008),(.008,.003),(.011,0),(.014,-.001),(.010,-.008),(.005,-.011),(0,-.008),(-.006,-.011),(-.011,-.008),(-.014,-.002),(-.015,.004)]
    verts=[(2.041+x,5.876+y,.4443) for x,y in logo]
    label=mesh('Laptop | inset maker mark',verts,[tuple(reversed(range(len(verts))))],charcoal)
    note(label,'inlay','Dark inset fruit-shaped emblem on the closed lid.',parent=lid)
    pages=mat('Book | paper edge','#D9D4C7',.93);jacket=mat('Book | sky blue jacket','#9BAEBF',.71)
    little=note(box('Book | dark lower volume',(2.043,6.106,.435),(.261,.217,.012),dark,.002),'book',
                'Thin dark volume beneath the larger table book; largely occluded.',on=top.name)
    book=note(box('Book | Los Angeles hardcover',(2.042,6.100,.465),(.273,.224,.045),pages,.003),
              'book','Thick pale blue Los Angeles coffee-table book with an exposed light page block.',on=little.name)
    b=box('Book | blue printed jacket',(2.042,6.100,.489),(.281,.229,.003),jacket,.001)
    note(b,'book cover','Pale blue upper cover of the Los Angeles volume.',parent=book)
    # The visible lower strip carries silver-white typesetting. It stays attached to the cover.
    from common import link
    txt=bpy.data.curves.new('Volume title lettering','FONT');txt.body='Los Angeles';txt.size=.032;txt.align_x='CENTER';txt.align_y='CENTER';txt.extrude=.00005;txt.materials.append(ivory)
    tt=link('Book | jacket title',txt);tt.location=(2.04,6.014,.4908);note(tt,'lettering','Title printed on the visible edge of the cover.',parent=b)
    pad=controller('Controller | powder blue',(1.790,5.975,.430),-1.44,blue,dark,alu)
    note(pad,'game controller','Second twin-stick gamepad with a pale blue molded shell and black controls.',on=top.name)
    for name,x,y,le,ang in [('Television remote | long',1.807,6.335,.141,-.16),
                             ('Streaming remote | small',1.877,6.100,.116,-.13)]:
        note(remote(name,x,y,.428,le,ang,dark,charcoal),'remote control','Dark handheld remote with raised keypad.',on=top.name)
    mini=uv('Key fob | dark oval',(1.902,6.365,.439),(.090,.045,.021),charcoal);mini.rotation_euler[2]=.19
    note(mini,'small accessory','Small gray-black oval item behind the glass candle; identity uncertain.',on=top.name)
    stick=note(rounded_soft('Remote | pocket-size black',(2.011,6.070,.501),(.033,.079,.017),dark,.006,.002,(0,0,-.20)),
                'small accessory','Small smooth black item resting on the large book in front of the candle.',on=book.name)
    # Two stacks of separate cream stone coasters in gold-tone wire stands.
    brass=mat('Accessories | brass','#A28A52',.34,.57);stone=mat('Coasters | cream stone','#E9E0D1',.73)
    for i,(x,y,count) in enumerate([(2.142,6.168,5),(2.134,6.061,5)]):
        z=.491
        base=cyl('Coasters | stand base '+str(i+1),(x,y,z+.002),.0485,.0038,brass)
        note(base,'coaster holder','Round gold-tone wire holder for stacked cream coasters.',on=book.name)
        for j in range(3):
            a=math.tau*j/3+.3;px=x+.048*math.cos(a);py=y+.048*math.sin(a)
            side=sweep('Coaster holder prong',[(px,py,z),(px,py,z+.038),(px-.002*math.cos(a),py-.002*math.sin(a),z+.041)],[(.0013,.0013)]*3,brass,8)
            # Permanently attached wire parts are one physical holder.
            base=joinparts([base,side],base.name)
        prev=base
        for j in range(count):
            co=cyl(f'Coaster | stack {i+1} disc {j+1}',(x+.001*(j%2),y,z+.008+.0074*j),.0445,.0065,stone)
            note(co,'coaster','Individual pale round coaster. Visible edge bands constrain the estimated count.',parent=None,on=prev.name)
            prev=co
    glass=mat('Candle | smoke-tinted glass','#968C83',.30,.16)
    cx,cy,cz=2.031,6.169,.491
    # Hollow beaker with rim, bottom, inner surface and a wax insert.
    rings=[(.038,cz),(.042,cz+.005),(.044,cz+.080),(.0408,cz+.080),(.0395,cz+.010),(.036,cz+.007)]
    verts=[];faces=[];N=48
    for radius,h in rings:
        for j in range(N):
            a=math.tau*j/N;verts.append((cx+radius*math.cos(a),cy+radius*math.sin(a),h))
    for k in range(len(rings)-1):
        for j in range(N):
            d=(j+1)%N;faces.append((k*N+j,k*N+d,(k+1)*N+d,(k+1)*N+j))
    faces.extend([tuple(reversed(range(N))),tuple(range((len(rings)-1)*N,len(rings)*N))])
    vessel=mesh('Candle | smoky glass tumbler',verts,faces,glass);note(vessel,'candle vessel','Translucent gray glass tumbler with open rim.',on=book.name)
    for f in list(vessel.data.polygons)[:-2]:f.use_smooth=True
    wax=note(cyl('Candle | ivory wax',(cx,cy,cz+.027),.039,.039,furry),'wax insert','Pale unlit wax inside the smoky glass.',parent=vessel)
    note(rod('Candle | unlit wick',(cx,cy,cz+.046),(cx+.001,cy,cz+.053),.0014,dark),'wick','Single dark unlit wick.',parent=wax)
    # Stepped aluminum can with open top and a printed violet lower label.
    can_label=mat('Beverage can | lavender','#A0A4C2',.49,.18)
    band=mat('Beverage can | silver white','#D3D3D5',.44,.34)
    cx,cy=2.124,5.604
    heights=[(.000,.025),(.002,.0285),(.006,.029),(.011,.032),(.015,.033),(.103,.033),(.111,.030),(.116,.0272),(.120,.0282),(.121,.0267)]
    verts=[];faces=[];N=56
    for h,rr in heights:
        for k in range(N):
            a=k*math.tau/N;verts.append((cx+rr*math.cos(a),cy+rr*math.sin(a),.428+h))
    for i in range(len(heights)-1):
        for k in range(N):
            d=(k+1)%N;faces.append((i*N+k,i*N+d,(i+1)*N+d,(i+1)*N+k))
    faces.extend([tuple(reversed(range(N))),tuple(range(len(verts)-N,len(verts)))])
    can=mesh('Beverage | opened citrus can',verts,faces,band);can.data.materials.append(alu)
    for f in list(can.data.polygons)[:-2]:f.use_smooth=True
    for f in list(can.data.polygons)[7*N:]:f.material_index=1
    note(can,'beverage can','Opened aluminum drink can with pale purple and silver-white label and citrus motif.',on=top.name)
    wrap=cyl('Beverage | lower violet print',(cx,cy,.455),.03312,.044,can_label)
    note(wrap,'printed label','Lavender lower portion of the can design.',parent=can)
    yellow=mat('Beverage can | lemon','#E1BD5B',.64)
    emblem=uv('Beverage | citrus print',(cx,cy-.0325,.474),(.032,.0013,.030),yellow)
    note(emblem,'printed motif','Small yellow citrus motif facing the room.',parent=can)
    opening=uv('Beverage | open aperture',(cx-.003,cy-.006,.5496),(.019,.011,.0007),dark)
    note(opening,'can opening','Dark drinking aperture in the top lid.',parent=can)
    ring=[(cx+.0075*math.cos(i*math.tau/22),cy+.007+.012*math.sin(i*math.tau/22),.5505) for i in range(23)]
    note(tube('Beverage | pull tab',ring,.0019,alu),'pull tab','Raised aluminum pull ring alongside the open aperture.',parent=can)
    for i,(x,y) in enumerate([(1.978,5.677),(1.967,5.692)]):
        note(uv('Small brass bead '+str(i+1),(x,y,.4337),(.011,.010,.011),brass),'small item','One of two closely spaced gold-tone or brass bead-like items; identity uncertain.',on=top.name)
    # Flat, rounded cream object leaning against the short face of the pedestal.
    spine=[];radii=[]
    for i in range(81):
        a=math.tau*i/80;spine.append((2.020+.070*math.cos(a),5.686+.012*math.sin(2*a),.079+.041*math.sin(a)))
        radii.append((.021,.018))
    knot=sweep('Floor textile | small cream knot',spine,radii,furry)
    extra=[]
    for s in [-1,1]:
        path=[(1.983+.073*j/9,5.674+.008*math.cos(j*math.pi/9),.077+s*.020*math.cos(j*math.pi/9)) for j in range(10)]
        extra.append(sweep('Folded knot interior',path,[(.012,.012)]*10,furry))
    knot=joinparts([knot]+extra,knot.name)
    note(knot,'small soft object','Round cream textile-like object with folded center resting beside the table pedestal. Its use is uncertain.',on=rug.name)
    # The table pedestal hides most of a separate black-spotted soft object.
    spotted=mat('Floor textile | black speckled pile','#9C9B97',.99)
    nodes=spotted.node_tree.nodes;links=spotted.node_tree.links
    coord=nodes.new('ShaderNodeTexCoord');scale=nodes.new('ShaderNodeVectorMath')
    scale.operation='MULTIPLY';scale.inputs[1].default_value=(108,95,131)
    links.new(coord.outputs['Object'],scale.inputs[0])
    noise=nodes.new('ShaderNodeTexVoronoi');noise.inputs['Scale'].default_value=1.
    links.new(scale.outputs[0],noise.inputs['Vector']);tone=nodes.new('ShaderNodeValToRGB')
    tone.color_ramp.elements[0].position=.30;tone.color_ramp.elements[0].color=(.018,.017,.019,1)
    tone.color_ramp.elements[1].position=.47;tone.color_ramp.elements[1].color=(.41,.40,.39,1)
    links.new(noise.outputs['Distance'],tone.inputs[0]);links.new(tone.outputs['Color'],nodes.get('Principled BSDF').inputs['Base Color'])
    low=rounded_soft('Floor textile | dark spotted item',(2.255,6.006,.054),(.241,.326,.075),spotted,.028,.009,(.02,.07,-.18))
    note(low,'small soft object','Gray-black spotted textile peeking out from under the right edge of the coffee table. Most of its shape is concealed.',
         'Visible outline in source frames 210 and 249 and local floor contact. Hidden extent, thickness and object identity are estimates.','low',on=rug.name,first=150)
    # Conservative first-observed anchor frames, zero-based, for time-specific manifest metadata.
    for obj in set(bpy.context.scene.objects)-existing_objects:
        name=obj.name
        if name.startswith(('Coffee table','Rug |')):obj['first_observed_frame']=107
        elif name.startswith(('Book |','Beverage |','Coaster','Candle |','Controller | powder','Television remote','Streaming remote','Key fob','Laptop |','Remote | pocket','Floor textile','Small brass')):obj['first_observed_frame']=157
        elif name.startswith('Throw |'):obj['first_observed_frame']=172
        elif name.startswith(('Controller | white','Power pack')):obj['first_observed_frame']=138
        elif name.startswith(('Sectional | window','Sectional | corner','Pillow | diagonal')):obj['first_observed_frame']=125

    # Approximate visibility from the source and a depth-tested projection of
    # evaluated meshes into the reconstructed cameras (three-frame sampling).
    # Hidden supports inherit identification of the sofa module supporting them.
    seen={
        'Rug | rectangular distressed weave':78,'Rug | stitched perimeter':78,
        'Pillow | ivory horseshoe bolster':81,'Pillow | diagonal corner':75,
        'Sectional | chaise seat':75,'Sectional | middle seat':69,'Sectional | inner seat':78,
        'Sectional | corner seat':78,'Sectional | window return seat':99,
        'Sectional | middle frame':141,'Sectional | middle inset foot':141,
        'Sectional | inner frame':93,'Sectional | inner inset foot':93,
        'Sectional | corner frame':96,'Sectional | corner inset foot':96,
        'Sectional | window return frame':96,'Sectional | window return inset foot':183,
        'Sectional | back cushion 5':78,'Sectional | back cushion 6':102,
        'Throw | taupe draped blanket':144,'Throw | ivory upper layer':165,
        'Controller | white and black':90,'Power pack | black square':78,
        'Power pack | charging lead':105,'Power pack | lead plug':569,
        'Coffee table | rounded oak top':99,'Coffee table | slab support 1':120,
        'Coffee table | slab support 2':120,'Controller | powder blue':105,
        'Laptop | aluminum base':120,'Laptop | silver display lid':114,
        'Laptop | inset maker mark':210,'Floor textile | small cream knot':120,'Floor textile | dark spotted item':150,
        'Beverage | opened citrus can':156,'Beverage | lower violet print':156,
        'Beverage | citrus print':159,'Beverage | open aperture':258,'Beverage | pull tab':207,
    }
    for name,source_frame in seen.items():
        obj=bpy.data.objects.get(name)
        if obj is not None:obj['first_observed_frame']=source_frame
