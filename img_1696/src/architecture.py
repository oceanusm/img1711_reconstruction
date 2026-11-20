"""Room envelope, openings, fireplace, window grouping and occasional furnishings."""
import bpy, math, random
from pathlib import Path
from mathutils import Vector
from common import box,mesh,uv,cyl,rod,tube,mat,tag,linear
from math import sin,cos,pi
ASSETS=Path(__file__).parent/'assets'
WALL_BASIS="Aligned floor and vertical wall samples in the VGGT cloud. Opening boundaries from video rays intersected with the fitted wall. Hidden build-up thickness estimated."
DETAIL_BASIS="Observed outlines, perspective and contacts across the source views. Thickness and concealed construction are estimated."
FIRE_BASIS="Colored display points and upper/lower firebox boundaries in the aligned cloud, cross-checked against source frames 385–491. Stand-off depth estimated."
CHAIR_BASIS="Seat contacts, arms and backs in source frames 346–385 and 952–991. Floor and height samples guide the rotated footprint."
def reg(o,k,desc,basis=DETAIL_BASIS,conf='low',parent=None,first=0):
    tag(o,k,desc,basis,conf,parent)
    if first:o['first_observed_frame']=first
    return o

def tex(name,file,color,rough=.65,emission=0):
    m=mat(name,color,rough); nodes=m.node_tree.nodes; links=m.node_tree.links
    im=nodes.new('ShaderNodeTexImage');im.image=bpy.data.images.load(str(ASSETS/file),check_existing=True)
    im.image.pack();bs=nodes.get('Principled BSDF');links.new(im.outputs['Color'],bs.inputs['Base Color'])
    if emission:links.new(im.outputs['Color'],bs.inputs['Emission Color']);bs.inputs['Emission Strength'].default_value=emission
    return m

def emit(name,color,strength):
    m=mat(name,color,.65);bs=m.node_tree.nodes.get('Principled BSDF')
    bs.inputs['Emission Color'].default_value=m.diffuse_color
    bs.inputs['Emission Strength'].default_value=strength
    return m

def finish_face(o,which,m,uvmap):
    data=o.data; layer=data.uv_layers.get('Surface UV') or data.uv_layers.new(name='Surface UV')
    data.materials.append(m)
    p=data.polygons[which];p.material_index=len(data.materials)-1
    for j,li in enumerate(p.loop_indices):layer.data[li].uv=uvmap[j]

def lathe(name,x,y,profile,m,n=64):
    vs=[(x+r*cos(2*pi*j/n),y+r*sin(2*pi*j/n),z) for z,r in profile for j in range(n)]
    annular=abs(profile[0][0]-profile[-1][0])<.0001
    fs=[] if annular else [tuple(reversed(range(n)))]
    for i in range(len(profile)-1):
        for j in range(n):fs.append((i*n+j,i*n+(j+1)%n,(i+1)*n+(j+1)%n,(i+1)*n+j))
    if annular:
        for j in range(n):fs.append(((len(profile)-1)*n+j,(len(profile)-1)*n+(j+1)%n,(j+1)%n,j))
    else:fs.append(tuple(range((len(profile)-1)*n,len(profile)*n)))
    o=mesh(name,vs,fs,m)
    for p in o.data.polygons[1:-1]:p.use_smooth=True
    return o

def timber_floor():
    base=mat('Oak | cut edges',(.66,.60,.49))
    top=tex('Whitewashed oak | grain','oak-grain.jpg',(.79,.74,.65))
    vs=[];fs=[];uvcoords=[];r=random.Random(28)
    for col in range(24):
        x=-.12+col*.215
        y=-4.35-r.random()*1.95
        while y<8.25:
            edge=y+[1.61,1.96,2.22][r.randrange(3)]
            y0=max(-4.25,y)+.00065;y1=min(8.24,edge)-.00065
            if y1>y0:
                x0=x+.00065;x1=x+.21435;start=len(vs)
                vs.extend([(xx,yy,z) for z in [-.038,0] for xx,yy in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]])
                fs.extend([tuple(start+n for n in face) for face in [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]])
                offset=r.random();shift=r.random()*.55; span=(y1-y0)/2.3
                uvcoords.append([(shift,offset),(shift+1,offset),(shift+1,offset+span),(shift,offset+span)])
            y=edge
    o=mesh('Floor | staggered oak planks',vs,fs,base)
    o.data.materials.append(top);uvl=o.data.uv_layers.new(name='Plank grain')
    for i,uvs in enumerate(uvcoords):
        p=o.data.polygons[6*i+1];p.material_index=1
        for li,uvp in zip(p.loop_indices,uvs):uvl.data[li].uv=uvp
    reg(o,'floor finish','Staggered, pale wide-plank floor with individual end joints.',WALL_BASIS,'medium')
    return o

def windows(plaster,dark):
    el=mat('White architectural enamel',(.935,.93,.91))
    shade=mat('Translucent ivory shade cloth',(.901,.903,.875),.92)
    glass=mat('Clear window glazing',(.96,.98,1),.08)
    bs=glass.node_tree.nodes.get('Principled BSDF');bs.inputs['Alpha'].default_value=.032
    glass.diffuse_color=(.86,.95,1,.032)
    plaster_sections=[(-.13,.015),(1.35,2.14),(2.65,3.37),(4.54,4.99)]
    for i,(a,b) in enumerate(plaster_sections):
        reg(box(f'Window wall | pier {i+1}',((a+b)/2,8.200,1.285),(b-a,.17,2.57),plaster),'wall','Painted wall between full-height openings.',WALL_BASIS,'medium')
        if b-a>.26:reg(box(f'Window wall | skirting {i+1}',((a+b)/2,8.102,.042),(b-a,.022,.084),el,.0015),'baseboard','Painted low skirting.',WALL_BASIS)
    reg(box('Window wall | head',(2.45,8.20,2.546),(5.10,.17,.047),plaster),'lintel','Continuous painted lintel above the window frames.',WALL_BASIS)
    for i,(a,b,lower,loc) in enumerate([(.015,1.35,1.57,0),(2.14,2.65,.913,160),(3.37,4.54,1.13,315)],1):
        mid=(a+b)/2; width=b-a
        root=reg(box(f'Window {i} | sill',(mid,8.175,.102),(width,.15,.064),el,.002),'window sill','Painted interior reveal at the base of the glazed opening.',WALL_BASIS,'low',first=loc)
        reg(box(f'Window {i} | lower apron',(mid,8.20,.038),(width,.17,.076),plaster),'wall','Painted substrate under the sill.',WALL_BASIS,parent=root,first=loc)
        for name,x in [('left',a+.015),('right',b-.015)]:
            reg(box(f'Window {i} | {name} stile',(x,8.197,1.33),(.030,.069,2.385),dark),'window frame','Powder-coated black metal jamb.',WALL_BASIS,parent=root,first=loc)
        for label,z,th in [('head',2.50,.035),('bottom rail',.147,.031),('meeting rails',1.006,.072)]:
            reg(box(f'Window {i} | {label}',(mid,8.192,z),(width,.071,th),dark),'window frame','Dark horizontal metal frame at the glazing.',WALL_BASIS,parent=root,first=loc)
        for label,zlo,zhi in [('upper',1.04,2.482),('lower',.163,.970)]:
            reg(box(f'Window {i} | {label} pane',(mid,8.213,(zlo+zhi)/2),(width-.05,.006,zhi-zlo),glass),'window glazing','Transparent pane held in the frame.',WALL_BASIS,parent=root,first=loc)
        screen=reg(box(f'Window {i} | roller blind',(mid,8.124,(lower+2.50)/2),(width-.01,.004,2.50-lower),shade,.001),'roller shade','Light fabric roller blind lowered partway.',WALL_BASIS,parent=root,first=loc)
        reg(box(f'Window {i} | weighted bottom rail',(mid,8.123,lower),(width-.009,.011,.014),el,.003),'shade rail','Narrow stiffener along the bottom of the blind.',DETAIL_BASIS,parent=screen,first=loc)
        roll=reg(rod(f'Window {i} | shade barrel',(a+.006,8.135,2.492),(b-.006,8.135,2.492),.027,shade),'roller spindle','Fabric-wrapped shade roller.',DETAIL_BASIS,parent=screen,first=loc)
    wood=mat('Botanical picture | frame',(.26,.235,.188))
    art=reg(box('Window pier | botanical artwork',(1.787,8.086,1.694),(.500,.041,.993),wood,.001),'framed print','Portrait-format landscape with two white herons, reeds and overhanging branches.',WALL_BASIS,'medium',first=126)
    finish_face(art,2,tex('Botanical picture | printed surface','botanical-etching.jpg',(.65,.6,.49)),[(0,0),(1,0),(1,1),(0,1)])
    for i,(x,z) in enumerate([(3.050,.433),(2.893,1.99)]):
        plate=reg(box(f'Window pier | electrical plate {i+1}',(x,8.095,z),(.071,.014,.114),el,.004),'wall plate','Ivory electrical receptacle or switch plate.',DETAIL_BASIS,first=327)
        for j in range(2):
            if i==0:
                for k in [-1,1]:reg(box(f'Receptacle {j+1} | slot {k}',(x+k*.009,8.086,z+(.021 if j==0 else -.022)),(.003,.002,.009),dark),'socket detail','Recessed contact slot.',DETAIL_BASIS,parent=plate,first=327)
            else:reg(box('Switch | paddle '+str(j+1),(x+.015*(2*j-1),8.085,z),(.021,.008,.045),el,.001),'switch','Rocker paddle.',DETAIL_BASIS,parent=plate,first=327)

def media_wall(dark):
    slope=-.0374
    def wx(y):return 4.5834+slope*(y-5.665)
    def P(x,y,z):return (wx(y)+x,y,z)
    cladding=mat('Media wall | charcoal ridged stone',(.375,.36,.359),.77)
    n=cladding.node_tree.nodes;ln=cladding.node_tree.links;noise=n.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=135;noise.inputs['Detail'].default_value=2
    ramp=n.new('ShaderNodeValToRGB')
    for e,c in zip(ramp.color_ramp.elements,[.30,.46]):e.color=(linear(c),linear(c*.975),linear(c*.97),1)
    ln.new(noise.outputs['Fac'],ramp.inputs[0]);ln.new(ramp.outputs[0],n.get('Principled BSDF').inputs['Base Color'])
    vs=[];fs=[]
    def section(ya,yb,za,zb):
        N=max(1,round((zb-za)/.0165))
        for k in range(N):
            z=za+(zb-za)*k/N;h=(zb-za)/N
            ring=[(.105,z),(.002,z),(-.002,z+.0023),(-.002,z+h-.0025),(.003,z+h),(.105,z+h)]
            idx=len(vs)
            for y in (ya,yb):vs.extend(P(x,y,zz) for x,zz in ring)
            fs.append(tuple(idx+i for i in range(6)));fs.append(tuple(idx+6+i for i in reversed(range(6))))
            for i in range(6):fs.append((idx+i,idx+i+6,idx+(i+1)%6+6,idx+(i+1)%6))
    section(3.58,7.61,0,.392);section(3.58,7.61,.787,2.57);section(3.58,4.30,.392,.787);section(6.72,7.61,.392,.787)
    wall=reg(mesh('Media wall | fluted charcoal cladding',vs,fs,cladding),'wall finish','Horizontally ribbed charcoal cladding around the recessed linear fire.',FIRE_BASIS,'medium',first=358)
    def wb(name,y,z,dx,dy,dz,offset,material):
        o=box(name,P(offset,y,z),(dx,dy,dz),material)
        o.rotation_euler[2]=math.atan(-slope)
        return o
    black=mat('Firebox | blackened metal',(.09,.08,.075),.33,.48)
    fire=reg(wb('Fireplace | inner dark back',5.51,.59,.025,2.42,.395,.098,black),'electric fireplace','Wide inset firebox, with dark interior, a lit ember strip and slim surround.',FIRE_BASIS,'medium',first=388)
    effect=mat('Fireplace | animated simulated flames',(.26,.20,.125),.38,.28)
    fire.data.materials.clear();fire.data.materials.append(effect)
    nd=effect.node_tree.nodes;lk=effect.node_tree.links;bs=nd.get('Principled BSDF')
    coords=nd.new('ShaderNodeTexCoord');sep=nd.new('ShaderNodeSeparateXYZ');lk.new(coords.outputs['Generated'],sep.inputs[0])
    axis=nd.new('ShaderNodeVectorMath');axis.operation='MULTIPLY';axis.inputs[1].default_value=(0,33,3.4)
    lk.new(coords.outputs['Generated'],axis.inputs[0])
    flick=nd.new('ShaderNodeTexNoise');flick.noise_dimensions='4D';flick.inputs['Scale'].default_value=1.7;flick.inputs['Detail'].default_value=2.5;flick.inputs['Distortion'].default_value=.42
    lk.new(axis.outputs['Vector'],flick.inputs['Vector'])
    for f,w in [(1,0.0),(992,23.5)]:
        flick.inputs['W'].default_value=w;flick.inputs['W'].keyframe_insert(data_path='default_value',frame=f)
    height=nd.new('ShaderNodeMath');height.operation='MULTIPLY';height.inputs[1].default_value=1.38;lk.new(sep.outputs['Z'],height.inputs[0])
    sub=nd.new('ShaderNodeMath');sub.operation='SUBTRACT';lk.new(flick.outputs['Fac'],sub.inputs[0]);lk.new(height.outputs[0],sub.inputs[1])
    colors=nd.new('ShaderNodeValToRGB');cr=colors.color_ramp;cr.elements[0].position=.19;cr.elements[0].color=(.083,.059,.037,1)
    cr.elements[1].position=.67;cr.elements[1].color=(1,.79,.225,1)
    for v,col in [(.30,(.175,.115,.047,1)),(.395,(.60,.308,.031,1)),(.49,(1,.66,.075,1))]:
        e=cr.elements.new(v);e.color=col
    lk.new(sub.outputs[0],colors.inputs[0]);lk.new(colors.outputs[0],bs.inputs['Base Color']);lk.new(colors.outputs[0],bs.inputs['Emission Color']);bs.inputs['Emission Strength'].default_value=1.0
    for name,y,z,dy,dz in [('bottom',5.51,.399,2.43,.026),('top',5.51,.781,2.43,.026),('left jamb',6.706,.59,.027,.38),('right jamb',4.313,.59,.027,.38)]:
        reg(wb('Fireplace | '+name,y,z,.105,dy,dz,.035,black),'firebox surround','Slim dark metal perimeter of inset opening.',FIRE_BASIS,parent=fire,first=388)
    # Separate physical bed of molded translucent ember stones with time-varying illumination.
    amber=emit('Fireplace | illuminated amber',(.98,.66,.14),2.0)
    ember=reg(wb('Fireplace | ember bed',5.51,.422,.080,2.34,.020,.049,amber),'ember bed','Narrow illuminated bed immediately above the firebox sill.',FIRE_BASIS,parent=fire,first=388)
    r=random.Random(1696);vs=[];fs=[];mats=[]
    for i in range(177):
        y=r.uniform(4.35,6.65);x=r.uniform(.013,.069);z=.427;w=r.uniform(.008,.026);h=r.uniform(.012,.043);b=len(vs)
        for zz,rr in [(0,.7),(h*.42,1),(h,.43)]:
            for j in range(5):
                a=j*2*pi/5+i;vs.append(P(x+rr*w*cos(a),y+rr*w*sin(a),z+zz))
        for k in range(2):
            for j in range(5):fs.append((b+k*5+j,b+k*5+(j+1)%5,b+(k+1)*5+(j+1)%5,b+(k+1)*5+j));mats.append(i%3)
        fs.append(tuple(b+10+j for j in range(5)));mats.append(i%3)
    stones=reg(mesh('Fireplace | decorative crystal fuel',vs,fs,amber),'decorative fire media','Faceted warm crystals across the firebed.',DETAIL_BASIS,parent=fire,first=388)
    for material in (emit('Fireplace | gold crystals',(1.,.84,.35),2.2),emit('Fireplace | caramel crystals',(.75,.42,.074),1.4)):stones.data.materials.append(material)
    for p,i in zip(stones.data.polygons,mats):p.material_index=i
    for j,mt in enumerate(stones.data.materials):
        s=mt.node_tree.nodes.get('Principled BSDF').inputs['Emission Strength']
        for f in range(1,993,10):
            s.default_value=1.4+.40*sin(f*.21+j)+.25*sin(f*.062-j);s.keyframe_insert(data_path='default_value',frame=f)
    glass=mat('Firebox | heat-resistant glazing',(.41,.33,.19),.24)
    bs=glass.node_tree.nodes.get('Principled BSDF');bs.inputs['Alpha'].default_value=.16
    reg(wb('Fireplace | protective glass',5.51,.59,.006,2.38,.356,-.004,glass),'glazing','Slightly warm safety glazing across the recessed box.',DETAIL_BASIS,parent=fire,first=388)
    cy=5.49;cz=1.54
    screen=reg(wb('Television | chassis',cy,cz,.052,1.68,.956,-.063,dark),'television','Thin dark television fixed above the electric fireplace.',FIRE_BASIS,'medium',first=391)
    face=reg(wb('Television | display',cy,cz+.002,.006,1.661,.930,-.092,dark),'display','Recreated purple nighttime media graphic, with illuminated architecture and a weather indicator.',FIRE_BASIS,parent=screen,first=391)
    # -X face of the local box: first loop is Y+, lower corner. Viewer right points toward decreasing world Y.
    finish_face(face,5,tex('TV | purple architectural broadcast','display-clean.jpg',(.47,.15,.66),.32,.72),[(0,0),(1,0),(1,1),(0,1)])
    reg(wb('Television | lower bezel badge',cy,1.068,.004,.038,.010,-.095,mat('TV badge | silver',(.70,.70,.71),.32,.65)),'badge','Small center badge on the lower screen bezel.',DETAIL_BASIS,parent=screen,first=405)
    pts=[P(-.035,y,z) for y,z in [(5.24,1.07),(5.21,1.03),(5.22,.96),(5.17,.934),(5.10,.94),(5.09,.89),(5.14,.86),(5.22,.87),(5.24,.90),(5.20,.94)]]
    reg(tube('Television | hanging black cable',pts,.0041,dark),'cable','Short loose cable loop beneath the television.',DETAIL_BASIS,parent=screen,first=414)
    return wall

def foliage(name,cx,cy,z0,h,span,seed,first):
    rng=random.Random(seed);clay=mat('Plant pot | graphite clay',(.30,.305,.30),.85)
    potheight=.255 if seed==9 else .33;rad=.129 if seed==9 else .155
    pot=reg(lathe(name+' | planter',cx,cy,[(z0,rad*.77),(z0+.018,rad*.82),(z0+potheight-.012,rad),(z0+potheight,rad*.99),(z0+potheight,rad*.90),(z0+potheight-.050,rad*.88)],clay),'planter','Dark tapered planter supporting the arching fronds.',DETAIL_BASIS,first=first)
    soil=reg(cyl(name+' | potting soil',(cx,cy,z0+potheight-.041),rad*.89,.017,mat('Planting medium',(.123,.095,.054))),'soil','Dark growing medium inside the pot.',DETAIL_BASIS,parent=pot,first=first)
    foliage_colors=[(.248,.359,.147),(.32,.443,.174),(.421,.529,.214),(.204,.320,.154),(.506,.570,.226)]
    ms=[mat(f'Fern | leaflet {i+1}',c,.70) for i,c in enumerate(foliage_colors)]
    verts=[];faces=[];midx=[]
    count=24
    for i in range(count):
        angle=(i*2.399963+rng.uniform(-.17,.17));v=Vector((cos(angle),sin(angle),0));perp=Vector((-sin(angle),cos(angle),0))
        reach=span*rng.uniform(.55,1)*(1 if i%5 else .52);height=h*rng.uniform(.67,1.05)
        center=Vector((cx,cy,z0+potheight-.04))+v*rng.uniform(.005,.04)
        def stem(t):return center+v*reach*t**1.52+Vector((0,0,height*(1.48*t-.67*t*t)))
        for k in range(18):
            t=k/18;v1=stem(t);v2=stem((k+1)/18)
            j=len(verts);rr=.0028*(1-.65*t);verts.extend((v1+perp*rr,v1-perp*rr,v2-perp*rr,v2+perp*rr));faces.append((j,j+1,j+2,j+3));midx.append(0)
        for k in range(3,24):
            t=k/25;c=stem(t);length=(.022+span*.22*sin(pi*t)**.70)*rng.uniform(.76,1.12)
            for s in (-1,1):
                sweep=perp*(s*length)+v*(.28*length)
                tip=c+sweep+Vector((0,0,.12*length))
                mid=c+sweep*.52+Vector((0,0,.13*length))
                w=length*.13
                j=len(verts);verts.extend([c,mid-v*w,mid+Vector((0,0,w*.42)),mid+v*w,tip])
                faces.extend([(j,j+1,j+2),(j,j+2,j+3),(j+1,j+4,j+2),(j+2,j+4,j+3)])
                midx.extend([(i+k)%len(ms)]*4)
    fern=reg(mesh(name+' | fern',verts,faces,ms[0]),'potted plant','Arching pinnate fronds with individually shaped leaflets.',DETAIL_BASIS,parent=pot,first=first)
    for m in ms[1:]:fern.data.materials.append(m)
    for p,idx in zip(fern.data.polygons,midx):p.material_index=idx
    return pot

def occasional():
    dark=mat('Side table | black stained timber',(.139,.134,.132),.38)
    cream=mat('Lamp | cream ceramic',(.868,.849,.759),.35)
    linen=mat('Lamp | off-white fabric shade',(.893,.879,.780),.82)
    x,y=3.21,7.72
    table=reg(lathe('Black end table | turned body',x,y,[(.005,.151),(.011,.161),(.034,.179),(.096,.184),(.16,.163),(.217,.115),(.259,.094),(.286,.115),(.320,.153),(.37,.187),(.45,.201),(.49,.198),(.503,.192)],dark),'side table','Black hourglass-shaped pedestal end table beside the armchair.',DETAIL_BASIS,'medium',first=320)
    x,y=3.245,7.781
    body=reg(lathe('Table lamp | ceramic base',x,y,[(.505,.052),(.512,.076),(.521,.083),(.551,.092),(.581,.095),(.612,.083),(.673,.057),(.741,.037),(.766,.031),(.774,.031)],cream),'table lamp','Cream ceramic bulbous lamp with a tapered neck on the black side table.',DETAIL_BASIS,parent=None,first=327)
    body['relationships_json']='{"on":"Black end table | turned body"}'
    reg(lathe('Table lamp | open drum shade',x,y,[(.718,.144),(.725,.146),(.978,.136),(.987,.133),(.987,.129),(.979,.131),(.724,.141),(.718,.140)],linen),'lampshade','Ivory drum shade with open interior and narrow rims.',DETAIL_BASIS,parent=body,first=327)
    reg(rod('Table lamp | cord',(3.2,7.78,.50),(3.03,8.055,.018),.0028,mat('Ivory cable',(.79,.76,.66))),'cable','Pale power lead trailing from the lamp, route behind the table estimated.',DETAIL_BASIS,parent=body,first=327)
    # Tall fixture beside the left window. Its foot is concealed by the sectional.
    x,y=.135,7.60
    f=reg(cyl('Floor lamp | weighted foot',(x,y,.020),.122,.035,dark),'floor lamp','Tall slim lamp at the end of the console beside the left window.',DETAIL_BASIS,first=43)
    reg(rod('Floor lamp | stem',(x,y,.029),(x,y,.88),.012,cream),'lamp stem','Slim support inferred behind the sofa.',DETAIL_BASIS,parent=f,first=43)
    reg(lathe('Floor lamp | cylindrical shade',x,y,[(.829,.131),(.834,.134),(1.478,.134),(1.486,.131),(1.486,.126),(1.477,.128),(.838,.128),(.829,.126)],linen),'lampshade','Tall pale cylindrical shade visible beyond the console.',DETAIL_BASIS,parent=f,first=43)
    # Lounge chair axes determined from the projected front edge at z=.405.
    a=math.radians(-31.5); right=Vector((cos(a),sin(a),0)); back=Vector((-sin(a),cos(a),0));origin=Vector((3.849,7.446,0))
    def pos(x,y,z):return origin+right*x+back*y+Vector((0,0,z))
    fabric=mat('Armchair | oatmeal weave',(.84,.805,.726),.91)
    frame=mat('Armchair | dark stained wood',(.206,.171,.137),.52)
    def chairbox(name,xyz,size,material,bevel,parent=None):
        o=box(name,pos(*xyz),size,material,bevel);o.rotation_euler.z=a
        return reg(o,'chair' if parent is None else 'chair component',name.split('|')[-1].strip().capitalize()+' of the freestanding upholstered lounge chair.',CHAIR_BASIS,'medium' if parent is None else 'low',parent,343)
    chair=chairbox('Window armchair | seat frame',(0,0,.203),(.658,.638,.145),fabric,.023)
    cushion=chairbox('Window armchair | seat cushion',(0,-.046,.325),(.52,.522,.156),fabric,.047,chair)
    backc=chairbox('Window armchair | back cushion',(0,.241,.521),(.562,.147,.417),fabric,.042,chair)
    backc.rotation_euler.x=math.radians(-6.0)
    for side,s in [('left',-1),('right',1)]:
        chairbox('Window armchair | '+side+' arm',(s*.282,-.018,.376),(.104,.575,.323),fabric,.030,chair)
        chairbox('Window armchair | '+side+' rail',(s*.308,-.005,.147),(.040,.641,.052),frame,.004,chair)
        for end,yy in [('front',-.287),('rear',.278)]:
            chairbox(f'Window armchair | {side} {end} leg',(s*.308,yy,.120),(.040,.047,.239),frame,.004,chair)
    foliage('Window fern',4.17,7.892,0,1.14,.50,12,339)
    foliage('Bar fern',4.768,3.103,0,1.22,.49,9,472)
    silver=mat('Mirror | brushed thin rim',(.50,.51,.50),.29,.82)
    mirror=reg(cyl('Bar mirror | metal rim',(3.775,2.865,1.603),.434,.034,silver,96),'wall mirror','Round mirror with a narrow metallic surround centered over the bar cart.',WALL_BASIS,'medium',first=486)
    mirror.rotation_euler.x=pi/2
    m=mat('Mirror | silvered glass',(.90,.935,.93),.035,1.)
    disk=cyl('Bar mirror | silvered glass',(3.775,2.887,1.603),.425,.006,m,96);disk.rotation_euler.x=pi/2
    reg(disk,'mirror glass','Reflective inset with independent surface material.',WALL_BASIS,'medium',mirror,486)

def build():
    plaster=mat('Walls | warm white plaster',(.914,.909,.885),.88)
    el=mat('White architectural enamel',(.935,.93,.91))
    dark=mat('Architectural black metal',(.107,.106,.103),.33,.40)
    timber_floor()
    def panel(name,center,size,desc,first=0):
        return reg(box(name,center,size,plaster),'wall',desc,WALL_BASIS,'medium',first=first)
    panel('Art wall | continuous plaster',(-.096,2.09,1.285),(.148,12.15,2.57),'Continuous long wall supporting the art and low console.')
    reg(box('Art wall | baseboard',(-.009,2.10,.043),(.026,12.14,.086),el,.001),'skirting','White skirting along the art wall.',WALL_BASIS)
    panel('Bar wall | partition',(3.97,2.752,1.285),(2.00,.175,2.57),'Short partition behind the bar cart, terminating at the kitchen passage.',486)
    reg(box('Bar wall | baseboard',(3.97,2.847,.041),(2,.022,.082),el,.001),'skirting','Baseboard below the circular mirror.',WALL_BASIS,first=486)
    panel('Media wall | inside return',(4.835,5.455,1.285),(.18,5.34,2.57),'Outside wall behind the recessed media feature.',358)
    # End wall behind the kitchen has a real opening in line with the modeled sash.
    for i,(a,b,z,h) in enumerate([(-.125,1.24,1.285,2.57),(2.24,4.97,1.285,2.57),(1.24,2.24,.66,1.32),(1.24,2.24,2.55,.04)]):
        panel('Kitchen end wall | section '+str(i+1),((a+b)/2,-3.967,z),(b-a,.175,h),'Painted surround of the distant kitchen window.',504)
    panel('Kitchen | side wall',(4.94,-.55,1.285),(.16,6.72,2.57),'Side wall beyond the brief view into the kitchen and passage.',504)
    reg(box('Ceiling | plaster slab',(2.42,2.12,2.639),(5.19,12.44,.137),plaster),'ceiling','Flat white ceiling above the kitchen, dining and living spaces.',WALL_BASIS,'medium')
    reg(box('Dining soffit | dropped section',(.293,.47,2.467),(.657,6.315,.205),plaster,.002),'ceiling soffit','Shallow dropped bulkhead above the dining/art wall.',WALL_BASIS,'low')
    for i,(x,y) in enumerate([(2.12,2.00),(1.98,4.68),(.78,6.96),(3.56,5.19),(3.92,3.49),(2.27,-.85),(1.04,-2.63),(3.55,-2.12)]):
        trim=reg(cyl(f'Ceiling downlight {i+1} | trim',(x,y,2.562),.047,.012,el),'recessed downlight','Circular ceiling light with a thin white trim and luminous aperture.',DETAIL_BASIS,first=0 if i<5 else 504)
        reg(cyl(f'Ceiling downlight {i+1} | diffuser',(x,y,2.553),.037,.004,emit('Downlight | warm white',(.99,.967,.91),3)),'light diffuser','Recessed warm-white diffuser.',DETAIL_BASIS,parent=trim,first=0 if i<5 else 504)
    windows(plaster,dark)
    media_wall(dark)
    occasional()
