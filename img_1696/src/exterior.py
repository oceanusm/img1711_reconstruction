"""Window views reconstructed as low-confidence, editable exterior geometry.

The exterior has negligible VGGT coverage. Red SUV localization uses the same
windshield/roof area viewed through different window bays (frames 350 and 920).
The other depths follow projected outlines, roof/ground heights and perspective.
"""
import math
import random
import bpy
import common as G


def build():
    exterior_basis = ('Exterior image rays at source frames 325–375 and 820–990; '
                      'weak through-glass VGGT returns. Street elevation, depths, '
                      'and unseen sides are estimates based on apparent size and parallax.')
    def material(name,color,rough=.7,metal=0):
        m=G.mat(name,color,rough,metal)
        if m.get('exterior_daylight'):return m
        m['exterior_daylight']=True
        bs=m.node_tree.nodes.get('Principled BSDF')
        bs.inputs['Emission Color'].default_value=m.diffuse_color
        # Outdoor daylight is appreciably brighter than the room. A directional
        # sky-fill term preserves light/shade differences even behind the windows.
        nodes=m.node_tree.nodes; links=m.node_tree.links
        geometry=nodes.new('ShaderNodeNewGeometry')
        dot=nodes.new('ShaderNodeVectorMath');dot.operation='DOT_PRODUCT'
        dot.inputs[1].default_value=(-.58,-.40,.71)
        links.new(geometry.outputs['Normal'],dot.inputs[0])
        sky=nodes.new('ShaderNodeMapRange')
        sky.inputs['From Min'].default_value=-.65;sky.inputs['From Max'].default_value=.92
        sky.inputs['To Min'].default_value=.29;sky.inputs['To Max'].default_value=1.78
        links.new(dot.outputs['Value'],sky.inputs['Value']);links.new(sky.outputs['Result'],bs.inputs['Emission Strength'])
        if 'shingle' in name.lower() or 'composition roof' in name.lower():
            tile=nodes.new('ShaderNodeTexBrick');tile.offset=.50;tile.offset_frequency=2
            tile.inputs['Scale'].default_value=1.
            tile.inputs['Brick Width'].default_value=.32;tile.inputs['Row Height'].default_value=.19
            tile.inputs['Mortar Size'].default_value=.008
            c=m.diffuse_color;tile.inputs['Color1'].default_value=(*[v*.80 for v in c[:3]],1.)
            tile.inputs['Color2'].default_value=(*[v*1.17 for v in c[:3]],1.)
            tile.inputs['Mortar'].default_value=(*[v*.55 for v in c[:3]],1.)
            links.new(geometry.outputs['Position'],tile.inputs['Vector'])
            links.new(tile.outputs['Color'],bs.inputs['Base Color'])
            links.new(tile.outputs['Color'],bs.inputs['Emission Color'])
        return m
    def obj(o,kind,desc,parent=None,basis=exterior_basis):
        o=G.tag(o,kind,desc,basis,'low',parent=parent)
        if parent is not None:
            first=int(parent.get('first_observed_frame',0))
        elif 'Red curbside' in o.name:
            first=338
        elif 'Silver curbside' in o.name:
            first=948
        elif kind in ('fence','curb','verge','pavement','driveway','utility cable'):
            first=815
        elif kind=='road' or 'boundary wall' in o.name:
            first=338
        elif kind=='tree trunk':
            first=103
        else:
            first=245
        o['first_observed_frame']=first
        return o
    def block(name,c,d,m,kind,desc,parent=None,bevel=0):
        return obj(G.box('Street | '+name,c,d,m,bevel),kind,desc,parent)
    z=-6.82
    asphalt=material('Street asphalt | warm gray','#7b7978')
    curb=material('Cast curb and path','#d1ccbf')
    lawn=material('Outdoor ground cover','#9fa46d')
    plaster=material('Neighboring stucco | warm white','#e9e6dc')
    white=material('Pickets and painted timber','#efefea')
    roof=material('Exterior | gray composition roof','#83878a')
    roof2=material('Exterior | gray-brown shingle','#82776c')
    roof3=material('Exterior | dark shingle','#66696e')
    cream=material('Exterior | pale flat membrane','#dfded8')
    dark=material('Exterior | door gaps and dark metal','#524f48')
    block('Road surface',(2,15.15,z-.095),(68,12.0,.19),asphalt,'road','Visible asphalt roadway below the windows')
    block('Curb',(2,21.18,z+.035),(68,.23,.22),curb,'curb','Far road edge and raised curb')
    block('Grass verge',(2,22.12,z+.045),(68,1.8,.14),lawn,'verge','Grass strip between curb and sidewalk')
    block('Sidewalk',(2,23.69,z+.077),(68,1.40,.14),curb,'pavement','Public sidewalk across the street')
    block('House frontage ground',(2,42,z-.035),(68,36,.17),lawn,'ground','Yards and ground supporting the visible neighboring buildings')
    block('Front driveway',(0.6,25.2,z+.11),(8.15,8.2,.11),curb,'driveway','Pale concrete driveway leading to the visible garage')

    def hip(name,x0,x1,y0,y1,eave,ridge,m,parent):
        # Symmetric hips with an elongated crest; the roof has a closed soffit.
        d=min((x1-x0)*.40,(y1-y0)*.5)
        ym=(y0+y1)/2
        v=[(x0,y0,eave),(x1,y0,eave),(x1,y1,eave),(x0,y1,eave),
           (x0+d,ym,ridge),(x1-d,ym,ridge)]
        faces=[(0,1,5,4),(1,2,5),(2,3,4,5),(3,0,4),(3,2,1,0)]
        return obj(G.mesh('Street | '+name,v,faces,m),'roof',
                   'Thick hipped roof volume with continuous sloping planes',parent)
    house=block('White house across street',(-3.8,32,z+1.48),(14.8,8.5,2.96),plaster,
             'house','Single-storey white house with mixed roofs visible behind white pickets')
    hip('Left front gray roof',-11.45,-3.65,26.6,32.8,z+2.95,z+4.52,roof,house)
    hip('Main gray hip roof',-12.4,4.5,28.05,37.1,z+2.97,z+4.91,roof,house)
    block('Flat roof segment',(-.4,29.8,z+3.11),(7.3,5.6,.23),cream,'roof','Pale, near-flat part of the roof beside the gray hip',house)
    block('Rooftop vent',(-1.1,30.1,z+3.31),(.72,.43,.23),white,'vent','One low rectangular roof vent',house,.05)
    block('Garage door',(1.28,27.70,z+1.1),(4.05,.09,2.16),white,'garage door','Broad white segmented garage door',house)
    for i in range(1,5):
        block('Garage joint %d'%i,(1.28,27.646,z+.16+i*.40),(4.00,.012,.014),
              curb,'door panel joint','Horizontal garage-door panel joint',house)
    block('Timber side gate',(-2.08,27.69,z+1.03),(.88,.09,1.92),material('Vertical picket gate','#b8ab8c'),
          'gate','Narrow wood-colored side entrance adjoining garage',house)
    # The white pickets and horizontal rails form one continuous physical fence.
    verts=[];faces=[]
    for i in range(86):
        x=-14.27+i*.122; y=24.55
        w=.065;h=.98 if i%22 else 1.14;t=.055
        p=[(-w/2,0),(w/2,0),(w/2,h-.08),(0,h),(-w/2,h-.08)]
        base=len(verts)
        verts.extend((x+xx,y+side,z+.14+zz) for side in [-t/2,t/2] for xx,zz in p)
        faces.extend([tuple(base+k for k in range(4,-1,-1)),tuple(base+5+k for k in range(5))])
        faces.extend((base+j,base+(j+1)%5,base+5+(j+1)%5,base+5+j) for j in range(5))
    fence=obj(G.mesh('Street | Continuous white picket fence',verts,[face[::-1] for face in faces],white),'fence',
              'Row of pointed vertical timber pickets enclosing the frontage')
    for h in [.35,.82]:
        block('Fence rail %.2f'%h,(-9.08,24.61,z+h),(10.52,.07,.065),white,
              'fence rail','Horizontal wooden rail connecting pickets',fence)
    block('White boundary wall',(11.5,25.9,z+1.26),(15.5,.17,2.52),plaster,
          'wall','Pale boundary behind the red parked vehicle')
    distant=block('Rear roofline house',(-4.3,45,z+2.3),(10.8,6.2,4.6),plaster,'house',
                  'House behind the front roof, observed mainly by roof and upper wall')
    hip('Rear brown roof',-10.3,1.6,41.65,48.5,z+4.55,z+6.06,roof2,distant)
    house_other=block('Right background house',(7.7,42,z+1.97),(9.0,7.0,3.94),plaster,'house',
                       'Neighboring volume behind right part of frontage; outline uncertain')
    hip('Right background roof',2.9,12.5,38.2,45.8,z+3.96,z+5.46,roof3,house_other)
    side=block('Right low outbuilding',(10,32,z+1.43),(12.4,7.4,2.86),plaster,'outbuilding',
               'Pale low building and dark eave glimpsed over the white wall')
    hip('Outbuilding roof',3.6,16.4,28.1,35.9,z+2.89,z+4.01,roof3,side)

    trunkmat=material('Trunks outside','#615d46')
    foliage=[material('Exterior canopy '+str(i),c) for i,c in enumerate(
             ['#617345','#627f4c','#758951','#849759','#8c955c','#527344','#3d6844'])]
    rng=random.Random(1696)
    def tree(name,x,y,h,rad,slender=False):
        # Each physical tree is a single editable tessellated canopy with a separate trunk.
        base=z+.05
        trunk=obj(G.cyl('Street | '+name+' trunk',(x,y,base+h*.38),max(.09,h*.025),h*.76,trunkmat,12),
                   'tree trunk','Woody trunk below the exterior foliage')
        vertices=[];faces=[];colors=[]
        for j in range(38 if not slender else 26):
            az=rng.uniform(0,2*math.pi)
            if slender:
                fraction=rng.uniform(.27,.95)
                offset=(1-fraction)*rad*rng.uniform(.30,1.0)
                cx=x+math.cos(az)*offset;cy=y+math.sin(az)*offset
                cz=base+h*fraction
                radius=rad*(1-fraction)**.52*rng.uniform(.6,.98)
                zoom=(radius,radius,radius*1.9)
            else:
                fraction=rng.uniform(-.65,.90)
                offset=rad*math.sqrt(1-fraction*fraction)*rng.uniform(.45,.87)
                cx=x+math.cos(az)*offset;cy=y+math.sin(az)*offset
                cz=base+h*.73+fraction*rad*.65
                radius=rad*rng.uniform(.32,.55)
                zoom=(radius,radius*.92,radius*.87)
            index=len(vertices); rows=6; slices=9
            for r in range(rows+1):
                ph=math.pi*r/rows
                for s in range(slices):
                    a=2*math.pi*s/slices
                    n=rng.uniform(.85,1.14)
                    vertices.append((cx+zoom[0]*n*math.sin(ph)*math.cos(a),cy+zoom[1]*n*math.sin(ph)*math.sin(a),cz+zoom[2]*n*math.cos(ph)))
            tone=rng.randrange(len(foliage))
            for r in range(rows):
                for s in range(slices):
                    v=index+r*slices+s; q=index+r*slices+(s+1)%slices
                    faces.append((v,v+slices,q+slices,q));colors.append(tone if rng.random()>.14 else rng.randrange(len(foliage)))
        o=G.mesh('Street | '+name+' crown',vertices,faces,foliage[0])
        for m in foliage[1:]:o.data.materials.append(m)
        for face,color in zip(o.data.polygons,colors):
            face.material_index=color;face.use_smooth=True
        obj(o,'tree foliage','Three-dimensional foliage envelope and leaf clusters; photographic silhouette approximate',trunk)
    for args in [('Far spreading tree',-16,41,8.5,3.9),('Garden broadleaf',-9.5,40,6.7,2.7),
                 ('Background cypress',-9.6,56,12.,2.0,True),('Tall narrow tree',-7.2,56,12.8,1.5,True),
                 ('Distant broadleaf',-4.4,58,11.3,4.2),('Right green canopy',4.8,54,9.8,3.8),
                 ('Front garden shrub',-11.7,25.8,1.6,1.0),('Right verge shrub',15.8,25.3,2.25,1.5)]:
        tree(*args)
    for i,(xx,hh,rr) in enumerate([(-13.5,.92,.59),(-12.8,.78,.50),(-10.6,.69,.48),
                  (-9.6,1.08,.75),(-8.6,.55,.35),(-6.1,.78,.63),(-4.5,.64,.42)]):
        tree('Low garden foliage '+str(i+1),xx,25.15,hh,rr)
    # Small objects on the far verge were individually identifiable near frame 840.
    stumpmat=material('Weathered stump bark','#817059')
    st=obj(G.cyl('Street | Low roadside stump',(-7.13,22.48,z+.32),.29,.41,stumpmat,11),
           'tree stump','Cut trunk visible between two leaning objects on the grass verge')
    st['first_observed_frame']=819
    panel=block('Light panel by stump',(-7.48,22.52,z+.62),(.59,.065,1.13),white,
                'loose panel','Pale object leaning against the stump; underside uncertain')
    panel.rotation_euler[1]=.59;panel['first_observed_frame']=819
    barrel=obj(G.cyl('Street | Pale rolled curbside object',(-6.91,22.52,z+.61),.205,1.15,
               material('Pale tan curbside covering','#c2b493'),24),'rolled object',
               'Long pale rounded item leaning against the roadside stump; identity uncertain')
    barrel.rotation_euler[1]=-.36;barrel['first_observed_frame']=819
    steel=material('Street chair | green-gray metal','#46666a',.5,.24)
    seat=block('Small outdoor chair seat',(-8.00,22.59,z+.50),(.43,.40,.055),steel,
               'chair','Dark blue-green chair with four thin legs and a framed back',bevel=.022)
    seat['first_observed_frame']=819
    for k,(sx,sy) in enumerate([(-1,-1),(-1,1),(1,-1),(1,1)]):
        obj(G.rod('Street | Outdoor chair leg '+str(k),(-8+sx*.23,22.59+sy*.22,z+.12),
            (-8+sx*.18,22.59+sy*.15,z+.48),.019,steel),'chair leg','Thin metal support',seat)
    obj(G.tube('Street | Outdoor chair back frame',[(-8.20,22.75,z+.5),(-8.20,22.79,z+.85),
              (-7.80,22.79,z+.85),(-7.80,22.75,z+.5)],.022,steel),
              'chair back','Open metal backrest frame',seat)
    stone=material('Limestone verge rocks','#b6aa8e')
    for k,(sx,sy,d) in enumerate([(-8.93,22.47,.25),(-8.50,22.56,.22),(-5.38,22.62,.33),
                                  (-4.57,22.75,.27),(-3.42,22.89,.30)]):
        rock=obj(G.uv('Street | Verge rock '+str(k+1),(sx,sy,z+.13+d*.23),
                      (d,.72*d,.60*d),stone),'rock','Individual pale rock on the verge')
        rock['first_observed_frame']=819
    cable=material('Utility cable dark grey','#514f4a')
    points=[(-22+i*1.5,18.7+i*.20,-1.32-.35*math.sin(math.pi*i/31)) for i in range(32)]
    obj(G.tube('Street | Aerial utility line',points,.016,cable),'utility cable',
        'Thin dark utility line visible diagonally across the opposite roofs')
    _vehicle('Red curbside SUV',7.37,20.40,z+.02,1.0,'#ad373a',material,obj,exterior_basis)
    _vehicle('Silver curbside car',14.10,20.5,z+.02,.92,'#a7adad',material,obj,exterior_basis)


def _vehicle(name,x,y,z,size,paint,material,obj,basis):
    car_basis=(('Cross-window triangulation of the side windshield area at frames 350 and '
               '920 gives approximately [6.9, 19.7, -5.0] m; contour and wheel positions are '
               'estimated against frames 345–370. ') if name.startswith('Red') else
               'Only the light-colored roof and hood appear behind the chair at frames 970–990. ') + basis
    col=material(name+' metallic finish',paint,.33,.35)
    glass=material('Exterior car tinted windows','#495660',.22,.26)
    rubber=material('Automotive tire rubber','#343538',.85)
    hardware=material('Wheel aluminum','#aaafb0',.32,.7)
    trim=material('Vehicle black trim','#3b3e40',.48,.2)
    tail=material('Red tail lamp','#bb433a',.3)
    light=material('Headlamp diffuser','#e9e6d7',.23)
    def world(a,b,c):
        # The silver vehicle shows a lower roofline than the red SUV.
        height=.99+.74*(c-.99) if name.startswith('Silver') and c>.99 else c
        return (x+a*size,y+b*size,z+height*size)
    def component(o,typ,desc,parent):
        return obj(o,typ,desc,parent,basis=car_basis)
    def hull(label,stations,mat,parent,kind):
        # Cross-sections have shoulders and narrowed lower/upper edges.
        verts=[];faces=[]
        for a,half,low,high in stations:
            profile=[(-half*.91,low),(-half,low+.09),(-half,high-.07),(-half*.95,high),
                     (half*.95,high),(half,high-.07),(half,low+.09),(half*.91,low)]
            verts.extend(world(a,b,c) for b,c in profile)
        faces.append(tuple(range(7,-1,-1)))
        for j in range(len(stations)-1):
            for k in range(8):
                faces.append((j*8+k,j*8+(k+1)%8,(j+1)*8+(k+1)%8,(j+1)*8+k))
        faces.append(tuple(range((len(stations)-1)*8,len(stations)*8)))
        o=G.mesh('Street | '+name+' '+label,verts,[face[::-1] for face in faces],mat)
        mod=o.modifiers.new('Panel edge rounding','BEVEL');mod.width=.031*size;mod.segments=2
        return component(o,kind,label,parent)
    body=hull('lower body',[(-2.35,.70,.45,.77),(-2.08,.87,.43,.95),(-1.30,.92,.44,1.03),
          (-.68,.94,.44,1.05),(.75,.94,.44,1.07),(1.91,.90,.48,1.12),(2.12,.79,.51,1.07)],
          col,None,'parked vehicle')
    cabin=hull('glazed passenger cabin',[(-1.26,.84,1.00,1.08),(-.65,.74,1.01,1.57),
          (-.33,.73,1.01,1.66),(.96,.75,1.03,1.65),(1.44,.77,1.05,1.53),(1.99,.82,1.06,1.13)],glass,body,'window assembly')
    roof=hull('painted roof',[(-.68,.68,1.54,1.63),(-.27,.73,1.60,1.71),(.95,.75,1.58,1.70),
          (1.43,.73,1.47,1.58)],col,body,'roof')
    for sign in (-1,1):
        for a1,a2 in [(-1.2,-.62),(.16,.18),(1.88,1.39)]:
            o=G.rod('Street | '+name+' window pillar '+str(sign)+' '+str(a1),
                    world(a1,.851*sign,1.04),world(a2,.738*sign,1.57),.033*size,trim if a1==.16 else col)
            component(o,'window pillar','Pillar separating side windows',body)
        for a in [-.4,.65]:
            component(G.box('Street | '+name+' door handle '+str(sign)+' '+str(a),world(a,.948*sign,.98),
                         (.16*size,.035*size,.04*size),hardware,.012*size),'door handle','Exterior door pull',body)
        for a in [-1.44,1.30]:
            for typ,offset,r,th,mat in [('tire',.863,.332,.230,rubber),('rim',.983,.218,.030,hardware),
                                        ('hubcap',1.002,.103,.014,trim)]:
                o=G.cyl('Street | '+name+' '+typ+' '+str(sign)+' '+str(a),world(a,offset*sign,.34),r*size,th*size,mat,28)
                o.rotation_euler[0]=math.pi/2
                component(o,typ,'Road wheel '+typ,body)
        component(G.uv('Street | '+name+' mirror '+str(sign),world(-.87,sign*.991,1.18),(.20*size,.18*size,.12*size),col),
                  'wing mirror','Door-mounted side mirror',body)
        for end,mat,typ in [(-2.23,light,'headlamp'),(2.04,tail,'taillight')]:
            o=G.uv('Street | '+name+' '+typ+' '+str(sign),world(end,sign*.65,.85),
                  (.17*size,.35*size,.16*size),mat)
            component(o,typ,'Corner light assembly',body)
    for end in [-2.25,2.04]:
        o=G.box('Street | '+name+' lower bumper '+str(end),world(end,0,.53),
                (.16*size,1.48*size,.12*size),trim,.046*size)
        component(o,'bumper trim','Dark lower bumper insert',body)
    return body
