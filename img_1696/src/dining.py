"""Dining group and kitchen. Metres, right-handed fitted room axes.

Table, frame, art and seating use frame 0 and reverse views 530/539. The
kitchen is largely outside the retained cloud and was constructed from
perspective intersections and standard cabinetry dimensions.
"""
import math
from pathlib import Path
import numpy as np
import bpy
from mathutils import Vector
from common import box, cyl, uv, rod, tube, mesh, mat, tag

TABLE_EVIDENCE = "Frames 0 and 530. Per-frame VGGT support on the tabletop and its edges, checked by intersecting corner rays with z=0.75 m. Global scale assumes this tabletop height."
CHAIR_EVIDENCE = "Six identifiable chairs in frames 0 and 539. Seat and backrest silhouettes intersected with estimated .45 and .77 m horizontal planes; obscured joinery inferred from matching chairs."
KITCHEN_EVIDENCE = "Brief reverse pass in frames 500–540. Perspective and floor contacts are constrained by the reconstructed camera. Hidden depth and standard built-in proportions are estimates; VGGT provides little support in the distant kitchen."
ART_EVIDENCE = "All four outside edges viewed in frame 539 and checked at the fitted wall plane. Depth and frame profile estimated."
SUPPORT = "Rigid attachment and visual proportions in frames 0, 66 and 539; concealed connections inferred."

class Assembly:
    """A single editable mesh for the continuous structure of one article."""
    def __init__(self):
        self.v=[]; self.f=[]
    def rings(self, rings):
        base=len(self.v); n=len(rings[0])
        for r in rings:self.v.extend(r)
        self.f.append(tuple(base+j for j in reversed(range(n))))
        for i in range(len(rings)-1):
            for j in range(n):
                a=base+i*n+j; b=base+i*n+(j+1)%n
                self.f.append((a,b,b+n,a+n))
        self.f.append(tuple(base+(len(rings)-1)*n+j for j in range(n)))
    def cuboid(self,c,d):
        a,b,z=[i/2 for i in d]
        self.rings([[(c[0]+x,c[1]+y,c[2]+h) for x,y in [(-a,-b),(a,-b),(a,b),(-a,b)]] for h in [-z,z]])
    def sweep(self,ps,radii,n=14,flat=1):
        pp=np.array(ps,float); rings=[]
        for i,p in enumerate(pp):
            tangent=pp[min(i+1,len(pp)-1)]-pp[max(i-1,0)]
            tangent/=max(np.linalg.norm(tangent),1e-9)
            helper=np.array([0,1,0]) if abs(tangent[1])<.95 else np.array([1,0,0])
            a=np.cross(helper,tangent);a/=np.linalg.norm(a); b=np.cross(tangent,a)
            rad=radii[i] if hasattr(radii,'__len__') else radii
            rings.append([p+rad*(math.cos(2*math.pi*j/n)*a+flat*math.sin(2*math.pi*j/n)*b) for j in range(n)])
        self.rings(rings)
    def create(self,name,m,smooth=False):
        obj=mesh(name,self.v,self.f,m)
        if smooth:
            for p in obj.data.polygons:p.use_smooth=True
        return obj

def bevel(o,r=.003):
    q=o.modifiers.new('Finished edges','BEVEL');q.width=r;q.segments=3
    o.modifiers.new('Face normals','WEIGHTED_NORMAL')
    return o

def surface(name,vertices,faces,m,thickness=.002):
    obj=mesh(name,vertices,faces,m)
    mod=obj.modifiers.new('Physical thickness','SOLIDIFY');mod.thickness=thickness
    for p in obj.data.polygons:p.use_smooth=True
    return obj

def grid_surface(name,nu,nv,point,m,thickness=.002):
    vs=[point(i/(nu-1),j/(nv-1)) for j in range(nv) for i in range(nu)]
    fs=[(j*nu+i,j*nu+i+1,(j+1)*nu+i+1,(j+1)*nu+i) for j in range(nv-1) for i in range(nu-1)]
    return surface(name,vs,fs,m,thickness)

def timber_material(name,color):
    m=mat(name,color,.54)
    n=m.node_tree.nodes;ln=m.node_tree.links
    tc=n.new('ShaderNodeTexCoord'); coord=n.new('ShaderNodeVectorMath');coord.operation='MULTIPLY'
    coord.inputs[1].default_value=(20,2.1,10);ln.new(tc.outputs['Generated'],coord.inputs[0])
    noise=n.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=4;noise.inputs['Detail'].default_value=2.8;noise.inputs['Roughness'].default_value=.7
    ln.new(coord.outputs['Vector'],noise.inputs['Vector'])
    ramp=n.new('ShaderNodeValToRGB');rgb=m.diffuse_color
    for e,k in zip(ramp.color_ramp.elements,[.79,1.13]):e.color=(*[min(v*k,1) for v in rgb[:3]],1)
    ln.new(noise.outputs['Fac'],ramp.inputs[0]);ln.new(ramp.outputs[0],n.get('Principled BSDF').inputs['Base Color'])
    return m

def seat(name):
    """Round-edged trapezoidal seat, pierced Y splat and steam-bent arm bow."""
    s=Assembly()
    for sign_x in [-1,1]:
        s.sweep([(sign_x*.204,-.183,.001),(sign_x*.199,-.165,.26),(sign_x*.193,-.16,.435),
                 (sign_x*.204,-.155,.59),(sign_x*.24,-.093,.713)], [.012,.016,.019,.017,.014])
        s.sweep([(sign_x*.211,.187,.001),(sign_x*.203,.175,.26),(sign_x*.191,.157,.446),
                 (sign_x*.172,.151,.54)], [.012,.016,.019,.015])
        s.sweep([(sign_x*.199,-.174,.20),(sign_x*.204,.174,.20)],.012)
        s.sweep([(sign_x*.195,-.16,.433),(sign_x*.194,.167,.433)],.019)
    for y,z in [(-.166,.184),(.174,.185),(-.166,.436),(.168,.436)]:
        s.sweep([(-.198,y,z),(.198,y,z)],.012 if z<.3 else .020)
    angles=np.linspace(math.radians(-28),math.radians(208),40)
    arm=[(.255*math.cos(a),.052+.202*math.sin(a),.707+.058*math.sin(a)) for a in angles]
    s.sweep(arm,[.014+.004*max(0,math.sin(a)) for a in angles],18)
    for b in [-1,1]:
        s.sweep([(0,.174,.435),(b*.012,.177,.52),(b*.032,.205,.61),
                 (b*.080,.230,.69),(b*.134,.231,.750)],
                [.027,.025,.020,.018,.014],16,.42)
    return s

def dining_group():
    oak=timber_material('Dining · warm oak', '#C6AB7F')
    topwood=timber_material('Dining top · blond oak','#D6C69B')
    black=mat('Dining · charcoal powder coat','#242329',.34,.66)
    linen=mat('Dining · natural flax','#DFD9C6',.91)
    cord=mat('Dining · papercord','#D9CCAA',.87)
    gray=mat('Dining · gray woven garment','#85858C',.95)
    top=tag(box('Dining table · oak top',(1.184,2.44,.7315),(.862,1.738,.037),topwood,.004),
            'dining table','Rectangular pale oak top above black end trestles.',TABLE_EVIDENCE,'medium')
    for side,y in [('kitchen',1.681),('living',3.217)]:
        parts=Assembly()
        for x in [.814,1.56]:parts.cuboid((x,y,.35),(.039,.044,.70))
        parts.cuboid((1.187,y,.019),(.78,.044,.038));parts.cuboid((1.187,y,.710),(.78,.044,.017))
        tag(bevel(parts.create('Dining table · '+side+' steel loop',black),.002),
            'table trestle','Rectangular steel loop with open center and floor stretcher.',TABLE_EVIDENCE,'medium',top)
    def runner_point(u,v):
        y=1.559+1.768*v; side=2*u-1
        x=1.21-.024*(y-2.44)+side*(.175+.003*math.sin(66*v)+.0017*math.sin(124*v))
        z=.754+.0006*math.sin(230*v+2.5*u)
        if y<1.57:z-=min(.008,(1.57-y)*.75)
        if y>3.309:z-=min(.015,(y-3.309)*.95)
        return x,y,z
    runner=tag(grid_surface('Dining table · ivory runner',9,100,runner_point,linen,.0015),
               'textile','Ivory lengthwise woven runner with gently irregular selvages.',SUPPORT,'medium',relations={'on':top.name})
    # The open boat-shaped bowl is a closed shell, including the inside and rolled rim.
    v=[];f=[];N=38;W=25
    for inner in [False,True]:
        for i in range(N):
            t=-1+2*i/(N-1)
            w=.094*max(.012,(1-t*t)**.67)
            bottom=.762+.034*t*t+.067*abs(t)**4-.012*t
            for j in range(W):
                u=-1+2*j/(W-1)
                z=bottom+(.040+.030*abs(t))*abs(u)**1.65
                x=1.24-.028*t+u*w
                if inner:z+=.007+(.003*(1-u*u));x-=.005*u*max(.05,1-abs(t)**3)
                v.append((x,2.42+.46*t,z))
    for k in range(2):
        offset=k*N*W
        for i in range(N-1):
            for j in range(W-1):
                a=offset+i*W+j;q=(a,a+1,a+1+W,a+W)
                f.append(q if k else q[::-1])
    border=list(range(W))+[i*W+W-1 for i in range(1,N)]+list(range(N*W-2,(N-1)*W-1,-1))+[i*W for i in range(N-2,0,-1)]
    for a,b in zip(border,border[1:]+border[:1]):f.append((a,b,b+N*W,a+N*W))
    bowl=mesh('Dining table · carved canoe bowl',v,f,timber_material('Carved bowl · weathered wood','#B59F80'))
    for p in bowl.data.polygons:p.use_smooth=True
    tag(bowl,'decorative vessel','Long, shallow, irregular wood bowl with visibly open concavity and raised ends.',
        'Endpoint triangulation in both viewing directions; inner depth and wall thickness estimated.','medium',relations={'on':runner.name})
    loose_basis='Near views 0 and 65 show two pale pieces, a pink piece and a darker slim item in the open bowl. Sizes are estimated from the bowl opening; functions uncertain.'
    for label,y,ww,ll,hh,color in [
        ('ivory oblong piece',2.22,.063,.044,.024,'#D9D9D3'),
        ('pale pink piece',2.45,.039,.056,.020,'#BBA6A7'),
        ('white flat piece',2.70,.042,.075,.013,'#E8E7DE'),
        ('dark narrow piece',2.36,.019,.072,.014,'#50493F')]:
        t=(y-2.42)/.46
        z=.762+.034*t*t+.067*abs(t)**4-.012*t+.010
        item=box('Dining bowl · '+label,(1.24-.028*t,y,z+hh/2),(ww,ll,hh),mat('Dining · '+label,color,.57),.006)
        item.rotation_euler.x=math.atan((.068*t+.268*t**3-.012)/.46)
        tag(item,'loose bowl contents',label.capitalize()+' resting inside the carved bowl.',loose_basis,'low',relations={'inside':bowl.name})
    tag(box('Dining table · aisle corner guard',(1.602,1.580,.733),(.032,.033,.043),black,.003),
        'corner protector','Small dark cap at the kitchen-facing aisle corner.',SUPPORT,'low',top)
    # The supported long edge runs 4–5 cm toward the wall over the table's length.
    top.rotation_euler.z=.024
    # Small pale item at the table edge appears as a slip/coaster in both directions.
    tag(box('Dining table · small white card',(1.47,3.23,.7515),(.061,.047,.002),mat('Dining · ivory paper','#ECEAE2'),.001),
        'paper item','Small pale flat rectangle near the living-room end. Exact function unresolved.',SUPPORT,'low',relations={'on':top.name})
    centers=[('kitchen end',1.20,1.43,math.pi),('living end',1.21,3.55,0),
             ('wall kitchen',.645,2.08,math.pi/2),('wall living',.651,2.89,math.pi/2),
             ('aisle kitchen',1.759,2.04,-math.pi/2),('aisle living',1.756,2.88,-math.pi/2)]
    chair_mesh=seat('chair')
    tacks={}
    for label,x,y,a in centers:
        frame=chair_mesh.create('Dining chair · '+label+' · timber frame',oak,True)
        frame.location=(x,y,0);frame.rotation_euler.z=a
        tag(frame,'dining chair','Bentwood wishbone dining chair with wraparound arm bow, forked splat and four tapered legs.',CHAIR_EVIDENCE,'medium')
        def seatpoint(u,v):
            xx=(u-.5)*(.419-.023*v);yy=(v-.5)*.391
            z=.456-.011*(1-(2*u-1)**2)*(1-(2*v-1)**2)+.0006*math.cos(200*u)
            return (xx,yy,z)
        seat_obj=grid_surface('Dining chair · '+label+' · cord seat',27,19,seatpoint,cord,.020)
        seat_obj.location=(x,y,0);seat_obj.rotation_euler.z=a
        tag(seat_obj,'chair seat','Pale woven seat over the timber rails.',CHAIR_EVIDENCE,'medium',frame)
        tacks[label]=seat_obj
    def textile(u,v):
        x=1.76+(u-.5)*.460+.025*math.sin(6*v);y=2.06+(v-.5)*.440+.014*math.cos(8*u)
        z=.485+.123*math.exp(-((u-.62)/.30)**2)+.035*math.sin(17*v+9*u)+.015*math.sin(27*u-6*v)
        return x,y,z
    tag(grid_surface('Dining chair · folded gray textile',29,39,textile,gray,.011),
        'textile','Loose gray garment or folded fabric resting on the aisle chair nearest the kitchen.',
        'First pass and reverse view silhouettes; interior folds underdetermined.','low',relations={'on':tacks['aisle kitchen'].name})
    pendant=tag(uv('Dining pendant · wooden shade',(1.146,2.344,1.846),(.133,1.247,.063),topwood),
        'pendant light','Narrow suspended tapered, lens-shaped blond timber light.',
        'Opposite-view triangulation of both ends in frames 0 and 539; cross-section estimated.','medium')
    cap=tag(box('Dining pendant · ceiling canopy',(1.146,2.344,2.541),(.100,.38,.040),oak,.016),
            'canopy','Suspension mounting canopy at the ceiling.',SUPPORT,'low',pendant)
    for k,y in enumerate([2.155,2.533],1):
        tag(rod('Dining pendant · suspension %d'%k,(1.146,y,1.873),(1.146,y,2.524),.0017,mat('Dining · light cord','#D6D2CC')),
            'suspension cable','Fine vertical suspension line.',SUPPORT,'low',pendant)
    tag(uv('Dining pendant · diffuser',(1.146,2.344,1.820),(.074,1.04,.018),mat('Dining · diffuser','#EEE5C8')),
        'diffuser','Recessed pale underside diffuser.',SUPPORT,'low',pendant)

def dining_art():
    x=.026;cy=2.35;cz=1.756;w=.99;h=.644;b=.014
    oak=mat('Dining print · oak surround','#AF946F')
    r=Assembly()
    for y in [cy-w/2+b/2,cy+w/2-b/2]:r.cuboid((x,y,cz),(.052,b,h))
    for z in [cz-h/2+b/2,cz+h/2-b/2]:r.cuboid((x,cy,z),(.052,w-2*b,b))
    frame=tag(bevel(r.create('Dining wall print · light wood frame',oak),.0015),'picture frame',
        'Narrow timber frame around a landscape-format dark abstract print.',ART_EVIDENCE,'medium')
    # Closed artwork panel. UVs use observed pigment on the front face only.
    paper=mat('Dining wall print · warm paper','#D2CFBE')
    pigment=mat('Dining wall print · rectified pigment','#DEDACE')
    image=bpy.data.images.load(str(Path(__file__).resolve().parent/'assets/dining_print.jpg'),check_existing=True)
    image.pack()
    nd=pigment.node_tree.nodes.new('ShaderNodeTexImage');nd.image=image
    pigment.node_tree.nodes.active=nd
    pigment.node_tree.links.new(nd.outputs['Color'],pigment.node_tree.nodes.get('Principled BSDF').inputs['Base Color'])
    panel=box('Dining wall print · painted board',(.038,cy,cz),(.011,w-2*b,h-2*b),paper)
    panel.data.materials.append(pigment)
    uvmap=panel.data.uv_layers.new(name='Artwork coordinates')
    for poly in panel.data.polygons:
        if poly.normal.x>.9:
            poly.material_index=1
        for li in poly.loop_indices:
            v=panel.data.vertices[panel.data.loops[li].vertex_index].co
            uvmap.data[li].uv=(.5+v.y/(w-2*b),.5+v.z/(h-2*b))
    tag(panel,'framed artwork','Interconnected indigo loops with fine linework on a cream panel. Pigment rectified from the artwork alone.',ART_EVIDENCE,'medium',frame)

def kitchen_group():
    cabinet=mat('Kitchen · warm white lacquer','#E7E5DE',.31)
    counter=mat('Kitchen · white quartz','#E9E8E1',.30)
    steel=mat('Kitchen · brushed stainless','#A6A9A9',.31,.68)
    dark=mat('Kitchen · black appliance enamel','#26272A',.27,.32)
    glass=mat('Kitchen · smoked glazing','#41474B',.18,.44)
    oak=timber_material('Kitchen · peninsula oak','#AC9075')
    blackmetal=mat('Kitchen · dark fittings','#36383B',.41,.65)
    def kit(o,kind,description,parent=None,relation=None):
        o=tag(o,kind,description,KITCHEN_EVIDENCE,'low',parent,relation)
        o['first_observed_frame']=494
        return o
    def block(n,c,d,m,ty,descr,p=None,r=.003):
        return kit(box('Kitchen · '+n,c,d,m,r),ty,descr,p)
    # Front at +Y, joins aligning with the back countertop.
    carc=block('back base units',(1.172,-3.57,.47),(2.16,.576,.79),cabinet,'fitted cabinet',
        'White sink cabinet row below the rear kitchen window.')
    pl=block('back cabinet plinth',(1.174,-3.59,.050),(2.13,.48,.095),dark,'cabinet plinth','Recessed shadow plinth.',carc)
    for i,(x,w) in enumerate([(.36,.50),(.88,.51),(1.43,.55),(1.985,.54)],1):
        face=block('back cabinet front %02d'%i,(x,-3.271,.47),(w-.009,.027,.757),cabinet,
                   'cabinet door','Flat lacquered base-unit front with a narrow reveal.',carc)
        block('back cabinet pull %02d'%i,(x,-3.250,.815),(.235,.019,.010),blackmetal,'pull','Low-profile upper-edge pull.',face)
    # Real through-opening in the slab and a sloped, recessed basin below its rim.
    x0,x1,y0,y1=.074,2.265,-3.864,-3.231
    sx0,sx1,sy0,sy1=1.41,1.938,-3.689,-3.325
    slab=Assembly()
    for xa,xb,ya,yb in [(x0,sx0,y0,y1),(sx1,x1,y0,y1),(sx0,sx1,y0,sy0),(sx0,sx1,sy1,y1)]:
        slab.cuboid(((xa+xb)/2,(ya+yb)/2,.883),(xb-xa,yb-ya,.040))
    kt=kit(bevel(slab.create('Kitchen · back quartz worktop',counter),.002),'countertop','Quartz worktop with an opening for the sink basin.',carc)
    basin=Assembly()
    def rectangle(xa,xb,ya,yb,z):
        return [(xa,ya,z),(xb,ya,z),(xb,yb,z),(xa,yb,z)]
    # Closed base, open upper lip. Ring geometry joins to a bottom polygon.
    outer=rectangle(sx0-.013,sx1+.013,sy0-.013,sy1+.013,.904)
    inn=rectangle(sx0+.005,sx1-.005,sy0+.005,sy1-.005,.905)
    base=rectangle(sx0+.043,sx1-.043,sy0+.043,sy1-.043,.772)
    sink=kit(mesh('Kitchen · inset metal sink',outer+inn+base,[(i,(i+1)%4,4+(i+1)%4,4+i) for i in range(4)]+[(4+i,4+(i+1)%4,8+(i+1)%4,8+i) for i in range(4)]+[(8,9,10,11)],steel),
             'sink','Open inset sink inferred beneath the rear window; exact rim profile uncertain.',kt)
    inside=sink.modifiers.new('Pressed steel shell','SOLIDIFY');inside.thickness=.001
    kit(tube('Kitchen · arched faucet',[(1.68,-3.747,.905),(1.68,-3.747,1.075),(1.68,-3.74,1.115),(1.68,-3.705,1.14),(1.68,-3.652,1.14),(1.68,-3.621,1.111),(1.68,-3.621,1.087)],.009,steel),
        'tap','Arched metal tap at the back basin.',kt)
    # Dark surround, bright pane, sill and modest reveals for the kitchen window.
    win=block('rear window frame',(1.737,-3.834,1.919),(1.014,.05,1.178),mat('Kitchen · window reveal','#C4C3BC'),'window frame',
              'Trim surrounding the bright rear kitchen window.')
    pane=block('rear window glass',(1.737,-3.800,1.919),(.95,.012,1.105),mat('Kitchen · diffuse daylight glass','#D3DEE2',.24),
               'window glazing','Backlit glass, outdoor detail unresolved.',win,.001)
    block('window sill',(1.737,-3.817,1.327),(1.045,.12,.025),cabinet,'sill','Pale window stool.',win)
    ref=block('refrigerator housing',(2.714,-3.57,.948),(.886,.677,1.898),dark,'refrigerator','Tall stainless refrigerator with lower freezer section.')
    upper=block('refrigerator door',(2.714,-3.202,1.175),(.871,.061,1.39),steel,'refrigerator door','Stainless upper door; interior concealed.',ref,.009)
    lower=block('freezer drawer',(2.714,-3.203,.255),(.871,.06,.425),steel,'freezer drawer','Lower freezer drawer.',ref,.009)
    block('freezer drawer handle',(2.714,-3.153,.438),(.674,.039,.023),steel,'appliance pull','Horizontal drawer grip.',lower)
    block('refrigerator dispenser',(2.70,-3.166,1.298),(.25,.015,.22),dark,'appliance fitting','Dark inset door feature; internal dispenser details uncertain.',upper)
    block('refrigerator upper note',(2.69,-3.156,1.61),(.223,.004,.118),mat('Kitchen · note','#D9D8CF'),'paper item',
          'Pale paper rectangle on the refrigerator.',upper,.0004)
    block('above refrigerator cupboard',(2.714,-3.596,2.175),(.906,.47,.49),cabinet,'wall cabinet','Storage above refrigerator.')
    # Return cabinets along x=0. Integrated microwave and cooking appliance.
    ret=block('side base units',(.307,-2.015,.460),(.572,2.45,.79),cabinet,'fitted cabinet','Return side of the fitted kitchen.')
    block('side countertop',(.321,-2.015,.885),(.635,2.47,.04),counter,'countertop','Pale fitted worktop.',ret)
    for i,y in enumerate([-2.94,-2.35,-1.08],1):
        block('side base front %d'%i,(.61,y,.463),(.027,.53,.745),cabinet,'cabinet door','Flat front divided by narrow seam.',ret)
    cooker=block('stove',(.357,-1.73,.455),(.575,.606,.84),dark,'cooking appliance','Dark oven and cooktop below a microwave.')
    block('oven door',(.659,-1.73,.482),(.028,.55,.488),glass,'oven door','Dark glazing of oven front.',cooker)
    block('oven handle',(.692,-1.73,.740),(.044,.484,.024),steel,'appliance pull','Horizontal oven pull.',cooker)
    block('cooktop',(.367,-1.73,.914),(.526,.598,.018),dark,'cooktop','Dark flat cooktop.',cooker)
    for i,y in enumerate([-3.03,-2.54,-1.155,-.708],1):
        block('side wall cupboard %d'%i,(.223,y,1.997),(.397,.450,.702),cabinet,'wall cabinet','White upper cupboard on the kitchen return.')
    microwave=block('over-range microwave',(.252,-1.77,1.765),(.431,.615,.386),dark,'microwave oven','Dark over-range microwave in the upper cabinet row.')
    block('microwave door',(.469,-1.81,1.766),(.018,.438,.295),glass,'microwave door','Window in appliance door.',microwave)
    block('microwave handle',(.490,-1.566,1.765),(.037,.018,.233),steel,'appliance pull','Vertical door handle.',microwave)
    for i,(x,z) in enumerate([(.70,1.31),(.70,1.62)],1):
        block('back open shelf %d'%i,(x,-3.68,z),(.86,.31,.031),oak,'wall shelf','Short open display shelf beside the kitchen window.')
    # Breakfast peninsula meets the long wall at x=0. Front in +Y has fluted wood.
    island=block('peninsula cabinet',(.878,.073,.446),(1.718,1.060,.852),oak,'peninsula cabinet','Oak peninsula with prominent vertical battens facing the dining table.')
    ribs=Assembly()
    for i in range(42):
        ribs.cuboid((.052+i*.0410,.609,.449),(.021,.023,.840))
    kit(bevel(ribs.create('Kitchen · fluted peninsula face',oak),.002),'cabinet fascia','Repetitive oak battens with shadowed spaces between.',island)
    itop=block('peninsula quartz slab',(.898,.081,.895),(1.796,1.162,.038),counter,'countertop','White peninsula top with slight overhang.',island)
    itop['measurement_basis']='Dense early PLY corner returns at 0.912–0.918 m and frame 0 corner rays at x=1.784, y=0.661. Rear depth uses kitchen reverse view.'
    end=block('peninsula waterfall end',(1.775,.080,.438),(.038,1.160,.876),counter,'stone facing',
              'Pale waterfall end panel visible in the opening view.',island)
    for early in [island,itop,end]:early['first_observed_frame']=0
    # Small objects resolved in silhouette in the reverse view, deliberately simple.
    blue=mat('Kitchen · cobalt glass','#433E8F',.28,.08);ceramic=mat('Kitchen · pale ceramic','#DBD7CA')
    def vessel(name,x,y,z,r,h,m,support,kind):
        return kit(cyl('Kitchen · '+name,(x,y,z+h/2),r,h,m,28),kind,'Countertop vessel with silhouette estimated from the distant view.',relation={'on':support.name})
    bottle=vessel('cobalt bottle',1.653,-.005,.914,.033,.207,blue,itop,'bottle')
    kit(cyl('Kitchen · cobalt bottle neck',(1.653,-.005,1.1365),.018,.031,blue), 'bottle neck','Narrow cylindrical neck.',bottle)
    vessel('white tumbler',1.343,-.05,.914,.044,.155,ceramic,itop,'tumbler')
    vessel('short dark blue container',.975,-.035,.914,.067,.072,blue,itop,'vessel')
    vessel('light canister',.650,-.073,.914,.045,.124,ceramic,itop,'canister')
    vessel('green storage jar',.209,-.055,.914,.049,.115,mat('Kitchen · bottle green','#5F745C'),itop,'jar')
    pot=vessel('small tree pot',.450,-.260,.914,.091,.165,dark,itop,'plant pot')
    leaf=mat('Kitchen · tree foliage','#526B45',.83)
    branches=Assembly()
    branches.sweep([(.45,-.26,1.066),(.445,-.269,1.29),(.458,-.265,1.48)],.008)
    rng=np.random.default_rng(3621)
    lv=[];lf=[]
    for j in range(10):
        ang=j*2.40;z=1.18+(j%5)*.049
        end=np.array([.45+.11*math.cos(ang),-.26+.10*math.sin(ang),z+.11])
        start=np.array([.45,-.26,z])
        branches.sweep([start,end],.0035,8)
        for k in range(8):
            p=start+(end-start)*(k/9)+rng.normal(0,.008,3)
            a=ang+(-1 if k%2 else 1)*1.1
            d=np.array([.044*math.cos(a),.044*math.sin(a),.012]);side=np.array([-.011*math.sin(a),.011*math.cos(a),.002])
            ix=len(lv);lv.extend([p,p+d*.55+side,p+d,p+d*.55-side]);lf.append((ix,ix+1,ix+2,ix+3))
    tree=kit(branches.create('Kitchen · small tree stems',oak,True),'houseplant','Small branching countertop plant; species unresolved.',relation={'inside':pot.name})
    kit(surface('Kitchen · small tree leaves',lv,lf,leaf,.0004),'foliage','Simple lanceolate leaves on the countertop plant.',tree)
    for i,(x,z,h,r) in enumerate([(.46,1.33,.11,.043),(.84,1.33,.16,.042),(.67,1.64,.135,.039),(.90,1.64,.08,.057)],1):
        vessel('shelf vessel %02d'%i,x,-3.614,z,r,h,ceramic if i%2 else glass,bpy.data.objects['Kitchen · back open shelf %d'%(1 if i<3 else 2)],'storage vessel')

def build():
    dining_group()
    dining_art()
    kitchen_group()
