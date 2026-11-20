"""Editable stationary decor derived from the img_1696 front, side and close views."""
import json, math, random
from pathlib import Path
import bpy
from mathutils import Vector
from common import box, cyl, uv, mesh, mat, rod, tag, tube

ASSET=Path(__file__).resolve().with_name('assets')
TAU=2*math.pi
RNG=random.Random(1696)
basis_console='Support plane fitted to VGGT subsets at source frames 655, 689, 724 and 753. Silhouette and contact constraints from the moving close view; fine dimensions estimated.'
basis_cart='Cart shelf and castor positions estimated from views 494–510 and VGGT returns. Bottles use visible ordering, silhouette and rack spacing; hidden depths inferred.'
M={}
def material(key,color,rough=.54,metal=0):
    M[key]=mat('Decor | '+key,color,rough,metal)
    return M[key]
def finish(o,kind,desc,basis=basis_console,parent=None,first=0,support=None,confidence='low'):
    relations={'on':support.name} if support else None
    tag(o,kind,desc,basis,confidence,parent,relations)
    o['first_observed_frame']=first
    return o

class Parts:
    """A mesh for one physical item. Its fixed joined surfaces share a material table."""
    def __init__(self):
        self.v=[];self.f=[];self.m=[];self.ids=[];self.smooth=[]
    def poly(self,ids,material,smooth=False):
        if material not in self.m:self.m.append(material)
        self.f.append(tuple(ids));self.ids.append(self.m.index(material));self.smooth.append(smooth)
    def add(self,verts,faces,material,smooth=False):
        start=len(self.v);self.v.extend(verts)
        for f in faces:self.poly([start+i for i in f],material,smooth)
    def block(self,c,d,material):
        x,y,z=c;dx,dy,dz=[a*.5 for a in d]
        v=[(x+sx*dx,y+sy*dy,z+sz*dz) for sz in (-1,1) for sx,sy in [(-1,-1),(1,-1),(1,1),(-1,1)]]
        self.add(v,[(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)],material)
    def lathe(self,xy,rings,materials,n=32):
        start=len(self.v);x,y=xy
        for z,r in rings:
            self.v.extend([(x+r*math.cos(TAU*i/n),y+r*math.sin(TAU*i/n),z) for i in range(n)])
        for j in range(len(rings)-1):
            at=start+j*n;m=materials[j]
            for i in range(n):
                ii=(i+1)%n;self.poly((at+i,at+ii,at+n+ii,at+n+i),m,True)
        self.poly(range(start+n-1,start-1,-1),materials[0])
        self.poly(range(start+(len(rings)-1)*n,start+len(rings)*n),materials[-1])
    def stick(self,a,b,r,m,n=8):
        a,b=Vector(a),Vector(b);axis=(b-a).normalized();u=axis.cross(Vector((1,0,0)))
        if u.length<.1:u=axis.cross(Vector((0,1,0)))
        u.normalize();v=axis.cross(u)
        pts=[tuple(p+r*(math.cos(TAU*i/n)*u+math.sin(TAU*i/n)*v)) for p in (a,b) for i in range(n)]
        faces=[(i,(i+1)%n,n+(i+1)%n,n+i) for i in range(n)]+[tuple(range(n-1,-1,-1)),tuple(range(n,2*n))]
        self.add(pts,faces,m,True)
    def object(self,name):
        o=mesh(name,self.v,self.f)
        for m in self.m:o.data.materials.append(m)
        for f,idx,s in zip(o.data.polygons,self.ids,self.smooth):f.material_index=idx;f.use_smooth=s
        return o

def image_material(name,filename,base):
    m=mat('Decor | '+name,base)
    nodes=m.node_tree.nodes
    im=bpy.data.images.load(str(ASSET/filename),check_existing=True)
    im.pack()
    node=nodes.new('ShaderNodeTexImage');node.image=im
    m.node_tree.links.new(node.outputs['Color'],nodes.get('Principled BSDF').inputs['Base Color'])
    return m

def canvas(i,y0,y1):
    import numpy as np
    panels=json.loads((ASSET/'diptych_contours.json').read_text())
    up,lo,tr,br=panels[i]
    frontmats=[image_material(f'Canvas {i+1} | '+suffix,f'diptych_{i+1}.jpg',rgb)
               for suffix,rgb in [('blush','#cbb6ba'),('chalk','#f4efe8'),('charcoal','#191a20')]]
    material_ids=[frontmats[0],frontmats[1],frontmats[2],frontmats[1],frontmats[0]]
    sect=190;parts=Parts();z0=.933;z1=1.831
    front=.007 if i==0 else .002
    back=-.022
    parts.block(((back+front)/2,(y0+y1)*.5,(z0+z1)*.5),(front-back,y1-y0,z1-z0),M['pink paint'])
    front+=.00035; start=len(parts.v)
    for j in range(sect+1):
        t=j/sect;u=t*270
        def get(arr): return float(np.interp(u,*np.array(arr).T))
        wob=math.sin(u*1.9)*.55+math.sin(u*.72)*.46
        profile=[0,get(up)-get(tr)+wob,get(up)+wob,get(lo)-wob,get(lo)+get(br)-wob,216]
        for k,v in enumerate(profile):
            parts.v.append((front+[0,.0008,.002,.002,.0008,0][k],y0+(y1-y0)*t,z1-min(216,max(0,v))/216*(z1-z0)))
    for j in range(sect):
        for k in range(5):
            n=start+j*6+k
            parts.poly([n,n+1,n+7,n+6],material_ids[k])
    o=parts.object('Torn horizon canvas '+('A' if not i else 'B'))
    uvmap=o.data.uv_layers.new(name='Traced paint')
    for poly in o.data.polygons:
        for li in poly.loop_indices:
            co=o.data.vertices[o.data.loops[li].vertex_index].co
            uvmap.data[li].uv=((co.y-y0)/(y1-y0),(co.z-z0)/(z1-z0))
    return finish(o,'wall artwork','Pink relief canvas with a charcoal band and irregular white raised edges.',basis='Frames 80, 157, 612–785. Rectified image boundaries and cloud-constrained front surface. Fine relief depth estimated.',confidence='medium')

def capsule(name,center,size,m):
    x,y,z=center;w,l,h=size;r=w/2;off=l*.5-r;path=[]
    for cy,a0 in [(y+off,0),(y-off,math.pi)]:
        for i in range(33):
            a=a0+math.pi*i/32;path.append((x+r*math.cos(a),cy+r*math.sin(a)))
    n=len(path)
    verts=[(a,b,z+v*h/2) for v in (-1,1) for a,b in path]
    faces=[tuple(range(n-1,-1,-1)),tuple(range(n,2*n))]+[(i,(i+1)%n,(i+1)%n+n,n+i) for i in range(n)]
    o=mesh(name,verts,faces,m);mod=o.modifiers.new('Eased rim','BEVEL');mod.width=.003;mod.segments=2
    return o

def ringvase(name,x,y,z,width,height,depth,opening,oriented=False):
    """Hollow ring body, connected tubular throat with an open, recessed mouth."""
    p=Parts();n=64;k=16
    def P(a,b,c):return (x+b,y+a,z+c) if oriented else (x+a,y+b,z+c)
    # Main ceramic ring: front faces +X. Cross-section narrows in depth toward inner hole.
    verts=[]
    for i in range(n):
        a=TAU*i/n;cs,sn=math.cos(a),math.sin(a)
        for j in range(k):
            b=TAU*j/k;cb,sb=math.cos(b),math.sin(b)
            mid=(width*.5+opening*.5)/2;rad=(width*.5-opening*.5)/2
            zmid=(height*.43+opening*.52)/2;rz=(height*.43-opening*.52)/2
            yy=(mid+rad*cb)*cs;zz=height*.43+(zmid+rz*cb)*sn
            verts.append(P(depth*.5*sb,yy,max(.001,zz)))
    mm=M['ink ceramic'] if oriented else M['ivory bisque']
    faces=[(i*k+j,((i+1)%n)*k+j,((i+1)%n)*k+(j+1)%k,i*k+(j+1)%k) for i in range(n) for j in range(k)]
    p.add(verts,faces,mm,True)
    rings=[(z+height*.79,width*.104),(z+height*.84,width*.12),(z+height*.94,width*.102),(z+height,width*.095),(z+height,width*.074),(z+height*.88,width*.07)]
    p.lathe((x,y),rings,[mm,mm,mm,mm,M['inner clay']],32)
    return p.object(name)

def flowering_spray(name,base):
    """One branching spray with actual cupped, volumetric five-petal florets."""
    p=Parts();seed=random.Random(740)
    root=Vector(base)
    def pt(offset):return root+Vector(offset)
    def petal_bud(center,right,up,normal,length,width,color):
        n=7
        verts=[center+normal*.0005]
        for h,scale in ((.17,.78),(.56,1.0),(.90,.72)):
            for i in range(n):
                a=TAU*i/n
                verts.append(center+right*(length*(h+.18))+
                             up*(width*math.cos(a)*scale)+normal*((.0028*math.sin(a)+.0016)*scale+h*.0028))
        verts.append(center+right*length*1.17+normal*.005)
        faces=[(0,1+(i+1)%n,1+i) for i in range(n)]
        for j in range(2):
            for i in range(n):
                a=1+j*n+i;b=1+j*n+(i+1)%n
                faces.append((a,b,b+n,a+n))
        for i in range(n):faces.append((1+2*n+i,1+2*n+(i+1)%n,len(verts)-1))
        p.add(verts,faces,color,True)
    def floret(center):
        face=Vector((seed.uniform(-1,1),seed.uniform(-1,1),seed.uniform(-.7,1))).normalized()
        side=face.cross(Vector((0,0,1)))
        if side.length<.02:side=face.cross(Vector((1,0,0)))
        side.normalize();vertical=face.cross(side).normalized();spin=seed.random()*TAU
        for j in range(5):
            phi=spin+j*TAU/5
            out=math.cos(phi)*side+math.sin(phi)*vertical
            across=face.cross(out)
            petal_bud(center,out,across,face,seed.uniform(.0048,.0075),seed.uniform(.0036,.0052),M['blossom'] if j%3 else M['ivory bisque'])
        p.stick(center-face*.002,center+face*.004,.0024,M['young stem'],6)
    # Main axis initially rises together, then divides between z≈1.11 and 1.28.
    # Endpoints match distinct sprays on the left and right of source frame 740.
    stems=[((.008,-.006,.16),(-.015,-.066,.32),(-.035,-.168,.50)),
           ((-.009,.001,.20),(-.045,-.024,.36),(-.062,-.025,.51)),
           ((.001,.006,.20),(.015,.094,.35),(.005,.198,.447)),
           ((.011,.024,.21),(.011,.130,.33),(.015,.278,.355)),
           ((.012,.010,.155),(.030,.065,.26),(.035,.173,.297)),
           ((.001,-.003,.155),(.052,-.028,.25),(.050,-.094,.330))]
    for branch_idx,(fork,mid,end) in enumerate(stems):
        fork,mid,end=map(pt,(fork,mid,end))
        for a,b,r in [(root,fork,.0025),(fork,mid,.0020),(mid,end,.0014)]:
            p.stick(a,b,r,M['branch'],8)
        sites=5 if branch_idx<4 else 4
        for site in range(sites):
            t=(site+.56)/sites;anchor=fork.lerp(mid,min(1,t*1.75)) if t<.56 else mid.lerp(end,(t-.56)/.44)
            sign=(-1 if site%2 else 1)
            tip=anchor+Vector((seed.uniform(-.028,.028),sign*seed.uniform(.012,.027),seed.uniform(.015,.036)))
            p.stick(anchor,tip,.0009,M['young stem'],6)
            for f in range(9):
                ang=TAU*f/9+seed.uniform(-.35,.35)
                off=Vector((math.cos(ang)*seed.uniform(.004,.013),math.sin(ang)*seed.uniform(.005,.015),seed.uniform(-.018,.016)))
                p.stick(tip,tip+off,.0006,M['young stem'],5);floret(tip+off)
    return p.object(name)

def foliage(name,base,h,w,flower=False):
    if flower:return flowering_spray(name,base)
    p=Parts();stem=M['young stem'];seed=random.Random(1696+int(h*100))
    x,y,z=base; shrub=4 if flower else 6
    def petal(c,r,theta,color):
        a=theta;dx,dy=math.cos(a),math.sin(a)
        perp=Vector((-dy,dx,0));u=Vector((dx,dy,.4));v=Vector(c)
        p.add([tuple(v),tuple(v+u*r*.45+perp*r*.38),tuple(v+u*r+Vector((0,0,.003))),tuple(v+u*r*.45-perp*r*.38)],[(0,1,2),(0,2,3)],color,True)
    for i in range(shrub):
        angle=TAU*(i+.24)/shrub;end=Vector((x+w*(.8+seed.random()*.2)*math.cos(angle),y+w*math.sin(angle),z+h*(.70+seed.random()*.3)))
        start=Vector(base);mid=start.lerp(end,.48);mid.x+=.012;mid.y+=.012
        p.stick(start,mid,.0024 if flower else .0018,stem);p.stick(mid,end,.0017,stem)
        for t in range(3,9):
            at=start.lerp(end,t/9);spin=angle+t*2.16
            L=w*(.20+seed.random()*.18);tip=at+Vector((L*math.cos(spin),L*math.sin(spin),L*.48))
            p.stick(at,tip,.0012,stem,6)
            if flower:
                for m in range(3):
                    a=spin+m*TAU/3;c=tip+Vector((.010*math.cos(a),.010*math.sin(a),.006*(m%2)))
                    for b in range(5):petal(c,.008,TAU*b/5,M['blossom'])
            else:
                for sign in (-1,1):
                    d=spin+sign*.55;L2=.024+seed.random()*.014
                    A=Vector((math.cos(d),math.sin(d),.4))*L2;B=Vector((-math.sin(d),math.cos(d),.1))*L2*.21
                    midpt=tip+.5*A;midpt.z+=.003
                    p.add([tip,midpt+B,tip+A,midpt-B,midpt],[(0,1,4),(1,2,4),(2,3,4),(3,0,4)],M['leaf light' if t%2 else 'leaf'])
    return p.object(name)

def bottle(name,x,y,z,h,r,surface,label,lid,profile='wine'):
    if profile=='thermos':
        ring=[(0,.84),(.03,.96),(.10,1),(.79,1),(.82,.97),(.84,.96),(.86,.98),(.93,1),(.96,1),(.99,.92),(1,.83)]
        mats=[surface,surface,surface,surface,surface,label,lid,lid,lid,lid]
    elif profile=='jar':
        ring=[(0,.85),(.03,1),(.13,1),(.20,1),(.65,1),(.76,1),(.80,.72),(.86,.72),(.87,.83),(.97,.83),(1,.75)]
        mats=[surface,surface,surface,label,surface,surface,surface,lid,lid,lid]
    else:
        shoulder=.52 if profile=='decanter' else .60
        ring=[(0,.83),(.018,.94),(.045,1),(.14,1),(.18,1),(.43,1),(shoulder,1),(shoulder+.07,.83),(.77,.37),(.89,.34),(.9,.38),(.97,.38),(1,.34)]
        mats=[surface,surface,surface,surface,label,surface,surface,surface,surface,lid,lid,lid]
    p=Parts();p.lathe((x,y),[(z+dz*h,rr*r) for dz,rr in ring],mats,28)
    return p.object(name)

def pages(name,c,size,edge,cover,angle=0):
    x,y,z=c;dx,dy,dz=size
    p=Parts();p.block((0,0,0),(dx*.986,dy*.99,dz*.78),M['paper'])
    for sg in (-1,1):p.block((0,0,sg*dz*.445),(dx,dy,.11*dz),edge)
    p.block((0,0,dz*.509),(dx*.83,dy*.86,.0003),cover)
    if name=='Lavender architectural magazine':
        # Printed plum border, photographic window, perspective framing and ruled headline.
        def ink(px,py,w,h,color,level):
            p.block((px*dx,py*dy,dz*.51+level*.00007),(dx*w,dy*h,.00005),color)
        ink(.015,.005,.64,.72,M['paper'],1)
        ink(.032,.025,.48,.49,M['warm gray'],2)
        ink(-.060,.087,.20,.21,M['paper'],3)
        ink(.15,-.085,.16,.21,M['mauve cover'],4)
        for sid in (-1,1):
            ink(.01,sid*.30,.55,.013,M['mauve cover'],4)
            ink(sid*.25,0,.013,.52,M['mauve cover'],4)
        for j in range(9):
            ink(-.324,-.19+j*.045,.066,.015,M['warm gray'],4)
        ink(.365,0,.035,.65,M['mauve cover'],4)
    o=p.object(name);o.location=c;o.rotation_euler.z=angle
    return o

def game_box(i,center,dim,angle):
    x,y,z=center;dx,dy,dz=dim
    edge=M['ink blue'] if i==0 else M['red book']
    o=box('Catan '+('Explorers and Pirates' if i==0 else 'base game')+' box',center,dim,edge,0)
    mt=image_material('Game cover '+str(i),'game_'+str(i)+'_top.jpg','#435782' if not i else '#ba863e')
    ms=image_material('Game spine '+str(i),'game_'+str(i)+'_spine.jpg','#465982' if not i else '#667842')
    o.data.materials.append(mt);o.data.materials.append(ms);uvmap=o.data.uv_layers.new(name='Carton print')
    for poly in o.data.polygons:
        idx=poly.index;poly.material_index=1 if idx==1 else 2 if idx in (2,3,4,5) else 0
        for li in poly.loop_indices:
            co=o.data.vertices[o.data.loops[li].vertex_index].co
            if idx==1:u,v=co.y/dy+.5,.5-co.x/dx
            elif idx in (3,5):u,v=co.y/dy+.5,co.z/dz+.5
            else:u,v=co.x/dx+.5,co.z/dz+.5
            uvmap.data[li].uv=(u,v)
    o.rotation_euler.z=angle
    return o

def cap(center):
    x,y,z=center;p=Parts();n=48
    vs=[]
    for j in range(13):
        a=math.pi*.5*j/12
        for k in range(n):
            t=TAU*k/n;r=math.cos(a);vs.append((x+.097*r*math.cos(t),y+.09*r*math.sin(t),z+.100*math.sin(a)))
    faces=[(j*n+k,j*n+(k+1)%n,(j+1)*n+(k+1)%n,(j+1)*n+k) for j in range(12) for k in range(n)]
    p.add(vs,faces,M['navy cloth'],True)
    crown=[(x+(.083+.073*math.cos(t)),y+.092*math.sin(t),z+.005-.013*math.sin(t)**2) for t in [TAU*i/44 for i in range(44)]]
    p.add(crown+[(a,b,c-.004) for a,b,c in crown],[tuple(range(44)),tuple(range(87,43,-1))]+[(i,(i+1)%44,44+(i+1)%44,i+44) for i in range(44)],M['navy cloth'],True)
    p.lathe((x,y),[(z+.098,.006),(z+.102,.005)],[M['navy cloth']],12)
    return p.object('Navy six-panel baseball cap')

def ship():
    p=Parts();al=M['model gray'];nav=M['black'];red=M['model red'];base=(.131,6.296,.766)
    p.block((0,0,.006),(.105,.078,.012),nav);p.stick((-.018,0,.012),(0,0,.069),.009,nav,12)
    # Swept-back tail, a center dorsal ridge and a thin pointed forward hull.
    outline=[(-.084,-.055),(-.071,-.026),(-.089,-.022),(-.078,0),(-.089,.022),
             (-.071,.026),(-.084,.055),(-.054,.058),(.119,0),(-.054,-.058)]
    top=[(x,y,.085-.055*x) for x,y in outline];bot=[(x,y,.079-.055*x) for x,y in outline]
    count=len(outline)
    p.add(bot+top,[tuple(reversed(range(count))),tuple(range(count,2*count))]+
          [(i,(i+1)%count,(i+1)%count+count,i+count) for i in range(count)],al)
    ridge=[(-.080,-.010,.093),(.108,0,.083),(-.080,.010,.093),(-.056,0,.102)]
    p.add(ridge,[(0,1,3),(1,2,3),(2,0,3)],red)
    for st in (-1,1):
        p.add([(-.068,st*.044,.091),(.086,st*.008,.083),(-.041,st*.034,.094)],[(0,1,2)],M['silver'])
        p.add([(-.073,st*.027,.091),(.059,st*.008,.087),(-.027,st*.025,.092)],[(0,1,2)],al)
        for j in range(5):
            k=-.045+j*.015;span=.027-j*.002
            p.block((k,st*span,.091-.06*k),(.004,.005,.003),M['dark walnut'])
    p.block((-.049,0,.100),(.044,.027,.014),al)
    p.block((-.062,0,.111),(.021,.029,.010),al)
    p.stick((-.073,0,.116),(-.084,0,.137),.0028,M['silver'],8)
    o=p.object('Miniature wedge spacecraft on display stand');o.location=base;o.rotation_euler.z=1.14
    return o

def game_and_console():
    # Short woven shelf sits immediately beside the much longer oval console.
    p=Parts();sx,sy=.206,5.119;top=.756
    for h in (.130,.403,.746):
        p.block((sx,sy,h),(.39,.40,.018),M['cane'])
        for yy in (sy-.197,sy+.197):p.stick((sx-.20,yy,h),(sx+.20,yy,h),.009,M['rattan'],12)
        for xx in (sx-.194,sx+.194):p.stick((xx,sy-.20,h),(xx,sy+.20,h),.009,M['rattan'],12)
    for xx in (sx-.185,sx+.185):
        for yy in (sy-.185,sy+.185):p.stick((xx,yy,.016),(xx,yy,.754),.012,M['rattan'],12)
    for i in range(36):
        yy=sy-.18+.36*i/35;p.stick((sx-.18,yy,.7546),(sx+.18,yy,.7546),.0009,M['rattan dark'],6)
    short=finish(p.object('Rattan three-tier side shelf'),'side table','Three woven cane shelves on four round bamboo legs.',confidence='medium')
    finish(box('Dark woven basket under rattan shelf',(sx,sy,.247),(.31,.31,.197),M['basket'],.018),'basket','Reddish storage basket visible below partly obscured shelves.',support=short)
    textile=uv('Folded ivory cloth on middle shelf',(sx,sy,.509),(.36,.37,.195),M['blossom']);finish(textile,'textile','White folded item on the middle wicker shelf, partly hidden by sofa back.',support=short)
    bowl=bottle('Small ivory round pot',.100,5.010,top,.077,.039,M['ivory bisque'],M['ivory bisque'],M['blossom'],'jar')
    finish(bowl,'container','Short ivory round lidded pot behind the dark ornament.',support=short,first=610)
    p=Parts();p.lathe((.192,4.985),[(top,.022),(top+.015,.040),(top+.052,.050),(top+.065,.038)],[M['ink ceramic']]*3)
    p.stick((.18,4.953,.797),(.18,4.928,.805),.010,M['ink ceramic'])
    p.lathe((.18,4.915),[(.787,.013),(.798,.019),(.810,.018),(.816,.010)],[M['ink ceramic']]*3,20)
    finish(p.object('Black ceramic low handled ornament'),'tabletop ornament','Rounded low black dish or vessel with small projecting round tip. Function uncertain.',support=short,first=600)
    blk=finish(ringvase('Black loop-handled ceramic vase',.237,5.133,top,.15,.204,.078,.060,True),'vase','Dark open-loop silhouette and narrow mouth; edge-on in the close pass.',support=short)
    finish(foliage('Green sprigs in black vase',(.237,5.133,.95),.163,.136),'cut foliage','Several narrow green leaves on branched stems.',support=blk)
    mags=[]
    for i,(dx,dy,ang) in enumerate([(.265,5.268,-.25),(.245,5.231,.12)]):
        mag=pages('Yellow-bordered journal '+str(i+1),(dx,dy,.762+i*.009),(.236,.158,.009),M['yellow trim'],M['ink blue'],ang)
        mags.append(finish(mag,'magazine','Yellow-bordered dark photographic journal, partly beneath green flask.',support=short if i==0 else mags[0],first=577))
    finish(bottle('Olive thermal flask',.226,5.244,.778,.239,.040,M['olive'],M['silver'],M['black'],'thermos'),'drink flask','Olive metal travel bottle with black lid and silver collar.',support=mags[-1])
    console=finish(capsule('Pale oak oval sofa console top',(.226,6.061,.751),(.435,1.495,.026),M['blonde wood']),'console tabletop','Long blond capsule-ended tabletop behind the sectional.',confidence='medium')
    for i,yy in enumerate([5.519,6.580]):
        leg=box('Console trestle panel '+str(i+1),(.226,yy,.369),(.292,.042,.738),M['blonde wood'],.004)
        finish(leg,'support panel','Concealed support inferred beneath the oval console top.',parent=console)
    lower=finish(game_box(0,(.283,5.553,.803),(.309,.319,.078),-.08),'board game','Dark blue Catan Explorers & Pirates carton. Cover art is an illustrative reconstruction.',support=console)
    upper=finish(game_box(1,(.234,5.611,.882),(.303,.307,.080),.085),'board game','Catan red and green base-game carton with yellow title.',support=lower)
    finish(cap((.213,5.655,.925)),'baseball cap','Navy soft cap with projecting brim and crown button.',support=upper)
    book=finish(pages('Brown hardbound album',(.230,5.857,.773),(.243,.295,.018),M['brown cover'],M['brown cover']),'book','Wide brown album or binder underneath the printed periodicals.',support=console,first=693)
    j=finish(pages('Yellow-bordered journal on console',(.169,5.889,.790),(.216,.157,.012),M['yellow trim'],M['ink blue'],.023),'magazine','Narrow yellow-bordered magazine next to ring vases.',support=book,first=706)
    finish(pages('White editorial magazine',(.189,5.835,.803),(.210,.154,.009),M['paper'],M['warm gray'],-.048),'magazine','White magazine; portrait artwork replaced by neutral printed field.',support=j,first=706)
    lg=finish(ringvase('Large chalk ceramic ring vase',.107,6.127,.764,.248,.302,.091,.082),'vase','Broad cream sculptural vase with narrow hollow mouth behind smaller vessel.',support=console)
    sm=finish(ringvase('Small chalk ceramic ring vase',.178,6.146,.764,.176,.180,.065,.074),'vase','Smaller annular vase in front of larger, with real through-hole and neck.',support=console)
    finish(foliage('White flowering branches',(.178,6.146,.932),.573,.257,True),'cut flowers','Branching dark twigs and white blossoms rise between the canvases.',support=sm)
    finish(ship(),'display model','Gray wedge miniature on a black pedestal with red central stripe.',support=console,first=717)
    finish(pages('Lavender architectural magazine',(.293,6.284,.770),(.183,.230,.011),M['paper'],M['mauve cover'],.215),'magazine','Light lavender architecture publication beside the miniature.',support=console,first=714)
    finish(bottle('White supplement bottle',.407,6.181,.764,.103,.032,M['paper'],M['warm gray'],M['violet cap'],'jar'),'small bottle','White cylindrical plastic bottle with dark purple screw cap.',support=console,first=717)
    mag=finish(pages('Yellow-bordered journal at far end',(.228,6.575,.768),(.224,.163,.008),M['yellow trim'],M['ink blue'],.095),'magazine','Thin yellow-bordered journal partially covered by boxed item.',support=console,first=740)
    # Die-cut rectangular package frame with transparent display window.
    p=Parts();dx,dy=.243,.178;h=.023
    p.block((0,0,0),(dx,dy,.004),M['paper'])
    for sign in (-1,1):
        p.block((sign*(dx*.5-.014),0,h*.5),(.028,dy,h),M['yellow trim'] if sign<0 else M['red book'])
        p.block((0,sign*(dy*.5-.008),h*.5),(dx-.055,.016,h),M['paper'])
    p.block((0,0,.014),(dx-.056,dy-.032,.002),M['muted glass'])
    pkg=p.object('Windowed model kit carton');pkg.location=(.236,6.602,.776);pkg.rotation_euler.z=.33
    finish(pkg,'retail package','Shallow illustrated carton with transparent display window; identity uncertain.',support=mag,first=743)

def bar():
    x,y=3.750,3.095;w,d=.770,.380;bot,mid,top=.142,.670,.827
    p=Parts();wood=M['walnut'];dark=M['dark walnut']
    for zz in (bot,mid,top):
        p.block((x,y,zz),(w,d,.022),wood)
    for xx in (x-w*.5+.029,x+w*.5-.029):
        for yy in (y-d*.5+.024,y+d*.5-.024):
            p.block((xx,yy,.425),(.029,.032,.754),wood)
    for xx in (x-w*.5+.010,x+w*.5-.010):
        p.block((xx,y,top+.015),(.020,d,.032),wood)
    for yy in (y-d*.5+.009,y+d*.5-.009):
        p.block((x,yy,top+.015),(w,.018,.033),wood)
        p.block((x,yy,bot-.017),(w,.021,.039),wood)
        p.stick((x-w*.46,yy,bot+.040),(x+w*.46,yy,bot+.040),.0034,M['silver'])
    for t in range(5):
        xx=x-w*.31+t*.119;p.block((xx,y,mid-.025),(.045,.271,.026),dark)
    cart=finish(p.object('Walnut drinks trolley'),'bar cart','Dark timber two-shelf bar cart with a shallow intermediate rack, raised tray edges and stemware runners.',basis_cart,first=485,confidence='medium')
    for i,(xx,yy) in enumerate([(x+a*(w/2-.028),y+b*(d/2-.023)) for a in (-1,1) for b in(-1,1)]):
        caster=rod('Trolley caster '+str(i+1),(xx-.016,yy,.031),(xx+.016,yy,.031),.027,M['black'])
        finish(caster,'caster','Small dark swivel caster. No motion observed.',basis_cart,parent=cart,first=490)
    surfaces=[M['aged spirit'],M['flint glass'],M['honey glass'],M['green glass'],M['smoked glass']]
    labels=[M['parchment'],M['paper'],M['ink blue'],M['red book'],M['silver']]
    caps=[M['ink blue'],M['silver'],M['black'],M['red book'],M['aged spirit']]
    # From right to left in the source front view: distinct front bottles plus clearly visible rear necks.
    upper=[(-.295,.048,.239,.030,2,0,1),(-.213,.088,.233,.032,1,4,3),(-.120,.089,.222,.041,3,0,2),(-.013,.110,.219,.046,2,0,3),(.085,.085,.228,.036,1,2,1),(.178,.091,.244,.034,1,1,1),(.288,.074,.275,.036,2,0,0),(-.120,-.055,.259,.035,1,1,3),(.111,-.060,.265,.030,1,4,1),(.008,-.089,.215,.040,4,0,4)]
    lower=[(-.263,.052,.257,.030,1,1,1),(-.195,.092,.190,.029,1,1,2),(-.129,.099,.188,.029,2,1,2),(-.038,.084,.212,.037,2,0,4),(.061,.083,.247,.036,1,4,3),(.157,.083,.228,.032,1,3,3),(.251,.040,.292,.036,2,0,4),(-.090,-.055,.260,.034,4,0,4),(.041,-.066,.265,.032,1,0,2)]
    for name,z,arr in [('Upper',top+.011,upper),('Lower',bot+.011,lower)]:
        for i,(xx,yy,height,r,sm,lb,ld) in enumerate(arr):
            o=bottle(name+' cart bottle '+str(i+1).zfill(2),x+xx,y+yy,z,height,r,surfaces[sm],labels[lb],caps[ld],'decanter' if i%3==0 else 'wine')
            o=finish(o,'bottle','Individually modeled bottle with body, shoulders, narrow neck, closure and label.',basis_cart,first=490,support=cart)
            if name=='Upper' and i==6:
                finish(foliage('Green stems on bar trolley',(x+xx,y+yy,z+height),.280,.131),'cut foliage','Small leafy sprigs visible over upper trolley bottles.',basis_cart,first=490,support=o)
    for i in range(3):
        p=Parts();cx=x-.156+i*.128;cy=y+.03;zz=mid-.014
        # Trace the inside first and turn over the rim to keep a real open bowl.
        rings=[(zz-.071,.002),(zz-.079,.013),(zz-.095,.028),(zz-.122,.036),
               (zz-.153,.031),(zz-.153,.033),(zz-.122,.038),(zz-.095,.030),
               (zz-.078,.015),(zz-.065,.004),(zz-.012,.004),(zz-.007,.031),(zz,.033)]
        p.lathe((cx,cy),rings,[M['muted glass']]*(len(rings)-1))
        finish(p.object('Suspended stem glass '+str(i+1)),'wine glass','Transparent stemware hung upside down beneath the middle rack.',basis_cart,first=493,support=cart)
    o=bottle('Horizontal wine bottle on rack',0,0,0,.268,.036,M['green glass'],M['parchment'],M['silver'])
    o.rotation_euler.x=math.pi/2;o.location=(x+.072,y+.13,mid+.049)
    finish(o,'bottle','Bottle resting horizontally on the middle shelf.',basis_cart,first=498,support=cart)
    strip=finish(box('Black power strip by trolley',(4.484,3.109,.021),(.267,.060,.032),M['black'],.008),'power strip','Slim black electrical bar on the floor between the trolley and potted fern.',basis_cart,first=488)
    finish(tube('Trolley black power lead',[(4.35,3.109,.020),(4.30,3.11,.009),(4.265,3.155,.007),(4.235,3.225,.007),(4.14,3.28,.007),(4.05,3.304,.008),(4.01,3.29,.011),(3.98,3.24,.038),(3.96,3.13,.26)],.004,M['black']),'cable','Black power cable curves over the floor and behind the cart.',basis_cart,first=488,parent=strip)
    finish(tube('White lead behind trolley',[(4.34,2.936,.016),(4.39,2.979,.007),(4.446,3.015,.007),(4.488,3.007,.007),(4.504,2.990,.007),(4.49,2.982,.008),(4.38,2.976,.009),(4.23,3.02,.030),(4.13,3.04,.085)],.003,M['paper']),'cable','Lighter lead partly coiled along the skirting.',basis_cart,first=490)

def build():
    colors={'pink paint':'#cbb6ba','chalk paint':'#f4efe8','charcoal paint':'#191a20','ink ceramic':'#23252a','ink blue':'#35436b','ivory bisque':'#eee7d6','inner clay':'#aaa08b','branch':'#69604d','young stem':'#677954','leaf':'#496341','leaf light':'#738561','blossom':'#f4f1e5','cane':'#bb975f','rattan':'#bb9a6a','rattan dark':'#9b794b','basket':'#75523a','olive':'#646d36','black':'#23232a','yellow trim':'#f1cc36','paper':'#eeeae3','blonde wood':'#bda98c','red book':'#ab4230','brown cover':'#895333','warm gray':'#bcb7b1','mauve cover':'#bea8b8','violet cap':'#3d326b','navy cloth':'#252941','model gray':'#a3a5a9','model red':'#913b40','walnut':'#624b3a','dark walnut':'#3a2e27','aged spirit':'#916f48','flint glass':'#c3bbac','honey glass':'#b7995d','green glass':'#58684c','smoked glass':'#716756','parchment':'#d4c49c','muted glass':'#cdd0c9'}
    for name,value in colors.items():material(name,value)
    material('silver','#ababa9',.29,.48)
    for key in ['flint glass','green glass','honey glass','smoked glass','muted glass','aged spirit']:
        shader=M[key].node_tree.nodes.get('Principled BSDF')
        shader.inputs['Roughness'].default_value=.18;shader.inputs['Transmission Weight'].default_value=.32 if key=='muted glass' else .19
    canvas(0,4.757,5.970);canvas(1,6.163,7.365)
    game_and_console();bar()
