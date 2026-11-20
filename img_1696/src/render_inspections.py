"""Render matched orthographic images of evaluated authored triangles and VGGT vertices."""
import argparse,ctypes,json,shutil,subprocess,tempfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from vg_reference import reference_for,read_vertices

def font(size):
    for path in ['/System/Library/Fonts/Supplemental/Arial.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        if Path(path).exists():return ImageFont.truetype(path,size)
    return ImageFont.load_default()

def render(video,output,cache):
    output=Path(output);folder=output/'inspection';folder.mkdir(exist_ok=True)
    align=json.loads((output/'vggt_alignment.json').read_text());m=np.array(align['matrix'])
    stored=np.load(cache);verts=stored['vertices'];tri=stored['triangles'];labels=stored['labels'];base=stored['colors'];names=stored['names'].tolist()
    raw=read_vertices(reference_for(video)/'point_cloud.ply')[::17]
    pts=np.einsum('nj,ij->ni',np.column_stack([raw[k] for k in ('x','y','z')]),m[:3,:3])+m[:3,3]
    cloud_rgb=np.column_stack([raw[k] for k in ('red','green','blue')]).astype('f4')
    argv=[shutil.which('cc') or 'clang','-O3','-shared','-fPIC',str(Path(__file__).with_name('audit_raster.c')),'-o']
    normals=np.cross(verts[tri[:,1]]-verts[tri[:,0]],verts[tri[:,2]]-verts[tri[:,0]])
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-8)
    def make(key,title,subtitle,r,d,f,limits,depth_limits,scale,explanation,oblique=False):
        ax=np.array([r,d,f],dtype='f8');lo,hi,bt,tp=limits;w=int(np.ceil((hi-lo)*scale));h=int(np.ceil((tp-bt)*scale))
        def project(p):
            q=np.einsum('nj,ij->ni',p,ax);q[:,0]=(q[:,0]-lo)*scale;q[:,1]=(tp-q[:,1])*scale
            return q.astype('f4')
        pv=project(verts);pp=project(pts);a,b=depth_limits
        qtri=verts[tri];keep=np.ones(len(tri),'u1')
        if oblique:
            # Remove the roof and the near media-wall shell from this cutaway only.
            roof=np.array([('ceiling' in n.lower() or 'soffit' in n.lower() or n.startswith('Bar wall |') or n.startswith('Media wall |') or n=='Kitchen | side wall') for n in names])
            keep[roof[labels]]=0
            keep[np.all(qtri[:,:,1]>8.42,axis=1)|np.all(qtri[:,:,1]<.25,axis=1)]=0
            keep[np.all(qtri[:,:,0]<-.15,axis=1)|np.all(qtri[:,:,0]>4.95,axis=1)]=0
            keep[np.all(qtri[:,:,2]>2.40,axis=1)]=0
            keep[np.all(qtri[:,:,0]>4.25,axis=1)&np.all(qtri[:,:,1]>3.35,axis=1)]=0
        facing=np.abs(np.einsum('ij,j->i',normals,ax[2]));illum=.76+.24*facing
        rgb=((base.astype(float)*.69+np.array([216,211,197])*.31)*illum[:,None]).clip(0,255).astype('u1')
        pix=np.zeros((h,w,3),'u1')+[234,236,234];pix=np.ascontiguousarray(pix,dtype='u1')
        depth=np.full((h,w),-1e12,'f4');ids=np.full((h,w),-1,'i4')
        array=lambda arr: arr.ctypes.data_as(ctypes.c_void_p)
        dll.draw_mesh(w,h,len(tri),array(pv),array(tri),array(rgb),array(labels),array(keep),array(verts),int(oblique),ctypes.c_float(a),ctypes.c_float(b),array(depth),array(ids),array(pix))
        edge=np.zeros((h,w),bool)
        edge[1:]|=(ids[1:]!=ids[:-1]);edge[:,1:]|=(ids[:,1:]!=ids[:,:-1]);edge&=ids>=0
        pix[edge]=(pix[edge]*.60).astype('u1')
        src=Image.fromarray(pix)
        ix=np.floor(pp[:,0]).astype(int);iy=np.floor(pp[:,1]).astype(int);q=pp[:,2]
        flag=(ix>=0)&(ix<w)&(iy>=0)&(iy<h)&(q>=a)&(q<=b)
        if oblique:flag&=(pts[:,2]<2.4)&(pts[:,0]<4.25)&((pts[:,0]<2.93)|(pts[:,1]>2.87)|(pts[:,1]<2.55))
        ids0=np.flatnonzero(flag);wins=iy[flag]*w+ix[flag]
        order=np.lexsort((q[flag],wins));sort=wins[order];top=np.r_[sort[:-1]!=sort[1:],True]
        chosen=ids0[order[top]];py,px=iy[chosen],ix[chosen]
        # Faint dim points may show a receding surface immediately behind the authored model.
        gap=depth[py,px]-q[chosen]
        visible=(gap<.10)
        chosen=chosen[visible];py,px=py[visible],px[visible]
        lum=np.einsum('ij,j->i',cloud_rgb[chosen],[.2126,.7152,.0722])/255
        ink=np.column_stack((np.full(len(lum),7),122+54*lum,137+64*lum))
        over=pix.copy();over[py,px]=(pix[py,px]*.18+ink*.82).astype('u1')
        overlay=Image.fromarray(over)
        stack=w/h>1.45;pad=26;header=94;label_h=30;foot=70
        W=(w if stack else 2*w+pad)+2*pad;H=header+(2*(h+label_h)+12 if stack else h+label_h)+foot
        sheet=Image.new('RGB',(W,H),'#faf9f5');ds=ImageDraw.Draw(sheet)
        ds.text((pad,16),title,font=font(29),fill='#142a2c');ds.text((pad,55),subtitle,font=font(16),fill='#485858')
        for j,(im,label) in enumerate([(src,'Authored geometry'),(overlay,'Authored geometry + VGGT samples')]):
            ox=pad+(w+pad)*j*(not stack);oy=header+j*(h+label_h+12)*stack
            ds.text((ox,oy),label,font=font(16),fill='#253d3b');sheet.paste(im,(ox,oy+label_h))
            sx,sy=ox+16,oy+label_h+h-18;ds.line((sx,sy,sx+scale,sy),fill='#18322f',width=3)
            ds.line((sx,sy-4,sx,sy+4),fill='#18322f',width=2);ds.line((sx+scale,sy-4,sx+scale,sy+4),fill='#18322f',width=2)
            ds.text((sx+scale+6,sy-9),'1 m',font=font(13),fill='#132a2a',stroke_width=1,stroke_fill='#e9e7df')
        y=H-53;ds.ellipse((pad,y+3,pad+11,y+14),fill='#0a92a2')
        ds.text((pad+19,y),'VGGT samples in cyan. The scale bar uses the estimated scene scale.',font=font(15),fill='#334c4b')
        ds.text((pad,y+23),explanation,font=font(15),fill='#334c4b')
        sheet.save(folder/(key+'.png'),optimize=True)
        if key=='01_plan' and 'audit_points' in stored:
            # Flatten the 0–1.7 m slice to show the location of scarce or displaced evidence.
            ap=project(stored['audit_points']);ad=stored['audit_distances']
            xx=np.floor(ap[:,0]).astype(int);yy=np.floor(ap[:,1]).astype(int)
            ok=(xx>=1)&(xx<w-1)&(yy>=1)&(yy<h-1)&(ap[:,2]>=a)&(ap[:,2]<=b)
            sel=np.flatnonzero(ok);sel=sel[np.argsort(ap[sel,2])]
            panel=Image.fromarray(((pix.astype(float)*.52)+np.array([239,239,235])*.48).astype('u1'))
            dots=ImageDraw.Draw(panel)
            palette=['#24877f','#c69d15','#df732a','#a33a70']
            levels=np.searchsorted([.025,.075,.15],ad)
            for ip in sel:
                xx0,yy0=int(xx[ip]),int(yy[ip]);dots.rectangle((xx0,yy0,xx0+1,yy0+1),fill=palette[levels[ip]])
            W=2*w+pad*3;H=h+221
            canvas=Image.new('RGB',(W,H),'#faf9f5');draw=ImageDraw.Draw(canvas)
            draw.text((pad,15),'Reference-to-mesh discrepancy',font=font(29),fill='#142a2c')
            draw.text((pad,54),'Top view of reference points below 1.70 m. Each dot uses its unsigned distance to the nearest evaluated mesh.',font=font(16),fill='#40504f')
            draw.text((pad,93),'Authored geometry',font=font(16),fill='#253d3b');draw.text((w+2*pad,93),'Distance to nearest authored surface',font=font(16),fill='#253d3b')
            canvas.paste(src,(pad,123));canvas.paste(panel,(w+2*pad,123))
            yp=h+141
            for j,lab in enumerate(['Up to 2.5 cm','2.5–7.5 cm','7.5–15 cm','Over 15 cm']):
                xp=pad+175*j
                draw.rectangle((xp,yp,xp+13,yp+13),fill=palette[j]);draw.text((xp+20,yp-2),lab,font=font(15),fill='#273d3d')
            allreg=align['authored_surface_validation']['regions']['all_sampled_vertices']
            draw.text((pad,yp+27),f"All elevations, {allreg['number_of_reference_samples']:,} samples: median {allreg['median_m']*100:.1f} cm, 90th percentile {allreg['p90_m']*100:.1f} cm.",font=font(15),fill='#334c4b')
            draw.text((pad,yp+48),'Distances include cloud scatter and geometric approximation. Nearby objects can reduce the measured distance.',font=font(15),fill='#334c4b')
            canvas.save(folder/'06_surface_discrepancy.png',optimize=True)

    with tempfile.TemporaryDirectory() as t:
        lib=Path(t)/'audit.so';subprocess.run(argv+[str(lib)],check=True);dll=ctypes.CDLL(str(lib))
        make('01_plan','Room plan','Horizontal section below 1.70 m. Windows are at the top and the dining table is at the bottom.',
             [1,0,0],[0,1,0],[0,0,1],(-.22,4.98,.2,8.48),(-.10,1.70),101,
             'The open lane and the concealed floor have uneven sampling. Furniture surfaces carry most of the local observations.')
        make('02_artwork_wall','Artwork wall','Elevation viewed from the room, including the first 0.80 m of depth.',
             [0,1,0],[0,0,1],[1,0,0],(.35,8.33,-.06,2.63),(-.13,.80),150,
             'The wall and canvas faces are sampled. Areas behind seating and below the console have much less coverage.')
        make('03_media_wall','TV and fireplace','Elevation of the media wall, including the front surfaces of the mounted display and fire surround.',
             [0,-1,0],[0,0,1],[-1,0,0],(-7.45,-3.42,-.04,2.61),(-4.94,-4.21),186,
             'The display has multi-view depth scatter. The cloud traces the fireplace surround, with few points inside.')
        make('04_windows','Window wall','Elevation facing the far end of the room, with the rear 1.4 m of the living area visible.',
             [1,0,0],[0,0,1],[0,-1,0],(-.20,4.89,-.05,2.61),(-8.38,-6.90),184,
             'Opaque portions of the windows and shades return few points. The central pier and seat-back profiles are better observed.')
        east=np.array([.848,.53,0]);east/=np.linalg.norm(east);fore=np.array([.53,-.848,.77]);fore/=np.linalg.norm(fore);up=np.cross(fore,east);up/=np.linalg.norm(up)
        make('05_cutaway','Oblique room cutaway','The ceiling and near media-wall shell are removed from this inspection view.',
             east,up,fore,(-.2,8.96,-1.30,5.87),(-20,20),110,
             'Elevated seating, tables and console surfaces can be compared across depth. Thin supports have intermittent cloud coverage.',True)
    align['inspection_images']={
        'method':'Orthographic depth-buffered projections of Blender-evaluated triangular surfaces, at source frame 0. The clay-style material colors prioritize geometry and omit surface textures. Every 17th PLY point, transformed with the published matrix, supplies the cyan comparison layer; receding points more than 0.10 m behind the visible model are suppressed. The discrepancy panel uses the separate every-83rd-point BVH evaluation.',
        '01_plan.png':'XY room plan below 1.70 m height.',
        '02_artwork_wall.png':'YZ elevation of X=-0.13 to +0.80 m, isolating the artwork wall and nearby furniture.',
        '03_media_wall.png':'YZ elevation through the media display and fireplace.',
        '04_windows.png':'XZ elevation of the window wall and nearby seating, with partial returns on shades and glazing.',
        '05_cutaway.png':'Oblique orthographic interior cutaway viewed from above the dining/media side.',
        '06_surface_discrepancy.png':'Plan colored by reference-to-authored triangle distance within the lower room slice.'}
    (output/'vggt_alignment.json').write_text(json.dumps(align,indent=2)+'\n')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('video');ap.add_argument('output');ap.add_argument('triangle_cache')
    opt=ap.parse_args();render(opt.video,opt.output,opt.triangle_cache)
