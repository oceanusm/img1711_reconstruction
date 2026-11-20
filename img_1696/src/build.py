"""Run inside Blender 4.2+: blender --background --python build.py -- /path/src/img_1696.mov"""
import sys,os,json,hashlib,importlib
from pathlib import Path
SRC=Path(__file__).resolve().parent
if str(SRC) not in sys.path:sys.path.insert(0,str(SRC))
import bpy
import numpy as np
from mathutils import Vector
import common
from vg_reference import reference_for,import_reference

SOURCE=Path('~/Desktop/scene_reconstruction/src/img_1696.mov').expanduser()
PROFILE_SHA='078681b5795fc3141e95cd53e9385c2263286a48e4b8e520f03f5e1a2a8c3b10'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def lights(scene):
    collection=bpy.data.collections.new('Render lighting');scene.collection.children.link(collection)
    def area(name,loc,target,energy,size,clr):
        light=bpy.data.lights.new(name,'AREA');light.energy=energy;light.shape='DISK';light.size=size;light.color=clr
        o=bpy.data.objects.new(name,light);collection.objects.link(o);o.location=loc
        o.rotation_euler=(Vector(target)-Vector(loc)).to_track_quat('-Z','Y').to_euler()
    area('Window daylight | broad source',(2.02,7.96,1.69),(1.95,2.55,.53),405,3.3,(.965,.974,1))
    area('Living room | bounced ceiling light',(2.33,5.5,2.48),(2.17,5.73,.15),248,3.1,(1,.965,.91))
    area('Dining room | bounced ceiling light',(2.1,2,2.48),(1.32,2.5,.2),227,3.4,(1,.967,.92))
    area('Kitchen | daylight fill',(2.05,-2.0,2.51),(1.8,-1.2,.45),230,2.4,(.95,.974,1))
    area('Art wall | diffuse room bounce',(3.94,4.0,1.85),(0,5.02,1.25),110,3.0,(.97,.98,1))
    world=bpy.data.worlds.new('Daylight sky');world.use_nodes=True
    bg=world.node_tree.nodes.get('Background');bg.inputs[0].default_value=(.79,.86,1.,1);bg.inputs[1].default_value=.65
    scene.world=world
    scene.render.engine='BLENDER_EEVEE_NEXT';scene.eevee.taa_render_samples=4
    scene.view_settings.view_transform='AgX'
    scene.view_settings.exposure=.12
    scene.render.film_transparent=False
    scene.render.resolution_percentage=100

def screens(camera,inspect=False):
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                sp=area.spaces.active;sp.region_3d.view_perspective='CAMERA';sp.region_3d.view_camera_zoom=0
                sp.shading.type='SOLID';sp.shading.color_type='MATERIAL'
                sp.overlay.show_floor=False;sp.overlay.show_axis_x=False;sp.overlay.show_axis_y=False;sp.overlay.show_relationship_lines=False;sp.overlay.show_extras=False
                sp.overlay.show_overlays=True
    # A scene made from Blender's Python Console should reopen showing geometry.
    if getattr(bpy.context,'window_manager',None):
        for win in bpy.context.window_manager.windows:
            if win.workspace and win.workspace.name=='Layout':
                for area in win.screen.areas:
                    if area.type=='CONSOLE' and area.width>350:
                        area.type='VIEW_3D';sp=area.spaces.active;sp.region_3d.view_perspective='CAMERA';sp.shading.color_type='MATERIAL';sp.overlay.show_extras=False;sp.overlay.show_floor=False

def build(source=SOURCE,output=None,cache=None):
    video=Path(source).expanduser().resolve();refdir=reference_for(video)
    output=Path(output or (video.parent.parent/'output'/video.stem)).resolve()
    if digest(video)!=PROFILE_SHA:raise ValueError(f'Matched references in {refdir}. The authored scene profile is specific to img_1696. A new video requires its own object model and metric reference.')
    output.mkdir(parents=True,exist_ok=True)
    from fit_img1696_alignment import estimate
    if not (output/'vggt_alignment.json').exists():estimate(video,output)
    alignment=json.loads((output/'vggt_alignment.json').read_text());track=SRC/'camera_track.npz'
    with np.load(track) as data:
        if str(data['reference_sha256'])!=digest(refdir/'cameras.npz') or str(data['source_sha256'])!=PROFILE_SHA:raise ValueError('Camera track belongs to a different input')
    for name,h in alignment['reference_sha256'].items():
        fname=Path(alignment['reference_paths'].get(name,name)).name
        if digest(refdir/fname)!=h:raise ValueError('Original reference changed: '+fname)
    # All generated content is local to this scene. Launch in a new, dedicated Blender session.
    for o in list(bpy.data.objects):bpy.data.objects.remove(o,do_unlink=True)
    for c in list(bpy.data.collections):bpy.data.collections.remove(c)
    bpy.data.orphans_purge(do_recursive=True)
    scene=bpy.context.scene;scene.name=video.stem+' | measured interior';scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1;scene.unit_settings.length_unit='METERS'
    bpy.context.preferences.filepaths.save_version=0
    from blender_camera import add_camera
    camera=add_camera(scene,track,alignment['matrix'],output/'camera.json')
    print('Per-frame camera exported',flush=True)
    ref=import_reference(video,alignment['matrix'])
    print('Loaded source PLY in reconstruction scene',flush=True)
    for name,label in [('architecture','01 Room and fixtures'),('dining','02 Dining and kitchen'),('seating','03 Sectional and coffee table'),('decor','04 Console, art and bar'),('exterior','05 Exterior through windows')]:
        collection=bpy.data.collections.new(label);scene.collection.children.link(collection);common.COLLECTION=collection
        before=set(bpy.data.objects)
        mod=importlib.import_module(name);importlib.reload(mod);mod.build()
        for o in set(bpy.data.objects)-before:
            if collection not in list(o.users_collection):
                for old in list(o.users_collection):old.objects.unlink(o)
                collection.objects.link(o)
            if name=='exterior' and not o.get('first_observed_frame'):
                o['first_observed_frame']=332 if 'SUV' in o.name or 'car' in o.name.lower() else 108
        print(label,len(collection.objects),flush=True)
    common.COLLECTION=None;lights(scene)
    import lighting
    importlib.reload(lighting).configure(scene)
    scene.frame_set(1)
    from measure import write_manifest
    write_manifest(output,video);print('Manifest exported',flush=True)
    for image in bpy.data.images:
        if image.filepath and image.type=='IMAGE' and not image.packed_file and image.has_data:image.pack()
    scene.render.filepath='//render.mp4'
    scene.render.image_settings.file_format='FFMPEG';scene.render.ffmpeg.format='MPEG4';scene.render.ffmpeg.codec='H264'
    for o in bpy.context.selected_objects:o.select_set(False)
    screens(camera,True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output/'scene_with_vggt.blend'),compress=True)
    print('Inspection Blender scene saved',flush=True)
    if cache:
        import vggt_audit
        importlib.reload(vggt_audit).capture(video,output,cache);print('Evaluated comparison exported',flush=True)
    for obj in list(ref.objects):bpy.data.objects.remove(obj,do_unlink=True)
    bpy.data.collections.remove(ref)
    bpy.data.orphans_purge(do_recursive=True)
    scene.frame_set(1);screens(camera)
    bpy.ops.wm.save_as_mainfile(filepath=str(output/'scene.blend'),compress=True)
    print('Clean Blender scene saved',flush=True)
    return output

if __name__=='__main__':
    argv=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    build(argv[0] if argv else SOURCE)
