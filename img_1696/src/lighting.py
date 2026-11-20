"""Lighting, window transparency and reflections evaluated against upright source views."""
import bpy,math
def configure(scene,contrast='AgX - Medium High Contrast',scale=.64,ambient=.075,exposure=-.38):
    nodes=scene.world.node_tree.nodes;links=scene.world.node_tree.links
    bg=nodes.get('Background');bg.inputs['Strength'].default_value=ambient
    out=next(n for n in nodes if n.type=='OUTPUT_WORLD')
    if not nodes.get('Visible outdoor sky'):
        sky=nodes.new('ShaderNodeBackground');sky.name='Visible outdoor sky';sky.inputs['Color'].default_value=(.80,.87,1,1);sky.inputs['Strength'].default_value=1.17
        path=nodes.new('ShaderNodeLightPath');mix=nodes.new('ShaderNodeMixShader')
        links.new(path.outputs['Is Camera Ray'],mix.inputs[0]);links.new(bg.outputs[0],mix.inputs[1]);links.new(sky.outputs[0],mix.inputs[2]);links.new(mix.outputs[0],out.inputs[0])
    for o in scene.objects:
        if o.type=='LIGHT':
            if 'unadjusted_energy' not in o.data:o.data['unadjusted_energy']=o.data.energy
            o.data.energy=o.data['unadjusted_energy']*scale
    scene.eevee.taa_render_samples=12
    scene.eevee.use_raytracing=True
    scene.view_settings.view_transform='AgX';scene.view_settings.look=contrast;scene.view_settings.exposure=exposure
    for mt in bpy.data.materials:
        b=mt.use_nodes and mt.node_tree.nodes.get('Principled BSDF')
        if not b:continue
        n=mt.name.lower()
        if ('window glazing' in n or 'heat-resistant glazing' in n):
            mt.surface_render_method='BLENDED';mt.use_transparency_overlap=False
        if any(x in n for x in ('weave','linen','plaster','canvas','fabric','cloth','textile','faux fur','dining ·','rug','cotton','printed','diptych','tv | purple')):
            b.inputs['Specular IOR Level'].default_value=.16
    if not bpy.data.objects.get('Reflection | round bar mirror'):
        probe=bpy.data.lightprobes.new('Round mirror plane','PLANE')
        obj=bpy.data.objects.new('Reflection | round bar mirror',probe);bpy.data.collections['Render lighting'].objects.link(obj)
        obj.location=(3.775,2.896,1.603);obj.rotation_euler.x=-math.pi/2;obj.scale=(.445,.445,1);probe.influence_distance=2.7
