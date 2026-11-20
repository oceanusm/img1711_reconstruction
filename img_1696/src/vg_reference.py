"""Read matched VGGT inputs and preserve their original vertex attributes in Blender."""
from pathlib import Path
import re
import numpy as np

def reference_for(video):
    video = Path(video).expanduser().resolve()
    match = re.fullmatch(r'img_(\d+)', video.stem, re.IGNORECASE)
    if not match:
        raise ValueError(f'Expected an img_<ID> video, received {video.name}')
    directory = video.parent / (match.group(1) + '_vggt')
    for name in ('point_cloud.ply', 'index.json', 'cameras.npz'):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    return directory

def read_vertices(path):
    """Read the declared fields of a binary, vertex-first PLY as an immutable view."""
    types = {'float':'<f4','float32':'<f4','double':'<f8','uchar':'u1','uint8':'u1',
             'int':'<i4','uint':'<u4','int32':'<i4','uint32':'<u4','short':'<i2','ushort':'<u2'}
    with Path(path).open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise ValueError('Expected a PLY header')
        fields = []
        count = None
        active = None
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('Incomplete PLY header')
            parts = line.decode('ascii').split()
            if not parts:
                continue
            if parts[0] == 'end_header':
                break
            if parts[0] == 'format' and parts[1] != 'binary_little_endian':
                raise ValueError('This importer requires binary_little_endian PLY')
            if parts[0] == 'element':
                active = parts[1]
                if count is None and active != 'vertex':
                    raise ValueError('Expected vertex as the first element')
                if active == 'vertex':
                    count = int(parts[2])
            if parts[0] == 'property' and active == 'vertex':
                fields.append((parts[2], types[parts[1]]))
        offset = stream.tell()
    return np.memmap(path, dtype=fields, mode='r', offset=offset, shape=(count,))

def _display_nodes(obj, radius):
    import bpy
    nodes = bpy.data.node_groups.new('VGGT | inspection dots', 'GeometryNodeTree')
    nodes.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    nodes.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    source = nodes.nodes.new('NodeGroupInput')
    convert = nodes.nodes.new('GeometryNodeMeshToPoints')
    convert.mode = 'VERTICES'
    convert.inputs['Radius'].default_value = radius
    sink = nodes.nodes.new('NodeGroupOutput')
    nodes.links.new(source.outputs['Geometry'], convert.inputs['Mesh'])
    nodes.links.new(convert.outputs['Points'], sink.inputs['Geometry'])
    modifier = obj.modifiers.new('Visible inspection points', 'NODES')
    modifier.node_group = nodes

def import_reference(video, matrix, proxy_stride=50):
    """Import full cloud at native coordinates; the world matrix stores the similarity."""
    import bpy
    from mathutils import Matrix
    directory = reference_for(video)
    rows = read_vertices(directory / 'point_cloud.ply')
    if bpy.data.collections.get('VGGT_REFERENCE') is not None:
        raise RuntimeError('VGGT_REFERENCE is already present')
    collection = bpy.data.collections.new('VGGT_REFERENCE')
    bpy.context.scene.collection.children.link(collection)
    collection.hide_render = True
    collection['source_file'] = str(directory / 'point_cloud.ply')
    collection['inspection_only'] = True
    for full, data in ((True, rows), (False, rows[::proxy_stride])):
        label = 'VGGT | full-resolution source' if full else 'VGGT | visible inspection sample'
        me = bpy.data.meshes.new(label)
        me.vertices.add(len(data))
        coords = np.column_stack([data[k] for k in ('x','y','z')]).astype('f4',copy=False)
        me.vertices.foreach_set('co', coords.ravel())
        color = me.color_attributes.new('SourceRGB', 'BYTE_COLOR', 'POINT')
        rgba = np.ones((len(data),4),'f4')
        for channel, key in enumerate(('red','green','blue')):
            rgba[:,channel] = data[key].astype('f4') / 255
        color.data.foreach_set('color_srgb', rgba.ravel())
        if 'confidence' in data.dtype.names:
            attr = me.attributes.new('confidence', 'FLOAT', 'POINT')
            attr.data.foreach_set('value', np.ascontiguousarray(data['confidence'],dtype='f4'))
        if 'frame_index' in data.dtype.names:
            attr = me.attributes.new('source_frame_index', 'INT', 'POINT')
            attr.data.foreach_set('value', np.ascontiguousarray(data['frame_index'],dtype='i4'))
        me.update()
        obj = bpy.data.objects.new(label, me)
        collection.objects.link(obj)
        obj.matrix_world = Matrix(matrix)
        obj.hide_render = True
        obj['source_vertex_count'] = len(rows)
        obj['source_vertex_stride'] = 1 if full else proxy_stride
        obj['source_path'] = str(directory / 'point_cloud.ply')
        if full:
            obj.hide_set(True)
        else:
            obj.show_in_front = True
            _display_nodes(obj, .00175)
    return collection
