"""Export binary entity masks and timestamped, inspectable RGB/label previews."""
from __future__ import annotations

import colorsys
import html
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from .common import load_manifest, sha256, write_json
from .compare import read_mask
from .entity_manifest import load_entity_manifest
from .ffmpeg_compat import video_sync_args


def load_bundle(render_index, entities_path, manifest_path, *, sparse=False):
    render_index, entities_path, manifest_path = map(lambda p: Path(p).resolve(),
                                                    (render_index, entities_path, manifest_path))
    index = json.loads(render_index.read_text())
    entities = load_entity_manifest(entities_path)
    manifest = load_manifest(manifest_path, allow_sparse=sparse)
    if index.get('status') != 'complete':
        raise ValueError('Entity render must be complete')
    if index.get('entities_sha256') != sha256(entities_path) or index.get('source_blend_sha256') != entities['source_blend_sha256']:
        raise ValueError('Rendered labels and entity manifest provenance differ')
    canonical_hash = manifest.get('source_manifest_sha256', sha256(manifest_path)) if sparse else sha256(manifest_path)
    if index.get('source_manifest_sha256') != canonical_hash:
        raise ValueError('Entity render belongs to a different source video manifest')
    if sparse and index.get('selection_sha256') != manifest.get('selection_sha256'):
        raise ValueError('SAM subset and Blender frame selection differ')
    if (index['width'], index['height']) != (manifest['width'], manifest['height']):
        raise ValueError('Masks require the exact canonical pixel grid; render at resolution-scale 1')
    rows = index.get('frames', [])
    ids = [r['index'] for r in rows]
    if len(rows) != index['frame_count'] or ids != sorted(set(ids)) or not rows:
        raise ValueError('Entity frame records are incomplete or duplicated')
    originals = {f['index']: f for f in manifest['frames']}
    if sparse and set(ids) != set(originals):
        raise ValueError('Entity render does not cover the exact SAM frame subset')
    for row in rows:
        if row['index'] not in originals:
            raise ValueError('Entity render and source timestamps differ')
        for key in ('time_seconds', 'pts_seconds'):
            if not np.isfinite(row[key]) or abs(row[key]-originals[row['index']][key]) > 1e-6:
                raise ValueError('Entity render and source timestamps differ')
        if not (render_index.parent/row['entity_file']).is_file():
            raise FileNotFoundError(render_index.parent/row['entity_file'])
    return index, entities, manifest, originals


def load_labels(path, shape, ids):
    with np.load(path, allow_pickle=False) as archive:
        labels = archive['entity_ids']
    if labels.shape != shape or labels.dtype != np.uint16 or not np.isin(labels, [0, *ids]).all():
        raise ValueError('Entity label raster has invalid grid, dtype or entity IDs')
    return labels


def export_masks(args):
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    index, entities, manifest, _ = load_bundle(args.render_index, args.entities, args.manifest, sparse=True)
    rows = entities['entities']
    requested = getattr(args, 'objects', None)
    if requested:
        if not set(requested).issubset({e['name'] for e in rows}):
            raise ValueError('Requested mask names absent from entity manifest')
        rows = [e for e in rows if e['name'] in requested]
    shape = (index['height'], index['width'])
    allowed = [e['id'] for e in entities['entities']]
    for entity in rows:
        (output/'masks'/entity['name']).mkdir(parents=True, exist_ok=True)
    metadata = {'schema_version': 1, 'status': 'running', 'frame_count': len(index['frames']),
        'width': index['width'], 'height': index['height'], 'manifest_sha256': sha256(Path(args.manifest)),
        'source_manifest_sha256': index['source_manifest_sha256'], 'selection_sha256': index['selection_sha256'],
        'entities_sha256': index['entities_sha256'], 'render_index_sha256': sha256(Path(args.render_index)),
        'objects': {e['name']: e['object_names'] for e in rows},
        'mask_semantics': index['label_semantics'], 'rendering': index['rendering']}
    write_json(output/'metadata.json', metadata)
    try:
        for position, frame in enumerate(index['frames']):
            labels = load_labels(Path(args.render_index).parent/frame['entity_file'], shape, allowed)
            for entity in rows:
                Image.fromarray((labels == entity['id']).astype(np.uint8)*255).save(
                    output/'masks'/entity['name']/f"{frame['index']:06d}.png", compress_level=1)
            if position % 20 == 0:
                print(f'Binary entity masks: {position+1}/{len(index["frames"])} frames', flush=True)
    except BaseException as error:
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        write_json(output/'metadata.json', metadata)
        raise
    metadata['status'] = 'complete'
    write_json(output/'metadata.json', metadata)
    print(f'Exported {len(rows)} entity masks at {len(index["frames"])} timestamps: {output}')


def palette(entities):
    colors = np.zeros((max(e['id'] for e in entities)+1, 3), dtype=np.uint8)
    for entity in entities:
        colors[entity['id']] = np.rint(np.asarray(colorsys.hsv_to_rgb((entity['id']*.61803398875)%1, .7, 1))*255)
    return colors


def tint(rgb, labels, colors):
    values = np.asarray(rgb.convert('RGB')).copy()
    valid = labels > 0
    values[valid] = (.5*values[valid]+.5*colors[labels[valid]]).astype(np.uint8)
    return Image.fromarray(values)


def preview(args):
    index, entity_manifest, manifest, originals = load_bundle(args.render_index, args.entities, args.manifest)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    rows, entities = index['frames'], entity_manifest['entities']
    colors, allowed = palette(entities), [e['id'] for e in entities]
    render_root, source_root = Path(args.render_index).resolve().parent, Path(args.manifest).resolve().parent
    for row in rows:
        if not row.get('rgb_file') or not (render_root/row['rgb_file']).is_file():
            raise ValueError('RGB renders are required for the entity preview')
        for path in (render_root/row['rgb_file'], source_root/originals[row['index']]['file']):
            with Image.open(path) as image:
                if image.size != (index['width'], index['height']):
                    raise ValueError(f'RGB image grid differs from entity labels: {path}')
    sam_masks = getattr(args, 'sam_masks', None)
    reviewed = []
    if sam_masks:
        sam_masks = Path(sam_masks).resolve()
        meta = json.loads((sam_masks.parent/'metadata.json').read_text())
        if meta.get('status') != 'complete' or 'mapping_sha256' not in meta:
            raise ValueError('Use assembled physical-entity masks, not SAM concept unions')
        sam_manifest_path = Path(args.sam_manifest).resolve() if getattr(args, 'sam_manifest', None) else None
        if sam_manifest_path is None or sha256(sam_manifest_path) != meta.get('manifest_sha256'):
            raise ValueError('Supply the exact --sam-manifest used for assembled masks')
        _, _, subset, _ = load_bundle(args.render_index, args.entities, sam_manifest_path, sparse=True)
        if (meta.get('width'), meta.get('height')) != (index['width'], index['height']):
            raise ValueError('Assembled SAM mask metadata has a different pixel grid')
        if subset.get('source_manifest_sha256') != index['source_manifest_sha256'] or subset.get('selection_sha256') != index['selection_sha256']:
            raise ValueError('Assembled masks use a different source or selection')
        if [f['index'] for f in subset['frames']] != [f['index'] for f in rows]:
            raise ValueError('Assembled mask frames differ from Blender render frames')
        for frame in subset['frames']:
            if (sam_manifest_path.parent/frame['file']).resolve() != (source_root/originals[frame['index']]['file']).resolve():
                raise ValueError('SAM subset image path differs from canonical input')
        names = set(meta['objects'])
        if not names.issubset({e['name'] for e in entities}):
            raise ValueError('Assembled masks contain names outside this entity manifest')
        reviewed = [e for e in entities if e['name'] in names]
        for entity in reviewed:
            for row in rows:
                if not (sam_masks/entity['name']/f"{row['index']:06d}.png").is_file():
                    raise FileNotFoundError(f'Missing assembled mask: {entity["name"]}/{row["index"]}')
    height = int(getattr(args, 'max_height', 480))
    if height < 32:
        raise ValueError('max_height must be at least 32')
    height += height % 2
    width = int(round(height*index['width']/index['height']/2))*2
    size = (width, height)
    shape = (index['height'], index['width'])
    times = [row['time_seconds'] for row in rows]
    end = times[-1]+originals[rows[-1]['index']]['duration_seconds']
    durations = np.diff(times+[end])
    output.mkdir(parents=True, exist_ok=True)
    screenshots = []
    sync = video_sync_args('ffmpeg', 'vfr')
    with tempfile.TemporaryDirectory(prefix='entity-preview-') as tmp:
        tmp = Path(tmp); concat = ['ffconcat version 1.0']
        for position, row in enumerate(rows):
            labels = load_labels(render_root/row['entity_file'], shape, allowed)
            with Image.open(source_root/originals[row['index']]['file']) as image:
                source = image.convert('RGB')
            with Image.open(render_root/row['rgb_file']) as image:
                rendered = image.convert('RGB')
            source_title = 'Original video'
            if sam_masks:
                occupancy = np.zeros(shape, dtype=np.uint16)
                sam_labels = np.zeros(shape, dtype=np.uint16)
                for entity in reviewed:
                    mask = read_mask(sam_masks/entity['name']/f"{row['index']:06d}.png", (index['width'],index['height']))
                    occupancy += mask
                    sam_labels[mask] = entity['id']
                source = tint(source, sam_labels, colors)
                values = np.asarray(source).copy(); values[occupancy > 1] = [255,255,255]
                source = Image.fromarray(values)
                source_title = 'Original + reviewed SAM (overlap white)'
            tiles = [source, rendered, tint(rendered, labels, colors)]
            canvas = Image.new('RGB', (width*3,height+48),(19,25,33)); draw=ImageDraw.Draw(canvas)
            for j,(tile,title) in enumerate(zip(tiles,[source_title,'Blender RGB','Blender entity-ID tint'])):
                canvas.paste(tile.resize(size,Image.Resampling.LANCZOS),(width*j,48))
                draw.text((width*j+6,6),title,fill='white')
            draw.text((6,27),f"Source {row['index']} | {row['time_seconds']:.3f}s | {entity_manifest['scene_key']}",fill='white')
            name=f'{position:06d}.png'; canvas.save(tmp/name)
            concat.extend([f"file '{name}'",'option framerate 1000000',f'duration {durations[position]:.9f}'])
            if position in {0,len(rows)//2,len(rows)-1}:
                target=f'preview_{row["index"]:06d}.jpg'; canvas.save(output/target,quality=92)
                screenshots.append(target)
        concat.extend([f"file '{len(rows)-1:06d}.png'",'option framerate 1000000'])
        (tmp/'frames.ffconcat').write_text('\n'.join(concat)+'\n')
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-safe','0','-f','concat',
            '-i',str(tmp/'frames.ffconcat'),'-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',*sync,
            '-enc_time_base','1:1000000','-video_track_timescale','1000000','-movflags','+faststart',
            str(output/'comparison.mp4')],check=True)
    write_json(output/'preview.json',{'status':'complete','scene_key':entity_manifest['scene_key'],
        'render_index_sha256':sha256(Path(args.render_index)),'entities_sha256':sha256(Path(args.entities)),
        'frame_indices':[r['index'] for r in rows],'source_timestamps_seconds':times,'duration_seconds':float(sum(durations)),
        'display_size':[width*3,height+48],'mask_resolution':[index['width'],index['height']],
        'sam_present':bool(sam_masks),'reviewed_sam_entity_count':len(reviewed),
        'interpretation':'RGB/ID visualization only; source timing retained between selected frames. No interpolated geometry or fabricated SAM masks.',
        'endpoint_duplicate':'Last display frame repeated at clip endpoint for one microsecond.'})
    cells=[]
    for e in entities:
        color=','.join(map(str,colors[e['id']]))
        cells.append(f'<tr><td style="background:rgb({color});color:#111">{e["id"]}</td><td>{html.escape(e["label"])}</td><td>{html.escape(e["name"])}</td><td>{html.escape(e["grouping_status"])}</td></tr>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Entity render comparison</title>'
        '<style>body{font:15px system-ui;background:#131923;color:#eee;max-width:1200px;margin:30px auto;padding:15px}video,img{max-width:100%}td,th{padding:7px;text-align:left;border-bottom:1px solid #455}a{color:#8cd}</style>'
        f'<h1>{html.escape(entity_manifest["scene_key"])} · entity comparison</h1>'
        f'<p>{len(rows)} shared source timestamps. Full-resolution masks: {index["width"]} × {index["height"]}. '
        +('Reviewed SAM masks are tinted on the original; overlapping assignments are white.' if sam_masks else 'SAM has not run yet; the tint shows Blender entity IDs only.')
        +'</p><p>Original video · authored Blender RGB · Blender entity tint. Colors identify entities within this scene only. '
        f'Hard ID edges are not antialiased; RGB uses {index["rendering"]["samples"]} samples.</p>'
        '<video controls preload="metadata" src="comparison.mp4"></video><p><a href="comparison.mp4">Open video</a></p>'
        +''.join(f'<img src="{p}" alt="Timestamped entity preview">' for p in screenshots)
        +'<h2>Entity colors</h2><table><tr><th>ID</th><th>Entity</th><th>Stable name</th><th>Grouping status</th></tr>'
        +''.join(cells)+'</table>')
    print(f'Entity preview: {output/"index.html"}',flush=True)
