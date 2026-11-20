"""Independent, bounded grid-discovery/video-tracking experiment.

No Blender inputs are consumed. Quality-passing proposals are saved before
deduplication or tracking budgets. Independent seed tracks are reconciled only
after temporal propagation, retaining every original mask for later grouping.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image

from .common import load_manifest, sha256, write_json
from .dense_association import filter_candidates
from .sam3_dense_backend import UPSTREAM_COMMIT


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _positive(value, name, *, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return value


def make_plan(args):
    source_path = Path(args.manifest).resolve()
    source = load_manifest(source_path)
    if any(not math.isfinite(f['pts_seconds']) for f in source['frames']):
        raise ValueError('Source PTS timestamps must be finite')
    selected = source['frames']
    selection_path = Path(args.selection).resolve() if args.selection else None
    if selection_path:
        selection = json.loads(selection_path.read_text())
        if selection.get('source_manifest_sha256') != sha256(source_path):
            raise ValueError('Frame selection belongs to a different source manifest')
        indices = [f['index'] for f in selection['frames']]
        if not indices or indices != sorted(set(indices)) or any(type(i) is not int for i in indices):
            raise ValueError('Selected source frame IDs must be nonempty, unique and ordered')
        by_id = {f['index']: f for f in source['frames']}
        if not set(indices).issubset(by_id):
            raise ValueError('Selected frame is absent from source manifest')
        for frame in selection['frames']:
            for field in ('time_seconds', 'pts_seconds'):
                if not math.isfinite(frame[field]) or abs(frame[field] - by_id[frame['index']][field]) > 1e-6:
                    raise ValueError('Selected timestamps differ from source video')
        selected = [by_id[i] for i in indices]
    limit = _positive(args.frame_limit, 'frame_limit', allow_zero=True)
    if limit:
        selected = selected[:limit]
    count = _positive(args.anchor_count, 'anchor_count')
    # Anchor choices depend only on source time, never on Blender or text labels.
    times = np.array([f['time_seconds'] for f in selected])
    positions = sorted(set(int(np.argmin(abs(times - time))) for time in
                           np.linspace(times[0], times[-1], min(count, len(selected)))))
    parameters = {}
    for name in ('grid_side', 'point_batch_size', 'track_batch_size'):
        parameters[name] = _positive(getattr(args, name), name)
    parameters['max_per_anchor'] = _positive(args.max_per_anchor, 'max_per_anchor', allow_zero=True)
    parameters['min_mask_area'] = _positive(args.min_mask_area, 'min_mask_area')
    if args.crop_layers not in (0, 1):
        raise ValueError('crop_layers must be 0 or 1 for this pilot')
    parameters['crop_layers'] = args.crop_layers
    for name in ('min_quality', 'min_stability', 'dedup_iou'):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be in [0,1]')
        if name == 'dedup_iou' and value == 0:
            raise ValueError('dedup_iou must be greater than zero')
        parameters[name] = value
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    if checkpoint and not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    frames = []
    for frame in selected:
        path = (source_path.parent / frame['file']).resolve()
        with Image.open(path) as image:
            if image.size != (source['width'], source['height']):
                raise ValueError(f'Source image grid differs from manifest: {path}')
        frames.append({**frame, 'source_path': str(path), 'image_sha256': sha256(path)})
    identity = {'schema_version': 1, 'pipeline_version': 1, 'native_revision': UPSTREAM_COMMIT,
                'source_manifest_sha256': sha256(source_path),
                'selection_sha256': sha256(selection_path) if selection_path else None,
                'frame_ids': [f['index'] for f in frames],
                'image_sha256': [f['image_sha256'] for f in frames],
                'width': source['width'], 'height': source['height'],
                'anchors': [frames[i]['index'] for i in positions],
                'parameters': parameters,
                'checkpoint': sha256(checkpoint) if checkpoint else 'facebook/sam3:sam3.pt'}
    return {**identity, 'fingerprint': _fingerprint(identity), 'frames': frames,
            'source_manifest': str(source_path), 'checkpoint_path': str(checkpoint) if checkpoint else None,
            'scope': 'Unreviewed independent discovery/tracking pilot; no Blender inputs or completeness guarantee'}


def grid_points(width, height, side):
    x = (np.arange(side, dtype=np.float32) + .5) * width / side
    y = (np.arange(side, dtype=np.float32) + .5) * height / side
    return np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)


def crop_boxes(width, height, layers):
    boxes = [(0, 0, width, height)]
    if layers:
        # Overlapping quadrants retain context and magnify small objects.
        cw, ch = int(math.ceil(width * .6)), int(math.ceil(height * .6))
        boxes += [(x, y, x + cw, y + ch) for y in (0, height-ch) for x in (0, width-cw)]
    return list(dict.fromkeys(boxes))


def _read_complete(path, fingerprint):
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if value.get('fingerprint') != fingerprint:
        raise ValueError(f'Cached artifact has different inputs/settings: {path}')
    return value if value.get('status') == 'complete' else None


def _mask(path, shape):
    with Image.open(path) as image:
        value = np.asarray(image)
    if value.shape != shape or not np.isin(value, [0, 255]).all():
        raise ValueError(f'Invalid binary mask on source pixel grid: {path}')
    return value > 0


def _discover_anchor(backend, frame, plan, output):
    index = frame['index']
    destination = output / 'discovery' / f'{index:06d}'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'masks').mkdir(exist_ok=True)
    params = plan['parameters']
    width, height = plan['width'], plan['height']
    with Image.open(frame['source_path']) as image:
        rgb = np.asarray(image.convert('RGB'))
    candidates, records = [], []
    attempted, rejected_quality, rejected_edge = 0, 0, 0
    for crop_id, (x0, y0, x1, y1) in enumerate(crop_boxes(width, height, params['crop_layers'])):
        crop = rgb[y0:y1, x0:x1]
        points = grid_points(x1-x0, y1-y0, params['grid_side'])
        for proposal in backend.discover(crop, points, batch_size=params['point_batch_size']):
            attempted += 1
            mask = np.asarray(proposal['mask'])
            if mask.dtype != np.bool_ or mask.shape != (y1-y0, x1-x0):
                raise ValueError('Discovery must return Boolean masks on the supplied crop grid')
            score, stability = float(proposal['predicted_iou']), float(proposal['stability_score'])
            if not math.isfinite(score) or not math.isfinite(stability):
                raise ValueError('Discovery returned non-finite quality scores')
            if score < params['min_quality'] or stability < params['min_stability'] or np.count_nonzero(mask) < params['min_mask_area']:
                rejected_quality += 1
                continue
            # Reject masks cut by artificial crop edges; keep natural image edges.
            if ((x0 and mask[:, 0].any()) or (y0 and mask[0, :].any()) or
                    (x1 < width and mask[:, -1].any()) or (y1 < height and mask[-1, :].any())):
                rejected_edge += 1
                continue
            proposal_id = f'a{index:06d}_c{crop_id:02d}_p{int(proposal["point_index"]):05d}_m{int(proposal["alternative_index"])}'
            if any(r['proposal_id'] == proposal_id for r in records):
                raise ValueError(f'Discovery returned duplicate proposal identifier: {proposal_id}')
            full = np.zeros((height, width), dtype=bool)
            full[y0:y1, x0:x1] = mask
            relative = f'discovery/{index:06d}/masks/{proposal_id}.png'
            Image.fromarray(full.astype(np.uint8)*255).save(output / relative, compress_level=1)
            seed = [float(proposal['seed_xy'][0])+x0, float(proposal['seed_xy'][1])+y0]
            record = {'proposal_id': proposal_id, 'anchor_frame': index, 'mask': relative,
                      'score': score, 'stability': stability, 'seed_xy': seed,
                      'crop_xyxy': [x0, y0, x1, y1], 'area_pixels': int(full.sum())}
            records.append(record)
            candidates.append({**record, 'mask': full})
    filtered = filter_candidates(candidates, min_area=params['min_mask_area'],
                                 min_quality=params['min_quality'], min_stability=params['min_stability'],
                                 duplicate_iou=params['dedup_iou'], duplicate_area_ratio=.85)
    accepted = filtered['accepted']
    limit = params['max_per_anchor']
    chosen = accepted[:limit] if limit else accepted
    result = {'schema_version': 1, 'status': 'complete', 'fingerprint': plan['fingerprint'],
              'anchor_frame': index, 'proposal_hypotheses': attempted,
              'quality_rejected': rejected_quality, 'artificial_crop_edge_rejected': rejected_edge,
              'proposals': records, 'decisions': filtered['decisions'], 'part_whole_relations': filtered['relations'],
              'tracking_proposal_ids': [p['proposal_id'] for p in chosen],
              'budget_deferred_ids': [p['proposal_id'] for p in accepted[len(chosen):]],
              'raw_scope': 'Every quality-passing, non-crop-truncated proposal before deduplication and tracking budget'}
    write_json(destination / 'index.json', result)
    print(f'Discovery source {index}: {attempted} hypotheses; {len(records)} retained; '
          f'{len(accepted)} after dedup; {len(chosen)} tracking seeds; {len(accepted)-len(chosen)} budget-deferred', flush=True)
    return result


def _tracking_jobs(anchors, batch_size):
    jobs, next_id = [], 1
    for anchor in anchors:
        records = {p['proposal_id']: p for p in anchor['proposals']}
        chosen = [records[p] for p in anchor['tracking_proposal_ids']]
        for start in range(0, len(chosen), batch_size):
            seeds = []
            for proposal in chosen[start:start+batch_size]:
                seeds.append({**proposal, 'id': f'track_{next_id:06d}', 'object_id': next_id})
                next_id += 1
            jobs.append({'name': f'a{anchor["anchor_frame"]:06d}_b{start//batch_size:03d}', 'seeds': seeds})
    return jobs


def _track_job(backend, job, plan, output, resource_path):
    frames = plan['frames']
    frame_positions = {f['index']: i for i, f in enumerate(frames)}
    shape = plan['height'], plan['width']
    seeds = [{'object_id': p['object_id'], 'frame_index': frame_positions[p['anchor_frame']],
              'mask': _mask(output / p['mask'], shape)} for p in job['seeds']]
    observations = {p['object_id']: [] for p in job['seeds']}
    by_id = {p['object_id']: p for p in job['seeds']}
    for p in job['seeds']:
        (output / 'tracks' / p['id']).mkdir(parents=True, exist_ok=True)
    seen = set()
    for result in backend.track(resource_path, len(frames), seeds, plan['height'], plan['width']):
        position = result['frame_index']
        if type(position) is not int or position not in range(len(frames)) or position in seen:
            raise ValueError('Tracking returned duplicate or unknown local frame')
        seen.add(position)
        if set(result['masks']) != set(by_id):
            raise ValueError('Tracking omitted an object; missing output is not an empty mask')
        frame = frames[position]
        for object_id, raw_mask in result['masks'].items():
            mask = np.asarray(raw_mask)
            if mask.dtype != np.bool_ or mask.shape != shape:
                raise ValueError('Tracker mask must retain exact Boolean source pixel grid')
            track = by_id[object_id]
            relative = f'tracks/{track["id"]}/{frame["index"]:06d}.png'
            Image.fromarray(mask.astype(np.uint8)*255).save(output / relative, compress_level=1)
            observations[object_id].append({'index': frame['index'], 'time_seconds': frame['time_seconds'],
                'pts_seconds': frame['pts_seconds'], 'mask': relative, 'area_pixels': int(mask.sum()),
                'raw_tracker_score': result.get('scores', {}).get(object_id),
                'prediction_state': 'nonempty' if mask.any() else 'predicted_empty_not_verified_absent'})
    if seen != set(range(len(frames))):
        raise ValueError('Tracker did not cover every requested frame')
    tracks = []
    for p in job['seeds']:
        relative = f'tracks/{p["id"]}/index.json'
        metadata = {'schema_version': 1, 'status': 'complete', 'fingerprint': plan['fingerprint'],
                    'id': p['id'], 'proposal_id': p['proposal_id'], 'anchor_frame': p['anchor_frame'],
                    'seed_mask': p['mask'], 'seed_quality': p['score'], 'seed_stability': p['stability'],
                    'frames': sorted(observations[p['object_id']], key=lambda f: f['index']),
                    'mask_semantics': 'Independent tracker masks before cross-object non-overlap suppression',
                    'identity_status': 'Unreviewed seed track; catalog reconciliation follows propagation'}
        write_json(output / relative, metadata)
        tracks.append({k: metadata[k] for k in ('id', 'proposal_id', 'anchor_frame')})
        tracks[-1]['index_file'] = relative
    result = {'schema_version': 1, 'status': 'complete', 'fingerprint': plan['fingerprint'], 'tracks': tracks}
    write_json(output / 'tracking_batches' / f'{job["name"]}.json', result)
    return result


def run(args, backend_factory=None):
    started = time.perf_counter()
    plan = make_plan(args)
    if args.dry_run:
        print(json.dumps({k: v for k, v in plan.items() if k not in ('frames', 'image_sha256', 'frame_ids')}, indent=2))
        print(f'Will track {len(plan["frames"])} source frames; anchors {plan["anchors"]}; no Blender or text inputs.')
        return
    output = Path(args.output).resolve()
    previous_metadata = {}
    if output.exists() and any(output.iterdir()):
        if not args.resume:
            raise FileExistsError(f'Use a new output directory or --resume: {output}')
        previous = json.loads((output / 'plan.json').read_text())
        if previous['fingerprint'] != plan['fingerprint']:
            raise ValueError('Cannot resume with changed input images or settings; choose a new output directory')
        if (output / 'metadata.json').is_file():
            previous_metadata = json.loads((output / 'metadata.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'plan.json', plan)
    portable = [{k: v for k, v in f.items() if k not in ('source_path', 'image_sha256')} for f in plan['frames']]
    for record, frame in zip(portable, plan['frames']):
        record['file'] = Path(os.path.relpath(frame['source_path'], output)).as_posix()
    write_json(output / 'manifest.json', {'schema_version': 1, 'width': plan['width'], 'height': plan['height'],
        'frame_count': len(portable), 'source_manifest_sha256': plan['source_manifest_sha256'], 'frames': portable})
    metadata = {'schema_version': 1, 'status': 'running', 'fingerprint': plan['fingerprint'],
                'scope': plan['scope'], 'parameters': plan['parameters'],
                'frame_count': len(portable), 'width': plan['width'], 'height': plan['height']}
    if 'runtime' in previous_metadata:
        metadata['runtime'] = previous_metadata['runtime']
    write_json(output / 'metadata.json', metadata)
    backend = None
    def record_runtime():
        if backend and callable(getattr(backend, 'runtime_info', None)):
            metadata['runtime'] = backend.runtime_info()
    def get_backend():
        nonlocal backend
        if backend is None:
            factory = backend_factory
            if factory is None:
                from .sam3_dense_backend import Sam3DenseBackend
                factory = Sam3DenseBackend
            backend = factory(checkpoint=plan['checkpoint_path'], gpu_index=args.gpu_index)
        return backend
    try:
        anchors = []
        frames = {f['index']: f for f in plan['frames']}
        for index in plan['anchors']:
            cached = _read_complete(output / 'discovery' / f'{index:06d}' / 'index.json', plan['fingerprint'])
            if cached:
                for proposal in cached['proposals']:
                    if not (output / proposal['mask']).is_file():
                        raise FileNotFoundError(output / proposal['mask'])
                anchors.append(cached)
            else:
                if args.stage in ('track', 'catalog'):
                    raise ValueError('Discovery is incomplete; run --stage discover or all first')
                model = get_backend()
                model.load_discovery()
                anchors.append(_discover_anchor(model, frames[index], plan, output))
        metadata.update(discovery_anchors=len(anchors),
                        saved_proposals=sum(len(a['proposals']) for a in anchors),
                        budget_deferred_proposals=sum(len(a['budget_deferred_ids']) for a in anchors))
        if args.stage == 'discover':
            metadata['status'] = 'discovery_complete'
        else:
            if backend:
                backend.release_discovery()
            jobs = _tracking_jobs(anchors, plan['parameters']['track_batch_size'])
            all_tracks = []
            with tempfile.TemporaryDirectory(prefix='dense_sam_frames_') as temporary:
                for position, frame in enumerate(plan['frames']):
                    source = Path(frame['source_path'])
                    (Path(temporary) / f'{position:06d}{source.suffix.lower()}').symlink_to(source)
                for position, job in enumerate(jobs):
                    cached = _read_complete(output / 'tracking_batches' / f'{job["name"]}.json', plan['fingerprint'])
                    if cached:
                        for track in cached['tracks']:
                            track_index = json.loads((output / track['index_file']).read_text())
                            if track_index.get('status') != 'complete' or len(track_index['frames']) != len(portable):
                                raise ValueError('Cached track is incomplete')
                            for frame in track_index['frames']:
                                if not (output / frame['mask']).is_file():
                                    raise FileNotFoundError(output / frame['mask'])
                    else:
                        if args.stage == 'catalog':
                            raise ValueError('Tracking is incomplete; run --stage track or all first')
                        print(f'Tracking batch {position+1}/{len(jobs)}: {job["name"]}, {len(job["seeds"])} proposals', flush=True)
                        model = get_backend()
                        model.load_tracking()
                        cached = _track_job(model, job, plan, output, temporary)
                    all_tracks.extend(cached['tracks'])
            write_json(output / 'raw_tracks.json', {'schema_version': 1, 'status': 'complete',
                'fingerprint': plan['fingerprint'], 'width': plan['width'], 'height': plan['height'],
                'manifest': 'manifest.json', 'tracks': all_tracks})
            if backend:
                record_runtime()
                backend.close()
                backend = None
            metadata['raw_track_count'] = len(all_tracks)
            if args.stage in ('all', 'catalog'):
                from .dense_catalog import build_catalog
                catalog = build_catalog(output)
                metadata['canonical_track_count'] = len(catalog['tracks'])
                metadata['status'] = 'complete'
            else:
                metadata['status'] = 'tracking_complete'
        record_runtime()
        metadata['invocation_seconds'] = round(time.perf_counter() - started, 3)
        write_json(output / 'metadata.json', metadata)
        print(f'Dense SAM pilot: {metadata["status"]}: {output}', flush=True)
    except BaseException as error:
        record_runtime()
        metadata.update(status='failed', error=f'{type(error).__name__}: {error}')
        metadata['invocation_seconds'] = round(time.perf_counter() - started, 3)
        write_json(output / 'metadata.json', metadata)
        raise
    finally:
        if backend:
            backend.close()
