"""Automatic structural-plane alignment; camera correspondences are evaluation only.

Target naming profiles describe supported scene layouts. Source detection and
hypothesis ranking use geometry only, never an existing camera-derived fit.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .alignment import transform_points, validate_transform
from .common import sha256, write_json


def dot(points, normal):
    return np.einsum("...j,j->...", points, normal)


def sample(values, count):
    values = np.asarray(values)
    return values if len(values) <= count else values[np.linspace(0, len(values)-1, count).astype(int)]


def merge_fragments(extraction):
    """Consolidate nearby RANSAC remnants without merging separated planes."""
    points = np.asarray(extraction['points'])
    threshold = extraction['diagnostics']['distance_threshold']
    groups = []
    for i, p in enumerate(extraction['planes']):
        for group in groups:
            neighbors = [extraction['planes'][j] for j in group]
            distances = [max(abs(dot(p['centroid'], q['normal'])-q['offset']),
                             abs(dot(q['centroid'], p['normal'])-p['offset'])) for q in neighbors]
            angles = [abs(dot(p['normal'], q['normal'])) for q in neighbors]
            if (min(angles) >= np.cos(np.deg2rad(3)) and min(distances) < 3*threshold
                    and max(distances) < 6*threshold):
                union = points[np.unique(np.concatenate([p['indices']] + [q['indices'] for q in neighbors]))]
                _, _, axes = np.linalg.svd(union-union.mean(axis=0), full_matrices=False)
                residual = np.abs(dot(union-union.mean(axis=0), axes[-1]))
                if np.sqrt(np.mean(residual**2)) > 2*threshold or np.percentile(residual, 95) > 4*threshold:
                    continue
                group.append(i)
                break
        else:
            groups.append([i])
    planes = []
    for group in groups:
        indices = np.unique(np.concatenate([extraction['planes'][i]['indices'] for i in group]))
        p = points[indices]; center = p.mean(axis=0)
        _, singular, axes = np.linalg.svd(p-center, full_matrices=False)
        n = axes[-1]
        if n[np.argmax(np.abs(n))] < 0:
            n = -n
        xy = np.einsum('ij,nj->ni', axes[:2], p-center)
        widths = np.percentile(xy, 95, axis=0)-np.percentile(xy, 5, axis=0)
        frames = extraction.get('frame_indices')
        planes.append({'normal': n.tolist(), 'offset': float(n @ center), 'centroid': center.tolist(),
                       'indices': indices, 'count': len(indices), 'area': float(np.prod(widths)),
                       'rms': float(singular[-1]/np.sqrt(len(p))), 'raw_plane_indices': group,
                       'frame_count': None if frames is None else len(np.unique(frames[indices]))})
    return {**extraction, 'planes': planes, 'diagnostics': {**extraction['diagnostics'],
            'raw_plane_count': len(extraction['planes']), 'merged_plane_count': len(planes),
            'fragment_merge_angle_degrees': 3, 'fragment_merge_distance': 3*threshold,
            'fragment_merge_max_pair_distance': 6*threshold,
            'fragment_merge_union_rms_limit': 2*threshold}}


def solve_planes(normals_source, offsets_source, normals_target, offsets_target):
    """Initialize a proper similarity from oriented corresponding planes."""
    a, b = np.asarray(normals_source, float), np.asarray(normals_target, float)
    av, bv = np.asarray(offsets_source, float), np.asarray(offsets_target, float)
    if (a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) < 4
            or av.shape != (len(a),) or bv.shape != av.shape
            or not all(np.isfinite(x).all() for x in (a, b, av, bv))
            or not np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-6)
            or not np.allclose(np.linalg.norm(b, axis=1), 1, atol=1e-6)):
        raise ValueError("At least four paired, finite, oriented unit planes are required")
    if np.linalg.matrix_rank(b, tol=1e-5) < 3:
        raise ValueError("Plane normals do not constrain all translation directions")
    u, _, vt = np.linalg.svd(np.einsum('ni,nj->ij', b, a))
    r = (u * [1, 1, np.linalg.det(u @ vt)]) @ vt
    source_origin = np.linalg.lstsq(a, av, rcond=None)[0]
    target_origin = np.linalg.lstsq(b, bv, rcond=None)[0]
    centered_source = av - np.einsum('ij,j->i', a, source_origin)
    centered_target = bv - np.einsum('ij,j->i', b, target_origin)
    numerical_floor = max(1e-10, np.finfo(float).eps*max(1., np.linalg.norm(av))*100)
    if np.linalg.norm(centered_source) <= numerical_floor:
        raise ValueError("Concurrent planes leave scale unconstrained")
    matrix = np.column_stack((centered_source, b))
    norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix / np.maximum(norms, 1e-15)
    singular = np.linalg.svd(normalized, compute_uv=False)
    if singular[-1] < singular[0] * 1e-4:
        raise ValueError("Plane offsets leave scale/translation unconstrained or ill-conditioned")
    solution = np.linalg.lstsq(matrix, centered_target, rcond=None)[0]
    translation = solution[1:] + target_origin - solution[0]*(r @ source_origin)
    result = validate_transform({'scale': solution[0], 'rotation': r, 'translation': translation})
    return result


def rotation_vector(v):
    angle = float(np.linalg.norm(v))
    if angle < 1e-14:
        return np.eye(3)
    x, y, z = v / angle
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*k + (1-np.cos(angle))*(k @ k)


def refine_planes(patches, target_planes, initial, delta, max_iterations=30):
    """Balanced robust point-to-plane Gauss-Newton, with a rank check."""
    if delta <= 0 or not np.isfinite(delta):
        raise ValueError("Refinement robust distance must be positive")
    patches = [sample(p, 2000).astype(float) for p in patches]
    if len(patches) != len(target_planes) or any(len(p) < 3 for p in patches):
        raise ValueError("Every target needs a nonempty source patch")
    origin = np.mean([p.mean(axis=0) for p in patches], axis=0)
    points = np.concatenate(patches) - origin
    normals = np.concatenate([np.tile(t['normal'], (len(p), 1)) for p, t in zip(patches, target_planes)])
    offsets = np.concatenate([np.full(len(p), t['offset']) for p, t in zip(patches, target_planes)])
    balance = np.concatenate([np.full(len(p), 1/len(p)) for p in patches])
    s, r = initial['scale'], np.asarray(initial['rotation'])
    t = np.asarray(initial['translation']) + s*r @ origin

    def residual(scale, rotation, translation):
        q = scale*np.einsum('ij,nj->ni', rotation, points)
        return np.sum(normals*(q+translation), axis=1)-offsets, q

    def loss(e):
        z = np.abs(e)
        return float(np.sum(balance*np.where(z <= delta, .5*z*z, delta*(z-.5*delta))))

    for iteration in range(max_iterations):
        e, q = residual(s, r, t)
        jacobian = np.column_stack((np.cross(q, normals), np.sum(normals*q, axis=1), normals))
        weight = balance*np.minimum(1, delta/np.maximum(np.abs(e), 1e-15))
        j = jacobian*np.sqrt(weight[:, None])
        step = np.linalg.lstsq(j, -e*np.sqrt(weight), rcond=None)[0]
        if np.linalg.norm(step) < 1e-10:
            break
        old_loss = loss(e)
        accepted = False
        for fraction in (1, .5, .25, .125, .0625):
            ds = float(np.clip(fraction*step[3], -.3, .3))
            ns, nr, nt = s*np.exp(ds), rotation_vector(fraction*step[:3]) @ r, t+fraction*step[4:]
            if loss(residual(ns, nr, nt)[0]) < old_loss:
                s, r, t = ns, nr, nt
                accepted = True
                break
        if not accepted:
            break
    e, q = residual(s, r, t)
    j = np.column_stack((np.cross(q, normals), np.sum(normals*q, axis=1), normals))*np.sqrt(balance[:, None])
    norms = np.linalg.norm(j, axis=0)
    singular = np.linalg.svd(j/np.maximum(norms, 1e-15), compute_uv=False)
    rank = int(np.sum(singular > singular[0]*1e-5))
    if rank < 7:
        raise ValueError("Surface fit does not constrain all seven similarity parameters")
    result = validate_transform({'scale': s, 'rotation': r, 'translation': t-s*r @ origin})
    return result, {'jacobian_rank': rank, 'normalized_jacobian_condition': float(singular[0]/singular[-1]),
                    'iterations': iteration+1, 'huber_delta_blender_units': delta,
                    'balanced_robust_loss': loss(e), 'optimization_points': len(points)}


def plane_basis(normal):
    n = np.asarray(normal)
    axis = np.eye(3)[np.argmin(np.abs(n))]
    u = np.cross(n, axis); u /= np.linalg.norm(u)
    return np.array([u, np.cross(n, u)])


def finite_support(points, target, tolerance):
    """One-sided coverage by actual target triangles, ignoring unseen extensions."""
    basis = plane_basis(target['normal'])
    xy = np.einsum('ij,nj->ni', basis, points)
    triangles = np.einsum('ij,ntj->nti', basis, np.asarray(target['triangles']))
    inside = np.zeros(len(points), dtype=bool)
    # Half-plane tests with a geometric edge tolerance, in bounded batches.
    for start in range(0, len(triangles), 128):
        tri = triangles[start:start+128]
        edges = np.roll(tri, -1, axis=1)-tri
        signed = edges[:, 0, 0]*edges[:, 1, 1]-edges[:, 0, 1]*edges[:, 1, 0]
        nondegenerate = np.abs(signed) > 1e-12
        orientation = np.where(signed >= 0, 1., -1.)
        diff = xy[:, None, None, :]-tri[None]
        cross = edges[None, :, :, 0]*diff[:, :, :, 1]-edges[None, :, :, 1]*diff[:, :, :, 0]
        valid = np.all(cross*orientation[None, :, None] >= -tolerance*np.linalg.norm(edges, axis=2)[None], axis=2)
        inside |= np.any(valid & nondegenerate[None], axis=1)
    return inside


def _bounds_support(points, target, tolerance):
    basis = plane_basis(target['normal'])
    xy = np.einsum('ij,nj->ni', basis, points)
    target_xy = np.einsum('ij,nj->ni', basis, np.asarray(target['triangles']).reshape(-1, 3))
    low, high = target_xy.min(axis=0), target_xy.max(axis=0)
    outside = np.maximum(np.maximum(low-xy, xy-high), 0)
    return np.linalg.norm(outside, axis=1) <= tolerance


def patch_widths(points, normal):
    xy = np.einsum('ij,nj->ni', plane_basis(normal), points)
    xy -= xy.mean(axis=0)
    _, _, axes = np.linalg.svd(xy, full_matrices=False)
    principal = np.einsum('ij,nj->ni', axes, xy)
    return np.percentile(principal, 95, axis=0)-np.percentile(principal, 5, axis=0)


def score_candidate(transform, patches, targets, room_points, height, exact=False):
    coverages, errors, supports = [], [], []
    for p, target in zip(patches, targets):
        q = transform_points(sample(p, 350), transform)
        coverage = (finite_support if exact else _bounds_support)(q, target, height*.04)
        coverages.append(float(coverage.mean()))
        supported = q[coverage]
        if len(supported) >= 5:
            widths = patch_widths(supported, target['normal'])
            supports.append(float(min(1, np.prod(widths)/(height*height*.04), len(supported)/30)))
        else:
            supports.append(0.)
        errors.append(float(np.median(np.abs(dot(q, target['normal'])-target['offset'])))/height)
    q = transform_points(room_points, transform)
    # Interior-side tests use mesh-facing normals; no camera centers enter them.
    interior = [float(np.mean(dot(q, targets[i]['normal']) >= targets[i]['offset']-height*.05)) for i in (0, 2, 3)]
    score = float(np.mean(supports) + min(supports) + np.mean(interior) + .1*np.mean(coverages) - 4*np.mean(errors))
    return {'score': score, 'coverage': coverages, 'median_residual_fraction_height': errors,
            'support_quality': supports, 'interior_fraction_floor_back_side': interior, 'finite_triangles': exact}


def pool_patch(extraction, target, transform, height):
    """Pool nearby parallel detections, independent of a RANSAC fragment ID.

    Plane identities propose a placement. The final support is reassociated
    geometrically so an arbitrary detector split cannot choose the scale.
    """
    source_normal = np.asarray(transform['rotation']).T @ target['normal']
    plane_ids = [i for i, p in enumerate(extraction['planes'])
                 if abs(dot(p['normal'], source_normal)) >= np.cos(np.deg2rad(8))]
    if not plane_ids:
        return np.empty(0, int), np.empty(0, int), []
    ids = np.unique(np.concatenate([extraction['planes'][i]['indices'] for i in plane_ids]))
    q = transform_points(np.asarray(extraction['points'])[ids], transform)
    distance = np.abs(dot(q, target['normal'])-target['offset'])
    gate = max(height*.05, transform['scale']*extraction['diagnostics']['distance_threshold']*6)
    close = distance <= gate
    pool = ids[close]
    kept = pool[finite_support(q[close], target, height*.04)]
    contributing = [i for i in plane_ids if np.intersect1d(kept, extraction['planes'][i]['indices']).size]
    return kept, pool, contributing


def match_surfaces(extraction, target_bundle):
    """Enumerate floor/raised-surface/orthogonal-wall identities, signs and scale."""
    planes, points = extraction['planes'], np.asarray(extraction['points'])
    targets = target_bundle['planes']
    if [p['name'] for p in targets] != ['floor', 'counter', 'back', 'side']:
        raise ValueError("Structural matching needs ordered floor, counter (raised surface), back, side roles")
    b = np.asarray([p['normal'] for p in targets]); bd = np.asarray([p['offset'] for p in targets])
    height = float(bd[1]-bd[0])
    if height <= 0:
        raise ValueError("Raised surface must be above floor with consistently upward normals")
    normals = np.asarray([p['normal'] for p in planes])
    centers = np.asarray([p['centroid'] for p in planes])
    extent = extraction['diagnostics']['robust_extent']
    room = sample(points, 2000)
    candidates = []
    tested = 0
    for floor, counter in itertools.permutations(range(len(planes)), 2):
        if abs(dot(normals[counter], normals[floor])) < .985:
            continue
        direction = dot(centers[counter]-centers[floor], normals[floor])
        if abs(direction) < extent*.05:
            continue
        up = normals[floor]*np.sign(direction)
        verticals = [i for i in range(len(planes)) if i not in (floor, counter) and abs(dot(normals[i], up)) < .12]
        for back, side in itertools.permutations(verticals, 2):
            if abs(dot(normals[back], normals[side])) > .12:
                continue
            ids = (floor, counter, back, side)
            patches = [points[planes[i]['indices']] for i in ids]
            for back_sign, side_sign in itertools.product((-1, 1), repeat=2):
                signs = np.array([np.sign(direction), np.sign(dot(normals[counter], up)), back_sign, side_sign])
                a = normals[list(ids)]*signs[:, None]
                # Only proper axis correspondences can map to target normals.
                if np.linalg.det(a[[0, 2, 3]])*np.linalg.det(b[[0, 2, 3]]) <= 0:
                    continue
                ad = np.array([planes[i]['offset'] for i in ids])*signs
                try:
                    transform = solve_planes(a, ad, b, bd)
                except ValueError:
                    continue
                tested += 1
                score = score_candidate(transform, patches, targets, room, height)
                candidates.append({'transform': transform, 'plane_indices': list(ids), 'signs': signs.tolist(), **score})
    if not candidates:
        raise ValueError("No full-rank floor/raised-surface/two-wall hypothesis. Inspect source planes or supply surface labels.")
    candidates.sort(key=lambda c: -c['score'])
    # Diversify finalists so repeated nearby hypotheses cannot hide an alternative.
    comparison = sample(points, 500)
    diverse, diverse_q = [], []
    for candidate in candidates:
        q = transform_points(comparison, candidate['transform'])
        if any(np.sqrt(np.mean(np.sum((q-other)**2, axis=1))) < height*.04 for other in diverse_q):
            continue
        diverse.append(candidate); diverse_q.append(q)
        if len(diverse) == 32:
            break
    # Restrict refinement to the observed part of each named target face.
    finalists = []
    for candidate in diverse:
        patches = [points[planes[i]['indices']] for i in candidate['plane_indices']]
        transform = candidate['transform']
        try:
            for _ in range(5):
                pooled = [pool_patch(extraction, t, transform, height) for t in targets]
                selected = [points[p[0]] for p in pooled]
                if any(len(p) < 30 for p in selected):
                    raise ValueError('Not enough points on finite anchor faces')
                transform, diagnostics = refine_planes(selected, targets, transform, height*.015)
        except ValueError:
            continue
        patches = [points[pool_patch(extraction, t, transform, height)[1]] for t in targets]
        score = score_candidate(transform, patches, targets, room, height, exact=True)
        finalists.append({**candidate, 'transform': transform, **score, 'refinement': diagnostics})
    if not finalists:
        raise ValueError('No candidate has enough finite surface support for a full-rank fit')
    finalists.sort(key=lambda c: -c['score'])
    eligible = [c for c in finalists if min(c['interior_fraction_floor_back_side']) >= .90 and min(c['support_quality']) >= .8]
    best = (eligible or finalists)[0]
    # Near-identical patch fragments are not independent competing alignments.
    comparison = sample(points, 1000)
    best_q = transform_points(comparison, best['transform'])
    alternatives = []
    for candidate in (eligible or finalists)[1:]:
        difference = float(np.sqrt(np.mean(np.sum((transform_points(comparison, candidate['transform'])-best_q)**2, axis=1))))
        candidate['displacement_from_best_blender_units'] = difference
        if difference > height*.08:
            alternatives.append(candidate)
    margin = None if not alternatives else float(best['score']-alternatives[0]['score'])
    issues = []
    if min(best['support_quality']) < .8:
        issues.append('At least one anchor has limited finite-face spatial support.')
    if min(best['interior_fraction_floor_back_side']) < .90:
        issues.append('More than 10% of sampled geometry falls outside an anchor room half-space.')
    if margin is not None and margin < .12:
        issues.append('A materially different surface assignment has a similar geometric score.')
    if best['refinement']['normalized_jacobian_condition'] > 100:
        issues.append('The seven-parameter surface fit is weakly conditioned.')
    labels = np.full(len(points), -1, dtype=np.int32)
    anchors = []
    for j, (i, target) in enumerate(zip(best['plane_indices'], targets)):
        display_name = target.get('display_name', target['name'])
        indices, pool_indices, pooled_planes = pool_patch(extraction, target, best['transform'], height)
        q = transform_points(points[indices], best['transform'])
        labels[indices] = j
        errors = np.abs(dot(q, target['normal'])-target['offset'])
        widths = patch_widths(q, target['normal'])
        frames = extraction.get('frame_indices')
        frame_count = None if frames is None else int(len(np.unique(frames[indices])))
        if min(widths) < height*.05 or np.prod(widths) < height*height*.01 or (frame_count is not None and frame_count < 3):
            issues.append(f"The {display_name} anchor has weak spatial or temporal support.")
        if np.sqrt(np.mean(errors**2)) > height*.025 or np.percentile(errors, 95) > height*.05:
            issues.append(f"The {display_name} anchor has large point-to-plane residuals.")
        anchors.append({'name': target['name'], 'display_name': target.get('display_name', target['name']),
                        'source_plane_index': i,
                        'pooled_source_plane_indices': pooled_planes,
                        'raw_source_plane_indices': sorted(set(k for z in pooled_planes for k in planes[z].get('raw_plane_indices', [z]))),
                        'candidate_pool_points': len(pool_indices),
                        'source_normal': planes[i]['normal'], 'source_offset': planes[i]['offset'],
                        'target_normal': target['normal'], 'target_offset': target['offset'],
                        'point_count': len(indices), 'frame_count': frame_count,
                        'widths_90_percent_blender_units': widths.tolist(),
                        'rms': float(np.sqrt(np.mean(errors**2))), 'median': float(np.median(errors)),
                        'p95': float(np.percentile(errors, 95)), 'coverage': best['coverage'][j],
                        'target_objects': target['object_names']})
    warnings = [
        'Automatic structural-plane correspondence is a geometric hypothesis, not verified semantic labeling.',
        f"Target object-name matching uses profile {target_bundle.get('profile', 'explicit_planes')}; it is not a universal scene recognizer.",
        'Anchor surfaces and floor-to-raised-surface scale are calibration inputs, not independent accuracy evidence.',
        'Camera correspondences were not used for detection, initialization, selection, or refinement.',
        'Finite support is one-sided: unobserved Blender surface extensions are not penalized.',
        'Other cloud geometry influenced hypothesis selection; it is not a strictly held-out object test.',
        'Ambiguity diagnostics are conditional on the enumerated hypotheses and diversified finalists.',
    ] + issues
    report = {'schema_version': 1, 'status': 'needs_review' if issues else 'automatic_candidate',
              'transform': best['transform'], 'anchors': anchors,
              'target_profile': target_bundle.get('profile', 'explicit_planes'),
              'diagnostics': {**best['refinement'], 'candidate_count': tested, 'finalist_count': len(finalists),
                              'best_score': best['score'], 'distinct_runner_up_margin': margin,
                              'eligible_finalist_count': len(eligible),
                              'support_reassociation': 'nearby parallel plane detections, point distance gate, finite target faces; five refinements',
                              'support_distance_gate': 'max(0.05*raised_surface_height, 6*scale*source_plane_threshold)',
                              'selection_gates': {'minimum_room_interior_fraction': .90, 'minimum_spatial_support_quality': .8,
                                  'maximum_anchor_rms_fraction_height': .025, 'maximum_anchor_p95_fraction_height': .05,
                                  'minimum_distinct_candidate_margin': .12},
                              'interior_fraction_floor_back_side': best['interior_fraction_floor_back_side'],
                              'uses_camera_correspondences': False, 'source_extraction': extraction['diagnostics'],
                              'quality_issues': issues},
              'candidates': [{**{k: v for k, v in c.items() if k != 'transform'},
                              'eligible_room_and_support': any(c is x for x in eligible)} for c in finalists[:8]],
              'warnings': warnings}
    return report, labels


def run(args):
    from . import alignment, alignment_viewer
    from .surface_planes import extract_planes
    from .surface_targets import extract_target_planes
    from .surface_report import write_report
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Surface alignment output must be empty: {output}")
    stability_runs = getattr(args, 'stability_runs', 2)
    if not isinstance(stability_runs, int) or not 1 <= stability_runs <= 8:
        raise ValueError('stability_runs must be between 1 and 8')
    if args.viewer_points < 0:
        raise ValueError('viewer_points must be nonnegative')
    # Validate paired bundles/timing before expensive fitting; never use positions.
    vggt, blender = alignment._load_cameras(args.vggt, 'VGGT'), alignment._load_cameras(args.blender_export, 'Blender')
    if vggt['manifest_sha256'] != blender['manifest_sha256']:
        raise ValueError('Source manifests differ')
    print('Detecting source planes without camera correspondence inputs...', flush=True)
    cloud = alignment.load_vggt_ply(Path(args.vggt)/'point_cloud.ply', args.sample_points)
    extraction = extract_planes(cloud['points'], cloud['frame_indices'],
                                voxel_size=args.voxel_size, distance_threshold=args.plane_threshold,
                                max_planes=args.max_planes, seed=args.seed)
    extraction = merge_fragments(extraction)
    print(f"Detected {len(extraction['planes'])} planes from {len(extraction['points'])} spatially balanced points.", flush=True)
    targets = extract_target_planes(args.blender_export, profile=getattr(args, 'surface_profile', 'auto'))
    report, labels = match_surfaces(extraction, targets)
    height = targets['planes'][1]['offset']-targets['planes'][0]['offset']
    comparison = sample(cloud['points'], 2000)
    base_points = transform_points(comparison, report['transform'])
    stability = {'runs': stability_runs, 'checked': stability_runs > 1,
                 'thresholds': {'scale_fraction': .01, 'rotation_degrees': 1.,
                                'cloud_rms_fraction_counter_height': .03}, 'repeats': []}
    for repeat in range(1, stability_runs):
        seed = args.seed + repeat
        print(f'Checking surface-fit stability with detector seed {seed}...', flush=True)
        try:
            other_extraction = merge_fragments(extract_planes(cloud['points'], cloud['frame_indices'],
                voxel_size=args.voxel_size, distance_threshold=args.plane_threshold, max_planes=args.max_planes, seed=seed))
            other, _ = match_surfaces(other_extraction, targets)
            scale_change = abs(other['transform']['scale']/report['transform']['scale']-1)
            r0, r1 = np.asarray(report['transform']['rotation']), np.asarray(other['transform']['rotation'])
            angle = float(np.rad2deg(np.arccos(np.clip((np.trace(r0.T @ r1)-1)/2, -1, 1))))
            rms = float(np.sqrt(np.mean(np.sum((transform_points(comparison, other['transform'])-base_points)**2, axis=1))))
            transform_stable = scale_change <= .01 and angle <= 1 and rms <= height*.03
            quality_passed = other['status'] == 'automatic_candidate'
            passed = transform_stable and quality_passed
            result = {'seed': seed, 'status': other['status'], 'passed': bool(passed),
                      'transform_stable': bool(transform_stable), 'quality_checks_passed': quality_passed,
                      'scale_relative_change': scale_change, 'rotation_degrees': angle,
                      'cloud_displacement_rms_blender_units': rms, 'transform': other['transform'],
                      'anchors': [{k: a[k] for k in ('name', 'display_name', 'source_plane_index',
                          'pooled_source_plane_indices', 'raw_source_plane_indices')} for a in other['anchors']],
                      'quality_issues': other['diagnostics']['quality_issues']}
        except ValueError as error:
            result = {'seed': seed, 'passed': False, 'error': str(error)}
        stability['repeats'].append(result)
    stability['passed'] = None if stability_runs == 1 else all(x['passed'] for x in stability['repeats'])
    stability['transform_stable'] = None if stability_runs == 1 else all(x.get('transform_stable', False) for x in stability['repeats'])
    stability['repeat_quality_checks_passed'] = None if stability_runs == 1 else all(x.get('quality_checks_passed', False) for x in stability['repeats'])
    report['diagnostics']['stability'] = stability
    if stability['passed'] is False:
        issue = ('Repeated detections produced stable transforms, but a repeat failed geometric quality checks.'
                 if stability['transform_stable'] else
                 'Independent detector seeds did not produce a sufficiently stable surface transform.')
        report['status'] = 'needs_review'
        report['diagnostics']['quality_issues'].append(issue)
        report['warnings'].append(issue)
    if stability_runs == 1:
        report['warnings'].append('Detector-seed stability was not checked (--stability-runs 1).')
    report['inputs'] = {'vggt_point_cloud_sha256': sha256(Path(args.vggt)/'point_cloud.ply'),
                        'blender_geometry_sha256': sha256(Path(args.blender_export)/'geometry.npz')}
    report['settings'] = {k: getattr(args, k) for k in ('sample_points', 'voxel_size', 'plane_threshold', 'max_planes', 'seed')}
    report['settings']['stability_runs'] = stability_runs
    report['settings']['surface_profile'] = getattr(args, 'surface_profile', 'auto')
    result_path = alignment.run(SimpleNamespace(vggt=args.vggt, blender_export=args.blender_export,
                                                output=output, surface_fit=report))
    result = json.loads(result_path.read_text())
    report['camera_metrics'] = result['camera_metrics']
    write_json(output/'surface_fit.json', report)
    write_json(output/'target_planes.json', targets)
    metadata = {'diagnostics': extraction['diagnostics'], 'planes': [
        {k: v for k, v in p.items() if k != 'indices'} for p in extraction['planes']]}
    write_json(output/'source_planes.json', metadata)
    np.savez_compressed(output/'surface_points.npz', points=extraction['points'], labels=labels,
                        frame_indices=extraction['frame_indices'],
                        **{f'plane_{i}_indices': p['indices'] for i, p in enumerate(extraction['planes'])})
    write_report(output/'index.html', report, extraction['points'], labels, targets)
    if args.viewer_points:
        alignment_viewer.run(SimpleNamespace(blender_export=args.blender_export, vggt=args.vggt,
            alignment=result_path, manifest=args.manifest, output=output/'viewer.html',
            max_points=args.viewer_points, max_triangles=30000, max_images=36))
    print(f"Surface alignment: {report['status']}; report: {output/'index.html'}", flush=True)
    return output/'surface_fit.json'
