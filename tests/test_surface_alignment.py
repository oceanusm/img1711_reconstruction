"""Independent analytic tests for surface fitting and camera-only evaluation."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from scene_reconstruction.alignment import run as evaluate_alignment, validate_transform
from scene_reconstruction.surface_alignment import (
    finite_support, match_surfaces, merge_fragments, pool_patch, refine_planes, rotation_vector, solve_planes,
)


def transform_fixture(seed=7):
    rng = np.random.default_rng(seed)
    return validate_transform({'scale': rng.uniform(.5, 5),
                               'rotation': rotation_vector(rng.normal(size=3)),
                               'translation': rng.uniform(-2, 2, 3)})


def inverse_points(points, transform):
    return np.einsum('ni,ij->nj', np.asarray(points)-transform['translation'],
                     np.asarray(transform['rotation'])) / transform['scale']


def rectangle(a, b, c, d):
    return np.asarray([[a, b, c], [a, c, d]], dtype=float)


def kitchen_fixture(transform=None, distractors=False):
    """Asymmetric finite supports distinguish floor from counter and walls."""
    if transform is None:
        transform = transform_fixture()
    rng = np.random.default_rng(90)
    target = [
        {'name': 'floor', 'normal': [0, 0, 1], 'offset': 0.,
         'triangles': rectangle([0,-3,0], [4,-3,0], [4,0,0], [0,0,0])},
        {'name': 'counter', 'normal': [0, 0, 1], 'offset': 1.,
         'triangles': rectangle([0,-.8,1], [4,-.8,1], [4,0,1], [0,0,1])},
        {'name': 'back', 'normal': [0, -1, 0], 'offset': 0.,
         'triangles': rectangle([0,0,1], [4,0,1], [4,0,2.4], [0,0,2.4])},
        {'name': 'side', 'normal': [1, 0, 0], 'offset': 0.,
         'triangles': rectangle([0,-3,1], [0,0,1], [0,0,2.4], [0,-3,2.4])},
    ]
    xyz = []
    for entry in target:
        corners = entry['triangles'][[0,0,1],[0,1,2]]
        uv = rng.uniform(.03, .97, (180, 2))
        xyz.append(corners[0] + uv[:, :1]*(corners[1]-corners[0]) + uv[:, 1:]*(corners[2]-corners[0]))
        entry['object_names'] = [entry['name']]
        entry['triangles'] = entry['triangles'].tolist()
    source = [inverse_points(p, transform) for p in xyz]
    normals = np.einsum('ni,ij->nj', np.asarray([p['normal'] for p in target]), transform['rotation'])
    offsets = (np.asarray([p['offset'] for p in target])-
               np.einsum('ni,i->n', np.asarray([p['normal'] for p in target]), transform['translation'])) / transform['scale']
    planes = []
    position = 0
    for i, (p, n, d) in enumerate(zip(source, normals, offsets)):
        sign = -1 if i % 2 else 1
        planes.append({'normal': (sign*n).tolist(), 'offset': float(sign*d),
                       'indices': np.arange(position, position+len(p)), 'centroid': p.mean(axis=0).tolist(),
                       'frame_count': 12})
        position += len(p)
    if distractors:
        # A distant parallel vertical patch and an oblique patch are both real
        # candidates, but neither is part of the target's structural anchors.
        for center, n in [(np.array([7.,-6.,1.5]), np.array([0.,-1.,0.])),
                          (np.array([1.,-1.,1.2]), np.array([1.,1.,1.])/np.sqrt(3))]:
            u = np.cross(n, [1.,0,0]); u /= np.linalg.norm(u)
            v = np.cross(n, u)
            uv = rng.uniform(-.25,.25,(60,2))
            p = center + uv[:, :1]*u + uv[:, 1:]*v
            sp = inverse_points(p, transform)
            sn = np.einsum('i,ij->j', n, transform['rotation'])
            sd = float((n@center-n@np.asarray(transform['translation']))/transform['scale'])
            planes.append({'normal':sn.tolist(),'offset':sd,'indices':np.arange(position,position+len(sp)),
                           'centroid':sp.mean(axis=0).tolist(),'frame_count':6})
            source.append(sp);position += len(sp)
    cloud = np.concatenate(source)
    return {'points': cloud, 'planes': planes,
            'diagnostics': {'robust_extent': float(np.linalg.norm(np.ptp(cloud,axis=0))),
                            'distance_threshold': .001}}, {'planes': target}, xyz


class SurfaceSolverTests(unittest.TestCase):
    def test_cli_routes_quality_cloud_and_keeps_detection_separate_from_display(self):
        from scene_reconstruction.cli import parser
        args=parser().parse_args(['align-surfaces','--blender-export','example','--output','result'])
        self.assertEqual(args.module,'surface_alignment')
        self.assertEqual(args.vggt,Path('outputs/vggt_quality'))
        self.assertEqual(args.sample_points,300000)
        self.assertEqual(args.viewer_points,5000000)
        self.assertEqual(args.stability_runs,2)
        self.assertEqual(args.surface_profile,'auto')
        self.assertIsNone(args.voxel_size)
        self.assertIsNone(args.plane_threshold)

    def test_exact_recovery_for_random_similarities(self):
        nb = np.array([[0.,0,1], [0,0,1], [0,-1,0], [1,0,0]])
        db = np.array([0., 1., -.2, .1])
        for seed in range(10):
            expected = transform_fixture(seed)
            ns = np.einsum('ni,ij->nj', nb, expected['rotation'])
            ds = (db-np.einsum('ni,i->n', nb, expected['translation']))/expected['scale']
            actual = solve_planes(ns, ds, nb, db)
            np.testing.assert_allclose(actual['matrix'], expected['matrix'], atol=1e-11)

    def test_scale_degeneracy_and_parallel_normals_rejected(self):
        normals = np.vstack((np.eye(3), np.ones(3)/np.sqrt(3)))
        # All four planes pass through one point: translation and scale cannot
        # be distinguished, despite three independent normal directions.
        point = np.array([1.,2,3])
        offsets = np.einsum('ni,i->n',normals,point)
        with self.assertRaisesRegex(ValueError,'scale|translation|ill-conditioned'):
            solve_planes(normals, offsets, normals, offsets)
        with self.assertRaisesRegex(ValueError,'translation'):
            solve_planes(np.tile([0.,0,1],(4,1)), np.arange(4.),
                         np.tile([0.,0,1],(4,1)), np.arange(4.))
        with self.assertRaisesRegex(ValueError,'four'):
            solve_planes(np.eye(3), np.zeros(3), np.eye(3), np.zeros(3))

    def test_large_source_origin_does_not_create_false_degeneracy(self):
        normals = np.array([[0.,0,1], [0,0,1], [0,-1,0], [1,0,0]])
        target_offsets = np.array([0.,1,0,0])
        expected = transform_fixture(3)
        expected['translation'] = [1e6,-2e6,3e6]
        source_normals = np.einsum('ni,ij->nj',normals,expected['rotation'])
        source_offsets = (target_offsets-np.einsum('ni,i->n',normals,expected['translation']))/expected['scale']
        actual = solve_planes(source_normals,source_offsets,normals,target_offsets)
        np.testing.assert_allclose(actual['translation'],expected['translation'],rtol=1e-8,atol=.01)
        self.assertAlmostEqual(actual['scale'],expected['scale'],delta=1e-8)

    def test_robust_refinement_recovers_pose_with_outliers(self):
        expected = transform_fixture(4)
        extraction, target, target_patches = kitchen_fixture(expected)
        rng = np.random.default_rng(56)
        patches = []
        for patch in target_patches:
            noisy = patch + rng.normal(0,.001,patch.shape)
            outliers = rng.uniform([-1,-4,-1],[5,1,3],(20,3))
            patches.append(inverse_points(np.concatenate((noisy,outliers)), expected))
        initial = validate_transform({'scale':expected['scale']*1.07,
                                      'rotation':rotation_vector(np.array([.03,-.025,.015]))@np.asarray(expected['rotation']),
                                      'translation':np.asarray(expected['translation'])+[.03,-.02,.04]})
        fitted, diagnostic = refine_planes(patches,target['planes'],initial,.007)
        least_squares, _ = refine_planes(patches,target['planes'],initial,100.)
        self.assertEqual(diagnostic['jacobian_rank'],7)
        error = abs(fitted['scale']/expected['scale']-1)
        ordinary_error = abs(least_squares['scale']/expected['scale']-1)
        # Huber limits residual influence but is not a high-breakdown estimator
        # against arbitrary tangential leverage. Production patches are first
        # filtered by RANSAC and finite-face support; this is a stress test.
        self.assertLess(error,.03)
        self.assertLess(error,ordinary_error/5)
        np.testing.assert_allclose(fitted['rotation'],expected['rotation'],atol=.015)
        np.testing.assert_allclose(fitted['translation'],expected['translation'],atol=.025)

    def test_triangle_coverage_respects_hole_and_winding(self):
        # Four strips surround a square hole; its center lies inside the global
        # bounding box but outside every actual triangle.
        rectangles = [(0,0,3,1),(0,2,3,3),(0,1,1,2),(2,1,3,2)]
        triangles = np.concatenate([rectangle([x0,y0,0],[x1,y0,0],[x1,y1,0],[x0,y1,0])
                                    for x0,y0,x1,y1 in rectangles])
        target = {'normal':[0.,0,1], 'triangles':triangles}
        points = np.array([[.5,.5,0],[1.5,1.5,0],[2.5,1.5,0],[3.1,1.5,0],[1,1,0]])
        expected = [True,False,True,False,True]
        np.testing.assert_array_equal(finite_support(points,target,0),expected)
        target['triangles'] = triangles[:,::-1]
        np.testing.assert_array_equal(finite_support(points,target,0),expected)

    def test_zero_area_triangle_cannot_cover_entire_plane(self):
        target = {'normal':[0.,0,1], 'triangles':np.zeros((1,3,3))}
        points = np.array([[0.,0,0],[50,50,0]])
        np.testing.assert_array_equal(finite_support(points,target,0),[False,False])
        target['triangles'] = np.concatenate((target['triangles'],rectangle([0,0,0],[1,0,0],[1,1,0],[0,1,0])))
        np.testing.assert_array_equal(finite_support(points,target,0),[True,False])

    def test_fragments_merge_without_merging_separate_parallel_planes(self):
        extraction,_,_ = kitchen_fixture()
        original = extraction['planes'][0]
        first = dict(original);first['indices'] = original['indices'][:90]
        second = dict(original);second['indices'] = original['indices'][90:]
        for part in (first,second):
            part['centroid'] = extraction['points'][part['indices']].mean(axis=0).tolist()
        extraction['planes'] = [first]+extraction['planes'][1:]+[second]
        extraction['diagnostics']['distance_threshold'] = .001
        merged = merge_fragments(extraction)
        self.assertEqual(len(merged['planes']),4)
        self.assertEqual(merged['planes'][0]['raw_plane_indices'],[0,4])
        np.testing.assert_array_equal(merged['planes'][0]['indices'],original['indices'])
        self.assertGreater(abs(merged['planes'][1]['offset']-merged['planes'][0]['offset']),.01)

    def test_matcher_handles_normal_signs_and_distractors_without_cameras(self):
        expected = transform_fixture(2)
        extraction,target,_ = kitchen_fixture(expected,distractors=True)
        report, labels = match_surfaces(extraction,target)
        np.testing.assert_allclose(report['transform']['matrix'],expected['matrix'],atol=1e-9)
        self.assertFalse(report['diagnostics']['uses_camera_correspondences'])
        self.assertEqual([a['source_plane_index'] for a in report['anchors']],[0,1,2,3])
        self.assertEqual(set(labels),{-1,0,1,2,3})

    def test_parallel_fragment_proposals_share_final_pooled_support(self):
        expected=transform_fixture(3)
        extraction,target,_=kitchen_fixture(expected)
        original=extraction['planes'][1]
        fragments=[]
        # Two detector fragments represent opposite sides of a small depth
        # error on one countertop, not two different physical surfaces.
        for ids,height_error in zip(np.array_split(original['indices'],2),[.006,-.006]):
            delta=np.asarray(expected['rotation'])[2]*height_error/expected['scale']
            extraction['points'][ids] += delta
            fragments.append({**original,'indices':ids,
                              'centroid':extraction['points'][ids].mean(axis=0).tolist(),
                              'offset':float(original['offset']+np.dot(original['normal'],delta))})
        extraction['planes'][1]=fragments[0]
        extraction['planes'].append(fragments[1])
        results=[]
        for proposal in (1,4):
            ids=[0,proposal,2,3]
            # Restore matching oriented normals: the fixture deliberately flips
            # counter and side normals to test canonical-sign independence.
            signs=np.array([1.,-1,1,-1])
            normals=np.asarray([extraction['planes'][i]['normal'] for i in ids])*signs[:,None]
            offsets=np.asarray([extraction['planes'][i]['offset'] for i in ids])*signs
            initial=solve_planes(normals,offsets,[p['normal'] for p in target['planes']],
                                 [p['offset'] for p in target['planes']])
            fitted=initial
            for _ in range(5):
                pooled=[pool_patch(extraction,p,fitted,1.) for p in target['planes']]
                np.testing.assert_array_equal(pooled[1][0],original['indices'])
                self.assertEqual(pooled[1][2],[1,4])
                fitted,_=refine_planes([extraction['points'][p[0]] for p in pooled],
                                       target['planes'],fitted,.015)
            results.append(fitted)
        np.testing.assert_allclose(results[0]['matrix'],results[1]['matrix'],atol=1e-8)
        first,_=match_surfaces(extraction,target)
        extraction['planes'][1],extraction['planes'][4]=extraction['planes'][4],extraction['planes'][1]
        second,_=match_surfaces(extraction,target)
        np.testing.assert_allclose(first['transform']['matrix'],second['transform']['matrix'],atol=1e-8)
        self.assertFalse(first['diagnostics']['uses_camera_correspondences'])

    def test_symmetric_surface_assignment_is_reported_ambiguous(self):
        rng = np.random.default_rng(21)
        rectangles = [rectangle([0,-2,0],[2,-2,0],[2,0,0],[0,0,0]),
                      rectangle([0,-2,1],[2,-2,1],[2,0,1],[0,0,1]),
                      rectangle([0,0,0],[2,0,0],[2,0,1],[0,0,1]),
                      rectangle([0,-2,0],[0,0,0],[0,0,1],[0,-2,1])]
        targets=[];planes=[];patches=[]
        for i,(name,n,d,triangles) in enumerate(zip(['floor','counter','back','side'],
                [[0.,0,1],[0,0,1],[0,-1,0],[1,0,0]],[0.,1,0,0],rectangles)):
            a,b,c=triangles[[0,0,1],[0,1,2]]
            uv=rng.uniform(.05,.95,(100,2));p=a+uv[:,:1]*(b-a)+uv[:,1:]*(c-a)
            patches.append(p)
            planes.append({'normal':n,'offset':d,'centroid':p.mean(axis=0).tolist(),
                           'indices':np.arange(i*100,(i+1)*100),'frame_count':5})
            targets.append({'name':name,'normal':n,'offset':d,'triangles':triangles.tolist(),'object_names':[name]})
        report,_=match_surfaces({'points':np.concatenate(patches),'planes':planes,
                                'diagnostics':{'robust_extent':3.,'distance_threshold':.001}}, {'planes':targets})
        self.assertEqual(report['status'],'needs_review')
        self.assertLess(report['diagnostics']['distinct_runner_up_margin'],.12)

    def test_camera_evaluation_does_not_refit_supplied_surface_transform(self):
        from test_alignment import make_bundle
        expected = transform_fixture(1)
        poses = np.broadcast_to(np.eye(4),(4,4,4)).copy()
        poses[:,:3,3] = [[0,0,0],[1,0,0],[1,1,0],[1,1,1]]
        # Deliberately unrelated cameras: a camera fit would change the answer.
        wrong = poses.copy();wrong[:,:3,3] += [20,-30,40]
        fit = {'transform':expected,'diagnostics':{'uses_camera_correspondences':False},
               'status':'automatic_candidate','anchors':[],'warnings':[]}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            make_bundle(root/'vggt','vggt',poses,[0,1,2,3],np.arange(4)/30)
            make_bundle(root/'blender','blender',wrong,[0,1,2,3],np.arange(4)/30)
            result=evaluate_alignment(SimpleNamespace(vggt=root/'vggt',blender_export=root/'blender',
                                                       output=root/'result',surface_fit=fit))
            data=json.loads(result.read_text())
            np.testing.assert_array_equal(data['transform']['matrix'],expected['matrix'])
            self.assertEqual(data['fit']['method'],'automatic_surfaces')
            self.assertEqual(data['camera_metrics']['evaluation']['translation_blender_units']['count'],4)
            self.assertEqual(data['camera_metrics']['fit']['translation_blender_units']['count'],0)
            self.assertGreater(data['camera_metrics']['all']['translation_blender_units']['rmse'],20)


if __name__ == '__main__':
    unittest.main()
