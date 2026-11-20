"""Dense pilot contracts with deterministic fake discovery/tracking, no GPU."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scene_reconstruction.cli import parser
from scene_reconstruction.common import sha256, write_json
from scene_reconstruction.sam3_dense import crop_boxes, grid_points, make_plan, run


class FakeBackend:
    def __init__(self, **kwargs):
        self.closed = False
        self.track_calls = 0

    def load_discovery(self): pass
    def release_discovery(self): pass
    def load_tracking(self): pass
    def close(self): self.closed = True
    def runtime_info(self): return {'backend': 'fake CPU contract test'}

    def discover(self, image_rgb, points_xy, batch_size=16):
        height, width = image_rgb.shape[:2]
        for i in range(3):
            mask = np.zeros((height, width), dtype=bool)
            if i < 2:
                mask[1:4, 1:4] = True  # Two points rediscover one object.
            else:
                mask[1:3, 1:3] = True  # Contained part is NOT a duplicate.
            yield {'mask': mask, 'seed_xy': [2., 2.], 'point_index': i,
                   'alternative_index': 0, 'predicted_iou': .99-i*.01, 'stability_score': 1.}

    def track(self, resource_path, frame_count, seeds, height, width):
        self.track_calls += 1
        for i in reversed(range(frame_count)):
            yield {'frame_index': i, 'masks': {s['object_id']: s['mask'].copy() for s in seeds}}


class DensePilotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / 'frames/manifest.json'
        self.manifest.parent.mkdir()
        frames = []
        for i in range(10):
            Image.new('RGB', (8, 12), (i, 20, 30)).save(self.manifest.parent / f'{i:06d}.png')
            frames.append({'index': i, 'file': f'{i:06d}.png', 'time_seconds': i*.1, 'pts_seconds': 2+i*.1})
        write_json(self.manifest, {'schema_version': 1, 'width': 8, 'height': 12, 'frames': frames})
        self.selection = self.root / 'selection.json'
        write_json(self.selection, {'source_manifest_sha256': sha256(self.manifest),
                                   'frames': [frames[i] for i in (0, 4, 9)]})
        self.args = parser().parse_args(['sam-dense', '--manifest', str(self.manifest),
            '--selection', str(self.selection), '--output', str(self.root/'pilot'),
            '--anchor-count', '2', '--grid-side', '2', '--min-mask-area', '1',
            '--max-per-anchor', '1', '--track-batch-size', '1'])

    def test_grid_is_xy_centers_and_crops_stay_inside_image(self):
        np.testing.assert_allclose(grid_points(8, 12, 2), [[2, 3], [6, 3], [2, 9], [6, 9]])
        boxes = crop_boxes(8, 12, 1)
        self.assertEqual(boxes[0], (0, 0, 8, 12))
        self.assertEqual(len(boxes), 5)
        self.assertTrue(all(0 <= x0 < x1 <= 8 and 0 <= y0 < y1 <= 12 for x0, y0, x1, y1 in boxes))

    def test_source_time_anchors_and_input_hashes_are_reproducible(self):
        plan = make_plan(self.args)
        self.assertEqual(plan['frame_ids'], [0, 4, 9])
        self.assertEqual(plan['anchors'], [0, 9])
        self.assertEqual(make_plan(self.args)['fingerprint'], plan['fingerprint'])
        Image.new('RGB', (8, 12), 'white').save(self.manifest.parent/'000004.png')
        self.assertNotEqual(make_plan(self.args)['fingerprint'], plan['fingerprint'])

    def test_discovery_retains_duplicates_and_budget_deferred_parts_on_disk(self):
        self.args.stage = 'discover'
        run(self.args, FakeBackend)
        output = self.root/'pilot'
        anchor = json.loads((output/'discovery/000000/index.json').read_text())
        self.assertEqual(len(anchor['proposals']), 3)
        self.assertEqual(len(anchor['tracking_proposal_ids']), 1)
        self.assertEqual(len(anchor['budget_deferred_ids']), 1)
        self.assertEqual(sum(d['status']=='duplicate' for d in anchor['decisions']), 1)
        self.assertTrue(all((output/p['mask']).is_file() for p in anchor['proposals']))
        manifest = json.loads((output/'manifest.json').read_text())
        self.assertEqual([f['index'] for f in manifest['frames']], [0, 4, 9])
        self.assertTrue(all((output/f['file']).is_file() for f in manifest['frames']))

    def test_tracking_covers_sparse_frames_and_resume_needs_no_gpu(self):
        self.args.stage = 'discover'
        run(self.args, FakeBackend)
        self.args.stage, self.args.resume = 'track', True
        run(self.args, FakeBackend)
        output = self.root/'pilot'
        tracks = json.loads((output/'raw_tracks.json').read_text())['tracks']
        self.assertEqual(len(tracks), 2)
        for track in tracks:
            index = json.loads((output/track['index_file']).read_text())
            self.assertEqual([f['index'] for f in index['frames']], [0, 4, 9])
        def forbidden(**kwargs): raise AssertionError('Completed batches must not reload a model')
        run(self.args, forbidden)
        self.args.grid_side = 3
        with self.assertRaisesRegex(ValueError, 'changed input'):
            run(self.args, forbidden)

    def test_missing_tracker_object_fails_instead_of_becoming_empty(self):
        class Missing(FakeBackend):
            def track(self, resource_path, frame_count, seeds, height, width):
                yield {'frame_index': 0, 'masks': {}}
        self.args.stage = 'track'
        # First create the discovery cache, then fail during tracking.
        self.args.stage = 'discover'
        run(self.args, FakeBackend)
        self.args.stage, self.args.resume = 'track', True
        with self.assertRaisesRegex(ValueError, 'omitted an object'):
            run(self.args, Missing)
        self.assertEqual(json.loads((self.root/'pilot/metadata.json').read_text())['status'], 'failed')

    def test_source_selection_mismatch_and_nonempty_output_are_rejected(self):
        selection = json.loads(self.selection.read_text())
        selection['frames'][0]['time_seconds'] = 1.0
        write_json(self.selection, selection)
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            make_plan(self.args)
        selection['frames'][0]['time_seconds'] = 0.0
        write_json(self.selection, selection)
        self.args.stage = 'discover'
        run(self.args, FakeBackend)
        with self.assertRaises(FileExistsError):
            run(self.args, FakeBackend)

    def test_dry_run_does_not_create_outputs_or_import_backend(self):
        self.args.dry_run = True
        def forbidden(**kwargs): raise AssertionError('Dry run must not load models')
        run(self.args, forbidden)
        self.assertFalse((self.root/'pilot').exists())

    def test_all_stages_preserve_parts_and_reconcile_repeated_discovery(self):
        self.args.max_per_anchor = 0
        run(self.args, FakeBackend)
        output = self.root/'pilot'
        catalog = json.loads((output/'catalog/catalog.json').read_text())
        self.assertEqual(catalog['raw_track_count'], 4)
        self.assertEqual(len(catalog['tracks']), 2)
        self.assertEqual([len(t['aliases']) for t in catalog['tracks']], [2, 2])
        self.assertTrue((output/'catalog/index.html').is_file())
        metadata = json.loads((output/'metadata.json').read_text())
        self.assertEqual(metadata['status'], 'complete')
        self.assertEqual(metadata['runtime']['backend'], 'fake CPU contract test')
        self.assertEqual(metadata['canonical_track_count'], 2)
        self.args.stage, self.args.resume = 'catalog', True
        def forbidden(**kwargs): raise AssertionError('Catalog rebuild must not load GPU')
        run(self.args, forbidden)

    def test_nonfinite_selected_timestamp_is_rejected(self):
        selection = json.loads(self.selection.read_text())
        selection['frames'][0]['pts_seconds'] = float('nan')
        self.selection.write_text(json.dumps(selection))
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            make_plan(self.args)

    def test_invalid_dedup_threshold_fails_before_models_load(self):
        self.args.dedup_iou = 0
        with self.assertRaisesRegex(ValueError, 'greater than zero'):
            make_plan(self.args)


if __name__ == '__main__':
    unittest.main()
