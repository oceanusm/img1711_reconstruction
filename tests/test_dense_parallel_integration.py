"""CPU subprocess checks of serial/parallel artifacts and resume ownership."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image

from scene_reconstruction.cli import parser
from scene_reconstruction.common import sha256, write_json
from scene_reconstruction.sam3_dense import _output_lock, make_plan, run
from tests.test_sam_dense import FakeBackend


class DelayedCPUBackend(FakeBackend):
    """Importable in workers; GPU choice is recorded but never initializes CUDA."""
    def __init__(self, checkpoint=None, gpu_index=0):
        super().__init__()
        self.gpu_index = gpu_index
        self.tracked_counts = []

    def discover(self, image_rgb, points_xy, batch_size=16):
        # Earlier source frames finish later without affecting predictions.
        time.sleep(max(0, 5 - int(image_rgb[0, 0, 0])) * .015)
        yield from super().discover(image_rgb, points_xy, batch_size)

    def track(self, resource_path, frame_count, seeds, height, width):
        staged = sorted(Path(resource_path).glob('*.png'))
        if len(staged) != frame_count or frame_count != 6:
            raise AssertionError('Every tracking worker must receive the entire six-frame clip')
        self.tracked_counts.append(frame_count)
        time.sleep(max(0, 6 - seeds[0]['object_id']) * .015)
        yield from super().track(resource_path, frame_count, seeds, height, width)

    def runtime_info(self):
        return {'backend': 'fake CPU integration test', 'pid': os.getpid(),
                'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'gpu_index': self.gpu_index, 'tracked_frame_counts': self.tracked_counts}


class InterruptAfterFirstBatch(FakeBackend):
    def track(self, resource_path, frame_count, seeds, height, width):
        if self.track_calls:
            raise RuntimeError('Synthetic interruption after one complete batch')
        yield from super().track(resource_path, frame_count, seeds, height, width)


class DenseParallelIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        # Do not inherit the operator's CUDA mask in these CPU-only tests.
        self.environment = patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        self.manifest = self.root / 'frames/manifest.json'
        self.manifest.parent.mkdir()
        frames = []
        for i in range(6):
            Image.new('RGB', (8, 12), (i, 20, 30)).save(self.manifest.parent / f'{i:06d}.png')
            frames.append({'index': i, 'file': f'{i:06d}.png',
                           'time_seconds': i*.2, 'pts_seconds': 2+i*.2})
        write_json(self.manifest, {'schema_version': 1, 'width': 8, 'height': 12, 'frames': frames})
        self.selection = self.root / 'selection.json'
        write_json(self.selection, {'source_manifest_sha256': sha256(self.manifest), 'frames': frames})

    def args(self, output, gpus=None):
        flags = ['sam-dense', '--manifest', str(self.manifest), '--selection', str(self.selection),
                 '--output', str(self.root / output), '--anchor-count', '3', '--grid-side', '2',
                 '--min-mask-area', '1', '--max-per-anchor', '0', '--track-batch-size', '2']
        if gpus is not None:
            flags += ['--gpus', gpus]
        return parser().parse_args(flags)

    def run_quiet(self, args, backend=DelayedCPUBackend):
        with redirect_stdout(io.StringIO()):
            run(args, backend)

    def assert_artifacts_equal(self, first, second):
        # Exclude only execution/progress metadata and private worker logs.
        for name in ('plan.json', 'manifest.json', 'raw_tracks.json'):
            self.assertEqual((first/name).read_bytes(), (second/name).read_bytes(), name)
        for directory in ('discovery', 'tracks', 'tracking_batches', 'catalog'):
            first_files = sorted(p.relative_to(first) for p in (first/directory).rglob('*') if p.is_file())
            second_files = sorted(p.relative_to(second) for p in (second/directory).rglob('*') if p.is_file())
            self.assertEqual(first_files, second_files)
            for relative in first_files:
                self.assertEqual((first/relative).read_bytes(), (second/relative).read_bytes(), str(relative))

    def test_three_workers_preserve_every_serial_artifact_and_full_tracking_sequence(self):
        serial = self.args('serial')
        parallel = self.args('parallel', '2,5,7')
        self.assertEqual(make_plan(serial)['fingerprint'], make_plan(parallel)['fingerprint'])
        self.run_quiet(serial)
        self.run_quiet(parallel)
        self.assert_artifacts_equal(serial.output, parallel.output)
        metadata = json.loads((parallel.output/'metadata.json').read_text())
        self.assertEqual(metadata['execution']['physical_gpus'], [2, 5, 7])
        self.assertEqual(metadata['raw_track_count'], 6)
        self.assertEqual(metadata['canonical_track_count'], 2)
        runtimes = [worker['runtime'] for worker in metadata['worker_runs']]
        self.assertEqual(len(runtimes), 6)  # Three discovery workers, then three tracking workers.
        self.assertEqual({runtime['visible_devices'] for runtime in runtimes}, {'2', '5', '7'})
        self.assertTrue(all(runtime['gpu_index'] == 0 for runtime in runtimes))
        self.assertTrue(all(runtime['pid'] != os.getpid() for runtime in runtimes))
        self.assertEqual([count for runtime in runtimes for count in runtime['tracked_frame_counts']], [6, 6, 6])

    def test_resume_keeps_completed_serial_jobs_when_switching_worker_devices_and_count(self):
        resumed = self.args('resumed')
        resumed.stage = 'discover'
        self.run_quiet(resumed)
        discovery_before = {p: p.stat().st_mtime_ns for p in (resumed.output/'discovery').rglob('*') if p.is_file()}
        resumed.stage, resumed.resume = 'track', True
        with self.assertRaisesRegex(RuntimeError, 'Synthetic interruption'):
            self.run_quiet(resumed, InterruptAfterFirstBatch)
        completed = sorted((resumed.output/'tracking_batches').glob('*.json'))
        self.assertEqual(len(completed), 1)
        completed_before = completed[0].stat().st_mtime_ns
        resumed.stage, resumed.gpus = 'all', '2,7'
        self.run_quiet(resumed)
        self.assertEqual({p: p.stat().st_mtime_ns for p in discovery_before}, discovery_before)
        self.assertEqual(completed[0].stat().st_mtime_ns, completed_before)
        reference = self.args('reference')
        self.run_quiet(reference)
        self.assert_artifacts_equal(reference.output, resumed.output)
        metadata = json.loads((resumed.output/'metadata.json').read_text())
        self.assertEqual(len(metadata['worker_runs']), 2)  # Only the two unfinished batches launched.
        self.assertEqual(metadata['status'], 'complete')
        # A later device-list change on a complete run needs no workers or model.
        resumed.gpus = '5'
        def forbidden(**kwargs):
            raise AssertionError('A complete cache must not load any backend')
        with patch('scene_reconstruction.sam3_dense_parallel.run_parallel_stage', side_effect=AssertionError('No new tasks')):
            self.run_quiet(resumed, forbidden)

    def test_coordinator_lock_excludes_concurrent_writers_and_releases_after_error(self):
        args = self.args('locked')
        with _output_lock(args.output):
            with self.assertRaisesRegex(RuntimeError, 'Another dense SAM coordinator'):
                self.run_quiet(args)
        self.assertFalse(args.output.exists())
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with _output_lock(args.output):
                raise RuntimeError('test failure')
        self.run_quiet(args)
        self.assertEqual(json.loads((args.output/'metadata.json').read_text())['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
