"""Actual subprocess scheduling tests with CPU masks, not GPU inference."""
import json
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from scene_reconstruction.cli import parser
from scene_reconstruction.common import write_json
from scene_reconstruction.sam3_dense import _tracking_jobs, make_plan
from scene_reconstruction.sam3_dense_parallel import parse_gpu_ids, run_parallel_stage


class IsolatedBackend:
    """Produces overlapping masks and reports worker visibility/call reuse."""
    def __init__(self, checkpoint=None, gpu_index=0):
        self.gpu_index = gpu_index
        self.discoveries = 0
        self.tracking = []

    def load_discovery(self): pass
    def load_tracking(self): pass
    def close(self): pass

    def discover(self, image_rgb, points_xy, batch_size=16):
        self.discoveries += 1
        for ordinal, end in enumerate((4, 3)):
            mask = np.zeros(image_rgb.shape[:2], bool)
            mask[1:end, 1:end] = True
            yield {'mask': mask, 'seed_xy': [2., 2.], 'point_index': ordinal,
                   'alternative_index': 0, 'predicted_iou': .99-ordinal*.01, 'stability_score': 1.}

    def track(self, resource_path, frame_count, seeds, height, width):
        self.tracking.append({'frames': frame_count, 'files': len(list(Path(resource_path).iterdir())),
                              'seed_ids': [s['object_id'] for s in seeds]})
        for i in reversed(range(frame_count)):
            yield {'frame_index': i, 'masks': {s['object_id']: s['mask'].copy() for s in seeds}}

    def runtime_info(self):
        return {'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'logical_index': self.gpu_index, 'discoveries': self.discoveries,
                'tracking': self.tracking, 'pid': os.getpid(),
                'is_main': os.environ.get('IS_MAIN_PROCESS'),
                'distributed': {key: os.environ[key] for key in
                    ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT') if key in os.environ}}


class FailingBackend(IsolatedBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.children = []

    def close(self):
        for child in self.children:
            child.wait(timeout=1)  # Parent must signal the whole process group.

    def discover(self, image_rgb, points_xy, batch_size=16):
        # Other workers prove that parent failure cleanup also kills descendants.
        if os.environ['CUDA_VISIBLE_DEVICES'] == '2':
            time.sleep(.5)
            raise RuntimeError('deliberate worker failure')
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        self.children.append(child)
        Path(os.environ['DENSE_TEST_CHILD_PIDS']).joinpath(os.environ['CUDA_VISIBLE_DEVICES']).write_text(str(child.pid))
        time.sleep(60)
        yield from super().discover(image_rgb, points_xy, batch_size)


class HoldingBackend(IsolatedBackend):
    def discover(self, image_rgb, points_xy, batch_size=16):
        Path(os.environ['DENSE_TEST_WORKER_STARTED']).write_text(str(os.getpid()))
        release = Path(os.environ['DENSE_TEST_WORKER_RELEASE'])
        deadline = time.monotonic() + 20
        while not release.is_file():
            if time.monotonic() > deadline:
                raise RuntimeError('Test worker release deadline exceeded')
            time.sleep(.05)
        yield from super().discover(image_rgb, points_xy, batch_size)


class DenseParallelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root/'output'
        self.manifest = self.root/'frames/manifest.json'
        self.manifest.parent.mkdir()
        frames = []
        for i in range(6):
            Image.new('RGB', (8, 12), (i, 20, 30)).save(self.manifest.parent/f'{i:06d}.png')
            frames.append({'index': i, 'file': f'{i:06d}.png', 'time_seconds': i*.2, 'pts_seconds': i*.2})
        write_json(self.manifest, {'schema_version': 1, 'width': 8, 'height': 12, 'frames': frames})
        args = parser().parse_args(['sam-dense', '--manifest', str(self.manifest),
             '--anchor-count', '6', '--grid-side', '2', '--min-mask-area', '1'])
        args.selection = None
        self.plan = make_plan(args)
        self.visibility = patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,5,7'})
        self.visibility.start()
        self.addCleanup(self.visibility.stop)

    def test_parse_preserves_physical_ordinals_and_respects_visibility(self):
        self.assertEqual(parse_gpu_ids('2, 5,7'), [2, 5, 7])
        self.assertEqual(parse_gpu_ids([7, 2]), [7, 2])
        self.assertEqual(parse_gpu_ids(None), [])
        for values in ('', '2,2', '-1', '2,,7', [True], '2.5'):
            with self.subTest(values=values), self.assertRaises(ValueError):
                parse_gpu_ids(values)
        with self.assertRaisesRegex(ValueError, 'excluded'):
            parse_gpu_ids('2,5,7', visible_devices='7')
        with self.assertRaisesRegex(ValueError, 'numeric'):
            parse_gpu_ids('2', visible_devices='GPU-uuid')

    def test_discovery_reuses_one_model_per_gpu_and_tracks_entire_sequences(self):
        original = json.dumps(self.plan, sort_keys=True)
        with patch.dict(os.environ, {'IS_MAIN_PROCESS': '0', 'RANK': '4', 'LOCAL_RANK': '4',
                                    'WORLD_SIZE': '8', 'MASTER_ADDR': 'example', 'MASTER_PORT': '12345'}):
            records = run_parallel_stage('discover', self.plan['frames'], self.plan, self.output,
                                         [2, 5, 7], IsolatedBackend)
        self.assertEqual([r['physical_gpu'] for r in records], [2, 5, 7])
        self.assertEqual([r['runtime']['visible_devices'] for r in records], ['2', '5', '7'])
        self.assertEqual([r['runtime']['logical_index'] for r in records], [0, 0, 0])
        self.assertEqual([r['runtime']['is_main'] for r in records], ['1', '1', '1'])
        self.assertEqual([r['runtime']['distributed'] for r in records], [{}, {}, {}])
        self.assertEqual([r['runtime']['discoveries'] for r in records], [2, 2, 2])
        self.assertEqual(len({r['pid'] for r in records}), 3)
        self.assertEqual(json.dumps(self.plan, sort_keys=True), original)
        anchors = [json.loads((self.output/'discovery'/f'{i:06d}'/'index.json').read_text()) for i in range(6)]
        jobs = _tracking_jobs(anchors, 2)
        records = run_parallel_stage('track', jobs, self.plan, self.output, [2, 5, 7], IsolatedBackend)
        self.assertEqual([r['completed_tasks'] for r in records], [2, 2, 2])
        for record in records:
            for call in record['runtime']['tracking']:
                self.assertEqual((call['frames'], call['files']), (6, 6))
        for job in jobs:
            batch = json.loads((self.output/'tracking_batches'/f'{job["name"]}.json').read_text())
            self.assertEqual([t['id'] for t in batch['tracks']], [s['id'] for s in job['seeds']])
            for track in batch['tracks']:
                index = json.loads((self.output/track['index_file']).read_text())
                self.assertEqual([f['index'] for f in index['frames']], list(range(6)))
        self.assertFalse((self.output/'metadata.json').exists())
        self.assertFalse((self.output/'raw_tracks.json').exists())

    def test_empty_stage_does_not_spawn_or_create_outputs(self):
        self.assertEqual(run_parallel_stage('discover', [], self.plan, self.output, [2,5,7]), [])
        self.assertFalse(self.output.exists())

    def test_duplicate_artifact_destinations_rejected_before_launch(self):
        with self.assertRaisesRegex(ValueError, 'unique artifact'):
            run_parallel_stage('discover', [self.plan['frames'][0]]*2, self.plan, self.output, [2,5,7])
        self.assertFalse(self.output.exists())

    def test_worker_failure_is_reported_and_other_sessions_are_stopped(self):
        child_pids = self.root/'child_pids'
        child_pids.mkdir()
        started = time.monotonic()
        with patch.dict(os.environ, {'DENSE_TEST_CHILD_PIDS': str(child_pids)}):
            with self.assertRaisesRegex(RuntimeError, 'GPU 2 failed.*deliberate worker failure'):
                run_parallel_stage('discover', self.plan['frames'], self.plan, self.output,
                                   [2,5,7], FailingBackend)
        self.assertLess(time.monotonic()-started, 10)
        status_files = list((self.output/'workers').glob('*/*.status.json'))
        self.assertEqual(len(status_files), 3)
        statuses = [json.loads(p.read_text()) for p in status_files]
        for state in statuses:
            with self.assertRaises(ProcessLookupError):
                os.kill(state['pid'], 0)
        self.assertTrue(list(child_pids.iterdir()))
        for path in child_pids.iterdir():
            pid = int(path.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_worker_retains_output_lock_after_coordinator_is_hard_killed(self):
        lock_path = self.root/'output.lock'
        plan_path = self.root/'plan.json'
        write_json(plan_path, self.plan)
        started, release = self.root/'started', self.root/'release'
        env = os.environ.copy()
        env.update(DENSE_TEST_WORKER_STARTED=str(started), DENSE_TEST_WORKER_RELEASE=str(release))
        code = '''import fcntl, json, sys
from pathlib import Path
from scene_reconstruction.sam3_dense_parallel import run_parallel_stage
from tests.test_sam_dense_parallel import HoldingBackend
handle = open(sys.argv[1], 'a+')
fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
plan = json.loads(Path(sys.argv[2]).read_text())
run_parallel_stage('discover', plan['frames'][:1], plan, Path(sys.argv[3]), [2],
                   HoldingBackend, coordinator_lock_fd=handle.fileno())
'''
        coordinator = subprocess.Popen([sys.executable, '-c', code, str(lock_path), str(plan_path), str(self.output)],
                                       env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 10
            while not started.is_file() and coordinator.poll() is None and time.monotonic() < deadline:
                time.sleep(.05)
            self.assertTrue(started.is_file(), 'Worker failed to start under the coordinator lock')
            coordinator.kill()  # No finally/LOCK_UN: simulate an abrupt machine/process failure.
            coordinator.wait(timeout=5)
            with lock_path.open('a+') as contender:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            release.touch()
            deadline = time.monotonic() + 10
            acquired = False
            while time.monotonic() < deadline:
                with lock_path.open('a+') as contender:
                    try:
                        fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        time.sleep(.05)
                    else:
                        acquired = True
                        break
            self.assertTrue(acquired, 'Lock must release after the last worker exits')
            state = json.loads(next((self.output/'workers').glob('*/*.status.json')).read_text())
            self.assertEqual(state['status'], 'complete')
        finally:
            release.touch()
            if coordinator.poll() is None:
                coordinator.kill()
                coordinator.wait(timeout=5)
            coordinator.stderr.close()


if __name__ == '__main__':
    unittest.main()
