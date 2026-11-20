"""Independent dense-SAM jobs in isolated, persistent GPU subprocesses.

The coordinator assigns complete anchors or pre-numbered tracking batches.
Workers never publish the global manifest, track list, or catalog. Their only
shared writes are to disjoint task directories and their own status files.
"""
from __future__ import annotations

from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

from .common import write_json


def parse_gpu_ids(value, visible_devices=None):
    """Parse physical CUDA ordinals without overriding an existing restriction."""
    if value is None:
        return []
    values = value.split(',') if isinstance(value, str) else list(value)
    if not values:
        raise ValueError('At least one GPU is required')
    ids = []
    for item in values:
        if isinstance(item, bool) or not str(item).strip().isdigit():
            raise ValueError('GPU IDs must be nonnegative physical CUDA ordinals, e.g. 2,5,7')
        ids.append(int(str(item).strip()))
    if len(ids) != len(set(ids)):
        raise ValueError('GPU IDs must be unique')
    restriction = os.environ.get('CUDA_VISIBLE_DEVICES') if visible_devices is None else visible_devices
    if restriction is not None:
        allowed = [token.strip() for token in restriction.split(',')]
        if not all(token.isdigit() for token in allowed):
            raise ValueError('Numeric --gpus requires an unset CUDA_VISIBLE_DEVICES or a numeric '
                             'visibility list containing those physical GPU IDs')
        if not set(ids).issubset({int(token) for token in allowed}):
            raise ValueError(f'Requested GPUs {ids} are excluded by CUDA_VISIBLE_DEVICES={restriction}; '
                             'use the intended visibility list explicitly')
    return ids


def _factory_reference(factory):
    if factory is None:
        return None
    module, name = factory.__module__, factory.__qualname__
    if '<locals>' in name or module == '__main__':
        raise ValueError('Parallel backend_factory must be an importable module-level callable')
    reference = f'{module}:{name}'
    if _load_factory(reference) is not factory:
        raise ValueError('Parallel backend_factory is not importable by its module/name')
    return reference


def _load_factory(reference):
    if reference is None:
        from .sam3_dense_backend import Sam3DenseBackend
        return Sam3DenseBackend
    module, name = reference.split(':', 1)
    value = importlib.import_module(module)
    for component in name.split('.'):
        value = getattr(value, component)
    return value


def _task_ids(stage, tasks):
    if stage == 'discover':
        identities = [task['index'] for task in tasks]
    elif stage == 'track':
        identities = [task['name'] for task in tasks]
        tracks = [seed['id'] for task in tasks for seed in task['seeds']]
        if len(tracks) != len(set(tracks)):
            raise ValueError('Parallel tracking jobs must have disjoint preassigned track IDs')
    else:
        raise ValueError('Parallel stage must be discover or track')
    if len(identities) != len(set(identities)):
        raise ValueError('Parallel tasks must have unique artifact destinations')
    return identities


def _signal_group(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _stop_workers(processes):
    """Stop whole sessions, including any native SAM children, and reap workers."""
    for process in processes:
        _signal_group(process, signal.SIGTERM)
    deadline = time.monotonic() + 3
    for process in processes:
        try:
            process.wait(timeout=max(.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    # A worker can exit before an uncooperative descendant; signal the group
    # even when the direct child has already been reaped.
    for process in processes:
        _signal_group(process, signal.SIGKILL)
    for process in processes:
        process.wait()


def run_parallel_stage(stage, tasks, plan, output, gpu_ids, backend_factory=None,
                       coordinator_lock_fd=None):
    """Run exact existing tasks in fresh workers; return ordered worker records.

    Each worker retains its model between tasks, sees one physical GPU as cuda:0,
    and processes complete sequences. Completion order cannot change task/track
    IDs. GPU placement is execution metadata and does not alter the plan hash.
    Completed artifacts can be resumed by the coordinator after worker failure.
    Workers inherit the coordinator's output-lock descriptor when supplied, so
    a hard-killed coordinator cannot expose unfinished writes to a second run.
    """
    tasks = list(tasks)
    identities = _task_ids(stage, tasks)
    gpu_ids = parse_gpu_ids(gpu_ids)
    if not tasks:
        return []
    if not gpu_ids:
        raise ValueError('Parallel execution requires at least one GPU')
    if coordinator_lock_fd is not None:
        if type(coordinator_lock_fd) is not int or coordinator_lock_fd < 0:
            raise ValueError('coordinator_lock_fd must be an open file descriptor')
        os.fstat(coordinator_lock_fd)
    factory = _factory_reference(backend_factory)
    output = Path(output).resolve()
    run_dir = output / 'workers' / f'{stage}-{uuid.uuid4().hex[:12]}'
    run_dir.mkdir(parents=True, exist_ok=False)
    worker_count = min(len(gpu_ids), len(tasks))
    processes, specifications, records = [], [], []
    with ExitStack() as stack:
        try:
            for ordinal, gpu in enumerate(gpu_ids[:worker_count]):
                assigned = tasks[ordinal::worker_count]
                base = run_dir / f'worker_gpu_{gpu}'
                spec_path = base.with_suffix('.json')
                status_path = base.with_suffix('.status.json')
                log_path = base.with_suffix('.log')
                spec = {'schema_version': 1, 'stage': stage, 'physical_gpu': gpu,
                        'output': str(output), 'plan': plan, 'tasks': assigned,
                        'backend_factory': factory, 'status_path': str(status_path),
                        'coordinator_lock_fd': coordinator_lock_fd}
                write_json(spec_path, spec)
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                env['PYTHONUNBUFFERED'] = '1'
                env['IS_MAIN_PROCESS'] = '1'
                for variable in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                    env.pop(variable, None)
                # Keep import resolution equal to the coordinator, including
                # editable source trees and importable CPU-test backends.
                env['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(
                    str(Path(path or os.getcwd()).resolve()) for path in sys.path))
                log = stack.enter_context(log_path.open('w'))
                process = subprocess.Popen(
                    [sys.executable, '-m', 'scene_reconstruction.sam3_dense_parallel', str(spec_path)],
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    pass_fds=() if coordinator_lock_fd is None else (coordinator_lock_fd,),
                )
                processes.append(process)
                specifications.append((gpu, status_path, log_path))
                print(f'{stage}: GPU {gpu}, {len(assigned)} tasks; log {log_path}', flush=True)
            next_report = time.monotonic() + 15
            while True:
                codes = [process.poll() for process in processes]
                failed = next((i for i, code in enumerate(codes) if code is not None and code != 0), None)
                if failed is not None:
                    gpu, status_path, log_path = specifications[failed]
                    detail = ''
                    if status_path.is_file():
                        detail = json.loads(status_path.read_text()).get('error', '')
                    raise RuntimeError(f'Dense SAM {stage} worker on GPU {gpu} failed '
                                       f'(exit {codes[failed]}): {detail}. See {log_path}')
                if all(code is not None for code in codes):
                    break
                if time.monotonic() >= next_report:
                    finished = 0
                    for _, status_path, _ in specifications:
                        if status_path.is_file():
                            finished += json.loads(status_path.read_text()).get('completed_tasks', 0)
                    print(f'{stage}: {finished}/{len(identities)} tasks complete across {worker_count} GPUs', flush=True)
                    next_report = time.monotonic() + 15
                time.sleep(.1)
            for gpu, status_path, log_path in specifications:
                if not status_path.is_file():
                    raise RuntimeError(f'GPU {gpu} worker exited without status: {log_path}')
                record = json.loads(status_path.read_text())
                if record.get('status') != 'complete' or record.get('completed_tasks') != record.get('task_count'):
                    raise RuntimeError(f'GPU {gpu} worker exited without complete tasks: {log_path}')
                record['log'] = str(log_path.relative_to(output))
                record['status_file'] = str(status_path.relative_to(output))
                records.append(record)
        except BaseException:
            _stop_workers(processes)
            raise
    return records


def _worker_main(spec_path):
    """Private subprocess entry point; model imports happen after GPU isolation."""
    spec = json.loads(Path(spec_path).read_text())
    if spec.get('schema_version') != 1:
        raise ValueError('Unsupported dense SAM worker specification')
    stage, tasks, plan = spec['stage'], spec['tasks'], spec['plan']
    identities = _task_ids(stage, tasks)
    physical_gpu = spec['physical_gpu']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(physical_gpu):
        raise RuntimeError('Worker must see exactly its assigned physical GPU')
    lock_fd = spec.get('coordinator_lock_fd')
    if lock_fd is not None:
        os.fstat(lock_fd)
        # Keep this inherited descriptor open until process exit. Never call
        # LOCK_UN: the parent and workers share the same open-file description.
    output, status_path = Path(spec['output']), Path(spec['status_path'])
    started = time.monotonic()
    status = {'schema_version': 1, 'status': 'running', 'stage': stage,
              'physical_gpu': physical_gpu, 'logical_gpu_index': 0,
              'fingerprint': plan['fingerprint'], 'pid': os.getpid(),
              'task_count': len(tasks), 'completed_tasks': 0, 'task_ids': identities}
    write_json(status_path, status)
    backend = None
    error = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Worker received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        from .sam3_dense import _discover_anchor, _track_job
        backend = _load_factory(spec['backend_factory'])(checkpoint=plan['checkpoint_path'], gpu_index=0)
        with ExitStack() as stack:
            resource_path = None
            if stage == 'discover':
                backend.load_discovery()
            else:
                backend.load_tracking()
                resource_path = stack.enter_context(tempfile.TemporaryDirectory(prefix='dense_sam_worker_frames_'))
                for position, frame in enumerate(plan['frames']):
                    source = Path(frame['source_path'])
                    (Path(resource_path) / f'{position:06d}{source.suffix.lower()}').symlink_to(source)
            for task in tasks:
                if stage == 'discover':
                    _discover_anchor(backend, task, plan, output)
                else:
                    _track_job(backend, task, plan, output, resource_path)
                status['completed_tasks'] += 1
                write_json(status_path, status)
        status['status'] = 'complete'
    except BaseException as exc:
        error = exc
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    finally:
        if backend is not None:
            # A provenance failure must not skip native session shutdown.
            for operation in ('runtime_info', 'close'):
                try:
                    method = getattr(backend, operation, None)
                    if callable(method):
                        value = method()
                        if operation == 'runtime_info':
                            status['runtime'] = value
                except BaseException as exc:
                    traceback.print_exc()
                    if error is None:
                        error = exc
                        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        status['seconds'] = round(time.monotonic() - started, 3)
        write_json(status_path, status)
    return 1 if error is not None else 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m scene_reconstruction.sam3_dense_parallel WORKER_SPEC.json')
    raise SystemExit(_worker_main(sys.argv[1]))
