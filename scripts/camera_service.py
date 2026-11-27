#!/usr/bin/env python3
"""Launch isolated C++ camera workers with disjoint, topology-validated placement."""

import argparse
import csv
import json
import os
import platform
from pathlib import Path
import re
import signal
import subprocess
import time

import numpy as np

from common import YOLO_KEYS, model_files, model_type, stats
from inventory import topology
from mma_evidence import require_evidence
from run_record import artifact, atomic_json, identity, public_config, sha256


def positive_integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return value


# Keys a worker may set itself; they override the shared session so workers can run different models.
MODEL_KEYS = ('model', 'model_type', 'input_name', 'height', 'width', 'boxes_output', 'logits_output',
              'background_index', 'top_k', 'threshold', 'mean', 'std', *YOLO_KEYS, 'pad_value')


def validate_model_session(session):
    """Normalized per-worker session; rfdetr and yolo11 each require their own explicit keys."""
    session = dict(session)
    kind = session['model_type'] = model_type(session)
    if kind == 'yolo11':
        for key in YOLO_KEYS:
            if key not in session:
                raise ValueError(f'Explicit {key} required for yolo11')
    for key in ('height', 'width', 'queue_capacity', 'capture_timeout_ms',
                'top_k' if kind == 'rfdetr' else 'max_det'):
        positive_integer(session.get(key), key)
    positive_integer(session.get('warmup', 10), 'warmup', 0)
    required = ('model', 'input_name') + (('boxes_output', 'logits_output') if kind == 'rfdetr' else ('output_name',))
    for key in required:
        if not isinstance(session.get(key), str) or not session[key]:
            raise ValueError(f'Explicit {key} required')
    thresholds = {'duration_s': 0, 'max_frame_age_ms': 0}
    if kind == 'rfdetr':
        if 'background_index' not in session or (session['background_index'] is not None and
                                                 type(session['background_index']) is not int):
            raise ValueError('Explicit background_index must be null or integer')
        session['background_index'] = 'none' if session['background_index'] is None else session['background_index']
        thresholds['threshold'] = .25
    else:
        positive_integer(session['num_classes'], 'num_classes')
        thresholds.update(conf_threshold=None, iou_threshold=None)
        if 'pad_value' in session and (type(session['pad_value']) is not int or not 0 <= session['pad_value'] <= 255):
            raise ValueError('pad_value must be an integer in 0..255')
    for key, default in thresholds.items():
        value = float(session.get(key, default))
        if not np.isfinite(value) or value < 0 or (key not in ('duration_s', 'max_frame_age_ms') and value > 1):
            raise ValueError(f'Invalid {key}')
    return session


def validate_config(config, rows):
    """Return worker configs without inventing masks, node IDs or stream assignments."""
    common = dict(config['session'])
    mode = common.get('mode')
    if mode not in {'live', 'replay'}:
        raise ValueError('session.mode must be live or replay')
    if mode == 'live' and float(common.get('duration_s', 0)) <= 0:
        raise ValueError('Live service requires duration_s > 0')
    if mode == 'replay' and common.get('max_frame_age_ms', 0):
        raise ValueError('Replay cannot discard frames by age')
    by_cpu = {row['cpu']: row for row in rows}
    used, camera_ids, workers = set(), set(), []
    for worker in config['workers']:
        cpus = worker['cpus']
        if not cpus or any(type(cpu) is not int or cpu not in by_cpu for cpu in cpus):
            raise ValueError('Worker CPU mask outside effective topology')
        if len(set(cpus)) != len(cpus) or used.intersection(cpus):
            raise ValueError('Worker CPU masks must be unique and disjoint')
        used.update(cpus)
        threads = positive_integer(worker['threads'], 'threads')
        if threads > len(cpus):
            raise ValueError('ORT threads exceed worker CPU mask')
        node = worker.get('numa_node')
        if node is not None:
            positive_integer(node, 'numa_node', 0)
            if {by_cpu[cpu]['node'] for cpu in cpus} != {node}:
                raise ValueError('NUMA-local worker CPU mask spans other/unknown nodes')
        sources = worker['sources']
        if not sources:
            raise ValueError('Each worker requires at least one source')
        for source in sources:
            name = source['id']
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in camera_ids:
                raise ValueError('Camera identifiers must be safe and globally unique')
            if not isinstance(source['source'], str) or not source['source']:
                raise ValueError('Camera source must be a nonempty string')
            if mode == 'replay' and not Path(source['source']).is_file():
                raise ValueError('Replay requires existing local video files')
            if 'expected_frames' in source:
                positive_integer(source['expected_frames'], 'expected_frames')
            camera_ids.add(name)
        session = validate_model_session({**common, **{key: worker[key] for key in MODEL_KEYS if key in worker}})
        cfg = dict(session, cpus=sorted(cpus), threads=threads, sources=sources)
        if node is not None:
            cfg['numa_node'] = node
        workers.append(cfg)
    if not workers:
        raise ValueError('No workers configured')
    if len({worker['model'] for worker in workers}) > 1 and (config.get('require_mma') or config.get('mma_evidence')):
        raise ValueError('MMA evidence is per model: run one model per campaign when require_mma/mma_evidence is set')
    return workers


def stop_workers(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10
    for process in processes:
        try:
            process.wait(timeout=max(.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run(config, output, plan_only=False):
    workers = validate_config(config, topology())
    binary = Path(config['binary']).resolve(strict=True)
    if not os.access(binary, os.X_OK):
        raise ValueError('Camera binary is not executable')
    model_paths = [Path(worker['model']).resolve(strict=True) for worker in workers]
    distinct = sorted(set(model_paths))
    hashes = {path: (sha256(path), model_files(path)) for path in distinct}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {**identity('camera'), 'status': 'planned' if plan_only else 'running',
              'machine': platform.machine(),
              'config': public_config(config),
              'models': [{'worker': index, 'model_type': worker['model_type'], 'path': str(path),
                          'sha256': hashes[path][0]}
                         for index, (worker, path) in enumerate(zip(workers, model_paths))],
              'model_files': hashes[distinct[0]][1] if len(distinct) == 1 else None,
              'model_sha256': hashes[distinct[0]][0] if len(distinct) == 1 else None,
              'worker_binary': artifact(binary),
              'profile_instrumented': bool(config['session'].get('profile', 0))}
    record['source_files'] = {source['id']: artifact(source['source']) for worker in workers
                              for source in worker['sources'] if '://' not in source['source'] and Path(source['source']).is_file()}
    # This is the C++ ORT library actually linked by the binary, specified explicitly.
    runtime = artifact(config['runtime_library'])
    record['runtime_library'] = runtime
    record['mma_evidence'] = require_evidence(config, 'inference', record['model_sha256'], [runtime], record['model_files'])
    atomic_json(output / 'run.json', record)
    if plan_only:
        return record
    processes, logs, private = [], [], []
    started = time.perf_counter()
    try:
        for index, cfg in enumerate(workers):
            target = output / f'worker-{index:02d}'
            cfg = dict(cfg, model=str(model_paths[index]), output=str(target))
            path = output / f'.worker-{index:02d}.private.json'
            # Source credentials are visible only in a mode-0600 ephemeral worker config.
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            private.append(path)
            with os.fdopen(descriptor, 'w') as stream:
                json.dump(cfg, stream, allow_nan=False)
            log = (output / f'worker-{index:02d}.log').open('w')
            logs.append(log)
            env = dict(os.environ, OMP_NUM_THREADS=str(cfg['threads']), OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENCV_FFMPEG_LOGLEVEL='-8')
            processes.append(subprocess.Popen([str(binary), str(path)], stdout=log,
                                               stderr=subprocess.STDOUT, env=env))
        timeout = float(config.get('timeout_s', 0))
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError('Camera worker failed; all peers stopped')
            if timeout > 0 and time.perf_counter() - started > timeout:
                raise TimeoutError('Camera campaign exceeded timeout_s')
            time.sleep(.1)
        if any(process.returncode for process in processes):
            raise RuntimeError('Camera worker exited with failure')
        results, rows = [], []
        for index in range(len(workers)):
            target = output / f'worker-{index:02d}'
            result = json.loads((target / 'result.json').read_text())
            if result['status'] != 'ok' or result['images'] <= 0:
                raise RuntimeError('Worker has no valid completed frames')
            if sha256(result['runtime_library']) != runtime['sha256']:
                raise RuntimeError('Loaded ORT library differs from the declared/hashed runtime')
            if config['session']['mode'] == 'replay':
                by_id = {camera['id']: camera for camera in result['cameras']}
                for source in workers[index]['sources']:
                    if 'expected_frames' in source:
                        camera = by_id[source['id']]
                        if camera['received'] != source['expected_frames'] or camera['processed'] != source['expected_frames']:
                            raise RuntimeError('Replay frame-count integrity check failed')
            results.append(result)
            with (target / 'samples.csv').open() as stream:
                rows.extend(csv.DictReader(stream))
        begin = min(result['active_start_monotonic_s'] for result in results)
        end = max(result['active_end_monotonic_s'] for result in results)
        # Common overlap is computed from actual completion timestamps, not summed isolated FPS.
        overlap_begin = max(result['active_start_monotonic_s'] for result in results)
        overlap_end = min(result['active_end_monotonic_s'] for result in results)
        overlap_count = sum(overlap_begin <= float(row['completed_monotonic_s']) <= overlap_end for row in rows)
        record.update(status='ok', workers=results, images=sum(result['images'] for result in results),
                      active_elapsed_s=end - begin,
                      throughput_active_fps=len(rows) / (end - begin),
                      concurrent_overlap_s=max(0, overlap_end - overlap_begin),
                      concurrent_overlap_images=overlap_count,
                      concurrent_overlap_fps=overlap_count / (overlap_end - overlap_begin) if overlap_end > overlap_begin else None,
                      latency_ms={key: stats([float(row[key]) for row in rows]) for key in
                                  ('queue_age_ms', 'receive_to_result_ms', 'read_to_result_ms', 'inference_ms')},
                      timing_scope='Concurrent camera window after each worker warmup; includes capture startup/output IO; not steady-state or sensor-to-result')
        record['replay_frame_counts_verified'] = config['session']['mode'] == 'replay' and all(
            'expected_frames' in source for worker in workers for source in worker['sources'])
        if record['profile_instrumented']:
            profiles = [artifact(result['profile_path']) for result in results]
            record['operator_profile'] = profiles[0]
            record['operator_profiles'] = profiles
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        stop_workers(processes)
        for log in logs:
            log.close()
        for path in private:
            path.unlink(missing_ok=True)
        record['parent_makespan_s'] = time.perf_counter() - started
        atomic_json(output / 'run.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    # SIGTERM follows the same cleanup path as Ctrl-C.
    def terminate(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    record = run(json.loads(Path(args.config).read_text()), args.output, args.plan_only)
    print(json.dumps({key: record[key] for key in ('run_id', 'status')}))


if __name__ == '__main__':
    main()
