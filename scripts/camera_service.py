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

from common import model_files, stats
from inventory import topology
from mma_evidence import require_evidence
from run_record import artifact, atomic_json, identity, public_config, sha256


def positive_integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return value


def validate_config(config, rows):
    """Return worker configs without inventing masks, node IDs or stream assignments."""
    common = dict(config['session'])
    mode = common.get('mode')
    if mode not in {'live', 'replay'}:
        raise ValueError('session.mode must be live or replay')
    if mode == 'live' and float(common.get('duration_s', 0)) <= 0:
        raise ValueError('Live service requires duration_s > 0')
    for key in ('height', 'width', 'top_k', 'queue_capacity', 'capture_timeout_ms'):
        positive_integer(common[key], key)
    positive_integer(common.get('warmup', 10), 'warmup', 0)
    for key in ('model', 'input_name', 'boxes_output', 'logits_output'):
        if not isinstance(common.get(key), str) or not common[key]:
            raise ValueError(f'Explicit {key} required')
    if 'background_index' not in common or (common['background_index'] is not None and
                                            type(common['background_index']) is not int):
        raise ValueError('Explicit background_index must be null or integer')
    common['background_index'] = 'none' if common['background_index'] is None else common['background_index']
    for key in ('duration_s', 'max_frame_age_ms', 'threshold'):
        value = float(common.get(key, .25 if key == 'threshold' else 0))
        if not np.isfinite(value) or value < 0 or (key == 'threshold' and value > 1):
            raise ValueError(f'Invalid {key}')
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
        cfg = dict(common, cpus=sorted(cpus), threads=threads, sources=sources)
        if node is not None:
            cfg['numa_node'] = node
        workers.append(cfg)
    if not workers:
        raise ValueError('No workers configured')
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
    model = Path(config['session']['model']).resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = {**identity('camera'), 'status': 'planned' if plan_only else 'running',
              'machine': platform.machine(),
              'config': public_config(config), 'model_files': model_files(model),
              'model_sha256': sha256(model), 'worker_binary': artifact(binary),
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
            cfg = dict(cfg, model=str(model), output=str(target))
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
