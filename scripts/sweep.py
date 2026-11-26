#!/usr/bin/env python3
"""Topology-aware CPU/NUMA sweep. Concurrent worker throughput uses measured makespan."""
import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from inventory import topology


def selected_cpus(rows, smt):
    groups = {}
    for row in rows:
        group = tuple(row['siblings_allowed']) or (row['cpu'],)
        groups.setdefault(group, []).append(row['cpu'])
    selected = {cpu for group in groups.values() for cpu in sorted(group)[:smt]}
    return [row for row in rows if row['cpu'] in selected]


def pools(rows, layout):
    by_node = {}
    for row in rows:
        by_node.setdefault(row['node'], []).append(row['cpu'])
    if layout == 'numa-local':
        return [{'label': 'node-%s' % node, 'cpus': sorted(cpus), 'membind': node}
                for node, cpus in by_node.items() if node is not None]
    if layout == 'all':
        return [{'label': 'all', 'cpus': sorted(r['cpu'] for r in rows), 'membind': None}]
    if layout == 'spread':
        ordered = []
        for group in itertools.zip_longest(*(sorted(v) for v in by_node.values())):
            ordered.extend(x for x in group if x is not None)
        return [{'label': 'spread', 'cpus': ordered, 'membind': None}]
    raise ValueError('Unknown layout ' + layout)


def build_plan(config, rows):
    plan = []
    for smt, layout, threads, workers, repeat in itertools.product(
            config.get('smt_limits', [1, 2, 4, 8]),
            config.get('layouts', ['all', 'numa-local', 'spread']),
            config.get('threads', [1, 2, 4, 8, 16, 30]),
            config.get('workers', [1]), range(config.get('repeats', 3))):
        if min(smt, threads, workers) < 1:
            raise ValueError('smt_limits, threads and workers must be positive')
        for pool in pools(selected_cpus(rows, smt), layout):
            if len(pool['cpus']) < threads * workers:
                continue
            # Allocate disjoint affinity masks; never oversubscribe the logical CPUs.
            masks = [pool['cpus'][i * threads:(i + 1) * threads] for i in range(workers)]
            plan.append({'id': len(plan), 'smt_limit': smt, 'layout': layout,
                         'pool': pool['label'], 'membind': pool['membind'],
                         'threads': threads, 'workers': workers, 'repeat': repeat,
                         'cpu_masks': masks})
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--plan-only', action='store_true')
    args = ap.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    base_path = Path(config['base_config'])
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    base = json.loads(base_path.read_text())
    rows = topology()
    plan = build_plan(config, rows)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'plan.json').write_text(json.dumps({'topology': rows, 'cases': plan}, indent=2) + '\n')
    if not plan:
        raise RuntimeError('No valid sweep cases for allowed CPU topology/configuration; inspect plan.json')
    if args.plan_only:
        print('%d cases; review %s before running' % (len(plan), out / 'plan.json'))
        return
    results = []
    benchmark = Path(__file__).with_name('benchmark.py').resolve()
    for case in plan:
        target = out / ('case-%04d' % case['id'])
        target.mkdir()
        launched = []
        errors = []
        wall_start = time.perf_counter()
        try:
            for worker, cpus in enumerate(case['cpu_masks']):
                worker_out = target / ('worker-%02d' % worker)
                cfg = dict(base, threads=case['threads'], inter_threads=1,
                           cpus=cpus, sweep_case=case, worker_index=worker)
                cfg_path = target / ('worker-%02d.json' % worker)
                cfg_path.write_text(json.dumps(cfg, indent=2) + '\n')
                affinity = ','.join(map(str, cpus))
                if case['membind'] is not None:
                    if not shutil.which('numactl'):
                        raise RuntimeError('numactl missing: cannot enforce local memory placement')
                    prefix = ['numactl', '--physcpubind=' + affinity, '--membind=' + str(case['membind'])]
                else:
                    if not shutil.which('taskset'):
                        raise RuntimeError('taskset missing: cannot enforce CPU affinity')
                    prefix = ['taskset', '--cpu-list', affinity]
                argv = prefix + [sys.executable, str(benchmark), '--config', str(cfg_path), '--output', str(worker_out)]
                env = dict(os.environ, OMP_NUM_THREADS=str(case['threads']),
                           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                log = (target / ('worker-%02d.log' % worker)).open('w')
                process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
                launched.append((process, log, worker_out))
            for process, log, worker_out in launched:
                code = process.wait()
                log.close()
                if code:
                    errors.append('worker %s exit %d' % (worker_out.name, code))
        except BaseException:
            for process, log, _ in launched:
                if process.poll() is None:
                    process.terminate()
                process.wait()
                log.close()
            raise
        makespan = time.perf_counter() - wall_start
        images = 0
        for _, _, worker_out in launched:
            try:
                result = json.loads((worker_out / 'result.json').read_text())
                images += int(result['images'])
            except (OSError, ValueError, KeyError) as e:
                errors.append('%s result unavailable: %s' % (worker_out.name, e))
        summary = {**{k: v for k, v in case.items() if k != 'cpu_masks'},
                   'cpu_masks': case['cpu_masks'], 'makespan_s': makespan,
                   'images': images, 'valid': not errors, 'errors': errors,
                   'throughput_makespan_fps': images / makespan if not errors else None,
                   'throughput_definition': 'total measured images / parent wall time including process init, warmup, teardown; no sum of isolated worker FPS'}
        (target / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        results.append(summary)
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        with (out / 'results.csv').open('w', newline='') as stream:
            keys = list(results[0])
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in results)
        print('case-%04d valid=%s FPS=%s' % (case['id'], not errors, summary['throughput_makespan_fps']), flush=True)


if __name__ == '__main__':
    main()
