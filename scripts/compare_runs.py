#!/usr/bin/env python3
"""Compare repeated, unprofiled model-only runs with matching workload boundaries."""

import argparse
import json
from pathlib import Path
import statistics

from run_record import artifact, atomic_json


def load_group(paths):
    records = []
    seen = set()
    for path in paths:
        path = Path(path)
        if path.is_dir():
            path = path / 'result.json'
        record = json.loads(path.read_text())
        if path.resolve() in seen or (record.get('run_id') and record['run_id'] in seen):
            raise ValueError('Repeated references to the same run are not independent repeats')
        seen.add(path.resolve())
        if record.get('run_id'):
            seen.add(record['run_id'])
        if record.get('status') != 'ok' or record.get('profile_instrumented'):
            raise ValueError('Only successful unprofiled runs qualify as KPI repeats')
        if record.get('config', {}).get('mode', 'model') != 'model':
            raise ValueError('This comparison is model-only; camera timing has different boundaries')
        if 'synthetic' in record.get('result_kind', ''):
            raise ValueError('Synthetic smoke results cannot qualify as application KPI')
        if record.get('images', 0) <= 0 or record.get('throughput_fps', 0) <= 0:
            raise ValueError('Invalid measured sample count/throughput')
        records.append((record, artifact(path)))
    if len(records) < 3:
        raise ValueError('At least three distinct runs per configuration are required')
    return records


def workload_key(record):
    return (record['model_sha256'], record['input_sha256'], tuple(record['input_shape']),
            record.get('input_dtype'), record['timing_boundary'], record.get('result_kind'),
            tuple(item['sha256'] for item in record.get('model_files', [])))


def resource_key(record):
    config = record['config']
    return (record.get('machine'), tuple(record.get('affinity') or []), config.get('threads'),
            config.get('inter_threads', 1), config.get('spinning', False))


def configuration_key(record):
    return {key: value for key, value in record['config'].items()
            if key not in {'profile_prefix', 'sweep_case'}}


def summarize(records):
    fps = [record['throughput_fps'] for record, _ in records]
    return {'repeats': len(records), 'images': sum(record['images'] for record, _ in records),
            'throughput_fps': {'mean': statistics.mean(fps), 'stdev': statistics.stdev(fps),
                               'min': min(fps), 'max': max(fps)},
            'latency_repeat_p50_ms': [record['latency_ms']['p50'] for record, _ in records],
            'latency_repeat_p95_ms': [record['latency_ms']['p95'] for record, _ in records],
            'latency_repeat_p99_ms': [record['latency_ms']['p99'] for record, _ in records]}


def compare(reference, candidate, comparison='stack'):
    groups = [load_group(reference), load_group(candidate)]
    baseline = workload_key(groups[0][0][0])
    ids = [record.get('run_id') for group in groups for record, _ in group if record.get('run_id')]
    if len(ids) != len(set(ids)):
        raise ValueError('Reference and candidate must use independent runs')
    for group in groups:
        for record, _ in group:
            if workload_key(record) != baseline:
                raise ValueError('Model/input/precision/timing boundary changed across A/B runs')
        config = configuration_key(group[0][0])
        resources = resource_key(group[0][0])
        runtime = [item['sha256'] for item in group[0][0].get('runtime_binaries', [])]
        for record, _ in group:
            if configuration_key(record) != config or [item['sha256'] for item in record.get('runtime_binaries', [])] != runtime:
                raise ValueError('Each group must repeat one unchanged configuration and runtime')
            if resource_key(record) != resources:
                raise ValueError('CPU/thread resources changed within one repeat group')
    if comparison == 'kernel':
        resources = resource_key(groups[0][0][0])
        if any(resource_key(record) != resources for group in groups for record, _ in group):
            raise ValueError('Kernel A/B requires identical CPU/thread resources')
    summaries = [summarize(group) for group in groups]
    r, c = [group['throughput_fps'] for group in summaries]
    return {'schema_version': 1, 'comparison': comparison, 'reference': summaries[0], 'candidate': summaries[1],
            'throughput_speedup': c['mean'] / r['mean'],
            'candidate_range_above_reference': c['min'] > r['max'],
            'runs': {'reference': [item for _, item in groups[0]], 'candidate': [item for _, item in groups[1]]},
            'scope': 'Empirical repeat variation, not a significance test or acceptance. Accuracy/parity, camera SLA and MMA trace gates remain independent. Stack/hardware gain cannot be attributed solely to MMA.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', nargs='+', required=True)
    parser.add_argument('--candidate', nargs='+', required=True)
    parser.add_argument('--comparison', choices=('stack', 'kernel', 'hardware'), default='stack')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.comparison)
    atomic_json(args.output, result)
    print(json.dumps({'throughput_speedup': result['throughput_speedup'],
                      'candidate_range_above_reference': result['candidate_range_above_reference']}))


if __name__ == '__main__':
    main()
