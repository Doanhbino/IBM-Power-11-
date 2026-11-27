#!/usr/bin/env python3
"""Summary of a camera campaign: per camera (shards merged), latency percentiles and smoothness.

Smoothness is the gap between consecutive completed results of one camera. A camera processed at 12 FPS with a
p95 gap of 400 ms is not smooth even though its average rate looks fine. Never prints a camera source.
Usage: live_summary.py runs/<campaign>
"""
import csv
import json
from pathlib import Path
import re
import sys

LATENCY_COLUMNS = ('queue_age_ms', 'inference_ms', 'read_to_result_ms')


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def base_name(camera_id):
    """cam-s0, cam-s1, ... are shards of camera 'cam' (worker_plan --shards)."""
    return re.sub(r'-s\d+$', '', camera_id)


def summarize(run_dir):
    run_dir = Path(run_dir)
    run = json.loads((run_dir / 'run.json').read_text())
    window = run['active_elapsed_s']
    cameras = {}
    for worker in run['workers']:
        for camera in worker['cameras']:
            entry = cameras.setdefault(base_name(camera['id']), dict(received=0, processed=0, dropped=0, errors=0,
                                                                     reconnects=0, shards=0, completions=[], frame_ids=[]))
            entry['received'] += camera['received']
            entry['processed'] += camera['processed']
            entry['dropped'] += camera['queue_dropped']
            entry['errors'] += camera['errors']
            entry['reconnects'] += camera.get('reconnects', 0)
            entry['shards'] += 1
    latency = {key: [] for key in LATENCY_COLUMNS}
    for path in sorted(run_dir.glob('worker-*/samples.csv')):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                entry = cameras[base_name(row['camera_id'])] if base_name(row['camera_id']) in cameras else None
                if entry is None:
                    continue
                entry['completions'].append(float(row['completed_monotonic_s']))
                entry['frame_ids'].append(int(row['frame_id']))
                for key in LATENCY_COLUMNS:
                    latency[key].append(float(row[key]))
    result = {'status': run.get('status'), 'workers': len(run['workers']), 'active_s': window, 'cameras': {},
              'latency_ms': {}}
    for name, entry in cameras.items():
        times = sorted(entry['completions'])
        gaps = [(b - a) * 1000.0 for a, b in zip(times, times[1:])]
        ids = entry['frame_ids']
        result['cameras'][name] = {
            'shards': entry['shards'], 'received': entry['received'], 'processed': entry['processed'],
            'dropped': entry['dropped'], 'dropped_percent': 100.0 * entry['dropped'] / entry['received'] if entry['received'] else 0.0,
            'errors': entry['errors'], 'reconnects': entry['reconnects'],
            'processed_fps': entry['processed'] / window if window else 0.0,
            'duplicate_frame_ids': len(ids) - len(set(ids)),
            'gap_ms_p50': percentile(gaps, 50) if gaps else None, 'gap_ms_p95': percentile(gaps, 95) if gaps else None,
            'gap_ms_max': max(gaps) if gaps else None}
    for key, values in latency.items():
        if values:
            result['latency_ms'][key] = {'p50': percentile(values, 50), 'p95': percentile(values, 95),
                                         'p99': percentile(values, 99), 'max': max(values), 'samples': len(values)}
    return result


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    result = summarize(sys.argv[1])
    print(f"status {result['status']}, {result['workers']} workers, active window {result['active_s']:.1f} s")
    for name, cam in result['cameras'].items():
        print(f"camera {name}: {cam['shards']} shard(s), received {cam['received']}, processed {cam['processed']} "
              f"({cam['processed_fps']:.2f} FPS), dropped {cam['dropped']} ({cam['dropped_percent']:.1f}%), "
              f"errors {cam['errors']}, reconnects {cam['reconnects']}, duplicate frame ids {cam['duplicate_frame_ids']}")
        if cam['gap_ms_p50'] is not None:
            print(f"  gap between consecutive results: p50 {cam['gap_ms_p50']:.0f} ms, p95 {cam['gap_ms_p95']:.0f} ms, "
                  f"max {cam['gap_ms_max']:.0f} ms")
    for key, row in result['latency_ms'].items():
        print(f"{key:<20} p50 {row['p50']:7.1f}  p95 {row['p95']:7.1f}  p99 {row['p99']:7.1f}  max {row['max']:7.1f}  (n={row['samples']})")


if __name__ == '__main__':
    main()
