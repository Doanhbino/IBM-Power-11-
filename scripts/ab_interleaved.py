"""Run camera_service campaigns sequentially in a given interleaved order and summarise per label.

Example: ab_interleaved.py --config R0=configs/a.json --config R1=configs/b.json \
         --order R0,R1,R1,R0,R0,R1 --output-root runs/ab1
Runs never overlap. Each run gets a fresh output directory; load average is recorded before/after.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import sys

MIN_SAMPLES = 300


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def summarize(run_dir, label, index):
    run = json.loads((run_dir / 'run.json').read_text())
    worker = run['workers'][0]
    camera = worker['cameras'][0]
    with (run_dir / 'worker-00' / 'samples.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    inference = [float(row['inference_ms']) for row in rows]
    total = [float(row['read_to_result_ms']) for row in rows]
    return {'label': label, 'index': index, 'run_id': run.get('run_id'), 'status': run.get('status'),
            'samples': len(rows), 'enough_samples': len(rows) >= MIN_SAMPLES,
            'received': camera['received'], 'processed': camera['processed'],
            'dropped': camera['queue_dropped'], 'errors': camera['errors'],
            'fps': run['throughput_active_fps'],
            'inference_mean': st.mean(inference), 'inference_p50': percentile(inference, 50),
            'inference_p95': percentile(inference, 95), 'inference_p99': percentile(inference, 99),
            'read_to_result_p50': percentile(total, 50)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', action='append', required=True, metavar='LABEL=PATH')
    parser.add_argument('--order', required=True, help='comma separated labels, e.g. R0,R1,R1,R0,R0,R1')
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    configs = dict(item.split('=', 1) for item in args.config)
    order = args.order.split(',')
    unknown = set(order) - set(configs)
    if unknown:
        parser.error(f'order uses labels without --config: {sorted(unknown)}')
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name('camera_service.py')
    results = []
    for index, label in enumerate(order, 1):
        run_dir = root / f'{index:02d}-{label}'
        before = os.getloadavg()
        done = subprocess.run([sys.executable, str(script), '--config', configs[label], '--output', str(run_dir)])
        after = os.getloadavg()
        if done.returncode:
            print(f'run {index} {label} failed (exit {done.returncode}); stopping', file=sys.stderr)
            break
        row = summarize(run_dir, label, index)
        row.update(load1_before=before[0], load1_after=after[0])
        results.append(row)
        print(f"{index:02d} {label}: fps={row['fps']:.2f} inf p50={row['inference_p50']:.1f} "
              f"p95={row['inference_p95']:.1f} samples={row['samples']} load1={before[0]:.1f}->{after[0]:.1f}")
    if not results:
        sys.exit(1)
    with (root / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (root / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f"\n{'label':<8}{'n':>3}{'fps mean±sd':>20}{'inf p50':>16}{'inf p95':>16}{'inf p99':>16}")
    for label in dict.fromkeys(order):
        group = [row for row in results if row['label'] == label]
        if not group:
            continue
        cell = lambda key: f"{st.mean(r[key] for r in group):.2f}±{st.pstdev(r[key] for r in group):.2f}"
        print(f"{label:<8}{len(group):>3}{cell('fps'):>20}{cell('inference_p50'):>16}"
              f"{cell('inference_p95'):>16}{cell('inference_p99'):>16}")
    short = [row['index'] for row in results if not row['enough_samples']]
    if short:
        print(f'note: runs {short} have < {MIN_SAMPLES} measured frames; tail percentiles are not reliable')


if __name__ == '__main__':
    main()
