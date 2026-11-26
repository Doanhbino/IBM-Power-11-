#!/usr/bin/env python3
"""Unprivileged telemetry, with absent PowerVM/GPU sensors explicitly recorded."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import time


def fields(path):
    try:
        return Path(path).read_text().splitlines()
    except OSError:
        return []


def sample(pid=None):
    now = time.time()
    row = {'unix_s': now, 'monotonic_s': time.monotonic()}
    stat = fields('/proc/stat')
    for line in stat:
        if line.startswith('cpu '):
            nums = [int(x) for x in line.split()[1:]]
            # guest/guest_nice already included in user/nice; do not count twice.
            row['cpu_ticks_total'] = sum(nums[:8])
            row['cpu_ticks_idle_iowait'] = nums[3] + nums[4]
            row['cpu_ticks_steal'] = nums[7] if len(nums) > 7 else 0
    wanted = {'MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree', 'Dirty'}
    for line in fields('/proc/meminfo'):
        key, value = line.split(':', 1)
        if key in wanted:
            row[key + '_kib'] = int(value.split()[0])
    for line in fields('/proc/ppc64/lparcfg'):
        if '=' in line:
            k, v = line.split('=', 1)
            if k in {'purr', 'spurr', 'cmo_faults', 'cmo_fault_time_usec',
                     'partition_entitled_capacity', 'partition_active_processors',
                     'shared_processor_mode', 'pool_idle_time', 'pool_num_procs'}:
                row['lpar_' + k] = v
    row['load1'], row['load5'], row['load15'] = os.getloadavg()
    if pid is not None:
        proc = Path('/proc') / str(pid)
        for line in fields(proc / 'status'):
            for key in ('VmRSS', 'VmHWM', 'Threads'):
                if line.startswith(key + ':'):
                    row['process_' + key] = int(line.split()[1])
        for line in fields(proc / 'io'):
            key, val = line.split(':', 1)
            if key in ('read_bytes', 'write_bytes'):
                row['process_' + key] = int(val)
        row['process_alive'] = proc.exists()
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='runs/telemetry')
    ap.add_argument('--pid', type=int)
    ap.add_argument('--interval', type=float, default=1.0)
    ap.add_argument('--seconds', type=float, default=60.0)
    args = ap.parse_args()
    if args.interval <= 0 or args.seconds <= 0:
        ap.error('interval and seconds must be positive')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stop = False
    def stopping(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    metadata = {'schema_version': 1, 'pid': args.pid, 'interval_requested_s': args.interval,
                'external_tools': {k: shutil.which(k) for k in ['lparstat', 'numastat', 'perf', 'iostat', 'nvidia-smi']},
                'lparcfg_readable': bool(fields('/proc/ppc64/lparcfg')),
                'limits': 'procfs CPU ticks are whole-host/LPAR, process RSS excludes child workers. No energy, temperature, memory bandwidth or actual MMA instruction counters measured here. Use separate approved vendor/perf tools.'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    rows = []
    start = time.monotonic()
    previous = None
    with (out / 'samples.jsonl').open('w') as stream:
        while not stop and time.monotonic() - start < args.seconds:
            row = sample(args.pid)
            if previous and row.get('cpu_ticks_total', 0) > previous.get('cpu_ticks_total', 0):
                dt = row['cpu_ticks_total'] - previous['cpu_ticks_total']
                di = row['cpu_ticks_idle_iowait'] - previous['cpu_ticks_idle_iowait']
                row['cpu_busy_pct'] = 100.0 * (1 - di / dt)
                row['cpu_steal_pct'] = 100.0 * (row['cpu_ticks_steal'] - previous['cpu_ticks_steal']) / dt
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            rows.append(row)
            previous = row
            time.sleep(min(args.interval, max(0, args.seconds - (time.monotonic() - start))))
    keys = sorted({k for row in rows for k in row})
    with (out / 'samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == '__main__':
    main()
