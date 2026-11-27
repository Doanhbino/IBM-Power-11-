#!/usr/bin/env python3
"""CPU use per thread of a running process, grouped by affinity mask. Read-only (/proc only).

Typical use: measure what video decoding costs, by running the worker with capture_cpus and sampling the threads
whose mask equals that set:
  thread_cpu.py --comm rfdetr_camera --worker-config configs/cam-local.json --window 30
Everything is attributed over the same wall-clock window, so percentages are of one CPU (100 = one core busy).
"""
import argparse
import json
import os
import subprocess
import sys
import time

TICK = os.sysconf('SC_CLK_TCK')


def parse_cpulist(text):
    cpus = set()
    for part in text.strip().split(','):
        if not part:
            continue
        low, _, high = part.partition('-')
        cpus.update(range(int(low), int(high or low) + 1))
    return frozenset(cpus)


def snapshot(pid):
    """{tid: {'ticks': utime+stime, 'allowed': frozenset(cpus), 'comm': str}} for live threads."""
    result = {}
    for tid in os.listdir(f'/proc/{pid}/task'):
        try:
            with open(f'/proc/{pid}/task/{tid}/stat') as stream:
                stat = stream.read()
            with open(f'/proc/{pid}/task/{tid}/status') as stream:
                status = stream.read()
        except OSError:
            continue  # thread exited between listing and reading
        rest = stat[stat.rindex(')') + 2:].split()      # field 3 (state) is rest[0]
        allowed = next(line.split()[1] for line in status.splitlines() if line.startswith('Cpus_allowed_list'))
        result[int(tid)] = {'ticks': int(rest[11]) + int(rest[12]), 'allowed': parse_cpulist(allowed),
                            'comm': stat[stat.index('(') + 1:stat.rindex(')')]}
    return result


def measure(pid, window, group_cpus=None):
    before = snapshot(pid)
    started = time.monotonic()
    time.sleep(window)
    after = snapshot(pid)
    elapsed = time.monotonic() - started
    threads = []
    for tid, row in after.items():
        if tid not in before:
            continue  # started inside the window: no baseline
        percent = 100.0 * (row['ticks'] - before[tid]['ticks']) / TICK / elapsed
        threads.append({'tid': tid, 'allowed': sorted(row['allowed']), 'cpu_percent': percent})
    in_group = [t for t in threads if group_cpus is not None and frozenset(t['allowed']) == group_cpus]
    others = [t for t in threads if t not in in_group]
    return {'pid': pid, 'window_s': elapsed, 'threads': sorted(threads, key=lambda t: -t['cpu_percent']),
            'group_threads': len(in_group), 'group_cpu_percent': sum(t['cpu_percent'] for t in in_group),
            'other_threads': len(others), 'other_cpu_percent': sum(t['cpu_percent'] for t in others)}


def find_pid(comm):
    done = subprocess.run(['pgrep', '-nx', comm], capture_output=True, text=True)
    return int(done.stdout.split()[0]) if done.returncode == 0 and done.stdout.split() else None


def wait_for_group(pid, group_cpus, timeout):
    """Capture threads start only after model load and warmup, so wait until threads with the mask exist."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if any(row['allowed'] == group_cpus for row in snapshot(pid).values()):
                return True
        except OSError:
            return False
        time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pid', type=int)
    parser.add_argument('--comm', help='process name; the newest match is used (waits up to --wait seconds)')
    parser.add_argument('--group-cpus', help='CPU list such as 128,160 identifying the threads to group')
    parser.add_argument('--worker-config', help='generated worker config; uses workers[0].capture_cpus as the group')
    parser.add_argument('--window', type=float, default=30)
    parser.add_argument('--wait', type=float, default=150)
    parser.add_argument('--settle', type=float, default=5, help='seconds to wait after the group appears')
    args = parser.parse_args()
    group = None
    if args.worker_config:
        group = frozenset(json.load(open(args.worker_config))['workers'][0]['capture_cpus'])
    elif args.group_cpus:
        group = parse_cpulist(args.group_cpus)
    pid = args.pid
    deadline = time.monotonic() + args.wait
    while pid is None and time.monotonic() < deadline:
        pid = find_pid(args.comm) if args.comm else None
        if pid is None:
            time.sleep(1)
    if pid is None:
        sys.exit('process not found')
    if group is not None:
        if not wait_for_group(pid, group, max(1, deadline - time.monotonic())):
            sys.exit('no thread with the capture mask appeared (binary without capture_cpus support, or capture not started)')
        time.sleep(args.settle)
    result = measure(pid, args.window, group)
    print(f"pid {pid}, window {result['window_s']:.1f}s, percentages are of one CPU")
    print(f"{'tid':>9}  {'allowed cpus':<24} {'cpu %':>7}")
    for row in result['threads'][:12]:
        cpus = ','.join(map(str, row['allowed']))
        print(f"{row['tid']:>9}  {cpus[:24]:<24} {row['cpu_percent']:7.1f}")
    if group is not None:
        print(f"capture/decode group ({','.join(map(str, sorted(group)))}): {result['group_threads']} threads, "
              f"{result['group_cpu_percent']:.1f}% of one CPU")
        print(f"all other threads: {result['other_threads']} threads, {result['other_cpu_percent']:.1f}% of one CPU")


if __name__ == '__main__':
    main()
