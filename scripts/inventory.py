#!/usr/bin/env python3
"""Read-only inventory; works without root or third party packages."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import ctypes


def read(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def cpu_list(value):
    out = set()
    for item in (value or '').split(','):
        if not item.strip():
            continue
        bounds = item.strip().split('-')
        if len(bounds) == 1:
            out.add(int(bounds[0]))
        else:
            out.update(range(int(bounds[0]), int(bounds[1]) + 1))
    return sorted(out)


def topology():
    allowed = set(os.sched_getaffinity(0))
    online = set(cpu_list(read('/sys/devices/system/cpu/online')))
    rows = []
    for cpu in sorted(allowed & online):
        base = Path('/sys/devices/system/cpu') / ('cpu%d' % cpu)
        nodes = sorted(int(p.name[4:]) for p in base.glob('node[0-9]*'))
        siblings = cpu_list(read(base / 'topology/thread_siblings_list'))
        rows.append({'cpu': cpu, 'node': nodes[0] if nodes else None,
                     'package': read(base / 'topology/physical_package_id'),
                     'core': read(base / 'topology/core_id'),
                     'siblings_allowed': sorted(set(siblings) & allowed & online)})
    return rows


def command(args):
    if not shutil.which(args[0]):
        return {'status': 'missing', 'argv': args}
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=15)
        return {'status': 'ok' if p.returncode == 0 else 'failed',
                'argv': args, 'returncode': p.returncode,
                'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'status': 'failed', 'argv': args, 'error': str(e)}


def mma_capability():
    """Linux Power HWCAP2_MMA. CPU model names and PCI devices are not ISA proof."""
    machine = platform.machine()
    result = {'machine': machine, 'hwcap2': None, 'hwcap2_mma': False,
              'kernel_execution': 'unverified; requires RF-DETR hot-path profiling'}
    if machine not in {'ppc64le', 'powerpc64le'}:
        result['status'] = 'not_power_target'
        return result
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        getauxval = libc.getauxval
        getauxval.argtypes = [ctypes.c_ulong]
        getauxval.restype = ctypes.c_ulong
        ctypes.set_errno(0)
        value = getauxval(26)  # AT_HWCAP2 from Linux auxiliary-vector ABI.
        if ctypes.get_errno():
            raise OSError(ctypes.get_errno(), 'getauxval AT_HWCAP2 failed')
        result.update(hwcap2=value, hwcap2_mma=bool(value & 0x00020000),
                      status='exposed' if value & 0x00020000 else 'not_exposed')
    except (AttributeError, OSError) as error:
        result.update(status='unknown', error=str(error))
    return result


def collect():
    pci = []
    for dev in sorted(Path('/sys/bus/pci/devices').glob('*')):
        pci.append({'address': dev.name, 'vendor': read(dev / 'vendor'),
                    'device': read(dev / 'device'), 'class': read(dev / 'class'),
                    'numa_node': read(dev / 'numa_node'),
                    'driver': os.path.basename(os.path.realpath(dev / 'driver'))
                    if (dev / 'driver').exists() else None})
    disk_path = '/home/models' if Path('/home/models').exists() else str(Path.cwd())
    usage = shutil.disk_usage(disk_path)
    cgroup_files = {}
    for line in (read('/proc/self/cgroup') or '').splitlines():
        if line.startswith('0::'):
            cg = Path('/sys/fs/cgroup') / line[3:].lstrip('/')
            for name in ['cpu.max', 'cpuset.cpus.effective', 'memory.max', 'memory.current']:
                cgroup_files[str(cg / name)] = read(cg / name)
    return {'schema_version': 1, 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'host': platform.node(), 'machine': platform.machine(),
            'mma_capability': mma_capability(),
            'uname': list(platform.uname()), 'python': platform.python_version(),
            'cpu_topology_allowed': topology(),
            'files': {p: read(p) for p in ['/etc/os-release', '/proc/cpuinfo', '/proc/meminfo',
                         '/proc/ppc64/lparcfg', '/proc/self/cgroup', '/sys/devices/system/node/online',
                         '/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/cpuset.cpus.effective',
                         '/sys/fs/cgroup/memory.max']},
            'current_cgroup_v2_files': cgroup_files,
            'numa_memory': {str(p): read(p) for p in sorted(Path('/sys/devices/system/node').glob('node[0-9]*/meminfo'))},
            'disk': {'path': disk_path, 'total_bytes': usage.total, 'used_bytes': usage.used,
                     'free_bytes': usage.free},
            'pci': pci,
            'spyre_status': 'unverified: match lspci device identity + assigned driver + vendor runtime health; CPU MMA is not PCIe Spyre',
            'commands': [command(c) for c in [['lscpu', '-e'], ['numactl', '--hardware'],
                         ['lparstat', '-i'], ['lspci', '-Dnnk'],
                         ['gcc', '--version'], ['g++', '--version'], ['cmake', '--version'],
                         ['perf', '--version'],
                         ['rpm', '-qa', '--qf', '%{NAME}-%{VERSION}-%{RELEASE}.%{ARCH}\n']]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='runs/inventory.json')
    args = ap.parse_args()
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = collect()
    dest.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(dest)


if __name__ == '__main__':
    main()
