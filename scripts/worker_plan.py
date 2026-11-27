#!/usr/bin/env python3
"""Generate one camera worker per observed NUMA node without assuming CPU numbering."""

import argparse
import json
import os
from pathlib import Path

from inventory import topology
from run_record import atomic_json
from sweep import selected_cpus


PLACEMENT_KEYS = {'cpus', 'numa_node', 'threads', 'sources'}


def place(config, rows, threads=4, smt_limit=1, max_workers=4):
    """One worker per NUMA node. Input workers that set different model keys form separate groups;
    nodes are dealt to the groups round-robin so each model keeps its own nodes (one group = old behavior)."""
    if min(threads, smt_limit, max_workers) < 1:
        raise ValueError('threads, smt_limit and max_workers must be positive')
    sources = [source for worker in config['workers'] for source in worker['sources']]
    if not sources or len({source['id'] for source in sources}) != len(sources):
        raise ValueError('Provide globally unique camera sources in the input config')
    groups = {}
    for worker in config['workers']:
        keys = {key: value for key, value in worker.items() if key not in PLACEMENT_KEYS}
        group = groups.setdefault(json.dumps(keys, sort_keys=True), {'keys': keys, 'sources': []})
        group['sources'] += worker['sources']
    pools = {}
    for row in selected_cpus(rows, smt_limit):
        if row['node'] is not None:
            pools.setdefault(row['node'], []).append(row['cpu'])
    nodes = [(node, cpus) for node, cpus in sorted(pools.items()) if len(cpus) >= threads]
    if not nodes:
        raise ValueError('No observed NUMA node has enough allowed CPUs; reduce threads or inspect topology')
    if len(groups) > len(nodes):
        raise ValueError('More model groups than usable NUMA nodes; reduce groups or threads')
    workers = []
    for index, group in enumerate(groups.values()):
        mine = [{**group['keys'], 'cpus': sorted(cpus)[:threads], 'numa_node': node, 'threads': threads,
                 'sources': []} for node, cpus in nodes[index::len(groups)]]
        mine = mine[:min(max_workers, len(group['sources']))]
        for position, source in enumerate(sorted(group['sources'], key=lambda source: source['id'])):
            mine[position % len(mine)]['sources'].append(source)
        workers += mine
    return dict(config, workers=workers,
                placement={'smt_limit': smt_limit, 'threads_per_worker': threads,
                           'topology': rows, 'policy': 'one worker per observed NUMA node; disjoint masks; '
                                                       'model groups get separate nodes'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--smt-limit', type=int, default=1)
    parser.add_argument('--max-workers', type=int, default=4)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError('Use a new config file to preserve the previous placement')
    result = place(json.loads(Path(args.config).read_text()), topology(),
                   args.threads, args.smt_limit, args.max_workers)
    # A generated config may contain live camera credentials; retain owner-only access.
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    atomic_json(path, result)
    path.chmod(0o600)
    print(f'{len(result["workers"])} worker(s) planned in {path}')


if __name__ == '__main__':
    main()
