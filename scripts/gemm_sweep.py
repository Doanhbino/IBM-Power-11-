#!/usr/bin/env python3
"""FP32 reference/MMA microbenchmarks using shapes from a real operator profile."""

import argparse
import json
from pathlib import Path
import subprocess

from common import stats
from run_record import artifact, atomic_json, identity, sha256


def run(binary, shapes, output, backends, repeats=3, iterations=20):
    if repeats < 3 or iterations < 1:
        raise ValueError('Use >= 3 repeats and >= 1 iteration')
    binary = Path(binary).resolve(strict=True)
    source = json.loads(Path(shapes).read_text())
    if source.get('schema_version') != 1 or not source.get('cases'):
        raise ValueError('No concrete profile shapes; do not substitute invented RF-DETR dimensions')
    if sha256(source['profile']['path']) != source['profile']['sha256']:
        raise ValueError('Source profile changed after shape extraction')
    graph = source.get('optimized_model')
    if graph and sha256(graph['path']) != graph['sha256']:
        raise ValueError('Optimized graph changed after shape extraction')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    record = {**identity('gemm_microbenchmark'), 'status': 'running', 'binary': artifact(binary),
              'shapes': artifact(shapes), 'profile': source['profile'], 'results': [],
              'scope': 'Experimental 8x16 (8-accumulator) kernel, not linked into ORT/PyTorch; not RF-DETR execution evidence or optimized BLAS comparison'}
    atomic_json(out / 'run.json', record)
    try:
        capability = json.loads(subprocess.check_output([str(binary), '--capability'], text=True))
        record['mma_capability'] = capability
        if 'mma' in backends and not (capability['compiled'] and capability['exposed']):
            raise RuntimeError('MMA backend unavailable; refusing silent reference fallback')
        seen = set()
        for case in source['cases']:
            if any(str(case[key]).lower() not in {'float', 'float32'} for key in ('input_dtype', 'rhs_dtype')):
                raise ValueError('FP32 kernel cannot benchmark another precision')
            key = (case['m'], case['n'], case['k'], bool(case['transpose_b']))
            if key in seen:
                continue
            seen.add(key)
            for backend in backends:
                runs = []
                for repeat in range(repeats):
                    argv = [str(binary), *map(str, key[:3]), backend, str(iterations), str(int(key[3]))]
                    result = subprocess.run(argv, text=True, capture_output=True, check=True)
                    raw = json.loads(result.stdout)
                    atomic_json(out / f'case-{len(record["results"]):04d}-r{repeat}.json', raw)
                    runs.append(raw)
                record['results'].append({'shape': case, 'backend': backend, 'repeats': repeats,
                    'latency_ms': stats([value for raw in runs for value in raw['samples_ms']]),
                    'repeat_mean_ms': stats([sum(raw['samples_ms']) / len(raw['samples_ms']) for raw in runs]),
                    'pack_b_ms': stats([raw['pack_b_ms'] for raw in runs]),
                    'max_abs_error': max(raw['max_abs_error'] for raw in runs)})
                atomic_json(out / 'run.json', record)
        record['status'] = 'ok'
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        atomic_json(out / 'run.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='build/rfdetr_gemm')
    parser.add_argument('--shapes', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--backends', nargs='+', choices=('reference', 'mma'), default=['reference', 'mma'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args()
    run(args.binary, args.shapes, args.output, args.backends, args.repeats, args.iterations)


if __name__ == '__main__':
    main()
