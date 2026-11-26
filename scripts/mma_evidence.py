#!/usr/bin/env python3
"""Validate an analyst-reviewed MMA evidence dossier against immutable run artifacts.

This checks consistency and trace attachments. It cannot replace expert interpretation
of sampled stacks, build logs or PMU events, and never infers MMA from a provider name.
"""

import argparse
import json
from pathlib import Path
import re

from run_record import atomic_json, sha256


def validate_dossier(path, workload, model_hash, binaries=None, model_files=None):
    path = Path(path).resolve()
    dossier = json.loads(path.read_text())
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    def attachment(value, label):
        if not isinstance(value, dict):
            errors.append(label + ': missing hashed attachment')
            return None
        try:
            target = Path(value['path'])
            if not target.is_absolute():
                target = path.parent / target
            require(target.is_file() and target.stat().st_size > 0, label + ': missing/empty file')
            if not target.is_file():
                return None
            require(sha256(target) == value['sha256'], label + ': hash mismatch')
            return target
        except (KeyError, OSError, TypeError, ValueError):
            errors.append(label + ': invalid attachment')
            return None

    require(dossier.get('schema_version') == 1, 'Unsupported evidence schema')
    require(dossier.get('workload') == workload, 'Training/inference evidence cannot be substituted')
    require(dossier.get('model_sha256') == model_hash, 'Evidence model/checkpoint hash mismatch')
    def model_key(files):
        return sorted(('<model>' if i == 0 else item['file'], item['sha256'])
                      for i, item in enumerate(files))
    if model_files and len(model_files) > 1:
        require(model_key(dossier.get('model_files', [])) == model_key(model_files),
                'External model tensor files differ from the reviewed model')
    require(bool(dossier.get('reviewer')), 'An accountable reviewer is required')
    require(bool(dossier.get('reviewed_utc')), 'Review timestamp is required')
    capability = dossier.get('capability', {})
    require(capability.get('machine') in {'ppc64le', 'powerpc64le'}, 'Evidence target must be little-endian Power')
    require(capability.get('hwcap2_mma') is True, 'LPAR MMA capability has not been demonstrated')
    inventory = attachment(capability.get('inventory'), 'Inventory')
    if inventory:
        try:
            collected = json.loads(inventory.read_text())
            require(collected.get('machine') == capability.get('machine'), 'Inventory architecture mismatch')
            require(collected.get('mma_capability', {}).get('hwcap2_mma') is True,
                    'Inventory did not observe HWCAP2_MMA')
        except (OSError, ValueError):
            errors.append('Invalid inventory JSON')
    stack = dossier.get('stack', {})
    for key in ('revision', 'compiler', 'flags'):
        require(bool(stack.get(key)), 'Missing stack ' + key)
    attachment(stack.get('build_log'), 'Build log')
    stack_binaries = stack.get('binaries', [])
    require(bool(stack_binaries), 'Native runtime binaries are required')
    for item in stack_binaries:
        attachment(item, 'Runtime binary')
    if binaries is not None:
        hashes = {item['sha256'] for item in binaries}
        supplied = {item.get('sha256') for item in stack_binaries}
        require(bool(hashes) and hashes <= supplied, 'Current runtime binaries differ from the reviewed stack')
    run_path = attachment(dossier.get('run'), 'Profiled run record')
    record = {}
    if run_path:
        try:
            record = json.loads(run_path.read_text())
            require(record.get('run_id') == dossier.get('run_id') and bool(record.get('run_id')), 'Profile run ID mismatch')
            require(record.get('workload') in ({'inference', 'camera'} if workload == 'inference' else {'training'}),
                    'Profile run workload mismatch')
            require(record.get('model_sha256') == model_hash, 'Profile run model hash mismatch')
            require(record.get('machine') == capability.get('machine'), 'Profile run architecture differs from the LPAR inventory')
            if model_files and len(model_files) > 1:
                require(model_key(record.get('model_files', [])) == model_key(model_files),
                        'External tensor files differ from the profiled run')
            require(record.get('profile_instrumented') is True, 'Run record must identify instrumentation')
            require(record.get('status') in {'ok', 'completed'}, 'Profile run did not complete')
        except (OSError, ValueError):
            errors.append('Invalid profiled run JSON')
    attachment(dossier.get('operator_profile'), 'RF-DETR operator profile')
    require(bool(record.get('operator_profile')) and
            record.get('operator_profile', {}).get('sha256') == dossier.get('operator_profile', {}).get('sha256'),
            'Operator profile is not linked by hash from the profiled run')
    hotspots = dossier.get('hotspots', [])
    require(bool(hotspots), 'No RF-DETR hotspots linked to kernel traces')
    phases = set()
    for index, hotspot in enumerate(hotspots):
        label = f'Hotspot {index}'
        for key in ('node', 'operator', 'shape', 'dtype', 'kernel_symbol', 'phase'):
            require(bool(hotspot.get(key)), label + ': missing ' + key)
        phases.add(hotspot.get('phase'))
        require(hotspot.get('mma_executed') is True, label + ': reviewer has not confirmed MMA execution')
        require(bool(hotspot.get('interpretation')), label + ': explain how the trace proves execution')
        trace = attachment(hotspot.get('sampled_stack'), label + ' sampled stack')
        if trace:
            symbol = hotspot.get('kernel_symbol')
            require(isinstance(symbol, str) and bool(symbol) and
                    re.search(re.escape(symbol), trace.read_text(errors='replace')) is not None,
                    label + ': kernel symbol absent from sampled trace')
    if workload == 'training':
        require({'forward', 'backward'} <= phases, 'Training requires independent forward AND backward evidence')
    return {'status': 'dossier_valid' if not errors else 'rejected', 'errors': errors,
            'dossier_sha256': sha256(path), 'workload': workload,
            'scope': 'Reviewed evidence consistency; not automatic proof of instruction execution or performance gain'}


def require_evidence(config, workload, model_hash, binaries, model_files=None):
    path = config.get('mma_evidence')
    if not path:
        if config.get('require_mma', False):
            raise RuntimeError('require_mma=true requires a reviewed MMA evidence dossier')
        return {'status': 'unverified', 'workload': workload}
    result = validate_dossier(path, workload, model_hash, binaries, model_files)
    if result['errors']:
        raise RuntimeError('MMA evidence rejected: ' + '; '.join(result['errors']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--workload', choices=('training', 'inference'), required=True)
    parser.add_argument('--model', required=True, help='ONNX model, or initial checkpoint for training')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    files = None
    if args.workload == 'inference':
        from common import model_files
        files = model_files(args.model)
    result = validate_dossier(args.evidence, args.workload, sha256(args.model), model_files=files)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not result['errors'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
