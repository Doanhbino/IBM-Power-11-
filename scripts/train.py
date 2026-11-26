#!/usr/bin/env python3
"""Config-driven RF-DETR fine-tuning; no package installation or silent device fallback."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
from run_record import identity, runtime_binaries, atomic_json, sha256
from mma_evidence import require_evidence
from training_data import inspect_dataset


def versions(expected=None):
    result = {}
    for package in ('rfdetr', 'torch', 'torchvision', 'numpy', 'onnx', 'onnxruntime'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    for package, required in (expected or {}).items():
        actual = importlib.metadata.version(package)
        if actual != required:
            raise RuntimeError(f'{package}: expected {required}, installed {actual}')
    return result


def load_model(config, device):
    import rfdetr
    model_cfg = dict(config['model'])
    name = model_cfg.pop('class')
    if name not in ('RFDETRNano', 'RFDETRSmall', 'RFDETRMedium', 'RFDETRLarge', 'RFDETRBase'):
        raise ValueError('Baseline supports detection Nano/Small/Medium/Large/Base only')
    kwargs = dict(model_cfg.get('kwargs', {}))
    weights = kwargs.get('pretrain_weights')
    if not weights or not Path(weights).is_file():
        raise FileNotFoundError('Set model.kwargs.pretrain_weights to an existing local trusted checkpoint')
    kwargs['device'] = device
    if config.get('load_from_checkpoint', False):
        base = getattr(rfdetr, 'RFDETR', None)
        if base is None or not hasattr(base, 'from_checkpoint'):
            raise RuntimeError('Installed rfdetr lacks RFDETR.from_checkpoint; use matching model constructor')
        kwargs.pop('pretrain_weights')
        return base.from_checkpoint(weights, **kwargs)
    return getattr(rfdetr, name)(**kwargs)


def main(profile_instrumented=False):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--check-only', action='store_true', help='Validate config/dataset without importing torch or RF-DETR; YOLO requires PyYAML')
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    train = dict(cfg['train'])
    device = train.get('device')
    if device not in ('cpu', 'cuda'):
        raise ValueError('Set train.device explicitly to cpu or cuda')
    if not isinstance(train.get('batch_size'), int) or train['batch_size'] < 1:
        raise ValueError('Use explicit positive integer batch_size for reproducible comparisons')
    if train.get('amp_dtype', 'missing') is not None:
        raise ValueError('This baseline requires explicit amp_dtype=null (FP32); use separate AMP experiment')
    data = inspect_dataset(train)
    train['dataset_dir'] = data['root']
    train['dataset_file'] = data['format']
    cfg['train'] = train
    weights = cfg['model'].get('kwargs', {}).get('pretrain_weights')
    if not weights or not Path(weights).is_file():
        raise FileNotFoundError('Local pretrain_weights missing')
    if args.check_only:
        print(json.dumps({'status': 'config_files_ok', 'training_not_run': True,
                          'dataset': {key: value for key, value in data.items() if key != 'annotation_files'}}))
        return
    if device == 'cuda' and os.environ.get('NVIDIA_TF32_OVERRIDE') != '0':
        raise RuntimeError('Strict FP32 baseline requires launching with NVIDIA_TF32_OVERRIDE=0 before Python starts; RF-DETR trainer may override PyTorch TF32 flags')
    import numpy as np
    import torch
    recorded_versions = versions(cfg.get('expected_versions'))
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; refusing CPU fallback')
    seed = int(train['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(int(cfg.get('torch_threads', 1)))
    torch.set_num_interop_threads(int(cfg.get('torch_interop_threads', 1)))
    torch.use_deterministic_algorithms(bool(cfg.get('deterministic_algorithms', False)))
    out = Path(train['output_dir'])
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Training requires a new/empty output_dir to preserve run provenance')
    out.mkdir(parents=True, exist_ok=True)
    manifest = {**identity('training'), 'config': cfg, 'versions': recorded_versions, 'python': platform.python_version(),
                'machine': platform.machine(), 'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                'weights_sha256': sha256(weights), 'annotations_sha256': {p: sha256(p) for p in data['annotation_files']},
                'dataset': {key: value for key, value in data.items() if key != 'annotation_files'},
                'time_scope': 'model construction separately; train wall includes data loading, forward/backward, validation and checkpoint IO',
                'status': 'running', 'NVIDIA_TF32_OVERRIDE': os.environ.get('NVIDIA_TF32_OVERRIDE'),
                'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')}
    manifest['model_sha256'] = manifest['weights_sha256']
    manifest['profile_instrumented'] = profile_instrumented
    manifest['runtime_binaries'] = runtime_binaries(torch)
    manifest['torch_build_config'] = torch.__config__.show()
    if device == 'cpu':
        manifest['mma_evidence'] = require_evidence(cfg, 'training', manifest['model_sha256'], manifest['runtime_binaries'])
    atomic_json(out / 'baseline_manifest.json', manifest)
    started = time.perf_counter()
    try:
        model = load_model(cfg, device)
        manifest['model_load_seconds'] = time.perf_counter() - started
        started = time.perf_counter()
        model.train(**train)
        if device == 'cuda':
            torch.cuda.synchronize()
        manifest['train_wall_seconds'] = time.perf_counter() - started
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        manifest['max_rss_kib_linux'] = __import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss
        atomic_json(out / 'baseline_manifest.json', manifest)
        with open(out / 'pip-freeze.txt', 'w') as f:
            subprocess.run([os.sys.executable, '-m', 'pip', 'freeze'], stdout=f, check=False)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
