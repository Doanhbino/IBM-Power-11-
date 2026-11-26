#!/usr/bin/env python3
"""Export local RF-DETR checkpoint and save eager FP32 numerical reference on identical inputs."""
import argparse
import inspect
import json
from pathlib import Path
import time
from train import load_model, sha256, versions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, help='JSON with model, export, expected_versions')
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    import numpy as np
    import torch
    vers = versions(cfg.get('expected_versions'))
    export = dict(cfg['export'])
    if export.get('format', 'onnx') != 'onnx':
        raise ValueError('Use the NVIDIA workflow for TensorRT; this script validates FP32 ONNX')
    if export.get('quantization') not in (None, 'fp32'):
        raise ValueError('FP32 baseline only; quantization is a separate accuracy experiment')
    shape = export.get('shape')
    if not shape or len(shape) != 2:
        raise ValueError('Set explicit export.shape=[H,W] valid for the chosen RF-DETR variant')
    export['shape'] = tuple(shape)
    batch = int(export.get('batch_size', 1))
    if batch < 1:
        raise ValueError('batch_size must be positive')
    out = Path(export.get('output_dir', 'runs/export'))
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(cfg, 'cpu')
    started = time.perf_counter()
    # Uses detector's eager forward, before exporter mutates/folds modules. This internal
    # adapter is intentionally fail-fast and recorded; never compare ORT to itself.
    module = model.model.model.cpu().float().eval()
    rng = np.random.default_rng(int(cfg.get('seed', 42)))
    pixels = rng.random((batch, 3, *shape), dtype=np.float32)
    mean = np.asarray(cfg.get('mean', [0.485, 0.456, 0.406]), dtype=np.float32)[None, :, None, None]
    std = np.asarray(cfg.get('std', [0.229, 0.224, 0.225]), dtype=np.float32)[None, :, None, None]
    samples = [(pixels - mean) / std]
    if cfg.get('reference_inputs'):
        loaded = np.load(cfg['reference_inputs'], allow_pickle=False)
        if loaded.ndim != 4 or tuple(loaded.shape[1:]) != (3, *shape):
            raise ValueError('reference_inputs .npy must be normalized float32 NCHW at export shape')
        if len(loaded) % batch:
            raise ValueError('reference_inputs count must be divisible by export batch')
        samples = [loaded[i:i+batch].astype(np.float32) for i in range(0, len(loaded), batch)]
    refs = {}
    with torch.inference_mode():
        for i, x in enumerate(samples):
            result = module(torch.from_numpy(np.ascontiguousarray(x)))
            if not isinstance(result, dict) or not {'pred_boxes', 'pred_logits'} <= result.keys():
                raise RuntimeError('RF-DETR eager adapter changed; expected pred_boxes/pred_logits dict')
            refs[f'input_{i}'] = x
            refs[f'dets_{i}'] = result['pred_boxes'].detach().cpu().numpy()
            refs[f'labels_{i}'] = result['pred_logits'].detach().cpu().numpy()
    np.savez_compressed(out / 'eager_reference.npz', **refs)
    unsupported = set(export) - set(inspect.signature(model.export).parameters)
    if unsupported:
        raise RuntimeError(f'Installed RF-DETR export API does not accept {sorted(unsupported)}; align package revision')
    artifact = model.export(**export)
    artifact = Path(artifact) if artifact is not None else out / 'inference_model.onnx'
    if not artifact.is_file():
        raise FileNotFoundError(f'Exporter artifact not found: {artifact}')
    manifest = {'config': cfg, 'versions': vers, 'artifact': str(artifact), 'sha256': sha256(artifact),
                'reference': str(out / 'eager_reference.npz'), 'reference_cases': len(samples),
                'reference_source': 'FP32 eager model.model.model; pred_boxes->dets, pred_logits->labels',
                'export_seconds_including_reference': time.perf_counter() - started}
    (out / 'export_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
