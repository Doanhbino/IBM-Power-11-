#!/usr/bin/env python3
"""Fail on structural, shape, NaN, or eager-vs-ORT FP32 parity failure."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--backend-config', help='Optional CPU/ORT-TRT/native TRT config; model path must agree')
    ap.add_argument('--reference', required=True, help='NPZ of FP32 reference tensors from an independent PyTorch eager run; never an ORT-generated reference')
    ap.add_argument('--output', default='runs/validation.json')
    ap.add_argument('--atol', type=float, default=1e-4)
    ap.add_argument('--rtol', type=float, default=1e-3)
    ap.add_argument('--boxes-output', default='dets')
    ap.add_argument('--logits-output', default='labels')
    args = ap.parse_args()
    import numpy as np
    import onnx
    onnx.checker.check_model(args.model, full_check=True)
    if args.backend_config:
        from common import session as make_session
        cfg=json.loads(Path(args.backend_config).read_text())
        if Path(cfg['model']).resolve()!=Path(args.model).resolve():raise ValueError('Backend config model differs from --model')
        session=make_session(cfg)
        runtime_version=__import__('tensorrt').__version__ if cfg.get('backend')=='tensorrt_native' else __import__('onnxruntime').__version__
    else:
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        session=ort.InferenceSession(args.model,sess_options=options,providers=['CPUExecutionProvider'])
        runtime_version=ort.__version__
    if len(session.get_inputs()) != 1:
        raise ValueError('Baseline expects one image input; additional masks need explicit adapter')
    names = [o.name for o in session.get_outputs()]
    if args.boxes_output not in names or args.logits_output not in names:
        raise ValueError(f'Explicit output names required; available: {names}')
    archive = np.load(args.reference, allow_pickle=False)
    cases = sorted(int(k.removeprefix('input_')) for k in archive.files if k.startswith('input_'))
    if not cases:
        raise ValueError('No eager reference cases')
    report = {'model': args.model, 'reference': args.reference, 'atol': args.atol, 'rtol': args.rtol,
              'provider': session.get_providers(), 'runtime_version': runtime_version, 'cases': [], 'passed': True}
    for i in cases:
        outputs = dict(zip(names, session.run(None, {session.get_inputs()[0].name: archive[f'input_{i}']})))
        for actual_name, reference_name in [(args.boxes_output, f'dets_{i}'), (args.logits_output, f'labels_{i}')]:
            actual, expected = outputs[actual_name], archive[reference_name]
            same_shape = actual.shape == expected.shape
            finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
            passed = same_shape and finite and bool(np.allclose(actual, expected, rtol=args.rtol, atol=args.atol))
            report['passed'] &= passed
            report['cases'].append({'case': i, 'output': actual_name, 'actual_shape': list(actual.shape),
                                    'expected_shape': list(expected.shape), 'finite': finite, 'passed': passed,
                                    'max_abs_error': float(np.max(np.abs(actual - expected))) if same_shape and finite else None})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
