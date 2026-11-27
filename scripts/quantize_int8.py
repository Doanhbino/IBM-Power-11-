#!/usr/bin/env python3
"""Post-training INT8 (QDQ) quantization of the constant-weight linear layers of an ONNX model.

Why: on Power10/11 the INT8 outer-product instruction (xvi8ger4) does 4x the multiply-accumulates of the FP32 one
(xvf32ger) per instruction (IBM REDP-5612), and ONNX Runtime's MLAS has a POWER10 quantized GEMM kernel. Only layers
with constant weights (Gemm / MatMul against an initializer) are quantized by default; MatMuls whose inputs are both
activations (attention Q.K^T and attn.V) stay FP32 because quantizing softmax-bound tensors costs the most accuracy.

Calibration frames come from a real clip through the same preprocessing as the camera path. There are no labels here:
accuracy must be checked afterwards against the FP32 model (compare_detections.py / agreement) and labelled data.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

from common import preprocess_bgr
from run_record import atomic_json, sha256


def dynamic_matmuls(model):
    """MatMul nodes whose inputs are both activations (no initializer operand)."""
    constants = {tensor.name for tensor in model.graph.initializer}
    return [node.name for node in model.graph.node
            if node.op_type == 'MatMul' and not any(name in constants for name in node.input)]


def calibration_frames(clip, count, height, width, mean, std):
    import cv2
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        raise SystemExit(f'Cannot open calibration clip {clip}')
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        raise SystemExit('Clip reports no frames')
    wanted = sorted(set(np.linspace(0, total - 1, min(count, total)).astype(int).tolist()))
    tensors = []
    for index in range(total):
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            tensors.append(preprocess_bgr(frame, height, width, mean, std))
    capture.release()
    if not tensors:
        raise SystemExit('No calibration frames could be read')
    return tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--frames', type=int, default=64, help='calibration frames, evenly spaced over the clip')
    parser.add_argument('--input-name', default='input')
    parser.add_argument('--mean', type=float, nargs=3, default=(.485, .456, .406))
    parser.add_argument('--std', type=float, nargs=3, default=(.229, .224, .225))
    parser.add_argument('--op-types', nargs='+', default=['MatMul', 'Gemm'])
    parser.add_argument('--include-dynamic', action='store_true', help='also quantize attention MatMuls (more speed, more risk)')
    parser.add_argument('--method', choices=['minmax', 'percentile', 'entropy'], default='minmax')
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        sys.exit(f'{output} exists; choose a new file so the previous model is preserved')

    import onnx
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    model = onnx.load(args.model)
    excluded = [] if args.include_dynamic else dynamic_matmuls(model)
    frames = calibration_frames(args.clip, args.frames, args.height, args.width, args.mean, args.std)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.iterator = iter(frames)

        def get_next(self):
            tensor = next(self.iterator, None)
            return None if tensor is None else {args.input_name: tensor}

    method = {'minmax': CalibrationMethod.MinMax, 'percentile': CalibrationMethod.Percentile,
              'entropy': CalibrationMethod.Entropy}[args.method]
    quantize_static(args.model, str(output), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
                    op_types_to_quantize=args.op_types, nodes_to_exclude=excluded, calibrate_method=method)

    quantized = onnx.load(str(output))
    by_output = {name: node for node in quantized.graph.node for name in node.output}
    done = [node.name for node in quantized.graph.node if node.op_type in args.op_types and node.name not in excluded
            and any(name in by_output and by_output[name].op_type == 'DequantizeLinear' for name in node.input)]
    counts = {}
    for node in quantized.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    atomic_json(str(output) + '.json', {
        'source_model': str(Path(args.model).resolve()), 'source_sha256': sha256(args.model),
        'output_sha256': sha256(output), 'calibration_clip': str(Path(args.clip).resolve()),
        'calibration_frames': len(frames), 'calibration_method': args.method,
        'quantized_nodes': done, 'excluded_dynamic_matmuls': excluded,
        'op_counts': {key: counts.get(key, 0) for key in ('QuantizeLinear', 'DequantizeLinear', 'MatMul', 'Gemm')},
        'versions': {'onnx': onnx.__version__, 'onnxruntime': importlib.metadata.version('onnxruntime')},
        'note': 'Quantized on the machine named in versions; run accuracy checks against FP32 before use.'})
    print(f'quantized {len(done)} nodes, left {len(excluded)} dynamic MatMuls in FP32 -> {output}')


if __name__ == '__main__':
    main()
