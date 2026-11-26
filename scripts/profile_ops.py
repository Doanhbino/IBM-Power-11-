#!/usr/bin/env python3
"""Aggregate ORT/PyTorch profiles and extract unambiguous, concrete GEMM shapes."""

import argparse
import csv
import json
import math
from pathlib import Path

from run_record import artifact, atomic_json


def ort_tensors(items):
    result = []
    for item in items or []:
        if isinstance(item, dict):
            for dtype, shape in item.items():
                result.append({'dtype': dtype, 'shape': shape})
    return result


def graph_inputs(path):
    """Read initializers by unique node name from the SAME optimized graph as the trace."""
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    constants = {value.name: {'dtype': onnx.TensorProto.DataType.Name(value.data_type).lower(),
                             'shape': list(value.dims)} for value in model.graph.initializer}
    nodes = {}
    duplicates = set()
    for node in model.graph.node:
        if not node.name:
            continue
        if node.name in nodes:
            duplicates.add(node.name)
        nodes[node.name] = (node.op_type, list(node.input))
    for name in duplicates:
        del nodes[name]
    return nodes, constants


def restore_packed_inputs(event, traced, graph):
    nodes, constants = graph
    node = nodes.get(event['name'].removesuffix('_kernel_time'))
    if node is None or node[0] != event.get('args', {}).get('op_name'):
        return traced
    names = node[1]
    if len(traced) == len(names):
        return traced
    # Only restore when every omitted position is a graph initializer.
    if sum(name not in constants for name in names) != len(traced):
        return traced
    active = iter(traced)
    return [constants[name] if name in constants else next(active) for name in names]


def gemm_shape(operator, inputs, kind):
    transposed_b = False
    if kind == 'ort':
        if operator != 'MatMul' or len(inputs) != 2:
            return None  # FusedMatMul/Gemm attributes are not available reliably in a trace.
        a, b = inputs
    else:
        shapes = [item for item in inputs if item['shape']]
        if operator in {'aten::mm', 'aten::bmm', 'aten::matmul'} and len(shapes) >= 2:
            a, b = shapes[:2]
        elif operator == 'aten::addmm' and len(shapes) >= 3:
            a, b = shapes[1:3]
        elif operator == 'aten::linear' and len(shapes) >= 2:
            a, b = shapes[:2]
            if len(b['shape']) != 2:
                return None
            b = dict(b, shape=b['shape'][::-1])
            transposed_b = True
        else:
            return None
    ashape, bshape = a['shape'], b['shape']
    if not isinstance(ashape, list) or not isinstance(bshape, list) or min(len(ashape), len(bshape)) < 2:
        return None
    if any(type(dim) is not int or dim < 1 for dim in ashape + bshape) or ashape[-1] != bshape[-2]:
        return None
    m = math.prod(ashape[:-1]) if operator == 'aten::linear' else ashape[-2]
    return {'m': m, 'n': bshape[-1], 'k': ashape[-1], 'transpose_b': transposed_b,
            'input_dtype': a['dtype'], 'rhs_dtype': b['dtype'],
            'lhs_shape': ashape, 'rhs_logical_shape': bshape,
            'batch_scope': 'one matrix; batch/broadcast and original strides require separate profiling'}


def aggregate(profile, kind='ort', model=None):
    payload = json.loads(Path(profile).read_text())
    events = payload.get('traceEvents', []) if isinstance(payload, dict) else payload
    groups = {}
    graph = graph_inputs(model) if kind == 'ort' and model else None
    for event in events:
        args = event.get('args', {})
        if event.get('ph') != 'X':
            continue
        if kind == 'ort':
            if event.get('cat') != 'Node' or not event.get('name', '').endswith('_kernel_time'):
                continue
            operator, provider = args.get('op_name', 'unknown'), args.get('provider', 'unknown')
            inputs = ort_tensors(args.get('input_type_shape'))
            if graph:
                inputs = restore_packed_inputs(event, inputs, graph)
            outputs = ort_tensors(args.get('output_type_shape'))
        else:
            if event.get('cat') != 'cpu_op':
                continue
            operator, provider = event.get('name', 'unknown'), 'PyTorch CPU'
            dims, types = args.get('Input Dims', []), args.get('Input type', [])
            inputs = [{'dtype': types[i] if i < len(types) else 'unknown', 'shape': shape}
                      for i, shape in enumerate(dims)]
            outputs = []
        key = (provider, operator, event['name'], json.dumps(inputs, sort_keys=True), json.dumps(outputs, sort_keys=True))
        duration = float(event.get('dur', 0))
        if not math.isfinite(duration) or duration < 0:
            raise ValueError('Invalid profile duration')
        if key not in groups:
            groups[key] = {'provider': provider, 'operator': operator, 'node': event['name'],
                           'inputs': inputs, 'outputs': outputs, 'calls': 0, 'total_us': 0.0}
        groups[key]['calls'] += 1
        groups[key]['total_us'] += duration
    rows = sorted(groups.values(), key=lambda row: -row['total_us'])
    cases = []
    for row in rows:
        row['mean_us'] = row['total_us'] / row['calls']
        shape = gemm_shape(row['operator'], row['inputs'], kind)
        if shape:
            cases.append(dict(shape, node=row['node'], operator=row['operator'],
                              calls=row['calls'], total_us=row['total_us']))
    return rows, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile')
    parser.add_argument('--format', choices=('ort', 'torch'), default='ort')
    parser.add_argument('--output', required=True)
    parser.add_argument('--shapes')
    parser.add_argument('--model', help='The optimized ONNX graph saved by this profiled session; restores omitted packed initializer shapes')
    args = parser.parse_args()
    rows, cases = aggregate(args.profile, args.format, args.model)
    fields = ['provider', 'operator', 'node', 'calls', 'total_us', 'mean_us', 'inputs', 'outputs']
    with Path(args.output).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, list) else value
                         for key, value in row.items()} for row in rows)
    if args.shapes:
        atomic_json(args.shapes, {'schema_version': 1, 'profile': artifact(args.profile),
                                 'optimized_model': artifact(args.model) if args.model else None,
                                 'format': args.format, 'cases': cases,
                                 'limitations': 'Logical dimensions only; physical layout/strides and fused-node attributes require inspection'})
    print(json.dumps({'nodes': len(rows), 'gemm_shapes': len(cases),
                      'providers': sorted({row['provider'] for row in rows}),
                      'warning': 'Includes warmup; operator sums may overlap; no proof of MMA dispatch'}))


if __name__ == '__main__':
    main()
