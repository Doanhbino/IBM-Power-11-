#!/usr/bin/env python3
"""Replace ONNX LayerNormalization nodes with the VSX custom op com.rfdetr::VsxLayerNorm.

Only nodes that can be replaced exactly are touched: normalisation over the last axis (axis -1), float32, with the
Mean and InvStdDev outputs unused. Everything else is left as is. The rewritten model needs the custom-op library
(core/ops/layernorm_op.cpp, built as librfdetr_ops.so) registered in the session.
"""
import argparse
from pathlib import Path
import sys

import numpy as np

from run_record import atomic_json, sha256

DOMAIN = 'com.rfdetr'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        sys.exit(f'{output} exists; choose a new file so the previous model is preserved')

    import onnx
    from onnx import helper, numpy_helper, TensorProto
    model = onnx.load(args.model)
    graph = model.graph
    used = {name for node in graph.node for name in node.input} | {out.name for out in graph.output}
    replaced = skipped = 0
    reasons = {}
    for index, node in enumerate(list(graph.node)):
        if node.op_type != 'LayerNormalization' or node.domain not in ('', 'ai.onnx'):
            continue
        attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        reason = None
        if attributes.get('axis', -1) != -1:
            reason = 'axis is not -1'
        elif attributes.get('stash_type', 1) != 1:
            reason = 'stash_type is not float32'
        elif any(name and name in used for name in list(node.output)[1:]):
            reason = 'mean or inverse std-dev output is used'
        elif len(node.input) < 2:
            reason = 'no scale input'
        if reason:
            skipped += 1
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        inputs = list(node.input[:3])
        if len(inputs) < 3 or not inputs[2]:
            # The op needs a bias; use zeros of the scale's size when the model has none.
            scale = next((t for t in graph.initializer if t.name == inputs[1]), None)
            if scale is None:
                skipped += 1
                reasons['bias missing and scale is not an initializer'] = reasons.get('bias missing and scale is not an initializer', 0) + 1
                continue
            zero_name = f'{node.name or index}_zero_bias'
            graph.initializer.append(numpy_helper.from_array(np.zeros(list(scale.dims), np.float32), zero_name))
            inputs = [inputs[0], inputs[1], zero_name]
        replacement = helper.make_node('VsxLayerNorm', inputs, [node.output[0]], name=node.name or f'ln_{index}',
                                       domain=DOMAIN, epsilon=float(attributes.get('epsilon', 1e-5)))
        position = list(graph.node).index(node)
        graph.node.remove(node)
        graph.node.insert(position, replacement)
        replaced += 1
    if replaced and not any(o.domain == DOMAIN for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid(DOMAIN, 1))
    onnx.save(model, str(output))
    atomic_json(str(output) + '.json', {
        'source_model': str(Path(args.model).resolve()), 'source_sha256': sha256(args.model),
        'output_sha256': sha256(output), 'replaced': replaced, 'skipped': skipped, 'skipped_reasons': reasons,
        'custom_op': f'{DOMAIN}::VsxLayerNorm', 'requires': 'librfdetr_ops.so registered in the session',
        'onnx': onnx.__version__})
    print(f'replaced {replaced} LayerNormalization nodes, skipped {skipped} -> {output}')


if __name__ == '__main__':
    main()
