"""INT8 post-training quantization of the constant-weight linear layers; dynamic (attention) MatMuls stay FP32."""
import importlib.util, json, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
HAVE = all(importlib.util.find_spec(m) for m in ('onnx', 'onnxruntime', 'cv2'))


@unittest.skipUnless(HAVE, 'needs onnx, onnxruntime and cv2')
class QuantizeInt8(unittest.TestCase):
    def setUp(self):
        import onnx, cv2
        from onnx import helper as h, TensorProto, numpy_helper
        self.tmp = tempfile.TemporaryDirectory(); self.p = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        w = numpy_helper.from_array(rng.normal(0, .3, (64, 16)).astype(np.float32), 'W')
        w2 = numpy_helper.from_array(rng.normal(0, .3, (16, 8)).astype(np.float32), 'W2')
        b2 = numpy_helper.from_array(rng.normal(0, .1, (8,)).astype(np.float32), 'B2')
        shape = numpy_helper.from_array(np.array([3, 64], np.int64), 'shape')
        nodes = [h.make_node('Reshape', ['input', 'shape'], ['r']),
                 h.make_node('MatMul', ['r', 'W'], ['y'], name='const_matmul'),
                 h.make_node('Transpose', ['y'], ['yt'], perm=[1, 0]),
                 h.make_node('MatMul', ['y', 'yt'], ['z'], name='dynamic_matmul'),
                 h.make_node('Gemm', ['y', 'W2', 'B2'], ['g'], name='gemm')]
        graph = h.make_graph(nodes, 'g', [h.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 8, 8])],
                             [h.make_tensor_value_info('z', TensorProto.FLOAT, [3, 3]),
                              h.make_tensor_value_info('g', TensorProto.FLOAT, [3, 8])], [w, w2, b2, shape])
        model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)]); model.ir_version = 9
        onnx.save(model, self.p / 'fp32.onnx')
        writer = cv2.VideoWriter(str(self.p / 'clip.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 10, (16, 16))
        for _ in range(12):
            writer.write(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))
        writer.release()

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *extra, code=0):
        done = subprocess.run([sys.executable, str(ROOT / 'scripts/quantize_int8.py'), '--model', str(self.p / 'fp32.onnx'),
                               '--clip', str(self.p / 'clip.avi'), '--output', str(self.p / 'int8.onnx'),
                               '--height', '8', '--width', '8', '--frames', '8', *extra], capture_output=True, text=True)
        self.assertEqual(done.returncode, code, done.stdout + done.stderr)
        return done

    def producers(self, model):
        import onnx
        graph = onnx.load(model).graph
        by_output = {o: n for n in graph.node for o in n.output}
        return graph, by_output

    def consumers(self, graph, tensor):
        return [n for n in graph.node if tensor in n.input]

    def test_const_layers_quantized_dynamic_matmul_left_alone(self):
        self.run_tool()
        graph, by_output = self.producers(self.p / 'int8.onnx')
        ops = {n.name: n for n in graph.node}
        def weight_fed_by_dq(name):
            return any(by_output[i].op_type == 'DequantizeLinear' for i in ops[name].input[1:2] if i in by_output)
        self.assertTrue(weight_fed_by_dq('const_matmul'))
        self.assertTrue(weight_fed_by_dq('gemm'))
        # An excluded MatMul computes in float: its output is never re-quantized.
        self.assertFalse([n for n in self.consumers(graph, ops['dynamic_matmul'].output[0]) if n.op_type == 'QuantizeLinear'])

    def test_dynamic_matmul_can_be_included_on_request(self):
        self.run_tool('--include-dynamic')
        graph, _ = self.producers(self.p / 'int8.onnx')
        node = next(n for n in graph.node if n.name == 'dynamic_matmul')
        self.assertTrue([n for n in self.consumers(graph, node.output[0]) if n.op_type == 'QuantizeLinear'])

    def test_quantized_model_close_to_fp32_and_manifest_written(self):
        import onnxruntime as ort
        self.run_tool()
        from common import preprocess_bgr   # same preprocessing and distribution as the calibration frames
        x = preprocess_bgr(np.random.default_rng(1).integers(0, 255, (16, 16, 3), dtype=np.uint8), 8, 8)
        a = ort.InferenceSession(str(self.p / 'fp32.onnx'), providers=['CPUExecutionProvider']).run(None, {'input': x})
        b = ort.InferenceSession(str(self.p / 'int8.onnx'), providers=['CPUExecutionProvider']).run(None, {'input': x})
        for fa, fb in zip(a, b):
            self.assertEqual(fa.shape, fb.shape); self.assertTrue(np.isfinite(fb).all())
            self.assertLess(np.abs(fa - fb).max(), .25 * max(1.0, np.abs(fa).max()))
        manifest = json.loads((self.p / 'int8.onnx.json').read_text())
        self.assertEqual(manifest['calibration_frames'], 8)
        self.assertEqual(manifest['quantized_nodes'], ['const_matmul', 'gemm'])
        self.assertEqual(manifest['excluded_dynamic_matmuls'], ['dynamic_matmul'])
        self.assertEqual(len(manifest['output_sha256']), 64)

    def test_refuses_to_overwrite_output(self):
        self.run_tool()
        self.run_tool(code=1)


if __name__ == '__main__':
    unittest.main()
