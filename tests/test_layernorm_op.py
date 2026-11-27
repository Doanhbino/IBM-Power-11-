"""VsxLayerNorm custom op and the model rewrite: must match ORT's own LayerNormalization."""
import importlib.util, json, os, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = os.environ.get('ORT_INCLUDE_DIR')
HAVE = all(importlib.util.find_spec(m) for m in ('onnx', 'onnxruntime')) and INCLUDE and shutil.which('g++')


@unittest.skipUnless(HAVE, 'needs onnx, onnxruntime, g++ and ORT_INCLUDE_DIR (directory with onnxruntime_cxx_api.h)')
class LayerNormOp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); cls.dir = Path(cls.tmp.name)
        cls.lib = cls.dir / 'librfdetr_ops.so'
        done = subprocess.run(['g++', '-std=c++17', '-O2', '-fPIC', '-shared', f'-I{INCLUDE}', f'-I{ROOT}/core/include',
                               str(ROOT / 'core/ops/layernorm_op.cpp'), str(ROOT / 'core/src/layernorm.cpp'),
                               '-o', str(cls.lib)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def model(self, shape=(1, 12, 32), bias=True, extra_output=False, name='ln', axis=-1):
        import onnx
        from onnx import helper as h, TensorProto, numpy_helper
        rng = np.random.default_rng(3)
        inits = [numpy_helper.from_array(rng.normal(1, .3, shape[-1]).astype(np.float32), 'scale')]
        inputs = ['input', 'scale']
        if bias:
            inits.append(numpy_helper.from_array(rng.normal(0, .2, shape[-1]).astype(np.float32), 'bias')); inputs.append('bias')
        outputs = ['y'] + (['mean', 'inv'] if extra_output else [])
        node = h.make_node('LayerNormalization', inputs, outputs, name=name, axis=axis, epsilon=1e-5)
        graph = h.make_graph([node, h.make_node('Relu', ['y'], ['out'], name='relu')], 'g',
                             [h.make_tensor_value_info('input', TensorProto.FLOAT, list(shape))],
                             [h.make_tensor_value_info('out', TensorProto.FLOAT, list(shape))], inits)
        model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)]); model.ir_version = 9
        path = self.dir / f'{name}-{bias}-{extra_output}-{axis}.onnx'
        onnx.save(model, path)
        return path

    def rewrite(self, source, *extra, code=0):
        out = self.dir / (source.stem + '-vsx.onnx')
        if out.exists(): out.unlink()
        done = subprocess.run([sys.executable, str(ROOT / 'scripts/replace_layernorm.py'), '--model', str(source),
                               '--output', str(out), *extra], capture_output=True, text=True)
        self.assertEqual(done.returncode, code, done.stdout + done.stderr)
        return out

    def run_model(self, path, x, custom=False):
        import onnxruntime as ort
        options = ort.SessionOptions()
        if custom: options.register_custom_ops_library(str(self.lib))
        return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider']).run(None, {'input': x})[0]

    def test_rewritten_model_matches_ort_layernorm(self):
        import onnx
        source = self.model()
        rewritten = self.rewrite(source)
        ops = [n.op_type for n in onnx.load(rewritten).graph.node]
        self.assertIn('VsxLayerNorm', ops); self.assertNotIn('LayerNormalization', ops)
        x = np.random.default_rng(4).normal(0, 2, (1, 12, 32)).astype(np.float32)
        np.testing.assert_allclose(self.run_model(rewritten, x, True), self.run_model(source, x), atol=2e-5, rtol=1e-5)

    def test_missing_bias_is_handled(self):
        source = self.model(bias=False)
        rewritten = self.rewrite(source)
        x = np.random.default_rng(5).normal(0, 1, (1, 12, 32)).astype(np.float32)
        np.testing.assert_allclose(self.run_model(rewritten, x, True), self.run_model(source, x), atol=2e-5, rtol=1e-5)

    def test_many_rows_and_odd_width_use_the_parallel_path(self):
        source = self.model(shape=(1, 300, 37), name='wide')
        rewritten = self.rewrite(source)
        x = np.random.default_rng(6).normal(5, 3, (1, 300, 37)).astype(np.float32)
        np.testing.assert_allclose(self.run_model(rewritten, x, True), self.run_model(source, x), atol=3e-5, rtol=1e-5)

    def test_nodes_whose_extra_outputs_are_used_are_left_alone(self):
        import onnx
        from onnx import helper as h
        source = self.model(extra_output=True, name='extra')
        model = onnx.load(source)
        model.graph.output.append(h.make_tensor_value_info('mean', onnx.TensorProto.FLOAT, None))  # consumes output 1
        onnx.save(model, source)
        rewritten = self.rewrite(source)
        self.assertEqual([n.op_type for n in onnx.load(rewritten).graph.node].count('LayerNormalization'), 1)

    def test_non_last_axis_is_left_alone(self):
        import onnx
        source = self.model(name='axis1', axis=1, shape=(1, 12, 12))
        rewritten = self.rewrite(source)
        self.assertEqual([n.op_type for n in onnx.load(rewritten).graph.node].count('LayerNormalization'), 1)

    def test_refuses_to_overwrite_and_writes_manifest(self):
        source = self.model(name='manifest')
        rewritten = self.rewrite(source)
        manifest = json.loads(Path(str(rewritten) + '.json').read_text())
        self.assertEqual(manifest['replaced'], 1); self.assertEqual(manifest['skipped'], 0)
        self.assertEqual(len(manifest['output_sha256']), 64)
        done = subprocess.run([sys.executable, str(ROOT / 'scripts/replace_layernorm.py'), '--model', str(source),
                               '--output', str(rewritten)], capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)


if __name__ == '__main__':
    unittest.main()
