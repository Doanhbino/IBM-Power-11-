"""Fixtures validate evidence consistency, never claim physical MMA execution."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from camera_service import validate_config
from compare_runs import compare
from mma_evidence import require_evidence, validate_dossier
from profile_ops import aggregate, gemm_shape
from run_record import artifact, atomic_json, public_config, sha256
from worker_plan import place


class Records(unittest.TestCase):
    def test_atomic_write_and_recursive_redaction(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'run.json'
            atomic_json(path, {'status': 'ok'})
            with self.assertRaises(ValueError):
                atomic_json(path, {'value': float('nan')})
            self.assertEqual(json.loads(path.read_text()), {'status': 'ok'})
            self.assertEqual(len(list(path.parent.iterdir())), 1)
        cfg = public_config({'workers': [{'sources': [{'source': 'rtsp://user:password@host'}]}],
                             'description': 'error rtsp://user:password@host/path'})
        self.assertNotIn('password', json.dumps(cfg))


class Evidence(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name, value in {'model': 'fixture model', 'binary': 'fixture runtime',
                            'build': 'fixture build', 'profile': 'fixture profile',
                            'stack': 'sampled fixture_mma_kernel'}.items():
            (self.root / name).write_text(value)
        atomic_json(self.root / 'inventory', {'machine': 'ppc64le', 'mma_capability': {'hwcap2_mma': True}})
        self.hash = sha256(self.root / 'model')
        atomic_json(self.root / 'run', {'run_id': 'fixture-run', 'model_sha256': self.hash, 'machine': 'ppc64le',
                    'workload': 'inference', 'status': 'ok', 'profile_instrumented': True,
                    'operator_profile': artifact(self.root / 'profile')})
        self.data = {'schema_version': 1, 'reviewer': 'fixture only', 'reviewed_utc': '2026-11-25T00:00:00Z',
                     'run_id': 'fixture-run', 'workload': 'inference', 'model_sha256': self.hash,
                     'capability': {'machine': 'ppc64le', 'hwcap2_mma': True, 'inventory': artifact(self.root / 'inventory')},
                     'stack': {'revision': 'fixture', 'compiler': 'fixture', 'flags': ['fixture'],
                               'build_log': artifact(self.root / 'build'), 'binaries': [artifact(self.root / 'binary')]},
                     'run': artifact(self.root / 'run'), 'operator_profile': artifact(self.root / 'profile'),
                     'hotspots': [{'node': 'fixture-node', 'operator': 'MatMul', 'shape': [3, 4, 5], 'dtype': 'float32',
                                   'phase': 'forward', 'kernel_symbol': 'fixture_mma_kernel', 'mma_executed': True,
                                   'sampled_stack': artifact(self.root / 'stack'), 'interpretation': 'test fixture only'}]}

    def tearDown(self):
        self.temp.cleanup()

    def validate(self, workload='inference', binaries=None):
        path = self.root / 'evidence.json'
        atomic_json(path, self.data)
        return validate_dossier(path, workload, self.hash, binaries)

    def test_consistent_dossier_and_tampered_artifact(self):
        self.assertEqual(self.validate()['status'], 'dossier_valid')
        (self.root / 'profile').write_text('changed')
        self.assertIn('RF-DETR operator profile: hash mismatch', self.validate()['errors'])

    def test_training_cannot_reuse_inference_evidence(self):
        result = self.validate('training')
        self.assertIn('Training/inference evidence cannot be substituted', result['errors'])
        self.assertIn('Training requires independent forward AND backward evidence', result['errors'])

    def test_runtime_model_and_symbol_binding(self):
        (self.root / 'other').write_text('different runtime')
        self.assertIn('Current runtime binaries differ from the reviewed stack',
                      self.validate(binaries=[artifact(self.root / 'other')])['errors'])
        self.data['hotspots'][0]['kernel_symbol'] = 'absent_symbol'
        self.assertTrue(any('absent from sampled trace' in error for error in self.validate()['errors']))
        with self.assertRaises(RuntimeError):
            require_evidence({'require_mma': True}, 'inference', self.hash, [])

    def test_capability_claim_must_match_inventory(self):
        atomic_json(self.root / 'inventory', {'machine': 'x86_64'})
        self.data['capability']['inventory'] = artifact(self.root / 'inventory')
        self.assertIn('Inventory did not observe HWCAP2_MMA', self.validate()['errors'])


class CameraPlacement(unittest.TestCase):
    def setUp(self):
        self.rows = [{'cpu': 2, 'node': 0}, {'cpu': 10, 'node': 0}, {'cpu': 21, 'node': 3}]
        self.config = {'session': {'model': 'fixture.onnx', 'mode': 'live', 'duration_s': 1,
                       'input_name': 'input', 'boxes_output': 'dets', 'logits_output': 'labels',
                       'background_index': None, 'height': 8, 'width': 8, 'top_k': 3,
                       'queue_capacity': 2, 'capture_timeout_ms': 100},
                       'workers': [{'cpus': [2, 10], 'numa_node': 0, 'threads': 1,
                                    'sources': [{'id': 'cam-0', 'source': 'rtsp://fixture'}]}]}

    def test_irregular_topology_and_background(self):
        cfg = validate_config(self.config, self.rows)
        self.assertEqual(cfg[0]['cpus'], [2, 10])
        self.assertEqual(cfg[0]['background_index'], 'none')

    def test_overlapping_foreign_cpu_and_cross_node_rejected(self):
        for cpus in ([1], [2, 2], [2, 21]):
            cfg = copy.deepcopy(self.config)
            cfg['workers'][0]['cpus'] = cpus
            with self.assertRaises(ValueError):
                validate_config(cfg, self.rows)
        cfg = copy.deepcopy(self.config)
        cfg['workers'].append(copy.deepcopy(cfg['workers'][0]))
        with self.assertRaises(ValueError):
            validate_config(cfg, self.rows)

    def test_generated_workers_use_observed_nodes_and_disjoint_cpus(self):
        rows = [dict(row, siblings_allowed=[row['cpu']]) for row in self.rows]
        cfg = copy.deepcopy(self.config)
        cfg['workers'][0]['sources'] += [{'id': 'cam-1', 'source': 'rtsp://fixture'},
                                         {'id': 'cam-2', 'source': 'rtsp://fixture'}]
        generated = place(cfg, rows, threads=1)
        workers = validate_config(generated, rows)
        self.assertEqual([worker['numa_node'] for worker in workers], [0, 3])
        self.assertEqual([worker['cpus'] for worker in workers], [[2], [21]])
        self.assertEqual(sum(len(worker['sources']) for worker in workers), 3)


class Comparison(unittest.TestCase):
    def test_repeats_and_workload_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            groups = []
            for name, speed in [('reference', 10), ('candidate', 20)]:
                paths = []
                for i in range(3):
                    path = root / f'{name}-{i}.json'
                    atomic_json(path, {'run_id': f'{name}-{i}', 'status': 'ok', 'profile_instrumented': False,
                         'model_sha256': 'fixture', 'input_sha256': 'fixture-input', 'input_shape': [1, 3, 8, 8],
                         'input_dtype': 'float32', 'timing_boundary': 'fixture-same-boundary',
                         'result_kind': 'replayed preprocessed tensor', 'config': {'mode': 'model', 'profile_prefix': str(path)},
                         'images': 100, 'throughput_fps': speed, 'latency_ms': {'p50': 1, 'p95': 2, 'p99': 3}})
                    paths.append(path)
                groups.append(paths)
            self.assertEqual(compare(*groups)['throughput_speedup'], 2)
            with self.assertRaises(ValueError):
                compare(groups[0][:2], groups[1])
            bad = json.loads(groups[1][0].read_text())
            bad['input_dtype'] = 'float16'
            atomic_json(groups[1][0], bad)
            with self.assertRaises(ValueError):
                compare(*groups)


class Profiles(unittest.TestCase):
    def test_ort_shape_groups_and_unknown_fused_attributes(self):
        event = {'name': 'node_kernel_time', 'cat': 'Node', 'ph': 'X', 'dur': 10,
                 'args': {'op_name': 'MatMul', 'provider': 'CPUExecutionProvider',
                          'input_type_shape': [{'float': [1, 3, 5]}, {'float': [5, 7]}]}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'profile.json'
            atomic_json(path, [event, event])
            rows, cases = aggregate(path)
            self.assertEqual(rows[0]['calls'], 2)
            self.assertEqual((cases[0]['m'], cases[0]['n'], cases[0]['k']), (3, 7, 5))
            self.assertIsNone(gemm_shape('FusedMatMul', rows[0]['inputs'], 'ort'))

    def test_torch_linear_transpose(self):
        shape = gemm_shape('aten::linear', [{'dtype': 'float', 'shape': [2, 3, 5]},
                                           {'dtype': 'float', 'shape': [7, 5]}], 'torch')
        self.assertEqual((shape['m'], shape['n'], shape['k'], shape['transpose_b']), (6, 7, 5, True))


if __name__ == '__main__':
    unittest.main()
