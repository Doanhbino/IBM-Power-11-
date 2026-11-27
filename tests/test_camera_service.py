"""Per-worker model selection in the camera launcher and placement planner (pure functions)."""
import copy, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from camera_service import validate_config
from worker_plan import place

ROWS = [{'cpu': i, 'node': i // 8, 'package': '0', 'core': str(i), 'siblings_allowed': [i]} for i in range(16)]
YOLO = dict(model_type='yolo11', model='y.onnx', input_name='images', height=640, width=640,
            output_name='output0', num_classes=4, conf_threshold=.25, iou_threshold=.7, max_det=300)


class CameraService(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video = Path(self.tmp.name) / 'a.avi'
        self.video.write_bytes(b'x')
        self.session = dict(model='r.onnx', input_name='input', boxes_output='dets', logits_output='labels',
                            height=576, width=576, background_index=-1, top_k=300, threshold=.25, mode='replay',
                            queue_capacity=2, capture_timeout_ms=5000, warmup=1)

    def tearDown(self):
        self.tmp.cleanup()

    def worker(self, cpus, node, name, **extra):
        return dict(cpus=cpus, threads=2, numa_node=node, sources=[dict(id=name, source=str(self.video))], **extra)

    def config(self, *workers, **top):
        return dict(binary='b', session=copy.deepcopy(self.session), workers=list(workers), **top)

    def test_legacy_config_defaults_to_rfdetr(self):
        out = validate_config(self.config(self.worker([0, 1], 0, 'a')), ROWS)
        self.assertEqual(out[0]['model_type'], 'rfdetr')
        self.assertEqual(out[0]['background_index'], -1)
        self.assertNotIn('output_name', out[0])

    def test_mixed_workers_merge_per_worker_keys(self):
        out = validate_config(self.config(self.worker([0, 1], 0, 'a'), self.worker([8, 9], 1, 'b', **YOLO)), ROWS)
        self.assertEqual([w['model_type'] for w in out], ['rfdetr', 'yolo11'])
        self.assertEqual([w['model'] for w in out], ['r.onnx', 'y.onnx'])
        self.assertEqual([w['height'] for w in out], [576, 640])
        self.assertEqual(out[1]['num_classes'], 4)

    def test_yolo_only_config_needs_no_rfdetr_keys(self):
        for key in ('model', 'input_name', 'boxes_output', 'logits_output', 'background_index', 'top_k',
                    'threshold', 'height', 'width'):
            del self.session[key]
        out = validate_config(self.config(self.worker([0, 1], 0, 'a', **YOLO)), ROWS)
        self.assertEqual(out[0]['model_type'], 'yolo11')

    def test_rfdetr_still_requires_top_k(self):
        del self.session['top_k']
        with self.assertRaises(ValueError) as caught:
            validate_config(self.config(self.worker([0, 1], 0, 'a')), ROWS)
        self.assertIn('top_k', str(caught.exception))

    def test_yolo_requires_explicit_thresholds(self):
        bad = {k: v for k, v in YOLO.items() if k != 'iou_threshold'}
        with self.assertRaises(ValueError) as caught:
            validate_config(self.config(self.worker([0, 1], 0, 'a', **bad)), ROWS)
        self.assertIn('iou_threshold', str(caught.exception))

    def test_unknown_model_type_rejected(self):
        with self.assertRaises(ValueError):
            validate_config(self.config(self.worker([0, 1], 0, 'a', model_type='yolo9')), ROWS)

    def test_mixed_models_refuse_require_mma_or_evidence(self):
        for extra in (dict(require_mma=True), dict(mma_evidence='dossier.json')):
            with self.assertRaises(ValueError) as caught:
                validate_config(self.config(self.worker([0, 1], 0, 'a'), self.worker([8, 9], 1, 'b', **YOLO), **extra), ROWS)
            self.assertIn('per model', str(caught.exception))

    def test_same_model_on_two_workers_is_allowed_with_require_mma(self):
        out = validate_config(self.config(self.worker([0, 1], 0, 'a'), self.worker([8, 9], 1, 'b'), require_mma=True), ROWS)
        self.assertEqual(len(out), 2)

    def test_worker_plan_preserves_model_keys(self):
        config = self.config(self.worker([], 0, 'a'), self.worker([], 0, 'b', **YOLO))
        planned = place(config, ROWS, threads=2, smt_limit=1, max_workers=4)['workers']
        self.assertEqual(sorted(w['model_type'] for w in planned if 'model_type' in w), ['yolo11'])
        self.assertEqual(len(planned), 2)
        yolo = next(w for w in planned if w.get('model_type') == 'yolo11')
        self.assertEqual(yolo['model'], 'y.onnx'); self.assertEqual([s['id'] for s in yolo['sources']], ['b'])
        self.assertFalse(set(planned[0]['cpus']) & set(planned[1]['cpus']))
        self.assertNotEqual(planned[0]['numa_node'], planned[1]['numa_node'])

    def test_worker_plan_legacy_single_group_unchanged(self):
        config = self.config(self.worker([], 0, 'a'), self.worker([], 0, 'b'))
        planned = place(config, ROWS, threads=2, smt_limit=1, max_workers=4)['workers']
        self.assertEqual(len(planned), 2)
        self.assertTrue(all(set(w) == {'cpus', 'numa_node', 'threads', 'sources'} for w in planned))


if __name__ == '__main__':
    unittest.main()
