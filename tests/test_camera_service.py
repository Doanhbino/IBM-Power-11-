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

    def test_capture_cpus_are_passed_through(self):
        out = validate_config(self.config(self.worker([0, 1], 0, 'a', capture_cpus=[2, 3])), ROWS)
        self.assertEqual(out[0]['capture_cpus'], [2, 3])

    def test_capture_cpus_default_absent(self):
        self.assertNotIn('capture_cpus', validate_config(self.config(self.worker([0, 1], 0, 'a')), ROWS)[0])

    def test_capture_cpus_must_not_overlap_any_worker(self):
        for bad in ([1, 2], [8, 9]):  # own inference CPUs; another worker's CPUs
            with self.assertRaises(ValueError):
                validate_config(self.config(self.worker([0, 1], 0, 'a', capture_cpus=bad),
                                            self.worker([8, 9], 1, 'b')), ROWS)
        with self.assertRaises(ValueError):  # two workers sharing a capture CPU
            validate_config(self.config(self.worker([0, 1], 0, 'a', capture_cpus=[2]),
                                        self.worker([8, 9], 1, 'b', capture_cpus=[2])), ROWS)

    def test_capture_cpus_must_stay_on_worker_numa_node(self):
        with self.assertRaises(ValueError):
            validate_config(self.config(self.worker([0, 1], 0, 'a', capture_cpus=[10])), ROWS)

    def test_capture_cpus_must_exist_and_be_unique_integers(self):
        for bad in ([99], [2, 2], [], ['2']):
            with self.assertRaises(ValueError):
                validate_config(self.config(self.worker([0, 1], 0, 'a', capture_cpus=bad)), ROWS)

    def test_worker_plan_reserves_capture_cpus(self):
        config = self.config(self.worker([], 0, 'a'))
        worker = place(config, ROWS, threads=2, smt_limit=1, max_workers=1, capture_cpus=2)['workers'][0]
        self.assertEqual(len(worker['cpus']), 2)
        self.assertEqual(len(worker['capture_cpus']), 2)
        self.assertFalse(set(worker['cpus']) & set(worker['capture_cpus']))
        self.assertEqual({ROWS[c]['node'] for c in worker['cpus'] + worker['capture_cpus']}, {worker['numa_node']})

    def test_worker_plan_default_has_no_capture_cpus(self):
        worker = place(self.config(self.worker([], 0, 'a')), ROWS, threads=2, smt_limit=1, max_workers=1)['workers'][0]
        self.assertNotIn('capture_cpus', worker)

    def test_worker_plan_rejects_node_too_small_for_capture(self):
        with self.assertRaises(ValueError):
            place(self.config(self.worker([], 0, 'a')), ROWS, threads=2, smt_limit=1, max_workers=1, capture_cpus=7)

    def test_allow_spinning_is_validated_and_passed(self):
        self.session['allow_spinning'] = 1
        self.assertEqual(validate_config(self.config(self.worker([0, 1], 0, 'a')), ROWS)[0]['allow_spinning'], 1)
        self.session['allow_spinning'] = 2
        with self.assertRaises(ValueError):
            validate_config(self.config(self.worker([0, 1], 0, 'a')), ROWS)

    def test_workers_per_node_packs_disjoint_workers_on_each_node(self):
        sources = [dict(id=f'cam-{i}', source=str(self.video)) for i in range(8)]
        config = self.config(dict(cpus=[], threads=2, sources=sources))
        workers = place(config, ROWS, threads=2, smt_limit=1, max_workers=8, workers_per_node=2)['workers']
        self.assertEqual(len(workers), 4)                       # 2 nodes x 2 workers
        self.assertEqual(sorted(w['numa_node'] for w in workers), [0, 0, 1, 1])
        cpus = [c for w in workers for c in w['cpus']]
        self.assertEqual(len(cpus), len(set(cpus)))              # disjoint
        for w in workers:
            self.assertEqual({ROWS[c]['node'] for c in w['cpus']}, {w['numa_node']})
            self.assertEqual(len(w['cpus']), 2)
        self.assertEqual(sorted(s['id'] for w in workers for s in w['sources']), sorted(s['id'] for s in sources))

    def test_workers_per_node_default_is_one(self):
        workers = place(self.config(self.worker([], 0, 'a')), ROWS, threads=2, smt_limit=1, max_workers=8)['workers']
        self.assertEqual(len(workers), 1)
        sources = [dict(id=f'cam-{i}', source=str(self.video)) for i in range(4)]
        workers = place(self.config(dict(cpus=[], threads=2, sources=sources)), ROWS, threads=2, smt_limit=1,
                        max_workers=8)['workers']
        self.assertEqual(sorted(w['numa_node'] for w in workers), [0, 1])

    def test_workers_per_node_respects_max_workers_and_spreads_nodes_first(self):
        sources = [dict(id=f'cam-{i}', source=str(self.video)) for i in range(8)]
        workers = place(self.config(dict(cpus=[], threads=2, sources=sources)), ROWS, threads=2, smt_limit=1,
                        max_workers=3, workers_per_node=4)['workers']
        self.assertEqual(len(workers), 3)
        self.assertEqual(sorted(w['numa_node'] for w in workers), [0, 0, 1])   # alternate nodes before filling one

    def test_workers_per_node_rejects_when_node_too_small(self):
        sources = [dict(id=f'cam-{i}', source=str(self.video)) for i in range(8)]
        with self.assertRaises(ValueError):
            place(self.config(dict(cpus=[], threads=2, sources=sources)), ROWS, threads=2, smt_limit=1,
                  max_workers=8, workers_per_node=5)   # 5 x 2 = 10 CPUs > 8 per node

    def test_workers_per_node_includes_capture_cpus_per_worker(self):
        sources = [dict(id=f'cam-{i}', source=str(self.video)) for i in range(4)]
        workers = place(self.config(dict(cpus=[], threads=2, sources=sources)), ROWS, threads=2, smt_limit=1,
                        max_workers=4, workers_per_node=2, capture_cpus=1)['workers']
        used = [c for w in workers for c in w['cpus'] + w['capture_cpus']]
        self.assertEqual(len(used), len(set(used)))
        self.assertEqual(len(workers), 4)

    def shard_config(self, **source_extra):
        worker = dict(cpus=[0, 1], threads=2, numa_node=0, sources=[dict(id='a', source=str(self.video), **source_extra)])
        return self.config(worker)

    def test_frame_shard_keys_are_validated_and_passed(self):
        out = validate_config(self.shard_config(frame_stride=3, frame_offset=1), ROWS)
        self.assertEqual((out[0]['sources'][0]['frame_stride'], out[0]['sources'][0]['frame_offset']), (3, 1))
        for bad in (dict(frame_stride=0, frame_offset=0), dict(frame_stride=3, frame_offset=3),
                    dict(frame_stride=3, frame_offset=-1), dict(frame_stride='3', frame_offset=0),
                    dict(frame_stride=2)):   # offset required with stride
            with self.assertRaises(ValueError):
                validate_config(self.shard_config(**bad), ROWS)

    def test_worker_plan_shards_one_camera_across_distinct_workers(self):
        config = self.config(dict(cpus=[], threads=2, sources=[dict(id='cam', source=str(self.video))]))
        workers = place(config, ROWS, threads=2, smt_limit=1, max_workers=8, workers_per_node=2, shards=3)['workers']
        self.assertEqual(len(workers), 3)
        sources = [s for w in workers for s in w['sources']]
        self.assertEqual(sorted(s['id'] for s in sources), ['cam-s0', 'cam-s1', 'cam-s2'])
        self.assertEqual(sorted((s['frame_stride'], s['frame_offset']) for s in sources), [(3, 0), (3, 1), (3, 2)])
        self.assertTrue(all(len(w['sources']) == 1 for w in workers))     # shards of a camera never share a worker
        validate_config(dict(binary='b', session=self.session, workers=workers), ROWS)   # the plan is launchable

    def test_worker_plan_shards_two_cameras_keep_shards_apart(self):
        config = self.config(dict(cpus=[], threads=2, sources=[dict(id='c1', source=str(self.video)),
                                                                dict(id='c2', source=str(self.video))]))
        workers = place(config, ROWS, threads=2, smt_limit=1, max_workers=4, workers_per_node=2, shards=3)['workers']
        for camera in ('c1', 'c2'):
            holders = [i for i, w in enumerate(workers) for s in w['sources'] if s['id'].startswith(camera + '-s')]
            self.assertEqual(len(holders), len(set(holders)), camera)

    def test_worker_plan_rejects_more_shards_than_workers(self):
        config = self.config(dict(cpus=[], threads=2, sources=[dict(id='cam', source=str(self.video))]))
        with self.assertRaises(ValueError):
            place(config, ROWS, threads=2, smt_limit=1, max_workers=2, shards=3)

    def test_worker_plan_default_does_not_shard(self):
        config = self.config(dict(cpus=[], threads=2, sources=[dict(id='cam', source=str(self.video))]))
        sources = [s for w in place(config, ROWS, threads=2, smt_limit=1, max_workers=2)['workers'] for s in w['sources']]
        self.assertEqual(sources, [dict(id='cam', source=str(self.video))])

    def test_custom_ops_library_must_exist_and_is_passed_through(self):
        lib = Path(self.tmp.name) / 'librfdetr_ops.so'
        lib.write_bytes(b'x')
        self.session['custom_ops_library'] = str(lib)
        out = validate_config(self.config(self.worker([0, 1], 0, 'a')), ROWS)
        self.assertEqual(out[0]['custom_ops_library'], str(lib))
        self.session['custom_ops_library'] = str(Path(self.tmp.name) / 'missing.so')
        with self.assertRaises(ValueError):
            validate_config(self.config(self.worker([0, 1], 0, 'a')), ROWS)
        self.session['custom_ops_library'] = ''
        with self.assertRaises(ValueError):
            validate_config(self.config(self.worker([0, 1], 0, 'a')), ROWS)

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
