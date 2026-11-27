"""live_summary combines the shards of each camera and reports smoothness, latency and drops."""
import csv, json, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from live_summary import summarize

HEADER = ['camera_id', 'frame_id', 'completed_monotonic_s', 'queue_age_ms', 'inference_ms', 'read_to_result_ms']


def make_run(root, workers):
    """workers: list of (camera_id, received, dropped, [(frame_id, completed_s, queue_age, inference, read_to_result)])"""
    run = {'status': 'ok', 'active_elapsed_s': 10.0, 'workers': []}
    for index, (camera, received, dropped, rows) in enumerate(workers):
        run['workers'].append({'cameras': [dict(id=camera, received=received, processed=len(rows), queue_dropped=dropped,
                                                errors=0, reconnects=0)]})
        target = root / f'worker-{index:02d}'
        target.mkdir(parents=True)
        with (target / 'samples.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADER)
            for frame, completed, age, inference, total in rows:
                writer.writerow([camera, frame, completed, age, inference, total])
    (root / 'run.json').write_text(json.dumps(run))


class Summary(unittest.TestCase):
    def test_shards_of_one_camera_are_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, [('cam-s0', 30, 24, [(0, 1.0, 50, 200, 260), (3, 1.4, 60, 210, 280), (6, 1.8, 55, 205, 270)]),
                            ('cam-s1', 30, 24, [(1, 1.2, 70, 220, 300), (4, 1.6, 65, 215, 290), (7, 2.0, 58, 212, 275)])])
            result = summarize(root)
        self.assertEqual(list(result['cameras']), ['cam'])
        cam = result['cameras']['cam']
        self.assertEqual((cam['received'], cam['processed'], cam['dropped']), (60, 6, 48))
        self.assertAlmostEqual(cam['processed_fps'], 0.6)          # 6 frames over the 10 s active window
        self.assertEqual(cam['duplicate_frame_ids'], 0)
        self.assertEqual(result['workers'], 2)

    def test_smoothness_uses_gaps_between_consecutive_completions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, [('cam', 10, 5, [(0, 1.0, 1, 1, 1), (2, 1.1, 1, 1, 1), (4, 1.2, 1, 1, 1), (6, 2.2, 1, 1, 1)])])
            cam = summarize(root)['cameras']['cam']
        self.assertAlmostEqual(cam['gap_ms_max'], 1000.0, places=3)     # the 1.2 -> 2.2 stall
        self.assertAlmostEqual(cam['gap_ms_p50'], 100.0, places=3)

    def test_duplicate_frame_ids_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, [('cam-s0', 5, 2, [(0, 1.0, 1, 1, 1), (3, 1.2, 1, 1, 1)]),
                            ('cam-s1', 5, 2, [(3, 1.1, 1, 1, 1)])])
            self.assertEqual(summarize(root)['cameras']['cam']['duplicate_frame_ids'], 1)

    def test_plain_camera_ids_pass_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, [('camera-0', 4, 1, [(0, 1.0, 1, 1, 1), (1, 1.5, 1, 1, 1), (2, 2.0, 1, 1, 1)])])
            self.assertEqual(list(summarize(root)['cameras']), ['camera-0'])

    def test_latency_percentiles_cover_all_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, [('cam-s0', 3, 0, [(0, 1.0, 10, 100, 200)]), ('cam-s1', 3, 0, [(1, 1.1, 30, 300, 400)])])
            result = summarize(root)
        self.assertEqual(result['latency_ms']['read_to_result_ms']['max'], 400)
        self.assertEqual(result['latency_ms']['inference_ms']['p50'], 100)


if __name__ == '__main__':
    unittest.main()
