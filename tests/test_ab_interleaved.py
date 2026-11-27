"""A/B summaries must cover every worker, not just the first."""
import csv, json, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ab_interleaved import summarize


def make_run(root, workers):
    """workers: list of (camera_ids_with_counts, inference_values)."""
    run = {'run_id': 'x', 'status': 'ok', 'throughput_active_fps': 7.5, 'workers': []}
    for index, (cameras, inference) in enumerate(workers):
        run['workers'].append({'cameras': [dict(id=name, received=r, processed=p, queue_dropped=d, errors=0)
                                            for name, r, p, d in cameras]})
        target = root / f'worker-{index:02d}'
        target.mkdir(parents=True)
        with (target / 'samples.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['inference_ms', 'read_to_result_ms'])
            for value in inference:
                writer.writerow([value, value + 100])
    (root / 'run.json').write_text(json.dumps(run))


class Summaries(unittest.TestCase):
    def test_two_workers_are_combined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, [([('a', 10, 6, 4), ('b', 8, 5, 3)], [100] * 6 + [200] * 5),
                            ([('c', 20, 12, 8)], [300] * 12)])
            row = summarize(root, 'L', 1)
        self.assertEqual(row['workers'], 2)
        self.assertEqual(row['samples'], 23)
        self.assertEqual((row['received'], row['processed'], row['dropped']), (38, 23, 15))
        self.assertEqual(row['fps'], 7.5)
        self.assertEqual(row['inference_p50'], 300)       # 12 of 23 samples are 300
        self.assertEqual(row['enough_samples'], False)

    def test_single_worker_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, [([('a', 400, 400, 0)], [250] * 400)])
            row = summarize(root, 'L', 1)
        self.assertEqual((row['workers'], row['samples'], row['processed']), (1, 400, 400))
        self.assertTrue(row['enough_samples'])


if __name__ == '__main__':
    unittest.main()
