"""The correctness gate must fail on any real difference and pass on identical or tolerably close output."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'compare_detections.py'


def box(x, score=0.9, slot=1):
    return {'xyxy': [x, 10, x + 50, 60], 'score': score, 'class_slot': slot}


class CompareDetections(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_dir(self, name, frames):
        directory = self.root / name / 'worker-00'
        directory.mkdir(parents=True)
        with (directory / 'detections.jsonl').open('w') as stream:
            for frame_id, detections in frames.items():
                stream.write(json.dumps({'camera_id': 'camera-0', 'frame_id': frame_id,
                                         'detections': detections}) + '\n')
        return str(self.root / name)

    def gate(self, reference, candidate, *extra):
        out = self.root / 'gate.json'
        done = subprocess.run([sys.executable, str(SCRIPT), '--reference', reference, '--candidate', candidate,
                               '--output', str(out), '--min-frames', '3', *extra], capture_output=True, text=True)
        return done.returncode, json.loads(out.read_text())

    def frames(self, change=None):
        base = {i: [box(100), box(300, 0.6, 2)] for i in range(5)}
        base.update(change or {})
        return base

    def test_identical_passes(self):
        code, result = self.gate(self.run_dir('a', self.frames()), self.run_dir('b', self.frames()))
        self.assertEqual(code, 0)
        self.assertTrue(result['passed'])
        self.assertEqual(result['frames_exact'], 5)

    def test_tiny_numeric_noise_passes(self):
        noisy = {i: [box(100.001, 0.9004), box(300, 0.6, 2)] for i in range(5)}
        code, result = self.gate(self.run_dir('a', self.frames()), self.run_dir('b', noisy))
        self.assertEqual(code, 0, result)
        self.assertLessEqual(result['max_score_delta'], 1e-3)

    def test_missing_detection_fails(self):
        code, result = self.gate(self.run_dir('a', self.frames()), self.run_dir('b', self.frames({3: [box(100)]})))
        self.assertEqual(code, 2)
        self.assertEqual(result['only_reference'], 1)

    def test_class_change_fails(self):
        changed = self.frames({1: [box(100, slot=3), box(300, 0.6, 2)]})
        code, result = self.gate(self.run_dir('a', self.frames()), self.run_dir('b', changed))
        self.assertEqual(code, 2)
        self.assertEqual((result['only_reference'], result['only_candidate']), (1, 1))

    def test_score_shift_fails(self):
        changed = self.frames({0: [box(100, 0.8), box(300, 0.6, 2)]})
        code, _ = self.gate(self.run_dir('a', self.frames()), self.run_dir('b', changed))
        self.assertEqual(code, 2)

    def test_box_shift_fails(self):
        changed = self.frames({4: [box(110), box(300, 0.6, 2)]})
        code, _ = self.gate(self.run_dir('a', self.frames()), self.run_dir('b', changed))
        self.assertEqual(code, 2)

    def test_too_few_frames_fails(self):
        few = {0: [box(100)], 1: [box(100)]}
        code, result = self.gate(self.run_dir('a', few), self.run_dir('b', few))
        self.assertEqual(code, 2)
        self.assertEqual(result['frames_compared'], 2)

    def test_frames_without_overlap_fail(self):
        a = {i: [box(100)] for i in range(5)}
        b = {i + 10: [box(100)] for i in range(5)}
        code, result = self.gate(self.run_dir('a', a), self.run_dir('b', b))
        self.assertEqual(code, 2)
        self.assertEqual(result['frames_compared'], 0)


class Agreement(unittest.TestCase):
    setUp = CompareDetections.setUp
    tearDown = CompareDetections.tearDown
    run_dir = CompareDetections.run_dir
    gate = CompareDetections.gate

    """Looser mode for lower precision: how much of the FP32 output does the candidate reproduce?"""
    def gate_agree(self, reference, candidate, *extra):
        out = self.root / 'agree.json'
        done = subprocess.run([sys.executable, str(SCRIPT), '--agreement', '--reference', reference, '--candidate',
                               candidate, '--output', str(out), '--min-frames', '3', *extra], capture_output=True, text=True)
        return done.returncode, json.loads(out.read_text())

    def many(self, change=None, count=10):
        base = {i: [box(20 * k, 0.9 - k / 100, k % 3) for k in range(count)] for i in range(5)}
        base.update(change or {})
        return base

    def test_identical_is_full_agreement(self):
        code, r = self.gate_agree(self.run_dir('a', self.many()), self.run_dir('b', self.many()))
        self.assertEqual(code, 0)
        self.assertEqual((r['recall'], r['precision']), (1.0, 1.0))

    def test_small_jitter_counts_as_agreement_at_half_iou(self):
        jitter = {i: [dict(d, xyxy=[d['xyxy'][0] + 3, d['xyxy'][1], d['xyxy'][2] + 3, d['xyxy'][3]], score=d['score'] - .02)
                      for d in dets] for i, dets in self.many().items()}
        code, r = self.gate_agree(self.run_dir('a', self.many()), self.run_dir('b', jitter))
        self.assertEqual(code, 0, r)
        self.assertEqual(r['recall'], 1.0)
        strict, _ = self.gate(self.run_dir('c', self.many()), self.run_dir('d', jitter))
        self.assertEqual(strict, 2)                                # strict mode still rejects the same output

    def test_missing_detections_lower_recall_and_fail_threshold(self):
        fewer = {i: dets[:8] for i, dets in self.many().items()}   # 2 of 10 missing in every frame
        code, r = self.gate_agree(self.run_dir('a', self.many()), self.run_dir('b', fewer))
        self.assertAlmostEqual(r['recall'], .8)
        self.assertEqual(r['precision'], 1.0)
        self.assertEqual(code, 2)                                  # default min_recall 0.95

    def test_extra_detections_lower_precision(self):
        extra = {i: dets + [box(900, .5, 0)] for i, dets in self.many().items()}
        _, r = self.gate_agree(self.run_dir('a', self.many()), self.run_dir('b', extra), '--min-precision', '0.5')
        self.assertEqual(r['recall'], 1.0)
        self.assertAlmostEqual(r['precision'], 10 / 11, places=4)

    def test_recall_is_reported_by_reference_score_band(self):
        # Every frame: two confident detections (0.9) and two borderline ones (0.3). The candidate drops the borderline ones.
        ref = {i: [box(0, .9, 0), box(100, .9, 1), box(200, .3, 0), box(300, .3, 1)] for i in range(5)}
        cand = {i: [box(0, .9, 0), box(100, .9, 1)] for i in range(5)}
        _, r = self.gate_agree(self.run_dir('a', ref), self.run_dir('b', cand))
        bands = {tuple(b['range']): b for b in r['recall_by_reference_score']}
        self.assertEqual(r['recall'], .5)
        self.assertEqual(bands[(0.8, 1.01)]['recall'], 1.0)
        self.assertEqual(bands[(0.25, 0.4)]['recall'], 0.0)
        self.assertEqual(bands[(0.25, 0.4)]['reference'], 10)

    def test_precision_is_reported_by_candidate_score_band(self):
        ref = {i: [box(0, .9, 0)] for i in range(5)}
        cand = {i: [box(0, .9, 0), box(500, .3, 2)] for i in range(5)}
        _, r = self.gate_agree(self.run_dir('a', ref), self.run_dir('b', cand), '--min-precision', '0.1')
        bands = {tuple(b['range']): b for b in r['precision_by_candidate_score']}
        self.assertEqual(bands[(0.8, 1.01)]['precision'], 1.0)
        self.assertEqual(bands[(0.25, 0.4)]['precision'], 0.0)

    def test_min_score_filters_each_side_before_matching(self):
        ref = {i: [box(0, .9, 0), box(200, .3, 0)] for i in range(5)}
        cand = {i: [box(0, .9, 0)] for i in range(5)}
        _, loose = self.gate_agree(self.run_dir('a', ref), self.run_dir('b', cand))
        self.assertEqual(loose['recall'], .5)
        code, strict = self.gate_agree(self.run_dir('c', ref), self.run_dir('d', cand),
                                       '--reference-min-score', '0.5', '--candidate-min-score', '0.5')
        self.assertEqual((code, strict['recall'], strict['reference_detections']), (0, 1.0, 5))

    def test_signed_score_shift_shows_direction(self):
        ref = {i: [box(0, .5, 0), box(200, .3, 1)] for i in range(5)}
        lower = {i: [box(0, .45, 0), box(200, .25, 1)] for i in range(5)}
        _, r = self.gate_agree(self.run_dir('a', ref), self.run_dir('b', lower))
        self.assertAlmostEqual(r['mean_signed_score_delta'], -0.05, places=6)   # candidate minus reference
        bands = {tuple(b['range']): b for b in r['recall_by_reference_score']}
        self.assertAlmostEqual(bands[(0.25, 0.4)]['mean_signed_score_delta'], -0.05, places=6)

    def test_thresholds_are_configurable(self):
        fewer = {i: dets[:8] for i, dets in self.many().items()}
        code, _ = self.gate_agree(self.run_dir('a', self.many()), self.run_dir('b', fewer), '--min-recall', '0.7')
        self.assertEqual(code, 0)


if __name__ == '__main__':
    unittest.main()
