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


if __name__ == '__main__':
    unittest.main()
