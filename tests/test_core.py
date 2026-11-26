"""Independent NumPy/C++ parity and lifecycle checks for the portable core."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import decode, preprocess_bgr

BINARY = Path(os.environ.get('RFDETR_ADAPTER_BINARY', ROOT / 'build/rfdetr_adapter'))


@unittest.skipUnless(BINARY.is_file(), 'Build C++ core first (cmake -S . -B build)')
class CoreParity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rng = np.random.default_rng(42)

    def tearDown(self):
        self.temp.cleanup()

    def test_preprocess_resize_and_padded_stride(self):
        for ih, iw in [(1, 1), (1, 9), (11, 1), (3, 7), (23, 19)]:
            stride = iw * 3 + 7
            storage = self.rng.integers(0, 256, size=(ih, stride), dtype=np.uint8)
            frame = np.ndarray((ih, iw, 3), dtype=np.uint8, buffer=storage,
                               strides=(stride, 3, 1))
            source, output = self.root / 'image.bin', self.root / 'tensor.bin'
            storage.tofile(source)
            for h, w in [(1, 1), (2, 13), (19, 2), (8, 9)]:
                subprocess.run([str(BINARY), 'preprocess', str(source), str(ih), str(iw),
                                str(stride), str(h), str(w), str(output)], check=True, capture_output=True)
                actual = np.fromfile(output, dtype=np.float32).reshape(1, 3, h, w)
                expected = preprocess_bgr(frame, h, w)
                np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=2e-6)

    def test_decoder_global_topk_background_and_ties(self):
        for q, c in [(1, 1), (3, 4), (17, 8), (300, 8)]:
            boxes = self.rng.uniform(-.5, 1.5, size=(1, q, 4)).astype(np.float32)
            logits = self.rng.normal(0, 5, size=(1, q, c)).astype(np.float32)
            for tied in (False, True):
                if tied:
                    logits.fill(0)
                for background in ([None] if c == 1 else [None, 0, -1, 1]):
                    cfg = dict(boxes_output='boxes', logits_output='logits',
                               background_index=background, top_k=min(30, q * c), threshold=.25)
                    boxes.tofile(self.root / 'boxes.bin')
                    logits.tofile(self.root / 'logits.bin')
                    output = self.root / 'detections.csv'
                    subprocess.run([str(BINARY), 'decode', str(self.root / 'boxes.bin'),
                                    str(self.root / 'logits.bin'), str(q), str(c), '123', '241',
                                    'none' if background is None else str(background),
                                    str(cfg['top_k']), str(cfg['threshold']), str(output)],
                                   check=True, capture_output=True)
                    b, scores, slots = decode(dict(boxes=boxes, logits=logits), cfg, (123, 241))
                    if not len(scores):
                        self.assertEqual(output.read_text(), '')
                        continue
                    actual = np.loadtxt(output, delimiter=',', ndmin=2)
                    np.testing.assert_allclose(actual[:, :4], b, atol=3e-5)
                    np.testing.assert_allclose(actual[:, 4], scores, atol=1e-7)
                    np.testing.assert_equal(actual[:, 5], slots)

    def test_mma_request_cannot_silently_fallback(self):
        binary = BINARY.with_name('rfdetr_gemm')
        capability = json.loads(subprocess.check_output([str(binary), '--capability'], text=True))
        if capability['compiled'] and capability['exposed']:
            self.skipTest('MMA actually available; covered by native core_tests')
        result = subprocess.run([str(binary), '5', '7', '3', 'mma', '1'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('refusing fallback', result.stderr)


if __name__ == '__main__':
    unittest.main()
