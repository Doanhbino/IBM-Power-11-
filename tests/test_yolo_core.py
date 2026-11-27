"""NumPy/C++ parity for YOLO11 letterbox and decoding (needs the built rfdetr_adapter)."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import decode_yolo, preprocess_letterbox

BINARY = Path(os.environ.get('RFDETR_ADAPTER_BINARY', ROOT / 'build/rfdetr_adapter'))


@unittest.skipUnless(BINARY.is_file(), 'Build C++ core first (cmake -S . -B build)')
class YoloCoreParity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rng = np.random.default_rng(7)

    def tearDown(self):
        self.temp.cleanup()

    def run_adapter(self, *args):
        subprocess.run([str(BINARY), *map(str, args)], check=True, capture_output=True)

    def test_letterbox_matches_numpy(self):
        for ih, iw in [(1, 1), (1, 9), (11, 1), (3, 7), (23, 19), (100, 200), (200, 100)]:
            stride = iw * 3 + 7
            storage = self.rng.integers(0, 256, size=(ih, stride), dtype=np.uint8)
            frame = np.ndarray((ih, iw, 3), dtype=np.uint8, buffer=storage, strides=(stride, 3, 1))
            storage.tofile(self.root / 'image.bin')
            for h, w in [(8, 8), (19, 2), (64, 64), (8, 9)]:
                self.run_adapter('letterbox', self.root / 'image.bin', ih, iw, stride, h, w, 114,
                                 self.root / 'tensor.bin', self.root / 'meta.txt')
                actual = np.fromfile(self.root / 'tensor.bin', dtype=np.float32).reshape(1, 3, h, w)
                expected, meta = preprocess_letterbox(frame, h, w)
                np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=2e-6)
                scale, pad_x, pad_y = (self.root / 'meta.txt').read_text().strip().split(',')
                self.assertEqual((int(pad_x), int(pad_y)), (meta['pad_x'], meta['pad_y']))
                self.assertAlmostEqual(float(scale), meta['scale'], delta=meta['scale'] * 1e-6)

    def test_decode_matches_numpy(self):
        meta = dict(scale=.5, pad_x=3, pad_y=5)
        for classes, anchors in [(1, 1), (1, 17), (4, 17), (4, 300)]:
            for variant in ('random', 'tied_scores', 'duplicates'):
                out = np.zeros((1, 4 + classes, anchors), np.float32)
                out[0, 0:2] = self.rng.uniform(0, 100, (2, anchors))
                out[0, 2:4] = self.rng.uniform(1, 40, (2, anchors))
                out[0, 4:] = self.rng.uniform(0, 1, (classes, anchors))
                if variant == 'tied_scores':
                    out[0, 4:] = .6
                if variant == 'duplicates':
                    out[0, :4, 1::2] = out[0, :4, 0::2][:, :out[0, :4, 1::2].shape[1]]
                out.tofile(self.root / 'output.bin')
                for max_det in (3, 300):
                    cfg = dict(output_name='o', num_classes=classes, conf_threshold=.25, iou_threshold=.5, max_det=max_det)
                    self.run_adapter('yolo-decode', self.root / 'output.bin', 4 + classes, anchors, 123, 241,
                                     repr(meta['scale']), meta['pad_x'], meta['pad_y'], classes, .25, .5, max_det,
                                     self.root / 'detections.csv')
                    boxes, scores, slots = decode_yolo({'o': out}, cfg, (123, 241), meta)
                    text = (self.root / 'detections.csv').read_text()
                    if not len(scores):
                        self.assertEqual(text, '')
                        continue
                    actual = np.loadtxt(self.root / 'detections.csv', delimiter=',', ndmin=2)
                    np.testing.assert_allclose(actual[:, :4], boxes, atol=3e-5)
                    np.testing.assert_allclose(actual[:, 4], scores, atol=1e-7)
                    np.testing.assert_equal(actual[:, 5], slots)

    def test_decode_rejects_wrong_channel_count(self):
        np.zeros((84, 3), np.float32).tofile(self.root / 'output.bin')
        result = subprocess.run([str(BINARY), 'yolo-decode', str(self.root / 'output.bin'), '84', '3', '100', '100',
                                 '1', '0', '0', '4', '.25', '.5', '300', str(self.root / 'o.csv')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('num_classes=4 expects 8 channels', result.stderr)
        self.assertIn('84', result.stderr)


if __name__ == '__main__':
    unittest.main()
