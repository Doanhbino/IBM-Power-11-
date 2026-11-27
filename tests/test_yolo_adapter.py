"""NumPy reference for YOLO11 letterbox, NMS and decoding; no ONNX or OpenCV needed."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
from common import MODEL_TYPES, decode_yolo, model_type, nms, preprocess_letterbox

CFG = dict(output_name='output0', num_classes=2, conf_threshold=.25, iou_threshold=.5, max_det=300)
IDENTITY = dict(scale=1.0, pad_x=0, pad_y=0)


def raw(anchors, classes=2):
    """anchors: [(cx, cy, w, h, [class scores])] -> {'output0': [1, 4+C, N]}"""
    out = np.zeros((1, 4 + classes, len(anchors)), np.float32)
    for i, (cx, cy, w, h, scores) in enumerate(anchors):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4:, i] = scores
    return {'output0': out}


class Letterbox(unittest.TestCase):
    def test_letterbox_wide_image(self):
        x, meta = preprocess_letterbox(np.zeros((100, 200, 3), np.uint8), 640, 640)
        self.assertEqual(x.shape, (1, 3, 640, 640))
        self.assertAlmostEqual(meta['scale'], 3.2)
        self.assertEqual((meta['pad_x'], meta['pad_y']), (0, 160))
        np.testing.assert_equal(x[0, :, 159, :], np.float32(114 / 255))
        np.testing.assert_equal(x[0, :, 480, :], np.float32(114 / 255))
        np.testing.assert_equal(x[0, :, 160, :], 0)

    def test_letterbox_tall_and_tiny(self):
        rng = np.random.default_rng(0)
        for shape in [(1, 1), (200, 100)]:
            x, _ = preprocess_letterbox(rng.integers(0, 256, (*shape, 3), dtype=np.uint8), 64, 64)
            self.assertEqual(x.shape, (1, 3, 64, 64))
            self.assertTrue(x.min() >= 0 and x.max() <= 1)

    def test_letterbox_extreme_aspect_keeps_one_pixel(self):
        x, meta = preprocess_letterbox(np.full((1, 1000, 3), 255, np.uint8), 64, 64)
        self.assertEqual(x.shape, (1, 3, 64, 64))
        self.assertEqual(meta['pad_x'], 0)
        self.assertGreaterEqual(float(x[0, 0, meta['pad_y']].max()), 0.99)

    def test_letterbox_round_trip(self):
        cfg = dict(CFG, num_classes=1)
        for ih, iw in [(100, 200), (200, 100), (37, 91), (480, 640), (1080, 1920)]:
            _, meta = preprocess_letterbox(np.zeros((ih, iw, 3), np.uint8), 640, 640)
            new_w, new_h = round(iw * meta['scale']), round(ih * meta['scale'])
            px, py = meta['pad_x'], meta['pad_y']
            out = raw([(px + new_w / 2, py + new_h / 2, new_w, new_h, [.9])], classes=1)
            boxes, _, _ = decode_yolo(out, cfg, (ih, iw), meta)
            np.testing.assert_allclose(boxes[0], [0, 0, iw, ih], atol=1.0)


class Nms(unittest.TestCase):
    def test_nms_same_class_and_threshold_equality(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 5]], np.float32)  # IoU exactly 0.5
        scores = np.array([.9, .8], np.float32)
        self.assertEqual(nms(boxes, scores, .5), [0, 1])
        self.assertEqual(nms(boxes, scores, .49), [0])

    def test_nms_tie_prefers_lower_index(self):
        same = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], np.float32)
        self.assertEqual(nms(same, np.array([.5, .5], np.float32), .5), [0])
        apart = np.array([[0, 0, 1, 1], [5, 5, 6, 6], [9, 9, 10, 10]], np.float32)
        self.assertEqual(nms(apart, np.array([.5, .5, .5], np.float32), .5), [0, 1, 2])


class Decode(unittest.TestCase):
    def test_decode_per_class_nms_keeps_other_class(self):
        different = raw([(50, 50, 20, 20, [.9, 0]), (50, 50, 20, 20, [0, .8])])
        _, scores, slots = decode_yolo(different, CFG, (100, 100), IDENTITY)
        self.assertEqual(list(slots), [0, 1]); np.testing.assert_allclose(scores, [.9, .8], atol=1e-6)
        same = raw([(50, 50, 20, 20, [.9, 0]), (50, 50, 20, 20, [.8, 0])])
        _, scores, slots = decode_yolo(same, CFG, (100, 100), IDENTITY)
        self.assertEqual(list(slots), [0]); self.assertEqual(len(scores), 1)

    def test_decode_best_class_per_anchor(self):
        out = raw([(50, 50, 20, 20, [0, .9, .8])], classes=3)
        _, scores, slots = decode_yolo(out, dict(CFG, num_classes=3), (100, 100), IDENTITY)
        self.assertEqual(list(slots), [1]); self.assertEqual(len(scores), 1)

    def test_decode_empty_below_threshold(self):
        boxes, scores, slots = decode_yolo(raw([(50, 50, 20, 20, [.1, .1])]), CFG, (100, 100), IDENTITY)
        self.assertEqual((boxes.shape, len(scores), len(slots)), ((0, 4), 0, 0))

    def test_decode_nonfinite_raises(self):
        out = raw([(50, 50, 20, 20, [.9, 0])]); out['output0'][0, 4, 0] = np.nan
        with self.assertRaises(ValueError):
            decode_yolo(out, CFG, (100, 100), IDENTITY)

    def test_decode_max_det(self):
        anchors = [(10 + 20 * i, 50, 10, 10, [.9 - .1 * i, 0]) for i in range(5)]
        _, scores, _ = decode_yolo(raw(anchors), dict(CFG, max_det=3), (100, 200), IDENTITY)
        np.testing.assert_allclose(scores, [.9, .8, .7], atol=1e-6)

    def test_decode_maps_back_and_clips(self):
        out = raw([(250, 250, 500, 500, [.9, 0])])
        boxes, _, _ = decode_yolo(out, CFG, (100, 200), dict(scale=.5, pad_x=10, pad_y=20))
        np.testing.assert_allclose(boxes[0], [0, 0, 200, 100])

    def test_channel_mismatch_names_both_numbers(self):
        out = {'output0': np.zeros((1, 84, 5), np.float32)}
        with self.assertRaises(ValueError) as caught:
            decode_yolo(out, dict(CFG, num_classes=4), (100, 100), IDENTITY)
        self.assertIn('4', str(caught.exception)); self.assertIn('80', str(caught.exception))

    def test_missing_cfg_key_is_named(self):
        cfg = {k: v for k, v in CFG.items() if k != 'iou_threshold'}
        with self.assertRaises(ValueError) as caught:
            decode_yolo(raw([(50, 50, 20, 20, [.9, 0])]), cfg, (100, 100), IDENTITY)
        self.assertIn('iou_threshold', str(caught.exception))


class ModelType(unittest.TestCase):
    def test_model_type_default_and_unknown(self):
        self.assertEqual(MODEL_TYPES, ('rfdetr', 'yolo11'))
        self.assertEqual(model_type({}), 'rfdetr')
        self.assertEqual(model_type({'model_type': 'yolo11'}), 'yolo11')
        with self.assertRaises(ValueError):
            model_type({'model_type': 'yolo9'})


if __name__ == '__main__':
    unittest.main()
