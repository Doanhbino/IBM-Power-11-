"""Dataset preflight fixtures test paths/labels; image decoding and training are separate."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from training_data import class_names, inspect_dataset


class TrainingData(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.train = {'dataset_dir': str(self.root), 'dataset_file': 'yolo', 'run_test': False}
        self.yaml = self.root / 'data.yaml'
        self.yaml.write_text('names: [car, motorcycle]\nnc: 2\ntrain: train/images\nval: valid/images\n')
        for split in ('train', 'valid'):
            (self.root / split / 'images').mkdir(parents=True)
            (self.root / split / 'labels').mkdir()
            (self.root / split / 'images/one.jpg').write_bytes(b'file-path-fixture-not-an-image')
            (self.root / split / 'labels/one.txt').write_text('0 0.5 0.5 0.2 0.4\n1 0.3 0.4 0.1 0.2\n')

    def tearDown(self):
        self.temporary.cleanup()

    def test_detection_counts_and_optional_test_split(self):
        result = inspect_dataset(self.train)
        self.assertEqual(result['format'], 'yolo')
        self.assertEqual(result['class_names'], ['car', 'motorcycle'])
        self.assertEqual(result['splits']['train']['objects_by_class'], [1, 1])
        self.assertEqual(len(result['annotation_files']), 3)
        self.assertNotIn('test', result['splits'])
        with self.assertRaises(FileNotFoundError):
            inspect_dataset(dict(self.train, run_test=True))

    def test_images_train_layout_and_dictionary_names(self):
        for split, yaml_split in [('train', 'train'), ('valid', 'val')]:
            (self.root / 'images' / yaml_split).mkdir(parents=True)
            (self.root / 'labels' / yaml_split).mkdir(parents=True)
            (self.root / split / 'images/one.jpg').rename(self.root / 'images' / yaml_split / 'one.jpg')
            (self.root / split / 'labels/one.txt').rename(self.root / 'labels' / yaml_split / 'one.txt')
        self.yaml.write_text('path: .\nnames: {"0": car, "1": motorcycle}\nnc: 2\ntrain: images/train\nval: images/val\n')
        result = inspect_dataset(self.train)
        self.assertEqual(result['splits']['val']['images'], 1)
        self.assertEqual(result['class_names'], ['car', 'motorcycle'])

    def test_invalid_box_class_and_polygon_rows_rejected(self):
        for line in ('2 0.5 0.5 0.2 0.4', '0 nan 0.5 0.2 0.4', '0 1.1 0.5 0.2 0.4',
                     '0 0.5 0.5 0 0.4', '0 0.5 0.5 -0.1 0.4', '0.0 0.5 0.5 0.2 0.4',
                     '0 0.1 0.1 0.5 0.1 0.5 0.5'):
            with self.subTest(line=line):
                (self.root / 'train/labels/one.txt').write_text(line)
                with self.assertRaisesRegex(ValueError, 'one.txt:1'):
                    inspect_dataset(self.train)

    def test_missing_label_and_explicit_negative_image(self):
        label = self.root / 'valid/labels/one.txt'
        label.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Missing YOLO label'):
            inspect_dataset(self.train)
        label.write_text('')
        result = inspect_dataset(self.train)
        self.assertEqual(result['splits']['val']['negative_images'], 1)
        (self.root / 'train/labels/one.txt').write_text('')
        with self.assertRaisesRegex(ValueError, 'no labeled objects'):
            inspect_dataset(self.train)

    def test_class_mapping_must_be_unambiguous(self):
        for cfg in ({'names': ['car'], 'nc': 2}, {'names': {1: 'car'}},
                    {'names': {0: 'car', '0': 'motorcycle'}}, {'names': []},
                    {'names': {True: 'car'}}, {'names': ['']}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                class_names(cfg)

    def test_path_escape_and_shared_splits_rejected(self):
        self.yaml.write_text('names: [car, motorcycle]\ntrain: ../images\nval: valid/images\n')
        with self.assertRaisesRegex(ValueError, 'escapes dataset_dir'):
            inspect_dataset(self.train)
        self.yaml.write_text('names: [car, motorcycle]\ntrain: train/images\nval: train/images\n')
        with self.assertRaisesRegex(ValueError, 'splits share'):
            inspect_dataset(self.train)

    def test_coco_auto_detection_preserved_and_explicit_yolo_wins(self):
        for split in ('train', 'valid', 'test'):
            (self.root / split).mkdir(exist_ok=True)
            (self.root / split / '_annotations.coco.json').write_text('{}')
        auto = inspect_dataset({'dataset_dir': str(self.root)})
        self.assertEqual(auto['format'], 'roboflow')
        self.assertEqual(len(auto['annotation_files']), 3)
        self.assertEqual(inspect_dataset(self.train)['format'], 'yolo')

    def test_cli_check_only_does_not_load_weights_or_train(self):
        cfg = json.loads((ROOT / 'configs/train-traffic-smoke.json').read_text())
        cfg['train'].update(self.train, output_dir=str(self.root / 'output'))
        cfg['model']['kwargs']['pretrain_weights'] = str(self.root / 'dummy.pth')
        (self.root / 'dummy.pth').write_bytes(b'existence-check-only')
        path = self.root / 'config.json'
        path.write_text(json.dumps(cfg))
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/train.py'), '--config', str(path), '--check-only'],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report['training_not_run'])
        self.assertEqual(report['dataset']['format'], 'yolo')
        self.assertFalse((self.root / 'output').exists())


if __name__ == '__main__':
    unittest.main()
