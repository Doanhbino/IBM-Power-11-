#!/usr/bin/env python3
"""Evaluate exported detection graph on a fixed COCO split with shared preprocessing/decoder."""
import argparse
import json
from pathlib import Path
import time


def evaluate_files(annotations, predictions, report_path, image_ids=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    ground = COCO(str(annotations))
    rows = json.loads(Path(predictions).read_text())
    if rows:
        result = ground.loadRes(rows)
    else:
        # pycocotools.loadRes indexes anns[0]; explicitly support a valid empty result.
        result = COCO()
        result.dataset = {'images': list(ground.dataset['images']),
                          'categories': list(ground.dataset['categories']), 'annotations': []}
        result.createIndex()
    evaluator = COCOeval(ground, result, 'bbox')
    evaluator.params.imgIds = sorted(image_ids if image_ids is not None else ground.getImgIds())
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    keys = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
            'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large']
    report = {'annotations': str(annotations), 'predictions': str(predictions),
              'images': len(evaluator.params.imgIds), 'iou_type': 'bbox',
              'max_dets': list(evaluator.params.maxDets),
              'metrics_0_to_1': dict(zip(keys, map(float, evaluator.stats)))}
    Path(report_path).write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--annotations', required=True)
    ap.add_argument('--images', help='COCO image root; required unless --predictions supplied')
    ap.add_argument('--config', help='Shared benchmark JSON; required unless --predictions supplied')
    ap.add_argument('--predictions', help='Existing COCO predictions, e.g. GPU or eager reference')
    ap.add_argument('--image-ids', help='JSON array of exact image IDs; omit for entire split')
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected = json.loads(Path(args.image_ids).read_text()) if args.image_ids else None
    predictions = args.predictions
    if predictions is None:
        if not args.config or not args.images:
            ap.error('--config and --images required when generating predictions')
        import cv2
        from common import session, preprocess_bgr, decode, sha256
        cfg = json.loads(Path(args.config).read_text())
        # Low floor for COCO AP: a deployment confidence=0.5 would censor PR curves.
        cfg['threshold'] = float(cfg.get('accuracy_threshold', 0.001))
        cfg['top_k'] = int(cfg.get('accuracy_top_k', 300))
        category_ids = cfg.get('category_ids')
        if category_ids is None:
            raise ValueError('category_ids is required: raw logit slot -> COCO category ID; do not assume +1')
        gt = json.loads(Path(args.annotations).read_text())
        available_ids = {int(image['id']) for image in gt['images']}
        if selected is not None and not set(selected) <= available_ids:
            raise ValueError('image-ids contains IDs absent from annotation split')
        allowed_categories = {int(cat['id']) for cat in gt['categories']}
        selected_set = set(selected) if selected is not None else available_ids
        records = sorted((image for image in gt['images'] if image['id'] in selected_set), key=lambda x: x['id'])
        sess = session(cfg)
        input_meta = sess.get_inputs()[0]
        outputs = [x.name for x in sess.get_outputs()]
        h, w = cfg.get('size', [cfg.get('height'), cfg.get('width')])
        if not h or not w:
            shape = input_meta.shape
            if len(shape) != 4 or not all(isinstance(d, int) for d in shape[2:4]):
                raise ValueError('Set height/width for dynamic ONNX spatial dimensions')
            h, w = shape[2:4]
        if any(isinstance(d, int) and d != wanted for d, wanted in zip(input_meta.shape[2:4], (h, w))):
            raise ValueError('Configured size does not match fixed ONNX input shape')
        mean = cfg.get('mean', [0.485, 0.456, 0.406])
        std = cfg.get('std', [0.229, 0.224, 0.225])
        rows = []
        started = time.perf_counter()
        for image in records:
            image_path = (Path(args.images) / image['file_name']).resolve()
            if not image_path.is_relative_to(Path(args.images).resolve()):
                raise ValueError('COCO file_name escapes image root')
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise ValueError(f'Cannot decode {image_path}')
            if tuple(frame.shape[:2]) != (image['height'], image['width']):
                raise ValueError(f'Image geometry differs from COCO annotations: {image_path}')
            x = preprocess_bgr(frame, h, w, mean, std)
            raw = dict(zip(outputs, sess.run(None, {input_meta.name: x})))
            boxes, scores, classes = decode(raw, cfg, frame.shape[:2])
            for box, score, slot in zip(boxes, scores, classes):
                slot = int(slot)
                try:
                    category = category_ids[str(slot)] if isinstance(category_ids, dict) else category_ids[slot]
                except (KeyError, IndexError):
                    raise ValueError(f'Missing category mapping for raw logit slot {slot}')
                if category is None:
                    continue
                category = int(category)
                if category not in allowed_categories:
                    raise ValueError(f'COCO category {category} absent from annotations')
                x1, y1, x2, y2 = map(float, box)
                rows.append({'image_id': int(image['id']), 'category_id': category,
                             'bbox': [x1, y1, max(0.0, x2-x1), max(0.0, y2-y1)], 'score': float(score)})
        predictions = out / 'predictions.coco.json'
        predictions.write_text(json.dumps(rows))
        (out / 'inference_manifest.json').write_text(json.dumps({'config': cfg, 'images': len(records),
            'wall_seconds_including_io_pre_post': time.perf_counter() - started,
            'providers': sess.get_providers(), 'model_sha256': sha256(cfg['model']),
            'annotations_sha256': sha256(args.annotations), 'predictions_sha256': sha256(predictions)}, indent=2))
    report = evaluate_files(args.annotations, predictions, out / 'accuracy.json', selected)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
