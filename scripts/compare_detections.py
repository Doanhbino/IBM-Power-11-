#!/usr/bin/env python3
"""Correctness gate: compare decoded detections of two camera runs frame by frame.

Run both configurations on the same replay clip (replay blocks the producer, so frame IDs line up),
then: compare_detections.py --reference runs/R0 --candidate runs/R1 --output gate.json
Exit status: 0 gate passed, 2 gate failed or too few comparable frames.
"""
import argparse
import json
from pathlib import Path
import sys

from run_record import atomic_json


def load(run_dir):
    frames = {}
    for path in sorted(Path(run_dir).glob('worker-*/detections.jsonl')):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                frames[(row['camera_id'], row['frame_id'])] = row['detections']
    if not frames:
        raise SystemExit(f'No detections.jsonl under {run_dir}')
    return frames


def iou(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(reference, candidate, iou_min):
    """Greedy highest-IoU matching within the same class slot. Returns pairs and unmatched counts."""
    pairs = sorted(((iou(r['xyxy'], c['xyxy']), i, j) for i, r in enumerate(reference)
                    for j, c in enumerate(candidate) if r['class_slot'] == c['class_slot']), reverse=True)
    used_r, used_c, matched = set(), set(), []
    for value, i, j in pairs:
        if value < iou_min:
            break
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        matched.append((value, abs(reference[i]['score'] - candidate[j]['score'])))
    return matched, len(reference) - len(used_r), len(candidate) - len(used_c)


def compare(reference, candidate, iou_min, score_tol, min_frames):
    common = sorted(set(reference) & set(candidate))
    totals = dict(matched=0, only_reference=0, only_candidate=0, frames_exact=0)
    min_iou, max_score_delta = 1.0, 0.0
    for key in common:
        matched, only_r, only_c = match(reference[key], candidate[key], iou_min)
        totals['matched'] += len(matched)
        totals['only_reference'] += only_r
        totals['only_candidate'] += only_c
        within = all(delta <= score_tol for _, delta in matched)
        if not only_r and not only_c and within:
            totals['frames_exact'] += 1
        for value, delta in matched:
            min_iou = min(min_iou, value)
            max_score_delta = max(max_score_delta, delta)
    passed = (len(common) >= min_frames and totals['frames_exact'] == len(common))
    return {'frames_compared': len(common), 'frames_only_in_reference': len(set(reference) - set(candidate)),
            'frames_only_in_candidate': len(set(candidate) - set(reference)), **totals,
            'min_matched_iou': min_iou, 'max_score_delta': max_score_delta,
            'thresholds': {'iou_min': iou_min, 'score_tol': score_tol, 'min_frames': min_frames},
            'passed': passed}


def agreement(reference, candidate, iou_min, min_frames, min_recall, min_precision):
    """Lower-precision check: what fraction of the reference detections does the candidate reproduce (recall), and
    what fraction of the candidate's detections exist in the reference (precision). Same class slot, IoU >= iou_min.
    This is agreement with the reference model, NOT accuracy against ground truth."""
    common = sorted(set(reference) & set(candidate))
    matched = ref_total = cand_total = 0
    iou_sum, score_delta_sum = 0.0, 0.0
    for key in common:
        pairs, only_r, only_c = match(reference[key], candidate[key], iou_min)
        matched += len(pairs)
        ref_total += len(reference[key])
        cand_total += len(candidate[key])
        for value, delta in pairs:
            iou_sum += value
            score_delta_sum += delta
    recall = matched / ref_total if ref_total else 1.0
    precision = matched / cand_total if cand_total else 1.0
    return {'mode': 'agreement', 'frames_compared': len(common), 'reference_detections': ref_total,
            'candidate_detections': cand_total, 'matched': matched, 'recall': recall, 'precision': precision,
            'mean_matched_iou': iou_sum / matched if matched else None,
            'mean_score_delta': score_delta_sum / matched if matched else None,
            'thresholds': {'iou_min': iou_min, 'min_recall': min_recall, 'min_precision': min_precision,
                           'min_frames': min_frames},
            'passed': len(common) >= min_frames and recall >= min_recall and precision >= min_precision}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--iou-min', type=float, default=0.99)
    parser.add_argument('--score-tol', type=float, default=1e-3)
    parser.add_argument('--min-frames', type=int, default=100)
    parser.add_argument('--agreement', action='store_true',
                        help='tolerant mode for lower precision: recall/precision against the reference, IoU >= 0.5')
    parser.add_argument('--min-recall', type=float, default=0.95)
    parser.add_argument('--min-precision', type=float, default=0.95)
    args = parser.parse_args()
    if args.agreement:
        iou_min = args.iou_min if args.iou_min != 0.99 else 0.5     # strict default is not meant for this mode
        result = agreement(load(args.reference), load(args.candidate), iou_min, args.min_frames,
                           args.min_recall, args.min_precision)
        keys = ('frames_compared', 'reference_detections', 'candidate_detections', 'matched', 'recall', 'precision',
                'mean_matched_iou', 'mean_score_delta', 'passed')
    else:
        result = compare(load(args.reference), load(args.candidate), args.iou_min, args.score_tol, args.min_frames)
        keys = ('frames_compared', 'frames_exact', 'matched', 'only_reference', 'only_candidate', 'min_matched_iou',
                'max_score_delta', 'passed')
    result.update(reference=args.reference, candidate=args.candidate)
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in keys}))
    sys.exit(0 if result['passed'] else 2)


if __name__ == '__main__':
    main()
