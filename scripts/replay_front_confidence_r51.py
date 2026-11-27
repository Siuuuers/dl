#!/usr/bin/env python3
"""Replay one declared confidence probe from PR143 tensors; no model execution.

Exact reproduction of all 20 returned native 0.13 outputs is a prerequisite.
The only diagnostic change is box confidence 0.13 to 0.05. Reference-derived
proposal selections explain errors and never become a deployed ranking policy.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import platform
import zipfile

import numpy as np

from audit_additional_front_pilot_r45_results import corner_groups
from audit_additional_front_pilot_r47_results import REFERENCES
from evaluate_pose import parse_pose_labels
from export_front_error_review_r48 import corner_details
from pose_scoring import _box_iou, aggregate_scores, score_image
from predict_pose_folder import postclip_nms

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = '892c8ee79ff470990bb47dc59f947ad6bd2577a86bc9827e9dd57f81c56213f3'
ARCHIVE_BYTES = 15631804
R50_PATH = 'data/reviews/fronts-2026-11-27-r50/manifest.json'
R50_SHA = 'f639459bdd986f6553b9e00f0ecfdf1eee3c7f5344e95c300a4a17a66352c04f'
HIGH, LOW, NMS, MATCH, CAP = .13, .05, .70, .50, 300


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def put(path, data):
    path = Path(path)
    if path.exists():
        need(path.is_file() and path.read_bytes() == data, 'Different existing output preserved: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def input_boxes(raw):
    xywh = raw[0, :4].T
    boxes = np.empty_like(xywh)
    boxes[:, :2] = xywh[:, :2] - xywh[:, 2:] / 2
    boxes[:, 2:] = xywh[:, :2] + xywh[:, 2:] / 2
    return boxes


def original_boxes(boxes):
    # This fixed return uses unit gain and eight vertical padding pixels.
    boxes = boxes.copy()
    boxes[:, [1, 3]] -= np.float32(8)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, 1280)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, 720)
    return boxes


def native_select(boxes, scores, confidence):
    above = np.flatnonzero(scores > confidence)
    need(len(above) < 30000, 'Native proposal pre-cap requires a separate interpretation')
    need(len(set(scores[above].tolist())) == len(above), 'Native score ties require explicit handling')
    order = above[np.argsort(-scores[above], kind='stable')]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    kept, suppressed = [], {}
    while len(order) and len(kept) < CAP:
        index = int(order[0])
        kept.append(index)
        rest = order[1:]
        intersection = np.prod(np.maximum(
            np.minimum(boxes[rest, 2:], boxes[index, 2:]) -
            np.maximum(boxes[rest, :2], boxes[index, :2]), 0), axis=1)
        union = area[index] + area[rest] - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        for other, value in zip(rest, overlap):
            if value > NMS:
                suppressed[int(other)] = dict(suppressor=index, pair_iou_fp32=float(value))
        order = rest[overlap <= NMS]
    need(len(kept) < CAP, 'Output cap requires a separate interpretation')
    return kept, suppressed


def prediction_records(raw, boxes, indices):
    scaled = original_boxes(boxes[indices])
    points = raw[0, 5:].T[indices].copy().reshape(-1, 2, 3)
    points[:, :, 1] -= np.float32(8)
    points[:, :, 0] = np.clip(points[:, :, 0], 0, 1280)
    points[:, :, 1] = np.clip(points[:, :, 1], 0, 720)
    return [dict(class_id=0, bbox_xyxy=box.tolist(), confidence=float(raw[0, 4, index]), keypoints=point.tolist())
            for index, box, point in zip(indices, scaled, points)]


def load_references(root):
    references, contracts = {}, {}
    descriptions = dict(REFERENCES)
    descriptions['r50'] = (R50_PATH, R50_SHA, 654)
    for name, (relative, expected, count) in descriptions.items():
        path = root / relative
        data = path.read_bytes()
        need(sha(data) == expected, 'Reference manifest differs: ' + name)
        gold, labels = {}, {}
        for item in json.loads(data)['files']:
            if item.get('split', 'val') != 'val':
                continue
            data = (path.parent / item['label_path']).read_bytes()
            need(sha(data) == item['label_sha256'], 'Reference label differs: ' + item['image_id'])
            gold[item['image_id']] = parse_pose_labels(data.decode(), 1280, 720)
            labels[item['image_id']] = item['label_sha256']
        need(len(gold) == 20 and sum(map(len, gold.values())) == count, 'Reference membership differs: ' + name)
        references[name] = gold
        contracts[name] = dict(manifest_path=relative, manifest_sha256=expected, objects=count, labels_sha256=labels)
    return references, contracts


def compact_score(score):
    return {name: value for name, value in score.items() if name not in (
        'endpoint_errors_px', 'matches', 'unmatched_gold_indices', 'unmatched_prediction_indices',
        'duplicate_prediction_indices', 'other_unmatched_prediction_indices')}


def analyze(archive, root):
    data = archive.read_bytes()
    need(len(data) == ARCHIVE_BYTES and sha(data) == ARCHIVE_SHA, 'PR143 archive differs')
    references, reference_contract = load_references(root)
    outputs, validation, mechanism_rows = {}, [], []
    with zipfile.ZipFile(archive) as bundle:
        saved = json.loads(bundle.read('predictions.json'))
        traces = json.loads(bundle.read('trace.json'))
        original_cases = json.loads(bundle.read('missed-cases.json'))
        need(set(saved) == set(traces) == set(references['r38']), 'Image membership differs')
        for image, result in saved.items():
            trace = traces[image]
            need(trace['input_shape'] == [1, 3, 736, 1280], 'Fixed input geometry differs: ' + image)
            raw = np.load(io.BytesIO(bundle.read('decoded/' + image + '.npy')), allow_pickle=False)
            need(raw.dtype == np.dtype('float32') and raw.shape[:2] == (1, 11)
                 and np.isfinite(raw).all() and sha(raw.tobytes()) == trace['raw_tensor_sha256'], 'Decoded tensor differs')
            boxes = input_boxes(raw)
            indices, suppressed = native_select(boxes, raw[0, 4], HIGH)
            need(indices == result['ordinary_indices'], 'Native anchor reconstruction differs: ' + image)
            records = prediction_records(raw, boxes, indices)
            need(records == result['hard_raw'], 'Native original-coordinate boxes, scores or points differ: ' + image)
            high, high_audit = postclip_nms(records)
            need(high == result['hard'] and high_audit == result['hard_audit'], 'Native postclip replay differs: ' + image)
            low_indices, _ = native_select(boxes, raw[0, 4], LOW)
            low_raw = prediction_records(raw, boxes, low_indices)
            low, low_audit = postclip_nms(low_raw)
            low_final_indices = [low_indices[j - 1] for j in low_audit['retained_primary_prediction_ids']]
            outputs[image] = dict(hard013=high, hard005=low, raw005=low_raw,
                ordinary_indices005=low_indices, postclip_indices005=low_final_indices, postclip_audit005=low_audit)
            validation.append(dict(image=image, decoded_sha256=trace['raw_tensor_sha256'],
                native_anchor_indices_exact=True, native_full_predictions_exact=True,
                native_postclip_audit_exact=True, raw013=len(records), final013=len(high),
                raw005=len(low_raw), final005=len(low)))
            scaled = original_boxes(boxes).tolist()
            for case in (c for c in original_cases if c['image'] == image):
                gold = case['gold']
                overlaps = [_box_iou(box, gold['bbox_xyxy']) for box in scaled]
                eligible = [index for index, value in enumerate(overlaps) if value >= MATCH]
                need(eligible == [e['anchor_index'] for e in case['eligible']], 'Eligible proposal inventory differs')
                best = max(range(len(overlaps)), key=lambda index: (overlaps[index], float(raw[0, 4, index]), -index))
                need(best == case['best_decoded']['anchor_index'], 'Best decoded proposal differs')
                def describe(index):
                    prediction = prediction_records(raw, boxes, [index])[0]
                    return dict(anchor_index=index, iou=overlaps[index], **prediction)
                highest = max(eligible, key=lambda index: (float(raw[0, 4, index]), -index)) if eligible else None
                conflicts = []
                for index in eligible:
                    if float(raw[0, 4, index]) <= HIGH or index not in suppressed:
                        continue
                    winner = suppressed[index]['suppressor']
                    a, b = boxes[index].astype(np.float64), boxes[winner].astype(np.float64)
                    intersection = np.prod(np.maximum(np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]), 0))
                    overlap = float(intersection / (np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection))
                    conflicts.append(dict(candidate=describe(index), suppressor=describe(winner),
                        suppressor_primary_prediction_id=indices.index(winner) + 1,
                        pair_iou_before_clip=overlap,
                        score_after_this_linear_soft_decay=float(raw[0, 4, index]) * (1 - overlap)))
                mechanism_rows.append(dict(image=image, r38_row=case['r38_row'], stable_id=case['stable_id'],
                    locatable_corners=case['locatable_corners'], recorded_stage=case['mechanism'],
                    target=gold, same_target_in_r50=gold == references['r50'][image][case['r38_row'] - 1],
                    eligible_count=len(eligible), above013_eligible_count=sum(float(raw[0, 4, j]) > HIGH for j in eligible),
                    best_decoded=describe(best), highest_confidence_eligible=describe(highest) if highest is not None else None,
                    ordinary_nms_conflicts=conflicts))

    reports, internal_scores = {}, {}
    for reference, gold in references.items():
        reports[reference], internal_scores[reference] = {}, {}
        for arm in ('hard013', 'hard005'):
            scores, groups = {}, {str(n): Counter() for n in (0, 1, 2)}
            for image, targets in gold.items():
                predictions = outputs[image][arm]
                score = score_image(targets, predictions, 1280, 720, MATCH, .5)
                scores[image] = score
                for key, values in corner_groups(targets, predictions, score).items():
                    groups[key].update(values)
            combined = aggregate_scores(list(scores.values()))
            reports[reference][arm] = dict(detection=compact_score(combined),
                corners_by_locatable_count={key: dict(value) for key, value in groups.items()},
                empty_scene_outputs={image: score['prediction_count'] for image, score in scores.items() if not score['gold_count']})
            internal_scores[reference][arm] = scores
        high, low = (internal_scores[reference][arm] for arm in ('hard013', 'hard005'))
        reports[reference]['paired_delta'] = {
            image: dict(recovered_rows=[j + 1 for j in sorted(set(high[image]['unmatched_gold_indices']) - set(low[image]['unmatched_gold_indices']))],
                        newly_missed_rows=[j + 1 for j in sorted(set(low[image]['unmatched_gold_indices']) - set(high[image]['unmatched_gold_indices']))])
            for image in gold if high[image]['unmatched_gold_indices'] != low[image]['unmatched_gold_indices']}
    for case in mechanism_rows:
        image, index = case['image'], case['r38_row'] - 1
        low_score = internal_scores['r38']['hard005'][image]
        match = next((m for m in low_score['matches'] if m['gold_index'] == index), None)
        case['recovered_by_probe'] = match is not None
        if match:
            j = match['prediction_index']
            prediction = outputs[image]['hard005'][j]
            case['probe_match'] = dict(anchor_index=outputs[image]['postclip_indices005'][j],
                prediction_number=j + 1, iou=match['iou'], prediction=prediction,
                corners=corner_details(case['target'], prediction))
    helpers = ('pose_scoring.py', 'predict_pose_folder.py', 'evaluate_pose.py',
               'audit_additional_front_pilot_r45_results.py', 'audit_additional_front_pilot_r47_results.py',
               'export_front_error_review_r48.py')
    summary = dict(schema='r51-one-confidence-probe-v1', archive_sha256=ARCHIVE_SHA,
        source_pr=143, post_hoc_diagnostic=True, same_decoded_tensors=True,
        fresh_inference=False, fresh_training=False, automatic_promotion=False,
        settings=dict(confidences=[HIGH, LOW], nms_iou=NMS, matching_iou=MATCH, endpoint_confidence=.5,
                      max_det=CAP, original_image_size=[1280, 720], input_shape=[1, 3, 736, 1280],
                      coordinate_mapping='unit gain, x padding0, y padding8; native clipping', same_postclip=True),
        source_files_sha256={name: sha((root / 'scripts' / name).read_bytes()) for name in helpers},
        script_sha256=sha(Path(__file__).read_bytes()), runtime=dict(python=platform.python_version(), numpy=np.__version__),
        references=reference_contract, baseline_reconstruction=validation, reports=reports,
        interpretation='One diagnostic replay; no threshold sweep or new operating default. Reference-derived proposal selection is explanatory only.')
    mechanisms = dict(schema='r51-pr143-miss-mechanisms-v1', archive_sha256=ARCHIVE_SHA,
        reference='R38; same target geometry in R50 is recorded per case',
        units='zero-based decoded anchor indices; one-based R38 rows and displayed prediction numbers',
        count=len(mechanism_rows), recorded_stage_counts=dict(Counter(c['recorded_stage'] for c in mechanism_rows)),
        cases=mechanism_rows)
    return summary, mechanisms, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-predictions', action='store_true')
    args = parser.parse_args()
    summary, mechanisms, predictions = analyze(args.archive, args.root)
    put(args.output / 'summary.json', encoded(summary))
    put(args.output / 'mechanisms.json', encoded(mechanisms))
    if args.include_predictions:
        put(args.output / 'predictions.json', encoded(predictions))
    for reference, reports in summary['reports'].items():
        for arm in ('hard013', 'hard005'):
            score = reports[arm]['detection']
            print(reference, arm, {key: score[key] for key in ('tp', 'fn', 'fp', 'duplicate_fp', 'other_fp')})
    print('Native 0.13 reconstruction exact on all', len(summary['baseline_reconstruction']), 'images')


if __name__ == '__main__':
    main()
