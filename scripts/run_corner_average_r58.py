#!/usr/bin/env python3
"""Average paired original/reflected corners while preserving original detections."""
import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import gzip
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import platform
import re
import subprocess
import sys
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

from audit_reflection_return import (
    ARCHIVE_SHA256, REFERENCES, COUNT_KEYS, digest, encoded, require,
    verify_archive, load_references, verify_geometry, valid_point)

PAIR_THRESHOLD = .5
ASSOCIATION_IOU = .5
ARCHIVE_NAME = 'baomai-corner-average-r58-results.zip'
RETURN_COMMIT = 'dfe5392109433fcee51f41716710001bb123a3d5'
RETURN_PATH = 'results/a100/baomai-flip-inference-r56-a100-001/baomai-flip-inference-r56-results.zip'


def overlap(a, b):
    if a['class_id'] != b['class_id']:
        return 0.
    x0, y0, x1, y1 = a['bbox_xyxy']
    u0, v0, u1, v1 = b['bbox_xyxy']
    shared = max(0., min(x1, u1) - max(x0, u0)) * max(0., min(y1, v1) - max(y0, v0))
    union = (x1 - x0) * (y1 - y0) + (u1 - u0) * (v1 - v0) - shared
    return shared / union if union > 0 else 0.


def average_predictions(original, reflected):
    output = deepcopy(original)
    matrix = [[overlap(a, b) for b in reflected] for a in original]
    forward = {i: max(range(len(reflected)), key=lambda j: (row[j], -j))
               for i, row in enumerate(matrix) if reflected}
    backward = {j: max(range(len(original)), key=lambda i: (matrix[i][j], -i))
                for j in range(len(reflected)) if original}
    rows = []
    for i, source in enumerate(original):
        row = {'original_index': i, 'original_prediction_id': source.get('primary_prediction_id'),
               'status': 'no_mutual_match', 'reflected_index': None}
        j = forward.get(i)
        if j is not None and backward[j] == i and matrix[i][j] >= ASSOCIATION_IOU:
            row.update(reflected_index=j, box_iou=matrix[i][j],
                       original_eligible_partners=sum(x >= ASSOCIATION_IOU for x in matrix[i]),
                       reflected_eligible_partners=sum(matrix[k][j] >= ASSOCIATION_IOU for k in range(len(original))))
            row['status'] = 'original_pair_unavailable' if not all(valid_point(k) for k in source['keypoints']) else \
                'reflected_pair_unavailable' if not all(valid_point(k) for k in reflected[j]['keypoints']) else 'averaged'
            if row['status'] == 'averaged':
                for point in range(2):
                    for axis in range(2):
                        output[i]['keypoints'][point][axis] = (
                            source['keypoints'][point][axis] + reflected[j]['keypoints'][point][axis]) / 2
        rows.append(row)
    require(len(original) == len(output), 'Output record count changed')
    for before, after in zip(original, output):
        require({k: v for k, v in before.items() if k != 'keypoints'} ==
                {k: v for k, v in after.items() if k != 'keypoints'}, 'Detection record changed')
        require([p[2] for p in before['keypoints']] == [p[2] for p in after['keypoints']], 'Point scores changed')
        for threshold in (.5, .8):
            require(all(valid_point(p) and p[2] >= threshold for p in before['keypoints']) ==
                    all(valid_point(p) and p[2] >= threshold for p in after['keypoints']), 'Accepted output identity changed')
    return output, rows


def clipped_prediction_indices(predictions):
    """Return indexes of finite boxes with no positive area. Other records stay untouched."""
    if isinstance(predictions, (str, bytes)) or not isinstance(predictions, Sequence):
        return []
    invalid = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, Mapping):
            continue
        box = prediction.get('bbox_xyxy')
        if isinstance(box, (str, bytes)) or not isinstance(box, Sequence) or len(box) != 4:
            continue
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in box):
            continue
        coords = [float(value) for value in box]
        if not all(math.isfinite(value) for value in coords):
            continue
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            invalid.append(index)
    return invalid


def score_image_counting_clipped(gold, predictions, width, height, matching_iou, endpoint_conf):
    """Count clipped non-positive predictions as other unmatched outputs.

    pose_scoring.score_image still rejects a non-positive box. Positive-area
    records are scored unchanged and their indexes are mapped back. Each clipped
    box stays in the prediction and false-output totals, keeps its original
    index, and cannot match.
    """
    from pose_scoring import score_image
    invalid = clipped_prediction_indices(predictions)
    if not invalid:
        return score_image(gold, predictions, width, height, matching_iou, endpoint_conf)
    skipped = set(invalid)
    valid = [index for index in range(len(predictions)) if index not in skipped]
    score = score_image(gold, [predictions[index] for index in valid], width, height, matching_iou, endpoint_conf)
    for match in score['matches']:
        match['prediction_index'] = valid[match['prediction_index']]
    for key in ('unmatched_prediction_indices', 'duplicate_prediction_indices', 'other_fp_prediction_indices'):
        score[key] = [valid[index] for index in score[key]]
    for key in ('unmatched_prediction_indices', 'other_fp_prediction_indices'):
        score[key] = sorted([*score[key], *invalid])
    score['prediction_count'] += len(invalid)
    score['fp'] += len(invalid)
    score['other_fp'] += len(invalid)
    return score


def state(target, prediction):
    complete = all(p[2] == 2 for p in target['keypoints'])
    usable = prediction is not None and all(valid_point(p) for p in prediction['keypoints'])
    error = max(math.hypot(p[0] - g[0], p[1] - g[1]) for p, g in zip(prediction['keypoints'], target['keypoints'])) \
        if complete and usable else None
    return {'complete_reference': complete, 'detected': prediction is not None,
            'maximum_corner_error_px': error, 'within5': error is not None and error <= 5,
            'within10': error is not None and error <= 10}


def report(gold, predictions):
    from pose_scoring import aggregate_scores
    scores, rows = {}, []
    for image, targets in sorted(gold.items()):
        values = predictions[image]
        result = score_image_counting_clipped(targets, values, 1280, 720, .5, .5)
        scores[image] = result
        matches = {m['gold_index']: m['prediction_index'] for m in result['matches']}
        for gi, target in enumerate(targets):
            pi = matches.get(gi)
            rows.append({'image': image, 'reference_index': gi, 'prediction_index': pi,
                         **state(target, values[pi] if pi is not None else None)})
    eligible = sum(r['complete_reference'] for r in rows)
    complete = [r for r in rows if r['complete_reference']]
    c = {'eligible': eligible, 'missed_box': sum(not r['detected'] for r in complete),
         'returned_pairs': sum(r['maximum_corner_error_px'] is not None for r in complete),
         'matched_pair_unavailable': sum(r['detected'] and r['maximum_corner_error_px'] is None for r in complete),
         'within5': sum(r['within5'] for r in complete), 'within10': sum(r['within10'] for r in complete)}
    c['over5'] = c['returned_pairs'] - c['within5']
    require(c['missed_box'] + c['matched_pair_unavailable'] + c['returned_pairs'] == eligible, 'Pair denominator differs')
    detection = aggregate_scores(list(scores.values()))
    return {'detection': detection, 'pairs': c}, scores, rows


def accepted_report(gold, predictions, threshold):
    from pose_scoring import score_image
    totals = Counter({k: 0 for k in ('accepted', 'confirmed5', 'confirmed10', 'measured_over5',
                                     'matched_incomplete', 'unmatched', 'duplicates')})
    for image, targets in gold.items():
        accepted = [p for p in predictions[image] if all(valid_point(k) and k[2] >= threshold for k in p['keypoints'])]
        score = score_image(targets, accepted, 1280, 720, .5, threshold)
        totals.update(accepted=len(accepted), unmatched=score['fp'], duplicates=score['duplicate_fp'])
        for match in score['matches']:
            s = state(targets[match['gold_index']], accepted[match['prediction_index']])
            if not s['complete_reference']:
                totals['matched_incomplete'] += 1
            else:
                totals['confirmed5'] += s['within5']
                totals['confirmed10'] += s['within10']
                totals['measured_over5'] += not s['within5']
    require(totals['accepted'] == sum(totals[k] for k in ('confirmed5', 'measured_over5', 'matched_incomplete', 'unmatched')),
            'Accepted pair accounting differs')
    return {'corner_threshold': threshold, **totals}


def analyze(root, archive):
    data = verify_archive(archive)
    refs = load_references(root, data['summary']['contract'])
    verify_geometry(data['predictions'], data['native-predictions'])
    require(all(r['passed'] and r['exact_json_equal'] for r in data['repeatability']), 'Source repeats did not pass')
    original = {i: r['postclip'] for i, r in data['predictions']['control'].items()}
    reflected = {i: r['postclip'] for i, r in data['predictions']['treatment'].items()}
    averaged, associations = {}, {}
    for image in original:
        averaged[image], associations[image] = average_predictions(original[image], reflected[image])
    tallies = Counter(r['status'] for rows in associations.values() for r in rows)
    summary = {'schema': 'cassette-corner-average-r58-v1', 'status': 'completed',
        'experiment': 'fixed-original-boxes_equal-original-reflection-corners',
        'fresh_inference': False, 'training': False, 'automatic_promotion': False,
        'protocol': {'association': 'same-class mutual-best predicted-box IoU, lower partner record index in each direction for ties',
                     'association_iou_inclusive': ASSOCIATION_IOU, 'corner_score_inclusive': PAIR_THRESHOLD,
                     'coordinate_weight_original': .5, 'coordinate_weight_reflection': .5,
                     'fallback': 'unchanged original record', 'preserved': ['boxes', 'box scores', 'point scores', 'class', 'record order', 'record identifiers'],
                     'scores_interpretation': 'Original scores define a fixed comparison cohort; they are not calibrated confidence in averaged coordinates.',
                     'image_size_px': [1280, 720]},
        'association_counts': {'original_records': sum(map(len, original.values())), 'reflected_records': sum(map(len, reflected.values())),
             **{k: tallies[k] for k in ('averaged','no_mutual_match','original_pair_unavailable','reflected_pair_unavailable')},
             'averaged_with_multiple_eligible_partners': sum(r['status'] == 'averaged' and
                (r['original_eligible_partners'] > 1 or r['reflected_eligible_partners'] > 1)
                for rows in associations.values() for r in rows)},
        'source': {'return_commit': RETURN_COMMIT, 'return_path': RETURN_PATH, 'return_sha256': ARCHIVE_SHA256,
                   'checkpoint_sha256': data['summary']['contract']['checkpoint_sha256']},
        'references': {}, 'interpretation': 'Exploratory development analysis. No reference coordinates select links or averaging weights.'}
    cases = {}
    for reference, gold in refs.items():
        arm_reports, arm_scores, arm_rows = {}, {}, {}
        for arm, values in (('control', original), ('reflection', reflected), ('average', averaged)):
            arm_reports[arm], arm_scores[arm], arm_rows[arm] = report(gold, values)
            arm_reports[arm]['accepted_pairs'] = {str(q): accepted_report(gold, values, q) for q in (.5, .8)}
            if arm != 'average':
                saved = data['summary']['reports']['control' if arm == 'control' else 'treatment'][reference]['postclip']
                require(arm_scores[arm] == saved['per_image'] and arm_reports[arm]['detection'] == saved['detection'],
                        'Saved scoring did not reproduce: ' + reference + '/' + arm)
        for image in gold:
            before, after = arm_scores['control'][image], arm_scores['average'][image]
            require(all(before[k] == after[k] for k in COUNT_KEYS), 'Detection counts changed')
            require([(m['gold_index'],m['prediction_index']) for m in before['matches']] ==
                    [(m['gold_index'],m['prediction_index']) for m in after['matches']], 'Box assignment changed')
        comparisons = []
        for before, after in zip(arm_rows['control'], arm_rows['average']):
            require((before['image'], before['reference_index'], before['prediction_index']) ==
                    (after['image'], after['reference_index'], after['prediction_index']), 'Target identity changed')
            comparisons.append({'image': before['image'], 'reference_index': before['reference_index'],
                                'prediction_index': before['prediction_index'], 'control': before, 'average': after})
        changes = {}
        for t in (5, 10):
            gained = [r for r in comparisons if r['control']['complete_reference'] and not r['control']['within'+str(t)] and r['average']['within'+str(t)]]
            lost = [r for r in comparisons if r['control']['complete_reference'] and r['control']['within'+str(t)] and not r['average']['within'+str(t)]]
            changes[str(t)] = {'gained': len(gained), 'lost': len(lost),
                              'gained_ids': [r['image'] + '/R' + str(r['reference_index']+1).zfill(3) for r in gained],
                              'lost_ids': [r['image'] + '/R' + str(r['reference_index']+1).zfill(3) for r in lost]}
        summary['references'][reference] = {'arms': arm_reports, 'paired_changes': changes,
                                            'all_detection_records_and_assignments_preserved': True}
        cases[reference] = comparisons
    return summary, {'averaged-predictions.json': averaged, 'associations.json': associations, 'case-changes.json': cases}


def source_identity(root, source_commit, script_sha):
    root = root.resolve()
    require(root == Path(__file__).resolve().parents[1], 'Driver and source root differ')
    require(re.fullmatch('[0-9a-f]{40}', source_commit) is not None, 'Use an immutable source commit')
    require(re.fullmatch('[0-9a-f]{64}', script_sha) is not None, 'Use a complete driver hash')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    require(git('rev-parse', 'HEAD') == source_commit, 'Source commit differs')
    require(not git('status', '--porcelain', '--untracked-files=all'), 'Source checkout changed')
    require(digest(Path(__file__).read_bytes()) == script_sha, 'Driver hash differs')
    paths = ['scripts/run_corner_average_r58.py', 'scripts/audit_reflection_return.py',
             'scripts/pose_scoring.py', 'scripts/evaluate_pose.py', 'scripts/pose_checkpoint.py', 'scripts/predict_pose_folder.py']
    return {'commit': source_commit, 'files_sha256': {p: digest((root/p).read_bytes()) for p in paths},
            'python': platform.python_version()}


def bundle(summary, records, source):
    payload = {'summary.json': encoded(summary), 'source.json': encoded(source)}
    for name, values in records.items():
        payload[name+'.gz'] = gzip.compress(encoded(values), compresslevel=9, mtime=0)
    payload['inventory.json'] = encoded({k: {'bytes': len(v), 'sha256': digest(v)} for k,v in payload.items()})
    return payload


def verify_result(path, expected):
    with ZipFile(path) as z:
        require(len(z.namelist()) == len(set(z.namelist())) and set(z.namelist()) == set(expected), 'Result membership differs')
        require(z.testzip() is None, 'Result CRC failed')
        actual = {name: z.read(name) for name in z.namelist()}
    inventory = json.loads(actual['inventory.json'])
    require(set(inventory) == set(actual) - {'inventory.json'}, 'Result inventory membership differs')
    for name in inventory:
        require(inventory[name] == {'bytes': len(actual[name]), 'sha256': digest(actual[name])},
                'Result inventory differs: ' + name)
        observed = gzip.decompress(actual[name]) if name.endswith('.gz') else actual[name]
        recomputed = gzip.decompress(expected[name]) if name.endswith('.gz') else expected[name]
        require(json.loads(observed) == json.loads(recomputed), 'Recomputed result differs: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run','verify'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--script-sha256', required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    source = source_identity(args.root, args.source_commit, args.script_sha256)
    summary, records = analyze(args.root, args.archive)
    target = args.work / ARCHIVE_NAME
    if args.action == 'verify':
        with ZipFile(target) as archive:
            recorded_source = json.loads(archive.read('source.json'))
        require(set(recorded_source) == set(source), 'Source record fields differ')
        require(recorded_source['commit'] == source['commit'] and
                recorded_source['files_sha256'] == source['files_sha256'], 'Result source differs')
        require(isinstance(recorded_source['python'], str), 'Original Python version is missing')
        source = recorded_source
    payload = bundle(summary, records, source)
    if args.action == 'run':
        args.work.mkdir(parents=True, exist_ok=True)
        for name, value in {'summary.json': payload['summary.json'], 'source.json': payload['source.json']}.items():
            output = args.work/name
            require(not output.exists() or output.read_bytes() == value, 'Different existing work preserved: ' + str(output))
        if target.exists():
            verify_result(target, payload)
        else:
            with ZipFile(target, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
                for name, value in payload.items():
                    info = ZipInfo(name, date_time=(1980,1,1,0,0,0));info.compress_type=ZIP_DEFLATED
                    archive.writestr(info, value)
        for name in ('summary.json','source.json'):
            (args.work/name).write_bytes(payload[name])
    verify_result(target, payload)
    print(json.dumps({'status':'verified','archive':str(target),'sha256':digest(target.read_bytes()),
                      'association_counts':summary['association_counts']}))
    for reference in ('frozen666','r50'):
        for arm in ('control','reflection','average'):
            r=summary['references'][reference]['arms'][arm]
            print(reference,arm,{k:r['detection'][k] for k in ('fn','fp','duplicate_fp')},r['pairs'])


if __name__ == '__main__':
    main()
