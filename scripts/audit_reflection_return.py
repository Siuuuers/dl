#!/usr/bin/env python3
"""Replay the returned horizontal-reflection experiment using saved predictions."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from zipfile import ZipFile

ARCHIVE_SHA256 = '88689ce9f3d03178683ea98ce98d50f5ddc4e07ba1fb24262e7820e9488bd114'
SOURCE_COMMIT = 'dc7eec630bb0b613fc1fe784e4d9818a0893658e'
CHECKPOINT_SHA256 = '222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9'
DRIVER_SHA256 = 'bf2ff9056f12976ca8c4aa971de2d2a7af1c679369c44dbf73de9f9d2542700d'
HELPER_SHA256 = '8c8a37c3dcaccde5c568e9df6f44651cdb05965ca5f5ed5b0f2978852387dcc1'
REFERENCES = ('frozen666', 'historical663', 'r38', 'r50')
COUNT_KEYS = ('gold_count', 'prediction_count', 'tp', 'fn', 'fp', 'duplicate_fp', 'other_fp')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def verify_archive(path):
    payload = Path(path).read_bytes()
    require(len(payload) == 2931102 and digest(payload) == ARCHIVE_SHA256, 'Returned archive bytes differ')
    with ZipFile(path) as archive:
        expected = {'baseline-equivalence.json', 'cases.json', 'complementarity.json', 'input-audit.json',
                    'inventory.json', 'native-predictions.json', 'predictions.json', 'preflight.json',
                    'repeatability.json', 'repeated-native-predictions.json', 'repeated-predictions.json',
                    'run_flip_inference_r56.py', 'run_inference_ablation_r53.py', 'runtime.json',
                    'summary.json', 'timing.json'}
        names = archive.namelist()
        require(len(names) == len(set(names)) and set(names) == expected and archive.testzip() is None,
                'Archive membership or CRC differs')
        members = {name: archive.read(name) for name in names}
    inventory = json.loads(members['inventory.json'])
    require(set(inventory) == expected - {'inventory.json'}, 'Inventory membership differs')
    for name, value in inventory.items():
        require(value == {'bytes': len(members[name]), 'sha256': digest(members[name])}, 'Inventory differs: ' + name)
    require(digest(members['run_flip_inference_r56.py']) == DRIVER_SHA256, 'Driver differs')
    require(digest(members['run_inference_ablation_r53.py']) == HELPER_SHA256, 'Inference helper differs')
    decoded = {name[:-5]: json.loads(data) for name, data in members.items() if name.endswith('.json')}
    contract = decoded['summary']['contract']
    require(contract['source'] == SOURCE_COMMIT and contract['checkpoint_sha256'] == CHECKPOINT_SHA256,
            'Source or model identity differs')
    require(contract['script_sha256'] == DRIVER_SHA256 and contract['r53_driver_sha256'] == HELPER_SHA256,
            'Declared driver identity differs')
    require(contract == decoded['preflight']['contract'] and
            digest(encoded(contract)) == decoded['preflight']['contract_sha256'], 'Preflight contract differs')
    require(decoded['input-audit'] == decoded['preflight']['input_audit'], 'Input audit differs')
    settings = contract['settings']
    require(settings['control'] == {'geometry': 'identity', 'imgsz': 1280} and
            settings['treatment'] == {'geometry': 'horizontal_flip', 'imgsz': 1280}, 'Comparison differs')
    common = settings['common']
    require(all(common[k] == v for k, v in {'conf': .13, 'iou': .70, 'matching_iou': .50,
            'endpoint_conf': .50, 'max_det': 300, 'augment': False}.items()), 'Scoring settings differ')
    require(decoded['summary']['status'] == 'completed' and decoded['summary']['interpretation_allowed'],
            'Returned run is not interpretable')
    return decoded


def load_references(root, contract):
    root = Path(root).resolve()
    sys.path.insert(0, str(root / 'scripts'))
    for name in ('pose_scoring.py', 'evaluate_pose.py', 'pose_checkpoint.py', 'predict_pose_folder.py'):
        relative = 'scripts/' + name
        require(digest((root / relative).read_bytes()) == contract['source_files_sha256'][relative],
                'Required implementation differs: ' + relative)
    from evaluate_pose import parse_pose_labels
    result = {}
    for reference in REFERENCES:
        descriptor = contract['references'][reference]
        require(digest((root / descriptor['manifest_path']).read_bytes()) == descriptor['manifest_sha256'],
                'Reference manifest differs: ' + reference)
        images = {}
        for identity, record in descriptor['labels'].items():
            path = (root / record['path']).resolve()
            require(path.is_relative_to(root), 'Reference path escapes repository')
            data = path.read_bytes()
            require(digest(data) == record['sha256'], 'Reference label differs: ' + identity)
            images[identity] = parse_pose_labels(data.decode(), 1280, 720, reference + '/' + identity)
        require(sum(map(len, images.values())) == descriptor['objects'], 'Reference count differs')
        require(set(images) == set(contract['images']), 'Image membership differs')
        result[reference] = images
    return result


def independent_inverse(prediction):
    result = deepcopy(prediction)
    x0, y0, x1, y1 = prediction['bbox_xyxy']
    result['bbox_xyxy'] = [1280 - x1, y0, 1280 - x0, y1]
    result['keypoints'] = [[None if point[0] is None else 1280 - point[0], point[1], point[2]]
                           for point in reversed(prediction['keypoints'])]
    return result


def verify_geometry(outputs, native):
    from predict_pose_folder import postclip_nms
    require(set(outputs) == set(native) == {'control', 'treatment'}, 'Arm membership differs')
    records = 0
    for arm in ('control', 'treatment'):
        require(set(outputs[arm]) == set(native[arm]), 'Native image membership differs')
        for identity, row in outputs[arm].items():
            restored = deepcopy(native[arm][identity]) if arm == 'control' else [
                independent_inverse(p) for p in native[arm][identity]]
            require(restored == row['raw'], 'Independent inverse mapping differs: ' + arm + '/' + identity)
            kept, audit = postclip_nms(restored)
            require(kept == row['postclip'] and audit == row['filter'], 'Suppression replay differs')
            records += len(restored)
    return records


def valid_point(point):
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in point) and \
        0 <= point[0] < 1280 and 0 <= point[1] < 720 and point[2] >= .5


def pair_errors(target, prediction):
    if prediction is None or not all(p[2] == 2 for p in target['keypoints']) or \
            not all(valid_point(p) for p in prediction['keypoints']):
        return None
    return [math.hypot(p[0] - g[0], p[1] - g[1])
            for p, g in zip(prediction['keypoints'], target['keypoints'])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = verify_archive(args.archive)
    refs = load_references(args.root, data['summary']['contract'])
    from pose_scoring import score_image, aggregate_scores, _box_iou
    geometry_records = verify_geometry(data['predictions'], data['native-predictions'])
    repeats = data['repeated-predictions']
    require(len(repeats) == len(data['repeated-native-predictions']) == 3, 'Repeat count differs')
    for output, native in zip(repeats, data['repeated-native-predictions']):
        geometry_records += verify_geometry(output, native)
        require(output == data['predictions'] and native == data['native-predictions'], 'Repeated bytes differ')
    require(all(r['passed'] and r['exact_json_equal'] for r in data['repeatability']), 'Repeatability differs')
    compact, per_target = {}, {}
    for reference, gold_images in refs.items():
        compact[reference], per_target[reference] = {}, {}
        for view in ('raw', 'postclip'):
            compact[reference][view] = {}; per_target[reference][view] = []
            arm_rows = {}
            for arm in ('control', 'treatment'):
                scores = []; counts = Counter(eligible=0, matched=0, within5=0, within10=0, returned_pairs=0)
                arm_rows[arm] = {}
                for identity, targets in sorted(gold_images.items()):
                    predictions = data['predictions'][arm][identity][view]
                    score = score_image(targets, predictions, 1280, 720, .5, .5)
                    require(score == data['summary']['reports'][arm][reference][view]['per_image'][identity],
                            'Independent per-image score differs')
                    scores.append(score)
                    matches = {m['gold_index']: m for m in score['matches']}
                    for index, target in enumerate(targets):
                        match = matches.get(index)
                        prediction = None if match is None else predictions[match['prediction_index']]
                        errors = pair_errors(target, prediction)
                        complete = all(p[2] == 2 for p in target['keypoints'])
                        if complete:
                            counts['eligible'] += 1
                            counts['matched'] += match is not None
                            counts['returned_pairs'] += errors is not None
                            counts['within5'] += errors is not None and max(errors) <= 5
                            counts['within10'] += errors is not None and max(errors) <= 10
                        case = reference + '/' + identity + '/R' + str(index + 1).zfill(3)
                        arm_rows[arm][case] = {'case_id': case, 'image_id': identity,
                            'locatable_count': sum(p[2] == 2 for p in target['keypoints']),
                            'matched': match is not None,
                            'prediction_index': None if match is None else match['prediction_index'],
                            'box_iou': None if match is None else match['iou'], 'errors_px': errors,
                            'minimum_corner_score': None if prediction is None else min(p[2] for p in prediction['keypoints']),
                            'within5': errors is not None and max(errors) <= 5,
                            'within10': errors is not None and max(errors) <= 10}
                aggregate = aggregate_scores(scores)
                require(aggregate == data['summary']['reports'][arm][reference][view]['detection'],
                        'Independent aggregate differs')
                expected = data['summary']['reports'][arm][reference][view]['corners_by_locatable_count']['2']
                require(all(counts[key] == value for key, value in expected.items()), 'Independent pair counts differ')
                compact[reference][view][arm] = {
                    'detection': {key: aggregate[key] for key in COUNT_KEYS},
                    'complete_pairs': dict(counts),
                    'endpoint_error_summary': {key: aggregate[key] for key in
                        ('returned_endpoints', 'visible_endpoints', 'conditional_median_error_px', 'conditional_p95_error_px')}}
            changes = Counter(recovered=0, newly_missed=0, matched_by_both=0, missed_by_both=0,
                              complete_within5_gains=0, complete_within5_losses=0,
                              complete_within10_gains=0, complete_within10_losses=0)
            for case, control in arm_rows['control'].items():
                flip = arm_rows['treatment'][case]
                row = {'case_id': case, 'image_id': control['image_id'], 'locatable_count': control['locatable_count'],
                       'control': control, 'reflection': flip,
                       'recovered': not control['matched'] and flip['matched'],
                       'newly_missed': control['matched'] and not flip['matched'],
                       'within5_gain': not control['within5'] and flip['within5'],
                       'within5_loss': control['within5'] and not flip['within5'],
                       'within10_gain': not control['within10'] and flip['within10'],
                       'within10_loss': control['within10'] and not flip['within10']}
                changes['recovered'] += row['recovered']; changes['newly_missed'] += row['newly_missed']
                changes['matched_by_both'] += control['matched'] and flip['matched']
                changes['missed_by_both'] += not control['matched'] and not flip['matched']
                for tolerance in (5, 10):
                    for kind in ('gain', 'loss'):
                        changes[f'complete_within{tolerance}_{kind}s' if kind == 'gain' else f'complete_within{tolerance}_losses'] += row[f'within{tolerance}_{kind}']
                per_target[reference][view].append(row)
            compact[reference][view]['changes'] = dict(changes)
            if view == 'postclip':
                original = {r['case_id']: r for r in data['cases']['paired_rows'] if r['reference'] == reference}
                for row in per_target[reference][view]:
                    require(all(row[k] == original[row['case_id']][k] for k in ('recovered', 'newly_missed')),
                            'Detection changes differ')
                    if row['locatable_count'] == 2:
                        require(all(row[k] == original[row['case_id']][k] for k in
                                    ('within5_gain', 'within5_loss', 'within10_gain', 'within10_loss')),
                                'Complete pair changes differ')
    image = 'new_000057'; target = refs['frozen666'][image][11]; boundary_case = {'case_id': 'frozen666/new_000057/R012'}
    for arm in ('control', 'treatment'):
        predictions = data['predictions'][arm][image]['postclip']
        index = max(range(len(predictions)), key=lambda i: _box_iou(predictions[i]['bbox_xyxy'], target['bbox_xyxy']))
        prediction = predictions[index]
        boundary_case[arm] = {'prediction_index': index, 'box_iou': _box_iou(prediction['bbox_xyxy'], target['bbox_xyxy']),
                              'box_confidence': prediction['confidence'], 'corner_errors_px': pair_errors(target, prediction)}
    result = {'schema': 'reflection-return-independent-audit-v1', 'status': 'passed',
              'archive_sha256': ARCHIVE_SHA256, 'source_commit': SOURCE_COMMIT,
              'checkpoint_sha256': CHECKPOINT_SHA256,
              'method': 'Original scorer replay plus independent complete-pair accounting and inverse mapping; no model inference.',
              'verified_prediction_records_including_repeats': geometry_records,
              'scoring_tables_checked': 16, 'per_image_score_tables_checked': 320,
              'repeat_count_per_arm': 3, 'repeats_exact': True,
              'baseline_equivalence': data['baseline-equivalence'], 'results': compact,
              'box_matching_boundary_example': boundary_case,
              'runtime': {k: data['runtime'][k] for k in ('gpu', 'python', 'torch', 'cuda', 'backend_fp16', 'tf32')},
              'timing': data['timing']['summary'],
              'empty_images': {identity: {arm: len(rows[identity]['postclip'])
                  for arm, rows in data['predictions'].items()} for identity, targets in refs['r50'].items() if not targets},
              'script_sha256': digest(Path(__file__).read_bytes())}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'audit.json').write_bytes(encoded(result))
    (args.output / 'case-changes.json.gz').write_bytes(gzip.compress(encoded(per_target), mtime=0))
    print(json.dumps({'status': 'passed', 'tables_checked': 16,
        'fixed': compact['frozen666']['postclip'], 'reviewed': compact['r50']['postclip']}, indent=2))


if __name__ == '__main__':
    main()
