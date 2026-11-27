#!/usr/bin/env python3
"""R53: two prespecified, same-checkpoint inference comparisons; never train.

The unchanged control must reproduce the PR137 cache before a treatment runs.
All scoring uses original image coordinates and the four existing references.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata as md
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import statistics
import sys
import time
import zipfile

sys.dont_write_bytecode = True
from diagnose_fronts_r50 import (CONF, MATCH, R50_SHA, checked, encoded, git, need,
                                 prepare as prepare_inputs, put, sha,
                                 source_check as inherited_source_check)

R50_DRIVER_SHA = 'cba4f1df6dcf431c933af1506a4de15718636d9477406eb34779ac4d8ab6ee3e'
PARENT_SOURCE = '9983d996e087aa6fd348a1861b61a1c99ec88a0d'
PINNED_PACKAGES = {'ultralytics': '8.4.182', 'opencv-python': '5.0.0.93',
                   'numpy': '2.4.6', 'Pillow': '13.0.0'}
ULTRALYTICS_WHEEL = {'name': 'ultralytics-8.4.182-py3-none-any.whl',
    'sha256': '11b90b26fa30f878807b0b8c4ed1870eb3c54e7e876a17a936bd92065a7e743b'}
INFERENCE_SOURCE_HASHES = {
    'nn/modules/head.py': 'b471bb8a39cc1e38672c20feaa8c28bdf43739adb8eafd14226807b34b31f3ee',
    'nn/tasks.py': '1048ab8db7504e7c57a77d05bfbd855e9de0ef477e15a5aae8cdb7e8a958246a',
    'engine/predictor.py': '9b29a5322293167a705a6d6054ef653a62d27f6e4bff589b242946df89b5b62f',
    'models/yolo/pose/predict.py': '8f0ce1afb26df2d30c38c5e7a550377cfa23c334cb0effba6927672779335c3e',
    'utils/nms.py': '386cd0e3e38fddef5d434fbcfd58264bedfa37295534e9a2b7174a3436d4db62',
    'utils/ops.py': '499672757c9463187ce2b354585a9ada28325e57178f4bc3307d9548bd552089',
}
TOLERANCES = {'coordinate_component_px': .05, 'probability': .00001,
              'endpoint_error_px': .075}
TIMING_REPEATS = 3
COUNT_KEYS = ('gold_count', 'prediction_count', 'tp', 'fn', 'fp', 'duplicate_fp',
              'other_fp', 'visible_endpoints', 'returned_endpoints', 'missing_endpoints')
INDEX_KEYS = ('unmatched_gold_indices', 'unmatched_prediction_indices',
              'duplicate_prediction_indices', 'other_fp_prediction_indices')
EXPERIMENTS = {
    'clahe': {'control': {'photometry': 'identity', 'imgsz': 1280},
              'treatment': {'photometry': 'lab_l_clahe', 'imgsz': 1280},
              'hypothesis': 'Fixed local luminance contrast may recover dim partial fronts; it may also amplify texture and false outputs.'},
    'scale': {'control': {'photometry': 'identity', 'imgsz': 1280},
              'treatment': {'photometry': 'identity', 'imgsz': 1600},
              'hypothesis': 'A larger network input may change feature sampling for small fronts; interpolation adds no new captured detail.'},
}


def preprocess_bgr(image, method):
    """Return a new, same-geometry uint8 BGR array; never modify source pixels."""
    import numpy as np
    need(isinstance(image, np.ndarray) and image.dtype == np.uint8
         and image.ndim == 3 and image.shape[2] == 3
         and min(image.shape[:2]) > 0, 'Expected nonempty uint8 H x W x 3 BGR image')
    if method == 'identity':
        return np.ascontiguousarray(image.copy())
    need(method == 'lab_l_clahe', 'Unknown fixed photometric arm')
    import cv2
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    output = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    need(output.shape == image.shape and output.dtype == np.uint8, 'Photometric geometry changed')
    return np.ascontiguousarray(output)


def settings(experiment):
    from evaluate_pose import INFERENCE
    return {'experiment': experiment, **EXPERIMENTS[experiment],
        'common': {**INFERENCE, 'conf': CONF, 'matching_iou': MATCH, 'endpoint_conf': .5},
        'clahe': {'color_space': 'OpenCV uint8 BGR to Lab; L only; back to BGR',
                  'clipLimit': 2.0, 'tileGridSize': [8, 8],
                  'training_adaptation': False, 'geometry_changes': False},
        'baseline_tolerances': TOLERANCES, 'timing_repeats': TIMING_REPEATS,
        'training': False, 'automatic_promotion': False}


def source_check(args):
    root = args.root.resolve()
    need(re.fullmatch('[0-9a-f]{64}', args.script_sha256) is not None, 'Full driver SHA256 required')
    checked(root / 'scripts/run_inference_ablation_r53.py', args.script_sha256)
    need(Path(__file__).resolve() == root / 'scripts/run_inference_ablation_r53.py',
         'Run the R53 driver from its pinned source checkout')
    context = inherited_source_check(root, args.source_commit, R50_DRIVER_SHA, R50_SHA)
    inherited = context['contract']
    need(inherited['source_files_sha256'].get('scripts/run_inference_ablation_r53.py') == args.script_sha256,
         'R53 driver is not tracked by the pinned Git tree')
    metadata = {}
    for relative in ('metadata/model_card.json', 'models/cassette_lower_yolo26s_pose.yaml'):
        data = (root / relative).read_bytes()
        blob = git(root, 'rev-parse', 'HEAD:' + relative)
        need(hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == blob,
             'Model metadata differs from pinned Git source')
        inherited['source_files_sha256'][relative] = sha(data)
        metadata[relative] = sha(data)
    card = json.loads((root / 'metadata/model_card.json').read_text())
    need(card['model'] == 'YOLO26s-Pose' and card['kpt_shape'] == [2, 3]
         and card['checkpoint_dependency'] == 'ultralytics 8.4.182', 'Model card contract differs')
    context['contract'] = {
        'schema': 'baomai-r53-inference-inputs-v1', 'source': args.source_commit,
        'parent_pr': 147, 'parent_source': PARENT_SOURCE, 'script_sha256': args.script_sha256,
        'source_files_sha256': inherited['source_files_sha256'],
        'references': inherited['references'], 'row_maps': inherited['row_maps'],
        'images': inherited['images'], 'settings': settings(args.experiment),
        'primary_reference': 'frozen666', 'reviewed_reference': 'r50',
        'model': {'architecture': card['model'], 'task': 'pose', 'kpt_shape': [2, 3],
                  'classes': ['cassette_front'], 'native_head': 'Pose26',
                  'checkpoint_role': 'PR137 candidate; research checkpoint, not deployment selection',
                  'epoch173_initializer_sha256': card['sha256'],
                  'original_initialization_sha256': card['original_initialization_sha256'],
                  'original_pretraining_source': 'not established by the recorded hash alone',
                  'metadata_sha256': metadata},
        'required_packages': PINNED_PACKAGES, 'ultralytics_wheel_provenance': ULTRALYTICS_WHEEL,
        'required_inference_source_sha256': INFERENCE_SOURCE_HASHES, 'development_only': True,
        'training': False, 'automatic_promotion': False}
    return prepare_inputs(args, context)


def endpoint_state(target, prediction):
    visible = [j for j, p in enumerate(target['keypoints']) if p[2] == 2]
    errors, valid = [], []
    for j in visible:
        p = None if prediction is None else prediction['keypoints'][j]
        okay = p is not None and p[0] is not None and p[1] is not None and p[2] >= .5
        okay = bool(okay and 0 <= p[0] < 1280 and 0 <= p[1] < 720)
        valid.append(j) if okay else None
        errors.append(math.hypot(p[0] - target['keypoints'][j][0],
                                 p[1] - target['keypoints'][j][1]) if okay else None)
    complete = bool(visible) and all(e is not None for e in errors)
    return {'locatable_count': len(visible), 'returned_visible_slots': valid,
            'errors_px': errors,
            'within5': bool(complete and max(errors) <= 5),
            'within10': bool(complete and max(errors) <= 10)}


def height_group(target):
    height = target['bbox_xyxy'][3] - target['bbox_xyxy'][1]
    return '<8' if height < 8 else '8-<16' if height < 16 else '16-<32' if height < 32 else '>=32'


def reports(refs, outputs):
    from audit_additional_front_pilot_r45_results import corner_groups
    from pose_scoring import aggregate_scores, score_image
    result = {}
    for name, gold in refs.items():
        result[name] = {}
        for view in ('raw', 'postclip'):
            scores, corners = {}, {str(n): Counter() for n in (0, 1, 2)}
            heights = {s: Counter(eligible=0, matched=0, missed=0) for s in ('<8', '8-<16', '16-<32', '>=32')}
            for image, targets in sorted(gold.items()):
                predictions = outputs[image][view]
                score = score_image(targets, predictions, 1280, 720, MATCH, .5)
                scores[image] = score
                for key, values in corner_groups(targets, predictions, score).items():
                    corners[key].update(values)
                matches = {m['gold_index'] for m in score['matches']}
                for j, target in enumerate(targets):
                    group = heights[height_group(target)]
                    group['eligible'] += 1
                    group['matched' if j in matches else 'missed'] += 1
            result[name][view] = {'detection': aggregate_scores(list(scores.values())),
                'corners_by_locatable_count': corners, 'height_strata_original_px': heights,
                'per_image': scores}
    return json.loads(encoded(result))


def numeric_drift(expected, actual):
    """Require corresponding output order; quantify float drift without rounding."""
    maxima = {'coordinate_component_px': 0., 'probability': 0.}
    failures = []
    if len(expected) != len(actual):
        return maxima, ['prediction_count']
    for number, (a, b) in enumerate(zip(expected, actual), 1):
        if a['class_id'] != b['class_id'] or a.get('primary_prediction_id') != b.get('primary_prediction_id'):
            failures.append(f'out{number:03d}/class_or_raw_id')
        pairs = [('coordinate_component_px', x, y) for x, y in zip(a['bbox_xyxy'], b['bbox_xyxy'])]
        pairs.append(('probability', a['confidence'], b['confidence']))
        for x, y in zip(a['keypoints'], b['keypoints']):
            pairs.extend([('coordinate_component_px', x[0], y[0]),
                          ('coordinate_component_px', x[1], y[1]), ('probability', x[2], y[2])])
        for field, x, y in pairs:
            if x is None or y is None:
                if x != y:
                    failures.append(f'out{number:03d}/{field}/missingness')
            elif not math.isfinite(x) or not math.isfinite(y):
                failures.append(f'out{number:03d}/{field}/nonfinite')
            else:
                delta = abs(x-y)
                maxima[field] = max(maxima[field], delta)
                if delta > TOLERANCES[field]:
                    failures.append(f'out{number:03d}/{field}')
    return maxima, sorted(set(failures))


def baseline_equivalence(refs, expected, actual):
    """Numerical tolerance never permits changed discrete conclusions or IDs."""
    failed, maximum = [], {k: 0. for k in TOLERANCES}
    if set(expected) != set(actual):
        return {'passed': False, 'exact_json_equal': False, 'failures': ['image_membership'],
                'maximum_drift': maximum, 'tolerances': TOLERANCES}
    for image in sorted(expected):
        for view in ('raw', 'postclip'):
            delta, reasons = numeric_drift(expected[image][view], actual[image][view])
            for key, value in delta.items():
                maximum[key] = max(maximum[key], value)
            failed.extend(f'{image}/{view}/{r}' for r in reasons)
        for key in ('primary_prediction_count', 'retained_primary_prediction_ids'):
            if expected[image]['filter'][key] != actual[image]['filter'][key]:
                failed.append(f'{image}/filter/{key}')
        for side in ('primary_prediction_id', 'suppressor_primary_prediction_id'):
            if [p[side] for p in expected[image]['filter']['suppressed_predictions']] != [p[side] for p in actual[image]['filter']['suppressed_predictions']]:
                failed.append(f'{image}/filter/{side}')
    reference_reports, current_reports = reports(refs, expected), reports(refs, actual)
    for name, gold in refs.items():
        for view in ('raw', 'postclip'):
            a, b = reference_reports[name][view], current_reports[name][view]
            for key in ('corners_by_locatable_count', 'height_strata_original_px'):
                if a[key] != b[key]:
                    failed.append(f'{name}/{view}/{key}')
            for image, targets in sorted(gold.items()):
                x, y = a['per_image'][image], b['per_image'][image]
                if any(x[k] != y[k] for k in (*COUNT_KEYS, *INDEX_KEYS)):
                    failed.append(f'{name}/{view}/{image}/discrete_metrics_or_unmatched_ids')
                pairs = lambda s: [(m['gold_index'], m['prediction_index']) for m in s['matches']]
                if pairs(x) != pairs(y):
                    failed.append(f'{name}/{view}/{image}/match_ids')
                for old, new in zip(x['endpoint_errors_px'], y['endpoint_errors_px']):
                    delta = abs(old-new)
                    maximum['endpoint_error_px'] = max(maximum['endpoint_error_px'], delta)
                    if delta > TOLERANCES['endpoint_error_px']:
                        failed.append(f'{name}/{view}/{image}/endpoint_error')
                mx, my = ({m['gold_index']: m['prediction_index'] for m in s['matches']} for s in (x, y))
                for j, target in enumerate(targets):
                    sx = endpoint_state(target, expected[image][view][mx[j]] if j in mx else None)
                    sy = endpoint_state(target, actual[image][view][my[j]] if j in my else None)
                    if any(sx[k] != sy[k] for k in ('returned_visible_slots', 'within5', 'within10')):
                        failed.append(f'{name}/{view}/{image}/R{j+1:03d}/endpoint_threshold_state')
    return {'passed': not failed, 'exact_json_equal': expected == actual,
            'failures': sorted(set(failed)), 'maximum_drift': maximum, 'tolerances': TOLERANCES,
            'discrete_policy': 'Exact per-image match/miss/unmatched/duplicate IDs, counts, and per-row endpoint threshold states across all four references; raw and postclip.'}


def case_inventory(refs, maps, arms, scored):
    rows, unmatched, pairs = [], [], []
    for name, gold in refs.items():
        for image, targets in sorted(gold.items()):
            states = {}
            for arm, output in arms.items():
                score = scored[arm][name]['postclip']['per_image'][image]
                matches = {m['gold_index']: m['prediction_index'] for m in score['matches']}
                states[arm] = {}
                for j, target in enumerate(targets):
                    pi = matches.get(j)
                    prediction = None if pi is None else output[image]['postclip'][pi]
                    state = endpoint_state(target, prediction)
                    state.update(matched=pi is not None, output_number=None if pi is None else pi+1,
                                 primary_prediction_id=None if prediction is None else prediction['primary_prediction_id'])
                    stable = (maps[image]['rows'][j]['stable_id'] if name in ('r38', 'r50') else f'{name}/{image}:row{j+1:03d}')
                    record = {'case_id': f'{name}/{image}/R{j+1:03d}', 'reference': name,
                        'image_id': image, 'row_number': j+1, 'stable_row_id': stable, 'arm': arm,
                        'original_bbox_xyxy': target['bbox_xyxy'], 'height_group': height_group(target), **state}
                    rows.append(record); states[arm][j] = record
                for pi in score['unmatched_prediction_indices']:
                    p = output[image]['postclip'][pi]
                    unmatched.append({'case_id': f'{name}/{image}/{arm}/out{pi+1:03d}(P{p["primary_prediction_id"]:03d})',
                        'reference': name, 'image_id': image, 'arm': arm, 'output_number': pi+1,
                        'primary_prediction_id': p['primary_prediction_id'],
                        'formal_type': 'duplicate' if pi in score['duplicate_prediction_indices'] else 'other_unmatched',
                        'prediction': p, 'visual_cause': 'not inferred from geometry alone'})
            if 'treatment' in states:
                for j in range(len(targets)):
                    a, b = states['control'][j], states['treatment'][j]
                    pairs.append({'case_id': a['case_id'], 'stable_row_id': a['stable_row_id'],
                        'reference': name, 'image_id': image, 'row_number': j+1,
                        'locatable_count': a['locatable_count'], 'height_group': a['height_group'],
                        'recovered': not a['matched'] and b['matched'],
                        'newly_missed': a['matched'] and not b['matched'],
                        'within5_gain': not a['within5'] and b['within5'],
                        'within5_loss': a['within5'] and not b['within5'],
                        'within10_gain': not a['within10'] and b['within10'],
                        'within10_loss': a['within10'] and not b['within10'],
                        'control': {k: a[k] for k in ('output_number', 'primary_prediction_id', 'errors_px')},
                        'treatment': {k: b[k] for k in ('output_number', 'primary_prediction_id', 'errors_px')}})
    return {'coordinate_system': 'original 1280 x 720 pixels; x right, y down',
            'prediction_identity': 'Output number and raw P-number are arm-specific, not persistent object IDs.',
            'rows': rows, 'unmatched_outputs': unmatched, 'paired_rows': pairs}


def summarize_timing(measurements):
    result = {}
    for arm in ('control', 'treatment'):
        rows = [r for r in measurements if r['arm'] == arm]
        if not rows:
            continue
        summary = {'measured_calls': len(rows), 'images': len({r['image_id'] for r in rows}),
                   'input_tensor_shapes': sorted({tuple(r['input_tensor_shape']) for r in rows})}
        for key in ('photometric_ms', 'predict_call_ms', 'record_and_postclip_ms', 'array_to_output_ms'):
            values = sorted(r[key] for r in rows)
            summary[key] = {'median': statistics.median(values),
                            'p95_nearest_rank': values[math.ceil(.95*len(values))-1],
                            'mean': statistics.mean(values)}
        summary['stage_ms_mean'] = {k: statistics.mean(r['stage_ms'][k] for r in rows) for k in rows[0]['stage_ms']}
        for key in ('cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes', 'cuda_allocated_before_bytes'):
            values = [r[key] for r in rows if r[key] is not None]
            summary[key] = max(values) if values else None
        result[arm] = summary
    return json.loads(encoded(result))


def archive_name(experiment):
    return f'baomai-{experiment}-inference-r53-results.zip'


def package(context, payload):
    """Write immutable evidence and a full member inventory, including blocked runs."""
    work, experiment = context['work'], context['contract']['settings']['experiment']
    entries = {name: encoded(value) for name, value in payload.items()}
    entries['preflight.json'] = encoded(context['preflight'])
    entries['input-audit.json'] = encoded(context['receipt'])
    entries['run_inference_ablation_r53.py'] = Path(__file__).read_bytes()
    inventory = {name: {'bytes': len(data), 'sha256': sha(data)} for name, data in entries.items()}
    for name, data in entries.items():
        put(work / name, data)
    entries['inventory.json'] = encoded(inventory)
    destination = work / archive_name(experiment)
    need(not destination.exists(), 'Completed archive exists; verify and reuse it')
    temp = destination.with_suffix('.part')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(entries.items()):
            z.writestr(name, data)
    temp.replace(destination)
    return verify_return(context)


def verify_return(context):
    from predict_pose_folder import postclip_nms
    target = context['work'] / archive_name(context['contract']['settings']['experiment'])
    with zipfile.ZipFile(target) as z:
        names = z.namelist()
        expected = {'summary.json', 'predictions.json', 'baseline-equivalence.json', 'cases.json',
                    'timing.json', 'runtime.json', 'repeatability.json', 'repeated-predictions.json', 'preflight.json',
                    'input-audit.json', 'run_inference_ablation_r53.py', 'inventory.json'}
        need(len(names) == len(set(names)) and set(names) == expected and z.testzip() is None,
             'Completed return membership/CRC differs; preserved')
        inventory = json.loads(z.read('inventory.json'))
        need(set(inventory) == expected - {'inventory.json'}, 'Return inventory membership differs')
        for name, entry in inventory.items():
            data = z.read(name)
            need(entry == {'bytes': len(data), 'sha256': sha(data)}, 'Return hash differs: ' + name)
        need(z.read('run_inference_ablation_r53.py') == Path(__file__).read_bytes(), 'Returned driver differs')
        need(json.loads(z.read('preflight.json')) == context['preflight'], 'Return input contract differs')
        need(json.loads(z.read('input-audit.json')) == context['receipt'], 'Return input audit differs')
        summary, arms = (json.loads(z.read(n)) for n in ('summary.json', 'predictions.json'))
        need(summary['contract'] == context['contract'], 'Return source/settings differ')
        need(set(arms) in ({'control'}, {'control', 'treatment'}), 'Return arm membership differs')
        for arm, values in arms.items():
            need(set(values) == set(context['records']), 'Return image membership differs: ' + arm)
            for image, row in values.items():
                kept, audit = postclip_nms(row['raw'])
                need(kept == row['postclip'] and audit == row['filter'], 'Return postclip differs: ' + image)
        baseline = baseline_equivalence(context['refs'], context['prior'], arms['control'])
        need(json.loads(z.read('baseline-equivalence.json')) == baseline, 'Return baseline gate differs')
        need(not ('treatment' in arms and not baseline['passed']), 'Treatment ran after failed baseline')
        scored = {arm: reports(context['refs'], output) for arm, output in arms.items()}
        need(scored == summary['reports'], 'Return score replay differs')
        need(json.loads(z.read('cases.json')) == case_inventory(context['refs'], context['maps'], arms, scored),
             'Return paired case replay differs')
        repeated = json.loads(z.read('repeated-predictions.json'))
        need(len(repeated) == (TIMING_REPEATS if baseline['passed'] else 0), 'Repeat membership differs')
        replayed_repeatability = []
        for index, observation in enumerate(repeated, 1):
            need(set(observation) == {'control', 'treatment'}, 'Repeat arm membership differs')
            for arm, outputs in observation.items():
                need(set(outputs) == set(context['records']), 'Repeat image membership differs')
                for image, row in outputs.items():
                    kept, audit = postclip_nms(row['raw'])
                    need(kept == row['postclip'] and audit == row['filter'], 'Repeated postclip differs')
                replayed_repeatability.append({'arm': arm, 'repeat': index,
                    **baseline_equivalence(context['refs'], arms[arm], outputs)})
        need(replayed_repeatability == json.loads(z.read('repeatability.json')), 'Repeatability replay differs')
        repeatable = all(r['passed'] for r in replayed_repeatability)
        valid = baseline['passed'] and repeatable
        expected_status = 'completed' if valid else 'blocked_baseline' if not baseline['passed'] else 'blocked_repeatability'
        need(summary['status'] == expected_status and summary['baseline_passed'] == baseline['passed']
             and summary['repeated_outputs_stable'] == repeatable
             and summary['interpretation_allowed'] == valid
             and summary['treatment_executed'] == ('treatment' in arms), 'Returned status/gates differ')
        timing = json.loads(z.read('timing.json'))
        need(timing['summary'] == summarize_timing(timing['measurements']), 'Return timing summary differs')
        expected_calls = {(arm, image, repeat) for arm in ('control', 'treatment')
                          for image in context['records'] for repeat in range(1, TIMING_REPEATS+1)} if baseline['passed'] else set()
        actual_calls = [(r['arm'], r['image_id'], r['repeat']) for r in timing['measurements']]
        need(len(actual_calls) == len(set(actual_calls)) and set(actual_calls) == expected_calls,
             'Timing call membership differs')
        need(summary['training'] is False and summary['automatic_promotion'] is False
             and summary['fresh_inference'] is True, 'Return experiment declarations differ')
    print('Verified R53 evidence:', target, 'SHA256:', sha(target.read_bytes()), flush=True)
    return summary


def run(context, device):
    if (context['work'] / archive_name(context['contract']['settings']['experiment'])).exists():
        return verify_return(context)
    for package_name, required in PINNED_PACKAGES.items():
        need(md.version(package_name) == required, f'Expected {package_name}=={required}')
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.pose.predict import PosePredictor
    from evaluate_pose import INFERENCE, _prediction_records
    from partial_classification_r51 import verify_native_sources
    from pose_checkpoint import check_loaded_pose_model
    from predict_pose_folder import postclip_nms
    need(device == 'cpu' or torch.cuda.is_available(), 'Requested GPU is unavailable; select a GPU runtime')
    native = verify_native_sources()
    package_root = Path(importlib.util.find_spec('ultralytics').origin).parent
    for relative, expected in INFERENCE_SOURCE_HASHES.items():
        checked(package_root / relative, expected)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    class ShapePredictor(PosePredictor):
        def preprocess(self, im):
            need(len(im) == 1 and not self.scale_fill and self.model.format == 'pt',
                 'Expected one-image, aspect-preserving native PT letterbox')
            original_h, original_w = im[0].shape[:2]
            tensor = super().preprocess(im)
            self.r53_input_shape = list(tensor.shape)
            gain = min(self.imgsz[0]/original_h, self.imgsz[1]/original_w)
            resized_h, resized_w = round(original_h*gain), round(original_w*gain)
            pad_h, pad_w = tensor.shape[2]-resized_h, tensor.shape[3]-resized_w
            need(pad_h >= 0 and pad_w >= 0, 'Letterbox padding/gain contract differs')
            self.r53_letterbox = {'scale_gain': gain, 'resized_hw': [resized_h,resized_w],
                'padded_hw': list(tensor.shape[2:]),
                'padding_ltrb': [round(pad_w/2-.1),round(pad_h/2-.1),
                                 round(pad_w/2+.1),round(pad_h/2+.1)],
                'derivation': 'Verified native LetterBox default scaleup=True, scale_fill=False; original shape and self.imgsz.'}
            return tensor

    model = YOLO(str(context['weights'])); check_loaded_pose_model(model)
    head = model.model.model[-1]
    need(type(head).__name__ == 'Pose26' and head.nc == 1 and head.reg_max == 1,
         'Expected original native Pose26 checkpoint architecture')
    config = context['contract']['settings']
    def sync():
        if device != 'cpu':
            torch.cuda.synchronize(backend.device)
    for _ in range(2):
        model.predict(source=np.zeros((720, 1280, 3), dtype=np.uint8),
            predictor=ShapePredictor, device=device, conf=CONF, **INFERENCE)
    backend = model.predictor.model
    need(not backend.end2end and not backend.fp16, 'Expected ordinary one-to-many NMS and FP32')
    torch.set_num_threads(4)
    runtime = {'python': platform.python_version(), 'platform': platform.platform(),
        'device': str(backend.device), 'gpu': torch.cuda.get_device_name(backend.device) if device != 'cpu' else None,
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'packages': {n: md.version(n) for n in (*PINNED_PACKAGES, 'torchvision')},
        'native_sources_verified': native, 'inference_sources_verified': INFERENCE_SOURCE_HASHES,
        'ultralytics_wheel_provenance': ULTRALYTICS_WHEEL,
        'backend_end2end': bool(backend.end2end), 'backend_fp16': bool(backend.fp16),
        'torch_threads': torch.get_num_threads(), 'tf32': False, 'cudnn_benchmark': False,
        'prior_runtime': context['old_summary']['runtime'],
        'warmup': 'Two zero-image1280 predictions before baseline; two predictions per arm before timing, all excluded.',
        'timing_scope': 'Original decoded BGR array through photometry, prediction, CPU record conversion and postclip. Excludes file I/O, model load, archive audit and camera acquisition. This is not camera FPS.',
        'memory_scope': 'CUDA allocator peak during each call; allocated and reserved differ and do not measure total device use.'}
    images = {}
    for identity in sorted(context['records']):
        array = cv2.imread(str(context['originals'] / (identity + '.png')), cv2.IMREAD_COLOR)
        need(array is not None and array.shape == (720, 1280, 3) and array.dtype == np.uint8,
             'Decoded image dimensions/type differ: ' + identity)
        images[identity] = array
    pixel_hashes = {i: sha(a.tobytes()) for i, a in images.items()}
    runtime['decoded_bgr_sha256'] = pixel_hashes
    measurements = []

    def infer(identity, arm, repeat, measured):
        source = images[identity]
        choice = config[arm]
        sync()
        if measured and device != 'cpu':
            torch.cuda.reset_peak_memory_stats(backend.device)
        before = torch.cuda.memory_allocated(backend.device) if device != 'cpu' else None
        start = time.perf_counter()
        transformed = preprocess_bgr(source, choice['photometry'])
        after_photo = time.perf_counter()
        result = model.predict(source=transformed, device=device, conf=CONF,
            **{**INFERENCE, 'imgsz': choice['imgsz']})[0]
        sync(); after_predict = time.perf_counter()
        need(tuple(result.orig_shape) == (720, 1280), 'Original-coordinate contract changed')
        raw = _prediction_records(result)
        kept, filtered = postclip_nms(raw)
        sync(); finished = time.perf_counter()
        row = {'raw': raw, 'postclip': kept, 'filter': filtered}
        if measured:
            measurements.append({'image_id': identity, 'arm': arm, 'repeat': repeat,
                'photometric_ms': (after_photo-start)*1000,
                'predict_call_ms': (after_predict-after_photo)*1000,
                'record_and_postclip_ms': (finished-after_predict)*1000,
                'array_to_output_ms': (finished-start)*1000,
                'stage_ms': {k: float(v) for k, v in result.speed.items()},
                'input_tensor_shape': model.predictor.r53_input_shape,
                'letterbox': model.predictor.r53_letterbox,
                'cuda_allocated_before_bytes': before,
                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(backend.device) if device != 'cpu' else None,
                'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(backend.device) if device != 'cpu' else None,
                'transformed_bgr_sha256': sha(transformed.tobytes())})
        return row

    control = {identity: infer(identity, 'control', 0, False) for identity in sorted(images)}
    baseline = baseline_equivalence(context['refs'], context['prior'], control)
    arms, repeatability, repeated = {'control': control}, [], []
    if baseline['passed']:
        print('Baseline reproduction passed; running the one prespecified treatment.', flush=True)
        first = sorted(images)[0]
        for arm in ('control', 'treatment'):
            for _ in range(2):
                infer(first, arm, -1, False)
        for repeat in range(TIMING_REPEATS):
            observed = {'control': {}, 'treatment': {}}
            for number, identity in enumerate(sorted(images)):
                order = ('control', 'treatment') if (number+repeat) % 2 == 0 else ('treatment', 'control')
                for arm in order:
                    observed[arm][identity] = infer(identity, arm, repeat+1, True)
            if repeat == 0:
                arms['treatment'] = observed['treatment']
            repeated.append(observed)
            for arm in ('control', 'treatment'):
                repeatability.append({'arm': arm, 'repeat': repeat+1,
                    **baseline_equivalence(context['refs'], arms[arm], observed[arm])})
            print(f'Finished paired timing repeat {repeat+1}/{TIMING_REPEATS}', flush=True)
    else:
        print('BASELINE GATE FAILED: treatment was not run. Download the diagnostic archive.', flush=True)
    need(pixel_hashes == {i: sha(a.tobytes()) for i, a in images.items()}, 'Original pixel array mutated')
    scored = {arm: reports(context['refs'], output) for arm, output in arms.items()}
    repeatable = all(r['passed'] for r in repeatability)
    valid = baseline['passed'] and repeatable
    summary = {'schema': 'baomai-r53-inference-results-v1', 'contract': context['contract'],
        'status': 'completed' if valid else 'blocked_baseline' if not baseline['passed'] else 'blocked_repeatability',
        'baseline_passed': baseline['passed'], 'repeated_outputs_stable': repeatable,
        'treatment_executed': 'treatment' in arms, 'interpretation_allowed': valid,
        'fresh_inference': True, 'training': False, 'automatic_promotion': False,
        'reports': scored, 'interpretation': 'Development comparison only. No benefit is established until this returned evidence is reviewed; no model, label, filter or threshold is promoted.'}
    payload = {'summary.json': summary, 'predictions.json': arms,
        'baseline-equivalence.json': baseline, 'cases.json': case_inventory(context['refs'], context['maps'], arms, scored),
        'timing.json': {'measurements': measurements, 'summary': summarize_timing(measurements),
                        'order': 'Alternating arm order by image index and repeat; fixed20 images,3 repeats.'},
        'runtime.json': runtime, 'repeatability.json': repeatability,
        'repeated-predictions.json': repeated}
    return package(context, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'verify'))
    parser.add_argument('--experiment', choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--script-sha256', required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--originals-dir', type=Path)
    parser.add_argument('--device', default='0')
    args = parser.parse_args(); args.root = args.root.resolve()
    context = source_check(args)
    if args.action == 'prepare':
        print('R53 CPU preflight passed. No model was loaded and no fresh inference was performed.')
    else:
        result = run(context, args.device) if args.action == 'run' else verify_return(context)
        print(json.dumps({'status': result['status'], 'interpretation_allowed': result['interpretation_allowed'],
                          'treatment_executed': result['treatment_executed']}, indent=2))


if __name__ == '__main__':
    main()
