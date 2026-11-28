#!/usr/bin/env python3
"""Trace current missed fronts through fixed PR143 decoded proposals on CPU.

No model execution, threshold search, training, or annotation changes. Historical
0.13 replay must be exact; current 0.25 replay must preserve measured outcomes
across the GPU/CPU runtime difference before its proposal stages are interpreted.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

sys.dont_write_bytecode = True
from audit_front_diagnosis_r49_results import (
    ARCHIVE_BYTES as R49_BYTES, ARCHIVE_SHA as R49_SHA, NAME as R49_NAME,
    PR_HEAD as R49_HEAD, SOURCE as R49_SOURCE, SCRIPT_SHA as R49_SCRIPT_SHA)
from audit_reflection_return import valid_point
from evaluate_pose import parse_pose_labels
from pose_scoring import _box_iou, score_image
from predict_pose_folder import postclip_nms
from replay_front_confidence_r51 import input_boxes, original_boxes, native_select, prediction_records
from run_corner_average_r58 import report
from run_original_comparison_r59 import assemble
from run_reference_matrix_r60 import verified_zip

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = 'docs/thesis/evidence/annotation-reference/baomai-annotation-reference-r60-results.zip'
REFERENCE_SHA = 'ce8df4bcf223b6bee49fa67e829a443acb572e4a6aad75fa0f92657a16cc2601'
REFERENCE_BYTES = 972658
MODEL_SHA = '222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9'
CONFIDENCE, MATCHING_IOU, NMS_IOU = .25, .50, .70
COORDINATE_TOLERANCE_PX, SCORE_TOLERANCE = .05, 1e-5
STAGES = ('no_iou_eligible_decoded_box', 'confidence_gate', 'ordinary_nms',
          'postclip_suppression', 'retained_assignment_contention')
COUNT_KEYS = ('gold_count', 'prediction_count', 'tp', 'fn', 'fp', 'duplicate_fp',
              'other_fp', 'visible_endpoints', 'returned_endpoints', 'missing_endpoints')
IDENTITY_KEYS = ('unmatched_gold_indices', 'unmatched_prediction_indices',
                 'duplicate_prediction_indices', 'other_fp_prediction_indices')
SOURCES = ('analyze_proposal_stages_r63.py', 'audit_front_diagnosis_r49_results.py',
           'replay_front_confidence_r51.py', 'pose_scoring.py', 'predict_pose_folder.py',
           'evaluate_pose.py', 'run_reference_matrix_r60.py', 'run_original_comparison_r59.py',
           'run_corner_average_r58.py', 'audit_reflection_return.py')
LIMITATIONS = [
    'Exploratory diagnosis of 20 repeatedly used development photographs, not independent test evidence.',
    'The two archives record identical model and photograph SHA-256 values; this replay does not load model weights or photograph bytes.',
    'Decoded tensors came from the historical Tesla T4 runtime. Current retained outputs came from a newer CPU execution with different Python and PyTorch versions. Numeric bytes are not identical; measured identities and threshold outcomes are verified separately.',
    'Only the original-view trained checkpoint has these decoded tensors. Reflected proposals and future newly trained checkpoints are not diagnosed.',
    'IoU eligibility is a geometric relation to the reviewed annotations. Many eligible proposals can describe the same physical object, and one proposal can overlap multiple references.',
    'Maximum one-to-one matching is a diagnostic upper bound for each fixed candidate set, not a measured detector result or a deployable oracle.',
    'A stage label identifies where eligible support disappears in this fixed computation. It does not prove an optimal intervention, correct physical annotations, or recoverable performance without extra predictions.',
    'No eligible decoded box means no box reaches this reference IoU threshold in the finite decoded tensor; it does not mean the neural network had no visual response.',
    'The audit does not change the current confidence threshold, suppression policy, labels, retained method, or two-view corner measurements.',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def read_json(members, name):
    data = members[name]
    return json.loads(gzip.decompress(data) if name.endswith('.gz') else data)


def score_identity(score):
    return {**{key: score[key] for key in COUNT_KEYS + IDENTITY_KEYS},
            'matches': [(m['gold_index'], m['prediction_index']) for m in score['matches']]}


def endpoint_outcomes(gold, predictions, score):
    matches = {m['gold_index']: m['prediction_index'] for m in score['matches']}
    outcomes = []
    for gi, target in enumerate(gold):
        pi = matches.get(gi)
        for ki, point in enumerate(target['keypoints']):
            if point[2] != 2:
                continue
            predicted = predictions[pi]['keypoints'][ki] if pi is not None else None
            usable = predicted is not None and valid_point(predicted)
            error = math.hypot(predicted[0] - point[0], predicted[1] - point[1]) if usable else None
            outcomes.append((gi, ki, usable, error is not None and error <= 5,
                             error is not None and error <= 10))
    return outcomes


def record_difference(replayed, current):
    require(len(replayed) == len(current), 'Native output count changed between runtimes')
    coordinates = scores = 0.
    for old, new in zip(replayed, current):
        require(old['class_id'] == new['class_id'], 'Native record class/order differs')
        coordinates = max(coordinates, *(abs(a - b) for a, b in zip(old['bbox_xyxy'], new['bbox_xyxy'])))
        scores = max(scores, abs(old['confidence'] - new['confidence']))
        for a, b in zip(old['keypoints'], new['keypoints']):
            coordinates = max(coordinates, abs(a[0] - b[0]), abs(a[1] - b[1]))
            scores = max(scores, abs(a[2] - b[2]))
    require(coordinates <= COORDINATE_TOLERANCE_PX and scores <= SCORE_TOLERANCE,
            'Cross-runtime numeric difference exceeds declared tolerances')
    return {'maximum_coordinate_difference_px': coordinates, 'maximum_score_difference': scores}


def iou_matrix(boxes, gold):
    """Vectorized float64 form of the unchanged scorer's scaled-area formula."""
    boxes = np.asarray(boxes, dtype=np.float64)
    if not gold:
        return np.empty((len(boxes), 0), dtype=np.float64)
    targets = np.asarray([g['bbox_xyxy'] for g in gold], dtype=np.float64)
    width = np.maximum(0., np.minimum(boxes[:, None, 2], targets[None, :, 2]) -
                       np.maximum(boxes[:, None, 0], targets[None, :, 0]))
    height = np.maximum(0., np.minimum(boxes[:, None, 3], targets[None, :, 3]) -
                        np.maximum(boxes[:, None, 1], targets[None, :, 1]))
    intersection = width * height
    first = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[:, None]
    second = ((targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1]))[None, :]
    scale = np.maximum(first, second)
    scaled_intersection = intersection / scale
    return scaled_intersection / (first / scale + second / scale - scaled_intersection)


def eligibility_graph(overlaps, indices):
    """Count geometric support without treating shared proposals as independent."""
    indices = [int(i) for i in indices]
    neighbors = {i: np.flatnonzero(overlaps[i] >= MATCHING_IOU).tolist() for i in indices}
    support = Counter(j for values in neighbors.values() for j in values)
    assigned = {}

    def augment(candidate, seen):
        for reference in neighbors[candidate]:
            if reference in seen:
                continue
            seen.add(reference)
            if reference not in assigned or augment(assigned[reference], seen):
                assigned[reference] = candidate
                return True
        return False

    for index in indices:
        if len(assigned) == overlaps.shape[1]:
            break
        augment(index, set())
    return {
        'candidate_count': len(indices),
        'eligible_candidate_count': sum(bool(v) for v in neighbors.values()),
        'candidates_eligible_for_multiple_references': sum(len(v) > 1 for v in neighbors.values()),
        'references_with_any_candidate': len(support),
        'references_with_multiple_candidates': sum(n > 1 for n in support.values()),
        'maximum_one_to_one_matches': len(assigned),
        'uncovered_reference_indices': [j for j in range(overlaps.shape[1]) if j not in support],
    }


def stage_label(eligible, confidence, native_indices, final_indices):
    eligible = set(eligible)
    above = {i for i in eligible if confidence[i] > CONFIDENCE}
    if not eligible:
        return STAGES[0]
    if not above:
        return STAGES[1]
    if not above.intersection(native_indices):
        return STAGES[2]
    if not above.intersection(final_indices):
        return STAGES[3]
    return STAGES[4]


def describe_cases(image, gold, raw, boxes, overlaps, indices, suppressed, final,
                   postclip_audit, final_indices, old_predictions, old_indices):
    score = score_image(gold, final, 1280, 720, MATCHING_IOU, .5)
    old_score = score_image(gold, old_predictions, 1280, 720, MATCHING_IOU, .5)
    matches = {final_indices[m['prediction_index']]: m['gold_index'] for m in score['matches']}
    old_matches = {m['gold_index']: m for m in old_score['matches']}
    native_position = {anchor: position for position, anchor in enumerate(indices)}
    final_position = {anchor: position for position, anchor in enumerate(final_indices)}
    after_clip = {indices[row['primary_prediction_id'] - 1]: row
                  for row in postclip_audit['suppressed_predictions']}
    cases = []
    for gi in score['unmatched_gold_indices']:
        target = gold[gi]
        # This direct scorer check also verifies the vectorized eligibility table.
        direct = np.array([_box_iou(box, target['bbox_xyxy']) for box in original_boxes(boxes).tolist()])
        require(np.array_equal(direct, overlaps[:, gi]), 'Vectorized/scorer overlap differs: ' + image)
        eligible = np.flatnonzero(overlaps[:, gi] >= MATCHING_IOU).tolist()
        above = [i for i in eligible if raw[0, 4, i] > CONFIDENCE]

        def describe(anchor):
            return {'anchor_index': anchor, 'anchor_number': anchor + 1,
                    **prediction_records(raw, boxes, [anchor])[0],
                    'target_iou': float(overlaps[anchor, gi]),
                    'eligible_reference_indices': np.flatnonzero(overlaps[anchor] >= MATCHING_IOU).tolist(),
                    'native_prediction_index': native_position.get(anchor),
                    'native_prediction_position': native_position[anchor] + 1 if anchor in native_position else None,
                    'postclip_prediction_index': final_position.get(anchor),
                    'postclip_prediction_position': final_position[anchor] + 1 if anchor in final_position else None,
                    'matched_reference_index': matches.get(anchor)}

        best = max(range(raw.shape[2]), key=lambda i: (float(overlaps[i, gi]), float(raw[0, 4, i]), -i))
        highest = max(eligible, key=lambda i: (float(raw[0, 4, i]), -i)) if eligible else None
        stage = stage_label(eligible, raw[0, 4], indices, final_indices)
        retained = [i for i in eligible if i in final_position]
        if retained:
            require(all(i in matches and matches[i] != gi for i in retained),
                    'Unmatched target has an unassigned eligible retained prediction')
        native_losses = []
        for anchor in above:
            if anchor in native_position:
                continue
            require(anchor in suppressed, 'Eligible native loss lacks a suppression witness')
            witness = suppressed[anchor]
            native_losses.append({'candidate': describe(anchor),
                                  'suppressor': describe(witness['suppressor']),
                                  'pair_iou_before_clipping_fp32': witness['pair_iou_fp32']})
        clip_losses = []
        for anchor in eligible:
            if anchor in after_clip:
                witness = after_clip[anchor]
                clip_losses.append({'candidate': describe(anchor),
                                    'suppressor': describe(indices[witness['suppressor_primary_prediction_id'] - 1]),
                                    'pair_iou_after_clipping': witness['postclip_prediction_iou']})
        prior = old_matches.get(gi)
        cases.append({
            'reference': 'r50', 'method': 'proposed_original_view_boxes', 'image': image,
            'reference_index': gi, 'reference_row': gi + 1, 'target': target,
            'available_corners': sum(p[2] == 2 for p in target['keypoints']), 'stage': stage,
            'eligible_decoded_count': len(eligible), 'eligible_above_confidence_count': len(above),
            'eligible_native_count': sum(i in native_position for i in eligible),
            'eligible_postclip_count': len(retained),
            'eligible_decoded_anchor_indices': eligible, 'eligible_above_confidence_anchor_indices': above,
            'best_decoded': describe(best),
            'highest_confidence_eligible': describe(highest) if highest is not None else None,
            'ordinary_nms_conflicts': native_losses, 'postclip_conflicts': clip_losses,
            'retained_assignment_conflicts': [describe(i) for i in retained],
            'historical_013_match': {'prediction_index': prior['prediction_index'],
                'anchor_index': old_indices[prior['prediction_index']], 'target_iou': prior['iou'],
                'prediction': old_predictions[prior['prediction_index']]} if prior is not None else None,
        })
    return cases, score


def analyze(r49_archive, reference_archive, root=ROOT):
    root = Path(root).resolve()
    require(root == ROOT, 'Analysis source root differs from imported helpers')
    old = verified_zip(r49_archive, R49_SHA, R49_BYTES)
    current = verified_zip(reference_archive, REFERENCE_SHA, REFERENCE_BYTES)
    old_summary, saved = (read_json(old, n) for n in ('summary.json', 'predictions.json'))
    traces = read_json(old, 'trace.json')
    current_summary = read_json(current, 'summary.json')
    contract = current_summary['contract']
    require(old_summary['source'] == R49_SOURCE and old_summary['script_sha256'] == R49_SCRIPT_SHA
            and digest(old['diagnose_r49.py']) == R49_SCRIPT_SHA, 'Historical source identity differs')
    require(old_summary['checkpoint_sha256'] == contract['adapted_model_sha256'] == MODEL_SHA
            and read_json(old, 'input-audit.json')['model_sha256']['candidate'] == MODEL_SHA,
            'Model identity differs')
    require(old_summary['runtime']['ultralytics'] == contract['prediction_runtime']['packages']['ultralytics'] == '8.4.182',
            'Ultralytics version differs')
    require({i: t['image_sha256'] for i, t in traces.items()} == contract['photographs'], 'Photograph hashes differ')
    require(old_summary['settings']['conf'] == .13 and old_summary['settings']['iou'] == NMS_IOU
            and old_summary['settings']['max_det'] == 300
            and contract['protocol']['box_confidence_strictly_greater'] == CONFIDENCE
            and contract['protocol']['matching_iou_inclusive'] == MATCHING_IOU
            and contract['protocol']['native_nms_iou'] == NMS_IOU
            and contract['protocol']['postclip_suppression_iou_strictly_greater'] == NMS_IOU,
            'Fixed pipeline settings differ')
    native = read_json(current, 'native-predictions.json.gz')
    arms, current_suppression, associations = assemble(native)
    for name, value in (('predictions', arms), ('suppression', current_suppression), ('associations', associations)):
        require(value == read_json(current, name + '.json.gz'), 'Current reconstruction differs: ' + name)
    texts = read_json(current, 'reference-labels.json.gz')
    refs = {}
    for reference, labels in texts.items():
        require(set(labels) == set(traces), 'Reference image membership differs')
        refs[reference] = {}
        for image, text in labels.items():
            require(digest(text.encode()) == contract['references'][reference]['labels'][image]['sha256'], 'Reference label hash differs')
            refs[reference][image] = parse_pose_labels(text, 1280, 720, reference + '/' + image)
    require(set(refs) == {'r50', 'frozen666', 'supplied664'} and len(traces) == 20 and set(traces) == set(saved),
            'Unexpected reference or image population')
    saved_scores = read_json(current, 'per-image.json.gz')
    replayed, historical, verification, cases, graph_rows = {}, {}, [], [], []
    for image in sorted(traces):
        trace = traces[image]
        require(trace['input_shape'] == [1, 3, 736, 1280] and trace['shape'] == [1, 11, 19320], 'Decoded geometry differs')
        raw = np.load(io.BytesIO(old['decoded/' + image + '.npy']), allow_pickle=False)
        require(raw.dtype == np.dtype('float32') and list(raw.shape) == trace['shape'] and np.isfinite(raw).all()
                and digest(raw.tobytes()) == trace['raw_tensor_sha256'], 'Decoded tensor identity differs')
        require(np.all((raw[0, 4] >= 0) & (raw[0, 4] <= 1)), 'Decoded score range differs')
        boxes = input_boxes(raw)
        old_indices, _ = native_select(boxes, raw[0, 4], .13)
        old_native = prediction_records(raw, boxes, old_indices)
        old_final, old_audit = postclip_nms(old_native)
        require(old_indices == saved[image]['ordinary_indices'] and old_native == saved[image]['hard_raw']
                and old_final == saved[image]['hard'] and old_audit == saved[image]['hard_audit'],
                'Historical 0.13 replay is not exact: ' + image)
        historical[image] = old_final
        old_final_indices = [old_indices[j - 1] for j in old_audit['retained_primary_prediction_ids']]
        require(old_final_indices == saved[image]['hard_indices'], 'Historical postclip anchor map differs')
        indices, suppressed = native_select(boxes, raw[0, 4], CONFIDENCE)
        records = prediction_records(raw, boxes, indices)
        require(all(p['confidence'] > CONFIDENCE for p in native['trained'][image]), 'Current native confidence truncation differs')
        differences = record_difference(records, native['trained'][image])
        final, audit = postclip_nms(records)
        replayed[image] = final
        final_indices = [indices[j - 1] for j in audit['retained_primary_prediction_ids']]
        require(audit['retained_primary_prediction_ids'] == current_suppression['trained'][image]['retained_primary_prediction_ids'],
                'Current postclip retained identities differ')
        verification.append({'image': image, 'decoded_tensor_sha256': trace['raw_tensor_sha256'],
                             'historical_013_native_and_postclip_exact': True,
                             'native025_records': len(records), 'postclip025_records': len(final), **differences})
        gold = refs['r50'][image]
        overlaps = iou_matrix(original_boxes(boxes), gold)
        image_cases, score = describe_cases(image, gold, raw, boxes, overlaps, indices, suppressed,
                                             final, audit, final_indices, old_final, old_final_indices)
        native_score = score_image(gold, records, 1280, 720, MATCHING_IOU, .5)
        cases.extend(image_cases)
        graph_rows.append({'image': image, 'reference_count': len(gold),
            'decoded': eligibility_graph(overlaps, range(raw.shape[2])),
            'above_confidence': eligibility_graph(overlaps, np.flatnonzero(raw[0, 4] > CONFIDENCE)),
            'native_nms': eligibility_graph(overlaps, indices),
            'postclip': eligibility_graph(overlaps, final_indices),
            'actual_native_counts': {k: native_score[k] for k in COUNT_KEYS},
            'actual_postclip_counts': {k: score[k] for k in COUNT_KEYS},
            'actual_duplicate_prediction_indices': score['duplicate_prediction_indices'],
            'actual_duplicate_anchor_indices': [final_indices[i] for i in score['duplicate_prediction_indices']]})
    comparisons = []
    for reference, gold in sorted(refs.items()):
        replay_report, replay_scores, replay_states = report(gold, replayed)
        current_report, current_scores, current_states = report(gold, arms['trained'])
        require(replay_report['pairs'] == current_report['pairs'], 'Complete-pair threshold outcomes differ')
        for a, b in zip(replay_states, current_states):
            require({k: v for k, v in a.items() if k != 'maximum_corner_error_px'} ==
                    {k: v for k, v in b.items() if k != 'maximum_corner_error_px'}, 'Per-front corner outcome differs')
        for image in gold:
            require(current_scores[image] == saved_scores[reference]['trained'][image], 'Current stored score differs')
            require(score_identity(replay_scores[image]) == score_identity(current_scores[image]), 'Current matching/outcome identity differs')
            require(endpoint_outcomes(gold[image], replayed[image], replay_scores[image]) ==
                    endpoint_outcomes(gold[image], arms['trained'][image], current_scores[image]), 'Per-corner threshold outcome differs')
            proposed_score = score_image(gold[image], arms['proposed'][image], 1280, 720, MATCHING_IOU, .5)
            require(proposed_score == saved_scores[reference]['proposed'][image], 'Current proposed stored score differs')
            require(score_identity(proposed_score) == score_identity(current_scores[image]), 'Corner averaging changed detection identities')
        comparisons.append({'reference': reference, 'images': len(gold),
                            'discrete_scores_and_matching_indices_exact': True,
                            'per_front_and_per_corner_threshold_outcomes_exact': True,
                            'trained_original_view_counts': {k: replay_report['detection'][k] for k in COUNT_KEYS},
                            'trained_original_view_pairs': replay_report['pairs']})
    stage_counts = {stage: sum(case['stage'] == stage for case in cases) for stage in STAGES}
    primary = next(row for row in current_summary['matrix'] if row['reference'] == 'r50' and row['method'] == 'proposed')
    require(sum(stage_counts.values()) == len(cases) == primary['missed'] == 18, 'Miss partition differs')
    require(sum(row['reference_count'] for row in graph_rows) == primary['fronts'] == 654, 'Target denominator differs')
    graph_totals = {stage: {key: sum(row[stage][key] for row in graph_rows)
                            for key in graph_rows[0][stage] if key != 'uncovered_reference_indices'}
                    for stage in ('decoded', 'above_confidence', 'native_nms', 'postclip')}
    detection_totals = {stage: {key: sum(row['actual_' + stage + '_counts'][key] for row in graph_rows)
                                for key in COUNT_KEYS} for stage in ('native', 'postclip')}
    for old_key, new_key in (('fronts', 'gold_count'), ('outputs', 'prediction_count'), ('matched', 'tp'),
                             ('missed', 'fn'), ('extras', 'fp'), ('duplicates', 'duplicate_fp')):
        require(primary[old_key] == detection_totals['postclip'][new_key], 'Primary measurement count differs')
    proposed_report, _, _ = report(refs['r50'], arms['proposed'])
    require(proposed_report['pairs']['eligible'] == primary['eligible_pairs']
            and all(proposed_report['pairs'][key] == primary[key] for key in ('within5', 'within10')),
            'Current proposed corner measurement differs')
    historical_report, _, _ = report(refs['r50'], historical)
    summary = {
        'schema': 'cassette-proposal-stages-r63-v1', 'status': 'completed',
        'fresh_inference': False, 'training': False, 'labels_modified': False,
        'automatic_promotion': False, 'development_only': True,
        'source_archives': {
            'decoded': {'name': R49_NAME, 'bytes': R49_BYTES, 'sha256': R49_SHA, 'pr_head': R49_HEAD, 'source': R49_SOURCE},
            'current_reference': {'path': REFERENCE_PATH, 'bytes': REFERENCE_BYTES, 'sha256': REFERENCE_SHA}},
        'model_sha256': MODEL_SHA, 'photographs_sha256': contract['photographs'],
        'references': contract['references'],
        'runtime_comparison': {'decoded_tensor_generation': old_summary['runtime'],
                              'current_prediction_generation': contract['prediction_runtime'],
                              'audit': {'python': platform.python_version(), 'numpy': np.__version__, 'device': 'cpu'},
                              'prediction_byte_identity_claimed': False},
        'analysis_sources_sha256': {'scripts/' + name: digest((root / 'scripts' / name).read_bytes()) for name in SOURCES},
        'protocol': {'box_confidence_strictly_greater': CONFIDENCE, 'matching_iou_inclusive': MATCHING_IOU,
                     'native_nms_iou_strictly_greater_fp32': NMS_IOU, 'postclip_iou_strictly_greater': NMS_IOU,
                     'native_cap': 300, 'proposal_pre_cap': 30000, 'caps_reached': False,
                     'native_score_ties_above_threshold': False, 'coordinate_tolerance_px': COORDINATE_TOLERANCE_PX,
                     'score_tolerance': SCORE_TOLERANCE, 'corner_outcomes_px': [5, 10]},
        'verification': {'both_archive_member_inventories': 'exact', 'recorded_model_and_image_hashes': 'identical',
                         'historical013_replayed_images': len(verification), 'current025_compared_images': len(verification),
                         'per_reference_image_outcomes_compared': sum(row['images'] for row in comparisons),
                         'maximum_coordinate_difference_px': max(row['maximum_coordinate_difference_px'] for row in verification),
                         'maximum_score_difference': max(row['maximum_score_difference'] for row in verification),
                         'images': verification, 'references': comparisons},
        'current_retained_measurement': primary,
        'current_miss_stages': stage_counts,
        'current_misses_by_available_corners': {str(n): sum(c['available_corners'] == n for c in cases) for n in range(3)},
        'eligibility_graph_totals': graph_totals, 'eligibility_graph_by_image': graph_rows,
        'measured_detection_counts_by_stage': detection_totals,
        'historical013_same_reviewed_reference': {k: historical_report['detection'][k] for k in COUNT_KEYS},
        'indices': 'Index fields and anchor_index are zero-based. Row, position, and anchor_number fields are one-based. Anchor identities apply only to this fixed tensor and image.',
        'limitations': LIMITATIONS,
    }
    return summary, {'schema': 'cassette-proposal-stage-cases-r63-v1', 'cases': cases}


def markdown(summary, cases):
    stages = summary['current_miss_stages']
    v = summary['verification']
    lines = ['# Current missed-front proposal stages', '',
        'This CPU replay explains the 18 missed reviewed fronts of the retained method at confidence > 0.25. '
        'It changes no labels, outputs, or model weights.', '',
        '| Stage where IoU-eligible support disappears | Missed fronts |', '|---|---:|']
    lines += [f'| {stage.replace("_", " ")} | {stages[stage]} |' for stage in STAGES]
    lines += ['', 'All 18 missed fronts have incomplete corner references: 13 have one available corner and five have none. '
        'The retained measurement remains 636/654 matched fronts, 51 extra predictions including five duplicates, '
        'and 569/583 complete pairs within five pixels.', '',
        '## Continuity checks', '',
        'Both hash-pinned archives and every inventoried member were verified. The archived model and photograph hashes agree. '
        'All 20 historical 0.13 native outputs, native anchor indices, and postclip outputs replay exactly. '
        'At 0.25, all 20 current native record orders and postclip identities agree; all 60 reference-image comparisons '
        'preserve matching, missed/extra/duplicate identities, and corner threshold outcomes.', '',
        f'The historical tensors were generated on a Tesla T4; the current cache was generated on CPU. '
        f'The maximum coordinate difference is {v["maximum_coordinate_difference_px"]:.12g} pixels, and the maximum box or '
        f'corner score difference is {v["maximum_score_difference"]:.12g}. These are measured agreement checks, not byte identity.', '',
        '## Geometric support and competition', '',
        '| Candidate stage | Candidates | References with support | Maximum one-to-one matches | Candidates eligible for multiple references |',
        '|---|---:|---:|---:|---:|']
    for stage, row in summary['eligibility_graph_totals'].items():
        lines.append(f'| {stage.replace("_", " ")} | {row["candidate_count"]} | {row["references_with_any_candidate"]} | '
                     f'{row["maximum_one_to_one_matches"]} | {row["candidates_eligible_for_multiple_references"]} |')
    lines += ['', 'A reference with one eligible proposal is not necessarily independently recoverable. '
        'The one-to-one bounds account for shared candidates; the actual scorer also retains its original greedy assignment. '
        'Multiple raw proposals are not automatically multiple physical fronts or duplicate errors. '
        'The detailed cases include every above-threshold eligible proposal lost to NMS and its actual suppressor. '
        'The existing extra postclip pass removes six duplicate predictions: native outputs have 57 extras including 11 duplicates; '
        'postclip outputs have 51 extras including five duplicates. Both stages match the same 636 references.', '',
        '## Every current missed front', '',
        '| Photograph / reviewed row | Available corners | Stage | Best decoded IoU | Highest eligible score | Matched at historical 0.13 |',
        '|---|---:|---|---:|---:|---|']
    for case in cases['cases']:
        high = case['highest_confidence_eligible']
        confidence = f'{high["confidence"]:.6f}' if high else 'none'
        lines.append(f'| {case["image"]} / {case["reference_row"]} | {case["available_corners"]} | '
                     f'{case["stage"].replace("_", " ")} | {case["best_decoded"]["target_iou"]:.6f} | {confidence} | '
                     f'{"yes" if case["historical_013_match"] else "no"} |')
    lines += ['', '## Interpretation limits', ''] + ['- ' + value for value in LIMITATIONS]
    return '\n'.join(lines) + '\n'


def write_outputs(output, summary, cases):
    payloads = {'summary.json': encoded(summary), 'cases.json': encoded(cases),
                'results.md': markdown(summary, cases).encode()}
    manifest = {'schema': 'cassette-proposal-stage-artifacts-r63-v1',
                'files': {name: {'bytes': len(data), 'sha256': digest(data)} for name, data in payloads.items()},
                'analysis_sources_sha256': summary['analysis_sources_sha256']}
    payloads['manifest.json'] = encoded(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        path = output / name
        require(not path.exists() or path.is_file() and path.read_bytes() == data, 'Different existing output preserved: ' + str(path))
        if not path.exists():
            path.write_bytes(data)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--r49-archive', type=Path, required=True)
    parser.add_argument('--reference-archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary, cases = analyze(args.r49_archive, args.reference_archive or args.root / REFERENCE_PATH, args.root)
    manifest = write_outputs(args.output, summary, cases)
    print(json.dumps({'status': summary['status'], 'current_miss_stages': summary['current_miss_stages'],
                      'files': manifest['files']}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
