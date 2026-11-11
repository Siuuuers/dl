#!/usr/bin/env python3
"""Review the confirmation report against independently recomputed evidence.

No production scoring or matching code is imported. This review reads artifacts
only; it does not load models, rerun inference, train, or modify the repository.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from independent_annotation_score import image_result, load_reference, load_revised, overlap


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def condensed(view):
    score, quality = view['score'], view['quality']
    radii = {r['radius_px']: r for r in quality['radii']}
    return {
        'missed_objects': score['fn'], 'extra_detections': score['fp'],
        'missing_visible_endpoints': score['missing_endpoints'],
        'conditional_median_error_px': score['conditional_median_error_px'],
        'conditional_p95_error_px': score['conditional_p95_error_px'],
        'both_corners_within5px': radii[5]['both_visible_objects_within_radius'],
        'both_corners_within10px': radii[10]['both_visible_objects_within_radius'],
        'both_corner_eligible_objects': quality['both_visible_objects'],
        'objects_by_locatable_corners': view['objects_by_locatable_corners'],
        'misses_by_locatable_corners': view['misses_by_locatable_corners'],
        'returned_on_unlocatable_sides': quality['returned_on_unlocatable_reference'],
        'unlocatable_sides_on_matched_objects': quality['unlocatable_on_matched_objects'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.workspace.resolve()
    own = base / 'seed46-work/independent'
    source = base / 'dl'
    paths = {
        'report_markdown': base / 'dl-confirmation/docs/annotation-confirmation-2026-11-11.md',
        'report_json': base / 'dl-confirmation/docs/annotation-confirmation-2026-11-11.json',
        'seed45_audit': own / 'seed45-independent-numeric-audit.json',
        'seed46_audit': own / 'seed46-numerical-audit.json',
        'seed45_summary': own / 'seed45-comparison.json',
        'seed46_summary': base / 'seed46-work/evidence/seed-46/comparison-original-and-pr30.json',
        'case_details': own / 'seed46-case-details.json',
        'review_queue': base / 'seed46-work/review-queue/review-queue.json',
        'published_queue': base / 'dl-confirmation/results/evaluation/baomai-annotations-seed46-20261111/review-queue.json',
        'reference_manifest': source / 'data/references/visible-front-2026-11-10/manifest.json',
        'alternate_proposals': base / 'seed46-work/alternate-proposal-diagnostics.json',
        'baseline_continuity': base / 'seed46-work/baseline-continuity.json',
        'baseline_revised_continuity': base / 'seed46-work/baseline-revised-continuity.json',
        'notebook_protocol_review': base / 'seed46-work/audit/notebook-protocol-review.json',
    }
    report = read(paths['report_json'])
    markdown = paths['report_markdown'].read_text()
    audits = {seed: read(paths[f'seed{seed}_audit']) for seed in (45, 46)}
    summaries = {seed: read(paths[f'seed{seed}_summary']) for seed in (45, 46)}
    details = read(paths['case_details'])
    queue = read(paths['review_queue'])
    manifest = read(paths['reference_manifest'])
    failures = []
    scalar_count = 0
    maximum_float_difference = 0.0

    def compare(actual, expected, context):
        nonlocal scalar_count, maximum_float_difference
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                failures.append({'field': context, 'error': 'expected dict'}); return
            for key, value in expected.items():
                if key not in actual:
                    failures.append({'field': context + '/' + key, 'error': 'missing'}); continue
                compare(actual[key], value, context + '/' + key)
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                failures.append({'field': context, 'error': 'list length/type differs'}); return
            for index, value in enumerate(expected):
                compare(actual[index], value, context + '/' + str(index))
        else:
            scalar_count += 1
            if type(expected) is float and type(actual) in (float, int):
                delta = abs(actual - expected)
                maximum_float_difference = max(maximum_float_difference, delta)
                equal = math.isfinite(actual) and delta <= 1e-10
            else:
                equal = actual == expected and type(actual) is type(expected)
            if not equal:
                failures.append({'field': context, 'actual': actual, 'expected': expected})

    for seed_view in report['per_seed']:
        seed = seed_view['seed']
        audit, summary = audits[seed], summaries[seed]
        expected = {(m['model_id'], m['box_confidence']): m for m in audit['metrics']}
        compare(len(seed_view['metrics']), 9, f'seed{seed}/metric_count')
        for metric in seed_view['metrics']:
            key = (metric['model_id'], metric['box_confidence'])
            for ref in ('original', 'revised'):
                compare(metric[ref], condensed(expected[key][ref]), f'seed{seed}/{key}/{ref}')
            compare(metric['primary'], metric['box_confidence'] == 0.25, f'seed{seed}/{key}/primary')
        for key in ('recipe_id', 'recipe_sha256', 'training_runtime'):
            compare(seed_view[key], summary['experiment'][key], f'seed{seed}/{key}')
        compare(seed_view['inference_runtime'], summary['inference']['runtime'], f'seed{seed}/inference_runtime')
        compare(report['complete_summary_sha256'][f'seed{seed}'], sha(paths[f'seed{seed}_summary']), f'seed{seed}/summary_hash')
    for row in report['primary_candidate_minus_control']:
        seed, ref = row['seed'], row['reference']
        old = condensed(audits[seed]['primary'][ref]['m1'])
        new = condensed(audits[seed]['primary'][ref]['m2'])
        for key, value in row['delta'].items():
            compare(value, new[key] - old[key], f'primary_delta/{seed}/{ref}/{key}')
    compare(report['seed46_common_returned_endpoints'], audits[46]['common_returned_endpoints'], 'seed46_common_returned_endpoints')
    compare(report['seed46_source'], summaries[46]['source'], 'seed46_source')
    compare(report['seed46_models'], summaries[46]['models'], 'seed46_models')
    compare(report['candidate_manifest_sha256'], summaries[46]['experiment']['candidate_manifest_sha256'], 'candidate_manifest_sha256')
    for ref, denominators in audits[46]['reference_denominators'].items():
        compare(report['references'][ref], {
            'images': 20, 'objects': denominators['objects'],
            'visible_endpoints': denominators['visible_endpoints'],
            'both_corner_eligible_objects': denominators['both_visible_objects']}, 'reference/' + ref)
    compare(report['each_seed_reported_separately'], True, 'each_seed_reported_separately')
    compare(report['pooled_independent_observations_claimed'], False, 'pooled_independent_observations_claimed')
    compare(report['fixed_box_confidences'], [0.15, 0.25, 0.4], 'fixed_box_confidences')
    compare(report['primary_box_confidence'], 0.25, 'primary_box_confidence')

    original = load_reference(source)
    revised, _ = load_revised(source, original, source / 'data/references/visible-front-2026-11-10')
    misses = {}
    for variant, meta in queue['variants'].items():
        identity = audits[meta['summary_seed']]['primary_object_match_identities'][meta['model_id']]['revised']
        misses[variant] = {(image, revised[image]['objects'][i]['line'])
                           for image, view in identity.items() for i in view['missed_gold_indices']}
    rowmaps = {f['image_id']: {row: index + 1 for index, row in enumerate(f['retained_source_to_revised_row_one_based']) if row is not None}
               for f in manifest['files']}
    dispositions = {(r['image_id'], r['source_row_one_based']): r['disposition'] for r in manifest['queue_dispositions']}
    current = []
    for row in queue['rows']:
        key = row['image_id'], row['reference_row_one_based']
        compare(row['original_row_one_based'], rowmaps[key[0]][key[1]], f'queue/{key}/row_mapping')
        expected_flags = {v: key in missset for v, missset in misses.items()}
        compare(row['miss_flags'], expected_flags, f'queue/{key}/miss_flags')
        compare(row['persistent_across_all_five_variants'], all(expected_flags.values()), f'queue/{key}/persistent')
        compare(row['previous_queue_disposition'], dispositions.get((key[0], row['original_row_one_based']), 'not_in_previous_33_row_queue'), f'queue/{key}/disposition')
        if 'current_primary_candidate_miss' in row['roles']:
            current.append(row)
    current_ids = {(r['image_id'], r['reference_row_one_based']) for r in current}
    compare(sorted(current_ids), sorted(misses['seed46_candidate']), 'queue/all_current_misses')
    compare(len(queue['rows']), 26, 'queue/unique_rows')
    compare(len({(r['image_id'], r['reference_row_one_based']) for r in queue['rows']}), 26, 'queue/unique_object_identities')
    persistent = sum(r['persistent_across_all_five_variants'] for r in current)
    disposition_counts = dict(Counter(r['previous_queue_disposition'] for r in current))
    unresolved_persistent = sum(r['previous_queue_disposition'] == 'retained_unresolved' and r['persistent_across_all_five_variants'] for r in current)
    compare(persistent, 21, 'queue/persistent_current_misses')
    compare(disposition_counts, {'retained_unresolved': 19, 'amended': 2, 'not_in_previous_33_row_queue': 3}, 'queue/dispositions')
    compare(unresolved_persistent, 18, 'queue/unresolved_persistent')
    compare(sha(paths['published_queue']), sha(paths['review_queue']), 'queue/published_copy_hash')

    raw = [json.loads(line) for line in (base / 'seed46-work/evidence/seed-46/m2/predictions.jsonl').read_text().splitlines()][1:]
    raw = {(r['image_id'], r['threshold']): r['predictions'] for r in raw}
    direct_proposals = []
    for image, row, prediction_index in (('new_000007', 15, 52), ('new_000007', 15, 53), ('new_000087', 15, 14)):
        record = revised[image]
        gi = next(i for i, obj in enumerate(record['objects']) if obj['line'] == row)
        obj = record['objects'][gi]
        predictions = raw[image, 0.25]
        prediction = predictions[prediction_index]
        result = image_result(record, predictions)[0]
        assigned = [record['objects'][m['gold_index']]['line'] for m in result['matches'] if m['prediction_index'] == prediction_index]
        distances = [math.hypot(p[0] - g[0], p[1] - g[1]) for p, g in zip(prediction['keypoints'], obj['points'])]
        direct_proposals.append({'image_id': image, 'row_one_based': row, 'prediction_index': prediction_index,
            'confidence': prediction['confidence'], 'reference_iou': overlap(prediction['bbox_xyxy'], obj['box']),
            'diagnostic_corner_distances_px': distances, 'matched_to_rows': assigned,
            'official_unmatched_extra': prediction_index in result['unmatched_prediction_indices']})
    compare(direct_proposals[0]['matched_to_rows'], [15], '007/prediction52/assignment')
    compare(direct_proposals[1]['official_unmatched_extra'], True, '007/prediction53/unmatched_extra')
    compare(direct_proposals[1]['diagnostic_corner_distances_px'], [4.757171150983218, 3.9280074779108904], '007/prediction53/distances')
    compare(direct_proposals[1]['reference_iou'], 0.8508087177205839, '007/prediction53/iou')
    compare(direct_proposals[2]['diagnostic_corner_distances_px'], [3.162419332863515, 1.1790939991450617], '087/prediction14/distances')
    compare(direct_proposals[2]['reference_iou'], 0.49306877590677217, '087/prediction14/iou')
    compare(direct_proposals[2]['matched_to_rows'], [], '087/prediction14/unmatched')

    primary = audits[46]['primary']['revised']
    two_px = {m: next(r['both_visible_objects_within_radius'] for r in primary[m]['quality']['radii'] if r['radius_px'] == 2) for m in ('m0', 'm2')}
    compare(two_px, {'m0': 265, 'm2': 261}, 'baseline/2px_both_corner_success')
    baseline_common = audits[46]['common_returned_endpoints']['revised']['paired']['m2_vs_m0']
    compare(baseline_common['error_change_px']['mean'], 0.004167705498629524, 'baseline/common_mean_error_change')
    local_extra_change = next(r for r in details['object_match_transitions'] if r['reference'] == 'revised' and r['confidence'] == 0.25 and r['comparison'] == 'm2_vs_m0')
    compare(next(r['delta'] for r in local_extra_change['per_image_extra_changes'] if r['image_id'] == 'new_000089'), 4, '089/baseline_extra_change')
    empty = sorted(image for image, record in revised.items() if not record['objects'])
    compare(len(empty), 3, 'empty_images/count')
    for m in ('m0', 'm1', 'm2'):
        per_image = [r for r in summaries[46]['per_image'] if r['model_id'] == m and r['box_confidence'] == 0.25 and r['image_id'] in empty]
        compare([r['revised']['score']['fp'] for r in per_image], [0, 0, 0], f'empty_images/{m}/extras')

    small_corrections = []
    if "public detection/quality results remain unchanged; error quantiles differ slightly" in markdown:
        small_corrections.append({
            'id': 'm0-continuity-wording', 'severity': 'small_clarity_correction',
            'location': 'Return and execution verification / fresh M0 continuity paragraph',
            'issue': 'The phrase public detection/quality results is broader than intended because error quantiles are public score fields and do change slightly.',
            'suggestion': 'Specify detection counts, original match identities, endpoint availability and radius-success counts as unchanged, then state that error quantiles differ slightly.',
            'blocks_numerical_conclusion': False,
        })
    result = {
        'schema_version': 1, 'kind': 'independent-seed46-report-review',
        'status': 'pass_with_small_clarity_correction' if small_corrections and not failures else ('passed' if not failures else 'failed'),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': {key: {'path': str(path), 'sha256': sha(path)} for key, path in paths.items()},
        'scope': ['All 36 compact model/threshold/reference aggregates across the two seeds',
                  'Four primary candidate-control delta records and all seed46 common-returned diagnostics',
                  'Reference denominators, checkpoint/source/runtime bindings, and fixed operating points',
                  'All 26 queue row mappings/dispositions and all 130 five-variant miss flags',
                  'Independent raw-prediction recomputation of 007 competing proposals and 087 unmatched geometry',
                  'Manual report review against independent match, endpoint, and hidden-side transitions'],
        'scalar_comparisons': scalar_count, 'maximum_float_difference': maximum_float_difference,
        'substantive_blockers': failures, 'small_corrections': small_corrections,
        'queue_verified': {'objects': 26, 'current_candidate_misses': len(current), 'persistent_across_five': persistent,
            'dispositions': disposition_counts, 'unresolved_persistent': unresolved_persistent,
            'first_nonunresolved_current_misses': [r['object_id'] for r in current if r['previous_queue_disposition'] != 'retained_unresolved']},
        'direct_proposal_verification': direct_proposals,
        'causal_review': {
            'label_pair_vs_additional_training_distinction_preserved': True,
            'favorable_primary_candidate_vs_baseline_results_preserved': True,
            'original_and_revised_reference_effects_kept_separate': True,
            'all584_eligible_denominator_including_misses_preserved': True,
            '007_alternate_proposal_does_not_replace_official_match': True,
            '087_nearby_corner_distances_labeled_unmatched_diagnostics': True,
            'hidden_side_membership_change_not_called_coordinate_improvement': True,
            'training_A100_distinguished_from_reported_later_L4_packaging': True,
            'cross_seed_hardware_confound_and_nonindependence_disclosed': True,
            'no_unseen_scene_accuracy_or_physical10px_tolerance_claimed': True,
        },
        'outside_scope': ['Capsule packaging, extraction/replay and payload readback claims; packaging was still in progress at review time.',
                          'No new visual label adjudication; original-pixel review is separately recorded by the parent reviewer.',
                          'No independent remote GPU/Drive attestation; returned records and source-review interpretation only.'],
        'script_sha256': sha(Path(__file__)),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'scalar_comparisons', 'maximum_float_difference', 'substantive_blockers', 'small_corrections', 'queue_verified', 'direct_proposal_verification')}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
