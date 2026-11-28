#!/usr/bin/env python3
"""Describe missed-front box geometry without changing predictions or matching."""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
from audit_reflection_return import digest, encoded, require
from evaluate_pose import parse_pose_labels
from pose_scoring import _box_iou, score_image
from run_original_comparison_r59 import PROTOCOL, assemble
from run_reference_matrix_r60 import REFERENCE_NAMES, verified_zip

ARCHIVE_PATH = 'docs/thesis/evidence/annotation-reference/baomai-annotation-reference-r60-results.zip'
ARCHIVE_SHA256 = 'ce8df4bcf223b6bee49fa67e829a443acb572e4a6aad75fa0f92657a16cc2601'
ARCHIVE_BYTES = 972658
METHODS = ('original', 'proposed')
DIMENSIONS = ('width_px', 'height_px', 'short_side_px', 'sqrt_area_px', 'width_over_height')
DEPENDENCIES = (
    'audit_reflection_return.py', 'evaluate_pose.py', 'pose_checkpoint.py',
    'pose_scoring.py', 'predict_pose_folder.py', 'run_corner_average_r58.py',
    'run_flip_inference_r56.py', 'run_original_comparison_r59.py',
    'run_reference_matrix_r60.py',
)


def decoded(members, name):
    return json.loads(gzip.decompress(members[name + '.json.gz']))


def box_geometry(box):
    require(len(box) == 4 and all(math.isfinite(v) for v in box), 'Non-finite box')
    width, height = box[2] - box[0], box[3] - box[1]
    require(width > 0 and height > 0, 'Non-positive box extent')
    return {'width_px': width, 'height_px': height, 'short_side_px': min(width, height),
            'sqrt_area_px': math.sqrt(width * height), 'width_over_height': width / height}


def statistics_for(values):
    ordered = sorted(values)
    if not ordered:
        return {'n': 0}
    def percentile(fraction):
        index = (len(ordered) - 1) * fraction
        lower = math.floor(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    return {'n': len(ordered), 'min': ordered[0], 'p10': percentile(.1), 'p25': percentile(.25),
            'median': statistics.median(ordered), 'p75': percentile(.75), 'p90': percentile(.9),
            'max': ordered[-1]}


def geometry_group(rows):
    return {'n': len(rows), **{name: statistics_for([r['reference_geometry'][name] for r in rows])
                              for name in DIMENSIONS}}


def candidate_geometry(reference_box, prediction):
    box = prediction['bbox_xyxy']
    target, actual = box_geometry(reference_box), box_geometry(box)
    edge = {name: box[i] - reference_box[i]
            for i, name in enumerate(('left_dx', 'top_dy', 'right_dx', 'bottom_dy'))}
    dx = (edge['left_dx'] + edge['right_dx']) / 2
    dy = (edge['top_dy'] + edge['bottom_dy']) / 2
    dw = actual['width_px'] - target['width_px']
    dh = actual['height_px'] - target['height_px']
    distance_squared = dx * dx + dy * dy + (dw * dw + dh * dh) / 4
    require(math.isclose(distance_squared, sum(v * v for v in edge.values()) / 2,
                         rel_tol=1e-12, abs_tol=1e-9), 'Independent box-distance identity differs')
    return {'prediction_geometry': actual, 'signed_edge_deltas_px': edge,
            'center_delta_px': {'x': dx, 'y': dy},
            'width_ratio': actual['width_px'] / target['width_px'],
            'height_ratio': actual['height_px'] / target['height_px'],
            'normalized_edge_deltas': {
                'left_over_reference_width': edge['left_dx'] / target['width_px'],
                'right_over_reference_width': edge['right_dx'] / target['width_px'],
                'top_over_reference_height': edge['top_dy'] / target['height_px'],
                'bottom_over_reference_height': edge['bottom_dy'] / target['height_px']},
            'gaussian_wasserstein_distance_px': math.sqrt(distance_squared)}


def analyze(root):
    root = Path(root).resolve()
    members = verified_zip(root / ARCHIVE_PATH, ARCHIVE_SHA256, ARCHIVE_BYTES)
    saved = json.loads(members['summary.json'])
    require(saved['status'] == 'completed' and saved['contract']['protocol'] == PROTOCOL,
            'Frozen analysis or protocol differs')
    source_hashes = json.loads(members['preflight.json'])['source']['files_sha256']
    dependencies = {}
    for name in DEPENDENCIES:
        relative = 'scripts/' + name
        dependencies[relative] = digest((root / relative).read_bytes())
        require(dependencies[relative] == source_hashes[relative], 'Frozen helper differs: ' + relative)
    predictions, suppression, associations = assemble(decoded(members, 'native-predictions'))
    for name, value in (('predictions', predictions), ('suppression', suppression), ('associations', associations)):
        require(value == decoded(members, name), 'Frozen prediction replay differs: ' + name)
    label_text = decoded(members, 'reference-labels')
    saved_scores = decoded(members, 'per-image')
    summary = {'schema': 'cassette-box-geometry-r63-v1', 'status': 'completed',
        'development_only': True, 'fresh_inference': False, 'training': False,
        'predictions_modified': False, 'labels_modified': False, 'automatic_promotion': False,
        'source_archive': {'path': ARCHIVE_PATH, 'sha256': ARCHIVE_SHA256, 'bytes': ARCHIVE_BYTES},
        'source_dependencies_sha256': dependencies, 'driver_sha256': digest(Path(__file__).read_bytes()),
        'protocol': PROTOCOL, 'methods': list(METHODS),
        'diagnostic_definition': {
            'candidate': 'Highest same-class reference IoU among retained predictions; lower prediction index breaks ties.',
            'zero_overlap': 'No candidate is selected when every retained prediction has zero overlap.',
            'candidate_reuse': 'Diagnostic candidates may be reused or already assigned to another reference; this is not the official one-to-one matcher.',
            'near_overlap_miss': 'An officially unmatched reference with 0.30 <= maximum retained-prediction IoU < 0.50.',
            'ratio_flags': 'Width or height ratio strictly above 1.50 is descriptive only; continuous values are preserved.',
            'signed_edges': 'Prediction minus reference, in original 1280x720 image pixels; x rightward and y downward.',
            'gaussian_box_encoding': 'Mean (center_x, center_y), diagonal covariance (width^2/4, height^2/4).',
            'gaussian_wasserstein_distance': 'sqrt(dx_center^2 + dy_center^2 + delta_width^2/4 + delta_height^2/4).',
            'distance_scope': 'Unnormalized descriptive distance; no NWD normalization constant, training objective or evaluation replacement is selected.',
            'percentiles': 'Linear interpolation at (n-1)*p.',
            'physical_identity': 'Overlap does not establish that the candidate depicts the same physical front.'},
        'references': {}, 'verification': {'canonical_predictions_exactly_reconstructed': True,
            'per_image_score_tables_exactly_reproduced': 0, 'reference_targets_analyzed_per_method': 0,
            'missed_case_records': 0, 'independent_distance_identity_checked': True}}
    cases = {}
    for reference in REFERENCE_NAMES:
        descriptor = saved['contract']['references'][reference]
        gold = {}
        for image, text in sorted(label_text[reference].items()):
            require(digest(text.encode()) == descriptor['labels'][image]['sha256'], 'Reference label differs')
            gold[image] = parse_pose_labels(text, 1280, 720, reference + '/' + image)
        require(set(gold) == set(saved['contract']['photographs']) and
                sum(map(len, gold.values())) == descriptor['objects'], 'Reference membership differs')
        summary['verification']['reference_targets_analyzed_per_method'] += descriptor['objects']
        summary['references'][reference] = {'fronts': descriptor['objects'],
            'manifest_sha256': descriptor['manifest_sha256'], 'methods': {}}
        cases[reference] = {}
        for method in METHODS:
            rows = []
            for image, targets in gold.items():
                outputs = predictions[method][image]
                score = score_image(targets, outputs, 1280, 720, .5, .5)
                require(score == saved_scores[reference][method][image], 'Per-image score replay differs')
                summary['verification']['per_image_score_tables_exactly_reproduced'] += 1
                assigned = {m['prediction_index']: m['gold_index'] for m in score['matches']}
                missed = set(score['unmatched_gold_indices'])
                for index, target in enumerate(targets):
                    row = {'case_id': f'{reference}/{image}/R{index+1:03d}', 'image': image,
                        'reference_index': index, 'locatable_corners': sum(p[2] == 2 for p in target['keypoints']),
                        'reference_box_xyxy': target['bbox_xyxy'],
                        'reference_geometry': box_geometry(target['bbox_xyxy']), 'matched': index not in missed}
                    if index in missed:
                        ranked = [(_box_iou(target['bbox_xyxy'], p['bbox_xyxy']), i)
                                  for i, p in enumerate(outputs) if p['class_id'] == target['class_id']]
                        iou, prediction_index = max(ranked, key=lambda item: (item[0], -item[1])) if ranked else (0., None)
                        row['maximum_retained_iou'] = iou
                        row['near_overlap_miss'] = .3 <= iou < .5
                        row['candidate'] = None
                        if iou > 0:
                            prediction = outputs[prediction_index]
                            owner = assigned.get(prediction_index)
                            row['candidate'] = {'prediction_index': prediction_index,
                                'primary_prediction_id': prediction.get('primary_prediction_id'),
                                'confidence': prediction['confidence'], 'bbox_xyxy': prediction['bbox_xyxy'],
                                'assigned_reference_index': owner,
                                'assigned_reference_case': None if owner is None else f'{reference}/{image}/R{owner+1:03d}',
                                **candidate_geometry(target['bbox_xyxy'], prediction)}
                    rows.append(row)
            missed_rows = [r for r in rows if not r['matched']]
            near = [r for r in missed_rows if r['near_overlap_miss']]
            partial = [r for r in rows if r['locatable_corners'] < 2]
            groups = {'all': rows, 'complete': [r for r in rows if r['locatable_corners'] == 2],
                      'incomplete_matched': [r for r in partial if r['matched']],
                      'incomplete_missed': [r for r in partial if not r['matched']],
                      'incomplete_near_overlap_missed': [r for r in near if r['locatable_corners'] < 2]}
            summary['references'][reference]['methods'][method] = {
                'matched': len(rows) - len(missed_rows), 'missed': len(missed_rows),
                'geometry_groups': {name: geometry_group(value) for name, value in groups.items()},
                'corner_availability_groups': {str(n): {status: geometry_group([r for r in rows
                    if r['locatable_corners'] == n and (status == 'all' or r['matched'] == (status == 'matched'))])
                    for status in ('all', 'matched', 'missed')} for n in (0, 1, 2)},
                'near_overlap_misses': {'n': len(near), 'case_ids': [r['case_id'] for r in near],
                    'locatable_corner_counts': dict(Counter(str(r['locatable_corners']) for r in near)),
                    'candidate_already_matched_count': sum(r['candidate']['assigned_reference_index'] is not None for r in near),
                    'width_ratio_above_1_5': sum(r['candidate']['width_ratio'] > 1.5 for r in near),
                    'height_ratio_above_1_5': sum(r['candidate']['height_ratio'] > 1.5 for r in near),
                    'width_ratio': statistics_for([r['candidate']['width_ratio'] for r in near]),
                    'height_ratio': statistics_for([r['candidate']['height_ratio'] for r in near]),
                    'gaussian_wasserstein_distance_px': statistics_for([r['candidate']['gaussian_wasserstein_distance_px'] for r in near])}}
            cases[reference][method] = missed_rows
            summary['verification']['missed_case_records'] += len(missed_rows)
    require(summary['verification']['per_image_score_tables_exactly_reproduced'] == 120,
            'Expected all three references, both methods and twenty images')
    return summary, cases


def markdown_result(summary, cases):
    lines = ['# Box geometry diagnostic', '',
        'Saved predictions and all three annotation references are unchanged. These are descriptive development-data measurements.', '',
        '| Reference | Method | Missed | Near-overlap misses | Candidate width >1.5x | Candidate height >1.5x |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for reference in REFERENCE_NAMES:
        for method in METHODS:
            values = summary['references'][reference]['methods'][method]
            near = values['near_overlap_misses']
            lines.append(f"| {reference} | {method} | {values['missed']} | {near['n']} | {near['width_ratio_above_1_5']} | {near['height_ratio_above_1_5']} |")
    lines.extend(['', '## Reviewed-reference geometry of the retained method', '',
        '| Group | Fronts | Median width (px) | Median height (px) | Median square-root area (px) |',
        '| --- | ---: | ---: | ---: | ---: |'])
    groups = summary['references']['r50']['methods']['proposed']['geometry_groups']
    for name, values in groups.items():
        numbers = [f"{values[k]['median']:.3f}" if values['n'] else 'undefined'
                   for k in ('width_px', 'height_px', 'sqrt_area_px')]
        lines.append(f"| {name} | {values['n']} | " + ' | '.join(numbers) + ' |')
    lines.extend(['', '## The ten near-overlap misses under the reviewed reference', '',
        '| Case | Reference width x height (px) | Max IoU | Width ratio | Height ratio | Signed left / top / right / bottom errors (px) | W2 distance (px) |',
        '| --- | --- | ---: | ---: | ---: | --- | ---: |'])
    for row in cases['r50']['proposed']:
        if not row['near_overlap_miss']:
            continue
        geometry, candidate = row['reference_geometry'], row['candidate']
        edge = candidate['signed_edge_deltas_px']
        errors = ' / '.join(f'{edge[k]:+.2f}' for k in ('left_dx', 'top_dy', 'right_dx', 'bottom_dy'))
        lines.append(f"| {row['case_id']} | {geometry['width_px']:.2f} x {geometry['height_px']:.2f} | {row['maximum_retained_iou']:.4f} | {candidate['width_ratio']:.3f} | {candidate['height_ratio']:.3f} | {errors} | {candidate['gaussian_wasserstein_distance_px']:.3f} |")
    lines.extend(['', '## Interpretation', '',
        'A near-overlap miss is an unmatched reference whose highest retained-prediction IoU is at least 0.30 and below 0.50. The diagnostic does not change the official one-to-one assignment. It may reuse a prediction or select a box already assigned to a different reference; cases.json records that status.', '',
        'The overlap-selected candidate is not verified as the same physical front. All-zero-overlap cases have no selected candidate. Width/height ratios and signed edge errors indicate what to inspect, not which label or prediction should be changed.', '',
        'The pixel distance is the 2-Wasserstein distance between Gaussian box representations with mean (center_x, center_y) and diagonal covariance (width^2/4, height^2/4). No normalization constant is selected and no distance replaces IoU in evaluation. A narrow visible fragment is not necessarily a tiny full object.', '',
        'These measurements motivate an audit of visible extent, occlusion ownership and training augmentation before another loss-function experiment. They do not establish the training-time cause or a model improvement.', ''])
    return '\n'.join(lines).encode()


def write_result(output, summary, cases):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = {'summary.json': encoded(summary), 'cases.json': encoded(cases),
               'results.md': markdown_result(summary, cases)}
    payload['manifest.json'] = encoded({name: {'bytes': len(data), 'sha256': digest(data)}
                                       for name, data in payload.items()})
    for name, data in payload.items():
        path = output / name
        require(not path.exists() or path.read_bytes() == data, 'Different existing output preserved: ' + str(path))
        path.write_bytes(data)
    return sorted(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary, cases = analyze(args.root)
    files = write_result(args.output, summary, cases)
    print(json.dumps({'status': 'completed', 'files': files, 'verification': summary['verification']}, sort_keys=True))


if __name__ == '__main__':
    main()
