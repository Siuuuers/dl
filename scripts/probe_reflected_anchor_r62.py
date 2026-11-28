#!/usr/bin/env python3
"""Replay one fixed reflected-anchor corner average from frozen predictions."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
from audit_reflection_return import COUNT_KEYS, digest, encoded, require
from evaluate_pose import parse_pose_labels
from run_corner_average_r58 import accepted_report, average_predictions, report
from run_original_comparison_r59 import PROTOCOL, assemble, measure
from run_reference_matrix_r60 import REFERENCE_NAMES, verified_zip
from verify_corner_average_independent import independent_average

ARCHIVE_PATH = 'docs/thesis/evidence/annotation-reference/baomai-annotation-reference-r60-results.zip'
ARCHIVE_SHA256 = 'ce8df4bcf223b6bee49fa67e829a443acb572e4a6aad75fa0f92657a16cc2601'
ARCHIVE_BYTES = 972658
METHODS = ('original', 'proposed', 'reflected', 'reflected_anchor')
DEPENDENCIES = (
    'audit_reflection_return.py', 'evaluate_pose.py', 'pose_checkpoint.py',
    'pose_scoring.py', 'predict_pose_folder.py', 'run_corner_average_r58.py',
    'run_flip_inference_r56.py', 'run_original_comparison_r59.py',
    'run_reference_matrix_r60.py', 'verify_corner_average_independent.py',
)


def decoded(members, name):
    return json.loads(gzip.decompress(members[name + '.json.gz']))


def changes(before, after):
    require([(r['image'], r['reference_index']) for r in before] ==
            [(r['image'], r['reference_index']) for r in after], 'Paired target order differs')
    result = {}
    for name, key in (('detection', 'detected'), ('5', 'within5'), ('10', 'within10')):
        gained, lost = [], []
        for first, second in zip(before, after):
            if name != 'detection' and not first['complete_reference']:
                continue
            identity = first['image'] + '/R' + str(first['reference_index'] + 1).zfill(3)
            if not first[key] and second[key]:
                gained.append(identity)
            if first[key] and not second[key]:
                lost.append(identity)
        result[name] = {'gained': len(gained), 'lost': len(lost),
                        'gained_ids': gained, 'lost_ids': lost}
    return result


def association_audit(gold, per_image, associations, arms):
    all_links, complete_links, details = Counter(), Counter(), []
    for image, rows in sorted(associations.items()):
        assigned = {view: {m['prediction_index']: m['gold_index']
                           for m in per_image[view][image]['matches']}
                    for view in ('reflected', 'trained')}
        for row in rows:
            if row['status'] != 'averaged':
                continue
            anchor, partner = row['original_index'], row['reflected_index']
            first, second = assigned['reflected'].get(anchor), assigned['trained'].get(partner)
            category = ('both_unassigned' if first is None and second is None else
                        'anchor_unassigned' if first is None else
                        'partner_unassigned' if second is None else
                        'same_target' if first == second else 'different_target')
            all_links[category] += 1
            complete = first is not None and all(p[2] == 2 for p in gold[image][first]['keypoints'])
            if complete:
                complete_links[category] += 1
            details.append({'image': image, 'anchor_index': anchor, 'partner_index': partner,
                'anchor_prediction_id': arms['reflected'][image][anchor].get('primary_prediction_id'),
                'partner_prediction_id': arms['trained'][image][partner].get('primary_prediction_id'),
                'anchor_reference_index': first, 'partner_reference_index': second,
                'category': category, 'anchor_complete_reference': complete, 'box_iou': row['box_iou']})
    return {'all_averaged_links': dict(all_links),
            'complete_anchor_links': dict(complete_links),
            'different_target_links': [r for r in details if r['category'] == 'different_target']}, details


def analyze(archive, root=None):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    members = verified_zip(archive, ARCHIVE_SHA256, ARCHIVE_BYTES)
    saved = json.loads(members['summary.json'])
    require(saved['status'] == 'completed' and saved['development_only'] and
            not saved['fresh_inference'] and not saved['training'] and not saved['labels_modified'],
            'Frozen reference analysis status differs')
    require(saved['contract']['protocol'] == PROTOCOL, 'Published protocol differs')
    preflight = json.loads(members['preflight.json'])
    source_hashes = preflight['source']['files_sha256']
    dependencies = {}
    for name in DEPENDENCIES:
        relative = 'scripts/' + name
        dependencies[relative] = digest((root / relative).read_bytes())
        require(dependencies[relative] == source_hashes[relative], 'Frozen helper differs: ' + relative)

    native = decoded(members, 'native-predictions')
    arms, suppression, canonical_associations = assemble(native)
    for name, actual in (('predictions', arms), ('suppression', suppression),
                         ('associations', canonical_associations)):
        require(actual == decoded(members, name), 'Canonical prediction replay differs: ' + name)

    probe, associations = {}, {}
    for image in sorted(arms['reflected']):
        probe[image], associations[image] = average_predictions(arms['reflected'][image], arms['trained'][image])
    independent_input = {'predictions': {
        'control': {i: {'postclip': p} for i, p in arms['reflected'].items()},
        'treatment': {i: {'postclip': p} for i, p in arms['trained'].items()}}}
    independent, independent_links, independent_rows = independent_average(independent_input)
    require(independent == probe, 'Independent coordinate average differs')
    links = {(image, r['original_index'], r['reflected_index'])
             for image, rows in associations.items() for r in rows if r['status'] == 'averaged'}
    require(links == {(image, a, b) for image, rows in independent_links.items() for a, b in rows},
            'Independent association differs')
    reverse_links = {(image, r['reflected_index'], r['original_index'])
                     for image, rows in canonical_associations.items() for r in rows if r['status'] == 'averaged'}
    require(links == reverse_links, 'Mutual associations are not symmetric under anchor reversal')
    for image, anchor, partner in links:
        require([p[:2] for p in probe[image][anchor]['keypoints']] ==
                [p[:2] for p in arms['proposed'][image][partner]['keypoints']],
                'Linked equal-weight coordinates differ under anchor reversal')

    label_text = decoded(members, 'reference-labels')
    references = {}
    require(set(label_text) == set(REFERENCE_NAMES), 'Reference membership differs')
    for reference in REFERENCE_NAMES:
        descriptor = saved['contract']['references'][reference]
        references[reference] = {}
        for image, text in label_text[reference].items():
            require(digest(text.encode()) == descriptor['labels'][image]['sha256'], 'Label identity differs')
            references[reference][image] = parse_pose_labels(text, 1280, 720, reference + '/' + image)
        require(set(references[reference]) == set(saved['contract']['photographs']) and
                sum(map(len, references[reference].values())) == descriptor['objects'],
                'Reference image or object membership differs')

    measured, canonical = measure(references, native)
    for name, actual in canonical.items():
        require(actual == decoded(members, name), 'Canonical score reconstruction differs: ' + name)
    for row in saved['matrix']:
        result = measured['references'][row['reference']][row['method']]
        expected = {'fronts': result['detection']['gold_count'], 'outputs': result['detection']['prediction_count'],
            'matched': result['detection']['tp'], 'missed': result['detection']['fn'],
            'extras': result['detection']['fp'], 'duplicates': result['detection']['duplicate_fp'],
            'eligible_pairs': result['pairs']['eligible'], 'within5': result['pairs']['within5'],
            'within10': result['pairs']['within10'], 'complete_pair_missed_boxes': result['pairs']['missed_box']}
        require(all(row[k] == v for k, v in expected.items()), 'Published matrix row differs')

    arms['reflected_anchor'] = probe
    statuses = Counter(r['status'] for rows in associations.values() for r in rows)
    summary = {'schema': 'cassette-reflected-anchor-probe-r62-v1', 'status': 'completed',
        'development_only': True, 'fresh_inference': False, 'training': False,
        'labels_modified': False, 'automatic_promotion': False, 'threshold_search': False,
        'source_archive': {'path': ARCHIVE_PATH, 'sha256': ARCHIVE_SHA256, 'bytes': ARCHIVE_BYTES},
        'source_dependencies_sha256': dependencies,
        'driver_sha256': digest(Path(__file__).read_bytes()),
        'protocol': {**PROTOCOL, 'fallback': 'unchanged reflected-view record',
            'anchor': 'reflected', 'partner': 'trained',
            'scores': 'Reflected-view scores are retained; they are not calibrated confidence in averaged coordinates.',
            'comparison': 'One fixed exploratory anchor reversal; no new inference, training, threshold fitting or label-based links.'},
        'geometry_verification': {'independent_prediction_and_link_replay': True,
            'mutual_links_symmetric_under_anchor_reversal': True,
            'linked_coordinates_equal_to_existing_average': True,
            'anchor_boxes_scores_order_and_ids_preserved': True,
            'canonical_views_exactly_reproduced': ['original', 'trained', 'reflected'],
            'existing_proposed_exactly_reproduced': True,
            'native_reflection_transform': 'Already restored to photograph coordinates in the frozen input; no new reflection transform.'},
        'association_counts': {'anchor_records': sum(map(len, probe.values())),
            'partner_records': sum(map(len, arms['trained'].values())), **dict(statuses),
            'averaged_with_multiple_eligible_partners': sum(r['multiple_overlap_candidates'] for r in independent_rows)},
        'references': {}, 'matrix': [],
        'interpretation': 'Exploratory reuse of reviewed development photographs. Sensitivity references are not independent test sets. No candidate is promoted automatically.'}
    per_image, cases, identity_details = {}, {}, {}
    for reference, gold in references.items():
        per_image[reference] = dict(canonical['per-image'][reference])
        states = dict(canonical['cases'][reference])
        results = {}
        candidate_report, candidate_scores, candidate_states = report(gold, probe)
        per_image[reference]['reflected_anchor'] = candidate_scores
        states['reflected_anchor'] = candidate_states
        for image in gold:
            before, after = per_image[reference]['reflected'][image], candidate_scores[image]
            for key in (*COUNT_KEYS, 'matches', 'unmatched_gold_indices', 'unmatched_prediction_indices',
                        'duplicate_prediction_indices'):
                require(before[key] == after[key], 'Reflected detection identity changed: ' + key)
        for method in METHODS:
            result = candidate_report if method == 'reflected_anchor' else measured['references'][reference][method]
            errors = [r['maximum_corner_error_px'] for r in states[method]
                      if r['maximum_corner_error_px'] is not None]
            results[method] = {'detection': {k: v for k, v in result['detection'].items() if k != 'endpoint_errors_px'},
                'pairs': result['pairs'], 'accepted_pairs': accepted_report(gold, arms[method], .5),
                'mean_maximum_corner_error_px': statistics.mean(errors) if errors else None}
            detection, pairs = results[method]['detection'], results[method]['pairs']
            summary['matrix'].append({'reference': reference, 'method': method,
                'fronts': detection['gold_count'], 'outputs': detection['prediction_count'],
                'matched': detection['tp'], 'missed': detection['fn'], 'extras': detection['fp'],
                'duplicates': detection['duplicate_fp'], 'eligible_pairs': pairs['eligible'],
                'within5': pairs['within5'], 'within10': pairs['within10']})
        audit, identity_details[reference] = association_audit(gold, per_image[reference], associations, arms)
        summary['references'][reference] = {'arms': results, 'association_identity_audit': audit,
            'paired_changes_to_reflected_anchor': {method: changes(states[method], candidate_states)
                for method in ('original', 'proposed', 'reflected')},
            'reflected_detection_records_and_assignments_preserved': True}
        cases[reference] = []
        for offset, candidate in enumerate(candidate_states):
            row = {'case_id': reference + '/' + candidate['image'] + '/R' +
                   str(candidate['reference_index'] + 1).zfill(3)}
            for method in METHODS:
                value = dict(states[method][offset])
                require((value['image'], value['reference_index']) ==
                        (candidate['image'], candidate['reference_index']), 'Case alignment differs')
                index = value['prediction_index']
                value['primary_prediction_id'] = None if index is None else arms[method][value['image']][index].get('primary_prediction_id')
                row[method] = value
            cases[reference].append(row)
    records = {'reflected-anchor-predictions': probe, 'reflected-anchor-associations': associations,
               'association-identities': identity_details, 'paired-cases': cases, 'per-image': per_image}
    return summary, records


def write_result(work, summary, records):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    payload = {'summary.json': encoded(summary)}
    payload.update({name + '.json.gz': gzip.compress(encoded(value), mtime=0) for name, value in records.items()})
    payload['inventory.json'] = encoded({name: {'bytes': len(data), 'sha256': digest(data)}
                                        for name, data in payload.items()})
    for name, data in payload.items():
        path = work / name
        require(not path.exists() or path.read_bytes() == data, 'Different existing output preserved: ' + str(path))
        path.write_bytes(data)
    for name, data in payload.items():
        require((work / name).read_bytes() == data, 'Saved output differs: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    require(not args.work.resolve().is_relative_to(args.root.resolve()), 'Keep outputs outside source checkout')
    summary, records = analyze(args.archive or args.root / ARCHIVE_PATH, args.root)
    write_result(args.work, summary, records)
    for row in summary['matrix']:
        print(row['reference'], row['method'], 'missed', row['missed'], 'extras', row['extras'],
              'duplicates', row['duplicates'], 'pairs5', str(row['within5']) + '/' + str(row['eligible_pairs']),
              'pairs10', str(row['within10']) + '/' + str(row['eligible_pairs']))
    print('Verified reflected-anchor probe:', args.work)


if __name__ == '__main__':
    main()
