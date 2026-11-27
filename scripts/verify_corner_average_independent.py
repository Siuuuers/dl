#!/usr/bin/env python3
"""Independently replay equal coordinate averaging from the verified reflection return.

Association uses separately sorted candidate lists in both directions. It does not
import the experiment's averaging implementation or read reference coordinates
until after constructing predictions. Python's standard library is sufficient.
"""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path

from audit_reflection_return import (
    ARCHIVE_SHA256, encoded, load_references, pair_errors, require, verify_archive)


def independent_overlap(first, second):
    if first['class_id'] != second['class_id']:
        return -1.
    a, b = first['bbox_xyxy'], second['bbox_xyxy']
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(
        0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else -1.


def complete_prediction(record):
    if len(record['keypoints']) != 2:
        return False
    for x, y, score in record['keypoints']:
        if x is None or y is None or score is None or not all(
                math.isfinite(value) for value in (x, y, score)):
            return False
        if not (0 <= x < 1280 and 0 <= y < 720 and score >= .5):
            return False
    return True


def independent_average(data):
    averaged, links, association_rows = {}, {}, []
    for image in sorted(data['predictions']['control']):
        original = data['predictions']['control'][image]['postclip']
        reflection = data['predictions']['treatment'][image]['postclip']
        output, selected = deepcopy(original), []
        forward, backward, eligible_original, eligible_reflection = {}, {}, {}, {}
        for i, prediction in enumerate(original):
            candidates = sorted(
                [(independent_overlap(prediction, other), j) for j, other in enumerate(reflection)
                 if prediction['class_id'] == other['class_id']], key=lambda value: (-value[0], value[1]))
            if candidates:
                forward[i] = candidates[0]
            eligible_original[i] = sum(value[0] >= .5 for value in candidates)
        for j, prediction in enumerate(reflection):
            candidates = sorted(
                [(independent_overlap(other, prediction), i) for i, other in enumerate(original)
                 if prediction['class_id'] == other['class_id']], key=lambda value: (-value[0], value[1]))
            if candidates:
                backward[j] = candidates[0]
            eligible_reflection[j] = sum(value[0] >= .5 for value in candidates)
        for i, (iou, j) in forward.items():
            if iou < .5 or backward[j][1] != i:
                continue
            if not complete_prediction(original[i]) or not complete_prediction(reflection[j]):
                continue
            for slot, (first, second) in enumerate(zip(original[i]['keypoints'], reflection[j]['keypoints'])):
                output[i]['keypoints'][slot] = [
                    (first[0] + second[0]) * .5, (first[1] + second[1]) * .5, first[2]]
            selected.append((i, j))
            association_rows.append({'image_id': image, 'original_index': i, 'reflected_index': j,
                                     'iou': iou, 'multiple_overlap_candidates':
                                     eligible_original[i] > 1 or eligible_reflection[j] > 1})
        require(len(output) == len(original), 'Original record count changed')
        for before, after in zip(original, output):
            require({key: value for key, value in before.items() if key != 'keypoints'} ==
                    {key: value for key, value in after.items() if key != 'keypoints'},
                    'Original detection record changed')
            require([point[2] for point in before['keypoints']] ==
                    [point[2] for point in after['keypoints']], 'Original corner score changed')
        averaged[image], links[image] = output, selected
    return averaged, links, association_rows


def analyze(root, archive):
    data = verify_archive(archive)
    averaged, links, association_rows = independent_average(data)
    references = load_references(root, data['summary']['contract'])
    from pose_scoring import score_image
    result = {'linked_pairs': len(association_rows),
              'total_original_outputs': sum(len(rows) for rows in averaged.values()),
              'ambiguous_links': sum(row['multiple_overlap_candidates'] for row in association_rows),
              'references': {}}
    for reference, images in references.items():
        rows = []
        totals = dict.fromkeys(('eligible', 'original5', 'average5', 'original10', 'average10'), 0)
        identity_counts, complete_identity_counts, different = Counter(), Counter(), []
        for image, targets in images.items():
            original = data['predictions']['control'][image]['postclip']
            reflection = data['predictions']['treatment'][image]['postclip']
            before = score_image(targets, original, 1280, 720, .5, .5)
            after = score_image(targets, averaged[image], 1280, 720, .5, .5)
            alternative = score_image(targets, reflection, 1280, 720, .5, .5)
            for key in ('matches', 'unmatched_gold_indices', 'unmatched_prediction_indices',
                        'duplicate_prediction_indices'):
                require(before[key] == after[key], 'Box identity or count changed: ' + key)
            matches = {row['gold_index']: row['prediction_index'] for row in before['matches']}
            link_map = dict(links[image])
            for index, target in enumerate(targets):
                if not all(point[2] == 2 for point in target['keypoints']):
                    continue
                totals['eligible'] += 1
                prediction_index = matches.get(index)
                first = pair_errors(target, None if prediction_index is None else original[prediction_index])
                average = pair_errors(target, None if prediction_index is None else averaged[image][prediction_index])
                row = {'case_id': f'{reference}/{image}/R{index + 1:03d}', 'original_errors': first,
                       'average_errors': average, 'linked_to_reflection': prediction_index in link_map,
                       'reflected_errors': None if prediction_index not in link_map else pair_errors(
                           target, reflection[link_map[prediction_index]])}
                for threshold in (5, 10):
                    was_good = first is not None and max(first) <= threshold
                    now_good = average is not None and max(average) <= threshold
                    totals['original' + str(threshold)] += was_good
                    totals['average' + str(threshold)] += now_good
                    row['gain' + str(threshold)] = not was_good and now_good
                    row['loss' + str(threshold)] = was_good and not now_good
                rows.append(row)
            assigned_original = {row['prediction_index']: row['gold_index'] for row in before['matches']}
            assigned_reflection = {row['prediction_index']: row['gold_index'] for row in alternative['matches']}
            for original_index, reflected_index in links[image]:
                a = assigned_original.get(original_index)
                b = assigned_reflection.get(reflected_index)
                category = ('both_unassigned' if a is None and b is None else
                            'original_unassigned' if a is None else
                            'reflection_unassigned' if b is None else
                            'same_target' if a == b else 'different_target')
                identity_counts[category] += 1
                if a is not None and all(point[2] == 2 for point in targets[a]['keypoints']):
                    complete_identity_counts[category] += 1
                if category == 'different_target':
                    different.append({'image': image, 'original_prediction': original_index,
                                      'reflected_prediction': reflected_index,
                                      'original_target': a, 'reflected_target': b})
        for threshold in (5, 10):
            totals['gains' + str(threshold)] = sum(row['gain' + str(threshold)] for row in rows)
            totals['losses' + str(threshold)] = sum(row['loss' + str(threshold)] for row in rows)
        result['references'][reference] = {
            'counts': totals, 'ten_pixel_losses': [row for row in rows if row['loss10']],
            'remaining_five_pixel_failures': [row for row in rows
                                            if row['average_errors'] is None or max(row['average_errors']) > 5],
            'five_pixel_gains': [row for row in rows if row['gain5']],
            'five_pixel_losses': [row for row in rows if row['loss5']],
            'association_identity_audit': {'all_links': dict(identity_counts),
                                           'original_matched_complete_links': dict(complete_identity_counts),
                                           'different_target_links': different}}
    return result, averaged


def compare_production(result, averaged, summary_path, predictions_path):
    summary = json.loads(summary_path.read_bytes())
    require(summary['source']['return_sha256'] == ARCHIVE_SHA256, 'Production input archive differs')
    counts = summary['association_counts']
    require(counts['averaged'] == result['linked_pairs'] and
            counts['original_records'] == result['total_original_outputs'] and
            counts['averaged_with_multiple_eligible_partners'] == result['ambiguous_links'],
            'Production association counts differ')
    payload = predictions_path.read_bytes()
    require(json.loads(gzip.decompress(payload) if predictions_path.suffix == '.gz' else payload) == averaged,
            'Production averaged prediction records differ')
    for reference, item in result['references'].items():
        published = summary['references'][reference]
        for arm, prefix in (('control', 'original'), ('average', 'average')):
            require(published['arms'][arm]['pairs']['eligible'] == item['counts']['eligible'],
                    'Production reference denominator differs')
            for threshold in (5, 10):
                require(published['arms'][arm]['pairs']['within' + str(threshold)] ==
                        item['counts'][prefix + str(threshold)], 'Production pair success count differs')
        for threshold in (5, 10):
            changes = published['paired_changes'][str(threshold)]
            require(changes['gained'] == item['counts']['gains' + str(threshold)] and
                    changes['lost'] == item['counts']['losses' + str(threshold)],
                    'Production paired gains or losses differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare-summary', type=Path)
    parser.add_argument('--compare-predictions', type=Path)
    args = parser.parse_args()
    if bool(args.compare_summary) != bool(args.compare_predictions):
        parser.error('Supply both production comparison files or neither.')
    result, averaged = analyze(args.root, args.archive)
    if args.compare_summary:
        compare_production(result, averaged, args.compare_summary, args.compare_predictions)
    payload = encoded(result)
    require(not args.output.exists() or args.output.read_bytes() == payload,
            'Different existing output preserved: ' + str(args.output))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(json.dumps({'status': 'passed', 'output': str(args.output),
                      'production_compared': bool(args.compare_summary),
                      'averaged_records': result['linked_pairs'],
                      'original_records': result['total_original_outputs']}))


if __name__ == '__main__':
    main()
