"""Verify a source-preserving annotation proposal; this does not approve its visual truth."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def member(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(), 'relative path required')
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), f'missing or unsafe path: {relative}')
    require(not path.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1'),
            f'actual file bytes required: {relative}')
    return path


def verify(root, candidate, source_root):
    sys.path.insert(0, str(root / 'scripts'))
    from evaluate_pose import parse_pose_labels
    from stage_pose_training import check_geometry

    manifest_path = candidate / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    require(manifest['schema_version'] == 1 and manifest['kind'] == 'supplementary_annotation_adaptation',
            'wrong candidate kind or schema')
    require(manifest['status'] == 'agent-reviewed-proposal', 'expected an explicitly proposed annotation version')
    require(manifest['physical_review_is_human_approval'] is False, 'review origin must be disclosed')
    require(digest(root / 'docs/annotation-policy.md') == manifest['policy_sha256'], 'policy bytes differ')
    require(digest(root / 'data/manifest.json') == manifest['frozen_reference_manifest_sha256'],
            'frozen development manifest differs')
    require(digest(source_root / 'manifest.json') == manifest['source_manifest_sha256'],
            'upstream source manifest bytes differ')
    source = json.loads((source_root / 'manifest.json').read_text(encoding='utf-8-sig'))
    source_by_id = {Path(row['image']).stem: row for row in source['images']}
    require(manifest['class_names'] == ['cassette_front'] and manifest['kpt_shape'] == [2, 3]
            and manifest['flip_idx'] == [1, 0], 'target semantics differ')
    ids, listed_labels, listed_sources = set(), set(), set()
    total = {'images': 0, 'source_objects': 0, 'candidate_objects': 0,
             'changed_rows': 0, 'box_changed_rows': 0, 'point_changed_rows': 0,
             'point_slots_changed': 0, 'visibility_changes': 0,
             'source_visible_endpoints': 0, 'candidate_visible_endpoints': 0,
             'quarantined_images': 0, 'unresolved_rows': 0}
    image_results = []
    for record in manifest['images']:
        name = record['id']
        require(name in source_by_id and name not in ids, f'unknown or repeated source image: {name}')
        ids.add(name)
        source_record = source_by_id[name]
        image = member(source_root, record['source_image'])
        original_label = member(source_root, record['source_label'])
        require(record['source_image'] == source_record['image'] and record['source_label'] == source_record['label'],
                f'source path differs: {name}')
        require(digest(image) == record['source_image_sha256'] == source_record['image_sha256'],
                f'source image hash differs: {name}')
        require(digest(original_label) == record['source_label_sha256'] == source_record['label_sha256'],
                f'source label hash differs: {name}')
        preserved = member(candidate, record['preserved_source_label'])
        require(preserved.read_bytes() == original_label.read_bytes(), f'preserved source bytes differ: {name}')
        listed_sources.add(record['preserved_source_label'])
        label = member(candidate, record['candidate_label'])
        listed_labels.add(record['candidate_label'])
        require(digest(label) == record['candidate_label_sha256'], f'candidate label hash differs: {name}')
        review_path = member(candidate, record['review_record'])
        require(digest(review_path) == record['review_sha256'], f'review record hash differs: {name}')
        review = json.loads(review_path.read_text())
        source_lines = original_label.read_text().splitlines()
        candidate_lines = label.read_text().splitlines()
        require(len(source_lines) == len(candidate_lines) == len(review['rows']) == record['objects'],
                f'row identity/coverage count differs: {name}')
        require([r['source_row_1based'] for r in review['rows']] == list(range(1, len(source_lines)+1)),
                f'review rows incomplete or reordered: {name}')
        require(review['whole_image_inspected'] is True and review['model_predictions_used'] is False,
                f'whole-image model-blind review required: {name}')
        replay_lines = source_lines[:]
        changes = set()
        for change in record['changes']:
            number = change['source_row_1based']
            require(isinstance(number, int) and 1 <= number <= len(source_lines) and number not in changes,
                    f'invalid/repeated changed row: {name}')
            changes.add(number)
            require(change['before'] == source_lines[number-1] and change['after'] == candidate_lines[number-1],
                    f'changed-row text differs: {name}:{number}')
            require(change['before'] != change['after'] and change['reason'].strip(),
                    f'empty change or physical reason: {name}:{number}')
            replay_lines[number-1] = change['after']
        require(replay_lines == candidate_lines, f'unrecorded label transformation: {name}')
        actual_changes = {i for i,(a,b) in enumerate(zip(source_lines,candidate_lines),1) if a != b}
        require(actual_changes == changes, f'change inventory differs: {name}')
        unresolved = []
        for row in review['rows']:
            number = row['source_row_1based']
            require(row['source_text'] == source_lines[number-1] and row['candidate_text'] == candidate_lines[number-1],
                    f'review does not refer to exact row bytes: {name}:{number}')
            require(row['status'] in ['retained-supported', 'corrected-supported', 'unresolved'],
                    f'unknown physical review status: {name}:{number}')
            require(bool(row['identity_assessment']) and bool(row['box_assessment'])
                    and bool(row['left_endpoint_assessment']) and bool(row['right_endpoint_assessment']),
                    f'incomplete physical review: {name}:{number}')
            if row['status'] == 'unresolved':
                unresolved.append(number)
            elif number in changes:
                require(row['status'] == 'corrected-supported', f'changed row not identified: {name}:{number}')
            else:
                require(row['status'] == 'retained-supported', f'retained row not identified: {name}:{number}')
        require(unresolved == record['unresolved_source_rows'], f'unresolved inventory differs: {name}')
        if unresolved or review['whole_image_unresolved_issues']:
            require(record['disposition'] == 'quarantined', f'unresolved image must be quarantined: {name}')
        else:
            require(record['disposition'] in ['reviewed-proposal', 'quarantined'], f'invalid disposition: {name}')
        require(record['eligible_for_model_scoring'] is False,
                f'this annotation proposal is not a released scoring reference: {name}')
        require(record['eligible_for_training'] is False,
                f'this quarantined proposal is not released training data: {name}')
        width, height = record['original_size']
        require([width,height] == [source_record['width'],source_record['height']], f'dimensions differ: {name}')
        old = parse_pose_labels(original_label.read_text(), width, height, str(original_label))
        new = parse_pose_labels(label.read_text(), width, height, str(label))
        check_geometry(new, width, height, str(label))
        require(len(record['unresolved_fields']) == len(unresolved), f'unresolved field inventory differs: {name}')
        field_rows = set()
        for issue in record['unresolved_fields']:
            number = issue['source_row_1based']
            require(number in unresolved and number not in field_rows, f'invalid unresolved field row: {name}')
            field_rows.add(number)
            require(bool(issue['edges']), f'unresolved field unspecified: {name}:{number}')
            for edge in issue['edges']:
                require(edge in ['xmin', 'ymin', 'xmax', 'ymax'], f'unknown bbox edge: {name}:{number}')
                i = ['xmin', 'ymin', 'xmax', 'ymax'].index(edge)
                require(abs(old[number-1]['bbox_xyxy'][i] - new[number-1]['bbox_xyxy'][i]) <= 1e-4,
                        f'unresolved source placeholder moved: {name}:{number}:{edge}')
        box_changes = sum(a['bbox_xyxy'] != b['bbox_xyxy'] for a,b in zip(old,new))
        point_changes = sum(a['keypoints'] != b['keypoints'] for a,b in zip(old,new))
        values = {'images': 1, 'source_objects': len(old), 'candidate_objects': len(new),
                  'changed_rows': len(changes), 'box_changed_rows': box_changes,
                  'point_changed_rows': point_changes,
                  'point_slots_changed': sum(p != q for a,b in zip(old,new)
                                             for p,q in zip(a['keypoints'],b['keypoints'])),
                  'visibility_changes': sum(p[2] != q[2] for a,b in zip(old,new)
                                            for p,q in zip(a['keypoints'],b['keypoints'])),
                  'source_visible_endpoints': sum(p[2] == 2 for obj in old for p in obj['keypoints']),
                  'candidate_visible_endpoints': sum(p[2] == 2 for obj in new for p in obj['keypoints']),
                  'quarantined_images': int(record['disposition'] == 'quarantined'),
                  'unresolved_rows': len(unresolved)}
        for key,value in values.items():
            total[key] += value
        image_results.append({'id': name, 'disposition': record['disposition'], **values})
    require(ids == set(manifest['selected_image_ids']), 'selected image inventory differs')
    require({p.relative_to(candidate).as_posix() for p in (candidate/'labels').glob('*.txt')} == listed_labels,
            'unexpected or missing candidate labels')
    require({p.relative_to(candidate).as_posix() for p in (candidate/'source-labels').glob('*.txt')} == listed_sources,
            'unexpected or missing preserved source labels')
    require(total == manifest['counts'], 'aggregate counts differ')
    require(manifest['released_for_model_scoring'] is False, 'annotation proposal must not be treated as a scoring release')
    require(manifest['eligible_image_ids'] == [], 'quarantined proposal must have no eligible images')
    return {'status': 'verified', 'candidate_id': manifest['candidate_id'],
            'manifest_sha256': digest(manifest_path), 'counts': total, 'images': image_results,
            'source_image_bytes_verified': True, 'source_label_bytes_preserved': True,
            'every_row_accounted_for': True, 'all_declared_transformations_reproduced': True,
            'candidate_schema_and_geometry_pass': True, 'visual_truth_certified_by_this_verifier': False,
            'human_approval_claimed': False, 'model_inference_or_training_run': False,
            'released_for_model_scoring': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.root.resolve(), args.candidate.resolve(), args.source_root.resolve())
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print(json.dumps({'status': 'invalid', 'error': str(error)}))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
