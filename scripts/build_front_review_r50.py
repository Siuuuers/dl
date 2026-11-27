#!/usr/bin/env python3
"""Replay the R50 user-clarified bottom boundary and keep references versioned."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

from build_front_review_r46 import amend_line
from build_front_review_r44 import changed_fields, geometry
from evaluate_pose import parse_pose_labels
from verify_pose_candidate import check_geometry
from pose_scoring import aggregate_scores, score_image
from audit_additional_front_pilot_r45_results import corner_groups
from audit_additional_front_pilot_r47_results import ARCHIVE_SHA

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path('data/reviews/fronts-2026-11-27-r48')
REVIEW = Path('data/reviews/fronts-2026-11-27-r50')
PARENT_SHA = '21838bd979a03f41a82ea17496f29bb4d4bd6357c94064112a91124a5a4723e4'
STABLE_ID = 'new_000010:R38:029'
IMAGE = 'new_000010'
ROW = 29


def need(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked(path, expected):
    data = Path(path).read_bytes()
    need(sha(data) == expected, 'Hash differs: ' + str(path))
    return data


def put(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_bytes() == data, 'Existing different file preserved: ' + str(path))
    else:
        path.write_bytes(data)


def build(root):
    parent_data = checked(root / PARENT / 'manifest.json', PARENT_SHA)
    parent = json.loads(parent_data)
    parent_audit = json.loads(checked(root / PARENT / 'audit-ledger.json', parent['audit_sha256']))
    parent_files = parent['files']
    need(len(parent_files) == 20 and len({r['image_id'] for r in parent_files}) == 20,
         'Expected twenty distinct development photographs')
    before_op = next(r for r in parent_audit['rows'] if r['stable_id'] == STABLE_ID)
    need([r['field'] for r in before_op['held_fields']] == ['ymax'], 'Expected R029 bottom hold')
    before = before_op['after_line']
    updates = {'ymax': 714}
    after = amend_line(before, updates)
    need(changed_fields(before, after) == ['ymax'], 'Only the physical bottom may change')
    need(before.split()[5:] == after.split()[5:], 'Corner coordinates or availability changed')
    reason = (
        'The user clarifies that the band below this cassette belongs to another cassette, '
        'resolving the R48 material-ownership hold. The original image shows the lower-left '
        'front transition near the retained left corner at (325,714). Set the tight '
        'axis-aligned visible-front bottom to nominal y714, excluding the neighboring '
        'lower band. This uses the stated ownership and visible junction, not a general '
        'rule to snap every box to a point. Keep the top, sides, visible left point and '
        'canonical missing right point exactly as recorded.'
    )
    operation = dict(stable_id=STABLE_ID, image_id=IMAGE, reference_row=ROW,
                     source_image_sha256=before_op['source_image_sha256'],
                     before_line=before, after_line=after, updates_px=updates,
                     changed_fields=['ymax'], original_pixels_inspected=True,
                     human_review_performed=False, user_ownership_clarification=True,
                     feedback='new_000010 / row 029 the bottom borderline need to excluding the other cassette, it is at around left point.',
                     reason=reason, nominal_visual_uncertainty_px=2,
                     resolved_fields=[dict(field='ymax', prior_review='R48',
                                          basis='User material-ownership clarification plus original-image lower-left transition.')])
    holds = [dict(stable_id=r['stable_id'], image_id=r['image_id'], reference_row=r['reference_row'], **h)
             for r in parent_audit['rows'] for h in r['held_fields'] if r['stable_id'] != STABLE_ID]
    need(len(holds) == 1 and holds[0]['stable_id'] == 'new_000089:R38:041'
         and holds[0]['field'] == 'xmax', 'Remaining hold differs')
    ledger = dict(schema_version=1, round=50, date='2026-11-27', timezone='Asia/Hong_Kong',
                  parent_manifest_sha256=PARENT_SHA, operations=[operation], remaining_holds=holds,
                  human_review_performed=False, user_feedback_adjudicates_one_material_owner=True,
                  scope='One development-row bottom amendment; inherited rows are not newly certified.',
                  coordinate_semantics='Original 1280 x 720: x right, y down; physical bottom is stored ymax.')
    files, entries = {}, []
    visible = missing = objects = 0
    for entry in sorted(parent_files, key=lambda r: r['image_id']):
        identity = entry['image_id']
        old = checked(root / PARENT / entry['label_path'], entry['label_sha256'])
        data = old
        if identity == IMAGE:
            lines = old.decode().splitlines()
            need(lines[ROW - 1] == before, 'Exact predecessor row differs')
            lines[ROW - 1] = after
            data = ('\n'.join(lines) + '\n').encode()
        parsed = parse_pose_labels(data.decode(), 1280, 720, identity)
        check_geometry(parsed, 1280, 720, identity)
        need(len(parsed) == entry['objects'], 'Instance population changed')
        count_visible = sum(p[2] == 2 for r in parsed for p in r['keypoints'])
        visible += count_visible
        missing += 2 * len(parsed) - count_visible
        objects += len(parsed)
        relative = 'labels/' + identity + '.txt'
        files[relative] = data
        entries.append(dict(image_id=identity, split='val', label_path=relative,
                            label_sha256=sha(data), parent_label_sha256=entry['label_sha256'],
                            source_image_sha256=entry['source_image_sha256'], objects=len(parsed),
                            locatable_corners=count_visible, missing_corners=2 * len(parsed) - count_visible))
    need((objects, visible, missing) == (654, 1220, 88), 'Development population differs')
    files['audit-ledger.json'] = encoded(ledger)
    manifest = dict(schema_version=1, kind='versioned_visible_front_development_reference', round=50,
                    review_id='fronts-2026-11-27-r50', date='2026-11-27',
                    status='assistant_review_with_user_ownership_clarification',
                    parent=dict(manifest_path=str(PARENT / 'manifest.json'), manifest_sha256=PARENT_SHA),
                    audit_sha256=sha(files['audit-ledger.json']), development_only=True,
                    full_dataset_certified=False, human_review_performed=False,
                    user_ownership_clarification=True, changed_training_rows=0,
                    coordinate_semantics=ledger['coordinate_semantics'],
                    row_identity_semantics='One-based R38 rows retained through R48 and R50; no instance addition or deletion.',
                    counts=dict(images=20, objects=654, locatable_corners=1220, missing_corners=88,
                                amended_rows=1, amended_box_fields=1, changed_corner_positions=0,
                                added_rows=0, deleted_rows=0, visibility_changes=0,
                                resolved_fields=1, held_rows=1, held_fields=1),
                    remaining_holds=holds, files=entries)
    files['manifest.json'] = encoded(manifest)
    return manifest, ledger, files


def verify_source(root, manifest, files):
    required = {str(root / REVIEW / name) for name in files}
    actual_labels = {str(p) for p in (root / REVIEW / 'labels').glob('*.txt')}
    need(actual_labels == {p for p in required if p.endswith('.txt')}, 'Unexpected or missing development label')
    for name, data in files.items():
        need((root / REVIEW / name).read_bytes() == data, 'Committed R50 replay differs: ' + name)
    return dict(passed=True, parent_manifest_sha256=PARENT_SHA,
                manifest_sha256=sha(files['manifest.json']), audit_sha256=sha(files['audit-ledger.json']),
                exact_label_files=20, counts=manifest['counts'],
                changed_training_rows=0, predictions_modified=False, frozen_references_modified=False)


def sensitivity(root, archive, files):
    with zipfile.ZipFile(archive) as z:
        need(sha(Path(archive).read_bytes()) == ARCHIVE_SHA, 'PR137 archive differs')
        predictions = json.loads(z.read('evaluation/predictions.json'))
    parent = json.loads((root / PARENT / 'manifest.json').read_bytes())
    labels = {'r48': {}, 'r50': {}}
    for entry in parent['files']:
        identity = entry['image_id']
        labels['r48'][identity] = parse_pose_labels((root / PARENT / entry['label_path']).read_text(), 1280, 720, identity)
        labels['r50'][identity] = parse_pose_labels(files['labels/' + identity + '.txt'].decode(), 1280, 720, identity)
    reports = {}
    assignment_changes = []
    for version, gold in labels.items():
        reports[version] = {}
        for role in ('reference', 'candidate'):
            scores = {}
            groups = {str(n): dict(eligible=0, matched=0, within5=0, within10=0) for n in (0, 1, 2)}
            for identity, targets in gold.items():
                outputs = predictions[role][identity]['postclip']
                scores[identity] = score_image(targets, outputs, 1280, 720, .5, .5)
                for key, values in corner_groups(targets, outputs, scores[identity]).items():
                    for name, value in values.items():
                        groups[key][name] += value
            reports[version][role] = dict(detection=aggregate_scores(list(scores.values())),
                                         corners_by_locatable_count=groups, per_image=scores)
    for role in ('reference', 'candidate'):
        for identity in labels['r48']:
            a, b = (reports[v][role]['per_image'][identity] for v in ('r48', 'r50'))
            pairs = lambda s: {(m['gold_index'], m['prediction_index']) for m in s['matches']}
            if pairs(a) != pairs(b) or a['unmatched_gold_indices'] != b['unmatched_gold_indices']:
                assignment_changes.append(dict(role=role, image_id=identity))
    return dict(schema_version=1, comparison='Fixed PR137 postclip predictions; R48 versus user-clarified R50 reference',
                archive_sha256=ARCHIVE_SHA, r50_manifest_sha256=sha(files['manifest.json']),
                fresh_inference=False, fresh_training=False, assignment_changes=assignment_changes,
                reports=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--write-source', action='store_true')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--originals', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--pdf', type=Path)
    args = parser.parse_args()
    manifest, ledger, files = build(args.root)
    if args.write_source:
        expected = {name for name in files if name.startswith('labels/')}
        existing = {p.relative_to(args.root / REVIEW).as_posix() for p in (args.root / REVIEW / 'labels').glob('*.txt')}
        need(existing <= expected, 'Unexpected existing label preserved')
        for name, data in files.items():
            path = args.root / REVIEW / name
            need(not path.exists() or path.read_bytes() == data, 'Different source export preserved: ' + name)
        for name, data in files.items():
            put(args.root / REVIEW / name, data)
    verification = verify_source(args.root, manifest, files)
    report = sensitivity(args.root, args.archive, files) if args.archive else None
    if args.output:
        put(args.output / 'verification.json', encoded(verification))
        if report:
            put(args.output / 'reference-sensitivity.json', encoded(report))
    if args.pdf:
        need(args.originals and report, 'PDF requires --originals and --archive')
        for identity in (IMAGE, 'new_000089'):
            entry = next(e for e in manifest['files'] if e['image_id'] == identity)
            checked(args.originals / (identity + '.png'), entry['source_image_sha256'])
        from render_front_review_r50 import render_pdf
        render_pdf(manifest, ledger, report, args.originals, args.pdf)
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
