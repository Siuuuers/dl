#!/usr/bin/env python3
"""Build the pinned r25 Colab notebook from the unchanged r24 runtime cells."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


BASE_NOTEBOOK = "notebooks/baomai_front_review_r24.ipynb"
BASE_NOTEBOOK_SHA256 = "36135c847fb460f7299628374bea907a01981e48d5336aa5321b1afda8fc8e2a"
BASE_SOURCE = "6307dae6a8db545a72446abb448f19a5dcd5d2b2"
BASE_PROTOCOL_SHA256 = "0a6322dbf4ee0cfca60b06478f8feecbd4dce516df92301cc74a2030b73021f5"
PROTOCOL_PATH = "experiments/front-review-2026-11-23-r25.json"
DRAFT_SOURCE = "SOURCE_COMMIT_TO_BE_PINNED_AFTER_PUBLICATION"
DRAFT_PROTOCOL = "PROTOCOL_HASH_TO_BE_PINNED_AFTER_BUILD"


INPUT_PREFLIGHT = r'''
import json
from collections import Counter
from pathlib import Path
import sys
sys.dont_write_bytecode = True
root = Path.cwd()
sys.path.insert(0, str(root / 'scripts'))
from evaluate_pose import file_sha256, preflight_dataset
from verify_incremental_pose_candidate import verify_incremental_candidate
from stage_pose_training import source_state
from rescore_reference_amendment import load_api, verify_amendment
from replay_corner_reference import reference_counts
from front_review_r25 import REQUESTED, ADJACENT, STANDARDS, PARENT, SOURCE_REVISION as REVIEW_SOURCE_REVISION

protocol_path = root / sys.argv[1]
assert file_sha256(protocol_path) == sys.argv[2]
protocol = json.loads(protocol_path.read_text())
assert protocol['experiment_id'] == 'front-review-r25'
candidate = root / protocol['candidate']['path']
assert file_sha256(candidate) == protocol['candidate']['sha256']
inspection = verify_incremental_candidate(root, candidate)
assert inspection['training_readiness'] == 'ready_for_controlled_incremental_experiment'
assert inspection['full_dataset_certified'] is False
assert inspection['named_rows_reviewed'] == 5
assert inspection['adjacent_rows_reviewed'] == len(ADJACENT)
assert inspection['endpoints_reviewed'] == 2 * (sum(map(len, REQUESTED.values())) + len(ADJACENT))
assert inspection['protected_standard_rows'] == len(STANDARDS)
assert inspection['changed_box_rows'] == 0
assert inspection['confirmed_reference_rows_preserved'] is True
assert inspection['classes_boxes_visibility_preserved'] is True
assert inspection['training_validation_preserved'] is True
assert inspection['instance_counts_preserved'] is True

state = source_state(root, candidate)
initial = preflight_dataset(root, require_weights=True)
assert initial['model_sha256'] == protocol['initial_checkpoint_sha256']
assert state['source']['initial_checkpoint_sha256'] == initial['model_sha256']
assert inspection['retained_legacy_image_ids'] == ['new_000015']
assert inspection['excluded_training_image_ids'] == protocol['excluded_training_image_ids']
assert len(inspection['excluded_training_image_ids']) == 31
assert inspection['retained_photo_labels_identical_between_arms'] is True
assert inspection['excluded_photo_labels_identical_between_arms'] is True
assert set(protocol['roles']) == {'reference', 'candidate'}
for role, spec in protocol['roles'].items():
    assert spec['arm'] == role
    assert all(state['source'][key] == spec[key] for key in
               ('candidate_manifest', 'candidate_manifest_sha256', 'candidate_id'))
    assert state['counts'][role] == spec['counts']
    assert state['counts'][role]['train']['images'] == 137
    assert state['counts'][role]['train']['empty_images'] == 13
    assert state['counts'][role]['val']['images'] == 20
    assert state['counts'][role]['val']['objects'] == 664
assert state['counts']['reference']['val'] == state['counts']['candidate']['val']

review_path = root / protocol['source_candidate']['path']
assert file_sha256(review_path) == protocol['source_candidate']['sha256']
review = json.loads(review_path.read_text())
assert review['review_id'] == 'fronts-2026-11-23-r25'
assert review['source_revision'] == REVIEW_SOURCE_REVISION
assert review['parent'] == PARENT
parent_path = root / review['parent']['path']
assert file_sha256(parent_path) == review['parent']['sha256']
parent = json.loads(parent_path.read_text())
assert parent['review_id'] == 'fronts-2026-11-23-r24'
assert len(review['review_evidence']) == 1
audit_descriptor = review['review_evidence'][0]
assert audit_descriptor['path'] == 'results/review/front-review-20261123-r25/point-review.json'
audit_path = root / audit_descriptor['path']
assert file_sha256(audit_path) == audit_descriptor['sha256']
audit = json.loads(audit_path.read_text())
assert audit['source_review'] == review['parent']
assert audit['source_revision'] == review['source_revision']
named = {(image_id, row) for image_id, rows in REQUESTED.items() for row in rows}
adjacent = set(ADJACENT)
standards = list(STANDARDS)
assert len(named) == 5 and len(standards) == 11 and not named.intersection(adjacent)
expected_rows = named | adjacent
audited = {(item['image_id'], item['row']): item for item in audit['rows']}
assert len(audit['rows']) == len(audited) == len(expected_rows)
assert set(audited) == expected_rows
assert [(item['image_id'], item['row']) for item in audit['preserved_standards']] == standards
assert sum(len(item['point_reviews']) for item in audit['rows']) == 2 * len(expected_rows)

records = state['records']
assert len(records) == 157
changed_rows, changed_images = set(), set()
point_changes, split_changes = Counter(), Counter()
endpoint_visibility, point_split_changes = Counter(), Counter()
review_lines, parent_lines, counts = {}, {}, {}
held = set(inspection['retained_legacy_image_ids'] + inspection['excluded_training_image_ids'])
for record in records:
    image_id, split = record['id'], record['split']
    assert record['original_size'] == [1280, 720]
    current_bytes = (review_path.parent / record['label']).read_bytes()
    parent_bytes = (parent_path.parent / record['label']).read_bytes()
    if current_bytes != parent_bytes:
        changed_images.add(image_id)
    if image_id not in {item[0] for item in expected_rows}:
        assert current_bytes == parent_bytes
    rows, previous = current_bytes.decode().splitlines(), parent_bytes.decode().splitlines()
    review_lines[image_id], parent_lines[image_id] = rows, previous
    assert len(rows) == len(previous)
    tally = counts.setdefault(split, {'images': 0, 'objects': 0, 'visible_endpoints': 0,
                                     'missing_endpoints': 0, 'empty_images': 0})
    tally['images'] += 1
    tally['objects'] += len(rows)
    tally['empty_images'] += not rows
    for row, (before, after) in enumerate(zip(previous, rows), 1):
        key = (image_id, row)
        old, new = before.split(), after.split()
        assert len(old) == len(new) == 11
        assert new[:5] == old[:5] and new[7] == old[7] and new[10] == old[10]
        for offset in (5, 8):
            assert new[offset + 2] in ('0', '2')
            if new[offset + 2] == '0':
                assert all(float(value) == 0 for value in new[offset:offset + 3])
                tally['missing_endpoints'] += 1
            else:
                tally['visible_endpoints'] += 1
        if key not in audited:
            assert after == before
            continue
        item = audited[key]
        assert item['scope'] == ('user_named' if key in named else 'adjacent_context')
        assert item['split'] == split
        assert item['source_image_sha256'] == record['image_sha256']
        assert item['before_line'] == before and item['after_line'] == after
        assert set(item['point_reviews']) == {'left', 'right'}
        assert new[7] == old[7] and new[10] == old[10]
        for side, offset in (('left', 5), ('right', 8)):
            decision = item['point_reviews'][side]
            assert decision['visibility'] == int(new[offset + 2])
            endpoint_visibility[int(new[offset + 2])] += 1
            if new[offset + 2] == '0':
                assert new[offset:offset + 3] == old[offset:offset + 3]
                assert all(float(value) == 0 for value in new[offset:offset + 3])
            changed = old[offset:offset + 2] != new[offset:offset + 2]
            assert decision['decision'] == ('corrected' if changed else 'retained')
            for label, tokens in (('before_pixel_xy', old), ('after_pixel_xy', new)):
                xy = decision[label]
                assert len(xy) == 2
                assert all(abs(float(tokens[offset + axis]) * scale - xy[axis]) < 1e-6
                           for axis, scale in enumerate((1280, 720)))
            if changed:
                point_changes[side] += 1
                point_split_changes[split] += 1
        if before != after:
            changed_rows.add(key)
            split_changes[split] += 1
            if split == 'val':
                assert image_id in {'new_000008', 'new_000010'}
    for role in ('reference', 'candidate'):
        plan = state['plans'][role][record['label']]
        if split == 'val':
            original_path = root / 'data' / record['label']
            assert plan['source'] == original_path
            assert file_sha256(original_path) == plan['sha256'] == record['label_sha256']
        elif image_id in held:
            other = state['plans']['reference'][record['label']]
            assert plan['sha256'] == other['sha256']
            assert plan['source'].read_bytes() == other['source'].read_bytes()
        elif role == 'candidate':
            assert plan['source'].read_bytes() == current_bytes

assert set(review['changed_image_ids']) == changed_images
assert inspection['changed_rows'] == len(changed_rows)
assert inspection['changed_photos'] == len(changed_images)
assert inspection['point_repairs'] == sum(point_changes.values())
assert inspection['point_repairs_by_side'] == dict(point_changes)
assert inspection['point_repairs_by_split'] == dict(point_split_changes)
assert inspection['locatable_endpoints_reviewed'] == endpoint_visibility[2]
assert inspection['missing_endpoints_reviewed'] == endpoint_visibility[0]
assert sum(endpoint_visibility.values()) == 2 * len(audited)
assert inspection['changed_rows_by_split'] == dict(split_changes)
assert inspection['unchanged_review_files'] == len(records) - len(changed_images)
assert inspection['development_review_labels_changed'] == bool(split_changes['val'])
for item in audit['preserved_standards']:
    image_id, row = item['image_id'], item['row']
    assert review_lines[image_id][row - 1] == parent_lines[image_id][row - 1] == item['line']
assert sum(bool(rows) for rows in review_lines.values()) == 141
assert sum(map(len, review_lines.values())) == 4448
assert counts == parent['expected_counts'] == protocol['complete_review_counts']
assert counts['val']['objects'] == 662
assert protocol['evaluation']['current_review_reference'] == protocol['source_candidate']
assert protocol['evaluation']['current_review_reference_is_exploratory'] is True

_, api = load_api(root)
assert protocol['evaluation']['diagnostic_reference_required'] is True
references = {}
for role, expected_objects in (('primary', 666), ('diagnostic', 663)):
    reference_path = root / protocol['evaluation'][role + '_reference']
    assert file_sha256(reference_path) == protocol['evaluation'][role + '_reference_sha256']
    reference, gold, *_ = verify_amendment(root, reference_path.parent, initial, api)
    reference_tally, groups = reference_counts(gold)
    assert reference_tally == reference['counts'] == protocol['evaluation'][role + '_counts']
    assert reference_tally['images'] == 20 and reference_tally['objects'] == expected_objects
    assert groups == reference['objects_by_locatable_corners']
    references[role] = {'counts': reference_tally, 'objects_by_locatable_corners': groups}
print(json.dumps({'status': inspection['training_readiness'],
                  'training_counts': state['counts'], 'complete_review_counts': counts,
                  'weights_verified': initial['weights_verified'],
                  'named_rows_reviewed': len(named), 'named_endpoints_reviewed': 2 * len(named),
                  'adjacent_rows_reviewed': len(adjacent), 'endpoints_reviewed': 2 * len(audited),
                  'locatable_endpoints_reviewed': endpoint_visibility[2],
                  'missing_endpoints_reviewed': endpoint_visibility[0],
                  'protected_standard_rows': len(standards),
                  'changed_rows': len(changed_rows), 'changed_photos': len(changed_images),
                  'changed_rows_by_split': dict(split_changes),
                  'point_repairs': sum(point_changes.values()), 'point_repairs_by_side': dict(point_changes),
                  'point_repairs_by_split': dict(point_split_changes),
                  'unchanged_review_files': len(records) - len(changed_images),
                  'changed_box_rows': 0, 'confirmed_reference_rows_preserved': True,
                  'original_training_validation_byte_identical': True,
                  'development_review_is_exploratory_only': True,
                  'held_training_photos': inspection['retained_legacy_image_ids'],
                  'old_photos_excluded_from_treatment': len(inspection['excluded_training_image_ids']),
                  'evaluation_references': references}, indent=2))
'''


INTRO = '''# Baomai cassette fronts - round 25

**Choose a GPU runtime, select Run all, and return `baomai-front-review-r25-results.zip`.**

This notebook uses the preserved R24 annotations plus the focused lower-front corner review of **five named rows: 008:35, 010:18-19 and 012:3,28**, with three related adjacent rows **010:20 and 012:1-2**. Both endpoint slots in all eight rows were inspected: **15 locatable points and one missing point**. Eight left points change, four in training and four in development; every right point is retained, including the unresolved soft boundary at 012:28. The **eleven accepted standard rows** remain byte-identical. Boxes, classes, visibility and object counts remain unchanged; the missing endpoint retains its exact zero tokens. Shadow and brightness support the local geometry, without defining a fixed threshold or offset.

Both arms use the same **137 training photos, including 13 empty scenes**. The reference uses PR35 supervision and the candidate uses eligible R25 labels; photo 015 and the 31 old-series exclusions retain identical PR35 labels in both arms. Each arm independently starts from original **epoch 173**, with **seed 46, 30 epochs, batch 4, input 1280, FP32 and the established AdamW recipe**. The resource trial never initializes a full run, and the final epoch is selected in advance.

Photos **008 and 010 belong to development validation**. Their revised review points enter only the separately versioned exploratory reference; original trainer validation remains **664 fronts**. The updated points in photo **012** enter candidate training. Evaluation retains the frozen **666-front primary** and **663-front historical** references. The **662-front R25 review** is exploratory. PR99 did not support promoting the R24 candidate, so **PR35 remains selected**. This new amendment experiment tests accumulated eligible supervision; it does not isolate the latest edits or assume they resolve partial-object misses.
'''

PREFLIGHT_MARKDOWN = '''## 2. Verify the exact experiment

Download the 157 original PNGs and epoch-173 initializer, verify their hashes, and replay R25 labels against the frozen R24 parent. Check both endpoint decisions in all five named rows and three adjacent rows, all eleven accepted standards, every preserved field and every unchanged row. Derive the correction, split and visibility counts directly from the hash-bound decision ledger. A missing endpoint remains missing rather than receiving an invented position.

Verify the actual training labels selected for each arm and the inherited training exclusions. Confirm every trainer-validation label remains byte-identical to the original data, including photos 008 and 010, and verify both fixed scoring references. A failed check stops before training. These checks establish that the intended annotations are used; the recorded visual review establishes their meaning.
'''

AFTER_RUN = '''### After the run

Compare both fresh final-epoch models with retained PR35 on the fixed references. Report unmatched targets, unmatched outputs, duplicates, empty scenes and both-corner successes within 5 and 10 original-image pixels. Retain eligible missed objects in each success denominator and inspect gained and lost identities, including partially visible fronts.

Keep R25 exploratory scores separate from the frozen references and from earlier R24 exploratory scores. Reference-box or corner changes can alter measured scores without changing model output; conflicting rankings do not establish a replacement. Do not select a favorable threshold, seed or epoch after seeing the results. Preparing this notebook does not train or promote a model, and the corner amendments do not establish a remedy for partial-object detection failures.
'''


def translated(text, source, protocol_sha):
    replacements = {
        "experiments/front-review-2026-11-23-r24.json": PROTOCOL_PATH,
        "baomai-front-review-r24-20261123": "baomai-front-review-r25-20261123",
        "r24": "r25", "r23": "r24", "PR97": "PR99",
        "round 24": "round 25", "Round 24": "Round 25",
        BASE_SOURCE: source, BASE_PROTOCOL_SHA256: protocol_sha,
    }
    pattern = re.compile("|".join(re.escape(value) for value in sorted(replacements, key=len, reverse=True)))
    return pattern.sub(lambda match: replacements[match.group()], text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", help="Full source commit containing the exact r25 protocol.")
    mode.add_argument("--draft", action="store_true", help="Write an execution-blocked draft outside the repository.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    preflight_output = (args.preflight_output or output.with_suffix(".preflight.py")).resolve()
    if args.draft:
        assert not output.is_relative_to(root), "Draft notebooks must remain outside the repository."
        assert not preflight_output.is_relative_to(root), "Draft preflights must remain outside the repository."
        source = DRAFT_SOURCE
    else:
        assert re.fullmatch("[0-9a-f]{40}", args.source or ""), "--source must be a full commit SHA."
        source = args.source
    baseline_bytes = (root / BASE_NOTEBOOK).read_bytes()
    assert hashlib.sha256(baseline_bytes).hexdigest() == BASE_NOTEBOOK_SHA256
    baseline = json.loads(baseline_bytes)
    notebook = json.loads(baseline_bytes)
    protocol_path, protocol_sha = root / PROTOCOL_PATH, DRAFT_PROTOCOL
    if protocol_path.is_file():
        protocol_bytes = protocol_path.read_bytes()
        assert json.loads(protocol_bytes)['experiment_id'] == 'front-review-r25'
        protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    if not args.draft:
        assert protocol_path.is_file(), "Build the r25 protocol before pinning the notebook."
        for relative, expected in ((PROTOCOL_PATH, protocol_bytes), (BASE_NOTEBOOK, baseline_bytes)):
            actual = subprocess.check_output(["git", "show", source + ":" + relative], cwd=root)
            assert actual == expected, f"Working file differs from source commit: {relative}"
    for cell in notebook['cells']:
        cell['source'] = translated(''.join(cell['source']), source, protocol_sha).splitlines(keepends=True)
        if cell['cell_type'] == 'code':
            cell['outputs'], cell['execution_count'] = [], None
    notebook['cells'][0]['source'] = INTRO.splitlines(keepends=True)
    notebook['cells'][3]['source'] = PREFLIGHT_MARKDOWN.splitlines(keepends=True)
    notebook['cells'][15]['source'] = AFTER_RUN.splitlines(keepends=True)
    bootstrap, separator, _ = translated(''.join(baseline['cells'][4]['source']), source, protocol_sha).partition('input_preflight =')
    assert separator, "The preserved download cell must contain its preflight assignment."
    notebook['cells'][4]['source'] = (
        bootstrap + 'input_preflight = r"""' + INPUT_PREFLIGHT + '"""\n'
        'run([sys.executable, "-B", "-c", input_preflight, PROTOCOL_PATH, PROTOCOL_SHA256], cwd=REPO)\n'
        'assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()\n'
        'print("Verified clean pinned source:", actual_revision)\n'
    ).splitlines(keepends=True)
    notebook['metadata']['colab']['name'] = 'baomai_front_review_r25.ipynb'
    code_cells = []
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])
            compile(code, f'r25-cell-{index}', 'exec')
            code_cells.append(index)
            if index != 4:
                assert code == translated(''.join(baseline['cells'][index]['source']), source, protocol_sha)
    assert code_cells == [2, 4, 6, 8, 10, 12, 14]
    pins = {}
    for node in ast.walk(ast.parse(''.join(notebook['cells'][2]['source']))):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('SOURCE_REVISION', 'PROTOCOL_PATH', 'PROTOCOL_SHA256', 'EXPERIMENT_ID'):
                    pins[target.id] = ast.literal_eval(node.value)
    assert pins == {'SOURCE_REVISION': source, 'PROTOCOL_PATH': PROTOCOL_PATH,
                    'PROTOCOL_SHA256': protocol_sha, 'EXPERIMENT_ID': 'front-review-r25'}
    assignments = [node for node in ast.walk(ast.parse(''.join(notebook['cells'][4]['source'])))
                   if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'input_preflight'
                                                          for target in node.targets)]
    assert len(assignments) == 1 and ast.literal_eval(assignments[0].value) == INPUT_PREFLIGHT
    compile(INPUT_PREFLIGHT, 'r25-input-preflight', 'exec')
    ids = [cell['id'] for cell in notebook['cells']]
    assert len(ids) == len(set(ids))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    preflight_output.parent.mkdir(parents=True, exist_ok=True)
    preflight_output.write_text(INPUT_PREFLIGHT)
    print(json.dumps({'notebook': str(output), 'draft': args.draft,
                      'notebook_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                      'base_notebook_sha256': BASE_NOTEBOOK_SHA256,
                      'code_cells_compiled': code_cells, 'pins': pins,
                      'runtime_cells_preserved_except_round_names_and_pins': [2, 6, 8, 10, 12, 14],
                      'input_preflight': str(preflight_output),
                      'input_preflight_sha256': hashlib.sha256(INPUT_PREFLIGHT.encode()).hexdigest(),
                      'input_preflight_executed': False, 'training_executed': False}, indent=2))


if __name__ == '__main__':
    main()
