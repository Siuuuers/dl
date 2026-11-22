#!/usr/bin/env python3
"""Build the pinned r23 Colab notebook from the unchanged r21 runtime cells."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


BASE_NOTEBOOK = "notebooks/baomai_front_review_r21.ipynb"
BASE_NOTEBOOK_SHA256 = "1de82d9dbfb53e331048f132d96df01272e2e732d48b1454bbf12c820b005f7a"
BASE_SOURCE = "c4fb76afe280552ed13ed3c9b78608deceb7b3c4"
BASE_PROTOCOL_SHA256 = "6e79440caa989de5327531ebde2242aa3ba7f25de7497ebac8542ee766400122"
PROTOCOL_PATH = "experiments/front-review-2026-11-22-r23.json"
DRAFT_SOURCE = "SOURCE_COMMIT_TO_BE_PINNED_AFTER_PUBLICATION"
DRAFT_PROTOCOL = "PROTOCOL_HASH_TO_BE_PINNED_AFTER_BUILD"


INPUT_PREFLIGHT = r'''
import json
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

protocol_path = root / sys.argv[1]
assert file_sha256(protocol_path) == sys.argv[2]
protocol = json.loads(protocol_path.read_text())
assert protocol['experiment_id'] == 'front-review-r23'
candidate = root / protocol['candidate']['path']
assert file_sha256(candidate) == protocol['candidate']['sha256']
inspection = verify_incremental_candidate(root, candidate)
assert inspection['training_readiness'] == 'ready_for_controlled_incremental_experiment'
assert inspection['full_dataset_certified'] is False
assert inspection['changed_rows'] == inspection['point_repairs'] == 4
assert inspection['changed_photos'] == 2
assert inspection['changed_box_rows'] == 0
assert inspection['confirmed_reference_rows_preserved'] is True
assert inspection['unchanged_review_files'] == 155
assert inspection['classes_boxes_left_points_visibility_preserved'] is True
assert inspection['development_labels_preserved'] is True
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
assert review['review_id'] == 'fronts-2026-11-22-r23'
assert review['parent'] == {
    'path': 'data/reviews/fronts-2026-11-22-r22/manifest.json',
    'sha256': 'a2af88f2e9e6a6e94db123d2d20f6d057f59062eaae294edbdf0ad94267e13ae'}
parent_path = root / review['parent']['path']
assert file_sha256(parent_path) == review['parent']['sha256']
parent = json.loads(parent_path.read_text())
assert parent['review_id'] == 'fronts-2026-11-22-r22'
targets = {
    ('new_000001', 27): (488, 518),
    ('new_000002', 25): (426, 543),
    ('new_000002', 26): (430, 571),
    ('new_000002', 27): (432, 600),
}
standards = [('new_000001', 1), ('new_000001', 19),
             ('new_000001', 25), ('new_000002', 28)]
records = state['records']
assert len(records) == 157
changed_rows, changed_images = set(), set()
review_lines, parent_lines = {}, {}
counts = {}
for record in records:
    image_id, split = record['id'], record['split']
    current_bytes = (review_path.parent / record['label']).read_bytes()
    parent_bytes = (parent_path.parent / record['label']).read_bytes()
    if current_bytes != parent_bytes:
        changed_images.add(image_id)
    if image_id not in {'new_000001', 'new_000002'}:
        assert current_bytes == parent_bytes
    if split == 'val':
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
        assert new[:8] == old[:8] and new[10] == old[10]
        for offset in (5, 8):
            assert new[offset + 2] in ('0', '2')
            if new[offset + 2] == '0':
                assert all(float(value) == 0 for value in new[offset:offset + 3])
                tally['missing_endpoints'] += 1
            else:
                tally['visible_endpoints'] += 1
        if key in targets:
            assert split == 'train' and old[10] == new[10] == '2'
            x, y = targets[key]
            assert abs(float(new[8]) * 1280 - x) <= 0.001
            assert abs(float(new[9]) * 720 - y) <= 0.001
            assert before != after
            changed_rows.add(key)
        else:
            assert after == before
assert changed_rows == set(targets)
assert changed_images == {'new_000001', 'new_000002'}
assert set(review['changed_image_ids']) == changed_images
assert len(records) - len(changed_images) == 155
for image_id, row in standards:
    assert review_lines[image_id][row - 1] == parent_lines[image_id][row - 1]
assert sum(bool(rows) for rows in review_lines.values()) == 141
assert sum(map(len, review_lines.values())) == 4448
assert counts == protocol['complete_review_counts']
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
                  'training_counts': state['counts'],
                  'complete_review_counts': counts,
                  'weights_verified': initial['weights_verified'],
                  'changed_rows': len(changed_rows), 'changed_photos': len(changed_images),
                  'changed_box_rows': 0, 'point_repairs': 4,
                  'unchanged_review_files': 155, 'confirmed_reference_rows_preserved': True,
                  'r22_development_labels_byte_identical': True,
                  'held_training_photos': inspection['retained_legacy_image_ids'],
                  'old_photos_excluded_from_treatment': len(inspection['excluded_training_image_ids']),
                  'evaluation_references': references}, indent=2))
'''


INTRO = '''# Baomai cassette fronts — round 23

**Choose a GPU runtime, select Run all, and return `baomai-front-review-r23-results.zip`.**

This notebook uses the completed r22 annotations plus four reviewed right-corner corrections: **001:27 and 002:25–27**. The targets are the actual lower corners of the cassette's front face. Boxes, left points, visibility, object identities and the accepted standards **001:1,19,25 and 002:28** remain unchanged. White/grey contrast helps inspection; it does not determine material ownership by itself. The completed r22 PDF and PR96 results remain preserved.

Both arms use the same **137 training photos, including 13 empty scenes**. The reference uses PR35 supervision and the candidate uses eligible r23 labels; photo 015 and the 31 old-series exclusions retain identical PR35 labels in both arms. Each arm starts independently from original **epoch 173**, with **seed 46, 30 epochs, batch 4, input 1280, FP32 and the established AdamW recipe**. The resource trial never initializes a full run, and the final epoch is selected in advance.

Original trainer validation remains **664 fronts**; evaluation retains the frozen **666-front primary** and **663-front historical** references. The **662-front r23 development review** is exploratory and has the same validation-label bytes as r22. **PR35 remains selected** pending the completed comparison. This run tests the accumulated eligible supervision, not the isolated effect of these four edits.
'''

PREFLIGHT_MARKDOWN = '''## 2. Verify the exact experiment

Download the 157 original PNGs and epoch-173 initializer, verify their hashes, and replay the r23 labels against the frozen r22 parent. Check the four right-point changes, every preserved field and row, all four accepted standards, and the other 155 unchanged label files.

Verify the actual labels selected for each arm, inherited training exclusions, unchanged trainer validation and both fixed scoring references. A failed check stops before training. The checks establish that the intended annotations are used; the recorded visual review establishes their meaning.
'''

AFTER_RUN = '''### After the run

Compare both fresh final-epoch models with retained PR35 on the fixed references. Report unmatched targets, unmatched outputs, duplicates, empty scenes and both-corner successes within 5 and 10 original-image pixels. Retain eligible missed objects in each success denominator and inspect gained and lost identities.

Keep the r23 exploratory scores separate. Reference-box changes can change IoU matching without moving a prediction, and conflicting rankings do not establish a replacement. Do not select a favorable threshold, seed or epoch after seeing the results. The notebook preserves the existing two-point architecture and training procedure; preparing it does not train or promote a model.
'''


def translated(text, source, protocol_sha):
    replacements = {
        "r21": "r23", "r20": "r22", "PR94": "PR96",
        "round 21": "round 23", "Round 21": "Round 23",
        BASE_SOURCE: source, BASE_PROTOCOL_SHA256: protocol_sha,
    }
    pattern = re.compile("|".join(re.escape(value) for value in sorted(replacements, key=len, reverse=True)))
    return pattern.sub(lambda match: replacements[match.group()], text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", help="Full source commit containing the exact r23 protocol.")
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
        assert json.loads(protocol_bytes)['experiment_id'] == 'front-review-r23'
        protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    if not args.draft:
        assert protocol_path.is_file(), "Build the r23 protocol before pinning the notebook."
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
    notebook['metadata']['colab']['name'] = 'baomai_front_review_r23.ipynb'
    code_cells = []
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])
            compile(code, f'r23-cell-{index}', 'exec')
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
                    'PROTOCOL_SHA256': protocol_sha, 'EXPERIMENT_ID': 'front-review-r23'}
    assignments = [node for node in ast.walk(ast.parse(''.join(notebook['cells'][4]['source'])))
                   if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'input_preflight'
                                                          for target in node.targets)]
    assert len(assignments) == 1 and ast.literal_eval(assignments[0].value) == INPUT_PREFLIGHT
    compile(INPUT_PREFLIGHT, 'r23-input-preflight', 'exec')
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
