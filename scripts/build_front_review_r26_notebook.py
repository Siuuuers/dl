#!/usr/bin/env python3
"""Build the pinned r26 Colab notebook from the unchanged r25 runtime cells."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


BASE_NOTEBOOK = "notebooks/baomai_front_review_r25.ipynb"
BASE_NOTEBOOK_SHA256 = "11d0cac7a492f7234e4778490c1ef70d445d5aa309f2bca11af4d02329553877"
BASE_SOURCE = "86a7ba078b348f2b1cbacf73b4c271e9216fdb63"
BASE_PROTOCOL_SHA256 = "80c3f2849890a664db9fea857342c4d78d38ba736b7a108ce13905d74aedb287"
PROTOCOL_PATH = "experiments/front-review-2026-11-23-r26.json"


INPUT_PREFLIGHT = r'''
import json
from collections import Counter
from pathlib import Path
import sys
sys.dont_write_bytecode = True
root = Path.cwd()
sys.path.insert(0, str(root / 'scripts'))
from evaluate_pose import file_sha256, preflight_dataset, parse_pose_labels
from verify_incremental_pose_candidate import verify_incremental_candidate
from stage_pose_training import source_state
from front_review_experiment import apply_operations
from front_review_r25 import STANDARDS
from verify_pose_candidate import totals
from rescore_reference_amendment import load_api, verify_amendment
from replay_corner_reference import reference_counts

def read_bound(descriptor):
    path = root / descriptor['path']
    assert file_sha256(path) == descriptor['sha256'], descriptor['path']
    return json.loads(path.read_text())

protocol_path = root / sys.argv[1]
assert file_sha256(protocol_path) == sys.argv[2]
protocol = json.loads(protocol_path.read_text())
assert protocol['experiment_id'] == 'front-review-r26'
assert protocol['recipe_id'] == 'reviewed-incremental-fronts-seed46-v1'
assert protocol['recipe_sha256'] == 'f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8'
candidate = root / protocol['candidate']['path']
read_bound(protocol['candidate'])
inspection = verify_incremental_candidate(root, candidate)
assert inspection['training_readiness'] == 'ready_for_controlled_incremental_experiment'
assert inspection['full_dataset_certified'] is False
assert inspection['named_rows_reviewed'] == 22
assert inspection['training_analogue_rows_reviewed'] == 3
assert inspection['protected_standard_rows'] == 11
assert inspection['classes_visibility_preserved'] is True
assert inspection['confirmed_reference_rows_preserved'] is True
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
previous_protocol = read_bound({'path': 'experiments/front-review-2026-11-23-r25.json',
    'sha256': '80c3f2849890a664db9fea857342c4d78d38ba736b7a108ce13905d74aedb287'})
for role, spec in protocol['roles'].items():
    assert spec['arm'] == role
    assert all(state['source'][key] == spec[key] for key in
               ('candidate_manifest', 'candidate_manifest_sha256', 'candidate_id'))
    assert state['counts'][role] == spec['counts'] == previous_protocol['roles'][role]['counts']
    assert state['counts'][role]['train']['images'] == 137
    assert state['counts'][role]['train']['empty_images'] == 13
    assert state['counts'][role]['val']['images'] == 20
    assert state['counts'][role]['val']['objects'] == 664
assert state['counts']['reference']['val'] == state['counts']['candidate']['val']

review_path = root / protocol['source_candidate']['path']
review = read_bound(protocol['source_candidate'])
assert review['review_id'] == 'fronts-2026-11-23-r26'
assert review['parent'] == {'path': 'data/reviews/fronts-2026-11-23-r25/manifest.json',
    'sha256': 'de80c21ed4b148efe6446e66b8a15f75c3b6363daf9c03d0b88a29e187483517'}
parent = read_bound(review['parent'])
parent_path = root / review['parent']['path']
assert parent['review_id'] == 'fronts-2026-11-23-r25'
operations = read_bound(review['operations'])
row_maps = read_bound(review['row_maps'])
for descriptor in review['review_evidence']:
    read_bound(descriptor)
operation_counts = Counter((item['split'], item['action']) for item in operations)
assert operation_counts == Counter({('train', 'replace'): 3, ('val', 'replace'): 16, ('val', 'delete'): 1})
training_operations = [item for item in operations if item['split'] == 'train']
assert {(item['image_id'], item['base_row']) for item in training_operations} == {
    ('new_000013', 29), ('new_000013', 30), ('new_000013', 31)}
point_rows, box_rows = [], []
for item in training_operations:
    before, after = item['before_line'].split(), item['after_line'].split()
    assert len(before) == len(after) == 11
    assert before[0] == after[0] and before[7:] == after[7:]
    assert before[2] == after[2] and before[4] == after[4]
    assert before[5:7] != after[5:7]
    point_rows.append(item['base_row'])
    if before[1:5] != after[1:5]:
        old_min = (float(before[1]) - float(before[3]) / 2) * 1280
        new_min = (float(after[1]) - float(after[3]) / 2) * 1280
        old_max = (float(before[1]) + float(before[3]) / 2) * 1280
        new_max = (float(after[1]) + float(after[3]) / 2) * 1280
        assert new_min > old_min and abs(new_max - old_max) <= 0.0001
        box_rows.append(item['base_row'])
assert len(point_rows) == 3 and len(box_rows) == 2

records = state['records']
assert len(records) == 157
files = {item['image_id']: item for item in review['files']}
parent_files = {item['image_id']: item for item in parent['files']}
held = set(inspection['retained_legacy_image_ids'] + inspection['excluded_training_image_ids'])
parsed, changed_images, protected = {}, set(), 0
for record in records:
    image_id, split = record['id'], record['split']
    assert record['original_size'] == [1280, 720]
    path = review_path.parent / record['label']
    old_path = parent_path.parent / record['label']
    assert file_sha256(path) == files[image_id]['label_sha256']
    assert file_sha256(old_path) == parent_files[image_id]['label_sha256']
    actual, before = path.read_bytes(), old_path.read_bytes()
    edits = [item for item in operations if item['image_id'] == image_id]
    assert all(item['split'] == split for item in edits)
    replay, mapping = apply_operations(before, edits, image_id)
    assert replay == actual and mapping == row_maps[image_id]
    if actual != before:
        changed_images.add(image_id)
    parsed[image_id] = parse_pose_labels(actual.decode(), 1280, 720, str(path))
    for standard_image, standard_row in STANDARDS:
        if standard_image != image_id:
            continue
        matches = [item for item in mapping['rows'] if item.get('base_row') == standard_row]
        assert len(matches) == 1
        assert actual.splitlines()[matches[0]['current_row'] - 1] == before.splitlines()[standard_row - 1]
        protected += 1
    for role in ('reference', 'candidate'):
        plan = state['plans'][role][record['label']]
        if split == 'val':
            original = root / 'data' / record['label']
            assert plan['source'] == original
            assert file_sha256(original) == plan['sha256'] == record['label_sha256']
        elif image_id in held:
            other = state['plans']['reference'][record['label']]
            assert plan['sha256'] == other['sha256']
            assert plan['source'].read_bytes() == other['source'].read_bytes()
        elif role == 'candidate':
            assert plan['source'].read_bytes() == actual
counts = totals(records, parsed)
assert counts == review['expected_counts'] == protocol['complete_review_counts']
assert counts['train'] == parent['expected_counts']['train']
assert counts['val']['images'] == 20 and counts['val']['objects'] == 661
assert set(review['changed_image_ids']) == changed_images
assert protected == len(STANDARDS) == 11
assert protocol['evaluation']['current_review_reference'] == protocol['source_candidate']
assert protocol['evaluation']['current_review_reference_is_exploratory'] is True

_, api = load_api(root)
assert protocol['evaluation']['diagnostic_reference_required'] is True
references = {}
for role, expected_objects in (('primary', 666), ('diagnostic', 663)):
    reference_path = root / protocol['evaluation'][role + '_reference']
    assert file_sha256(reference_path) == protocol['evaluation'][role + '_reference_sha256']
    reference, gold, *_ = verify_amendment(root, reference_path.parent, initial, api)
    tally, groups = reference_counts(gold)
    assert tally == reference['counts'] == protocol['evaluation'][role + '_counts']
    assert tally['images'] == 20 and tally['objects'] == expected_objects
    assert groups == reference['objects_by_locatable_corners']
    references[role] = {'counts': tally, 'objects_by_locatable_corners': groups}
print(json.dumps({'status': inspection['training_readiness'],
    'training_counts': state['counts'], 'complete_review_counts': counts,
    'weights_verified': initial['weights_verified'], 'original_images_verified': len(records),
    'training_changes': {'image_id': 'new_000013', 'rows': sorted(point_rows),
                         'left_corner_replacements': len(point_rows), 'xmin_rows': sorted(box_rows)},
    'development_replacements': operation_counts['val', 'replace'],
    'development_deletions': operation_counts['val', 'delete'],
    'protected_standard_rows': protected, 'changed_photos': len(changed_images),
    'original_training_validation_byte_identical': True,
    'development_review_is_exploratory_only': True,
    'held_training_photos': inspection['retained_legacy_image_ids'],
    'old_photos_excluded_from_treatment': len(inspection['excluded_training_image_ids']),
    'evaluation_references': references}, indent=2))
'''


INTRO = '# Baomai cassette fronts - round 26\n\n**Choose a GPU runtime, select Run all, and return `baomai-front-review-r26-results.zip`.**\n\nR26 preserves the accepted R25 review and adds a small, supported training intervention in **photo 013, rows 29–31: three lower-left corner corrections and two left box-boundary corrections**. Separately, the development review contains **16 row replacements and one deletion**. Those development changes refine the scoring reference; they do not enter training. All **11 accepted standard rows** retain their exact lines.\n\nBoth arms use **137 training photos, including 13 empty scenes**. The reference uses PR35 supervision and the candidate uses accumulated eligible R26 labels; photo 015 and the 31 old-series exclusions retain identical PR35 labels in both arms. Each arm independently starts from original **epoch 173**, with **seed 46, 30 epochs, batch 4, input 1280, FP32 and AdamW at initial learning rate 0.0001**. The resource trial never initializes a full run, and the final epoch is selected in advance. Both arms are trained together in the same recorded runtime; an old control is not silently substituted.\n\nOriginal trainer validation remains **20 images and 664 fronts**. Post-training scoring retains the frozen **666-front primary** and **663-front historical** references. The new **661-front R26 review** is exploratory and has its own identity. The completed PR101 evaluation retained **PR35**: R25 did not establish a replacement and did not improve on R24 overall. This new experiment tests whether the supported training corrections help; it makes no advance claim of improvement or of generalization to a new camera session.\n'

PREFLIGHT_MARKDOWN = '## 2. Verify the exact experiment\n\nDownload and verify the 157 original PNGs and epoch-173 initializer. Replay every R26 operation against the frozen R25 parent, including the single development-row deletion and its row mapping. Verify the three training left-corner changes, two left box-boundary changes, 16 development replacements, and all 11 preserved standards directly from the saved bytes.\n\nCheck the actual labels selected for both arms and the inherited training exclusions. Every trainer-validation image and label must remain byte-identical to the original data. Verify the frozen 666/663 scoring references separately from the current exploratory review. A failed check stops before training. These checks verify that the intended labels are used; they do not independently certify every physical annotation.\n'

AFTER_RUN = '### After the run\n\nCompare both returned final-epoch models with retained PR35 on unchanged frozen references, then report the separately named R26 exploratory scores. Include misses, unmatched outputs, duplicates, empty scenes and both-corner successes within 5 and 10 original-image pixels. Retain eligible misses in each denominator and inspect gained/lost identities.\n\nThe within-run candidate/control comparison tests all accumulated eligible review changes. To examine the small R25-to-R26 training change, compare their actual input/runtime receipts and score both models against identical reference bytes. Development-only corrections cannot establish model learning. Keep the prior R25 result and every reference version intact; no automatic replacement follows from completing training. No threshold, seed or epoch search is introduced.\n'


def translated(text, source, protocol_sha):
    replacements = {
        "experiments/front-review-2026-11-23-r25.json": PROTOCOL_PATH,
        "baomai-front-review-r25-20261123": "baomai-front-review-r26-20261123",
        "r25": "r26", "r24": "r25", "PR99": "PR101",
        "round 25": "round 26", "Round 25": "Round 26",
        BASE_SOURCE: source, BASE_PROTOCOL_SHA256: protocol_sha,
    }
    pattern = re.compile("|".join(re.escape(value) for value in sorted(replacements, key=len, reverse=True)))
    return pattern.sub(lambda match: replacements[match.group()], text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-revision", required=True,
                        help="Full published source commit containing the exact r26 protocol and builder.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    preflight_output = (args.preflight_output or output.with_suffix(".preflight.py")).resolve()
    assert re.fullmatch("[0-9a-f]{40}", args.source_revision), "--source-revision must be a full commit SHA."
    source = args.source_revision
    baseline_bytes = (root / BASE_NOTEBOOK).read_bytes()
    assert hashlib.sha256(baseline_bytes).hexdigest() == BASE_NOTEBOOK_SHA256
    baseline = json.loads(baseline_bytes)
    notebook = json.loads(baseline_bytes)
    protocol_path = root / PROTOCOL_PATH
    assert protocol_path.is_file(), "Build the r26 protocol before pinning the notebook."
    protocol_bytes = protocol_path.read_bytes()
    assert json.loads(protocol_bytes)['experiment_id'] == 'front-review-r26'
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    source_files = [PROTOCOL_PATH, BASE_NOTEBOOK, "scripts/front_review_r26.py",
                    "scripts/build_front_review_r26_notebook.py"]
    for relative in source_files:
        actual = subprocess.check_output(["git", "show", source + ":" + relative], cwd=root)
        assert actual == (root / relative).read_bytes(), f"Working file differs from source commit: {relative}"
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
    notebook['metadata']['colab']['name'] = 'baomai_front_review_r26.ipynb'
    code_cells = []
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            code = ''.join(cell['source'])
            compile(code, f'r26-cell-{index}', 'exec')
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
                    'PROTOCOL_SHA256': protocol_sha, 'EXPERIMENT_ID': 'front-review-r26'}
    assignments = [node for node in ast.walk(ast.parse(''.join(notebook['cells'][4]['source'])))
                   if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'input_preflight'
                                                          for target in node.targets)]
    assert len(assignments) == 1 and ast.literal_eval(assignments[0].value) == INPUT_PREFLIGHT
    compile(INPUT_PREFLIGHT, 'r26-input-preflight', 'exec')
    ids = [cell['id'] for cell in notebook['cells']]
    assert len(ids) == len(set(ids))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    preflight_output.parent.mkdir(parents=True, exist_ok=True)
    preflight_output.write_text(INPUT_PREFLIGHT)
    print(json.dumps({'notebook': str(output), 'draft': False,
                      'notebook_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                      'base_notebook_sha256': BASE_NOTEBOOK_SHA256,
                      'code_cells_compiled': code_cells, 'pins': pins,
                      'runtime_cells_preserved_except_round_names_and_pins': [2, 6, 8, 10, 12, 14],
                      'input_preflight': str(preflight_output),
                      'input_preflight_sha256': hashlib.sha256(INPUT_PREFLIGHT.encode()).hexdigest(),
                      'input_preflight_executed': False, 'training_executed': False}, indent=2))


if __name__ == '__main__':
    main()
