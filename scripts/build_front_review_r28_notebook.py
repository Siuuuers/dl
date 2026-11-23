#!/usr/bin/env python3
"""Build a source-pinned Colab notebook for fresh R26-versus-R28 supervision."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

BASE_NOTEBOOK = "notebooks/baomai_front_review_r26.ipynb"
BASE_NOTEBOOK_SHA256 = "37fe6bb66ce1c4d833b228466b3bce24769c42fa1b383d1384f465ca49561d4d"
BASE_SOURCE = "5c0a378dccfa16392eb3951531312efa3378889b"
BASE_PROTOCOL_SHA256 = "232f81cdf6d367fc9ece902e01b4ab5a03da5ae0898f5f75abae25ef5001372f"
PROTOCOL_PATH = "experiments/front-review-2026-11-23-r28.json"

INPUT_PREFLIGHT = r'''
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
root = Path.cwd()
sys.path.insert(0, str(root / 'scripts'))
from evaluate_pose import file_sha256, preflight_dataset
from front_review_r28_training import CONTROL, BASE_PROTOCOL, verify_candidate
from stage_pose_training import source_state
from rescore_reference_amendment import load_api, verify_amendment
from replay_corner_reference import reference_counts

def read_bound(descriptor):
    path = root / descriptor['path']
    assert file_sha256(path) == descriptor['sha256'], descriptor['path']
    return json.loads(path.read_text())

protocol_path = root / sys.argv[1]
assert file_sha256(protocol_path) == sys.argv[2]
protocol = json.loads(protocol_path.read_text())
assert protocol['experiment_id'] == 'front-review-r28'
assert protocol['recipe_id'] == 'reviewed-incremental-fronts-seed46-v1'
assert protocol['recipe_sha256'] == 'f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8'
candidate = root / protocol['candidate']['path']
read_bound(protocol['candidate'])
inspection = verify_candidate(root, candidate)
state = source_state(root, candidate)
initial = preflight_dataset(root, require_weights=True)
assert initial['model_sha256'] == protocol['initial_checkpoint_sha256']
assert state['source']['initial_checkpoint_sha256'] == initial['model_sha256']
assert inspection['training_readiness'] == 'ready_for_controlled_incremental_experiment'
assert inspection['full_dataset_certified'] is False
assert inspection['control']['manifest'] == CONTROL['manifest']
assert inspection['control']['manifest_sha256'] == CONTROL['sha256']
assert inspection['control']['supervision'] == 'r26_predecessor_control'
assert inspection['retained_legacy_image_ids'] == ['new_000015']
assert inspection['excluded_training_image_ids'] == protocol['excluded_training_image_ids']
assert len(inspection['excluded_training_image_ids']) == 31
assert inspection['retained_photo_labels_identical_between_arms'] is True
assert inspection['excluded_photo_labels_identical_between_arms'] is True
assert set(protocol['roles']) == {'reference', 'candidate'}
assert protocol['roles']['reference']['supervision'] == 'r26_predecessor_control'
assert protocol['roles']['candidate']['supervision'] == 'accepted_fronts_treatment'
previous_protocol = read_bound(BASE_PROTOCOL)
for role, spec in protocol['roles'].items():
    assert spec['arm'] == role
    assert all(state['source'][key] == spec[key] for key in
               ('candidate_manifest', 'candidate_manifest_sha256', 'candidate_id'))
    assert state['counts'][role] == spec['counts']
    assert state['counts'][role]['train']['images'] == 137
    assert state['counts'][role]['train']['empty_images'] == 13
    assert state['counts'][role]['val']['images'] == 20
    assert state['counts'][role]['val']['objects'] == 664
assert state['counts']['reference'] == previous_protocol['roles']['candidate']['counts']
assert state['counts']['reference'] == state['counts']['candidate']
review = read_bound(protocol['source_candidate'])
review_path = root / protocol['source_candidate']['path']
assert review['review_id'] == 'fronts-2026-11-23-r28'
delta = read_bound(protocol['training_input_delta'])
assert state['source']['training_input_delta']['report'] == delta
assert state['source']['training_input_delta']['sha256'] == protocol['training_input_delta']['sha256']
assert delta['summary']['changed_rows'] > 0
assert delta['summary']['added_rows'] == delta['summary']['deleted_rows'] == 0
assert delta['summary']['classes_visibility_preserved'] is True
control_path = root / CONTROL['manifest']
control_manifest = json.loads(control_path.read_text())
control_changes = {x['image_id']: x for x in control_manifest['changes']}
held = set(inspection['retained_legacy_image_ids'] + inspection['excluded_training_image_ids'])
observed_delta = []
for record in state['records']:
    image_id, split = record['id'], record['split']
    assert record['original_size'] == [1280, 720]
    original = root / 'data' / record['label']
    a, b = [state['plans'][role][record['label']]['source'].read_bytes()
            for role in ('reference', 'candidate')]
    change = control_changes.get(image_id)
    expected_control = control_path.parent / change['candidate_label'] if change else original
    assert a == expected_control.read_bytes(), 'Reference arm must use exact R26 supervision.'
    if split == 'val':
        assert a == b == original.read_bytes()
    elif image_id in held:
        assert a == b
    else:
        assert b == (review_path.parent / record['label']).read_bytes()
        assert len(a.splitlines()) == len(b.splitlines())
        observed_delta.extend((image_id, i, old.decode(), new.decode())
                              for i, (old, new) in enumerate(zip(a.splitlines(), b.splitlines()), 1)
                              if old != new)
assert observed_delta == [(x['image_id'], x['row'], x['before_line'], x['after_line']) for x in delta['rows']]
assert protocol['evaluation']['current_review_reference'] == protocol['source_candidate']
assert protocol['evaluation']['current_review_reference_is_exploratory'] is True
assert protocol['complete_review_counts'] == review['expected_counts']
assert review['expected_counts']['val']['objects'] == 661
assert protocol['execution']['same_runtime_required'] is True
assert protocol['execution']['historical_runtime_match_required'] is False

_, api = load_api(root)
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
delta_output = Path(sys.argv[3])
delta_bytes = (root / protocol['training_input_delta']['path']).read_bytes()
if delta_output.exists():
    assert delta_output.read_bytes() == delta_bytes, 'Existing input-delta report belongs to different labels.'
else:
    delta_output.parent.mkdir(parents=True, exist_ok=True)
    delta_output.write_bytes(delta_bytes)
print(json.dumps({'status': inspection['training_readiness'],
    'reference_supervision': 'R26', 'candidate_supervision': 'R28',
    'training_counts': state['counts'], 'complete_review_counts': review['expected_counts'],
    'original_images_verified': len(state['records']), 'weights_verified': initial['weights_verified'],
    'training_changes_vs_r26': delta['summary'], 'training_input_delta': str(delta_output),
    'changed_training_rows': [x['image_id'] + ':' + str(x['row']) for x in delta['rows']],
    'original_trainer_validation_byte_identical': True,
    'development_review_is_exploratory_only': True,
    'held_training_photos': inspection['retained_legacy_image_ids'],
    'old_photos_excluded_from_treatment': len(inspection['excluded_training_image_ids']),
    'evaluation_references': references, 'training_executed': False}, indent=2))
'''


def translated(value, source, protocol_sha):
    replacements = {BASE_SOURCE: source, BASE_PROTOCOL_SHA256: protocol_sha,
                    "r26": "r28", "round 26": "round 28", "Round 26": "Round 28"}
    pattern = re.compile("|".join(re.escape(v) for v in sorted(replacements, key=len, reverse=True)))
    return pattern.sub(lambda m: replacements[m.group()], value)


def publication_sources(root, protocol, fixed_files):
    """Include the label/evidence bytes needed by the pinned preflight replay."""
    source_files = set(fixed_files)
    source_files.add("scripts/rank_front_geometry_candidates.py")
    source_files.add(protocol['control']['manifest'])
    candidate_path = Path(protocol['candidate']['path'])
    review_path = Path(protocol['source_candidate']['path'])
    candidate = json.loads((root / candidate_path).read_bytes())
    review = json.loads((root / review_path).read_bytes())
    source_files.update((candidate_path.parent / item['candidate_label']).as_posix()
                        for item in candidate['changes'])
    source_files.update((review_path.parent / item['label_path']).as_posix()
                        for item in review['files'])

    def descriptors(value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and re.fullmatch(r'[0-9a-f]{64}', str(value.get('sha256', ''))):
                yield value
            for item in value.values():
                yield from descriptors(item)
        elif isinstance(value, list):
            for item in value:
                yield from descriptors(item)

    pending, expanded = [protocol, candidate, review], set()
    while pending:
        document = pending.pop()
        for item in descriptors(document):
            relative = item['path']
            source_files.add(relative)
            # Follow new review ledgers; inherited rounds are already committed.
            if (relative.startswith('results/review/front-review-20261123-r28/')
                    and relative.endswith('.json') and relative not in expanded):
                expanded.add(relative)
                pending.append(json.loads((root / relative).read_bytes()))
    return sorted(source_files)


def build_document(root, source, *, verify_source=True):
    base_bytes = (root / BASE_NOTEBOOK).read_bytes()
    assert hashlib.sha256(base_bytes).hexdigest() == BASE_NOTEBOOK_SHA256
    baseline = json.loads(base_bytes)
    notebook = json.loads(base_bytes)
    protocol_path = root / PROTOCOL_PATH
    protocol = json.loads(protocol_path.read_bytes())
    assert protocol['experiment_id'] == 'front-review-r28'
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    delta_path = root / protocol['training_input_delta']['path']
    delta = json.loads(delta_path.read_bytes())
    assert hashlib.sha256(delta_path.read_bytes()).hexdigest() == protocol['training_input_delta']['sha256']
    source_files = [PROTOCOL_PATH, BASE_NOTEBOOK, protocol['candidate']['path'],
                    protocol['source_candidate']['path'], protocol['training_input_delta']['path'],
                    "scripts/front_review_r28.py", "scripts/front_review_r28_training.py",
                    "scripts/build_front_review_r28_notebook.py", "scripts/stage_pose_training.py",
                    "scripts/verify_incremental_pose_candidate.py", "scripts/pose_checkpoint.py",
                    "scripts/train_pose_comparison.py"]
    source_files = publication_sources(root, protocol, source_files)
    lfs_paths = []
    if verify_source:
        for relative in source_files:
            actual = subprocess.check_output(["git", "show", source + ":" + relative], cwd=root)
            payload = (root / relative).read_bytes()
            pointer = re.fullmatch(rb'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n', actual)
            if pointer:
                assert hashlib.sha256(payload).hexdigest() == pointer[1].decode() and len(payload) == int(pointer[2]), f"Git LFS object differs: {relative}"
                lfs_paths.append(relative)
            else:
                assert actual == payload, f"Working file differs from source commit: {relative}"
    for cell in notebook['cells']:
        cell['source'] = translated(''.join(cell['source']), source, protocol_sha).splitlines(keepends=True)
        if cell['cell_type'] == 'code':
            cell['outputs'], cell['execution_count'] = [], None
    count, photos = delta['summary']['changed_rows'], delta['summary']['changed_photos']
    notebook['cells'][0]['source'] = f'''# Cassette fronts - R28 training comparison

**1. Select a GPU runtime (A100 recommended). 2. Select Run all. 3. Return `baomai-front-review-r28-results.zip`.**

This notebook trains **R26 supervision** and **R28 supervision** together. The exact treatment is **{count} coordinate-amended training rows across {photos} photographs**, including accepted R27 changes that have not yet been trained. Each arm independently starts from the original epoch-173 initializer. Both use seed 46, 30 epochs, batch 4, input 1280, FP32 and AdamW at initial learning rate 0.0001. The final epoch is selected in advance; the resource trial does not initialize either full run.

Both arms retain all 137 training photographs, including 13 empty scenes. Photo 015 and the 31 excluded old-series treatment photographs use identical R26 labels in both arms. The original trainer validation remains 20 images / 664 fronts. Development annotation changes do not enter training. Post-return scoring preserves the frozen 666-front primary and 663-front historical references, with the R28 review reported separately as exploratory evidence.

The earlier R25/R26 comparison used different GPUs. Here, both fresh arms and the resource trial must have the same recorded hardware/software runtime. A100 is recommended, but another CUDA GPU is valid if the complete pair runs in that same runtime. Completing this notebook does not automatically replace the retained PR35 model.

Google Drive stores verified completed bundles so they can be recovered. Keep the final ZIP even if the browser download succeeds. `training-input-delta.json` is also saved separately; its complete contents are embedded in each staged manifest inside the ZIP.
'''.splitlines(keepends=True)
    notebook['cells'][3]['source'] = '''## 2. Verify the exact experiment

Download the 157 original PNGs and the original initializer. Verify the accepted R28 review, the R26 predecessor, every selected label, and the exact training-row differences. The reference arm must use R26 supervision; the treatment arm must use eligible R28 supervision. Held/excluded photos remain identical, and every trainer-validation label remains original.

Verify the frozen 666/663 evaluation references separately. Save the complete before/after training-input report outside the repository. A failed input check stops before training. These checks establish which bytes will be trained; they do not certify every physical annotation.
'''.splitlines(keepends=True)
    bootstrap, separator, _ = translated(''.join(baseline['cells'][4]['source']), source, protocol_sha).partition('input_preflight =')
    assert separator
    notebook['cells'][4]['source'] = (
        bootstrap + 'TRAINING_DELTA = RUN_BASE / "training-input-delta.json"\n'
        'input_preflight = r"""' + INPUT_PREFLIGHT + '"""\n'
        'run([sys.executable, "-B", "-c", input_preflight, PROTOCOL_PATH, PROTOCOL_SHA256, str(TRAINING_DELTA)], cwd=REPO)\n'
        'save_to_drive(TRAINING_DELTA)\n'
        'assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()\n'
        'print("Verified clean pinned source:", actual_revision)\n'
    ).splitlines(keepends=True)
    # Runtime, training and packaging remain the established paired workflow.
    role_code = ''.join(notebook['cells'][8]['source'])
    role_code = role_code.replace("    assert receipt['arm'] == role\n",
        "    assert receipt['arm'] == role\n"
        "    assert receipt['training_labels']['role'] == spec['supervision']\n")
    role_code = role_code.replace('print("Starting:", folder, role, recipe_id, flush=True)',
        'print("Starting:", folder, CONTEXT["document"]["roles"][role]["display_name"], recipe_id, flush=True)')
    notebook['cells'][8]['source'] = role_code.splitlines(keepends=True)
    notebook['cells'][9]['source'] = '''## 5. Train the two missing arms

Train **R26 supervision** and **R28 supervision** with identical numerical settings, independently loading the original epoch-173 initializer. Both arms and their resource trial must share the recorded hardware/software runtime. Only the saved training-coordinate amendments differ between the two arms; trainer validation remains original. Completed runs are verified before reuse, including their exact supervision identities.
'''.splitlines(keepends=True)
    packaging = ''.join(notebook['cells'][12]['source'])
    marker = "check_incremental_pair(bundles, CONTEXT)\n"
    assert packaging.count(marker) == 1
    packaging = packaging.replace(marker, marker +
        "delta = json.loads(TRAINING_DELTA.read_text())\n"
        "for bundle in bundles.values():\n"
        "    embedded = bundle['staging']['source']['training_input_delta']\n"
        "    assert embedded['report'] == delta and embedded['sha256'] == sha256(TRAINING_DELTA)\n")
    notebook['cells'][12]['source'] = packaging.splitlines(keepends=True)
    notebook['cells'][15]['source'] = '''### After the run

Return `baomai-front-review-r28-results.zip`. It contains both final checkpoints, exact staging/label identities, the complete input delta, epoch results, raw logs and training receipts. The two arms must remain together.

Compare the fresh R26-supervision and R28-supervision models under identical evaluation rules, together with retained PR35. Keep the frozen 666/663 scores separate from the current R28 exploratory review. Report missed targets, unmatched outputs, duplicates, empty scenes and both-corner success within 5 and 10 original-image pixels; include eligible misses in denominators and inspect gained/lost identities.

The within-run comparison tests the saved training-coordinate changes. It does not establish new-session reliability, and development-label corrections cannot be reported as model learning. No threshold, seed or epoch search and no automatic model promotion follows from the run.
'''.splitlines(keepends=True)
    notebook['metadata']['colab']['name'] = 'baomai_front_review_r28.ipynb'
    code_cells = []
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'r28-cell-{i}', 'exec')
            code_cells.append(i)
    assert code_cells == [2, 4, 6, 8, 10, 12, 14]
    for i in (2, 6, 10, 14):
        assert ''.join(notebook['cells'][i]['source']) == translated(''.join(baseline['cells'][i]['source']), source, protocol_sha)
    pins = {}
    for node in ast.walk(ast.parse(''.join(notebook['cells'][2]['source']))):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('SOURCE_REVISION', 'PROTOCOL_PATH', 'PROTOCOL_SHA256', 'EXPERIMENT_ID'):
                    pins[target.id] = ast.literal_eval(node.value)
    assert pins == {'SOURCE_REVISION': source, 'PROTOCOL_PATH': PROTOCOL_PATH,
                    'PROTOCOL_SHA256': protocol_sha, 'EXPERIMENT_ID': 'front-review-r28'}
    compile(INPUT_PREFLIGHT, 'r28-input-preflight', 'exec')
    ids = [cell['id'] for cell in notebook['cells']]
    assert len(ids) == len(set(ids))
    return notebook, {'draft': not verify_source, 'pins': pins, 'source_files_verified': source_files if verify_source else [],
                      'git_lfs_pointer_paths_verified': lfs_paths,
                      'base_notebook_sha256': BASE_NOTEBOOK_SHA256, 'code_cells_compiled': code_cells,
                      'runtime_cells_preserved_except_round_names_and_pins': [2, 6, 10, 14],
                      'training_changes_vs_r26': delta['summary'],
                      'input_preflight_sha256': hashlib.sha256(INPUT_PREFLIGHT.encode()).hexdigest(),
                      'input_preflight_executed': False, 'training_executed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source-revision', required=True, help='Full published source commit containing this exact experiment.')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-output', type=Path)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    assert re.fullmatch(r'[0-9a-f]{40}', args.source_revision), 'Use a full source commit SHA.'
    notebook, receipt = build_document(args.root.resolve(), args.source_revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    preflight = args.preflight_output or args.output.with_suffix('.preflight.py')
    preflight.parent.mkdir(parents=True, exist_ok=True)
    preflight.write_text(INPUT_PREFLIGHT)
    receipt.update(notebook=str(args.output), notebook_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                   input_preflight=str(preflight))
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
