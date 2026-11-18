#!/usr/bin/env python3
"""Verify r6 staging, frozen references, and rejection of isolated artifact tampering.

Run with --output pointing to a new directory outside the checkout.
No model is loaded or trained, and original labels are never edited.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--output', type=Path, required=True,
                    help='fresh directory outside the checkout for staging and verification.json')
args = parser.parse_args()
ROOT = args.root.resolve()
OUTPUT = args.output.resolve()
assert not OUTPUT.exists(), 'Choose a fresh output directory.'
assert not OUTPUT.is_relative_to(ROOT) and not ROOT.is_relative_to(OUTPUT), 'Output overlaps the checkout.'
OUTPUT.mkdir(parents=True)
sys.path.insert(0, str(ROOT / 'scripts'))
from stage_pose_training import stage_datasets
from verify_front_review_r4 import verify_candidate as verify_r4
from verify_front_review_r5 import verify_candidate as verify_r5
from verify_front_review_r6 import EXPERIMENT, REVIEW, CANDIDATE, PROTOCOL_PATH

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def encoded(data):
    return (json.dumps(data, indent=2, ensure_ascii=False) + '\n').encode()

def expect_rejection(call, fragment):
    try:
        call()
    except ValueError as error:
        assert fragment in str(error), str(error)
        return str(error)
    raise AssertionError('An invalid artifact was accepted')

reference = json.loads((ROOT / 'data/manifest.json').read_text())
records = reference['records']
inspection = EXPERIMENT.verify_candidate(ROOT)
candidate = json.loads((ROOT / CANDIDATE).read_text())
protocol = json.loads((ROOT / PROTOCOL_PATH).read_text())
review = json.loads((ROOT / REVIEW['path']).read_text())
status = json.loads((ROOT / review['review_status']['path']).read_text())
expected_held = {c['image_id'] for c in status['cases']
                 if c['split'] == 'train' and c['status'] != 'resolved'}
assert expected_held == set(candidate['retained_legacy_image_ids'])
assert not expected_held.intersection(candidate['changed_images_vs_control'])
old_train = {r['id'] for r in records if r['split'] == 'train' and r['id'].startswith('old_')}
assert len(old_train) == 31 and old_train.issubset(expected_held)
control_path = ROOT / candidate['control']['manifest']
control = json.loads(control_path.read_text())
control_changes = {c['image_id']: c for c in control['changes']}
original_labels_before = {r['id']: sha(ROOT / 'data' / r['label']) for r in records}

stage_path = OUTPUT / 'staging'
assert not stage_path.exists(), 'Use the existing receipt; do not overwrite prior verification.'
staged = stage_datasets(ROOT, ROOT / CANDIDATE, stage_path)
assert staged['status'] == 'staged' and staged['actual_original_pngs_verified'] == 157
val_count, held_count, exploratory_changed = 0, 0, []
review_files = {f['image_id']: f for f in review['files']}
for record in records:
    original = (ROOT / 'data' / record['label']).read_bytes()
    arms = [(stage_path / arm / record['label']).read_bytes() for arm in ('reference', 'candidate')]
    if record['split'] == 'val':
        assert arms[0] == arms[1] == original
        val_count += 1
        reviewed = (ROOT / REVIEW['path']).parent / review_files[record['id']]['label_path']
        if reviewed.read_bytes() != original:
            exploratory_changed.append(record['id'])
    if record['id'] in expected_held:
        change = control_changes.get(record['id'])
        expected = ((control_path.parent / change['candidate_label']).read_bytes()
                    if change else original)
        assert arms[0] == arms[1] == expected
        held_count += 1
assert val_count == 20 and held_count == len(expected_held)
assert staged['counts']['reference']['val']['objects'] == 664
assert staged['counts']['candidate']['val']['objects'] == 664
assert exploratory_changed, 'This pass is expected to contain separately reviewed development changes.'
assert protocol['evaluation']['current_review_reference'] == REVIEW
prior_protocol = json.loads((ROOT / 'experiments/front-review-2026-11-19-r5.json').read_text())
for role, expected_objects in [('primary', 666), ('diagnostic', 663)]:
    for suffix in ('reference', 'reference_sha256', 'counts'):
        key = role + '_' + suffix
        assert protocol['evaluation'][key] == prior_protocol['evaluation'][key]
    assert sha(ROOT / protocol['evaluation'][role + '_reference']) == protocol['evaluation'][role + '_reference_sha256']
    assert protocol['evaluation'][role + '_counts']['objects'] == expected_objects

regressions = {'r4': verify_r4(ROOT)['status'], 'r5': verify_r5(ROOT)['status']}
negative_checks = {}
with tempfile.TemporaryDirectory(prefix='r6-isolated-pipeline-', dir=OUTPUT) as directory:
    snapshot = Path(directory)
    for name in ('data', 'docs', 'metadata', 'experiments', 'results', 'scripts'):
        def ignore(path, names):
            return {'images'} if Path(path) == ROOT / 'data' else {'__pycache__'}
        shutil.copytree(ROOT / name, snapshot / name, ignore=ignore)
    altered = json.loads((snapshot / CANDIDATE).read_text())
    change = altered['changes'][0]
    path = (snapshot / CANDIDATE).parent / change['candidate_label']
    lines = path.read_text().splitlines()
    tokens = lines[0].split()
    tokens[1] = f'{float(tokens[1]) + 0.0001:.9f}'
    lines[0] = ' '.join(tokens)
    path.write_text('\n'.join(lines) + '\n')
    change['candidate_label_sha256'] = sha(path)
    (snapshot / CANDIDATE).write_bytes(encoded(altered))
    negative_checks['candidate_rehash_cannot_bypass_review_replay'] = expect_rejection(
        lambda: EXPERIMENT.verify_candidate(snapshot), 'candidate differs from exact reviewed/held composition')

    # Exercise field-highlighting integrity beyond the outer manifest hash.
    altered_review = json.loads((snapshot / REVIEW['path']).read_text())
    op_path = snapshot / altered_review['operations']['path']
    operations = json.loads(op_path.read_text())
    operation = next(o for o in operations if o['action'] == 'replace')
    operation['changed_groups'] = []
    op_path.write_bytes(encoded(operations))
    altered_review['operations']['sha256'] = sha(op_path)
    (snapshot / REVIEW['path']).write_bytes(encoded(altered_review))
    altered_experiment = replace(EXPERIMENT, review={'path': REVIEW['path'], 'sha256': sha(snapshot / REVIEW['path'])})
    negative_checks['stale_field_highlighting_rejected'] = expect_rejection(
        lambda: altered_experiment.read_review(snapshot), 'field-highlighting metadata differs')

assert {r['id']: sha(ROOT / 'data' / r['label']) for r in records} == original_labels_before
receipt = {
    'status': 'verified', 'training_run': False,
    'review': REVIEW, 'candidate_manifest_sha256': sha(ROOT / CANDIDATE),
    'protocol_sha256': sha(ROOT / PROTOCOL_PATH),
    'original_images_verified': 157, 'all_original_label_bytes_unchanged': True,
    'staged_counts': staged['counts'], 'original_validation_photos_byte_identical_in_both_arms': val_count,
    'held_training_photos_exact_pr35_in_both_arms': held_count,
    'held_old_series_photos': len(old_train),
    'reviewed_development_photos_different_from_original_but_not_staged': exploratory_changed,
    'frozen_reference_object_counts': {'primary': 666, 'diagnostic': 663},
    'prior_round_verifiers': regressions, 'negative_checks': negative_checks,
    'staging_manifest_sha256': sha(stage_path / 'staging-manifest.json'),
    'source_files_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
        ROOT / 'scripts/check_front_review_r6.py',
        ROOT / 'scripts/front_review_experiment.py', ROOT / 'scripts/verify_front_review_r6.py',
        ROOT / 'scripts/stage_pose_training.py', ROOT / 'scripts/train_pose_comparison.py',
        ROOT / 'scripts/verify_incremental_pose_candidate.py']},
    'limitations': ['Mechanical and provenance verification does not certify physical annotation correctness.',
                   'No model has been trained or evaluated by this integration check.']
}
(OUTPUT / 'verification.json').write_bytes(encoded(receipt))
print(json.dumps(receipt, indent=2))
