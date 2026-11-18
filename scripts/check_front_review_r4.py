#!/usr/bin/env python3
"""Check the real r4 staging boundary and fail-closed provenance without a GPU."""

import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True

from evaluate_pose import preflight_dataset
from pose_checkpoint import _check_staging, training_label_identity
from replay_corner_reference import reference_counts
from rescore_reference_amendment import load_api, verify_amendment
from stage_pose_training import stage_datasets, verify_staging
from verify_front_review_r4 import CANDIDATE, PROTOCOL_PATH, REVIEW, read_review, verify_candidate
from verify_reconciled_pose_labels import encoded, sha


ROOT = Path(__file__).resolve().parents[1]


class R4Checks(unittest.TestCase):
    def snapshot(self):
        temporary = tempfile.TemporaryDirectory(prefix='r4-check-', dir=ROOT / 'tmp')
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / 'snapshot'
        for relative in ('data', 'docs', 'results/review', 'metadata', 'scripts', 'experiments'):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(ROOT / relative, target, copy_function=os.link)
        return root

    def replace(self, path, value):
        path.unlink()
        path.write_bytes(encoded(value))

    def test_actual_staging_holds_and_separate_references(self):
        receipt = verify_candidate(ROOT)
        protocol = json.loads((ROOT / PROTOCOL_PATH).read_text())
        review, status, records, _, _, _, reviewed = read_review(ROOT)
        held = set(receipt['retained_legacy_image_ids'])
        self.assertEqual(held, {'new_000014'})
        self.assertNotIn('new_000014', receipt['changed_images_vs_control'])
        self.assertEqual(review['expected_counts']['val']['objects'], 662)
        self.assertEqual(receipt['counts']['val']['objects'], 664)
        preflight = preflight_dataset(ROOT, require_weights=False)
        with tempfile.TemporaryDirectory(prefix='r4-staging-', dir=ROOT / 'tmp') as temporary:
            output = Path(temporary) / 'staged-data'
            stage_datasets(ROOT, ROOT / CANDIDATE, output)
            self.assertEqual(verify_staging(ROOT, ROOT / CANDIDATE, output)['status'], 'verified')
            staging = json.loads((output / 'staging-manifest.json').read_text())
            _check_staging(preflight, staging)
            self.assertEqual(staging['code_sha256']['scripts/verify_front_review_r4.py'],
                             sha(ROOT / 'scripts/verify_front_review_r4.py'))
            changed = []
            for record in records:
                reference = (output / 'reference' / record['label']).read_bytes()
                candidate = (output / 'candidate' / record['label']).read_bytes()
                if record['split'] == 'val':
                    self.assertEqual(reference, candidate)
                    self.assertEqual(candidate, (ROOT / 'data' / record['label']).read_bytes())
                elif record['id'] in held:
                    self.assertEqual(reference, candidate)
                else:
                    self.assertEqual(candidate, reviewed[record['id']]['bytes'])
                if reference != candidate:
                    changed.append(record['id'])
            self.assertEqual(sorted(changed), sorted(receipt['changed_images_vs_control']))
            for arm in ('reference', 'candidate'):
                self.assertEqual(staging['variants'][arm]['counts'], protocol['roles'][arm]['counts'])
                self.assertEqual(training_label_identity(staging, arm)['role'],
                                 'pr35_annotation_control' if arm == 'reference' else 'accepted_fronts_treatment')
            tampered = copy.deepcopy(staging)
            tampered['code_sha256']['scripts/verify_front_review_r4.py'] = '0' * 64
            (output / 'staging-manifest.json').write_bytes(encoded(tampered))
            with self.assertRaisesRegex(ValueError, 'configuration or code'):
                verify_staging(ROOT, ROOT / CANDIDATE, output)
        _, api = load_api(ROOT)
        for role, objects in (('primary', 666), ('diagnostic', 663)):
            path = ROOT / protocol['evaluation'][role + '_reference']
            reference, gold, *_ = verify_amendment(ROOT, path.parent, preflight, api)
            counts, groups = reference_counts(gold)
            self.assertEqual(counts, protocol['evaluation'][role + '_counts'])
            self.assertEqual(counts['objects'], objects)
            self.assertEqual(groups, reference['objects_by_locatable_corners'])

    def test_self_consistent_candidate_cannot_release_held_photo(self):
        root = self.snapshot()
        path = root / CANDIDATE
        candidate = json.loads(path.read_text())
        candidate['retained_legacy_image_ids'] = []
        candidate['unresolved_case_ids'] = []
        self.replace(path, candidate)
        protocol_path = root / PROTOCOL_PATH
        protocol = json.loads(protocol_path.read_text())
        protocol['candidate']['sha256'] = sha(path)
        protocol['retained_legacy_image_ids'] = []
        protocol['unresolved_case_ids'] = []
        for role in protocol['roles'].values():
            role['candidate_manifest_sha256'] = sha(path)
        self.replace(protocol_path, protocol)
        with self.assertRaisesRegex(ValueError, 'exact reviewed/held composition'):
            verify_candidate(root)

    def test_protocol_cannot_substitute_latest_review_or_promote(self):
        root = self.snapshot()
        path = root / PROTOCOL_PATH
        protocol = json.loads(path.read_text())
        protocol['evaluation']['primary_reference'] = REVIEW['path']
        protocol['evaluation']['primary_reference_sha256'] = REVIEW['sha256']
        protocol['execution']['automatic_promotion'] = True
        self.replace(path, protocol)
        with self.assertRaisesRegex(ValueError, 'fixed experiment scope'):
            verify_candidate(root)


if __name__ == '__main__':
    (ROOT / 'tmp').mkdir(exist_ok=True)
    unittest.main(verbosity=2)
