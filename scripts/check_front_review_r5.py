#!/usr/bin/env python3
"""Check actual r5 staging, stable row identities and candidate tamper rejection."""

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
from verify_front_review_r5 import CANDIDATE, PROTOCOL_PATH, read_review, verify_candidate
from verify_reconciled_pose_labels import encoded, sha

ROOT = Path(__file__).resolve().parents[1]


class R5Checks(unittest.TestCase):
    def test_actual_staging_and_stable_count_changes(self):
        receipt = verify_candidate(ROOT)
        protocol = json.loads((ROOT / PROTOCOL_PATH).read_text())
        review, status, records, _, _, _, reviewed = read_review(ROOT)
        held = set(receipt['retained_legacy_image_ids'])
        self.assertEqual(held, {'new_000023'})
        self.assertNotIn('new_000023', receipt['changed_images_vs_control'])
        self.assertIn('new_000014', receipt['changed_images_vs_control'])
        self.assertEqual(review['expected_counts']['val']['objects'], 662)
        self.assertEqual(receipt['counts']['val']['objects'], 664)
        maps = json.loads((ROOT / review['row_maps']['path']).read_text())
        self.assertEqual(maps['new_000014']['deleted_base_rows'], [11, 12])
        self.assertEqual(next(row['current_row'] for row in maps['new_000014']['rows'] if row.get('base_row') == 13), 11)
        added = [row for row in maps['new_000095']['rows'] if 'addition_id' in row]
        self.assertEqual(len(added), 1)
        self.assertEqual((added[0]['after_base_row'], added[0]['current_row']), (19, 20))
        self.assertEqual(next(row['current_row'] for row in maps['new_000095']['rows'] if row.get('base_row') == 20), 21)
        new_row = reviewed['new_000095']['bytes'].decode().splitlines()[19].split()
        self.assertEqual(list(map(float, new_row[5:])), [0] * 6)
        preflight = preflight_dataset(ROOT, require_weights=False)
        with tempfile.TemporaryDirectory(prefix='r5-staging-', dir=ROOT / 'tmp') as temporary:
            output = Path(temporary) / 'staged-data'
            stage_datasets(ROOT, ROOT / CANDIDATE, output)
            self.assertEqual(verify_staging(ROOT, ROOT / CANDIDATE, output)['status'], 'verified')
            staging = json.loads((output / 'staging-manifest.json').read_text())
            _check_staging(preflight, staging)
            for name in ['front_review_experiment.py', 'verify_front_review_r4.py', 'verify_front_review_r5.py']:
                self.assertEqual(staging['code_sha256']['scripts/' + name], sha(ROOT / 'scripts' / name))
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
                if reference != candidate: changed.append(record['id'])
            self.assertEqual(sorted(changed), sorted(receipt['changed_images_vs_control']))
            for arm in ['reference', 'candidate']:
                self.assertEqual(staging['variants'][arm]['counts'], protocol['roles'][arm]['counts'])
                self.assertEqual(training_label_identity(staging, arm)['role'],
                                 'pr35_annotation_control' if arm == 'reference' else 'accepted_fronts_treatment')
        _, api = load_api(ROOT)
        for role, expected_objects in [('primary', 666), ('diagnostic', 663)]:
            path = ROOT / protocol['evaluation'][role + '_reference']
            reference, gold, *_ = verify_amendment(ROOT, path.parent, preflight, api)
            counts, groups = reference_counts(gold)
            self.assertEqual(counts, protocol['evaluation'][role + '_counts'])
            self.assertEqual(counts['objects'], expected_objects)
            self.assertEqual(groups, reference['objects_by_locatable_corners'])

    def test_candidate_only_coordinate_edit_cannot_replace_review(self):
        with tempfile.TemporaryDirectory(prefix='r5-tamper-', dir=ROOT / 'tmp') as temporary:
            root = Path(temporary) / 'snapshot'
            for relative in ['data', 'docs', 'results/review', 'metadata', 'scripts', 'experiments']:
                target = root / relative; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(ROOT / relative, target, copy_function=os.link)
            path = root / CANDIDATE
            manifest = json.loads(path.read_text())
            change = next(item for item in manifest['changes'] if item['image_id'] == 'new_000095')
            label = path.parent / change['candidate_label']
            lines = label.read_text().splitlines()
            tokens = lines[0].split()
            tokens[5] = f'{float(tokens[5]) + .25 / 1280:.9f}'
            lines[0] = ' '.join(tokens)
            label.unlink(); label.write_text('\n'.join(lines) + '\n')
            change['candidate_label_sha256'] = sha(label)
            path.unlink(); path.write_bytes(encoded(manifest))
            protocol_path = root / PROTOCOL_PATH
            protocol = json.loads(protocol_path.read_text())
            protocol['candidate']['sha256'] = sha(path)
            for role in protocol['roles'].values(): role['candidate_manifest_sha256'] = sha(path)
            protocol_path.unlink(); protocol_path.write_bytes(encoded(protocol))
            with self.assertRaisesRegex(ValueError, 'exact reviewed/held composition'):
                verify_candidate(root)


if __name__ == '__main__':
    (ROOT / 'tmp').mkdir(exist_ok=True)
    unittest.main(verbosity=2)
