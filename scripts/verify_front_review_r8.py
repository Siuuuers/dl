#!/usr/bin/env python3
"""Replay r8 labels, coverage and four-extrema evidence against preserved r7."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from evaluate_pose import parse_pose_labels
from front_review_experiment import FrontReviewExperiment
from verify_pose_candidate import require
from verify_reconciled_pose_labels import bound

REVIEW = {'path': 'data/reviews/fronts-2026-11-19-r8/manifest.json', 'sha256': '07721c3e5ded57ae289650b9e0a96d424a793d595d769f42809c129e95061ca2'}
PARENT = {'path': 'data/reviews/fronts-2026-11-19-r7/manifest.json', 'sha256': 'dab46852de4dece61520190807a5ca48d4b51b9daf7e3b793d83645e6d088845'}
CANDIDATE_ID = 'front-review-2026-11-19-r8'
CANDIDATE = f'data/candidates/{CANDIDATE_ID}/manifest.json'
PROTOCOL_PATH = 'experiments/front-review-2026-11-19-r8.json'
SOURCE_REVISION = 'bb3b144372496781f8a892c758a64d1f0c27986d'


class ExtremaReviewExperiment(FrontReviewExperiment):
    def read_review(self, root):
        result = super().read_review(root)
        review, status, records, *_ = result
        root = Path(root).resolve()
        parent = json.loads(bound(root, self.parent).read_text())
        parents = {row['image_id']: row for row in parent['files']}
        coverage = json.loads(bound(root, review['request_coverage']).read_text())
        operations = json.loads(bound(root, review['operations']).read_text())
        bound(root, review['boundary_rubric'])
        inventory_descriptor = next(d for d in review['review_evidence']
                                    if d['path'].endswith('/inspection-artifacts.json'))
        inventory = json.loads(bound(root, inventory_descriptor).read_text())['artifacts']
        evidence_ids = {item['id'] for item in inventory}
        require(len(evidence_ids) == len(inventory), 'r8 repeated inspection artifact identity')
        by_id = {row['id']: row for row in records}
        require(coverage['base_review'] == self.parent and coverage['review_id'] == review['review_id']
                and coverage['source_revision'] == SOURCE_REVISION, 'r8 baseline or source differs')
        require(coverage['targeted_image_ids'] == review['requested_image_ids']
                == review['complete_atlas_image_ids'] == sorted(by_id), 'r8 whole-cohort scope differs')
        entries = coverage['images']
        require(len(entries) == 157 and {e['image_id'] for e in entries} == set(by_id),
                'r8 must record all 157 original photographs exactly once')
        for entry in entries:
            iid = entry['image_id']
            require(entry['source_image_sha256'] == by_id[iid]['image_sha256']
                    and entry['base_label_sha256'] == parents[iid]['label_sha256']
                    and entry['split'] == by_id[iid]['split']
                    and entry['inherited_coverage'] == parent['request_coverage'],
                    f'r8 coverage source differs: {iid}')
            require(entry['new_full_frame_inspection'] is True
                    and entry['inspection_origin'] == 'r8_original_pixel_review'
                    and entry['reviewed_base_rows'] == list(range(1, parents[iid]['objects'] + 1))
                    and entry['viewed_evidence_ids']
                    and set(entry['viewed_evidence_ids']) <= evidence_ids,
                    f'r8 inspection coverage or evidence differs: {iid}')
        for op in operations:
            if op['action'] == 'replace':
                before, after = op['before_line'].split(), op['after_line'].split()
                groups = [g for g, a, b in [('box', 1, 5), ('left', 5, 8), ('right', 8, 11)]
                          if before[a:b] != after[a:b]]
            else:
                groups = ['box', 'left', 'right'] if op['action'] == 'add' else []
            require(op['changed_groups'] == groups and op['evidence_ids']
                    and set(op['evidence_ids']) <= evidence_ids,
                    'r8 field highlighting or recorded visual evidence differs')
            if 'box' in groups:
                witnesses = op['extreme_witnesses']
                require(set(witnesses) == {'left', 'top', 'right', 'bottom'},
                        'r8 changed box lacks four recorded extrema')
                box = parse_pose_labels(op['after_line'], 1280, 720)[0]['bbox_xyxy']
                extremes = [witnesses['left'][0], witnesses['top'][1],
                            witnesses['right'][0], witnesses['bottom'][1]]
                require(max(abs(a - b) for a, b in zip(box, extremes)) < .0001,
                        'r8 recorded extrema do not define the serialized box')
                require(all(box[0] - .0001 <= x <= box[2] + .0001
                            and box[1] - .0001 <= y <= box[3] + .0001
                            for x, y in witnesses.values()), 'r8 witness falls outside its box')
        for case in status['cases']:
            require(all(type(n) is int and 1 <= n <= parents[case['image_id']]['objects']
                        for n in case['base_rows'])
                    and all(f in ('box', 'left', 'right') for f in case['fields']),
                    'r8 case identity or field differs')
        require(status['pending_train_photos'] == sorted({c['image_id'] for c in status['cases']
                    if c['split'] == 'train' and c['status'] != 'resolved'}),
                'r8 whole-photo training holds differ from case ledger')
        confirmed = root / 'data/reviews/fronts-2026-11-19-r8/labels/train/old_000013.txt'
        require(confirmed.read_bytes() == (root / 'data/labels/train/old_000013.txt').read_bytes(),
                'r8 changed the user-confirmed original old_000013 bytes')
        return result


EXPERIMENT = ExtremaReviewExperiment(
    review=REVIEW, parent=PARENT, candidate_id=CANDIDATE_ID,
    protocol_path=PROTOCOL_PATH, experiment_id='front-review-r8-seed46-v7',
    source_revision=SOURCE_REVISION, round_number=8, experiment_version=7,
    serialization_repairs=(), review_development_labels=True,
    protocol_limits=(
        'All 157 original photographs received a fresh visual pass; changed boxes record four observed extrema. Inspection coverage and mechanical checks do not certify every physical boundary.',
        'The target is the visible cassette front and its own frontal rim. Continuous rack members do not set its bounds. Actual lower-front corners are judged independently; unavailable corners remain canonical missing values.',
        'Every held training photograph retains exact PR35 labels in both arms. Old-series flange ownership and intended landmark definition are distinguished explicitly.',
        'Original 664-front training validation and fixed 666/663 scoring references remain unchanged. The separately versioned current development review is exploratory.',
        'PR74 fresh fixed-reference inference does not support replacing PR35. Scored box misses include overlap failures as well as absent retained predictions; keep the frozen matching rule unchanged.',
        'No training or automatic model promotion occurs during annotation preparation. The next notebook tests the eligible supervision package using the predeclared two-arm recipe; reused scenes do not establish unseen-session reliability.',
    ),
)
read_review = EXPERIMENT.read_review
compose = EXPERIMENT.compose
protocol_document = EXPERIMENT.protocol_document
verify_candidate = EXPERIMENT.verify_candidate
build = EXPERIMENT.build

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build(args.root) if args.build else verify_candidate(args.root), indent=2))
