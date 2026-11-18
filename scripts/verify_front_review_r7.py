#!/usr/bin/env python3
"""Verify the r7 follow-up against the preserved r6 review and fixed experiment."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from front_review_experiment import FrontReviewExperiment
from verify_pose_candidate import require
from verify_reconciled_pose_labels import bound

REVIEW = {'path':'data/reviews/fronts-2026-11-19-r7/manifest.json','sha256':'dab46852de4dece61520190807a5ca48d4b51b9daf7e3b793d83645e6d088845'}
PARENT = {'path':'data/reviews/fronts-2026-11-19-r6/manifest.json','sha256':'6d6591b8fa19614cbd10d3f2edb3a7b45848dcb2a2e0a2dea578452067d65094'}
CANDIDATE_ID = 'front-review-2026-11-19-r7'
CANDIDATE = f'data/candidates/{CANDIDATE_ID}/manifest.json'
PROTOCOL_PATH = 'experiments/front-review-2026-11-19-r7.json'
SOURCE_REVISION = '497da71bfdc07b49571a9eee3748b575817fca02'

class FollowupReviewExperiment(FrontReviewExperiment):
    def read_review(self, root):
        result = super().read_review(root)
        review, status, records, *_ = result
        root = Path(root).resolve()
        parent = json.loads(bound(root, self.parent).read_text())
        parents = {r['image_id']:r for r in parent['files']}
        coverage = json.loads(bound(root, review['request_coverage']).read_text())
        operations = json.loads(bound(root, review['operations']).read_text())
        targeted = set(coverage['targeted_image_ids'])
        require(coverage['base_review'] == self.parent and coverage['review_id'] == review['review_id']
                and review['requested_image_ids'] == sorted(targeted), 'r7 follow-up baseline or scope differs')
        by_id = {r['id']:r for r in records}
        entries = coverage['images']
        require(len(entries) == 157 and {e['image_id'] for e in entries} == set(by_id)
                and review['complete_atlas_image_ids'] == sorted(by_id), 'r7 atlas coverage differs')
        for e in entries:
            iid = e['image_id']; base = parents[iid]
            require(e['source_image_sha256'] == by_id[iid]['image_sha256']
                    and e['base_label_sha256'] == base['label_sha256']
                    and e['inherited_coverage'] == parent['request_coverage']
                    and e['split'] == by_id[iid]['split'], f'r7 coverage source differs: {iid}')
            fresh = iid in targeted
            require(e['new_full_frame_inspection'] is fresh
                    and e['inspection_origin'] == ('targeted_r7_original_pixel_review' if fresh else 'inherited_complete_r6_review')
                    and all(type(n) is int and 1 <= n <= base['objects'] for n in e['reviewed_base_rows'])
                    and (bool(e['reviewed_base_rows']) if fresh else not e['reviewed_base_rows']),
                    f'r7 inspection claim differs: {iid}')
        for op in operations:
            require(op['image_id'] in targeted, 'r7 operation outside follow-up scope')
            if op['action'] == 'replace':
                before, after = op['before_line'].split(), op['after_line'].split()
                groups = [g for g,a,b in [('box',1,5),('left',5,8),('right',8,11)] if before[a:b] != after[a:b]]
            else:
                groups = ['box','left','right'] if op['action'] == 'add' else []
            require(op['changed_groups'] == groups, 'r7 field-highlighting metadata differs from exact label changes')
        for c in status['cases']:
            require(all(type(n) is int and 1 <= n <= parents[c['image_id']]['objects'] for n in c['base_rows'])
                    and all(f in ('box','left','right') for f in c['fields']), 'r7 case identity differs')
        restored = root / 'data/reviews/fronts-2026-11-19-r7/labels/train/old_000013.txt'
        require(restored.read_bytes() == (root/'data/labels/train/old_000013.txt').read_bytes(),
                'r7 user-confirmed old_000013 restoration differs from original bytes')
        return result

EXPERIMENT = FollowupReviewExperiment(
    review=REVIEW, parent=PARENT, candidate_id=CANDIDATE_ID,
    protocol_path=PROTOCOL_PATH, experiment_id='front-review-r7-seed46-v6',
    source_revision=SOURCE_REVISION, round_number=7, experiment_version=6,
    serialization_repairs=(), review_development_labels=True,
    protocol_limits=(
        'The complete157-photo r6 review is preserved. R7 directly reviews seven photos prompted by six new screenshots and the old_000013 confirmation; unchanged photos inherit that earlier review.',
        'Visible target material and actual lower-front turns govern annotations. Approximate user marks are checked against original pixels; hidden corners are not guessed.',
        'Every held training photograph retains exact PR35 labels in both arms. Supported review edits on those photos are not treatment-only changes.',
        'Original664-front training validation and the fixed666/663 scoring references remain unchanged. The separately versioned current development review is exploratory.',
        'PR72 has been audited with fresh fixed-reference inference and does not support replacing PR35. The new supervision package is a hypothesis to test, not a demonstrated improvement.',
        'No training or automatic model promotion is performed during annotation preparation. Reused development scenes do not establish unseen-session reliability.',
    ),
)
read_review = EXPERIMENT.read_review
compose = EXPERIMENT.compose
protocol_document = EXPERIMENT.protocol_document
verify_candidate = EXPERIMENT.verify_candidate
build = EXPERIMENT.build

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--build',action='store_true')
    a=p.parse_args()
    print(json.dumps(build(a.root) if a.build else verify_candidate(a.root),indent=2))
