#!/usr/bin/env python3
"""Verify the complete r6 visual review and its controlled training composition."""

import argparse
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from front_review_experiment import FrontReviewExperiment, apply_operations
from verify_pose_candidate import require
from verify_reconciled_pose_labels import bound

REVIEW = {'path': 'data/reviews/fronts-2026-11-19-r6/manifest.json', 'sha256': '6d6591b8fa19614cbd10d3f2edb3a7b45848dcb2a2e0a2dea578452067d65094'}
PARENT = {'path': 'data/reviews/fronts-2026-11-19-r5/manifest.json', 'sha256': '666a7d85ec2783ab1a7c65a544abcfd444c37c07096703d04720dc132ea56636'}
CANDIDATE_ID = 'front-review-2026-11-19-r6'
CANDIDATE = f'data/candidates/{CANDIDATE_ID}/manifest.json'
PROTOCOL_PATH = 'experiments/front-review-2026-11-19-r6.json'
EXPERIMENT_ID = 'front-review-r6-seed46-v5'
SOURCE_REVISION = 'f6a8e9560c188fffe0a89eeee8e19c435d116800'


class CompleteReviewExperiment(FrontReviewExperiment):
    def read_review(self, root):
        result = super().read_review(root)
        review, status, records, *_ = result
        root = Path(root).resolve()
        coverage = json.loads(bound(root, review['request_coverage']).read_text())
        parent = json.loads(bound(root, self.parent).read_text())
        parents = {r['image_id']: r for r in parent['files']}
        operations = json.loads(bound(root, review['operations']).read_text())
        field_slices = (('box', 1, 5), ('left', 5, 8), ('right', 8, 11))
        for operation in operations:
            action = operation['action']
            if action == 'replace':
                before, after = operation['before_line'].split(), operation['after_line'].split()
                changed = [name for name, start, end in field_slices
                           if before[start:end] != after[start:end]]
            else:
                changed = [name for name, *_ in field_slices] if action == 'add' else []
            require(operation.get('changed_groups', []) == changed,
                    f"r6 field-highlighting metadata differs from exact label changes: {operation['image_id']}")
        entries = coverage['images']
        by_id = {r['id']: r for r in records}
        for case in status['cases']:
            rows, fields = case.get('base_rows'), case.get('fields')
            base_count = parents[case['image_id']]['objects']
            require(isinstance(rows, list)
                    and all(type(row) is int and 1 <= row <= base_count for row in rows)
                    and len(rows) == len(set(rows))
                    and isinstance(fields, list)
                    and all(field in ('box', 'left', 'right') for field in fields)
                    and len(fields) == len(set(fields)),
                    f"r6 case has invalid immutable rows or annotation fields: {case['case_id']}")
            region = case.get('region_xyxy')
            if not rows or region is not None:
                width, height = by_id[case['image_id']]['original_size']
                require(isinstance(region, list) and len(region) == 4
                        and all(type(value) in (int, float) and math.isfinite(value) for value in region)
                        and 0 <= region[0] < region[2] <= width
                        and 0 <= region[1] < region[3] <= height,
                        f"r6 case without base rows requires a valid original-pixel region: {case['case_id']}")
        require(coverage['base_review'] == self.parent
                and coverage['review_id'] == review['review_id']
                and len(entries) == len(by_id) == 157
                and {e['image_id'] for e in entries} == set(by_id),
                'r6 visual coverage does not match all original photos')
        for entry in entries:
            image_id = entry['image_id']
            source, base = by_id[image_id], parents[image_id]
            require(entry['split'] == source['split']
                    and entry['source_image_sha256'] == source['image_sha256']
                    and entry['base_label_sha256'] == base['label_sha256']
                    and entry['reviewed_base_rows'] == list(range(1, base['objects'] + 1))
                    and entry['visual_review_completed'] is True
                    and entry['full_frame_inspected'] is True
                    and entry['status'] in ('reviewed_no_change', 'reviewed_changed', 'reviewed_with_uncertainty')
                    and entry['reviewer'] and entry['notes'],
                    f'r6 visual coverage has an incomplete or mismatched entry: {image_id}')
        require(review['requested_image_ids'] == sorted(by_id), 'r6 declared scope is not the complete source cohort')
        return result


EXPERIMENT = CompleteReviewExperiment(
    review=REVIEW, parent=PARENT, candidate_id=CANDIDATE_ID,
    protocol_path=PROTOCOL_PATH, experiment_id=EXPERIMENT_ID,
    source_revision=SOURCE_REVISION, round_number=6, experiment_version=5,
    serialization_repairs=(), review_development_labels=True,
    protocol_limits=(
        'All 157 original photographs received a fresh visual pass; inspection is not physical certification.',
        'Corrections are supported nominal estimates on original pixels. Blur and physical surface ownership remain explicitly uncertain in the case ledger.',
        'A pending training case retains the entire photograph\'s exact PR35 labels in both arms. It is not removed or converted into an empty negative.',
        'Development review changes are versioned independently and remain exploratory. The original 664-front training-validation labels and fixed 666/663 scoring references are unchanged.',
        'The old-series front-rim versus projecting-support interpretation remains held where the originals cannot settle ownership.',
        'This prepares a controlled supervision comparison. No training result, model promotion, or claim of improved unseen-session reliability is implied.',
    ),
)
read_review = EXPERIMENT.read_review
compose = EXPERIMENT.compose
protocol_document = EXPERIMENT.protocol_document
verify_candidate = EXPERIMENT.verify_candidate
build = EXPERIMENT.build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build(args.root) if args.build else verify_candidate(args.root), indent=2))


if __name__ == '__main__':
    main()
