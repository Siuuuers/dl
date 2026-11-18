#!/usr/bin/env python3
"""Reconstruct and verify the independently versioned r5 review experiment."""

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from front_review_experiment import FrontReviewExperiment, apply_operations

REVIEW = {'path': 'data/reviews/fronts-2026-11-19-r5/manifest.json', 'sha256': '666a7d85ec2783ab1a7c65a544abcfd444c37c07096703d04720dc132ea56636'}
PARENT = {'path': 'data/reviews/fronts-2026-11-18-r4/manifest.json', 'sha256': 'ec40bab297e928c5a19a82dbc418548e2c35d4fff9764bf0e2c8879cc40ec773'}
CANDIDATE_ID = 'front-review-2026-11-19-r5'
CANDIDATE = f"data/candidates/{CANDIDATE_ID}/manifest.json"
PROTOCOL_PATH = 'experiments/front-review-2026-11-19-r5.json'
EXPERIMENT_ID = 'front-review-r5-seed46-v4'
SOURCE_REVISION = 'dba4d685fa7a62dbda1ec32e5f40a44b42bd6054'

EXPERIMENT = FrontReviewExperiment(
    review=REVIEW, parent=PARENT, candidate_id=CANDIDATE_ID,
    protocol_path=PROTOCOL_PATH, experiment_id=EXPERIMENT_ID,
    source_revision=SOURCE_REVISION, round_number=5, experiment_version=4,
    serialization_repairs=(),
    protocol_limits=(
        'This prepares a controlled supervision comparison; it is not a training result or a claim of improvement.',
        'Only the 17 requested photos were newly reviewed in r5. Other labels inherit the r4 review and its physical uncertainty.',
        'The two requested row deletions in new_000014 are user adjudications, not a new size filter or proof of physical absence.',
        "A pending training case retains the entire photograph's exact PR35 labels in both arms; no uncertain region becomes an empty negative.",
        'The 20 separately reviewed development labels are not substituted for the original 664-front training-validation labels or the fixed scoring references.',
        'The development scenes have already informed annotation and model decisions; their scores do not establish unseen-session reliability.',
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
