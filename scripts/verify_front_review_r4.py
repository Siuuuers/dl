#!/usr/bin/env python3
"""Reconstruct and verify the independently versioned r4 review experiment."""

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from front_review_experiment import FrontReviewExperiment, apply_operations

REVIEW = {'path': 'data/reviews/fronts-2026-11-18-r4/manifest.json', 'sha256': 'ec40bab297e928c5a19a82dbc418548e2c35d4fff9764bf0e2c8879cc40ec773'}
PARENT = {'path': 'data/reviews/fronts-2026-11-18/manifest.json', 'sha256': '256b750a0466b281750085f90acfa3056ee9320f44b45067a0e63415afc6ac35'}
CANDIDATE_ID = 'front-review-2026-11-18-r4'
CANDIDATE = f"data/candidates/{CANDIDATE_ID}/manifest.json"
PROTOCOL_PATH = 'experiments/front-review-2026-11-18-r4.json'
EXPERIMENT_ID = 'front-review-r4-seed46-v3'
SOURCE_REVISION = '36bb0bad2d93242720000111c9cb5ed2a3f79ed3'

EXPERIMENT = FrontReviewExperiment(
    review=REVIEW, parent=PARENT, candidate_id=CANDIDATE_ID,
    protocol_path=PROTOCOL_PATH, experiment_id=EXPERIMENT_ID,
    source_revision=SOURCE_REVISION, round_number=4, experiment_version=3,
    serialization_repairs=(('new_000061', 10), ('new_000119', 26)),
    protocol_limits=(
        'This prepares a controlled supervision comparison; it is not a training result or a claim of improvement.',
        'Only the requested photos were newly reviewed in r4. Other labels inherit the recovered r3 review and its physical uncertainty.',
        'The three specifically requested tiny-row deletions are user adjudications, not a new size filter or proof of physical absence.',
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
