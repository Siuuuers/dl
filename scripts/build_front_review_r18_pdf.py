#!/usr/bin/env python3
"""Render only the saved r18 coordinate changes, with no uncertainty flags."""
from build_front_review_r16_pdf import main


if __name__ == '__main__':
    main(default_review='data/reviews/fronts-2026-11-22-r18',
         default_ledger='results/review/front-review-20261122-r18',
         default_output='output/pdf/front-review-r18-changed-photos.pdf',
         default_changes_only=True)
