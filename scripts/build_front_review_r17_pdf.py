#!/usr/bin/env python3
"""Render the saved r17 edits with immutable r16 row provenance."""
from build_front_review_r16_pdf import main


if __name__ == '__main__':
    main(default_review='data/reviews/fronts-2026-11-22-r17',
         default_ledger='results/review/front-review-20261122-r17',
         default_output='output/pdf/front-review-r17-changed-photos.pdf')
