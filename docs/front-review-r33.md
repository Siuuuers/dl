# R33 front review and pinned notebook handoff

## Published evidence
- R33 full review: `data/reviews/fronts-2026-11-25-r33/`.
- Eligible training overlay: `data/candidates/front-review-2026-11-25-r33/`.
- R33 boundary guidance: [annotation-boundaries-r33.md](annotation-boundaries-r33.md).
- Exact decisions, replay operations, review coverage and training delta: `results/review/front-review-20261125-r33/`.
- Fixed experiment: `experiments/front-review-2026-11-25-r33.json`.

The review covered 126 new-series photos and 4,178 labeled rows. Fourteen rows changed: eight eligible training rows in seven photos and six development rows. Twelve uncertain cases remain recorded without forced changes. No objects or class/visibility statuses changed in this round. Original splits, held and legacy supervision, trainer validation, and frozen evaluation references are preserved.

## Reproducibility and running
The exact R33 source tree `d3ae26eebf73872a34ff722b9d6c6bc6c8527e69` is published at commit `1acb4061a99a4484be6226db5dafddd0cb17c103`. All 719 embedded files from the original standalone notebook were recovered and hash-checked, and the resulting Git tree was verified against that original tree. The slim notebook fetches the published commit and reconstructs only the original local commit metadata (`d4b905dc0ed7728815108fed04486bc4215f7a2b`) so PR114's source identity and existing receipts remain compatible. The original standalone notebook is preserved in the prior handoff (SHA-256 `f939e8564249b9c43cd5a2704e5f3168288fdcbfbd8b2f145ccc0f138882de0e`).

[Open the R33 notebook in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/publish-r33-source-20261125/notebooks/baomai_front_review_r33.ipynb). PR114 already returned the original R33 GPU pair; do not retrain solely because of notebook publication. The experiment uses native loss in both arms, independent epoch-173 initialization, seed 46, identical training budget, and final-epoch checkpoints. Only the eight eligible training-boundary corrections differ.

## Future notebook rule
Publish the exact source, labels, manifests and experiment protocol first. Pin the immutable source commit in the notebook and verify the protocol/input hashes before training. Keep notebook cells for orchestration and run recovery; do not embed copies of repository label snapshots, source scripts or review ledgers once published. Keep PDFs and bulky binary artifacts outside review PRs. Never rewrite an already returned run's notebook/source identity or silently change training or evaluation settings.

This publication does not select a new checkpoint or rerun training. PR35 remains selected under frozen 666; historical 663 and revised development references remain distinct.
