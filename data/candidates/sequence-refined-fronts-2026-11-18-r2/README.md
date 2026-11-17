# Visible cassette fronts: 18 November 2026, round 2

This complete training-label review contains **137 photos, 3,797 fronts, 6,860
locatable endpoints, 734 missing endpoints and 13 empty photos**. It follows
`sequence-refined-fronts-2026-11-18` with **387 box refinements and one deletion
across 28 photos**. All retained class and keypoint tokens are preserved exactly.

The user-confirmed deletion is `new_000049` parent row 18. Later rows in that
photo shift up by one; every original row identity and prior addition ID remains
in the manifest, including the deleted identity with a null current row.

The [review report](../../../results/review/front-refinement-20261118-r2/README.md)
explains the changes, the single focused PDF and all remaining questions.
The [status ledger](../../../results/review/front-refinement-20261118-r2/review-status.json)
records **10 case groups on 9 training photos**. Cyan means no recorded open
hold; it does not certify all inherited annotations.

This directory is the complete review, not the notebook's bounded treatment.
The separately versioned `sequence-front-corrections-2026-11-18` experiment keeps
exact PR35 labels in both arms for every held training photo, and selects this
review only on photos without holds. Its training counts therefore differ.
The requested human PDF review remains the gate before merging this draft;
preparing and running the controlled notebook is already authorized.

## Reproduce or verify

From the repository root, with original images materialized:

```bash
python scripts/compose_sequence_front_review.py \
  --review results/review/front-refinement-20261118-r2/accepted-review.json \
  --verify-only
python scripts/check_front_refinement_review.py --root .
```

Omit `--verify-only` on the first command to reconstruct the dated review from
its exact parent and operation ledger. The second command independently replays
the edits using only the Python standard library. The original 17 November and
first 18 November composer outputs remain byte-identical.

The original 20-photo training-validation labels remain at 664 fronts. The
separate `sequence-refined-reference-2026-11-18` development reference remains
at 666 fronts. No model training, inference or promotion occurs in this review.

Manifest SHA256:
`de3cd6e5605ec7bd1250b53b4eb52c3a688e2431a19f4016bbfa08f2e09b6849`.
