# Sequence-refined cassette fronts: 18 November 2026

This is the complete **137-photo training-label review** following the user's
second set of pixel and video-continuity corrections. It contains **3,798 front
instances, 6,860 locatable endpoints, 736 missing endpoints, and 13 empty photos**.
Relative to the 17 November review, 26 rows are replaced and three fronts are
added across nine photos. No row is deleted in this revision.

The candidate remains a review artifact awaiting the requested PDF review before
merge. It is not selected by an existing training notebook and does not certify
the whole dataset. Nine case groups on eight training photos remain recorded in
the [status ledger](../../../results/review/sequence-refinement-20261118/review-status.json).

The [review report](../../../docs/sequence-label-review-2026-11-18.md) explains
each correction, remaining questions, PDF page locations, and the saved-model
replay. The separately versioned development reference is
`sequence-refined-reference-2026-11-18`; original training-validation labels
remain unchanged.

## Reconstruct or verify

From the repository root, with the original images materialized:

```bash
python scripts/compose_sequence_front_review.py \
  --review results/review/sequence-refinement-20261118/accepted-review.json \
  --verify-only
```

Omit `--verify-only` to reconstruct the dated outputs from the exact parent and
operation ledger. The default composer invocation still reproduces the unchanged
17 November candidate. All source manifests, labels, original images, policy,
review sources, and status are hash-bound.

Every training TXT is present, including empty files. `files[].row_map` retains
original row identities and stable addition IDs. For this revision, `parent_row`
and `base_row` mean the current row in the immediate 17 November parent;
`candidate_row` means the displayed 18 November output row. New026 rows 54-56 are
the three additions. Existing row numbers do not shift because this pass has no
deletions.

Manifest SHA256:
`fb52d6946e598e977289ca025ac386c5044e9f15e9461b6e55f7c00a6b9c9171`.
