# R46 additional-front review

All 507 current fronts in twelve original photographs were visually reviewed.
29 rows change: 27 corner positions and 17 box boundaries. All 17 named requests
are recorded. There are no new/deleted objects or visibility changes this round.

`audit-ledger.json` contains every current row, exact before/after text, requested
fields and physical reasons. `outlines.json` holds nine illustrative contour
traces, two deliberately open. They are not segmentation labels.

`proposals/` contains all twelve label files, including held photos. **Do not use
that directory as a training split.** Only `complete-candidates/labels/` passes
the recorded whole-image annotation gate. It still contains only frame_0383,
with exactly the same 43 labels used in PR133. Eleven photos and 328 source-row
field holds remain. The original 137/20 split is unchanged.

The full historical manifest is deterministically rebuilt from the pinned R44
archive plus this ledger rather than duplicating its 508 identities in Git.
`release-status.json` pins the generated manifest and every exported label file.
See [review and notebook instructions](../../../docs/front-review-r46.md).
