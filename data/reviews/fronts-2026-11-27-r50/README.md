# R50 development reference: one clarified bottom boundary

This directory continues the R48 proposal using the user's clarification of
`new_000010 / R38 row 029`. Its physical bottom changes from nominal original-image
`ymax=720` to `ymax=714`, excluding the lower neighboring cassette. The retained
visible left corner is at `(325,714)`; the right slot stays exactly `0 0 0`.

All 20 development photographs and 654 instance identities remain. Exactly one
box field changes, with no point, visibility, addition, deletion or training-label
change. The remaining development hold is `new_000089:R38:041 / xmax`. R48's
supplementary training proposals and holds are unchanged.

The user identifies the disputed material owner; the assistant checks the
original pixels and proposes the nominal coordinate. This is not independent
human certification of the entire reference. The reported roughly 2 px visual
uncertainty is an explanatory estimate, not a measured confidence interval.

- [Manifest](manifest.json): complete membership, source-image hashes, label hashes
  and parent reference.
- [Audit ledger](audit-ledger.json): exact before/after line, user feedback,
  reasoning and remaining hold.
- [Review guide](../../../docs/front-review-r50.md): the amendment PDF and fixed-cache
  sensitivity replay.
- [New error analysis](../../../docs/front-error-review-r50.md): all 119 original
  R38 review cases, with the current R50 status separately identified.

Run `python scripts/build_front_review_r50.py` from the repository root to verify
the exported labels against the pinned R48 parent. It never overwrites an
incompatible export. The old R38 and frozen 666/663 references remain intact.
