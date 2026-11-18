# Evidence for the r4 relabeling pass

This pass starts from PR68 commit `36bb0bad2d93242720000111c9cb5ed2a3f79ed3`. The r3 parent is immutable. The current labels live in `data/reviews/fronts-2026-11-18-r4`.

- `user-requests.json`: 37 request groups across 31 photographs; `new_00002` resolves to `new_000002`.
- `operations.json`: exact before/after lines keyed by immutable parent rows; 521 visual replacements, 2 serialization-only replacements, 3 additions and 3 specific user-adjudicated deletions.
- `row-maps.json`: every parent/current identity, including new015 and new023 additions and deletion renumbering.
- `request-coverage.json`: each requested component is changed, explicitly retained after review, or held. The sole remaining requested uncertainty is new014 base11.
- `review-status.json`: the whole-photo training hold and inherited development amber rows.
- `group-a-review.json`, `group-b-review.json`, `group-c-review.json`: per-group evidence and limits. Intermediate crop filenames refer to session inspection files; the authoritative reproducible inputs are the pinned original PNGs and row operations.
- `independent-cross-review.json`: second review of new023's split and new051's disconnected visible patches.
- `serialization-repair-review.json`: exact Decimal checks of the two inherited numerical bounds and independent effective-label comparison.
- `new_000014-base11-review.png`: the preserved clipped-edge uncertainty crop.
- `user-references/`: the two supplied orange-outline screenshots, used as context while geometry is placed on original PNGs.
- `pdf-audit.json` and `pdf-visual-qa.json`: delivered PDF identity, page inventory and rendered review results.

The focused PDF is delivered separately. It shows all labels on every changed photo, with current rows in magenta when touched, inherited rows in cyan, and the unresolved row in amber. The final deletion page explicitly distinguishes base/current numbers. The [handoff guide](../../../docs/front-review-r4.md) describes the counts, limitations, notebook and reproduction commands.

No GPU training, inference or model promotion was performed by this pass.
