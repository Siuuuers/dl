# R47 cassette-front annotation release

All 507 current fronts in 12 original photographs were inspected. This round
changes 412 box boundaries across 320 rows, resolves 338 field holds, and
preserves every corner/visibility token. The full historical 508 identities
remain, including one earlier deletion and a separate added identity.

## Files

- `audit-ledger.json`: every current stable ID, exact R46 before/R47 after TXT,
  original-image SHA-256, local physical reason, concise PDF reason and
  explicit field-hold resolutions.
- `outlines.json`: 14 manually interpreted diagnostic traces; 11 closed,
  3 deliberately open. They are not segmentation training masks.
- `proposals/`: all 12 current label files, including held photographs.
- `complete-candidates/labels/`: only the 5 complete photographs:
  `frame_0146`, `frame_0160`, `frame_0178`, `frame_0371`, `frame_0383`.
- `release-status.json`: exact manifest/audit/outline/label hashes and gates.

The remaining 13 held rows in 7 photos contain 32 unresolved fields. The
complete-photo export is an assistant-reviewed annotation gate, not human
sign-off, a new model result or an independent test split. The historical
`frame_0383` training labels remain byte-identical to PR133.

Use [the R47 guide](../../../docs/front-review-r47.md) for the actual contour
decisions and the [new pilot](../../../docs/additional-front-pilot-r47.md) for
the explicit training membership. The five-photo candidate adds four photos
to the previous 138-photo dataset; the seven held photos remain excluded.

Rebuild the full historical manifest and offline review package from the
pinned original archive using `scripts/build_front_review_r47.py`. PDFs and
photographs are delivered or downloaded separately rather than committed
again. The existing HTML is still the R44 snapshot.
