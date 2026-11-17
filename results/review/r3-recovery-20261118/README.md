# Interrupted r3 review: recovery and completion

The current review is [data/reviews/fronts-2026-11-18](../../../data/reviews/fronts-2026-11-18/README.md). It contains **889 replacements and 12 deletions** relative to r2: the 888 replacements and 12 deletions recovered from the original r3 PDFs, plus one separately inspected box correction for `new_000101` row 52. No model has been trained on this complete review revision.

## Evidence and limits

- `manifest.json`, `operations.json`, `verification.json`, and `pdf-review-notes.json` are the immutable reconstruction receipts. The source PDFs and parent labels are pinned by SHA-256. Their vector geometry reproduces 157 label files with explicit row mappings, including 1,817 before/current panel-row checks. The two PDFs agree within 0.001 original-image pixel.
- `pdf-inventory.json` records the source PDF inventory and the outstanding user-request completeness check found before the follow-up.
- `new101-row52-followup.json` records the original PNG hash, physical boundary reasoning, exact before/after label rows, and the few-pixel uncertainty of the soft contour. `new101-row52.patch` applies that one correction after the historical recovery.
- `integration-review.json` verifies the assembled current bundle, unchanged original datasets, and the distinction between reconstructed and newly reviewed labels.
- `recovered-pdf-audit.json` and `recovered-pdf-qa.json` bind the two updated review PDFs to their exact bytes and record their visual and structural checks.

The original authoring TXT bytes were not recovered. Unchanged parent token groups remain exact; changed groups are newly serialized from PDF vectors. Numerical rendering agreement does **not** certify every physical annotation. Only `new_000007` rows 7 and 14 retain recorded amber holds; absence of an amber hold is not an independent physical approval.

The historical recovery manifest intentionally retains its original row-52-open statement. The current manifest records the completed follow-up. Historical `label_path` entries are relative to a reproduced recovery output directory, not to this evidence directory. Reproduce that tree with [scripts/recover_pdf_review.py](../../../scripts/recover_pdf_review.py); do not treat these paths as missing children of this folder. The original source PDFs, not the updated deliverables, are the script inputs. Exact commands and the separate patch step are in the [current review guide](../../../data/reviews/fronts-2026-11-18/README.md#reproduce-the-recovery).

The current labels remain review-only, with no selected training recipe and no automatic model promotion. Original training validation, PR67's frozen comparison references, and the selected PR35 checkpoint are unchanged. Review both updated PDFs before merging the label revision.

## Updated PDF navigation

| File | Pages | Useful places |
| --- | ---: | --- |
| `cassette-front-changes-2026-11-18-recovered.pdf` | 76 | Page 3: index; page 50: new101 comparison; page 75: row 52 detail; page 76: provenance |
| `cassette-label-atlas-2026-11-18-recovered.pdf` | 166 | Page 79: new101; page 146: the remaining new007 amber rows; page 166: provenance |

Original r3 page numbering, bookmarks and 389 combined internal links are preserved. The 234 unaffected original pages retain identical decoded content streams. The refreshed new101 page contains 53 fronts, 103 locatable corners and 3 missing corners, and its embedded photograph matches the original PNG pixels.
