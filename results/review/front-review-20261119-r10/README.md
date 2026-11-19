# Complete new-series review, round 10

See the [review and training guide](../../../docs/front-review-r10.md) for the decisions, PDF legend and notebook workflow.

All 126 new photographs were inspected; all 31 confirmed old review files were preserved byte for byte. Relative to r9, 65 photos changed: 269 existing rows revised, two removed and two added. Replacements changed 134 box, 124 left-point and 47 right-point field groups. There are 18 specific remaining cases across 15 training photos and one development photo.

## Evidence and exact replay

| Record | Purpose |
| --- | --- |
| [amendment-summary.json](amendment-summary.json) | Counts for this annotation revision |
| [request-coverage.json](request-coverage.json) | One visual review entry for every new photograph |
| [root-audit.json](root-audit.json), [early-audit.json](early-audit.json), [middle-audit.json](middle-audit.json), [late-audit.json](late-audit.json) | Four disjoint primary audits, with original-image and r9 label identities |
| [root-secondary-audit.json](root-secondary-audit.json) | Independent check of 13 specified root proposals |
| [new025-secondary-evidence.json](new025-secondary-evidence.json) | Independent count and ownership inspection near row57 |
| [user-reference-index.json](user-reference-index.json) | The ten uploaded screenshot hashes, photo matches and user decisions |
| [operations.json](operations.json), [row-maps.json](row-maps.json) | Exact before/after edits and immutable r9-to-current instance identities |
| [review-status.json](review-status.json), [status-updates.json](status-updates.json) | Current cases and the reasons for changed decisions |
| [pdf-case-notes.json](pdf-case-notes.json) | Concise display text bound to the full case ledger |

Inspection crops under `tmp/r10-*` are reproducible scratch views, not additional training images. Their hashes and source coordinates remain in the audit records. The original source PNGs and final complete PDF are the durable visual evidence.

## Deliverables and checks

- [PDF delivery record](pdf-delivery.json): final file link, hash and page/layout checks.
- [Notebook handoff](notebook-handoff.json): exact source commit, protocol identity and notebook checks.
- [Actual input verification](verification.json) and [per-file input inventory](training-input-inventory.json): materialized reference and candidate images/labels.
- [Training runner verification](runner-verification.json): the actual dispatcher and staging path used by the notebook.

Both arms contain all 137 training photos, including 13 empty scenes, and unchanged original validation on 20 photos. The candidate contains 3,778 training fronts versus 3,771 in the PR35-label reference. Exactly 77 training photos differ between those arms; this includes inherited accepted revisions and is a different comparison from the 65 photos changed relative to r9.

All 15 held training photographs and the separately excluded 31 old photographs use exact PR35 labels in both arms. The frozen 666-front primary and 663-front diagnostic references and the seed-46, 30-epoch recipe are unchanged. No r10 GPU run or model promotion has occurred.
