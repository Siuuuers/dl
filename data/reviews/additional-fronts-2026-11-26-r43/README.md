# R43 targeted amendments

R43 starts from the verified [PR129 R42 return](https://github.com/Siuuuers/dl/pull/129). It reviews ten requested existing source rows: nine amended and one retained. The original image pixels and inherited full-frame decisions remain the authority for this review.

- [User guide and physical decisions](../../../docs/front-review-r43.md)
- [Exact amendment ledger](amendments.json)
- [Cumulative current manifest](manifest.json)
- [Photo provenance and prior guidance](provenance.json)
- [Self-contained frame review](../../../results/review/front-mining-r43/review.html)
- [Portable ZIP with labels and history](../../../results/review/front-mining-r43/review.zip)

`manifest.json` preserves immutable upstream identities. Its `source_line` is the upstream label, `before_line` remains the R41 working baseline, and `after_line` is the current R43 candidate. Its main `counts` and `changed_token_indices` are cumulative from that working baseline. The separate `latest_counts`, `latest_update`, and amendment ledger describe only R42 → R43. The complete R42 manifest remains in the ZIP at `history/r42/manifest.json`.

All 508 identities remain, including the one earlier deletion and its separate addition. There are 507 current fronts. All 498 unrequested row records, existing unresolved fields and whole-image dispositions are preserved exactly. Nothing was added or deleted in R43.

Only `frame_0146`, `frame_0371`, and `frame_0383` remain eligible for complete-candidate export, totaling 129 fronts. Nine whole frames remain held. There is no assigned training split, human sign-off, new inference, training or model promotion.
