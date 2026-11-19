# Round 11 annotation evidence

This round applies the user's screenshot and text feedback to the exact round 10 review at commit `8230b8d592c792002bf97405d950264d22292a47`. All **58 requested new photographs** were inspected against their original camera pixels. **54 photographs change**, with **630 row replacements, two duplicate removals and one added front**. The four requested photographs preserved after inspection are new120, new121, new122 and new125. All 31 accepted old review files and all 68 unrequested new review files remain byte-identical to round 10.

The [round guide](../../../docs/front-review-r11.md) explains the annotation decisions, remaining questions, PDF legend, PR79 result and next experiment. [PR35 remains the selected model](../../../docs/current-model.md); the round 11 notebook is prepared, but no round 11 GPU training has run.

## Deliverables

- [PDF delivery record](pdf-delivery.json): 55 pages, comprising a linked index and only the 54 changed photographs. All current instances on each included photograph are displayed.
- [Notebook handoff](notebook-handoff.json): exact notebook, source and protocol identities for the next controlled Colab run.
- [Review manifest](../../../data/reviews/fronts-2026-11-19-r11/manifest.json), [candidate manifest](../../../data/candidates/front-review-2026-11-19-r11/manifest.json) and [experiment protocol](../../../experiments/front-review-2026-11-19-r11.json).
- [PR79 evaluation](../../evaluation/pr79-r10-20261119/README.md): verified returned round 10 results and fresh fixed scoring.

## Traceability

| Evidence | Purpose |
| --- | --- |
| [User reference index](user-reference-index.json) and [original text](feedback-text.txt) | Bind all ten marked screenshots and all text feedback; document interpretations of ambiguous left/right wording. |
| [Root](root-audit.json), [middle](middle-audit.json), [late](late-audit.json) and [late2](late2-audit.json) audits | Record original-pixel review, per-image decisions and exact proposed operations. |
| [Root secondary audit](root-secondary-audit.json) and [late secondary audit](late-secondary-audit.json) | Independently inspect the early screenshot amendments and selected identity/landmark changes. Their limited scopes and exact audit hashes are explicit. |
| [Operations](operations.json), [row maps](row-maps.json) and [amendment summary](amendment-summary.json) | Replay exact round 10 rows into round 11 and preserve identities across additions and removals. |
| [Request coverage](request-coverage.json) | Account for all 58 requested photographs and each recorded disposition. |
| [Complete review status](review-status.json) | Preserve all 19 unresolved cases across 17 photographs, including unchanged photos omitted from the PDF. |
| [Verification](verification.json), [actual training inventory](training-input-inventory.json) and [dispatcher check](runner-verification.json) | Verify all 157 originals, generated labels, candidate/control staging and frozen evaluation references. |
| [Notebook input preflight](notebook-input-preflight.json) | Record execution of the notebook's exact input-check code on real PNGs, labels and the epoch173 checkpoint. This is not a GPU training receipt. |
| [PDF layout](pdf-layout-qa.json), [final readback](pdf-final-readback.json), [lossless optimization](pdf-lossless-final-audit.json) and [visual review](pdf-visual-qa.json) | Verify page identities, original RGB pixels, links/bookmarks, 2,067 tags and bounded visual inspection. |

## Material decisions

Round 10 rows 23 and 28 in new050 are removed as duplicate upper rims, leaving four cassettes per stack. A separate visible front above parent row 48 in new090 is added as current row 48, with both true lower corners hidden and encoded `0 0 0`. The right corner on new087 row 33 is restored; unavailable points on new094 row 18 and new124 row 19 are removed.

The grouped requests in new124/new127/new128 that would collapse broad fronts by moving xmax to the left point are interpreted as referring to the right point. Explicit separate left-point requests retain their stated side. This decision is grounded in the original pixels and recorded in the reference index and audits.

Two requested changes remain incomplete: the lower boundary of new074 row 11 depends on an occluded instance separator; the proposed rightward shift of new124 row 25 would include rack material and displace the box beyond its existing left corner. The corresponding holds remain explicit. Other inherited holds are not resolved by unrelated edits.

## Actual training input

Both arms contain 137 training photos, including 13 empty scenes. The 16 held training photographs and separately excluded 31 old photographs use identical PR35 label files in both arms. Exactly 77 eligible training label files differ: the control has 3,771 fronts and the candidate 3,775. The complete review's 3,788 training fronts are a different inventory because held photos do not contribute amended supervision.

Original validation remains 20 photos and 664 fronts in both arms. The frozen 666-front primary and 663-front diagnostic scoring references are unchanged. Current development amendments are exploratory and do not silently alter those scorecards.

The protocol keeps the established seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW and learning rate 0.0001, with each full arm independently initialized from epoch173 and final-epoch selection fixed. The notebook's source commit is recorded in its final handoff; no round 11 model has been promoted.
