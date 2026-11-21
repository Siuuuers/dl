# Round 14: requested amendments and approval of displayed amber cases

Round 14 starts from the saved r13 review at `4795910cdeb37d23d4b99031eb480c94b883b478`. The review manifest is pinned by SHA-256 in the builder. The new text names 33 photographs. Every requested original is inspected, including requests ultimately retained unchanged. Exact r13 source rows, source image hashes, before/after lines and clause dispositions precede replay. Every unrequested label and every old-series review label retains its r13 bytes.

## Physical target and coordinate interpretation

Keep the [r13 physical contract](annotation-boundaries-r13.md): one visible cassette front with its owned frontal rim per box, and the two independently observed lower front-to-side junctions ordered by image X. A projecting rim may put a corner inside its enclosing box. Rack, hand and gripper edges do not establish cassette corners. A hidden or unlocatable point remains exactly `0 0 0`; a locatable point uses visibility 2. Deleting a point does not delete the cassette instance.

Image X increases rightward; image Y increases downward. Where feedback names physical top as `ymax` or physical bottom as `ymin`, translate to pixel `ymin` or `ymax` only when the original scene supports it. Cross-row point references normally align X only, retaining the target's own Y and visibility. Explicit Y references are honored when physically locatable. Anchors use immutable r13 coordinates, including a point later deleted in the same pass; a dependent edit using a revised point is identified in its disposition.

The text boundary between photos007 and008 is ambiguous. The clauses “17 xmax to16 rpt” and “16,17 lpt to15 lpt” refer to photo008: applying them to007 would jump to a different column. The source image establishes this interpretation. Photo009 row27's `xmax lpt` wording is interpreted as `xmin` and left X because the literal right-edge transfer would collapse the front. Neither correction is applied speculatively to both photos.

Perspective changes a stack's extent by tier. Reject a literal adjacent-row transfer when it places the target in rack, inside its visible front, or in a different column. A small requested directional correction may be bounded by the observed front turn rather than the literal box edge; the actual coordinate and reason are recorded. No new training hold is created merely because a request is ambiguous or unsupported.

## User approval and the remaining unboxed question

The user explicitly said “All amber are correct!” after the r13 atlas. `user-approval.json` binds that quote to the r13 manifest, status ledger, delivered PDF receipt and layout audit. It resolves the **17 cases with displayed amber rows**, across15 photos, releasing14 training photos. This records the user's acceptance of the saved displayed annotations; it does not claim a new physical inspection of every previously held photo.

Photo015's possible extra front had no labelled row and therefore no amber outline. That one unboxed instance question remains held. Its training label is exact PR35 in both arms. All31 old-series photos remain separately excluded from the treatment, also with exact PR35 labels in both arms. Those exclusions are distinct from unresolved physical cases. The review is not a certification of the complete dataset.

## PDF scope

Include exactly the nonempty `new_*` photographs whose saved labels changed relative to r13. Show every current label on each included photo. Magenta marks changed fields, cyan marks retained fields, and amber is reserved for a still-unresolved displayed field. Resolved amber cases are cyan unless amended in this pass. The unboxed photo015 question is described in text; do not draw an invented instance.

Unchanged, old-series and empty-label photos are excluded from this PDF only. All157 label files and the original images remain in the dataset. Every correction is saved in the r14 review, which the PDF renders. Eligible resolved training corrections enter the composed candidate; photo015's review amendments remain held out, and development-review amendments do not replace the original trainer-validation labels. Every row retains its identity because this pass adds or deletes no instances.

## Next comparison

PR84's r13 candidate improves the pre-return reviewed-label diagnostic but regresses against the frozen primary reference. PR35 remains the selected checkpoint. The r14 experiment tests the corrected and newly accepted label set while holding model architecture, numerical recipe, thresholds, original validation bytes and frozen666/663 scoring references fixed. The r13 diagnostic was pinned before PR84 returned; r14 review is a later diagnostic and cannot retroactively decide the PR84 winner.

Both fresh arms independently initialize from the original epoch173 weights and use seed46,30epochs,batch4,1280input,FP32,AdamW with initial learning rate0.0001, and final-epoch selection. This pass prepares and verifies the experiment; it does not run GPU training or promote a model automatically. The single experiment changes both new amendments and eligibility of user-approved labels, so its result estimates their combined effect rather than isolating those two causes.
