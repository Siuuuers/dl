# Round 17: recovered labels and the current notebook

22 November 2026, Asia/Hong_Kong.

**The previous corrections were saved as actual labels.** The recovered r16 source is commit `8d72f7b7869aae4e71cbe9066fb28dd54ba7f9a2`, available on GitHub. It contains 83 revised rows and the requested photo 066 row 22 deletion, its complete 157-file review, the composed training candidate, and the PR88 evaluation. The interrupted part was the notebook handoff: the local r16 notebook still had an unfinished source pin. The r17 notebook below replaces that unfinished entry.

## Current annotations

The [complete r17 review](../data/reviews/fronts-2026-11-22-r17/manifest.json) starts from those saved r16 bytes and applies the new requests for **067–075, 079 and 080**. These are all training photographs. The current labels are under `data/reviews/fronts-2026-11-22-r17/labels/`; `data/labels/` retains the original baomai dataset.

This pass changes all 11 requested photographs: **139 row revisions and one row deletion**, affecting 134 box groups, four left-point groups and three right-point groups. Some rows change several groups. All 146 unrequested review files retain their exact r16 bytes.

Use the [amendment summary](../results/review/front-review-20261122-r17/amendment-summary.json) for exact row and field counts and the [operation ledger](../results/review/front-review-20261122-r17/operations.json) for before/after labels. All references in this amendment list use the immutable r16 row numbers. Cross-row anchors use the r16 coordinates even when that source row is also changed. The [row map](../results/review/front-review-20261122-r17/row-maps.json) records the surviving identities after photo 074 row 13 is deleted.

The PDF is rendered from saved labels. It includes every surviving label on each changed photograph, with changed fields highlighted; an outlined deleted-row marker is historical context and is never supplied to training. Unrequested labels, all old-series review files, and the development-review labels retain their r16 bytes. The [boundary policy](annotation-boundaries-r17.md) and request dispositions record any interpreted or retained fields.

### Specific interpretations

- Photo 069 rows 3–5: “x negative” means move left, toward smaller image X. The left-point X coordinates move from 602/598/594 to 599/595/591 px, retaining their Y and visibility. “xmin to pt” uses those corrected own left points; the box edges already equal those X coordinates. No negative normalized coordinate is written.
- Photo 074 row 10: delete only the right point, using `0 0 0`; retain the cassette row. Row 13 is deleted as a whole instance.
- Photo 074 row 4: the requested “rpt to row 5 ymax” cannot be applied as written. Row 5 image ymax 433 lies below row 4's box; interpreting it as row 5's upper edge 381 also lacks a supported lower-corner location. The accepted right point `(790,389)` px is retained. This one Y-anchor question remains explicit; the other photo 074 corrections are saved.
- Photo 074 row 20: the preserved baomai labels do **not** contain a separate label for this lower-diamond instance. It was added on 17 November as `sequence-aperture-009`. Original baomai row 20 is a different upper-stack front, corresponding to r16 row 19 and r17 row 18. The user's question does not request deletion, so the accepted lower-diamond instance is retained; after deleting r16 row 13, it becomes r17 row 19. The [provenance trace](../results/review/front-review-20261122-r17/photo074-row20-provenance.json) binds both identities to their saved labels.

## Training comparison

**PR35 remains selected.** The [completed PR88 audit](../results/evaluation/pr88-r15-20261122/README.md) reports conflicting rankings between the frozen primary reference and the later reviewed labels. No new training or model promotion is implied by this annotation handoff.

The [r17 training candidate](../data/candidates/front-review-2026-11-22-r17/manifest.json) explicitly selects the eligible revised annotations. Both arms retain all 137 training photographs and 13 empty training scenes. Photo 015's existing possible-extra-front hold and the 31 separate old-series treatment exclusions use exact PR35 supervision in both arms. The 17 prior displayed-case approvals remain inherited.

| Materialized training input | Reference | Candidate |
| --- | ---: | ---: |
| Photographs | 137 | 137 |
| Empty scenes | 13 | 13 |
| Front instances | 3,771 | 3,784 |
| Locatable endpoints | 6,841 | 6,837 |
| Missing endpoint slots | 701 | 731 |

Exactly 92 training-label files differ between arms. The complete review has 3,786 training fronts; the eligible candidate has 3,784 because photo 015 keeps the PR35 control labels in both arms.

| Fixed choice | Value |
| --- | --- |
| Initialization | Original epoch 173, independently for both arms |
| Recipe | Seed 46, 30 epochs, batch 4, input 1280 |
| Optimizer | AdamW, initial learning rate 0.0001, FP32 |
| Selection | Final epoch; separate resource trial never initializes either arm |
| Trainer validation | Original 20 photographs / 664 fronts |
| Primary scoring reference | Frozen 666 fronts |
| Historical scoring reference | Frozen 663 fronts |
| Current development review | 662 fronts; same label bytes as r16; exploratory |

This comparison measures PR35 supervision against the complete eligible r17 set. It does not isolate the r16-to-r17 amendments. The notebook prints the materialized object and endpoint counts for both arms before starting training.

## Run the notebook

Open [baomai_front_review_r17.ipynb](../notebooks/baomai_front_review_r17.ipynb) or [open r17 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r17-20261122/notebooks/baomai_front_review_r17.ipynb).

1. Select **Runtime → Change runtime type → GPU**.
2. Select **Run all** and connect Drive when requested.
3. Let input verification, the resource trial and both independently initialized arms complete. Compatible completed work can be recovered from Drive.
4. Return **`baomai-front-review-r17-results.zip`** for evaluation.

Source commit and protocol hashes are fixed in the delivered notebook. Its preflight verifies the original image and initializer bytes, exact label replay, composed training inputs, unchanged validation and scoring references, and the held/excluded cohorts. The [notebook verification receipt](../results/review/front-review-20261122-r17/notebook-verification.json) records preparation checks. CUDA installation, Drive interaction and full GPU training are not executed during preparation.

## Remaining evidence limits

Keep box confidence 0.13, endpoint confidence 0.50, NMS IoU 0.70, matching IoU 0.50, and radii 5/10 original-image pixels when scoring the return. Report misses and unmatched outputs along with corner agreement; retain eligible misses in the denominator. Different reference rankings must remain visible in the report.

Repeated use of the same development photographs does not measure generalization to a new capture session. A later held-out session is the useful next evidence once this annotation pass is complete. The current work uses RGB; pixel thresholds do not establish robot positioning tolerance in millimetres. The [robot-grasp guide](robot-grasp-readiness.md) explains which camera, depth/plane and calibration information would change that later architecture.
