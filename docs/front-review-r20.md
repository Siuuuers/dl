# Round 20: horizontal rack cleanup and the next notebook

22 November 2026, Asia/Hong_Kong.

**The r20 notebook is the current training entry. PR35 remains the selected model.** The accepted r19 annotations and PDF are preserved. This round tightens **315 boxes across 37 photographs** and changes no corner, visibility value, vertical box field, object count or row identity.

## Run the notebook

[Open r20 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r20-20261122/notebooks/baomai_front_review_r20.ipynb), or download [baomai_front_review_r20.ipynb](../notebooks/baomai_front_review_r20.ipynb).

1. Select a GPU runtime.
2. Select **Run all** and connect Drive when offered.
3. Let the verification, four-batch resource trial and two training arms finish.
4. Return **baomai-front-review-r20-results.zip**. Completed bundles and the final ZIP are also preserved in the notebook's Drive folder when Drive is connected.

The notebook pins a full source commit and exact protocol SHA-256. [Final notebook verification](../results/review/front-review-20261122-r20/notebook-verification.json) records those identities. Input preparation did not run GPU training, install CUDA dependencies, exercise Drive or select a replacement model.

## What changed

All **40 supplied photographs / 1,715 existing rows** in 081-130 were inspected. IDs 091-093 and 104-110 are absent from the prepared data; they are not empty training scenes. There are **269 changed training rows** and **46 changed development rows**. Photos 081, 083 and 123 retain their accepted r19 labels because the remaining outward margins include projecting front/rim material.

The user's clean standard, **087 row 5**, is byte-identical to r19. The earlier accepted example **086 row 3** is also unchanged. Both edges of **087 rows 21-24** now meet their existing visible corner X coordinates. For example, row 24 changes approximately from xmin **806.21 to 811.79** and xmax **908.22 to 906.02** original-image pixels; its points and vertical bounds stay fixed.

The useful rule is to remove a visible rack strip while retaining the actual front and its owned rim. A lower corner is a good boundary proposal for an ordinary frontal face. For a sloping face, the upper rim can extend beyond that corner. Photo 081 row 14 and photo 116 row 1 illustrate why a universal endpoint clamp would cut cassette material. The accepted reference 087 row 5 itself has about 2.08 pixels beyond each corner.

This pass changes only the normalized horizontal center and width tokens in each YOLO row. All other tokens are preserved exactly. Missing endpoints stay missing. The r19 restoration of the left point on 127 row 32 is retained. No instances, holds or cases are added or removed. See the [annotation definition](annotation-boundaries-r20.md) and [exact amendment audit](../results/review/front-review-20261122-r20/horizontal-audit.json).

## Saved review and PDF

The [complete review](../data/reviews/fronts-2026-11-22-r20/manifest.json) contains all 157 annotation files. Its [training candidate](../data/candidates/front-review-2026-11-22-r20/manifest.json) selects eligible training supervision explicitly; the notebook does not fall back to original labels silently.

The changed-photo PDF contains **39 pages: two linked index pages and 37 photos**. Every current label on those photos is shown, totaling **1,624 row tags**. Magenta identifies changed boxes; cyan identifies retained fields. No uncertainty marks or hidden coordinate adjustments are added. [PDF delivery](../results/review/front-review-20261122-r20/pdf-delivery.json), [renderer checks](../results/review/front-review-20261122-r20/pdf-qa.json), and [visual review](../results/review/front-review-20261122-r20/pdf-visual-review.json) identify the delivered file and its checks.

All original images and labels, the accepted r19 files, and both fixed evaluation references remain unchanged. The exact replacement ledger preserves the accepted parent row for every change.

## Training remains the same comparison

| Actual training input | PR35 reference | r20 candidate |
| --- | ---: | ---: |
| Images | 137 | 137 |
| Empty scenes | 13 | 13 |
| Front instances | 3,771 | 3,784 |
| Locatable endpoints | 6,841 | 6,838 |
| Missing endpoint slots | 701 | 730 |

Exactly **92 training-label files differ** between the two arms. Photo 015 and the 31 old-series photos retain identical PR35 supervision in both arms; their images are not removed. The complete review has 3,786 training fronts, while the eligible candidate has 3,784 because the inherited photo-015 supervision is retained.

Both arms independently initialize from original epoch 173 with seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW and initial learning rate 0.0001. They run in the same runtime and use the final-epoch checkpoint. The resource trial never initializes either full run. This keeps the established Baomai two-lower-corner architecture and changes the supervised labels only.

The original trainer validation remains **20 photos / 664 fronts**. The frozen **666-front primary** and **663-front historical** scoring references remain unchanged. Photos 086-090, including the user's 087 examples, are development images: their reviewed labels never enter training or replace trainer validation. The current **662-front r20 development review** is reported separately as exploratory evidence.

## Verification and interpretation

The notebook's exact embedded input preflight passed. Actual staging copied and independently read back **314 images and 314 labels** across the two arms. Fingerprints of 476 original/r19/reference source files matched before and after. [Input preflight](../results/review/front-review-20261122-r20/input-preflight.json), [staging readback](../results/review/front-review-20261122-r20/staging-verification.json), and [review verification](../results/review/front-review-20261122-r20/verification.json) retain the evidence.

PR92 returned r18 training results; its [completed audit](../results/evaluation/pr92-r18-20261122/README.md) is preserved. The interrupted r19 notebook handoff is now repaired at [the recovered pinned notebook](../notebooks/baomai_front_review_r19.ipynb). Use r20 for the additional cleanup in this round.

This experiment compares accumulated eligible r20 annotations with PR35 annotations. It does not isolate this latest rack cleanup. Tightening labels can change IoU matches without moving a model prediction, so fixed and newly reviewed references must remain separate. Repeated development scenes do not establish generalization to a new capture session. Improved model accuracy is a result to measure after the returned training ZIP, not a claim made by preparing cleaner annotations.
