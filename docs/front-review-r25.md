# R25: eight corner amendments and the next notebook

23 November 2026, Asia/Hong_Kong.

**The R25 annotation amendment and pinned notebook are complete. The notebook
is ready for the requested controlled comparison; R25 has not been trained yet.**
PR35 remains the selected working model. The [completed PR99 comparison](../results/evaluation/pr99-r24-20261123/README.md)
and [model improvement report](model-improvement-after-r24.md) explain the
latest R24 result and the remaining failure patterns.

## Run the new notebook

[Open R25 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r25-20261123/notebooks/baomai_front_review_r25.ipynb)
or download [baomai_front_review_r25.ipynb](../notebooks/baomai_front_review_r25.ipynb).

1. Select a GPU runtime.
2. Choose **Run all**. Use the existing Drive option to retain completed runs.
3. Return **`baomai-front-review-r25-results.zip`** after both arms finish.

The final notebook pins source commit
`86a7ba078b348f2b1cbacf73b4c271e9216fdb63` and exact protocol. Its input
preflight verifies the saved review, candidate, source images, missing-point
representation, standard rows, actual training inputs and original validation.
The resource trial and both training arms retain the established behavior.

The [handoff verification](../results/review/front-review-20261123-r25/notebook-verification.json)
records seven compiled code cells, empty execution outputs, the resolved
source/protocol pins, and a successful run of the final notebook's exact input
preflight. That result reproduces the saved verification. The six runtime
cells outside the updated input check preserve the working R24 implementation
apart from round names and pins. GPU training is the next user-run step.

## Exact amendment

Only the left endpoint changes in each row below. Coordinates are original
1280x720 image pixels; old values are rounded here for reading, with exact
serialized values in the [point ledger](../results/review/front-review-20261123-r25/point-review.json).

| Photograph | Row | Old left point | New left point | Scope | Use |
| --- | ---: | --- | --- | --- | --- |
| `new_000008` | 35 | (1199.79, 516.16) | (1207, 516) | User named | Development only |
| `new_000010` | 18 | (798, 269) | (800, 270) | User named | Development only |
| `new_000010` | 19 | (791, 312) | (794, 312) | User named | Development only |
| `new_000010` | 20 | (780, 353) | (785, 352) | Related adjacent row | Development only |
| `new_000012` | 1 | (109.77, 411.21) | (116, 411) | Related adjacent row | Training |
| `new_000012` | 2 | (119.65, 445.39) | (125, 446) | Related adjacent row | Training |
| `new_000012` | 3 | (129.76, 479.60) | (134, 479) | User named | Training |
| `new_000012` | 28 | (203.73, 700.12) | (209, 694) | User named | Training |

The complete eight changed rows have 16 endpoint slots: 15 locatable corners
and one missing right corner. The right endpoint of `008:35` remains its exact
missing triple. All boxes, classes, visibility tokens, row identities, counts
and splits are unchanged; the other 154 review label files match R24 exactly.

The nine newly supplied standards `008:18-21,28-32`, plus `010:16` and
`016:24`, keep their entire exact lines. Each corrected point follows that
front's visible lower-face turn, without a fixed inward offset or a brightness
threshold. The [boundary policy](annotation-boundaries-r25.md) records the
physical definition and limits.

### One retained soft boundary

`012:28` right stays at approximately `(330.02,705.32)`. The shadowed lower
front edge merges with the neighboring layer, so a plausible upward alternative
cannot be established as a more accurate exact corner. Retention records
insufficient evidence to amend, not a claim that the old point is exact.
The clearer left-corner correction does not require moving the opposite point.

## Review the pictures

The compact **`front-review-r25-amendments.pdf`** is supplied in the chat
handoff. It contains a change summary and three original/R24/R25 comparison
pages. Magenta shows old R24 points, green shows changed R25 points, and cyan
shows retained coordinates. The original-pixel column is unmarked.

- [Photo 008 comparison](../results/review/front-review-20261123-r25/new_000008-front-face-comparison.png)
- [Photo 010 comparison](../results/review/front-review-20261123-r25/new_000010-front-face-comparison.png)
- [Photo 012 comparison](../results/review/front-review-20261123-r25/new_000012-front-face-comparison.png)

The [complete 116-page R24 atlas](front-review-r24-pdf.md) remains the earlier
snapshot of all nonempty new-series photos. The new compact packet displays
the R25 amendment; it does not silently replace that prior PDF.

## What this training can test

Exactly **four training endpoint locations in one photograph**, `012`, differ
from the R24 candidate. The other four edits belong to development photos
`008` and `010`. Those development edits cannot teach the model; they refine
the separately versioned review used to interpret output.

The new notebook preserves the existing paired experiment: independent
original-epoch173 initialization, seed46, 30 epochs, batch4, 1280 input, FP32,
AdamW with initial learning rate0.0001, and prespecified final-epoch selection.
It keeps 137 training images including13 empty scenes, the original20-image
664-front trainer validation, and inherited held/excluded-photo handling.

The control uses PR35 supervision; the candidate uses accumulated eligible
amendments. That within-run comparison tests accumulated changes. A direct
R24/R25 candidate comparison additionally requires matching runtime and input
receipts, and identical evaluation-reference bytes. The frozen666 primary,
historical663 diagnostic and current662 exploratory review stay separately
identified. No automatic model replacement follows from completion of training.

These eight point repairs do not change partial-front boxes or add new visual
examples. The [failure inventory](../results/review/front-review-20261123-r25/model-failure-inventory.md)
therefore provides a separate next-work list for occluded fragments, corner
errors, duplicates and background confusion. The notebook measures the effect
of the newly supported supervision; it does not promise to solve those other
failure mechanisms.
