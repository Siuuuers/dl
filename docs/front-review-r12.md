# Round 12 annotation amendments and training handoff

20 November 2026, Asia/Hong_Kong.

**The annotations are saved as YOLO label files.** This round reviewed the 46 requested photographs and changed 45 of them: 597 existing rows were amended and one visible front fragment was added. No existing instance was deleted. The PDF displays exactly those 45 changed photographs, with every current instance on each page.

The complete current review is [fronts-2026-11-20-r12](../data/reviews/fronts-2026-11-20-r12/manifest.json). Its `labels/train/` and `labels/val/` directories contain all 157 label files, including empty scenes. Original images remain under `data/images/`. The original `data/labels/` files are the immutable source dataset, not the latest review. The notebook explicitly selects the r12 training candidate, so it does not silently train the original labels or an older notebook's package.

## What changed

| Review result | Count |
| --- | ---: |
| Requested photographs inspected | 46 |
| Photographs with changed label bytes | 45 |
| Existing rows amended | 597 |
| Box field groups changed | 543 |
| Left-point field groups changed | 84 |
| Right-point field groups changed | 66 |
| Added visible front fragments | 1 |
| Removed instances | 0 |
| Unrequested review files preserved exactly | 111 |
| Existing physical cases resolved | 1 |
| Remaining physical cases / photographs | 18 / 16 |

Field counts overlap: one row can have both its box and a point changed. Photo 019 is the only requested photograph with no saved change; its inconsistent anchors remain in the request ledger. The complete review has **3,789 training fronts and 662 development fronts**. The [amendment summary](../results/review/front-review-20261120-r12/amendment-summary.json), [operations](../results/review/front-review-20261120-r12/operations.json) and [row maps](../results/review/front-review-20261120-r12/row-maps.json) record exact identities.

The new photo 023 fragment is appended as **row 37**, preserving every existing row number. A separate upper seam and pale upright support its identity below row 35. The hand hides both actual lower corners, which remain exactly `0 0 0`.

Four previously missing corner slots are now locatable in the original pixels: right corners of photo 026 rows 43/49 and photo 119 rows 27/32. Four requested unreliable corner slots are cleared: photo 056 row 8 right, photo 113 row 20 right, photo 114 row 17 right, and photo 118 row 33 left. These point removals do not delete the fronts.

## Coordinate and ownership decisions

Image X increases rightward; image Y increases downward. The repeated physical-top/physical-bottom wording in the feedback was interpreted against the original image, with each translation recorded. In particular, moving a front's upper edge to the lower seam of the cassette above is a pixel-`ymin` change. Moving its bottom to its own lower corner is a pixel-`ymax` change.

The [boundary policy](annotation-boundaries-r12.md) retains the distinction between an enclosing front/rim box and an actual lower front/side junction. A corner can be inset from a projecting rim. An independent second review corrected the initially proposed photo 026 points away from the dark rack to their visible front turns: row 43 at `(687,451)` and row 49 at `(686.5,470.5)`. The points are not forced to box corners.

The supplied screenshots are saved with their hashes in the [reference index](../results/review/front-review-20261120-r12/user-reference-index.json). Numerical coordinates come from the original PNGs. The [request ledger](../results/review/front-review-20261120-r12/request-coverage.json) contains 217 disposition entries: 112 applied, 74 interpreted, 3 already satisfied, 20 unsupported as written, and 8 unresolved. These are grouped review entries, not a count of individual point coordinates.

## Remaining questions

**Photo 124 row 25 is resolved.** The new instruction specifies the right edge alone at row 29's left-point X, approximately 330.64. It trims the previous right boundary 338.29 while retaining the supported left edge 212.67. This replaces the earlier ambiguous request to move both horizontal edges right.

**Photo 074 rows 11/20 remain held.** Row11's exposed top and right strip are amended to pixel `ymin418` and `xmax788`. Whether its lower aperture patch belongs to row 11 or a separate front is still unresolved; its lower boundary is retained. The pale triangle below photo 089 row 47 likewise lacks a decisive independent frontal separator, so no extra object is added there.

The remaining wording questions are concrete:

| Photo | Clarification needed |
| --- | --- |
| 011 | The clause `16 to?` does not name a field or target. |
| 019 | The initial `xmax negative to right point` has no row number. Rows24/30 and 27/37 identify different vertical tiers; rows 55/56 are in distant columns. A marked photo with the intended row identities would resolve these references. |
| 097 | `44 to lpt,15 to lpt` does not identify the field or edge to move. |
| 117 | Row4's right point is in a different column from rows 11–13/16. Confirm the intended right-edge anchor for those rows. |
| 118 | Row31's right point is in another column from 24/27/28; their supported own-right-point corrections were applied instead. |
| 128 | Row24's right edge is not a valid left-edge anchor for row 28's column; the supported row 30 alignment was applied. |

An incomplete instruction alone does not create a new training hold. A correctly missing corner also does not create one. The [physical case ledger](../results/review/front-review-20261120-r12/review-status.json) retains only actual unresolved front, boundary or corner ownership.

## New notebook

[Open baomai_front_review_r12.ipynb](../notebooks/baomai_front_review_r12.ipynb), or [open it in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r12-20261120/notebooks/baomai_front_review_r12.ipynb).

1. Choose **Runtime → Change runtime type → GPU**.
2. Choose **Runtime → Run all**. The notebook checks its pinned source and input hashes, downloads original images and epoch 173 initialization, preserves the runtime's Torch/torchvision pair, and runs a four-batch resource check.
3. Let both complete arms finish. Each starts independently from epoch 173 with seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW and learning rate0.0001. Final-epoch selection is fixed.
4. Return **`baomai-front-review-r12-results.zip`**. Compatible completed arms and the final ZIP are verified and backed up to Drive; a rerun can reuse them.

This is a new r12 experiment. [PR80](https://github.com/Siuuuers/dl/pull/80) already contains the completed r11 return. Preparing this notebook runs no training and makes no model-quality claim. The notebook pins source commit `4f4423a511b42bc804cbcbeb7a772823cef2eb3b`; its [verification receipt](../results/review/front-review-20261120-r12/notebook-verification.json) records all seven compiled code cells and the successful input preflight against the actual 157 PNGs and epoch173 bytes.

### Actual training inputs

| Input | Reference arm | Candidate arm |
| --- | ---: | ---: |
| Training images | 137 | 137 |
| Empty training scenes | 13 | 13 |
| Training fronts | 3,771 | 3,777 |
| Original validation images | 20 | 20 |
| Original validation fronts | 664 | 664 |

Exactly **78 training photographs** have different labels between the arms. The **15 unresolved training photographs** retain exact PR35 supervision in both arms. The **31 accepted old-series photographs** remain separately excluded from treatment and also retain identical PR35 labels. The complete current annotation review and the eligible training overlay therefore have different counts; this is deliberate and recorded, not a missing-save problem.

The [protocol](../experiments/front-review-2026-11-20-r12.json) keeps the **666-front primary** and **663-front historical** scoring references fixed. The current 662-front r12 development review is a separately reported exploratory diagnostic. After training, compare PR35 and both fresh final-epoch models using the fixed thresholds and report misses, unmatched outputs, duplicates, empty scenes and 5/10-pixel corner-pair results. Inspect gained and lost identities before choosing a model. These reused development scenes do not establish reliability on new camera sessions or robotic positioning.

## Verification and reproduction

The [materialized input verification](../results/review/front-review-20261120-r12/verification.json) and [training-runner verification](../results/review/front-review-20261120-r12/runner-verification.json) check actual files in both arms. They verify all 157 original image hashes, unchanged original labels, exact held/excluded labels, original validation and frozen scoring references. The complete annotation files replay exactly from the four source audits. Fifty critical proposals also have a separately bound physical review.

The [PDF QA receipt](../results/review/front-review-20261120-r12/pdf-qa.json) binds its exact label inventory, all 1,667 current row tags, navigation and original-pixel images. It reports zero overlapping or clipped tags. Saved embedded RGB pixels are compared byte-for-byte with the originals.

From the repository root:

```bash
python3 -B scripts/verify_front_review_r12.py
python3 -B scripts/check_front_review_r12.py --output /path/to/fresh-r12-check
python3 -B scripts/build_front_review_r12_pdf.py --output /path/to/r12-changes.pdf
```

For another manual editor session, make a fresh working folder containing a copy of `data/images/` as `images/` and a copy of `data/reviews/fronts-2026-11-20-r12/labels/` as `labels/`. Open that folder in the [HTML labeler](../tools/cassette_labeler/README.md). Keep the pinned review immutable; collect later changes into a new revision.
