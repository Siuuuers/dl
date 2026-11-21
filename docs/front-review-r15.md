# Round 15: saved amendments and the next notebook

21 November 2026, Asia/Hong_Kong.

**The amendments are saved as actual YOLO labels.** All 24 requested photographs were inspected. This pass changes 141 existing rows and deletes the explicitly requested photo 028 r14 row 63: 142 operations across 24 photos. The replacements affect 133 box groups, eight left-point groups and 11 right-point groups; a row may change several groups. No instances are added. The [complete review](../data/reviews/fronts-2026-11-21-r15/manifest.json) retains all 157 label files. [Exact amendment summary](../results/review/front-review-20261121-r15/amendment-summary.json).

**The new notebook is [baomai_front_review_r15.ipynb](../notebooks/baomai_front_review_r15.ipynb).** It selects the composed r15 training candidate explicitly. The original `data/labels/` directory remains the preserved source dataset; current review annotations are under `data/reviews/fronts-2026-11-21-r15/labels/`.

**PR35 remains selected.** PR86 is the completed r14 training return. Its candidate improves several reviewed-target scores while regressing against the frozen primary reference. This r15 preparation does not run new GPU training or replace the model.

## What was saved

The [operations ledger](../results/review/front-review-20261121-r15/operations.json) binds each changed row to its exact r14 before-line, after-line, original image hash and physical reason. All cross-row anchors use the immutable r14 coordinates, even when the anchor itself changes in this round. The [row maps](../results/review/front-review-20261121-r15/row-maps.json) record deletion of028:63. Because it was the final row, the surviving rows 1–62 keep their numbers.

All 133 unrequested label files, including the 31 old-series files, retain exact r14 bytes. The [request coverage](../results/review/front-review-20261121-r15/request-coverage.json) contains 82 grouped dispositions: 54 applied, 19 interpreted, eight unsupported as written and one self-assignment no-op. Five directional or approximate proposals have a separately bound physical corroboration. Every requested photo has at least one saved amendment.

The [physical boundary policy](annotation-boundaries-r15.md) distinguishes the enclosing visible front and owned rim from its actual lower corner. Image X increases rightward and Y downward. “Negative” means movement toward smaller coordinates. Requests naming the physical top as `ymax` are mapped to pixel `ymin` only where the original scene supports it. Point references normally align X while retaining the target point’s own Y and visibility.

Examples of documented interpretations include:

- Photo 049 row 18: box xmin 147, xmax 416 and left point (178,40) pixels, bounded by the clipped white front and its lower-left step.
- Photo 051 row 8: xmax moves left from 763 to 760 pixels; its hidden right point remains missing.
- Photo 054: “around lpt” trims retain each projecting upper/left frontal rim rather than cutting every box at an interior lower corner.
- Photo 055: rows 20/21 use xmax 290 and row 10 uses xmax 313. The lower-tier anchors315/319 would include rack/background at the target tiers.
- Photo 066 row 8 uses the original row 3 xmin 620, independent of row 3’s new xmin 623.

These are bounded visual judgments under the original image’s blur and occlusion, not subpixel physical measurements.

### Eight literal clauses retained

The supported changes on these same photographs are still saved. Retaining an invalid wording clause does not automatically create another whole-photo training hold.

| Photo / r14 row | Retained clause | Reason |
| --- | --- | --- |
| 001 /47 | xmin to left point | The point is missing behind the gripper. `0 0 0` is a sentinel, not an X=0 physical anchor. |
| 002 /32 | Left point to own xmax and row 26 right Y | The literal point (579,573) would cross to the right of its right point (576,578). |
| 028 /36 | ymax to right point | The sloping front’s left lower point is below its right lower point. The requested ymax would clip the left face and corner. |
| 033 /1 | xmin to row 6 xmin | The neighboring-stack anchor413 lies beyond this box’s xmax 407 and would invert it. |
| 033 /31 | xmin to row 25 xmin | The anchor371 belongs to a different stack and would enclose another front. |
| 040 /32 | Right point to row 1 left point | Full copying changes stacks; X-only copying moves the point inside its visible face. |
| 042 /8 | xmax to right point | Its owned upper/right rim reaches788, beyond the lower junction779. Its separate xmin amendment is applied. |
| 050 /20 | Left point to xmax and half box height | The literal (284,15) lies beyond its right point and on the opposite edge. The visible left point near (145,22) is retained. |

Photo 017 row 59’s `xmax to 59 xmax` is recorded as a literal no-op. No different row was guessed.

## What the PDF shows

The PDF contains **25 pages: one clickable index and all 24 changed photos**, displaying **947 current fronts**. Magenta identifies amended fields, cyan retained fields. A dashed D63 on photo 028 is a historical deletion marker, separate from current labels. The PDF’s box and point coordinates come directly from the saved review; moving a row tag does not move geometry.

Unchanged photographs, old-series photographs and empty-label scenes are omitted from this PDF only. The full dataset and both training arms retain their complete inventories. The [delivery receipt](../results/review/front-review-20261121-r15/pdf-delivery.json) records the PDF hash and checks, including exact original RGB pixels, label replay, navigation, row tags and page inspection.

## Inherited acceptance and the remaining hold

The 17 displayed-case approvals from r14 are inherited unchanged; no new photo is released in this pass. **Photo 015 remains the sole training hold** because the separate question concerns a possible additional front without a confirmed box. Its supported existing-row corrections are saved and visible in the PDF, while both training arms retain its exact PR35 supervision.

The 31 old-series photos are separately excluded from the treatment, also using exact PR35 labels in both arms. “Excluded from treatment” means the label changes are withheld, not that the photos are removed. The complete review has 3,788 training fronts; the eligible candidate has 3,786.

## PR86 result and why the selected model stays PR35

[PR86](https://github.com/Siuuuers/dl/pull/86) is audited at head `218f10ce7e964c958d0fb26159b01334951e5d7a`. Its complete 39,781,759-byte archive, 12 indexed members, notebook source pin, original images, epoch 173 initialization, recorded source hashes and actual staged inputs all passed. Both independent L4 arms completed 30 epochs, 1,050 batches and 78 optimizer steps. Full receipts, fresh predictions and reproducible scoring are in the [PR86 report](../results/evaluation/pr86-r14-20261121/README.md).

The same predictions are measured against the fixed references at confidence 0.13, endpoint confidence 0.50, NMS IoU 0.70, match IoU 0.50 and 1280 input. The corner-pair denominator retains eligible missed detections as failures.

| Reference | Model | Misses | Unmatched outputs | Both corners within 5 px | Within10 px |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen primary,666 fronts | PR35 | 18 | 80 | 533/584 | 582/584 |
| Frozen primary | Fresh r14 control | 20 | 83 | 531/584 | 583/584 |
| Frozen primary | r14 candidate | 35 | 101 | 520/584 | 575/584 |
| Pre-return r14 review,662 fronts | PR35 | 34 | 100 | 540/584 | 575/584 |
| Pre-return r14 review | Fresh r14 control | 38 | 105 | 535/584 | 574/584 |
| Pre-return r14 review | r14 candidate | 18 | 88 | 547/584 | 580/584 |

The candidate has 20 fewer misses and 17 fewer unmatched outputs than the fresh control on the reviewed target, but 15 more misses and 18 more unmatched outputs on the frozen primary. Its reviewed duplicate subset also increases from 16 to 30. All three models produce zero outputs on the three empty development scenes. These mixed results do not support automatic replacement or choosing whichever reference is favorable.

Against the previous r13 candidate on identical r14 review labels, misses improve 19→18, unmatched outputs 93→88 and 5-pixel pair successes 543→547, while 10-pixel successes fall 582→580. Some scored misses still have proposals just below the fixed IoU threshold; the report includes three original-image inspections and preserves this distinction.

### Separately named post-r15 diagnostic

Seven development label files change after PR86 returns: 007,008,009,010,054,055,059. Replaying the cached predictions against the new r15 review adds no inference or training and does not replace the pre-return result.

| Existing model on r15 review | Misses | Unmatched outputs | Pairs within 5 px | Within10 px |
| --- | ---: | ---: | ---: | ---: |
| PR35 | 35 | 101 | 538/584 | 573/584 |
| Fresh r14 control | 39 | 106 | 533/584 | 572/584 |
| Returned r14 candidate | 17 | 87 | 549/584 | 583/584 |

The model bytes and predictions are unchanged; the scoring target changed. The [post-r15 diagnostic](../results/evaluation/pr86-r14-20261121/post-r15-diagnostic.json) and [replay script](../results/evaluation/pr86-r14-20261121/measure_post_r15.py) provide an explicit baseline for the next experiment.

## Run the new notebook

[Open r15 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r15-20261121/notebooks/baomai_front_review_r15.ipynb).

1. Select **Runtime → Change runtime type → GPU**.
2. Choose **Run all** and connect Drive when requested.
3. Let the resource trial and both independently initialized arms finish. Compatible completed work can be recovered from Drive.
4. Return **`baomai-front-review-r15-results.zip`**.

| Actual training input | Reference | Candidate |
| --- | ---: | ---: |
| Photographs | 137 | 137 |
| Empty scenes | 13 | 13 |
| Front instances | 3,771 | 3,786 |
| Locatable endpoint labels | 6,841 | 6,838 |
| Missing endpoint slots | 701 | 734 |
| Original validation photos/fronts | 20/664 | 20/664 |

Exactly 92 training label files differ. Both arms independently initialize from original epoch 173, using seed 46, 30 epochs, batch 4, 1280 input, FP32,AdamW and initial learning rate 0.0001. Final-epoch selection is fixed; the resource trial does not initialize either full arm. Frozen 666/663 scoring references and original 664-front trainer validation remain unchanged.

The [materialized input verification](../results/review/front-review-20261121-r15/verification.json) checks actual images, label bytes, counts and held/excluded cohorts. The [notebook verification](../results/review/front-review-20261121-r15/notebook-verification.json) records the exact source pin, seven code-cell syntax checks, actual embedded input preflight and committed payload identity. CUDA installation, Drive interaction and full GPU training are not executed during preparation.

## Reproduce and continue editing

```bash
python3 -B scripts/verify_front_review_r15.py
python3 -B scripts/build_front_review_r15_pdf.py --output /path/to/r15-review.pdf
python3 -B results/evaluation/pr86-r14-20261121/measure_post_r15.py \
  --root . --output /path/to/fresh-post-r15-diagnostic.json
```

For manual editing, use a fresh working folder with `data/images/` copied as `images/` and the r15 review’s `labels/` copied as `labels/`. Open that folder in the [HTML labeler](../tools/cassette_labeler/README.md). Preserve this pinned revision and save later corrections in the next revision.

The next useful evidence is the paired r15 return. A later independent evaluation should reserve whole new camera sessions: repeated development scenes and image-pixel corner scores alone do not establish unseen-scene reliability or millimetre positioning accuracy.
