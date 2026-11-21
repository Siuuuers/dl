# Round 16: saved amendments and the next notebook

22 November 2026, Asia/Hong_Kong.

**The amendments are saved as actual YOLO labels.** All 17 requested photographs were inspected against the original images and immutable r15 rows. This pass revises 83 rows and deletes the expressly requested photo066 row22: 84 operations across 15 photos. Replacements change 77 box groups, four left-point groups and five right-point groups; some rows change several fields. No instances are added. The [complete review](../data/reviews/fronts-2026-11-22-r16/manifest.json) retains all 157 label files. [Exact amendment summary](../results/review/front-review-20261122-r16/amendment-summary.json).

**Use [baomai_front_review_r16.ipynb](../notebooks/baomai_front_review_r16.ipynb) for the next comparison.** It explicitly selects the composed r16 training candidate. Current reviewed annotations are under `data/reviews/fronts-2026-11-22-r16/labels/`; `data/labels/` remains the preserved original dataset. The PDF is generated from these saved annotations.

**PR35 remains selected.** The completed PR88 r15 result improves agreement with the reviewed labels but still regresses against the frozen primary reference. This handoff prepares the next controlled annotation experiment; no r16 GPU training or model promotion has occurred.

## What changed and what was retained

The [operation ledger](../results/review/front-review-20261122-r16/operations.json) binds every edit to the exact r15 before-line, after-line, original image hash and physical reason. Cross-row anchors always use immutable r15 coordinates, even when that anchor changes later in this pass. [Row maps](../results/review/front-review-20261122-r16/row-maps.json) preserve all surviving identities; photo066 rows1–21 keep their numbers after the final row22 is removed.

All 140 unrequested files retain exact r15 bytes. Requested photos007 and008 also remain byte-identical after inspection, giving 142 unchanged files overall. All 31 old-series review files are preserved. The [request ledger](../results/review/front-review-20261122-r16/request-coverage.json) contains 45 grouped dispositions: 19 applied, 13 interpreted, three partially applied, eight unsupported as written and two already satisfied. These describe grouped instructions, not individual row counts. A separately bound secondary review corroborates 23 selected operations on photos040/042; it does not certify every annotation in the dataset.

The [physical policy](annotation-boundaries-r16.md) distinguishes the visible front and owned rim from its lower corner. Image X increases rightward and Y downward. Physical-top requests become pixel `ymin` only when the original supports that interpretation; physical-bottom requests can become pixel `ymax`. Cross-row point references normally align X while retaining the target's own Y and visibility. Approximate and directional changes stop at the visible material.

Examples:

- Photo004 row49 uses the exact requested source-edge midpoint, xmax789px.
- Photo040 right points become row29=(786,54) and row32=(775,133)px. Rows30/31/32/33 have xmax785/782/777/774px, respectively.
- Photo042 row2 xmin286px and row8 xmax786px retain their projecting rims. Row7 right point becomes (788,373)px; rows13/14 use the **source** row8 xmax788, not its edited value786.
- Photo066 row22 is deleted. Its D22 marker in the PDF is historical context and never a training label.

These are bounded visual decisions under blur and occlusion, not subpixel physical measurements.

### Literal requests that could not be fully applied

Supported changes on the same photo are saved. Retaining a field does not introduce a whole-photo hold.

| Photo / r15 row | Retained field | Evidence-based reason |
| --- | --- | --- |
| 002 /30 | Left point to row28 left X | At this tier, X428 lies outside its own box and in the external dark gap; the own junction stays433. |
| 003 /1,3 | xmin to the proposed lower-point X | Projecting upper-left rims extend to418/420. Row1's point-X correction is saved separately. |
| 003 /14 | Physical bottom to own right Y | It would exclude the independently located left junction and lower rim; rows27/33's supported bottom changes are applied. |
| 003 /50 | Right point to xmax | The outer rim edge897 is beyond the lower junction892 at its own Y. |
| 004 /27,54 | Right point to xmax | Box edges enclose the outer rim/side; they are not the lower front-to-side junction at the target Y. |
| 007 /47 | Left point to row46 left X | The borrowed coordinate lies outside its own observed front at this tier. |
| 008 /35 | Left point to row31 right X | The borrowed coordinate lands inside the target face rather than on its left junction. Rows15/17's xmin requests are already satisfied at719px. |
| 009 /49 | xmax to own right point | The right point is missing behind the gripper; `0 0 0` cannot supply an edge. Rows7/10's supported trims are saved. |
| 009 /12 | Top seam to row11 left Y | The reference crosses columns and falls below the clipped target's entire box. Other same-stack top-seam changes are applied. |
| 010 /1 | Left point to row5 xmin | The anchor belongs to the neighboring stack, close to row1's right side. |
| 032 /37 | xmax to row35 xmax | Extending625→659px would include gripper material. Row36's small supported extension to659 is applied. |
| 033 /20 | Left Y equal to right Y | The lower seam slopes upward to the right; moving left Y80.26 to64 would place the point inside the face. |

## What the PDF shows

The changed-photo PDF has **16 pages: one clickable index plus 15 photos**, displaying **all 612 current fronts** on those photos. Magenta identifies changed fields; cyan identifies retained fields. Photo066's dashed D22 is a separate deletion marker. Row tags can move for readability without moving the saved geometry.

Photos007/008 are omitted because their bytes did not change. Unchanged photos, old-series photos and empty scenes are omitted from this PDF only; the complete review and both training arms retain their full inventories. The index also identifies the still-open, undrawn photo015 question even though that photo is not a changed page. The [PDF delivery receipt](../results/review/front-review-20261122-r16/pdf-delivery.json) records the saved file and verification evidence.

## PR88: measured results and the selected model

[PR88](https://github.com/Siuuuers/dl/pull/88) is audited at head `e05a985cb2ee21be82adc164c85c99a5ad15fa82`. Its complete 39,781,855-byte archive, all 12 indexed members, source/notebook pins, original images, epoch173 initializer and actual staged inputs passed verification. Both independent L4 arms completed 30 epochs, 1,050 batches and 78 recorded optimizer steps. The [full PR88 audit](../results/evaluation/pr88-r15-20261122/README.md) preserves fresh CPU predictions and a byte-reproducible comparison.

Scoring remains fixed at box confidence0.13, endpoint confidence0.50, NMS IoU0.70, matching IoU0.50 and 1280 input. Eligible missed objects remain failures in the corner-pair denominator. Unmatched outputs are association outcomes, not a claim that each has been independently judged to be background.

| Reference | Model | Misses | Unmatched outputs | Both corners within5px | Within10px |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen primary,666 fronts | PR35 | 18 | 80 | 533/584 | 582/584 |
| Frozen primary | Fresh r15 control | 20 | 83 | 531/584 | 583/584 |
| Frozen primary | r15 candidate | 35 | 101 | 521/584 | 575/584 |
| Pre-return r15 review,662 fronts | PR35 | 35 | 101 | 538/584 | 573/584 |
| Pre-return r15 review | Fresh r15 control | 39 | 106 | 533/584 | 572/584 |
| Pre-return r15 review | r15 candidate | 16 | 86 | 549/584 | 583/584 |

The candidate has 23 fewer misses and 20 fewer unmatched outputs than control on the reviewed target, but 15 more misses and 18 more unmatched outputs on the frozen primary. Its reviewed duplicate subset increases from16 to29. The same predictions produce both tables, so the reference-sensitive ranking cannot be resolved by choosing the favorable table. All three models produce zero outputs on the three empty development scenes.

Against the previous r14 candidate on **identical r15 labels**, the r15 candidate improves misses17→16, unmatched outputs87→86 and duplicates30→29. Aggregate pair counts remain549/584 at5px and583/584 at10px. The only new match is photo011 row56, whose IoU crosses the fixed0.50 boundary from0.496218 to0.504338. This is a small boundary-sensitive score change, not proof of a newly recovered physical front or improved millimetre precision. At5px, photo090 row19 gains success and row46 loses it. The report includes original-image case evidence.

### Separately named post-r16 diagnostic

Only development label files009/010 change in r16. Replaying the same PR88 prediction arrays against this later reference changes only the fresh control: it gains photo009 row19 at the matching threshold and both pixel radii. The existing PR35 and r15 candidate retain their matched and success identities. No inference or training occurs in this replay.

| Existing model on r16 review | Misses | Unmatched outputs | Pairs within5px | Within10px |
| --- | ---: | ---: | ---: | ---: |
| PR35 | 35 | 101 | 538/584 | 573/584 |
| Fresh r15 control | 38 | 105 | 534/584 | 573/584 |
| Returned r15 candidate | 16 | 86 | 549/584 | 583/584 |

The [post-r16 diagnostic](../results/evaluation/pr88-r15-20261122/post-r16-diagnostic.json) reproduces byte-for-byte and does not replace the pre-return or frozen results. It is a baseline for evaluating the next return on the same labels.

## Run the r16 notebook

[Open r16 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r16-20261122/notebooks/baomai_front_review_r16.ipynb).

1. Select **Runtime → Change runtime type → GPU**.
2. Choose **Run all** and connect Drive when requested.
3. Let the separate resource trial and both independently initialized arms complete. Compatible completed work can be recovered from Drive.
4. Return **`baomai-front-review-r16-results.zip`**.

| Actual training input | Reference | Candidate |
| --- | ---: | ---: |
| Photographs | 137 | 137 |
| Empty scenes | 13 | 13 |
| Front instances | 3,771 | 3,785 |
| Locatable endpoint labels | 6,841 | 6,838 |
| Missing endpoint slots | 701 | 732 |
| Original validation photos/fronts | 20/664 | 20/664 |

Exactly 92 training-label files differ. Both arms independently initialize from original epoch173 with seed46, 30epochs, batch4, 1280 input, FP32, AdamW and initial learning rate0.0001. Final-epoch selection stays fixed; the resource trial cannot initialize either arm.

The complete review contains3,787 training fronts. The eligible candidate has3,785 because photo015's separate undrawn possible-front question remains the sole training hold, with exact PR35 labels in both arms. The31 old-series photos remain separately excluded from the annotation treatment, also with exact PR35 labels in both arms. None of these photos is removed. The17 prior displayed-case approvals are inherited unchanged, with no new release this round.

The [materialized-input receipt](../results/review/front-review-20261122-r16/verification.json) checks actual photos, label bytes, counts and held/excluded cohorts. The [notebook verification](../results/review/front-review-20261122-r16/notebook-verification.json) records its source pin, syntax and extracted input preflight. Original664-front trainer validation and frozen666/663 scoring references remain fixed; the662-front r16 development review is a separately versioned exploratory diagnostic. CUDA installation, Drive interaction and full GPU training are not executed during preparation.

## What10px means, and which camera input is supported

**10px is an image-space scoring threshold, not an established gripper tolerance.** A pair passes only when both eligible corners meet the configured validity, confidence and distance conditions. This does not mean all predicted corners are within10px, and it cannot be converted to a single millimetre error using these PNGs alone.

The inspected baomai dataset and current/upstream capture paths provide color images. The capture code enables the BGR8 color stream and reads color frames only; its timestamps do not establish paired depth. D405 hardware can supply depth, but the supplied files do not establish recorded depth, intrinsics, camera-to-robot calibration or a physical gripper tolerance. Continue this notebook with RGB.

The [robot-grasp guide](robot-grasp-readiness.md) records the inspected sources and the geometric distinction. For an ideal calibrated image, `delta_X ≈ Z * delta_u / fx` at fixed depth; therefore the same10px changes physical scale with distance and focal length. Related upstream code supports rays and optional measured-plane localization, but its historical2mm mention is not a verified requirement for this project.

No hardware answer is needed to complete this round. If the task later expands to robot grasping, obtain the real gripper engagement/clearance and contact point, the camera mounting/calibration, and either paired metric depth or a valid measured-plane constraint. These determine the acceptable position/orientation errors and the smallest supported geometry route. Do not choose a physical acceptance limit merely because a pixel score passes it.

## Reproduce and continue editing

```bash
python3 -B scripts/verify_front_review_r16.py
python3 -B scripts/build_front_review_r16_pdf.py --output /path/to/r16-review.pdf
python3 -B results/evaluation/pr88-r15-20261122/measure_post_r16.py \
  --root . --output /path/to/fresh-post-r16-diagnostic.json
```

For manual editing, create a fresh working folder with `data/images/` copied as `images/` and the r16 review's `labels/` copied as `labels/`. Open that folder in the [HTML labeler](../tools/cassette_labeler/README.md). Preserve this pinned revision and save later corrections in a new revision.

The next training evidence is the paired r16 return. A later generalization evaluation should reserve whole new camera sessions; repeatedly reused development views and pixel scores do not establish unseen-session reliability or physical grasp accuracy.
