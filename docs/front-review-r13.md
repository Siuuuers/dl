# Round 13 annotations, complete new-photo atlas and next experiment

21 November 2026, Asia/Hong_Kong.

**The amendments are saved as actual YOLO label files.** All eight requested photographs were inspected against their original PNGs and immutable r12 rows. Seven photographs change: **28 existing rows, comprising 27 box groups, two right-point groups and one left-point group**. No instance or visibility flag is added or deleted. The complete current review still contains all 157 label files.

## PR82 result and the next decision

[PR82](https://github.com/Siuuuers/dl/pull/82) contains the completed r12 return. Its exact archive, both complete checkpoint bundles, all recorded training-source and input hashes, and original epoch 173 initialization passed the [return audit](../results/evaluation/pr82-r12-20261121/verified-return.json). Both arms completed 30 epochs independently from epoch 173 on NVIDIA L4.

All three models were freshly evaluated in one CPU environment at the fixed operating point. [The full measured report](../results/evaluation/pr82-r12-20261121/README.md) preserves the receipts, prediction arrays, fixed scoring code and a bounded inspection of gained/lost identities.

| Reference | Model | Misses | Unmatched outputs | Both corners within 5 px | Both within 10 px |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen primary,666 fronts | PR35 | 18 | 80 | 533/584 | 582/584 |
| Frozen primary,666 fronts | Fresh r12 control | 20 | 83 | 531/584 | 583/584 |
| Frozen primary,666 fronts | r12 candidate | 29 | 97 | 528/584 | 581/584 |
| Pre-return r12 review,662 fronts | PR35 | 36 | 102 | 534/584 | 573/584 |
| Pre-return r12 review,662 fronts | Fresh r12 control | 41 | 108 | 530/584 | 571/584 |
| Pre-return r12 review,662 fronts | r12 candidate | 20 | 92 | 541/584 | 581/584 |

The candidate fits the reviewed target better but regresses against the frozen primary and historical references. The same saved predictions produce both comparisons; no threshold was tuned between them. A scored miss can be an overlapping proposal below the fixed 0.50 matching-IoU cutoff, and an unmatched output can be a duplicate or an extent disagreement. These counts are not direct counts of physically absent/present cassettes.

**PR35 remains selected.** The next notebook keeps the matched recipe and tests the new labels. The candidate still has scored duplicates, and 27 of its 29 primary misses are in the zero/one-locatable-corner groups. These are concrete targets for later investigation. A different architecture is not justified by this one reference-sensitive result. Before any future change to the primary annotation reference, settle the physical target and declare the new rule prospectively; preserve the historical comparison.

The r12 review was pinned before PR82 returned. The later r13 review changes three development label files, so its [post-return diagnostic](../results/evaluation/pr82-r12-20261121/post-r13-diagnostic.json) is reported separately. It leaves all detection-match and 5/10 px pair-success identities unchanged for these three models; one fresh-control unmatched output changes category from duplicate to other. This is a replay of existing predictions, not another training run.

## Saved labels and this round's scope

Use [data/reviews/fronts-2026-11-21-r13/labels](../data/reviews/fronts-2026-11-21-r13/labels/) for the current review. The manifest binds the label bytes and original-image identities. The original `data/labels/` directory remains the preserved source dataset. The notebook explicitly selects the eligible r13 training candidate.

| Changed cohort | Photographs | Amended rows |
| --- | --- | ---: |
| Training | 005,012,026,043 | 19 |
| Development review | 009,011,056 | 9 |

All149 unrequested review files, including 31 old-series files, retain exact r12 bytes. Photo053 was inspected and remains unchanged. The [request ledger](../results/review/front-review-20261121-r13/request-coverage.json) contains 26 grouped dispositions: 13 applied, eight interpreted and five unsupported as written. The [operations](../results/review/front-review-20261121-r13/operations.json) bind every exact before/after row. Eighteen accepted proposals received a separately bound physical review.

The [physical policy](annotation-boundaries-r13.md) keeps visible front/rim extent and actual lower-corner position distinct. Image Y increases downward. Cross-row references use immutable r12 anchors, preserving the target point's other coordinate and visibility. Physical-top/bottom wording is translated explicitly. In005 row49, the stated target is Y528, an upward move from532.5 despite the positive sign in the text; the target removes the point from the rack below the front.

Five clauses remain unapplied for specific geometric reasons:

| Request | Decision |
| --- | --- |
| 005 rows23–25 xmax to row22 xmax | Retain the tapered right edges; X1021 would include background below row22's distinct protrusion. |
| 005 row51 left point to row50 left point | Retain(853,504). Copying the point changes the tier; copying X alone moves inside the visible lower seam. |
| 026 row27 xmax/right point to row26 xmax | Retain row27. X893 at its own lower-corner Y300 is in the dark rack. Row28's supported alignment uses the original row27 X887. |
| 053 row11 xmax/right point to row7 xmax | Retain the visible rim ending near540; X535.64 would clip it. |
| 053 row6 ymin to its right point | Retain the sloping front. Right Y49.36 cannot define either horizontal top or bottom without truncating the face or excluding its left corner at Y67.08. |

No new physical hold was introduced. The existing 18 cases across 16 photographs remain recorded, including photo074's lower-aperture ownership and photo089's possible extra fragment. The 15 affected training photographs retain exact PR35 supervision in both arms. A rejected wording clause is not itself a new physical hold.

## Complete nonempty new-photo PDF

The requested atlas contains **all 110 nonempty `new_*` photographs**, with **4,182 current objects**. Two clickable index pages bring the total to 112 pages. All31 `old_*` photos and all 16 empty-label photos are excluded from the PDF. They remain in the complete annotation package and declared dataset.

Magenta marks this round's changed fields; cyan marks retained fields; amber identifies existing unresolved cases. Every included photograph shows all its current rows, including unchanged photo053. Original RGB pixels, row identities, navigation and readable tags are checked. Final identity and delivery details are recorded in [pdf-delivery.json](../results/review/front-review-20261121-r13/pdf-delivery.json).

## Run the new notebook

[Open r13 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r13-20261121/notebooks/baomai_front_review_r13.ipynb).

1. Choose **Runtime → Change runtime type → GPU**.
2. Choose **Runtime → Run all** and connect Drive when requested.
3. Let both arms finish. Each independently starts from original epoch 173 with seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW and learning rate 0.0001. The final epoch is selected.
4. Return **`baomai-front-review-r13-results.zip`**. Compatible completed arms and the ZIP are verified and backed up to Drive; a rerun can reuse them.

The notebook pins its source commit and protocol hash. Its exact embedded input preflight has been executed against the actual 157 original PNGs and original checkpoint bytes; all seven code cells compile. The [notebook verification receipt](../results/review/front-review-20261121-r13/notebook-verification.json) records the final source pin. Preparation performs no r13 GPU training and no model replacement.

| Training input | Reference | Candidate |
| --- | ---: | ---: |
| Photographs | 137 | 137 |
| Empty scenes | 13 | 13 |
| Fronts | 3,771 | 3,777 |
| Original validation photographs/fronts | 20/664 | 20/664 |

Exactly 78 training label files differ between the arms. The 15 held training photos and the separate 31 old-series exclusions have identical PR35 labels in both arms. Development-review amendments are saved separately from original training validation. Frozen 666/663 scoring references stay fixed. The [runner receipt](../results/review/front-review-20261121-r13/runner-verification.json) verifies the materialized input bytes.

These remain reused development scenes. Their scores do not establish performance on new camera sessions, millimetre positioning accuracy or grasp success.

## Reproduce or edit

```bash
python3 -B scripts/verify_front_review_r13.py
python3 -B scripts/check_front_review_r13.py --output /path/to/fresh-r13-check
python3 -B scripts/build_front_review_r13_pdf.py --output /path/to/r13-all-nonempty-new.pdf
```

For manual editing, open a fresh working folder containing `data/images/` copied as `images/` and the r13 review's `labels/` in the [HTML labeler](../tools/cassette_labeler/README.md). Keep the pinned review immutable and collect later edits into a new revision.
