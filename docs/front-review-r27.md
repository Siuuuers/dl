# R27: PR103 result review and annotation amendments

23 November 2026, Asia/Hong_Kong.

**The PR103 returned candidate is evaluated, and 21 original-pixel annotation
changes are saved for review. PR35 remains selected under the frozen protocol.**
The two PDFs serve different purposes: one preserves the returned R26 result,
and the other shows every accepted R26-to-R27 label amendment.

## Visual reports

| Report | Complete scope |
| --- | --- |
| `pr103-r26-misses-and-errors.pdf` | 32 pages / 61 cases: all 15 scored misses, all 32 matched corner-failure targets, 8 additional low-overlap box examples, and 6 high-confidence unmatched-output examples. |
| `r27-annotation-amendments.pdf` | 14 pages: all 21 accepted changes across 10 photographs, grouped into 13 cropped original/previous/amended comparisons. Retained-only photographs are omitted. |

The result PDF uses the returned R26 candidate and **unchanged R26 exploratory
661-front reference**. The frozen primary 666-front result is stated separately.
It shows local context and makes no distance claim for an unavailable reference
corner. An overlapping output beside a miss is explicitly identified as
nonqualifying or assigned elsewhere. Unmatched outputs are not automatically
adjudicated background.

Four selected result cases subsequently change in R27: 008:33, 008:34, 009:52
and 056:12. They carry annotation-report cross-references. The saved result,
selection and miss identities remain unchanged. See the
[full PR103 analysis](../results/evaluation/pr103-r26-20261123/README.md),
[result inventory](../results/evaluation/pr103-r26-20261123/candidate-error-inventory.json)
and [amendment inventory](../results/review/front-review-20261123-r27/amendment-pdf-inventory.json).

## The four user-named questions

Coordinates below are original 1280 x 720 pixels. Row identities are one-based
R26 rows and remain the same in R27 because no rows are added or deleted.

| Image / row | Accepted amendment | Evidence and qualification |
| --- | --- | --- |
| 008:33 | xmin 1211.84 to 1230; left point about (1236.69, 450) to (1238, 447); ymax 456 to 455. | The previous left extent includes rack/neighbor material and the point sits on the shadow side of the turn. **Keep xmax 1280**: the visible cassette continues out of frame. A shorter predicted box is not annotation truth. |
| 009:49 | ymin 441 to 440; ymax 461 to 457. | The local dark separator supports the smaller vertical extent. The one-pixel top change has less certainty than the larger bottom cleanup; adjacent rows are checked individually. |
| 007:24 | xmin 652 to 655; left point about (654.06, 247.51) to (656, 246). | Remove the grey rail strip and use the visible lower-front turn. The prediction suggested the neighborhood, but its coordinates were not copied. |
| 007:41 | xmax 924 to 926; both points retained. | Include the visible outer rim. The farther predicted edge would include extra dark rack. The outermost rim and lower-front corner are distinct annotation features. |

The [exact row ledger](../results/review/front-review-20261123-r27/row-review.json)
preserves unrounded before/after token strings, source hashes, visual reasons,
uncertainty, explicit retained decisions and independent review evidence.

## All changed rows

| Photo | Split | Changed rows | Main correction |
| --- | --- | --- | --- |
| 007 | Development | 24, 33, 37, 41, 42 | Rack/neighbor exclusion, overlong lower boxes, two left turns, outer right rim. |
| 008 | Development | 33, 34, 35 | Left material bounds; two left turns; lower bounds at local separators. |
| 009 | Development | 48, 49, 50, 51, 52 | Vertical box limits at each front's own separator. |
| 013 | Training | 27 | Remove excessive left separator strip. |
| 014 | Training | 35 | Move left point from exterior shadow to rounded front turn. |
| 016 | Training | 26, 27 | Tighten left bounds to the sloping cassette outline. |
| 049 | Training | 3 | Right point from rail/shadow gap to visible turn. |
| 051 | Training | 20 | Right point from rack junction to visible turn. |
| 056 | Development | 12 | Left point from rail to turn; hidden right point remains unavailable. |
| 083 | Training | 12 | Right point from rail/lower-shadow junction to visible turn. |

The seven training-row changes comprise three xmin edits and four endpoint edits.
The full review retains 137 training images / 3,786 objects and 20 development
images / 661 objects. These are review-snapshot counts, not a newly staged training
candidate. Original trainer validation remains 20 images / 664 objects.

## What color can establish

The reproducible [color screen](../scripts/rank_front_shadow_candidates.py)
examines all 157 original images and 8,062 locatable endpoints. It compares a
small median-luminance patch at the saved point with nearby inward/upward patches;
8,050 endpoints have complete comparisons and 12 are skipped at borders.
Point/patch contrast is a **review-priority cue**, not an object-material
classifier or a coordinate generator. The tangent comes from two locatable
corners where available, with a documented horizontal fallback for one-point
fronts.

Exactly eight additional high-ranked cases on distinct eligible photos were
inspected. Four support amendments (083:12, 049:3, 051:20, 056:12); two have valid
dark-edge points (114:19, 050:30); two remain too soft to move confidently (037:7,
070:14). These are outcomes from a deliberately selected sample, not estimates
of annotation accuracy, cue precision or problem prevalence. The 74-row manual
review is bounded even though the numerical screen covers the full dataset.

Brightness alone does not identify material: real cassette surfaces can be grey
in shadow, and valid corners often straddle dark boundaries. Follow the visible
contour and the turn of this front, using neighboring fronts to interpret local
structure. Do not enforce a global white threshold, uniform vertical spacing,
or a rule that a box must stop at an adjacent front's corner. Perspective,
occlusion, tilted fronts and the difference between a rim and a keypoint can
invalidate those shortcuts.

Most accepted edges have roughly 1-2 original pixels of qualitative uncertainty;
some soft partial-front transitions are about 2-3 pixels. Fine decimal storage
does not establish subpixel physical accuracy. Retain unresolved coordinates
rather than choosing a new bright pixel without stronger evidence.

## Preserved invariants and next use

There are no deletions. Eleven protected standards, all class/visibility tokens,
all unavailable-point token triplets, image splits, frozen 666/663 references,
the original trainer validation, the photo 015 hold and 31 excluded old-series
treatment photos are preserved. Model predictions were never substituted for
annotation evidence. No whole-dataset certification is claimed.

The [reference-only rescore](../results/review/front-review-20261123-r27/exploratory-comparison.json)
reuses all five saved PR103 prediction sets. The R26 candidate retains the same
15 misses, 92 unmatched outputs and 558/582 two-corner successes. PR35 changes to
40 misses and 107 unmatched outputs because tightened boxes at 007:24 and 009:49
no longer pass matching IoU 0.50. These are reference effects, not model learning.
The result PDF deliberately continues to show the unamended R26 score basis.

A future R27 training comparison has seven eligible changed training rows. Use
the same hardware, software and fixed recipe for both R26 and R27 arms, because
the previous L4-to-A100 change also changed the unchanged control. No new GPU
training, candidate, protocol or notebook is part of this review. See
[current model and next experiment](current-model.md).

## Reproduce

Materialize the original image LFS objects first; their hashes are verified.
These commands run from the repository root and require the existing Python
analysis environment. Renderers additionally use Pillow and ReportLab.

```bash
python3 scripts/front_review_r27.py --root .
python3 results/review/front-review-20261123-r27/verify_independent.py
python3 scripts/rank_front_shadow_candidates.py
python3 scripts/score_front_review_r27.py --root . --output results/review/front-review-20261123-r27/exploratory-comparison.json
python3 scripts/render_front_review_r27_amendments.py --root . --output /tmp/r27-annotation-amendments.pdf --crop-dir /tmp/r27-amendment-crops
python3 scripts/render_pr103_failures.py --help
```

`front_review_r27.py --build` replays the exact accepted decisions from immutable
R26 labels and refuses to overwrite differing immutable artifacts. Default
verification is read-only. PDF inventories bind source images, labels, prediction
records where applicable, renderers and final PDF bytes. Independent review
verification and PDF visual checks are saved beside their respective inventories:
[snapshot audit](../results/verification/front-review-r27-independent.json),
[amendment PDF QA](../results/review/front-review-20261123-r27/amendment-pdf-verification.json),
and [result PDF QA](../results/evaluation/pr103-r26-20261123/pdf-verification.json).
