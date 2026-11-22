# R24: focused corner amendments and the next notebook

23 November 2026, Asia/Hong_Kong.

**The r24 notebook is the next training entry. PR35 remains the working model.**
The [PR97 evaluation](../results/evaluation/pr97-r23-20261123/README.md) is
complete. This round corrects 26 corner points on the original camera pixels,
keeps the two-point model and existing training recipe, and preserves every
earlier review and result.

## Run

[Open r24 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r24-20261123/notebooks/baomai_front_review_r24.ipynb)
or download [baomai_front_review_r24.ipynb](../notebooks/baomai_front_review_r24.ipynb).

1. Select a GPU runtime.
2. Choose **Run all**. Connect Drive when offered to retain completed bundles.
3. Return **`baomai-front-review-r24-results.zip`** after both arms finish.

The notebook pins a full source commit and protocol hash. It verifies the
inputs, runs the existing four-batch resource trial, and trains two independent
arms from original epoch 173. Both use seed 46, 30 epochs, batch 4, input 1280,
FP32, AdamW and initial learning rate 0.0001. The final epoch is selected in
advance, and the resource trial never initializes a full run. Completed-run
checks and Drive/result packaging retain the established behavior.

## What changed

Rows are resolved against immutable r23 labels. Both endpoints of all 21
named rows were inspected, together with the two accepted standards. One
adjacent row, `new_000012:5`, visibly shared the right-point problem between
named rows 4 and 6 and was included explicitly.

| Image | Reviewed amendment rows | Left changes | Right changes | Use |
| --- | --- | ---: | ---: | --- |
| `new_000010` | 15, 17–19 | 4 | 0 | Development review only |
| `new_000012` | 4–9; row 5 is the adjacent addition | 1 | 6 | Training |
| `new_000014` | 21–25, 31–32 | 7 | 0 | Training |
| `new_000016` | 18–19, 25–27 | 5 | 3 | Training |
| **Total** | **22 rows** | **17** | **9** | **26 points** |

Eighteen of the 44 reviewed endpoints remain unchanged. Both accepted
standard rows, **`010:16` and `016:24`**, retain their exact complete lines.
Boxes, class, visibility, row identity, object counts and image splits are
unchanged. All revised points remain locatable (`2`). The other **153 label
files** are byte-identical to r23. The complete review still contains 4,448
fronts and 8,062 locatable corners in 141 nonempty photos, with 16 empty photos
also retained.

The [point decisions](../results/review/front-review-20261123-r24/point-review.json)
record both the moved and retained endpoints, original-image hashes, exact
before/after lines, pixel coordinates and independent visual review reasons.
[Operations](../results/review/front-review-20261123-r24/operations.json) replay
the saved changes exactly; no row is identified by its nearest prediction.

### Original pixels and saved labels

Each sheet reads the actual r23 and r24 label files. The columns show original
pixels, old labels and saved new labels; pink marks old points, green changed
points and cyan retained points. Retained colour identifies an unchanged
coordinate, not a guarantee of physical truth.

- [Photo 010 comparison](../results/review/front-review-20261123-r24/new_000010-front-face-comparison.png)
- [Photo 012 comparison](../results/review/front-review-20261123-r24/new_000012-front-face-comparison.png)
- [Photo 014 comparison](../results/review/front-review-20261123-r24/new_000014-front-face-comparison.png)
- [Photo 016 comparison](../results/review/front-review-20261123-r24/new_000016-front-face-comparison.png)

The [corner policy](annotation-boundaries-r24.md) follows the lower front edge
to its turn into the side of the face. The surrounding shadow and sudden
bright transition are useful cues, considered with geometry and neighboring
context. They do not define a universal brightness threshold: a shadowed white
cassette can look grey, and a rack can reflect light. Existing boxes enclose
the cassette's material extent, so a correct front-face point may sit inside
the box's extreme edge.

Most inspected corners allow approximately 1–2 original pixels of placement
ambiguity. The shadowed right corners `016:25–27` have a softer transition,
approximately 2–3 pixels. These are qualitative visual estimates. Good points
were retained when a move would merely select a different position on the
same rounded arc, including `014:23` right.

## What the notebook actually trains

Both arms keep all **137 training images, including 13 empty scenes**. The
reference uses PR35 supervision; the candidate uses eligible r24 supervision.
Photo 015 and the 31 old-series treatment exclusions use exact PR35 labels in
both arms. The full review and effective training candidate remain distinct.

| Input | Reference | Candidate |
| --- | ---: | ---: |
| Training images | 137 | 137 |
| Training fronts | 3,771 | 3,784 |
| Locatable training points | 6,841 | 6,836 |
| Empty training images | 13 | 13 |

The 18 amended training rows contribute 22 corrected points. The four amended
rows in `new_000010` stay development-only. Original trainer validation remains
**20 images / 664 fronts**, byte-identical in both arms. The frozen **666-front
primary** and **663-front historical** scoring references are unchanged. The
**662-front r24 review** is a separately pinned exploratory reference; its
updated `010` points must not be mixed with PR97's pre-return r23 scores.

This run compares accumulated eligible supervision with PR35 supervision. It
does not isolate the effect of the latest point corrections. Keep the recipe,
operating point and final-epoch rule fixed; inspect the returned evidence before
selecting a replacement model.

## What PR97 establishes

Both r23 arms completed the intended 30-epoch L4 run with verified source,
initializer, input labels and final checkpoints. At the fixed confidence 0.13
operating point, the frozen 666-front reference gives:

| Model | Matched targets | Unmatched targets | Unmatched outputs | Both corners ≤5 px | Both corners ≤10 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained PR35 | 648 | 18 | 80 | 533/584 | 582/584 |
| Fresh r23 control | 646 | 20 | 83 | 531/584 | 583/584 |
| r23 candidate | 630 | 36 | 110 | 521/584 | 575/584 |

All three produce zero outputs on the three empty scenes. The r23 candidate
scores better on the separately pinned exploratory r23 review, which changes
reference geometry. The audit traces how fixed predictions cross the IoU
matching threshold when a reference box changes. An unmatched target is a
scoring outcome, not proof that a physical cassette has no detector output.
These conflicting rankings leave **PR35 selected**.

Against the prior r21 candidate on the same frozen reference, r23 has one fewer
match, ten more unmatched outputs and three more both-corner successes within
five pixels. That is a mixed result, not a demonstrated overall improvement.
See the [complete PR97 evidence](../results/evaluation/pr97-r23-20261123/README.md)
for provenance, identity sensitivity, denominators and reproduction.

## Verification and practical limit

[Candidate verification](../results/review/front-review-20261123-r24/verification.json),
[staged-input readback](../results/review/front-review-20261123-r24/staging-verification.json)
and [notebook verification](../results/review/front-review-20261123-r24/notebook-verification.json)
record the delivered inputs. Preparation checks exact row replay, unchanged
fields and standards, all original image hashes, actual per-arm labels,
validation separation, notebook syntax/schema and its embedded input preflight.
It does not execute a new GPU training run or establish an r24 model gain.

The next useful generalization check is a separately captured and independently
labeled camera session. Repeatedly correcting and measuring the same development
photos cannot establish performance on new scenes. That question can be addressed
without changing the present two-point architecture or making this notebook
harder to use.
