# R37 marked-crop amendments and the PR121 result

R37 accepts the two newly supplied marked images and resolves two previously held annotation fields. It prepares a controlled comparison of eligible R36 and R37 supervision. The PR121 R36 run has been verified and evaluated; R37 has not yet been trained. PR35 remains the selected deployment model.

The practical recommendation is to retain identifiable small fronts, keep the current inference path, and concentrate further review and new data on partial and occluded fronts. A same-checkpoint one-to-one-head probe reduced other unmatched outputs but increased misses and duplicate outputs. Neither that probe nor the R36 result supports switching the deployed model.

## Accepted annotations

Coordinates below use the original 1280 x 720 image: x increases rightward and y downward. Approximate physical placement uncertainty is 2 pixels; decimal serialization is not subpixel measurement accuracy.

| Image / stable row | R36 to R37 change | Physical basis |
| --- | --- | --- |
| `new_000006:53` | Right box edge `503 -> 495` | The yellow marker identifies the right extent of the lower sloping front. The marked upper seam meets the finger near `(495,572)`. Pale material extending toward x503 lies above this specified seam. |
| `new_000034:35` | Observed left point `(518.35,297.09) -> (517,294)`, visibility remains 2 | The orange region registers around `(517.2,294.0)`. The change is mainly upward with a smaller leftward movement. |

For `new_000006:53`, keep the other three edges, observed left point `(465,580)`, and unavailable right point. The thick lower pink line is consistent with approximately 1-2 pixels of placement ambiguity and does not justify changing the saved bottom edge 586. An enclosing rectangle can still contain interior finger pixels.

For `new_000034:35`, retain the entire R36 box `(511,255,637,300)` and right point `(635,284)`. The observed lower-left turn and the box's leftmost rim are different features; the point does not have to equal xmin 511.

The original uploaded JPEGs are preserved unchanged in [the review evidence](../results/review/front-review-20261126-r37/user-markup/). [Screenshot registration](../results/review/front-review-20261126-r37/screenshot-registration.json), original-PNG hashes, exact before/after rows and independent inspection are bound into [the row ledger](../results/review/front-review-20261126-r37/row-review.json). The [boundary policy](annotation-boundaries-r37.md) records the scope and uncertainty.

The earlier `new_000010:24` point request remains pending. Its left point is unavailable; the saved observed point is the right corner at `(909,491)`. Both point triplets are preserved until the intended feature is identified. This inherited hold does not block the two clarified training corrections. The earlier `new_000009:56/57` exclusions remain in place; this round adds or deletes no instance.

## What PR121 actually changed

The returned archive is from [PR121](https://github.com/Siuuuers/dl/pull/121), head `f7ad5dd9297a837bd052995fdf842e2d73b0cd1d`, using training source `83aeca67264e1f98872abeca40debc212c0775e9`. Both independently initialized arms completed the prescribed 30 epochs, 1,050 batches, 4,110 image presentations and 78 optimizer updates. The two R36 training changes were box-only edits in one image. The PR121 control reproduces the PR119 candidate's prediction records exactly on all 20 development images.

All tables retain box confidence 0.13, endpoint confidence 0.50, matching IoU 0.50 and 5/10-original-pixel corner criteria. Primary inference uses the trained one-to-many head followed by NMS at IoU 0.70. Duplicate outputs are a subset of unmatched outputs, not an additional count.

### Preserved R36 development labels: 659 fronts

| Measure | PR121 control: eligible R35 labels | PR121 candidate: eligible R36 labels |
| --- | ---: | ---: |
| Missed reference fronts | 15 | 15 |
| Unmatched outputs | 93 | 94 |
| Duplicate subset | 31 | 31 |
| Other unmatched outputs | 62 | 63 |
| Both corners within 5 px | 570 / 583 | 570 / 583 |
| Both corners within 10 px | 582 / 583 | 582 / 583 |
| Matched rows with a corner error | 19 | 19 |

All 15 missed identities remain the same. The candidate matches all 583 two-locatable-corner fronts, but only 61 of 76 fronts with fewer than two locatable corners. Twelve misses have one locatable corner and three have none. This is an association with visibility and truncation; it does not establish that missing corners cause detection failures.

Seven misses have no output with IoU at least 0.10; eight have an overlapping output below the fixed 0.50 matching threshold. These are geometric triage categories, not physical adjudications. In particular, `new_000011:55` has best IoU about 0.489 and `new_000090:39` about 0.493. An output may be present while failing the box-match criterion. Maximum-cardinality matching recovers no extra matches under the same IoU gate.

The frozen 666 reference remains the primary historical gate: misses stay 36, unmatched outputs rise 107 to 108, duplicates stay 29, 5-pixel pairs improve 517 to 518 out of 584, and 10-pixel pairs stay 573. On historical 663, misses stay 32 and unmatched outputs rise 106 to 107. These references are reported separately and are not pooled with the revised 659-front development labels. The selected PR35 model still has 18 misses and 80 unmatched outputs on frozen 666.

See [the complete PR121 evaluation](../results/evaluation/pr121-r36-20261126/README.md), [comparison](../results/evaluation/pr121-r36-20261126/comparison.json) and [archive verification](../results/evaluation/pr121-r36-20261126/archive-audit.json). The cropped errors PDF covers every candidate miss, corner-error row and unmatched output, with the separate probe and size analysis.

## Small labels: measure the problem before changing the task

The descriptive area threshold below is **1,024 original-image pixels squared**. It is an audit bin, not an adopted annotation or deployment cutoff. A long, thin front can have this area without being a 32 x 32 square.

| R36 reference group | Fronts | Matched | Missed | Duplicate outputs associated with the group |
| --- | ---: | ---: | ---: | ---: |
| Box area below 1,024 px² | 29 | 18 | 11 | 15 |
| Box area at least 1,024 px² | 630 | 626 | 4 | 16 |

Twenty-four of the 29 small references have fewer than two locatable corners. All five small references with two locatable corners are detected. Size and visibility are therefore entangled in this sample. Eleven duplicate outputs also associate with two-corner references; duplicates are not exclusively a tiny-fragment problem.

One reference-only thought experiment removes the 29 small rows in memory while holding all 738 predictions fixed. The score changes from 15 to 4 misses, but true matches fall 644 to 626 and unmatched outputs rise 94 to 112. The duplicate category falls 31 to 16 because its reference targets disappeared; the outputs still exist. This is target censoring, not learned improvement. No labels are deleted from the dataset by this analysis.

Use the established [visible-front policy](annotation-policy.md): retain each identifiable visible front once; correct confirmed duplicate or wrong-object rows using the original pixels; record uncertain ownership explicitly. Deleting a genuine positive does not create an ignore region. Sparse-annotation research explains why unlabeled foreground can be treated as background during detector training ([Suri et al., ICCV 2023](https://www.cs.umd.edu/~sakshams/SparseDet/)); that general mechanism is a reason for caution, not a measured training result for this dataset.

The geometric screen finds no equal reference boxes or reference pairs with IoU at least 0.50. Two containment flags remain review candidates: `new_000010:29/30` and `new_000053:1/2`. Slanted and clipped adjacent fronts can produce contained rectangles. Geometric overlap alone does not prove duplicate labels, and these rows have not been deleted. Exact bins, assignments and the unchanged-prediction thought experiment are in [the size and visibility audit](../results/evaluation/pr121-r36-20261126/size-visibility-audit.json).

## A measured alternative to changing models

The actual checkpoint is YOLO26s Pose with two custom endpoints. Its native training architecture contains both prediction heads. The one-to-one tensors exist and differ from the epoch-173 initializer, so it was possible to test that trained head without another GPU run. [Ultralytics documents the two inference paths](https://docs.ultralytics.com/guides/end2end-detection/); the probe also verifies the behavior of the pinned local 8.4.182 implementation.

The probe starts a fresh model instance from the exact PR121 candidate, sets `nms=False`, verifies that the actual backend is end-to-end, and retains all other inference settings. The NMS IoU argument does not perform suppression on this head. This is a change of trained output head, not simply removal of NMS from the same proposals.

| R36 development measure | Current one-to-many + NMS | One-to-one probe |
| --- | ---: | ---: |
| Returned outputs | 738 | 710 |
| Missed fronts | 15 | 16 |
| Unmatched outputs | 94 | 67 |
| Duplicate subset | 31 | 37 |
| Other unmatched outputs | 63 | 30 |
| Both corners within 5 px | 570 / 583 | 572 / 583 |
| Both corners within 10 px | 582 / 583 | 581 / 583 |

The head recovers five old misses and introduces six new misses. It also returns one zero-height clipped box in `new_000010`, original prediction 37, at y720 and confidence approximately 0.179. The raw cache is preserved. The strict production scorer rejects that invalid geometry; the diagnostic report explicitly counts it as an invalid other-unmatched output, scores the 709 valid boxes and maps every prediction ID back to the untouched cache. It is not silently filtered to improve the result.

On frozen 666, the probe increases misses 36 to 38 and duplicates 29 to 32, although unmatched outputs decrease 108 to 82 and 5-pixel pairs improve 518 to 525. The outcome is a trade-off, not a duplicate solution. The default and selected checkpoint remain unchanged. [The probe comparison](../results/evaluation/pr121-r36-20261126/head-probe-comparison.json) and [model audit](../results/evaluation/pr121-r36-20261126/head-probe-model-audit.json) retain the exact evidence. CPU jobs could overlap, so no latency comparison is claimed.

## R37 notebook and training contract

The next [notebook](../notebooks/baomai_front_review_r37.ipynb) isolates the two accepted training-row corrections. The control uses eligible R36 supervision; the candidate uses eligible R37 supervision. Both start independently from the same original epoch-173 initializer, with seed 46 and the unchanged native 30-epoch recipe. This is an annotation regression experiment, not a claim that two rows will solve the remaining partial-front misses.

The full review still contains 137 training images / 3,786 fronts and 20 development images / 659 fronts. The eligible training overlay has 137 images, 3,784 fronts, 6,837 locatable corners, 731 unavailable corners and 13 empty scenes. The held and excluded training-image policies are inherited. Original trainer validation remains 20 images / 664 fronts. R37 changes no development label bytes, so rescoring fixed PR121 predictions on R37 is equivalent to the R36 rescore; that equivalence is not new learning.

The experiment preserves architecture, loss, input size 1280, FP32, batch 4, AdamW at 1e-4, sampling, augmentation and inference thresholds. Expected exposure per arm is 1,050 batches, 4,110 presentations and 78 optimizer updates. The four-batch resource trial is discarded. Final-epoch checkpoint selection remains fixed.

Open the published notebook in Colab, select an L4 or A100 GPU, and run all cells. It verifies the pinned source, protocol, 157 original image hashes and actual epoch-173 initializer bytes, then runs the resource trial and matched pair. Return `baomai-front-review-r37-results.zip`. Compatible completed runs can be recovered; incompatible or incomplete folders are preserved. The notebook contains five code cells and no embedded repository source payload.

Source publication precedes final notebook generation. Full setup integration uses synthetic input fixtures to exercise checkout, restoration, clean reuse, dirty-source preservation/rejection and wrong-revision rejection. Separately, the generated notebook's original-input function is checked against the actual 157 PNGs and 106,838,542-byte epoch-173 initializer in an isolated clean checkout. These checks do not run R37 GPU training.

## The next improvement after this narrow comparison

1. **Prioritize partial-front coverage.** Review the seven low-overlap misses and eight extent mismatches separately, using the cropped catalogue. Confirm object ownership and visibility before changing boxes. Never move a reference toward a prediction merely to cross IoU 0.50.
2. **Add relevant evidence.** Capture or review training scenes with real finger/gripper occlusion, image-edge truncation, narrow visible fronts, shadow and adjacent rows. Keep labels complete in the scene. Put whole independent capture sessions into a separate final test; do not move the current repeatedly reviewed development photos into training.
3. **Treat the training starting point as a separate experiment.** Loading epoch173 with `resume=False` starts a fresh fine-tuning session: optimizer, learning-rate schedule, EMA and native head-weight schedule restart. It is not a continuation from epoch174. A later comparison could test fresh official pretrained YOLO26s-Pose against epoch173 on the same consolidated supervision, with a declared convergence budget. Random initialization and a different architecture are separate hypotheses. [Official pose documentation](https://docs.ultralytics.com/tasks/pose/) distinguishes these initialization routes, and [training documentation](https://docs.ultralytics.com/modes/train/) distinguishes resumed state from a fresh session.
4. **Use the application requirement to define acceptance.** Continue detecting identifiable visible fronts under the existing policy. If a downstream action needs a minimum visible span, valid corner pair or calibrated positional tolerance, make that an explicit eligibility decision with its own measurements. Missing or uncertain corners must remain distinguishable from observed points. A pixel-area filter should not quietly redefine the annotated world.

The returned curves do not justify simply adding more epochs: the native validation box metric declines during this short run while pose metrics are nearly flat, and trainer-validation boundary conventions differ from the reviewed development labels. Moreover, batch 4 with nominal batch 64 accumulates gradients, yielding 78 optimizer steps over 30 epochs. Native logged loss components describe the one-to-one branch while the optimized objective combines both branches. A broader training change needs its own fixed comparison and success criteria.

The [training-strategy audit](../results/evaluation/pr121-r36-20261126/training-strategy-audit.json) binds the actual checkpoint architecture, pinned native loss implementation, schedules, learning curves and inference-head comparison. A small visible box may describe only a clipped or occluded portion of a full-size cassette; it is not a direct physical-size measurement.

The repeated 20-image development set supports these diagnosis and regression decisions. It does not establish unseen-session reliability or a mechanical grasping tolerance. Better annotation fidelity, better development scores and a better deployed system are related claims that require different evidence.
