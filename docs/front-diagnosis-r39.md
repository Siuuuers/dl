# R39: improve partial-front recall and remove redundant border outputs

26 November 2026, Asia/Hong_Kong.

**Prioritize the remaining partial-front misses. A small additional NMS pass after clipping is a promising separate duplicate reduction.** R39 uses the returned PR123 candidate and accepted R38 annotations. It explains the current failures and measures two bounded inference alternatives before another learning experiment. The new [CPU notebook](../notebooks/baomai_front_diagnosis_r39.ipynb) replays the completed evidence; running it needs no GPU, images, weights or training.

All eight R38 amendments remain accepted. They changed development labels only; the eligible training supervision is unchanged. The existing [R38 notebook](../notebooks/baomai_front_review_r38.ipynb) and [review guide](front-review-r38.md) preserve that completed work. R39 changes no target labels or default deployment settings.

## What deserves attention

The accepted R38 development reference contains 654 fronts. PR123 matches all 583 with two locatable reference corners, and 60 of the 71 with fewer than two. Overall detection recall is 643/654 (98.3%); recall in the partial-visibility group is only 60/71 (84.5%). Every one of the 11 remaining misses is in that group. This is an observed association, not proof that missing endpoint labels cause detection failure.

The fixed evaluator has three related outcomes:

| Outcome | Meaning | PR123 on R38 |
| --- | --- | ---: |
| Missed reference | A target receives no one-to-one match at IoU 0.50 | 11 |
| Duplicate output | An unmatched output overlaps an already matched reference at IoU 0.50 | 29 |
| Other unmatched output | An output is unassigned and does not meet that duplicate rule | 65 |

There are **94 unmatched outputs in total: 29 + 65**. A duplicate prediction is not a duplicate annotation row. An unmatched output is not automatically confirmed background. A poorly positioned box can produce both a missed reference and an unmatched output. Six baseline misses have such overlapping, below-threshold outputs, so these counts are not independent physical mistakes.

## The new raw-proposal trace

Fresh CPU inference copied the decoded one-to-many predictions before confidence filtering and NMS. All twenty final baseline arrays exactly reproduced the accepted cache, including scores, coordinates, keypoints, ordering and the three empty scenes. The trace kept the established confidence 0.13, NMS IoU 0.70, input size 1280, FP32 and matching IoU 0.50.

| Immediate failure stage | Count | Next useful lever |
| --- | ---: | --- |
| Qualifying box exists but scores at or below 0.13 | 6 | Improve evidence for partial-front appearance; quantify recall versus extra outputs before lowering a global cutoff |
| No decoded box reaches reference IoU 0.50 | 4 | Improve extent/localization evidence for the difficult occlusion and neighboring-front pattern |
| Qualifying above-threshold box is suppressed by NMS | 1 | Inspect box ranking and the suppression interaction for this case |
| Output cap or retained matching assignment | 0 | Increasing max_det or rewriting the matcher does not address these recorded failures |

All four geometry cases are `new_000011` rows 55, 56, 57 and 60. Row 55 has maximum raw IoU 0.49935: record its threshold sensitivity without moving the label or cutoff to force a pass.

The concrete NMS case is `new_000090:39`. A qualifying proposal scores 0.67983 with reference IoU 0.53746. A higher-scored output scores 0.77317 but has reference IoU 0.48971. Their mutual input-space IoU is 0.89866, so ordinary NMS keeps the higher score. Confidence ranks class evidence; it does not guarantee the best geometric alignment with a particular reference.

These are diagnoses of where this pipeline drops a match. They do not, by themselves, establish which training intervention will fix it. The [complete trace](../results/evaluation/pr123-r39-diagnosis-20261126/README.md) preserves every case and its original R37 identity.

## Two small, distinct inference experiments

| Same PR123 weights; accepted R38 reference | Existing output | Extra NMS after clipping, IoU 0.70 | Lower confidence to 0.05 |
| --- | ---: | ---: | ---: |
| Returned outputs | 737 | 728 | 810 |
| Matched fronts | 643 | 643 | 648 |
| Missed fronts | 11 | 11 | 6 |
| All unmatched outputs | 94 | 85 | 162 |
| Duplicate subset | 29 | 21 | 48 |
| Other unmatched outputs | 65 | 64 | 114 |
| Both corners within 5 px | 570/583 | 570/583 | 570/583 |
| Both corners within 10 px | 582/583 | 582/583 | 582/583 |
| Outputs on three empty scenes | 0 | 0 | 0 |

### Why clipping explains some duplicates

Ordinary NMS compares boxes before clipping them to the original image. At a frame boundary, two boxes can differ substantially outside the image and become almost identical after clipping. Their prediction-to-prediction overlap also differs from the prediction-to-reference overlap used to classify a duplicate.

For all 29 baseline duplicates, the input-space overlap with the corresponding matched output is below 0.70 (range 0.47444–0.69889). Eight of those pairs cross 0.70 after clipping. Thus these outputs do not show a broken original NMS implementation.

One extra confidence-ordered, same-class NMS pass on the final clipped boxes, at the **same 0.70 threshold**, removes nine outputs: eight R38 duplicates and one other-unmatched output. It preserves every existing reference-to-original-prediction association, corner outcomes and empty-scene results. On frozen666 it also retains all 631 existing matches and reduces unmatched outputs from 106 to 97. This is a measured candidate improvement on these reused development images, not an unseen-scene guarantee or automatic change to the selected model.

An assistant original-pixel review of all nine removed outputs supports eight competing boxes on the same top-clipped cassette fronts. The ninth is a redundant box on a blue/gray background item outside the rack; its retained companion remains a false proposal in that inspected region. This is a bounded review of these nine removals, not adjudication of all other unmatched outputs. See [the visual review record](../results/evaluation/pr123-r39-diagnosis-20261126/clip-nms-visual-review.json).

The [duplicate overlap audit](../results/evaluation/pr123-r39-diagnosis-20261126/duplicate-overlap-audit.json) and [clipped-box probe](../results/evaluation/pr123-r39-diagnosis-20261126/clip-nms-probe.json) preserve the exact pairs and removed original prediction IDs. This additional filtering cannot recover a missing proposal; it is separate from the recall work.

### Why a global confidence cut is expensive

The 0.05 counterfactual reuses the exact same decoded tensors and NMS. It recovers five reference matches while preserving every original output. The cost is **73 additional proposals: five matches and 68 unmatched outputs**, including 19 additional duplicates. Partial-group recall becomes 65/71 (91.5%). Two-corner success does not change.

That trade-off can be useful for a human-assisted proposal queue, but it does not justify silently replacing the established cutoff. The added proposals have not all been physically adjudicated. The new notebook reports this alternative explicitly and leaves inference defaults unchanged.

## The next learning experiment should follow the failure pattern

Use the four `new_000011` failures and six confidence failures as a specification for relevant **training scenes**, not as training samples themselves. Review complete eligible scenes with comparable gripper occlusion, narrow exposed fronts, frame-edge truncation, shaded material and neighboring-front separation. Every identifiable target in a newly admitted image needs a consistent label; adding a crop or just one attractive annotation can hide incomplete supervision elsewhere in that scene.

Keep the current native loss and full-image input while testing one meaningful data change. Earlier partial-rich sampling, loss weighting, larger-resolution and output-head experiments already exist and had trade-offs. R39 does not present them as untried solutions or launch an identical supervision comparison.

New camera capture is not the only possible data source: the [additional-frame pilot](additional-annotation-pilot-2026-11-07.md) records 320 recovered upstream frames, with 317 still lacking complete review in that pilot and three quarantined examples. Their current eligibility and capture relationships must be established before use. These frames are neither automatically training-ready nor automatically an independent test set.

For the eventual selection decision, use a separate capture session under the accepted current annotation policy. Group related frames by capture sequence/session when splitting. A frozen label file on repeatedly inspected photographs is a regression reference, not fresh test evidence.

## A model-selection blind spot

| Reference | PR35 misses / unmatched | PR123 misses / unmatched |
| --- | ---: | ---: |
| Accepted R38 | 40 / 114 | 11 / 94 |
| Frozen666 | 18 / 80 | 35 / 106 |

The ranking reverses with the annotation convention on the same development photographs. PR35 remains the selected checkpoint under the established frozen gate. PR123 is stronger against the current accepted R38 target on these images. Both statements should remain visible; neither establishes performance on a new capture session. Do not pool the denominators or call one model universally better.

## Keep the physical target simple

The model already receives the full 1280×720 camera frame, including rack and gripper. Keep each target box around visible material owned by that cassette front, including its own rim where appropriate. Deliberately adding rack to the box changes the regression target without adding input context. Background incidentally inside a rectangular envelope is different from extending the target boundary to unrelated material.

The established endpoint contract also remains: the model may predict obscured corners, each predicted endpoint passes its own display gate, and the line is drawn when both qualify. Naturally unlocatable training/reference corners remain canonical missing slots. Prediction confidence is not proof of physical visibility or coordinate accuracy.

Before adding deployment machinery, clarify the next use: reviewed still-image proposals, unattended counts, or robot target selection. If robot action is next, specify whether re-observation is allowed and the physical positional tolerance. The existing 5/10-pixel reports do not supply a calibrated robot tolerance.

## Notebook use and reproducibility

Open `baomai_front_diagnosis_r39.ipynb` in Colab with a standard CPU runtime and run its three code cells. It fetches a small immutable subset of source/evidence files, recomputes the saved comparisons using Python's standard library, prints the per-case queue, and downloads `baomai-front-diagnosis-r39-results.zip`.

The notebook is an evidence replay of completed fresh inference. Full raw tensors are omitted to keep delivery small; each tensor is fingerprinted, and every IoU-eligible proposal for the eleven misses is retained. Reconstructing the completeness of those lists or the absence of other raw geometry requires the full diagnostic command with the original images and checkpoint. No new GPU training is needed to reproduce the compact scores.

External background: [Ultralytics annotation guidance](https://academy.ultralytics.com/courses/dataset-readiness-for-yolo/annotation-best-practices), [prediction/NMS settings](https://docs.ultralytics.com/modes/predict/), and [group-aware evaluation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data). The project counts above come from the linked repository evidence, not those general sources.
