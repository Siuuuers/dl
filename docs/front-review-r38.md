# R38: review PR123 and replay the amended reference

R38 applies the user's three annotation corrections and five named exclusions after [PR123](https://github.com/Siuuuers/dl/pull/123) returned the R37 training comparison. All eight cases are development images. The 137 training label files and the eligible R37 training inputs are unchanged. The new notebook therefore performs a CPU replay of the returned predictions; it does not launch another training pair with identical supervision.

The recommendation is to retain tight visible-front labels, use the surrounding camera image as context, and prioritize evidence for partial and occluded fronts. PR123 makes a small trade-off on the frozen historical reference and leaves all 15 misses unchanged on the preserved R37 development reference. PR35 remains the selected model.

## The requested annotations

These identities refer to **R37 rows before deletion**. Original image coordinates increase rightward in x and downward in y. The user's physical `ymax/top` corresponds to stored `ymin`; physical `ymin/bottom` corresponds to stored `ymax`. The table uses top and bottom to avoid reversing the intended movement.

| R37 image / row | Accepted change in original pixels | Reason |
| --- | --- | --- |
| `new_000010:29` | Top about 679.91 to 666; left point about (320.91,717.27) to (325,714) | The upper seam and actual lower-left turn support the prediction-guided correction. Other edges, the missing right point, class and visibility are retained. |
| `new_000008:7` | Bottom 33 to 23 | The owned lower edge lies near the left point at y=20 and slopes/fades toward the clipped right side. Bottom 23 encloses that visible extent without retaining the broad separator below. The left point is unchanged. |
| `new_000007:47` | Top 388 to 385 | The selected front's upper seam agrees with the p57 cue. The other edges and both points are unchanged. |

These are integer placements with approximately 2 pixels of visual ambiguity, not subpixel measurements. Model coordinates are a review aid; the original pixels and object ownership determine the accepted geometry. The amendment ledger records the exact prediction cache and ID used for each cue.

The following five R37 rows are removed as explicitly requested: `new_000086:57`, `new_000088:57`, `new_000053:1`, `new_000010:30`, and `new_000009:52`. They are case-specific user exclusions, not a general minimum-size rule and not a claim that all five were duplicate annotations or background. In particular, `new_000009:52` has a visible partial front; its removal changes the scored target set. The preserved R37 evaluation continues to show its original status.

R38 contains 654 development fronts, 1,220 locatable endpoints and 88 unavailable endpoints. The two-corner population remains 583. The original trainer validation set remains 664 fronts; neither it nor either frozen historical reference is rewritten. A deletion-aware row map links every R37 row to its R38 row or exclusion, so later line-number shifts do not change object identity.

The inherited `new_000010:24` landmark question remains unchanged in full. Earlier `new_000009:56/57` exclusions remain in place. No new point is invented for an unavailable corner.

See [the boundary decisions](annotation-boundaries-r38.md), [the row ledger](../results/review/front-review-20261126-r38/row-review.json), and [the R38 reference](../data/reviews/fronts-2026-11-26-r38/manifest.json). The amendment PDF shows the affected rows at matching crop extents with the original pixels, prior reference and accepted result.

## What the returned R37 model actually changed

Both PR123 arms completed the intended 30 epochs, 1,050 batches, 78 optimizer updates and 4,110 image presentations. The treatment was exactly two training rows: one box boundary in `new_000006:53` and one left point in `new_000034:35`. Both arms started independently from the epoch 173 initializer using the same recipe and seed 46. The archive, source, protocol, checkpoints and completed-run receipts have been verified.

Fresh inference uses the fixed primary path: input 1280, one-to-many head with NMS IoU 0.70, box confidence 0.13, endpoint confidence 0.50, matching IoU 0.50, CPU FP32 and the pinned inference runtime. No threshold or output-head search is part of this round. PR123's control reproduces the PR121 candidate prediction records exactly on all 20 development images, which provides a direct continuity check.

### Preserved R37 development reference: 659 fronts

| Measure | PR123 control: eligible R36 supervision | PR123 candidate: eligible R37 supervision |
| --- | ---: | ---: |
| Missed reference fronts | 15 | 15 |
| Unmatched outputs | 94 | 93 |
| Duplicate subset of unmatched outputs | 31 | 31 |
| Other unmatched outputs | 63 | 62 |
| Both corners within 5 px | 570 / 583 | 570 / 583 |
| Both corners within 10 px | 582 / 583 | 582 / 583 |
| Matched rows with a corner error | 19 | 19 |

All 15 missed identities remain the same. One fewer unmatched output is a small measured difference; it does not establish that the remaining partial-front detection problem is solved. Duplicate outputs are already included in the unmatched count and must not be added a second time.

The candidate matches all 583 references with two locatable corners, but only 61 of the 76 with fewer than two. Twelve misses have one locatable corner and three have none. This association supports prioritizing partial visibility and truncation; it does not prove that missing corner labels cause the detection failures. The three empty development scenes produce no outputs. A maximum-cardinality diagnostic finds no extra matches under the same IoU gate, so simply reassigning the existing matches does not remove these misses.

### Frozen historical comparisons

| Reference / measure | Control | Candidate |
| --- | ---: | ---: |
| Frozen666 misses | 36 | 35 |
| Frozen666 unmatched outputs | 108 | 106 |
| Frozen666 duplicate subset | 29 | 29 |
| Frozen666 pairs within 5 px | 518 / 584 | 517 / 584 |
| Frozen666 pairs within 10 px | 573 / 584 | 573 / 584 |
| Historical663 misses | 32 | 31 |
| Historical663 unmatched outputs | 107 | 105 |

The frozen comparison recovers `new_000089:43`, while losing one 5-pixel pair success overall. That is a trade-off, not uniform progress. These historical references have different annotation conventions and populations from R37/R38; their scores are never pooled or presented as interchangeable. The selected PR35 model has 18 misses and 80 unmatched outputs on frozen666, so this return does not justify replacing it.

The [PR123 evaluation](../results/evaluation/pr123-r37-20261126/README.md), [comparison JSON](../results/evaluation/pr123-r37-20261126/comparison.json) and [archive audit](../results/evaluation/pr123-r37-20261126/archive-audit.json) retain the evidence. The errors PDF's main catalogue uses the preserved R37 reference. Its separate R38 section reports only the effect of changing references with predictions held fixed. Removing a missed row from the reference is not a newly detected object.

### The same candidate predictions scored against R38

| Measure | Preserved R37 | Amended R38 |
| --- | ---: | ---: |
| Reference fronts | 659 | 654 |
| Returned predictions | 737 | 737 |
| Matched fronts | 644 | 643 |
| Missed reference fronts | 15 | 11 |
| Unmatched outputs | 93 | 94 |
| Duplicate subset | 31 | 29 |
| Other unmatched outputs | 62 | 65 |
| Matched rows with a corner error | 19 | 18 |

Four excluded rows were previously missed; the fifth, old `new_000010:30`, was matched by p29. Its unchanged prediction becomes other-unmatched. No retained row becomes newly detected or newly missed for this candidate. Two unchanged predictions, `new_000007/p59` and `new_000010/p34`, move from the duplicate category to other-unmatched after the reference geometry changes. Their outputs have not disappeared.

The corrected `new_000010:29` left point changes the measured error from about 5.13 to 0.50 pixels against the same prediction. Both-corner success remains 570/583 within 5 pixels and 582/583 within 10 pixels because this row has only one locatable reference corner. These are changes in measurement against the corrected reference, not effects of another learning run. [The reference-delta ledger](../results/evaluation/pr123-r37-20261126/r38-reference-delta.json) attributes each change using stable R37 identities.

## Should a cassette label include the rack or environment?

Keep the box tied to the visible cassette front. The original 1280x720 image is already the model input, so rack, gripper, adjacent fronts, shadows and other scene information are available outside the target rectangle. The row crops in the PDFs are presentation aids, not the training samples.

The PR123 receipts show rectangular tensors of shape `[4,3,736,1280]`, no mosaic/mixup/cutmix/copy-paste, and mild whole-image geometric and colour augmentation. Augmentation may change which pixels remain visible, but the pipeline does not crop each example to its label box. Tightening a label does not itself remove contextual pixels from the input. See [the input-context audit](../results/review/front-review-20261126-r38/input-context-audit.json).

There are three separate decisions:

| Decision | Current rule | Why it matters |
| --- | --- | --- |
| Input context | Start with the full camera image | The network can use neighboring geometry and scene appearance while recognizing a front. |
| Object extent | Tight rectangle around visible material owned by that front | The box retains a consistent physical meaning instead of mixing cassette and rack extents. |
| Corner target | Actual observed lower-front corners, with independent availability | A rack edge or inferred hidden corner does not become an observed cassette landmark. |

The cassette's own white rim, lip and shaded front material belong in the extent when contour continuity supports ownership. A brightness cutoff is not an ownership test. A slanted or occluded cassette's enclosing rectangle can contain some rack or finger pixels internally; this does not require extending its outer boundary into unrelated material.

Enlarging boxes deliberately to include rack would change the regression target and overlap between neighboring instances. It is not a demonstrated duplicate fix, and it would make box accuracy less directly related to the physical cassette extent. If context becomes a separate experimental hypothesis, change how context is supplied or evaluated while preserving a clearly defined output target.

This is consistent with [Ultralytics' annotation guidance](https://academy.ultralytics.com/courses/dataset-readiness-for-yolo/annotation-best-practices), which recommends tight visible-object boxes and consistent written policies. [Beery et al., ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Beery_Recognition_in_Terra_ECCV_2018_paper.pdf) demonstrate a generalization gap across camera locations. That study motivates testing new capture conditions; it does not prove this model has learned a rack shortcut. The earlier [context-corner pilot](context-corners-2026-11-12.md) already keeps context-derived estimates separate from observed labels.

## The simplest useful next improvement

First make the reference changes reviewable and replay them. The new labels improve the specification of the task, but all eight changes are outside training. Another nominal R37-versus-R38 GPU pair would have identical training supervision and would not test these corrections.

For a subsequent learning experiment, prioritize complete training scenes with the failure modes that persist: partial fronts, narrow visible strips, frame-edge truncation, finger/gripper occlusion, shadows and adjacent-front boundaries. Retain genuine rack/gripper-only negative scenes as well. Keep a whole independent capture session separate for final evaluation, rather than moving the repeatedly reviewed development frames into training.

Continue to track misses, duplicate outputs, other unmatched outputs and corner accuracy separately. A lower total prediction count is not enough: suppression can also remove a neighboring cassette. The prior [same-checkpoint head probe](front-review-r37.md#a-measured-alternative-to-changing-models) reduced unmatched outputs but increased misses and duplicates, so it is not adopted here. Starting with another model or extending training needs a specific hypothesis, a controlled comparison and application-level success criteria.

An operational decision still matters before deployment: whether the system must report every identifiable front or only offer a downstream action when enough geometry is observable. Under the current policy, an identifiable partial front remains a detection target; any action eligibility should be an explicit, separately measured decision. Pixel error is not a calibrated mechanical tolerance without camera geometry and task requirements.

## Notebook and review files

The [R38 notebook](../notebooks/baomai_front_review_r38.ipynb) is a CPU review/replay notebook. It verifies the pinned source, prediction caches, separate reference manifests, deletion-aware identity map and unchanged training supervision, then reruns the fixed comparison and exports `baomai-front-review-r38-replay-results.zip`. It needs no GPU, new checkpoint download or training run.

Use a standard CPU runtime and run all four code cells. The notebook verifies 645 saved label/cache/audit inputs using Python's standard library; it needs no Torch installation. Initial source retrieval requires Git/network access, and verified completed work can be reused offline. The output reports model differences under the same preserved reference separately from reference-only changes. The two PDFs accompany the review outside the Git tree: an eight-page amendment catalogue and a 42-page PR123 misses/errors catalogue. The latter preserves the 36-page R37 case catalogue and adds six pages covering the frozen-reference trade-off, full-frame input context, R38 reference effects and provenance. Their hashes and coverage inventories are recorded with the corresponding review/evaluation evidence.

The notebook pins source revision `d0cccb385633636d59211da915debb18097a0076`. All four complete code cells were executed in order against the actual committed labels and prediction caches, using a temporary local Git remote containing that exact published revision. The replay reproduced the saved comparison and reference-delta results exactly and produced a verified nine-member, 142,049-byte result ZIP. A clean rerun reproduced the same ZIP. Altered prediction caches and eligible training labels were rejected by their hashes; a modified checkout or wrong revision was preserved and rejected. This was a complete local execution of the notebook, not a live Colab UI test or another inference/training run. See [the notebook validation record](../results/review/front-review-20261126-r38/notebook-validation.json) and [delivery manifest](../results/review/front-review-20261126-r38/notebook-delivery.json).

The 20 development images have been reviewed repeatedly. This round supports annotation and regression decisions; it does not establish unseen-session reliability or certify the full dataset.
