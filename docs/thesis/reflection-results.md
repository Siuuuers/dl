# Horizontal reflection results

28 November 2026, Asia/Hong_Kong.

## Finding

The returned horizontal-reflection experiment passes its source, input, control-reproduction, repeated-output, coordinate-mapping, and scoring checks. Reflection reduces missed fronts and total extra predictions under the fixed and reviewed references, but increases the duplicate subset. Corner accuracy moves in different directions under the two annotation versions. The result supports further investigation of complementary coordinate estimates; it does not support replacing the unchanged input as an overall improvement.

This note records the fresh model experiment returned in [PR152](https://github.com/Siuuuers/dl/pull/152). Technical provenance belongs here; the thesis explains its methods and results directly without using project links as citations.

## Method

The same research checkpoint processed twenty 1280 by 720 colour photographs. The control used the original image. The treatment reversed the image columns before prediction, using the same input-size setting of 1280. The outputs were returned to the original continuous image frame with x_original = 1280 − x_reflected. The lower-left and lower-right point slots and their scores were exchanged. This was an explicit reflection comparison, with no generic augmented-prediction mode, coordinate clipping added by the experiment, combined detections, or training.

Both arms used box confidence greater than 0.13, non-maximum-suppression overlap 0.70, a maximum of 300 predictions, reference-matching overlap at least 0.50, and corner confidence at least 0.50. The existing suppression after boxes were clipped to the image boundary was applied in the original image frame. Pair-success denominators retain every reference front with two annotated corners, including missed fronts.

The four references label the same photographs. The fixed and earlier versions contain 666 and 663 fronts, respectively, with 584 complete corner pairs in each. The intermediate and reviewed versions each contain 654 fronts with 583 complete pairs. They are alternative annotation versions, not independent test sets.

## Results after the existing boundary suppression

Duplicates are included in extra predictions. The term “extra” describes an unmatched output; it does not by itself establish that the prediction depicts background.

| Reference and input | Missed fronts | Extra predictions | Duplicates | Both corners within 5 pixels | Both within 10 pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed, unchanged | 34 | 91 | 17 | 511/584 | 575/584 |
| Fixed, reflected | 32 | 84 | 23 | 518/584 | 572/584 |
| Earlier, unchanged | 30 | 90 | 17 | 511/584 | 575/584 |
| Earlier, reflected | 30 | 85 | 22 | 517/584 | 571/584 |
| Intermediate review, unchanged | 15 | 84 | 14 | 566/583 | 582/583 |
| Intermediate review, reflected | 11 | 75 | 21 | 562/583 | 583/583 |
| Reviewed, unchanged | 15 | 84 | 13 | 566/583 | 582/583 |
| Reviewed, reflected | 11 | 75 | 22 | 562/583 | 583/583 |

On the fixed reference, misses fall by two and extra predictions by seven, while duplicates increase by six. Five-pixel pair success rises by seven, but ten-pixel success falls by three. On the reviewed reference, misses fall by four and extra predictions by nine; duplicates increase by nine. Five-pixel pair success falls by four while ten-pixel success rises by one.

All three images without annotated fronts remain free of predictions in both arms.

### Paired changes

| Reference | Recovered fronts | Newly missed fronts | Complete pairs gaining 5-pixel success | Complete pairs losing 5-pixel success | Complete pairs gaining 10-pixel success | Complete pairs losing 10-pixel success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed | 4 | 2 | 21 | 14 | 0 | 3 |
| Earlier | 3 | 3 | 20 | 14 | 0 | 4 |
| Intermediate review | 6 | 2 | 8 | 12 | 1 | 0 |
| Reviewed | 6 | 2 | 8 | 12 | 1 | 0 |

The complete-pair rows use the two-locatable-corner group. The returned complementarity file also has an “all” group whose threshold gains include one-corner targets. Those totals must not be described as both-corner gains.

Reflection recovers no complete target under the fixed reference and newly misses one; matched complete targets change from 577 to 576. Under the reviewed reference, both arms match all 583 complete targets. The six recovered and two newly missed reviewed targets all have incomplete corner annotations.

### A box-matching boundary case

For fixed-reference case `new_000057/R012`, the unchanged prediction has box overlap 0.50516, just above the 0.50 assignment threshold. The reflected output still exists with confidence 0.87717, but its overlap is 0.47744. It is therefore counted as a miss even though its two returned corner estimates are only 0.564 and 0.713 pixels from the annotations. The unchanged corner errors were 0.336 and 0.522 pixels.

This example distinguishes losing a reference assignment from losing all useful image evidence. The completed [coordinate-only comparison](corner-average-experiment.md) preserves the original box and isolates the effect of combining point estimates.

## Numerical and runtime checks

The unchanged baseline has a maximum coordinate-component difference of 0.0001220703125 pixels from the archived reference prediction, below the declared tolerance of 0.05 pixels. Its discrete assignments, missed and extra prediction identities, duplicate identities, and endpoint threshold decisions agree exactly. The largest endpoint-error difference is 0.00006766371576 pixels, and the largest probability difference is 0.000001162290573.

Each arm was repeated three times on every image. The repeated prediction records are exactly equal. Independent inverse mapping and suppression reproduce every stored output, and direct use of the original scorer reproduces all 16 arm/reference/stage result tables and 320 per-image score tables. A separate complete-pair calculation reproduces all paired five- and ten-pixel changes.

The runtime records an NVIDIA A100-SXM4-40GB, 32-bit inference, and TF32 disabled. Median decoded-array-to-output time is 14.31 ms for the unchanged input and 20.62 ms for reflection over 60 calls per arm. Both use input tensors of shape 1 by 3 by 736 by 1280 after aspect-preserving padding. These times exclude camera acquisition, file reading, model loading, and input auditing. Timing is descriptive; recognition quality remains the selection priority.

## Completed follow-up and next investigation

A separate [saved-output experiment](corner-average-experiment.md) retains every original-view box, score, and record identity, links reflected detections using predicted boxes alone, and averages only corresponding corner coordinates. Fixed-reference five-pixel successes increase from 511 to 525 and reviewed-reference successes from 566 to 569, with unchanged detection counts. The fixed-reference ten-pixel count falls from 575 to 574. This separates coordinate averaging from the earlier box-union experiments, which increased extra outputs.

Two [local crop-refinement notebooks](corner-refinement-experiment.md) test the next training hypothesis. They preserve detection boxes, apply supervision only to available corners, and report the same all-eligible pair counts. One uses local image content alone; the other also receives the original corner estimates. Their accuracy results remain pending. The existing twenty photographs support development decisions; later acquisition groups are needed to evaluate transfer.

## Before boundary suppression

The same pair-success counts are retained, but the unsuppressed extra and duplicate totals are larger:

| Reference and input | Missed fronts | Extra predictions | Duplicates | Both corners within 5 pixels | Both within 10 pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed, unchanged | 34 | 100 | 24 | 511/584 | 575/584 |
| Fixed, reflected | 32 | 94 | 32 | 518/584 | 572/584 |
| Earlier, unchanged | 30 | 99 | 24 | 511/584 | 575/584 |
| Earlier, reflected | 30 | 95 | 31 | 517/584 | 571/584 |
| Intermediate review, unchanged | 15 | 93 | 22 | 566/583 | 582/583 |
| Intermediate review, reflected | 11 | 85 | 31 | 562/583 | 583/583 |
| Reviewed, unchanged | 15 | 93 | 21 | 566/583 | 582/583 |
| Reviewed, reflected | 11 | 85 | 32 | 562/583 | 583/583 |

## Evidence

The [reproduction instructions and source hashes](evidence/reflection/README.md) identify the immutable returned ZIP, independent audit script, and complete per-target changes. The returned archive remains attached to its original results branch rather than being copied into this documentation folder.
