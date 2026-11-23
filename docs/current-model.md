# Current model and R25 training handoff

23 November 2026, Asia/Hong_Kong.

**PR35 remains selected. R24 training and PR99 evaluation are complete;
the [R25 notebook](front-review-r25.md) prepares the newly requested amendment experiment.**
R25 corrects eight left corners in three photographs: four points in one
training photograph and four development-only points. Eleven accepted standard
rows and all non-coordinate fields are preserved. R25 has not been trained yet.

The [complete R24 photo atlas](front-review-r24-pdf.md) contains every nonempty
new-series photo, with the accepted R23-to-R24 corner changes highlighted.
The [PR99 evaluation](../results/evaluation/pr99-r24-20261123/README.md)
records the returned models, fresh inference and same-reference comparisons.
The [improvement report](model-improvement-after-r24.md) explains what changed
in the learned weights, the exact remaining failure rows and useful future
capture types. The R25 compact review packet accompanies the annotation handoff.

## Run inference

Use branch `codex/front-review-r25-20261123`. Follow the
[folder guide](predict-pose-folder.md#use-the-seed46-candidate-explicitly) to
obtain the verified PR35 bundle, retaining its manifest and companion files.

| Setting | Value |
| --- | --- |
| Checkpoint | PR35 seed 46 candidate |
| Checkpoint SHA-256 | `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` |
| Input size | 1280 |
| Box confidence | 0.13 |
| NMS IoU | 0.70 |
| Endpoint confidence | 0.50 |

## Latest completed comparison

Fresh inference uses one common CPU runtime and fixed settings for PR35, the
R24 control, the R24 candidate and the preceding R23 candidate. On the frozen
666-front primary reference:

| Model | Unmatched targets | Unmatched outputs | Both corners within 5 px | Both within 10 px |
| --- | ---: | ---: | ---: | ---: |
| Selected PR35 | 18 | 80 | 533/584 | 582/584 |
| R24 control | 20 | 83 | 531/584 | 583/584 |
| R23 candidate, freshly inferred | 36 | 110 | 521/584 | 575/584 |
| R24 candidate | 35 | 105 | 518/584 | 573/584 |

Relative to R23, R24 gains one match and removes five unmatched outputs; its
frozen-reference corner-success counts decrease. The 584-front denominator
includes every target with two locatable corners, including scored misses.
All four models produce zero outputs on the three empty development scenes.

On the separately identified current R24 exploratory reference (662 fronts),
PR35 has 37 unmatched targets, 103 unmatched outputs and 544/584 both-corner
successes within 5 px. R24 has 16, 90 and 557/584 respectively. Fresh R23,
scored on that same R24 reference, has 17, 95 and 557/584. Both R23 and R24
have 583/584 successes within 10 px there. The reference versions are not
combined and reference edits are not attributed to model learning.

R24 matches all 584 exploratory targets with two locatable corners. Its
remaining 16 unmatched targets have zero or one locatable corner (7 and 9),
and it has 33 duplicate outputs on this reference. Reference-box changes also
alter which predictions meet the fixed IoU matching requirement. An unmatched
target does not necessarily mean the network produced no nearby proposal.
The conflicting reference rankings do not establish a replacement for PR35.

## Completed R24 training and newly authorized R25 work

The [R24 notebook](../notebooks/baomai_front_review_r24.ipynb) remains an immutable
reproduction entry for the completed experiment. Both arms finished 30 epochs
in the returned [PR99 archive](https://github.com/Siuuuers/dl/pull/99).
The user's subsequent feedback supplied new point amendments. The
[R25 handoff](front-review-r25.md) and its new notebook preserve the recipe
while changing only the newly supported supervision.

The R23-to-R24 training input difference is 22 corner corrections across 18
rows in three training photos. The four corrections in photo 010 are in the
development review only. Original trainer validation and frozen scoring
references remain separate from the current exploratory review.

The [completed 16-case review](../results/evaluation/pr99-r24-20261123/partial-front-failure-queue.md)
maps every remaining partial-front miss to its exact accepted row, original
photograph and atlas page. Ten have retained overlapping boxes below the fixed
IoU 0.50 requirement; six have no retained overlap. None loses a qualifying
retained box through assignment to a competing target. These saved outputs
have already passed confidence filtering and NMS, so they do not establish
the raw-network cause of a missing output.

In `new_000011:55-57`, the locatable left endpoints are about 1.96, 3.36 and
1.43 px from the accepted points, while the retained boxes extend too far into
the occluded region. Comparable training examples should be reviewed for
visible-box extent and separate-layer coverage before designing a further
supervision intervention. Keep `008:33`, `008:34` and `009:55` as coverage
checks: PR35 and the control match them on the same R24 reference, but both
R23 and R24 candidates miss them.

The queue records the earlier proposal-filter and occlusion experiments so
that a future intervention starts from the actual remaining failure cases.
Hidden coordinates cannot be inferred from confidence alone. A separate
capture session remains the useful generalization check; repeated development
images do not establish new-scene reliability. The task remains RGB detection
and two 2-D corners.
