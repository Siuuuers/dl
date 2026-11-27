# Equal corner averaging with fixed detection boxes

Completed 28 November 2026, Asia/Hong_Kong. This is a CPU analysis of the returned horizontal-reflection predictions, not a new training or inference run.

## Question and decision

The [reflection experiment](reflection-results.md) recovers some cassette fronts but increases duplicate detections and loses four five-pixel successes against the reviewed annotations. Can its alternative corner estimates improve localization while retaining every original detection box?

The fixed equal-weight rule improves five-pixel pair success under both main annotation references. It also loses one ten-pixel success under the fixed reference. It is a useful development candidate, with these tradeoffs retained explicitly. The application defaults remain unchanged.

## Procedure

For each original prediction, find the reflected prediction of the same class with the largest intersection-over-union of their boxes. Accept the association only if each prediction selects the other, their overlap is at least 0.50, and both have two finite, in-image corner coordinates with scores of at least 0.50. Exact overlap ties select the lower partner record index in each direction. Average the two corresponding lower-left coordinates and the two corresponding lower-right coordinates with weights of 0.50 and 0.50. Keep the original prediction when any condition fails.

Every original box, box score, point score, class, identifier and record position remains unchanged. Reference annotations do not select associations, weights, or cutoffs. The rule was fixed before computing this follow-up comparison, after the reflection results had been inspected; this is an exploratory analysis of the same 20 development photographs.

Of 723 original and 718 reflected outputs, 666 original records receive averaged coordinates. There are 43 records without a mutual match, seven with an unavailable original pair, and seven with an unavailable reflected pair. Fifty-six averaged associations have more than one overlap-eligible candidate in at least one direction. A subsequent annotation-based identity audit found no association assigned to two different reference targets in this sample. The association rule itself never uses this identity audit.

## Results

All complete reference targets stay in the denominator, including missed detections. A pair passes only if both returned corners meet the distance criterion.

| Reference | Original pair successes at 5 px | Averaged pair successes at 5 px | Gains / losses at 5 px | Original pair successes at 10 px | Averaged pair successes at 10 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed, 584 complete targets | 511 | 525 | 19 / 5 | 575 | 574 |
| Earlier, 584 complete targets | 511 | 525 | 19 / 5 | 575 | 574 |
| Intermediate review, 583 complete targets | 566 | 569 | 6 / 3 | 582 | 583 |
| Reviewed, 583 complete targets | 566 | 569 | 6 / 3 | 582 | 583 |

The original detection counts and assignments are exactly preserved: fixed-reference misses/extra outputs/duplicates remain 34/91/17; reviewed-reference counts remain 15/84/13. This rule cannot recover a missed detection or remove an extra box because it changes only point coordinates.

The fixed-reference ten-pixel loss is `new_000008/R002`. Its maximum corner error changes from 8.9965 px to 10.0388 px. We retain the equal weights and report the failure rather than select a weight using this already inspected case. Against the reviewed annotations, 14 five-pixel failures remain across five photographs; nine occur in `new_000008`.

## Automatically accepted pairs

Original point scores keep the same accepted records at thresholds 0.50 and 0.80. These scores define comparison cohorts; they are not calibrated confidence in the new coordinates. For the reviewed reference:

| Corner-score threshold | Accepted records | Confirmed within 5 px, original → average | Complete matched pairs over 5 px, original → average | Matched incomplete reference | Unmatched outputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 712 | 566 → 569 | 17 → 14 | 51 | 78 |
| 0.80 | 654 | 566 → 569 | 17 → 14 | 35 | 36 |

The last two categories cannot be described as accurate two-corner outputs. Preserving their identity prevents an apparent localization gain from being manufactured by dropping difficult records. Full tables for all four annotation references are in the [saved summary](evidence/corner-average/summary.json).

## Run and inspect

[Open the CPU notebook in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/thesis-r56-report-and-flip-probe-20261127/notebooks/baomai_corner_average_r58.ipynb), then run its four code cells. The notebook checks its immutable source revision, the original result archive, the saved inference geometry, the reference files, and the reproduced original/reflected result tables before computing the new comparison. Its final cell verifies and downloads `baomai-corner-average-r58-results.zip`.

This replay is optional: the numerical analysis above is already completed and independently verified. A CPU runtime is sufficient. The next new model measurements come from the [two controlled local-refinement notebooks](corner-refinement-experiment.md).

The [evidence index](evidence/corner-average/README.md) identifies every saved output. An [independent implementation](evidence/corner-average/independent-verification.md) reproduces all 723 resulting records, every aggregate count, and the individual gains and losses without importing the averaging function.
