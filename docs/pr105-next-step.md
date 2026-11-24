# PR105 R28 return: diagnosis and next decision

This is a summary of the completed accepted-R28, PR105 result and the rationale for the R29 follow-up. Do not amend R28 annotations or replace PR35 automatically.

## Same accepted R28 reference: 661 fronts

| Model | Misses | Unmatched outputs | Both corners within 5 px (of 584) | Within 10 px |
|---|---:|---:|---:|---:|
| Fresh R26 control | 13 | 87 | 566 | 582 |
| R28 candidate | 13 | 89 | 565 | 582 |
| Retained PR35 | 40 | 107 | 545 | 572 |

For the R28 candidate: **584 two-locatable-corner fronts have zero misses**, while 58 one-corner fronts have nine misses and 19 zero-corner fronts have four misses. Overall, **13 misses** are concentrated entirely in the partial-front categories. The original 26-page PR105 PDF (provided separately in the user's Library) includes all 13 misses, 23 matched corner-failure targets, eight additional poor-overlap box cases, and six unmatched outputs.

The preserved decoded-proposal inspection classified seven misses as no qualifying confidence at 0.13 despite a box reaching matching IoU 0.50, and six as no decoded box reaching that overlap. None was attributable to NMS suppression or one-to-one assignment of an otherwise qualifying proposal. This is an output-path diagnosis, not proof of the training-time cause.

## Frozen 666-front selection reference

| Model | Misses | Unmatched outputs | Both corners within 5 px (of 584) | Within 10 px |
|---|---:|---:|---:|---:|
| Fresh R26 control | 36 | 105 | 519 | 573 |
| R28 candidate | 35 | 106 | 516 | 573 |
| Retained PR35 | 18 | 80 | 533 | 582 |

The unchanged-primary-reference ranking conflicts with accepted-R28 exploratory ranking. A revised reference box can alter IoU assignment without changing any model output. **PR35 remains selected under the existing frozen policy.**

## Next hypothesis

Use *the same accepted R28 labels for both training arms* and change only a bounded image-sampling schedule. Keep computation and protected images fixed; score partial and ordinary fronts together. It would be premature to change NMS, point visibility semantics, or lower-front-box conventions based on this output alone. The next run is defined in [R29 handoff](partial-front-sampling-r29.md).
