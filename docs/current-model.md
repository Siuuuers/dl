# Current model and R26 training handoff

23 November 2026, Asia/Hong_Kong.

**PR35 remains the selected checkpoint. The returned R25 experiment in
[PR101](https://github.com/Siuuuers/dl/pull/101) is evaluated; R26 annotations
and the [next notebook](front-review-r26.md) are prepared. R26 has not been trained.**

R26 corrects 19 rows, deletes the unsupported development patch 008:36, and
retains five named rows after original-pixel review. Only three rows in training
photo 013 change the supervision: three left corners and two xmin trims. All eight
user-named photographs remain in development. Eleven accepted standards are
preserved exactly. See the [boundary policy](annotation-boundaries-r26.md) and
[exact amendment ledger](../results/review/front-review-20261123-r26/row-review.json).

## Run the retained model

Use branch `codex/front-review-r26-20261123`. Follow the
[folder guide](predict-pose-folder.md#use-the-seed46-candidate-explicitly) to
obtain the verified PR35 bundle, retaining its manifest and companion files.

| Setting | Value |
| --- | --- |
| Checkpoint | PR35 seed 46 candidate |
| Checkpoint SHA-256 | `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` |
| Input size |1280 |
| Box confidence |0.13 |
| NMS IoU |0.70 |
| Endpoint confidence |0.50 |

## Returned R25 results

Fresh inference uses one common CPU runtime and fixed settings for PR35, R25
control, R24 candidate and R25 candidate. No training was repeated to evaluate
the return. Both R25 arms completed 30 epochs and their provenance audit passed.

| Frozen 666-front primary reference | Missed targets | Unmatched outputs | Both corners within 5 px | Both within 10 px |
| --- | ---: | ---: | ---: | ---: |
| Selected PR35 |18 |80 |533/584 |582/584 |
| R25 matched control |20 |83 |531/584 |583/584 |
| R24 candidate |35 |105 |518/584 |573/584 |
| R25 candidate |37 |108 |518/584 |574/584 |

The denominator includes all 584 targets with two locatable corners, including
scored misses. All four models produce zero outputs on the three empty scenes.
The historical 663-front reference also does not support replacing PR35.

On the identical pre-return R25 exploratory reference, R24/R25 have 16/17 misses,
90/92 unmatched outputs and 557/584 five-pixel pairs each. Their ten-pixel pair
counts are 583/584 each. R25 loses the match at 089:49; its best retained box has
IoU 0.445 rather than R24's 0.711. Under the frozen reference it additionally loses
089:43 at the fixed 0.50 overlap threshold. These saved filtered outputs do not
establish which raw proposal-generation or suppression mechanism caused that.

The [PR101 evaluation](../results/evaluation/pr101-r25-20261123/README.md)
contains the full same-reference comparison, exact failure identities and
provenance. The input audit finds exactly four changed training left corners
between R24 and R25, with 879/879 identical matched-control state tensors. This
supports a bounded four-point experiment interpretation, not a new-session
accuracy claim or a need to change the architecture.

## Separate R26 reference check

The new R26 exploratory review contains 661 development fronts. The same saved
predictions, without inference or retraining, give:

| Current R26 exploratory reference | Missed targets | Unmatched outputs | Both corners within 5 px | Both within 10 px |
| --- | ---: | ---: | ---: | ---: |
| PR35 |38 |105 |543/584 |573/584 |
| R25 matched control |40 |108 |539/584 |573/584 |
| R24 candidate |15 |90 |557/584 |582/584 |
| R25 candidate |15 |91 |557/584 |582/584 |

These numbers are a [reference-only rescore](../results/review/front-review-20261123-r26/exploratory-comparison.json).
They must not replace or be pooled with the frozen results. The deletion removes
one previously unmatched reference row; tighter 089:48/49 reference boxes cause two existing PR35
matches to fail, while 089:49 becomes a match for R25. No network weight changed.

The current reference favors recent candidates, while the frozen reference
favors PR35. Retention follows the existing selection protocol; it is not proof
that PR35 is physically best under every accepted annotation version. Resolve
that disagreement with a stable, physically adjudicated reference and a separate
capture-session check, rather than selecting whichever reference gives a desired
ranking. The reused 20 images remain development data.

## Focused prediction review

The requested PDF reviews retained PR35's 16 largest measurable matched-corner
errors above 5 px and eight other lowest-overlap matches below IoU 0.70. It excludes
missed targets and unmatched outputs. Each case shows an unmarked crop and a
reference/prediction comparison, including nearby scene context. Unlocatable
corners have no scored error. The [inventory](../results/review/front-review-20261123-r26/prediction-error-inventory.json)
contains all 623 matched R26 targets, the exact 24 selected identities and PDF hash.

Reproduce it from the hash-bound originals and saved prediction records:

```bash
python3 scripts/render_front_review_r26_errors.py --help
```

## Next improvement

Keep the architecture, inference settings and fixed paired recipe. Test the
three supported training-row corrections in R26 as a small, explicit experiment,
and compare its return with both PR35 and the saved R25 predecessor under the
same references. No favorable result is promised by the amendment itself.

The higher-value data addition is a separate camera session with more examples
of narrow fronts beside the gripper, frame-clipped fronts, rounded/shadowed lower
corners, and rack/hand lookalikes. Include ordinary and empty scenes. Preserve
identifiable small instances and missing-point visibility; do not remove hard
objects to improve a score. Split by capture session before training so that
near-duplicate views do not serve as independent evidence.

Current original trainer validation remains 20 images / 664 objects. Frozen 666/663
references, the whole-photo hold for 015, and 31 excluded old-series treatment
photos remain unchanged. The model task is still RGB front detection and two
image-plane corners; this work does not establish metric 3-D or robot grasp accuracy.
