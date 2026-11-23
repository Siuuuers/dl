# Current model and R27 annotation review

23 November 2026, Asia/Hong_Kong.

**PR35 remains the selected checkpoint. The returned R26 experiment in
[PR103](https://github.com/Siuuuers/dl/pull/103) is evaluated. R27 adds 21 supported
annotation amendments and two reproducible visual reports; it is not a new
training result.**

Start with the [R27 review guide](front-review-r27.md) for changed rows, the color
screen, PDF contents and reproduction commands. The complete
[PR103 evaluation](../results/evaluation/pr103-r26-20261123/README.md) preserves
the original returned result before the R27 reference changes.

## Run the retained model

Follow the [folder prediction guide](predict-pose-folder.md#use-the-seed46-candidate-explicitly)
to obtain the verified PR35 bundle, retaining its manifest and companion files.

| Setting | Value |
| --- | --- |
| Checkpoint | PR35 seed 46 candidate |
| Checkpoint SHA-256 | `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` |
| Input size | 1280 |
| Box confidence | 0.13 |
| NMS IoU | 0.70 |
| Endpoint confidence | 0.50 |

## Returned R26 result

Five checkpoints were freshly inferred on the same 20 original development
photos with one fixed CPU runtime. Both returned R26 arms completed 30 epochs;
archive, checkpoint, source, notebook and staged-input audits passed. No training
was repeated to evaluate the return. All five models return zero outputs on the
three empty scenes.

| Frozen 666-front primary reference | Missed targets | Unmatched outputs | Both corners within 5 px | Both within 10 px |
| --- | ---: | ---: | ---: | ---: |
| Selected PR35 | 18 | 80 | 533/584 | 582/584 |
| R25 control, L4 | 20 | 83 | 531/584 | 583/584 |
| R26 control, A100 | 18 | 80 | 533/584 | 582/584 |
| R25 candidate, L4 | 37 | 108 | 518/584 | 574/584 |
| R26 candidate, A100 | 36 | 108 | 519/584 | 573/584 |

Eligible missed targets remain in the two-corner denominator. An unmatched output
is an assignment outcome, not proof that its pixels are background. The
historical 663-front reference also does not support replacing PR35.

### Current-label diagnostics remain separate

| Unchanged R26 exploratory reference: 661 fronts | Missed targets | Unmatched outputs | Both corners within 5 px | Both within 10 px |
| --- | ---: | ---: | ---: | ---: |
| Selected PR35 | 38 | 105 | 543/584 | 573/584 |
| R25 candidate | 15 | 91 | 557/584 | 582/584 |
| R26 candidate | 15 | 92 | 558/584 | 582/584 |

R25 and R26 miss exactly the same 15 identities on this reference. All 584
two-locatable-corner targets match; the remaining misses have zero or one
locatable corner. The result PDF shows every miss and every matched corner-failure
target, followed by selected low-overlap boxes and unmatched outputs.

The current reference favors recent candidates, while the frozen reference favors
PR35. Retention follows the existing protocol; it is not proof that PR35 is
physically best under every annotation version. Resolve the disagreement through
a stable, physically adjudicated reference and a separate capture-session check.
These reused photos remain development data.

## Why this is not an isolated three-row experiment

The R26 candidate changes only training photo 013 rows 29-31 relative to R25:
three left corners and two xmin trims. The control supervision is identical.
However, R25 trained on L4 and R26 on A100. Their recorded software and recipe
agree, but 706 of 879 saved control tensors differ. The R26 control reproduces all
20 PR35 prediction arrays exactly in fresh common-runtime inference.

Consequently, the small R25-to-R26 change cannot be assigned solely to those three
rows. Common inference conditions do not remove a training-history difference.
R26 versus its own control tests the accumulated eligible annotation treatment.
See the [cross-round input audit](../results/evaluation/pr103-r26-20261123/crossround-input-verification.json).

## R27 amendment and next experiment

R27 replaces 21 rows across 10 photographs: 7 training rows and 14 development
rows. It deletes none. Seventy-four rows across 17 photographs were explicitly
reviewed; 53 are retained. All 11 standards, class/visibility tokens, unavailable
point triplets, splits, original trainer validation, frozen references, photo
015 hold and 31 old-series treatment exclusions are preserved. This is a bounded
review, not certification of every label in the dataset.

The [reference-only R27 rescore](../results/review/front-review-20261123-r27/exploratory-comparison.json)
uses these same saved prediction arrays. R26 stays at 15 misses / 92 unmatched
outputs / 558 and 582 two-corner successes. PR35 changes to 40 / 107 / 542 and 572:
tightened boxes at 007:24 and 009:49 no longer pass the fixed 0.50 matching IoU.
These are reference effects, not learning or weight changes. They do not rewrite
the PR103 return or the frozen results.

For a later training run, the actual new treatment is the seven training rows
013:27, 014:35, 016:26-27, 049:3, 051:20 and 083:12. Compare R26 and R27 training
labels under the same A100/software/recipe conditions, keeping frozen gates and
recording both arms' actual inputs. The 14 development edits do not teach the
model. No R27 candidate, protocol or training notebook is introduced by this
annotation-and-reporting change.

Fresh capture-session data is still the higher-value addition: partial fronts
beside the gripper, frame-clipped fronts, rounded/shadowed corners, rack/hand
lookalikes, ordinary scenes and empty scenes. Keep identifiable small objects and
honest missing-point visibility; do not remove difficult objects to improve a
score. Split by capture session before training. This task remains RGB front
detection and two image-plane corners; it does not establish metric 3-D or robot
grasp accuracy.
