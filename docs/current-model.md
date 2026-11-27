# Current model and review status

28 November 2026, Asia/Hong_Kong.

**PR35 remains selected under the established frozen-reference gate.** The latest
completed training is [PR146/R51](partial-classification-r51-results.md): both
arms use the same 142 photographs and 3,999 fronts. Doubling classification
weight for assigned partial targets changes R50 misses 15→11, unmatched 84→112,
formal duplicates 13→23, and both-corner 5 px successes 566→567/583. This is a
tradeoff and does not support automatic promotion. The reference predictions
exactly reproduce the PR137 candidate on all 20 development images.

**Current entry points:** [Final Year Project I report](thesis/README.md),
[completed contrast and input-size results](thesis/r53-results.md),
[completed reflection results](thesis/reflection-results.md),
[coordinate-average analysis and CPU notebook](thesis/corner-average-experiment.md),
and [local-refinement training notebooks](thesis/corner-refinement-experiment.md).

PR149 and PR150 complete the fixed contrast and 1280/1600 probes using the
immutable PR137 research checkpoint. Both reproduction checks pass, but neither
treatment passes the joint quality screen. Primary misses change 34→33 while
both-corner 5 px success falls 511→506/584 for CLAHE and 511→503/584 for size1600.
Fixed cached two-view fusion also increases unmatched and duplicate outputs.

PR152 completes the horizontal-reflection experiment. Control reproduction,
coordinate mapping, repeated predictions, and scoring pass independent replay.
Fixed-reference misses change 34→32 and extras 91→84, but duplicates increase
17→23. Five-pixel pair success rises 511→518/584 under the fixed reference and
falls 566→562/583 under R50. The original notebook and its immutable source pin
remain available for reproduction; a fresh reflection run is not required for
the next CPU analysis.

The completed fixed-box coordinate average modifies 666 of 723 original records
and preserves all original detection outcomes. Fixed-reference five-pixel pair
success rises 511→525/584, while ten-pixel success falls 575→574/584. R50 changes
566→569/583 and 582→583/583. Its rule uses predicted-box association and equal
weights fixed before calculation. The [CPU replay notebook](../notebooks/baomai_corner_average_r58.ipynb)
uses saved outputs; it neither trains nor reruns a model.

The [crop-containment check](thesis/research-corner-quality.md) shows that all 583
matched complete R50 targets retain both reference corners in fixed 1.5x crops,
including all 17 baseline five-pixel failures. The new GPU notebooks compare
[RGB with zero hint channels](../notebooks/baomai_corner_refinement_rgb_r58.ipynb)
against [RGB with initial-corner hints](../notebooks/baomai_corner_refinement_hints_r58.ipynb)
using the same architecture and recipe. Pretrained CPU forward/backward and
masked-gradient smoke checks pass; full GPU training and accuracy results are
pending. No local refiner has been selected for application use.

The self-contained FYP I report has 29 verified Word pages, a 278-word abstract,
40 numbered headings, 18 external IEEE references, nine equations, 11 tables,
and four figures. It contains no project GitHub citations or unexplained
experiment codes. The [source map](thesis/report-source-map.md) preserves exact
technical provenance separately.

The generic live launcher and `weights/base.json` currently select epoch 173,
with confidence 0.25, while the retained PR35 operating point below uses 0.13.
The camera audit explains explicit launch settings and checkpoint verification.
A selected-model document does not itself change the live application's
weights. The reflection and coordinate studies do not change these runtime defaults.

## Completed inference diagnosis and annotation work

[PR143](https://github.com/Siuuuers/dl/pull/143) returns the successful R49
inference experiment. Its fresh baseline exactly reproduces all 20 saved PR137
outputs. Soft-NMS removes one unmatched output but recovers no target; R38/R50
misses remain 15 and two-corner 5 px success remains 566/583. The equivalent
R49/R50 inference question is complete and does not need another run.

For that reference checkpoint, the 15 miss traces comprise ten confidence-gate
cases, three without an IoU-eligible decoded box and two ordinary-NMS cases.
One post-hoc CPU replay at confidence 0.05 recovers six R38/R50 targets but adds 74
unmatched outputs, including outputs in two previously empty scenes.
Confidence 0.13 remains unchanged. These stage causes were measured for the
PR137/reference model; they must not be copied onto the PR146 candidate
without its decoded-candidate evidence.

[The subsequent PR144 R50 return](../results/evaluation/pr144-r50-20261127/README.md)
confirms that prior diagnosis: all 20 decoded tensor files are byte-identical
to PR143, prediction and trace objects are identical, and all four reference
scores agree. This establishes reproducibility on those scenes, not independent
generalization.

[R52](front-review-r52.md) records 10 amended rows/11 boundary fields on
`frame_0160`, with all corner fields and existing holds retained. Its
[CPU study notebook](annotation-study-r52.md) replays the returned training
result and produces the review reports. The user permits an empirical thesis without a required new architecture;
new captures are unavailable for now. All four evaluation label references
share the same repeatedly reviewed development photographs.

The earlier [R40 folder filter](unmatched-output-r40.md) followed the accepted [R38 development amendments](front-review-r38.md), PR123 and [R39 diagnosis](front-diagnosis-r39.md). PR125 returned the identical R39 CPU replay archive. The old R28 notebook below is historical, not the next run.

PR123 has 11 misses and 94 unmatched outputs on accepted R38, compared with PR35's 40 and 114. The historical frozen666 ranking reverses (PR35: 18/80; PR123: 35/106). Current-target performance and historical compatibility remain separate; neither reference is a new capture session. The optional `--postclip-nms` folder flag now implements R39's fixed 0.70 clipped-box suppression. A fresh all-20-image CPU run exactly reproduced 85 unmatched outputs with every existing match and corner outcome preserved. It does not automatically promote PR123 or change inference defaults.

## Historical R28 handoff

23 November 2026, Asia/Hong_Kong.

**PR35 remains the selected checkpoint. The returned R26 experiment in
[PR103](https://github.com/Siuuuers/dl/pull/103) is evaluated. R28 adds 23 supported
annotation amendments after R27 and supplies the next runnable notebook.
R28 has no new training result yet.**

Start with the [R28 review and notebook guide](front-review-r28.md) for the new
amendment PDF, TXT geometry screen and paired training instructions. The
[R27 review guide](front-review-r27.md) preserves the prior annotation round.
The complete
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

## R27 amendment history

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

At the end of R27, the untrained treatment was the seven training rows
013:27, 014:35, 016:26-27, 049:3, 051:20 and 083:12. Compare R26 and R27 training
labels under the same hardware/software/recipe conditions, keeping frozen gates and
recording both arms' actual inputs. The 14 development edits do not teach the
model. No R27 candidate, protocol or training notebook is introduced by this
annotation-and-reporting change. R28 now supersedes that proposed handoff.

## R28 amendment and runnable experiment

R28 reviews 25 requested identities and 8 additional TXT-screen cases. It accepts
23 replacements across 15 photos (5 training / 18 development) and retains 10 rows.
No rows are added, deleted or renumbered; class/visibility tokens and
missing-point zero triplets remain unchanged.
All 11 standards remain exact, including the explicitly revisited 008:19/29.
The 21-page amendment PDF contains every changed row in cropped original/R27/R28
comparisons. Prediction numbers for 009:55 remain ambiguous; that label is kept.

The [R28 notebook](../notebooks/baomai_front_review_r28.ipynb) trains fresh R26
and R28 supervision together from the same original epoch173 initializer.
The exact accumulated treatment is 10 training rows across 9 photos: the seven
untrained R27 rows, refinements of 016:26/27, and new 017:18, 028:59, 094:9 edits.
The [input delta](../results/review/front-review-20261123-r28/training-input-delta.json)
records every before/after line and is embedded in both staged manifests.

Both arms keep 137 training photos / 3,784 objects, the 13 empty scenes, identical
R26 labels on held 015 and 31 old-series treatment exclusions, and original
20-photo / 664-front trainer validation. Use one actual GPU/software runtime
for both arms and the resource trial (A100 recommended), seed 46, 30 epochs,
batch 4, input 1280, FP32, AdamW at 0.0001, and final-epoch selection. The frozen 666/663
references remain unchanged; the amended 661-front development review is
separate exploratory evidence. Run all and return both checkpoints in
`baomai-front-review-r28-results.zip`.

No R28 training, performance gain or promotion is claimed. See the
[complete R28 guide](front-review-r28.md) for axis corrections, screen limits,
the runnable handoff and next-session data priorities.

Fresh capture-session data is still the higher-value addition: partial fronts
beside the gripper, frame-clipped fronts, rounded/shadowed corners, rack/hand
lookalikes, ordinary scenes and empty scenes. Keep identifiable small objects and
honest missing-point visibility; do not remove difficult objects to improve a
score. Split by capture session before training. This task remains RGB front
detection and two image-plane corners; it does not establish metric 3-D or robot
grasp accuracy.
