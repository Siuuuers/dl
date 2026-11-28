# Quality-focal training: fewer misses with more extra outputs

**Decision:** the returned candidate does not replace the retained two-view
method. The paired run recovers six reviewed-reference fronts but adds 34 extra
predictions, including 13 more duplicates. Its five-pixel complete-pair count
improves by two against its matched native control, while its ten-pixel count
and mean maximum-corner error worsen. These are development results on the
existing 20 photographs, with one paired seed and no automatic deployment.

The [frozen protocol](quality-focal-experiment.md) compares native classification
BCE with the same BCE multiplied by squared probability–target difference.
The native target-construction, assignment and normalization rules remain the
same. Realized targets and assignments can evolve differently as the models
learn. This is a QFL-style weighting experiment, not a full GFL detector.

## 1. Independently reproduced results

The return is PR157, commit `f971e880e8aff2859ce4d8ae2923efea826eddb4`, running
source `acaf5d4b4043583fdd81f5ea848b3a6a68ada728`. The evidence ZIP is
7,528,011 bytes with SHA-256
`cf7b7b71349b9412bd5272001f5071afcf8d3b356423deb6c233cb1e6c326c4a`.

The independent replay verifies every inventoried member and all 95 archived
Python files against their published Git blob identities. All three reference
label sets match the source commit. Recomputing from the saved native predictions
exactly reproduces all 18 reference/model/view metric groups, reconstructed
coordinates, associations, suppression records, per-image and per-target cases,
and all nine AP50 rankings. This replay uses no new inference or optimizer.
[Verification](evidence/quality-focal-return/verification.json) ·
[Reproduction instructions](evidence/quality-focal-return/README.md).

The paired receipts record the same 142 training photographs, 3,999 source
fronts, annotations, initial parameter/buffer hash, runtime, seed 46, image order,
and observed per-batch target counts. Each arm completes 30 epochs, 1,080 batches,
4,260 image presentations and the same 80 optimizer steps. The only settings
differences are role-specific data and output paths. Both complete controller
states pass the frozen validation function, and the recorded CPU guard passes
its frozen verifier. This validates the returned experiment record; it is not a
second execution of training or an independent capture-session test.

## 2. Detection and corners at the declared operating settings

Object confidence remains strictly above 0.25, matching IoU at least 0.50,
endpoint confidence at least 0.50, and native/postclip NMS IoU 0.70.
The original supplied model uses its original-view predictions as the benchmark.
Both newly trained arms use the declared two-view corner averaging.

| Reviewed reference: 654 fronts, 583 complete pairs | Misses | Extras | Duplicate subset | Both corners within 5 px | Both within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supplied original model | 34 | 61 | 5 | 555/583 | 577/583 |
| Matched native control | 18 | 49 | 4 | 568/583 | 583/583 |
| QFL candidate | 12 | 83 | 17 | 570/583 | 582/583 |

Duplicates are already included in extras. Relative to the matched control,
QFL adds 13 duplicates and 21 other extras. Precision falls from 92.85% to 88.55%,
while recall rises from 97.25% to 98.17%. All three empty photographs remain
empty at the fixed operating threshold. [All metrics](evidence/quality-focal-return/metrics.csv).

The six recovered references are `new_000008/R001`, `new_000010/R014`,
`new_000010/R030`, `new_000010/R032`, `new_000011/R059` and
`new_000089/R043`. None of the control's matched fronts becomes missed. Every
recovery has fewer than two known reference corners. Five-pixel corner gains
are `new_000008/R006` and `new_000053/R014`, with no five-pixel losses.
[Paired identities](evidence/quality-focal-return/paired-cases.json).

At ten pixels, `new_000008/R011` regresses: its maximum corner error changes from
7.5335 to 18.1026 px. Across the same 583 complete fronts, mean maximum-corner
error rises from 2.18584 to 2.19975 px. Both original-view arms score 567/583 at
five pixels; the two-view counts become 568 and 570. Thus the reported two-view
gain should not be described as an original-coordinate improvement of two cases.

One QFL output has a clipped zero-height box: `new_000057`, original prediction
identity 22, confidence 0.257153. Its rectangle is approximately
[783.870, 0, 880.281, 0]. The source repair in PR156 explicitly counts this as an
unmatched extra, excluded from matching; the strict matcher still rejects
nonpositive boxes. It has not been silently discarded to improve the score.

The earlier retained two-view method remains the practical comparator: 18
misses, 51 extras, five duplicates, 569/583 five-pixel and 583/583 ten-pixel
successes. It is a different historical run, so the matched control above is
the correct comparator for attributing this experiment's treatment difference.
Neither the old method nor the new native control is silently substituted for
the other.

### Reference sensitivity remains

| Annotation reference | Method | Misses | Extras | Duplicates | Within 5 px | Within 10 px |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Supplied annotations | Supplied original model | 39 | 56 | 11 | 521/584 | 568/584 |
| Supplied annotations | Matched native control | 46 | 67 | 7 | 511/584 | 565/584 |
| Supplied annotations | QFL candidate | 44 | 105 | 19 | 514/584 | 563/584 |
| Earlier corrected annotations | Supplied original model | 26 | 41 | 11 | 534/584 | 580/584 |
| Earlier corrected annotations | Matched native control | 34 | 53 | 8 | 522/584 | 574/584 |
| Earlier corrected annotations | QFL candidate | 32 | 91 | 19 | 525/584 | 572/584 |

Both older references retain the advantage of the supplied model in these
corner counts. QFL's small advantage over its matched control at five pixels
does not establish which annotation definition is physically correct.

## 3. Ranking and emitted-output support

AP50 uses original-view outputs, confidence above 0.001, at most 300 detections
per photograph, postclip suppression and 101 interpolated recall points. It is
the prespecified single-class ranking measure, not COCO mAP and not a selected
operating threshold.

| Reference | Supplied original | Native control | QFL candidate |
| --- | ---: | ---: | ---: |
| Reviewed | 0.95864967 | 0.98386492 | 0.98395336 |
| Supplied annotations | 0.94573730 | 0.92492165 | 0.92796918 |
| Earlier corrected | 0.97489236 | 0.95435646 | 0.95788832 |

The reviewed-reference difference is 0.00008844, or 0.008844 percentage point.
At the low floor both trained arms match 651/654 fronts, while the number of
retained predictions rises from 2,768 to 5,183. This small AP difference is not
compelling evidence of a general ranking improvement. It does not establish
statistical equivalence either. [Recomputed ranking](evidence/quality-focal-return/ranking.json).

Scores also change shape. Among matched complete fronts above 0.25, median
confidence falls from 0.9158 to 0.8815. The duplicate pool expands from four
to 17 and its median falls from 0.4521 to 0.2920. These pools are selected by
each model's own predictions, so their distributions are descriptive, not a
paired calibration study. No posthoc confidence threshold is selected.
[Score distributions](evidence/quality-focal-return/score-distributions.json).

At endpoint confidence at least 0.50, with finite in-image points, the native
control offers 676 complete coordinate pairs and QFL offers 712. Their breakdown
under the reviewed reference is:

| Emitted pair category | Native control | QFL candidate |
| --- | ---: | ---: |
| Matched complete reference, both within 5 px | 568 | 570 |
| Matched complete reference, over 5 px | 15 | 13 |
| Matched reference with unavailable corner(s) | 48 | 54 |
| Unmatched output | 45 | 75 |

The last two rows lack complete reference support; this does not prove physical
error. The denominator 583 describes eligible reference pairs, not all emitted
coordinates. These support counts matter for the intended robotic-arm consumer,
without making a claim about arm control, calibration or physical tolerance.
[Emitted-pair support](evidence/quality-focal-return/emitted-pair-support.json).

## 4. What the training traces do and do not explain

Both arms contain 18,626 partial-target presentations after augmentation.
The one-to-many branch leaves four unassigned in the native control and three
in QFL; the one-to-one branch leaves one unassigned in each arm. Every complete
target presentation receives an assignment. All recorded foreground anchors
also have positive class targets. These aggregates do not support widespread
assignment starvation as the explanation for the current errors.

Mean soft target q, weighted by foreground-anchor count across training, is:

| Branch and group | Native control | QFL candidate |
| --- | ---: | ---: |
| One-to-many, partial | 0.54759 | 0.56218 |
| One-to-many, complete | 0.79418 | 0.80037 |
| One-to-one, partial | 0.80833 | 0.81080 |
| One-to-one, complete | 0.92456 | 0.92495 |

Partial targets have lower mean supervision values, but the traces combine
zero and one known corner and do not retain individual p/q/CIoU/OKS values.
Size, appearance, corner availability and native selection co-vary. These means
cannot explain a particular evaluation miss or establish harmful pose coupling.

QFL changes numerical loss emphasis substantially. Summing per-batch BCE terms
after the recorded target-score normalization, classification gain and native
branch schedule, but before the batch-size factor, yields 358.187 for the native
control and 40.597 for the QFL objective. Background shares also differ by
branch. These are loss-value summaries, **not gradient norms**: reduced scalar
loss does not show that learning improved or quantify the effective parameter
update. In candidate traces, the recorded “native” BCE is calculated on the
candidate's evolving predictions; it is not the separately trained control.
[Trace definitions and totals](evidence/quality-focal-return/training-diagnostics.json) ·
[Per-epoch values](evidence/quality-focal-return/training-by-epoch.csv).

## 5. Next experiment

The next useful step is a frozen-checkpoint score/assignment audit, without an
optimizer. Use both returned checkpoints, the unchanged training photographs
and labels, deterministic unaugmented inputs, and the exact native loss sources.
Separate zero, one and two known corners; retain candidate eligibility,
pre-conflict selection, surviving assignment, predicted p, target q, clamped
CIoU, known-only OKS and their mixed quality. Confirm that observation leaves
the native assignments unchanged.

A box-only-overlap counterfactual may then describe how pose coupling changes
assignment at the same frozen predictions. It is not a trained detector or proof
that removing pose quality improves recognition. The current evidence does not
justify immediately adding VFL, NWD, another detector, or another threshold sweep.
The separate [controlled training-label comparison](training-label-results.md) is also complete. It improves agreement under the reviewed reference, while increasing duplicates and reversing rank under both older references. The [fixed-weight diagnostic guide](score-assignment-audit.md) records implementation and execution status.
