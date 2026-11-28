# Which notebook to run

28 November 2026, Asia/Hong_Kong.

Both paired GPU experiments have returned. Their saved outputs and training receipts have been independently verified, so there is no need to rerun them merely to obtain the reported comparisons. The score-target and duplicate-audit notebooks have also completed. The latest partial-front inclusion notebook replays the effect of four justified reference additions from saved outputs, with no model inference or training.

| Role | Notebook | Question | Runtime and status |
| --- | --- | --- | --- |
| Completed reference replay | [Partial-front inclusion](../../notebooks/baomai_partial_inclusion_r67.ipynb) | How do four restored fronts change the scores of six unchanged methods and the interpretation of existing fixed filters? | All four actual CPU cells completed locally from pinned source and preseeded archives; [guide and evidence](partial-inclusion-r67.md). |
| Completed diagnostic | [Duplicate and unmatched audit](../../notebooks/baomai_duplicate_audit_r66.ipynb) | What do reflected support and extra suppression remove, and does complete-pair scoring hide partial-front losses? | CPU; saved predictions only. [Results and execution evidence](duplicate-audit-r66.md). |
| Completed diagnostic | [Score and assignment audit](../../notebooks/baomai_score_assignment_r65.ipynb) | For each zero-, one- or two-corner target, how do native assignments, quality targets and predicted scores relate? | All four actual cells completed on CPU; 284 full-pass forwards, no optimizer. See [execution evidence](evidence/score-assignment-r65/README.md). |
| Completed comparison | [Classification-loss comparison](../../notebooks/baomai_quality_focal_r61.ipynb) | Does QFL improve misses, extras and corner results relative to a matched native control? | PR157 returned; [independent score/AP replay](quality-focal-results.md) complete. |
| Completed comparison | [Controlled training-label comparison](../../notebooks/baomai_training_label_comparison_r61.ipynb) | What changes when only the training annotations change on the same 137 photographs? | PR158 returned; [independent all-reference replay](training-label-results.md) complete. |
| Completed diagnostic | [Proposal-stage and box-geometry audit](../../notebooks/baomai_proposal_audit_r63.ipynb) | Where do the historical candidate's misses lose eligible proposals? | CPU; all four unchanged cells completed from published source; [evidence](evidence/proposal-stages/README.md). |
| Completed diagnostic | [Training-augmentation audit](../../notebooks/baomai_augmentation_audit_r63.ipynb) | How do the existing transforms affect target retention and corner availability? | CPU; all four unchanged cells completed from published source; [evidence](evidence/augmentation-audit/README.md). |

## Completed partial-front inclusion replay

The [r67 guide](partial-inclusion-r67.md) records four exact historical restorations across three photographs and three held cases. Its [manifest](../../data/reviews/fronts-2026-11-28-r67/manifest.json) preserves all 654 r50 rows and appends four rows: 658 fronts comprise 18 with no known corner, 57 with one and 583 with two, for 1,223 known corners. Historical evaluation versions and training labels remain unchanged. The [research note](research-partial-inclusion-r67.md) explains the supplied guidance and the inclusion/availability distinction.

All six methods gain one reference match, gain three misses and lose one other-extra; duplicates and complete-pair scores stay unchanged. The already emitted `new_000010/P029` matches the restored bottom band. No retained target is reassigned. The separate eight-row supplement repeats r66's existing support and additional-suppression restrictions for PR158 reviewed labels. Every restriction retains that P029, while losses of other partial fronts remain. The [evidence guide](evidence/partial-inclusion-r67/README.md) links all measurements, exact transitions and independent verification.

For optional reproduction, select a CPU runtime and run all four code cells. The notebook pins `2c3886271c12e501d236a5eff54e3eb78007d1b0` and verifies the candidate and three immutable prediction archives. All four actual cells completed unchanged in one local IPython session using a clean pinned checkout and preseeded archives. Live source/input acquisition, hosted Colab and its browser-download interface were not tested; the [execution receipt](evidence/partial-inclusion-r67/notebook-execution.json) states those limits. No model weights, photographs, new inference or training are required.

## What the completed GPU comparisons establish

The following results retain the frozen 654-front r50 evaluation reference. Their r67 reference-only counterparts appear in the guide above.

QFL changes reviewed-reference two-view results from 18 misses/49 extras/4 duplicates to 12/83/17. Five-pixel successes change from 568 to 570; ten-pixel successes from 583 to 582. Original-view AP50 changes from 0.9838649 to 0.9839534. The aggregate training traces show almost complete assignment of incomplete-corner targets; they do not log per-target predicted scores or separate zero and one known corner.

On the same 137 photographs, changing supplied training labels to frozen reviewed labels changes reviewed-reference two-view results from 47 misses/85 extras/6 duplicates to 15/51/13, and five-pixel successes from 541 to 572. Both older annotation references favour the supplied-label arm. One paired seed and reused development photographs do not establish independent generalization or human-label precision.

The returned notebooks execute source `acaf5d4b4043583fdd81f5ea848b3a6a68ada728`, pinned in notebook commit `587514e557db13be287a5483eaf952dfd5e14761`. This incorporates PR156's clipped-box accounting: finite non-positive-area predictions stay in extra-output totals and cannot match. The same repair is retained in this branch. Existing thresholds, training budgets and final-epoch selection are unchanged.

## Choose the next change from evidence

The [duplicate and unmatched audit](duplicate-audit-r66.md) compares unchanged final outputs, an existing reflected-partner requirement, and separate additional hard-suppression cutoffs of 0.60 and 0.50. It preserves surviving corner coordinates and reruns the original evaluator. The reviewed-label candidate loses four partial matches under either the reflected-support rule or the 0.50 rule, despite unchanged complete-pair scores. The maximum-cardinality diagnostic finds no reviewed-reference match gain available merely by replacing the greedy matcher on the five audited baseline sets. The [literature note](research-duplicates-r66.md) distinguishes these restrictions from a future pre-NMS Soft-NMS or learned-quality experiment.

Use the [score-target research](research-score-assignment.md) to interpret the fixed-weight diagnostic. A low supervision target, missing assignment and score below a useful target require different remedies. The completed box-only counterfactual lowers mean one-corner targets while preserving assignment for every front. It describes assignment at fixed predictions; it is not a trained-model result. Do not select a new operating threshold using the same twenty development photographs.

The [focused annotation review](annotation-review-r65.md) follows the existing visible-front definition and retains uncertain ownership judgments. The completed [r67 review](partial-inclusion-r67.md) resolves four historical exclusions through original-image inspection and records the other three as held. The changed-photo PDF and its [validation](evidence/partial-inclusion-r67/pdf-validation.json) cover only the three amended photographs. This focused agent-assisted review does not measure human annotation repeatability. Further amendments should preserve the same explicit versioning, stable identities and changed-photo evidence.

Additional matched seeds are worthwhile for a promising design. A controlled 137-versus-142-photo comparison would separately test the five extra photographs; the existing cross-run results cannot isolate that effect. The [model research note](research-next-models.md) keeps previous suppression, augmentation and corner-refinement attempts visible so they are not presented as untried methods.

## Application boundary

Vision outputs are intended for robotic arms to locate cassettes. Arm control and integration are outside this project. Recognition quality is the priority; latency and hardware capacity are not current decision criteria. The downstream message format and physical acceptance tolerance remain unspecified.
