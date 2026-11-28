# Which notebook to run

28 November 2026, Asia/Hong_Kong.

Both paired GPU experiments have returned. Their saved outputs and training receipts have been independently verified, so there is no need to rerun them merely to obtain the reported comparisons. The score-target, duplicate-audit and partial-front inclusion analyses have also completed. The latest R69 notebook completes fresh current-checkpoint inference and a separate fixed corner-conditioned suppression comparison. Both analyses are verified, and all four notebook cells have run unchanged.

| Role | Notebook | Question | Runtime and status |
| --- | --- | --- | --- |
| Completed current-model trace and corner comparison | [Current proposals and corner suppression](../../notebooks/baomai_current_proposals_r69.ipynb) | Where does the reviewed-label model lose eligible boxes, and can corner agreement remove extras while keeping matches? | All four actual CPU cells completed from source 41d0e6937c695377373cc55071ffb780345870ee. [Trace evidence](evidence/current-proposals-r69/README.md) and [corner results](evidence/corner-suppression-r69/README.md). |
| Completed diagnosis and fixed output comparison | [Partial-front detection](../../notebooks/baomai_partial_detection_r68.ipynb) | Where do the three restored misses lose eligible boxes, and what does adding unused reflected outputs recover or add? | CPU analysis and independent verification complete. [Research and interpretation](research-partial-detection-r68.md), [full results](evidence/partial-detection-r68/results.md). |
| Completed reference replay | [Partial-front inclusion](../../notebooks/baomai_partial_inclusion_r67.ipynb) | How do four restored fronts change the scores of six unchanged methods and the interpretation of existing fixed filters? | All four actual CPU cells completed locally from pinned source and preseeded archives; [guide and evidence](partial-inclusion-r67.md). |
| Completed diagnostic | [Duplicate and unmatched audit](../../notebooks/baomai_duplicate_audit_r66.ipynb) | What do reflected support and extra suppression remove, and does complete-pair scoring hide partial-front losses? | CPU; saved predictions only. [Results and execution evidence](duplicate-audit-r66.md). |
| Completed diagnostic | [Score and assignment audit](../../notebooks/baomai_score_assignment_r65.ipynb) | For each zero-, one- or two-corner target, how do native assignments, quality targets and predicted scores relate? | All four actual cells completed on CPU; 284 full-pass forwards, no optimizer. See [execution evidence](evidence/score-assignment-r65/README.md). |
| Completed comparison | [Classification-loss comparison](../../notebooks/baomai_quality_focal_r61.ipynb) | Does QFL improve misses, extras and corner results relative to a matched native control? | PR157 returned; [independent score/AP replay](quality-focal-results.md) complete. |
| Completed comparison | [Controlled training-label comparison](../../notebooks/baomai_training_label_comparison_r61.ipynb) | What changes when only the training annotations change on the same 137 photographs? | PR158 returned; [independent all-reference replay](training-label-results.md) complete. |
| Completed diagnostic | [Proposal-stage and box-geometry audit](../../notebooks/baomai_proposal_audit_r63.ipynb) | Where do the historical candidate's misses lose eligible proposals? | CPU; all four unchanged cells completed from published source; [evidence](evidence/proposal-stages/README.md). |
| Completed diagnostic | [Training-augmentation audit](../../notebooks/baomai_augmentation_audit_r63.ipynb) | How do the existing transforms affect target retention and corner availability? | CPU; all four unchanged cells completed from published source; [evidence](evidence/augmentation-audit/README.md). |

## Latest completed experiment

The [R69 notebook](../../notebooks/baomai_current_proposals_r69.ipynb) uses one frozen reviewed-label checkpoint and the same 20 photographs. It captures all decoded FP32 proposals in both views, verifies their native reconstruction, and records the first stage that removes each front's eligible support. A separate companion compares one fixed corner condition on five saved methods and four references.

Under r67, the current original view has 12 confidence-gate misses, five without an eligible decoded box and one native-suppression loss. Its candidates above the fixed 0.25 cutoff support at most 641 matches, compared with 640 after suppression. This gives a one-match upper bound for changing only suppression within that pool.

On the archived reviewed-label outputs, requiring two accepted corresponding corners within five pixels before additional suppression at IoU 0.60 changes extras from 50 to 44 and duplicates from 13 to 11. Misses stay at 18, complete-pair scores stay at 572/583 and 583/583, and every target's matched output and known-corner outcome is preserved under all four references. The supplied-label checkpoint loses one partial front under the same rule.

The fresh CPU outputs preserve all target detection and corner-threshold outcomes relative to the archived L4 outputs. One separate extra changes category, giving 14 baseline duplicates and 12 after the corner rule. Both runtimes retain the same 50-to-44 extra reduction. The full numerical differences and selected-candidate witnesses remain in the evidence.

All four unchanged cells completed in local IPython from a clean published checkout with preseeded, hash-verified inputs. The complete result ZIP preserves raw tensors, diagnostic ledgers and the companion records. For optional reproduction, choose a CPU runtime and run all four cells. No GPU training is required. Live network acquisition and the hosted Colab download interface have separate execution status in the [receipt](evidence/current-proposals-r69/notebook-execution.json).

The [research note](research-detection-decisions-r69.md) explains how RMPE motivates this small rule, why score ranking alone does not define an absolute confidence threshold, and how Repulsion Loss and learned suppression address different mechanisms. The next training question is the representation and scoring of partial fronts under the existing annotation definition.

## Completed partial-front diagnosis and union

The [r68 research note](research-partial-detection-r68.md) explains the measured stages, prior Soft-NMS and larger-input results, and conditional NWD or sliced-inference options. Its fixed union adds every reflected record without a mutual original-view box partner, then applies the existing score-ranked suppression above IoU 0.70. Reference annotations do not select additions. The [40-row result matrix](evidence/partial-detection-r68/matrix.csv) covers five methods and four reference versions.

For the reviewed-label model under r67, misses fall from 18 to 10 while extras rise from 50 to 60 and duplicates from 13 to 18. Complete-pair counts stay at 572/583 within five pixels and 583/583 within ten. Six one-known and two zero-known partial fronts are recovered. Of the six known corners, four are within five pixels and five within ten, with one error of 11.3027 pixels. The three restored focus misses remain. Either older reference gives only two additional matches and sixteen more extras. The union is not promoted.

For optional reproduction, run the four CPU cells in [baomai_partial_detection_r68.ipynb](../../notebooks/baomai_partial_detection_r68.ipynb). Source is pinned to `0e1f545a2cf33022b86bf316866b08cacfa55477`. The notebook uses immutable saved outputs and nominal training labels. It requires no model weights, original photographs, new inference or training. The bounded geometry and training-size observations do not by themselves justify a new GPU loss or sampling experiment.

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

The [duplicate and unmatched audit](duplicate-audit-r66.md) compares unchanged final outputs, an existing reflected-partner requirement, and separate additional hard-suppression cutoffs of 0.60 and 0.50. It preserves surviving corner coordinates and reruns the original evaluator. The reviewed-label candidate loses four partial matches under either the reflected-support rule or the 0.50 rule, despite unchanged complete-pair scores. The maximum-cardinality diagnostic finds no reviewed-reference match gain available merely by replacing the greedy matcher on the five audited baseline sets. The [literature addendum](research-duplicates-r66.md#retrospective-addendum-28-november-2026) records the already completed pre-NMS linear Soft-NMS comparison, which recovered no missed fronts. A learned-quality intervention remains conditional.

Use the [score-target research](research-score-assignment.md) to interpret the fixed-weight diagnostic. A low supervision target, missing assignment and score below a useful target require different remedies. The completed box-only counterfactual lowers mean one-corner targets while preserving assignment for every front. It describes assignment at fixed predictions; it is not a trained-model result. Do not select a new operating threshold using the same twenty development photographs.

The [focused annotation review](annotation-review-r65.md) follows the existing visible-front definition and retains uncertain ownership judgments. The completed [r67 review](partial-inclusion-r67.md) resolves four historical exclusions through original-image inspection and records the other three as held. The changed-photo PDF and its [validation](evidence/partial-inclusion-r67/pdf-validation.json) cover only the three amended photographs. This focused agent-assisted review does not measure human annotation repeatability. Further amendments should preserve the same explicit versioning, stable identities and changed-photo evidence.

Additional matched seeds are worthwhile for a promising design. A controlled 137-versus-142-photo comparison would separately test the five extra photographs; the existing cross-run results cannot isolate that effect. The [model research note](research-next-models.md) keeps previous suppression, augmentation and corner-refinement attempts visible so they are not presented as untried methods.

## Application boundary

Vision outputs are intended for robotic arms to locate cassettes. Arm control and integration are outside this project. Recognition quality is the priority; latency and hardware capacity are not current decision criteria. The downstream message format and physical acceptance tolerance remain unspecified.
