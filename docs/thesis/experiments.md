# Which notebook to run

28 November 2026, Asia/Hong_Kong.

Both paired GPU experiments have returned. Their saved outputs and training receipts have been independently verified, so there is no need to rerun them merely to obtain the reported comparisons. The next useful notebook inspects the existing score-target mechanism without training a new model.

| Role | Notebook | Question | Runtime and status |
| --- | --- | --- | --- |
| Next diagnostic | [Score and assignment audit](../../notebooks/baomai_score_assignment_r65.ipynb) | For each zero-, one- or two-corner target, how do native assignments, quality targets and predicted scores relate? | CPU or an already available GPU; no optimizer. See [exact validation and execution status](score-assignment-audit.md). |
| Completed comparison | [Classification-loss comparison](../../notebooks/baomai_quality_focal_r61.ipynb) | Does QFL improve misses, extras and corner results relative to a matched native control? | PR157 returned; [independent score/AP replay](quality-focal-results.md) complete. |
| Completed comparison | [Controlled training-label comparison](../../notebooks/baomai_training_label_comparison_r61.ipynb) | What changes when only the training annotations change on the same 137 photographs? | PR158 returned; [independent all-reference replay](training-label-results.md) complete. |
| Completed diagnostic | [Proposal-stage and box-geometry audit](../../notebooks/baomai_proposal_audit_r63.ipynb) | Where do the historical candidate's misses lose eligible proposals? | CPU; all four unchanged cells completed from published source; [evidence](evidence/proposal-stages/README.md). |
| Completed diagnostic | [Training-augmentation audit](../../notebooks/baomai_augmentation_audit_r63.ipynb) | How do the existing transforms affect target retention and corner availability? | CPU; all four unchanged cells completed from published source; [evidence](evidence/augmentation-audit/README.md). |

## What the completed comparisons establish

QFL changes reviewed-reference two-view results from 18 misses/49 extras/4 duplicates to 12/83/17. Five-pixel successes change from 568 to 570; ten-pixel successes from 583 to 582. Original-view AP50 changes from 0.9838649 to 0.9839534. The aggregate training traces show almost complete assignment of incomplete-corner targets; they do not log per-target predicted scores or separate zero and one known corner.

On the same 137 photographs, changing supplied training labels to frozen reviewed labels changes reviewed-reference two-view results from 47 misses/85 extras/6 duplicates to 15/51/13, and five-pixel successes from 541 to 572. Both older annotation references favour the supplied-label arm. One paired seed and reused development photographs do not establish independent generalization or human-label precision.

The returned notebooks execute source `acaf5d4b4043583fdd81f5ea848b3a6a68ada728`, pinned in notebook commit `587514e557db13be287a5483eaf952dfd5e14761`. This incorporates PR156's clipped-box accounting: finite non-positive-area predictions stay in extra-output totals and cannot match. The same repair is retained in this branch. Existing thresholds, training budgets and final-epoch selection are unchanged.

## Choose the next change from evidence

Use the [score-target research](research-score-assignment.md) to interpret the fixed-weight diagnostic. A low supervision target, missing assignment and score below a useful target require different remedies. A box-only overlap counterfactual, if produced, describes assignment at fixed predictions; it is not a trained-model result. Do not select a new operating threshold using the same twenty development photographs.

The [focused annotation review](annotation-review-r65.md) follows the existing visible-front definition and retains uncertain ownership judgments. Predictions may prioritize a region; the original pixels decide whether an edit is warranted. An independent second pass is required to measure annotation repeatability. If labels are amended later, freeze a new reference, retain the older matrices and produce a PDF of only changed photographs with exact changes highlighted.

Additional matched seeds are worthwhile for a promising design. A controlled 137-versus-142-photo comparison would separately test the five extra photographs; the existing cross-run results cannot isolate that effect. The [model research note](research-next-models.md) keeps previous suppression, augmentation and corner-refinement attempts visible so they are not presented as untried methods.

## Application boundary

Vision outputs are intended for robotic arms to locate cassettes. Arm control and integration are outside this project. Recognition quality is the priority; latency and hardware capacity are not current decision criteria. The downstream message format and physical acceptance tolerance remain unspecified.
