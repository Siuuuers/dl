# Which notebook to run

28 November 2026, Asia/Hong_Kong.

The next GPU comparison remains the classification-loss experiment, followed by the controlled training-label experiment. Two CPU audits explain the remaining errors and the existing augmentation path. They do not change either frozen training comparison.

| Priority | Notebook | Question | Runtime and status |
| --- | --- | --- | --- |
| 1 | [Model-improvement comparison](../../notebooks/baomai_quality_focal_r61.ipynb) | Does changing classification-loss emphasis improve misses, extras and corner results? | GPU; first two setup cells verified, comparative training result pending |
| 2 | [Controlled training-label comparison](../../notebooks/baomai_training_label_comparison_r61.ipynb) | What changes when only the training annotations change on the same 137 photographs? | GPU; first two setup cells verified, comparative training result pending |
| Diagnostic | [Proposal-stage and box-geometry audit](../../notebooks/baomai_proposal_audit_r63.ipynb) | Where do the current misses lose eligible proposals, and what box disagreements remain? | CPU; all four actual cells completed from published source; [evidence](evidence/proposal-stages/README.md) |
| Diagnostic | [Training-augmentation audit](../../notebooks/baomai_augmentation_audit_r63.ipynb) | How do the existing transforms affect target retention and corner availability? | CPU; all four actual cells completed from published source; [evidence](evidence/augmentation-audit/README.md) |

Each notebook has four code cells. The GPU notebooks retain their original source pins and paired training budgets. A run already in progress need not be restarted for these new CPU diagnostics.

Both CPU notebooks have already completed locally in IPython. Rerunning them is optional; hosted Colab and its browser download interface were not exercised. The augmentation diagnostic retains the current policy: 117 of 11,997 target presentations are removed, mostly because the transformed boxes lie entirely outside the canvas. The six reviewed panels reveal no concrete supervision inconsistency in that bounded sample. These findings do not measure a trained model's accuracy.

## Return the GPU results

For each GPU notebook, select one GPU runtime and run all four cells. The final cell evaluates, verifies and downloads one upload ZIP. Extract its outer ZIP and return the compact evidence ZIP, the return manifest and every wrapped model part together. The two training notebooks share the default verified asset cache within the same runtime. Their separate work directories prevent mixing results.

The model comparison downloads `baomai-quality-focal-r61-upload.zip`; the label comparison downloads `baomai-training-label-comparison-r61-upload.zip`. Follow their [model protocol](quality-focal-experiment.md) and [label protocol](training-label-comparison.md) for persistence and compatible-run reuse.

## How a result changes the plan

Assess the fixed-threshold detection and corner results together. For the classification-loss experiment also inspect the declared AP50 ranking measure; a shifted confidence distribution alone does not establish improved recognition. Report the incomplete-corner group explicitly and retain all three annotation-reference comparisons.

If a model has a useful tradeoff, repeat the matched comparison with additional declared seeds before a strong repeatability claim. If augmentation loses a useful target or leaves an inconsistent target, first verify the original and transformed pixels; then compare one precisely defined transform change. A measured target drop is not automatically a bug.

The [research note](research-next-models.md) records why further suppression sweeps, generic occlusion training, another corner-only refiner, or a new detector family are not the current priority. Prior attempts remain visible so that the same idea is not presented as an untried method.

## Application boundary

The user clarified that the vision outputs are intended for robotic arms to locate the cassettes. Arm control and its integration are outside this project. The vision study reports detection and corner-coordinate quality; the downstream message format and physical acceptance tolerance remain unspecified.
