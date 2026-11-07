# Continue the baomai image-localization experiment

**Current decision, 2026-11-07 (Asia/Hong_Kong): retain the supplied epoch173 model.** Both GPU fine-tunes completed correctly, but neither improves the predeclared combination of detection and corner measurements. The next executable handoff is [PR #15's matched bias-warmup experiment](https://github.com/Siuuuers/dl/pull/15). The [completed comparison](https://github.com/Siuuuers/dl/blob/155f1118f9773d22af944c4b5774fc406d51d943/docs/gpu-comparison-2026-11-07.md) and [PR #14](https://github.com/Siuuuers/dl/pull/14) explain the retained-model decision.

The user's current duty is finding/localizing cassette objects in the supplied baomai D405 images, preserving the existing one-class, two-lower-corner target. Camera settings are unknown. Camera/robot calibration and physical grasp qualification are outside this deliverable and are not prerequisites for image training. The user has GPU access and returned the first completed pair in [PR #13](https://github.com/Siuuuers/dl/pull/13).

## Latest completed work

The original return archive is at `results/gpu/baomai-a100-001/baomai-gpu-results.zip` on commit `f860c8cfa071da252edbf5fd1320bf83cdf8d51d`. Its SHA-256 is `9ec72f6124027889f8d6b8cf41b939aa4c164cc0ab23063af9064ca64098ad19`, and it has 39,776,601 bytes.

Both final-epoch checkpoint bundles pass content and paired-run validation. Each records epochs 1-30, 1,050 batches, 4,110 image presentations, 78 optimizer updates and successful finalization. They used matching settings, initialization and A100/software environment. The only training-label differences are the intended eleven-row deletion in five files. The 78 optimizer updates are consistent with gradient accumulation; they are not evidence of incomplete training.

New inference ran all three models in one pinned CPU environment from clean source `707ca254ac624ae956400216deffbefb52a4bc89`. Each completed 60 image/threshold pairs. The comparison code was then published and verified from clean revision `484f5671456f004e4198173537fd847e0a56db90`; final evidence is in `155f1118f9773d22af944c4b5774fc406d51d943`.

At primary box confidence 0.25, matching IoU 0.50 and endpoint confidence 0.50:

| Measure | M0 original | M1 original-label fine-tune | M2 cleaned-label fine-tune |
| --- | ---: | ---: | ---: |
| Missed objects | 39 | 36 | 37 |
| Extra detections | 63 | 80 | 80 |
| Missing visible endpoints | 43 | 43 | 44 |
| Conditional median error, px | 1.611790 | 2.468005 | 2.473908 |
| Conditional p95 error, px | 5.161683 | 6.326759 | 6.282496 |
| Both corners within illustrative 5 px / 584 | 521 | 479 | 479 |

Neither trained model passes PR #1's five-metric dominance rule against M0. M2 also fails against M1. The secondary 0.15 and 0.40 results do not reverse the decision. The regression persists on the same 1,176 returned reference endpoints: M2 worsens 919, and all 17 nonempty images have a higher common-endpoint median after either fine-tune.

The report preserves complete raw predictions, per-image metrics, source/runtime records, checkpoint evidence and a visually inspected common-image crop. Its 1.12 MiB archive restores and replays successfully. Two focused altered-evidence cases reject incomplete execution and an invented favorable median. The original model, labels, split, candidate overlay, training recipe and evaluator remain unchanged.

## Next bounded assignment

**Goal:** execute the matched bias-warmup experiment and return both complete model bundles.

**Owners:** the user supplies GPU execution; the next evaluation assistant validates the return and runs the common M0/control/warmup comparison.

**Implementation is complete:** [PR #15](https://github.com/Siuuuers/dl/pull/15) adds the isolated recipe, schema-2 recipe provenance, full resolved-settings checks, actual GPU/backend observations, matched return packaging and a pinned Colab notebook. The verified training-code revision is `679b87b38e7b0bca5fbd02fb1519a90fb2379541`; notebook, instructions and raw evidence are published at `b9a12a02c970bf8c0f43addb5648e0d54e26ab6b`.

**One GPU handoff:** [open the Colab notebook](https://colab.research.google.com/github/Siuuuers/dl/blob/b9a12a02c970bf8c0f43addb5648e0d54e26ab6b/notebooks/baomai_bias_warmup.ipynb), select a GPU runtime, and run its cells in order. [Full instructions](https://github.com/Siuuuers/dl/blob/b9a12a02c970bf8c0f43addb5648e0d54e26ab6b/docs/gpu-warmup-handoff.md) also give shell commands. Return `baomai-warmup-results.zip` as an attachment or a new result PR.

| Role | Recipe | Labels | Bias warmup start |
| --- | --- | --- | ---: |
| Fresh control | `paired-finetune-v1` | Original reference | 0.1 |
| Warmup experiment | `warmup-bias-lr-1e-4-v1` | Original reference | 0.0001 |

Both runs freshly initialize from epoch173. Keep seed 42, 30 epochs, batch 4, image size 1280, FP32, AdamW, nominal LR, augmentation, loss/RLE gains, accumulation and final-epoch selection. Run both in the same unchanged source/software/GPU session. The new packer requires a fresh schema-2 matched control and does not silently reuse historical M1. The candidate-label arm is excluded from this warmup experiment.

**Verified readiness:** [the implementation report](https://github.com/Siuuuers/dl/blob/b9a12a02c970bf8c0f43addb5648e0d54e26ab6b/docs/warmup-execution-verification-2026-11-07.md) records 96 passing focused checks and two actual clean-source CPU trials. Each completes four batches, sixteen image presentations and four parameter-changing updates while retaining the 30-epoch schedule. Initial bias rates are 0.1/0.0001; initial batches and first forward loss match. Both real trial receipts are correctly refused as completed models. Both PR #13 checkpoint identities remain exact. Notebook cells compile; two local subprocess checks validate interruption cleanup. CUDA and complete 30-epoch execution remain pending.

**Evidence and hypothesis:** the original explicit-AdamW recipe starts bias learning rate at 1,000 times nominal `lr0`, as the pinned trainer and returned logs show. The new experiment tests whether that shared setting contributes to the localization regression. It does not establish causality or an accuracy improvement before the complete comparison.

**Acceptance after return:**

- Verify the unchanged return ZIP, both role/recipe identities, complete epochs and finalized `last.pt` artifacts through the new packer/checkpoint gate.
- Require identical source, runtime/device/backend state and reference data, allowing only bias warmup and execution-path differences.
- Run fresh M0/control/warmup inference together in one evaluator environment; preserve the frozen split, scorer and thresholds.
- Compare the warmup model against both M0 and its fresh control using the same five unrounded primary measures and complete corner/availability diagnostics.
- Keep epoch173 unless the predeclared rule supports replacement. Lower early training loss or the trainer's automatic `best.pt` score is not the acceptance result.
- Preserve all raw predictions, receipts, settings, real update counts, completed epochs, finalized hashes, CSVs and logs. Never rewrite old results.

No full new GPU training or model-quality result exists yet. Do not change labels, augmentation, losses, thresholds or coordinates alongside this test. A later deduplicated-label run can follow if the shared training recipe proves useful.

## Reuse the completed work

| Work | Current source |
| --- | --- |
| Original experiment and conservative promotion rule | [PR #1](https://github.com/Siuuuers/dl/pull/1), `docs/refine-cassette-pose.md` |
| Baseline evaluator and first real inference | [PR #7](https://github.com/Siuuuers/dl/pull/7) |
| Validation review and direct-pixel quality diagnostics | [PR #8](https://github.com/Siuuuers/dl/pull/8) |
| Thirteen-image training pilot and eleven-row candidate | [PR #9](https://github.com/Siuuuers/dl/pull/9) |
| Adopted annotation policy and verified paired-data staging | [PR #10](https://github.com/Siuuuers/dl/pull/10) |
| Training execution and completed-checkpoint provenance | [PR #11](https://github.com/Siuuuers/dl/pull/11) |
| Portable input/return packaging | [PR #12](https://github.com/Siuuuers/dl/pull/12) |
| Actual returned GPU models and logs | [PR #13](https://github.com/Siuuuers/dl/pull/13) |
| Matched M0/M1/M2 result and replayable evidence | [PR #14](https://github.com/Siuuuers/dl/pull/14) |
| Isolated bias-warmup recipes, verified trials and pinned GPU notebook | [PR #15](https://github.com/Siuuuers/dl/pull/15) |

PR #15 is stacked on PR #14; PR #14 is stacked on PR #12, which depends on #11, #10, #9, #8 and #7. PR #13 contains the result archive on a separate branch based on main. Inspect current heads before continuing; do not assume this stack has merged or reimplement missing code from main.

The baseline hash is `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`. Its reference has 137 training images, 20 validation images, 3,772/664 objects and 13/3 empty scenes. Preserve `cassette_front`, `kpt_shape: [2, 3]`, `flip_idx: [1, 0]` and image-X corner order. Use raw 1280×720 PNGs; the annotated previews are review aids. Unlocatable endpoints stay exactly `0 0 0`; do not invent them from predictions.

The local `baomai-visible-front-v1` annotation policy is adopted. It retains identifiable visible fronts even when both target corners are unlocatable, and draws boxes around observed front-face extent. Frozen legacy labels have not been completely migrated or certified. Any new box or corner corrections must be separately versioned; the current candidate remains deletion-only, with all 6,839 visible training endpoints retained. PR #2's historical 3,765-object candidate remains unrecovered and is distinct from this verified 3,761-object proposal.

## Reproduce the comparison

From PR #14's branch, use the commands in [the report](https://github.com/Siuuuers/dl/blob/155f1118f9773d22af944c4b5774fc406d51d943/docs/gpu-comparison-2026-11-07.md). The evidence archive is `results/evaluation/baomai-a100-001/evaluation.zip`. The standard-library `scripts/summarize_pose_comparison.py` consumes its M0/M1/M2 and quality folders plus small reference/candidate checkpoint bundles. It rechecks all saved scores from predictions before deriving the decision. It needs original validation PNGs/labels but no weights or machine-learning runtime.

For new inference, use `scripts/evaluate_pose.py` and explicit `--checkpoint-manifest` for trained models; omitting it selects the frozen baseline. Keep image size 1280, rectangular preprocessing, FP32, ordinary NMS at IoU 0.70, batch 1, `max_det=300`, box thresholds 0.15/0.25/0.40, matching IoU 0.50 and endpoint confidence 0.50. Keep the same device/runtime across models. Never substitute the trainer's automatic `best.pt` console score for the predeclared finalized epoch-30 `last.pt`.

The baseline replacement rule requires all five unrounded measures to be no worse and at least one better: FN, total FP, missing visible endpoints, conditional median and conditional p95 error. Missing statistics do not pass. Radius success retains missed objects/points in its denominator. Reference-unlocatable output counts require their matched-object denominator and do not measure hidden-point coordinates.

## Independent work and limits

The existing browser label editor and historical label recovery may continue independently. Browser Save/reopen acceptance requires a fresh check of current editor branches in an accessible browser; older access trouble did not establish an HTML defect. Use disposable annotation copies. No editor rewrite or new hardware questionnaire is required for this training follow-up.

The 20 validation images already participated in model selection and development. One seeded comparison does not establish repeatability or independent generalization. The five excluded final-test images and nineteen gap images mentioned in the source data README are absent from this package; genuinely untouched test scenes are still needed for a final independent assessment. Preserve the original model while investigating the measured regression.
