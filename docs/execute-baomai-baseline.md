# Continue the baomai image-localization experiment

**Current decision, 2026-11-07 (Asia/Hong_Kong): retain the supplied epoch173 model.** Both GPU fine-tunes completed correctly, but neither improves the predeclared combination of detection and corner measurements. Start with [the completed comparison](https://github.com/Siuuuers/dl/blob/155f1118f9773d22af944c4b5774fc406d51d943/docs/gpu-comparison-2026-11-07.md) and [PR #14](https://github.com/Siuuuers/dl/pull/14).

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

**Goal:** isolate the shared training setting that may disturb precise localization.

**Owner:** next implementation assistant; user supplies the GPU execution.

**Scope:** implement an explicitly versioned reference-label experiment that freshly initializes from epoch173 and changes only `warmup_bias_lr: 0.1 -> 0.0001`. Preserve the original recipe and its returned evidence. The current runner/verifier deliberately pins the original recipe, so the existing GPU commands do not yet run this new experiment.

**Evidence and hypothesis:** explicit AdamW in pinned Ultralytics preserves bias warmup at 0.1 even though nominal `lr0=0.0001`. The first bias update therefore starts at 1,000 times the nominal LR. The returned end-of-epoch bias rates are 0.0676514, 0.0343493 and 0.00104489 before settling onto the shared schedule. The logged values match the implementation; both GPU runs followed our plan. This is a concrete hypothesis, not a demonstrated cause of the final regression.

**Depends on:** [PR #14's measured report](https://github.com/Siuuuers/dl/blob/155f1118f9773d22af944c4b5774fc406d51d943/docs/gpu-comparison-2026-11-07.md), the existing runner/checkpoint support from PR #11 and the GPU handoff from PR #12.

**Acceptance checks:**

- Existing returned bundles retain their validity and identity; a revised recipe cannot be mistaken for the original.
- Only the declared bias-warmup setting changes. Keep original reference labels, seed 42, 30 epochs, batch 4, image size 1280, FP32, AdamW, nominal LR, augmentation, loss gains, accumulation and final-epoch selection.
- A bounded trial retains the 30-epoch schedule; stopping after a few batches is different from changing `epochs` to a small number.
- Use the current M1 as control only if the new run keeps its recorded training environment/device. Otherwise run the original and revised reference recipes together in the new environment.
- Compare the revised run against M0 and its matched M1 control using the same primary metrics and complete corner/availability diagnostics.
- Preserve actual settings, optimizer updates, completed epochs, finalized checkpoint hash, raw log and results. Do not rewrite old receipts.
- A positive bias start of 0.0001 works with the existing positive-LR update witness. Zero-start warmup would require separately handling the first zero-rate optimizer call and is additional scope.

Do not infer a causal fix from the observed upward residual, add an empirical coordinate offset, choose a different threshold for each model or change augmentation/loss/labels simultaneously. The proposed warmup experiment is not yet implemented or executed. A later deduplicated-label run can follow if the shared training recipe proves useful.

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

PR #14 is stacked on PR #12, which depends on #11, #10, #9, #8 and #7. PR #13 contains the result archive on a separate branch based on main. Inspect current heads before continuing; do not assume this stack has merged or reimplement missing code from main.

The baseline hash is `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`. Its reference has 137 training images, 20 validation images, 3,772/664 objects and 13/3 empty scenes. Preserve `cassette_front`, `kpt_shape: [2, 3]`, `flip_idx: [1, 0]` and image-X corner order. Use raw 1280×720 PNGs; the annotated previews are review aids. Unlocatable endpoints stay exactly `0 0 0`; do not invent them from predictions.

The local `baomai-visible-front-v1` annotation policy is adopted. It retains identifiable visible fronts even when both target corners are unlocatable, and draws boxes around observed front-face extent. Frozen legacy labels have not been completely migrated or certified. Any new box or corner corrections must be separately versioned; the current candidate remains deletion-only, with all 6,839 visible training endpoints retained. PR #2's historical 3,765-object candidate remains unrecovered and is distinct from this verified 3,761-object proposal.

## Reproduce the comparison

From PR #14's branch, use the commands in [the report](https://github.com/Siuuuers/dl/blob/155f1118f9773d22af944c4b5774fc406d51d943/docs/gpu-comparison-2026-11-07.md). The evidence archive is `results/evaluation/baomai-a100-001/evaluation.zip`. The standard-library `scripts/summarize_pose_comparison.py` consumes its M0/M1/M2 and quality folders plus small reference/candidate checkpoint bundles. It rechecks all saved scores from predictions before deriving the decision. It needs original validation PNGs/labels but no weights or machine-learning runtime.

For new inference, use `scripts/evaluate_pose.py` and explicit `--checkpoint-manifest` for trained models; omitting it selects the frozen baseline. Keep image size 1280, rectangular preprocessing, FP32, ordinary NMS at IoU 0.70, batch 1, `max_det=300`, box thresholds 0.15/0.25/0.40, matching IoU 0.50 and endpoint confidence 0.50. Keep the same device/runtime across models. Never substitute the trainer's automatic `best.pt` console score for the predeclared finalized epoch-30 `last.pt`.

The baseline replacement rule requires all five unrounded measures to be no worse and at least one better: FN, total FP, missing visible endpoints, conditional median and conditional p95 error. Missing statistics do not pass. Radius success retains missed objects/points in its denominator. Reference-unlocatable output counts require their matched-object denominator and do not measure hidden-point coordinates.

## Independent work and limits

The existing browser label editor and historical label recovery may continue independently. Browser Save/reopen acceptance requires a fresh check of current editor branches in an accessible browser; older access trouble did not establish an HTML defect. Use disposable annotation copies. No editor rewrite or new hardware questionnaire is required for this training follow-up.

The 20 validation images already participated in model selection and development. One seeded comparison does not establish repeatability or independent generalization. The five excluded final-test images and nineteen gap images mentioned in the source data README are absent from this package; genuinely untouched test scenes are still needed for a final independent assessment. Preserve the original model while investigating the measured regression.
