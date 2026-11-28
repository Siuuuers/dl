# Frozen score and assignment audit

Status: all four actual notebook cells completed on CPU from published source `7528f2f3980b7abaef0f2644c0a98d5163440f11`, including the three-photograph smoke and the complete 142-photograph audit for both checkpoints. Updated 28 November 2026 (Asia/Hong_Kong). The [execution receipt](evidence/score-assignment-r65/notebook-execution.json) records the source, runtime, unchanged cell hashes and verification results.

## Why this diagnostic comes next

The paired quality-focal return recovered six detections at the common operating point, but added 34 extra predictions, including 13 more duplicates. Its training traces do not show widespread loss of positive assignments for fronts with fewer than two known corners. Those traces record aggregate target quality, but do not separate zero-known from one-known fronts or preserve each anchor's predicted score and quality components. The [verified result analysis](quality-focal-results.md) records the counts and the limits of that evidence.

This notebook inspects the two returned final checkpoints without training. It asks whether frozen predictions on the unchanged training photographs expose differences in candidate eligibility, assignment, target quality, or predicted score. It adds no candidate model to the training queue and selects no confidence threshold. The research rationale and native assignment rules are described in [the score and assignment research note](research-score-assignment.md).

## Fixed inputs and scope

- Models: the complete native-control and quality-focal final checkpoints returned in commit `f971e880e8aff2859ce4d8ae2923efea826eddb4`. Both whole-model hashes are checked before loading. Downloaded split wrappers additionally require their exact wrapper and member hashes.
- Data: the same 142 training photographs and 3,999 fronts used by both arms. The returned evidence archive supplies the original image inventory, image hashes and unchanged training labels; no development photograph is staged.
- Return archive: `baomai-quality-focal-r61-results.zip`, SHA256 `cf7b7b71349b9412bd5272001f5071afcf8d3b356423deb6c233cb1e6c326c4a`.
- Native-control checkpoint: SHA256 `15d7c97d5c58f16e40462807231c7bb01a5ee9a719225cb756493e1b41714d67`.
- Quality-focal checkpoint: SHA256 `8760602df814f47685dde25a2e389a554f623a7280473c90d5e94a0f30fe519f`.
- Runtime: Ultralytics 8.4.182, with hashes of the seven native source/configuration files used by the observer. The receipt records Python, package versions, device, GPU name when applicable, CUDA/cuDNN versions, deterministic settings and TF32 flags. Existing Torch/torchvision installations are retained and disclosed.

Each checkpoint runs in FP32 evaluation mode with gradients disabled. BatchNorm and dropout remain in evaluation mode. The driver uses the native letterbox transform, no random augmentation, batch size one and fixed image order. A 1,280 × 720 photograph becomes a 1,280 × 736 input with its labels transformed by the same operation. The model's ordinary evaluation forward returns both raw native heads; the audit reads those heads before confidence filtering, clipping or nonmaximum suppression. No model mode switch is needed to expose them.

This intentionally differs from augmented historical training. A final checkpoint evaluated on its own training photographs cannot reconstruct the optimization trajectory or provide independent recognition-accuracy evidence.

## What is measured

Each source target retains its image ID, original label-row number, box dimensions and zero-, one- or two-known-corner group. Each checkpoint and each native head has separate records for:

1. Eligible anchor centers under the native candidate rule.
2. Selected candidates before inter-target conflict resolution.
3. Surviving assignments after conflict resolution and the branch's final selection.
4. Predicted class probability `p`, normalized supervision target `q`, clamped box CIoU, known-only corner OKS and the combined assignment overlap.
5. Targets with no eligible center, no surviving assignment, and assignments with `q = 0`. These are distinct conditions.

Every eligible anchor ID is retained, with summaries of its score and component values. Detailed values are retained for the union of pre-conflict and surviving anchors. An anchor that another target wins has `q = 0` in the losing target's record; it never inherits the winner's target score.

For a front with a known corner, the native overlap is the mean of clamped CIoU and known-only OKS. For zero-known fronts it is clamped CIoU alone. The observer reproduces the native area factor, sigmas and visibility mask, and checks its decomposition against native overlap values exactly. The final `q` also depends on task alignment and native normalization; it must not be equated with raw IoU, raw OKS or their mean.

The group table averages over surviving anchors. The recorded `p − q` uses the same assigned anchor for both values. It describes agreement with this model-derived training target; it is not calibration against observed correctness, proof of under-training, or a physical coordinate-error bound. Different checkpoints can select different anchors, so an unpaired group mean is not an intervention effect.

The driver also records an explicit native edge case: a zero-overlap conflict tie can assign an anchor to a target outside that target's eligible pool. Such ownership is retained with `q = 0`, an explicit flag and null unevaluated component values. A synthetic fixture tests this logging behavior; its existence is not evidence that it occurs in these checkpoints.

## Descriptive box-only counterfactual

For each unchanged set of decoded predictions, a separate native box-only `TaskAlignedAssigner` uses the same candidate-center rule, alpha, beta, top-k settings and final selection rule. It removes the corner term from overlap, then allows native assignment and normalization to run normally. The counterfactual does not hold realized assignments or normalized targets fixed: changing overlap can change both.

This comparison can identify sensitivity to the corner term at the frozen predictions. It cannot show whether a model trained with box-only assignment would improve, whether the historical training scores were caused by pose quality, or whether the resulting detections would satisfy the endpoint task. Any later training intervention needs a separately stated hypothesis and a paired control.

## Observation and integrity checks

The three-photograph smoke uses `new_000001`, `new_000002` and the empty scene `new_000131`, providing 99 source fronts per checkpoint and all three known-corner groups. On samples covering each group and an empty scene, a hook captures the native `PoseLoss26.loss` assigner call. The constructed inputs must match exactly, and observing the assignment must leave every native returned tensor unchanged. Both heads and both models are checked.

All parameter and buffer bytes are hashed before and after each checkpoint's pass. The driver additionally requires every module to remain in evaluation mode and every parameter gradient to remain absent. Native fixtures verify zero-overlap conflict reporting and per-target ownership of `q`.

Each completed output contains exactly five payloads and one manifest. Verification replays group summaries from detailed records, checks identities and full/smoke scope, validates each payload hash, and checks frozen-state receipts. An existing completed run is verified and explicitly reported as reuse with its saved runtime; it is not described as new execution. An interrupted or incompatible run is preserved and requires a fresh work folder.

## Run the notebook

Open [the four-cell notebook](../../notebooks/baomai_score_assignment_r65.ipynb) and use **Run all**. A connected GPU is used if available; CPU is supported and no GPU allocation or purchase is performed. The notebook verifies a published, immutable source commit before staging data.

The cells load the fixed source, verify and stage inputs, run the three-photograph smoke followed by all 142 photographs, and package the results. The full pass requires 284 model forwards, inspecting both heads from each forward. The smoke is reported separately and can never satisfy the full-completion flag.

Optional environment variables `BAOMAI_SCORE_ASSIGNMENT_BASE` and `BAOMAI_R61_CACHE` select a fresh working directory and a reusable verified input cache. The notebook builder accepts `--source-commit` after the driver is published; a draft all-zero source pin deliberately fails before computation.

The returned ZIP contains the source/input plan, original return manifest, driver, five result payloads and their manifest, plus the smoke summary when present. It includes neither checkpoint weights nor photographs. The payloads are `summary.json`, `targets.json.gz`, `images.json`, `groups.csv` and `results.md`.

## Access to the detailed records

The [repository storage guide](evidence/score-assignment-r65/repository-storage.md) explains how to restore the exact full target-record gzip from four verified parts. The original notebook outputs, execution receipt and inventory retain their original bytes. After restoration, all original inventory hashes verify directly. This storage arrangement changes no experiment result.

## Completed execution and findings

The published notebook completed all four unchanged code cells in one local IPython session. It performed 284 fresh full-pass forwards and six separate smoke forwards on CPU, without an optimizer. The result contains 31,992 target records and 568 image/head records. All native-input, native-assignment and frozen-state guards passed; the 96-file source closure and all ten return-ZIP members were independently verified. This exercised the local notebook execution and file-link path, not a hosted Colab session or its browser download interface. See the [execution receipt](evidence/score-assignment-r65/notebook-execution.json), [full summary](evidence/score-assignment-r65/results/summary.json) and [complete group table](evidence/score-assignment-r65/results/groups.csv).

The source contains 137 zero-known, 465 one-known and 3,397 two-known fronts. Every one of the 3,999 fronts has an eligible center, a pre-conflict candidate and a surviving assignment under both heads, both checkpoints and both overlap variants. No surviving anchor has `q = 0`, and no owner-outside-eligible event occurs in these data. This frozen, unaugmented pass therefore provides no evidence of missing assignments in any known-corner group; it does not exclude transient problems during historical augmented training.

The native one-to-many head shows a clear group pattern:

| Known corners | Source fronts | Control mean `p` | Control mean `q` | QFL mean `p` | QFL mean `q` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 137 | 0.1673 | 0.2834 | 0.1926 | 0.2890 |
| 1 | 465 | 0.5598 | 0.6262 | 0.5525 | 0.6437 |
| 2 | 3,397 | 0.7909 | 0.8032 | 0.7464 | 0.8065 |

These are means over surviving anchors, whose membership can differ between checkpoints. Lower known-corner counts coincide with lower target quality and lower predicted scores. The zero-known group's predicted scores also fall below its normalized targets on average. The one-to-one head shows the same group ordering: its zero-known mean `p/q` is `0.2242/0.6257` for the control and `0.2469/0.6388` for QFL. Such differences are descriptive and do not establish a historical learning cause or a generalization result.

For one-known fronts under native one-to-many assignment, mean known-only OKS is approximately `0.985` for both checkpoints, while mean clamped CIoU is approximately `0.68`. The known-corner term therefore contributes higher overlap than the box component on these assigned anchors. In the separate box-only counterfactual, mean normalized `q` for this group falls from `0.6262` to `0.4658` for the control and from `0.6437` to `0.4771` for QFL. This comparison includes changed assignments and normalization; it is not a component-only subtraction at fixed anchors.

Box-only overlap changes the surviving one-to-many anchor set for 681 control fronts and 825 QFL fronts, while all fronts remain assigned. It also changes some zero-known assignments through competition with neighboring targets, even though the zero-known target's own overlap already uses the box alone. These findings show assignment sensitivity, but do not demonstrate a benefit from removing the corner term.

## Effect on the next-method decision

Retain the existing recognition candidate. This audit does not support a box-only training switch or a general missing-assignment explanation for incomplete-front errors. QFL's higher zero-known mean score is accompanied by lower means in the one-known and two-known groups; its separately measured increase in extra predictions remains the relevant adverse recognition result. The audit is not grounds to promote it.

If another score-learning intervention is proposed, first inspect the exact partial-front cases and their geometry, score targets and assigned anchors. The present records make that investigation possible without another training run. A later intervention would need a specific paired hypothesis and should explain how it addresses extra predictions as well as misses. No additional GPU experiment is justified solely by these group means.

No result from this notebook alone promotes either returned checkpoint, changes annotations, establishes robotic-arm tolerance, or supplies evidence from a new capture session.
