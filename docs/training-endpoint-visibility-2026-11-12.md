# One matched comparison for the reviewed endpoint corrections

Date: 2026-11-12, Asia/Hong_Kong  
Status: archived handoff; PR52 full training pair and evaluation complete, retain PR35

**This experiment is complete.** Read the [results and decision](endpoint-results-2026-11-12.md). The instructions below preserve how the fixed pair was produced; no repeat GPU run is requested.

**Keep PR35 selected at 1280 / box confidence 0.13 / NMS 0.70 / endpoint cutoff 0.50 with whole-image review.** This handoff tests the revised endpoint supervision under one fixed matched experiment. It does not replace PR35 automatically.

## The user-requested corner recheck

The original-image recheck accepts the upper of the three questioned right corners in `new_000023`, **original row 33 at `(612,549)`**. A small rounded return is visible a few pixels left of the gripper contour. Rows 34 and 35 remain missing behind the finger. This is a moderate-confidence manual decision: the two original-only reviewers disagreed on the upper junction, and the final record retains that disagreement. See the [pixel comparison and reasoning](../results/review/baomai-endpoint-recheck-20261112/review.md).

Only that one right triple changes relative to PR50. The cumulative intervention against PR35 supervision now contains ten endpoint slots on six rows: three coordinate corrections, three visible-to-missing changes, and four missing-to-visible changes. Two missing right slots remain unchanged. No box, object-list or split change is added. The separate seventeen-box/duplicate proposal is not combined with this study.

## Archived run instructions

1. Open [baomai_endpoint_visibility.ipynb](../notebooks/baomai_endpoint_visibility.ipynb) in [Google Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/endpoint-study-20261112/notebooks/baomai_endpoint_visibility.ipynb).
2. Select **Runtime → Change runtime type → GPU**, then **Runtime → Run all**. Keep the same GPU/runtime for the pair. Approve the notebook's Drive mount to preserve completed work.
3. The notebook downloads and verifies the original inputs, performs a four-batch resource trial when needed, and completes both 30-epoch runs. It uses fresh epoch173 initialization for each full run.
4. Return **`baomai-endpoint-visibility-seed46-results.zip`**. It contains both selected checkpoints and their result/provenance artifacts. Keep the verified Drive backup until evaluation is complete.

The dedicated backup folder is `MyDrive/baomai-endpoint-visibility-seed46-20261112`. The notebook restores and verifies compatible completed arms before starting missing work. If both full arms are already complete, it skips the trial and training. It can reuse a compatible completed trial. Partial runs are preserved and rejected as complete results; use a fresh `RUN_BASE` and restore verified completed artifacts. A changed runtime, source, candidate, recipe or protocol cannot silently join an earlier completed arm.

The final notebook pins implementation commit [`ce207c7`](https://github.com/Siuuuers/dl/commit/ce207c7a4b929fcf2ff02fb3f0db9929de08e1ea) and protocol SHA256 `0e257cc036c86ea6859b6ed3b86f10768a87994e8f3b8b82b7e553a1cc1f8f60`. It refuses an unset pin before starting work. Use this final handoff notebook rather than an intentionally unpinned preparation snapshot. Both full-run receipts must identify the same clean implementation checkout.

## What is compared

| Role | Training supervision | Recipe ID | Visible / missing endpoints |
| --- | --- | --- | ---: |
| `old_reviewed` | Exact eight-file annotation overlay selected for PR35 | `reviewed-annotations-seed46-v1` | 6,841 / 701 |
| `endpoint_reviewed` | The same parent plus the six reviewed endpoint rows | `reviewed-endpoints-seed46-v1` | 6,842 / 700 |

Both variants have **137 training images, 3,771 objects and 13 empty images**. Original in-training validation remains **20 images, 664 objects, 1,227 visible endpoints, 101 missing slots and three empty images**. The separately corrected **663-object development reference** is used for post-training scoring; it does not replace the training validation labels.

Both inner runner arms are `candidate`, because both roles use corrected overlays. The stager's original-label `reference` arm is not this experiment's control. Neither full run resumes PR35 weights, the resource trial or the other run.

The recipe IDs differ to preserve their annotation contracts. Their complete numerical settings are identical: **seed 46, 30 epochs, batch 4, input 1280, rectangular batches, FP32, AdamW, learning rate 0.0001 and bias warmup 0.0001**, with the established augmentation and loss settings. The numerical recipe SHA256 is `f18e0a813a1057e1b3ab3a44ea3ce9b2a3e7673cc2bf04ca09b394bfdd8c51b8`. Ultralytics stays at 8.4.182. The selected output is finalized epoch 30 `last.pt`.

The [fixed protocol](../experiments/endpoint-visibility-2026-11-12.json) pins candidate manifest `9c489bf57a526509de73e40cc73b02117398fba4ec97107bf047aa71e4951c41`, its adjudication, the PR35 parent and the original epoch173 initialization. Existing recipes continue to reject endpoint candidates. The new recipe accepts only the endpoint kind and requires `arm=candidate` in both execution and completed-receipt validation.

## The scientific question and decision

The hypothesis is that correcting misplaced or wrong-side visible targets can improve endpoint availability/localization on partially obscured fronts while preserving object completeness. A decrease in returned points caused by additional missed objects or blanket endpoint suppression does not support that hypothesis.

Evaluate both returned models and the exact retained PR35 checkpoint in the same inference environment. Use **confidence 0.13 only**, input 1280, NMS 0.70, matching IoU 0.50 and endpoint cutoff 0.50. This study adds no threshold, seed, epoch, resolution or annotation-subset sweep.

Report all-object misses and scored extras over 663 objects; recall for zero/one/two-locatable-corner groups; 5/10 px pair successes over all 584 eligible objects, including eligible detection misses; miss-free scenes; empty-image outputs; and every changed match/success identity. Inspect original images to distinguish actual object recovery from an IoU crossing or reassignment among existing proposals.

For reference-unlocatable slots, compare outputs on **the same objects matched by both models in each comparison pair**. Report exact numerators, denominators and identities, together with full-reference and common-matched locatable coverage. This prevents extra misses or suppressed useful corners from creating an apparent availability improvement. An unlocatable slot has no reliable hidden coordinate, so this is not a hidden-corner geometric error measurement.

Require a supported endpoint benefit while preserving object completeness, locatable coverage and existing 5/10 px successes against the matched control; review extras and require no added empty-scene outputs. Compare with PR35 separately before considering replacement. Manual annotation uncertainty, including row 33's recorded dissent, remains relevant when interpreting small gains. A single pair on reused development views cannot establish unseen-scene reliability.

## Preparation and return verification

The endpoint verifier now supports separate descriptive ranges for two visible corners on one row and checks that each recorded pixel point agrees with its normalized label. The exact historical PR50 nine-slot candidate still passes through the legacy single-visible-point evidence format.

This work also fixes a compatibility regression in PR50: a schema 1 receipt has no explicit recipe ID, and the newly strict recipe check rejected it. The legacy receipt path now selects the established original recipe explicitly; unknown public recipe IDs and candidate kinds remain rejected. Positive legacy-bundle and endpoint-admission checks cover both cases.

The [preparation receipt](endpoint-preparation-verification-2026-11-12.json) separates original-input/staging checks from synthetic bundle and notebook checks. Preparation checks do not deserialize a checkpoint or execute a GPU trial. PR52's completed full-run artifacts and fresh evaluation are documented in the [result](endpoint-results-2026-11-12.md). The returned archive does not independently establish resource-trial execution.

After a completed pair, the notebook invokes the dedicated packager:

```bash
python -B scripts/package_endpoint_results.py \
  --protocol experiments/endpoint-visibility-2026-11-12.json \
  --old-reviewed /path/to/old_reviewed \
  --endpoint-reviewed /path/to/endpoint_reviewed \
  --output /path/to/baomai-endpoint-visibility-seed46-results.zip
```

The packager reuses completed-checkpoint validation and binds the two truthful candidate-arm bundles to their exact roles, recipes, source, runtime, initialization, effective file inventories and endpoint-count delta. It verifies the ZIP after creation. Add `--verify-existing` to inspect an existing ZIP without rewriting it. The two full bundles do not themselves prove that a resource trial executed, and neither packaging nor successful training establishes improved accuracy.
