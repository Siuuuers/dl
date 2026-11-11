# One matched visible-extent training comparison

**Ready for one matched GPU comparison.** The candidate, protocol and implementation source are pinned and the preparation checks pass. The current PR35 seed 46 checkpoint at confidence 0.13 remains the selected reviewed-proposal model while this experiment is pending. No new GPU training or accuracy result is claimed here.

The [handoff notebook](../notebooks/baomai_visible_extent.ipynb) downloads implementation commit [`d43a58b9456a782ace14e20f0bf7ed2da3572ca1`](https://github.com/Siuuuers/dl/commit/d43a58b9456a782ace14e20f0bf7ed2da3572ca1). Its protocol SHA-256 is `42aa275a5d2ed1ccabb786c07b537a8f0d515b27b7804fec2b71e942d6c9022b`. The implementation commit contains the reviewed labels and code; the later notebook handoff supplies the completed source pin. Use the handoff notebook linked here, rather than the earlier draft notebook inside the implementation commit.

The question is specific: does replacing recorded hand/gripper and neighboring-band box supervision with supported observed-front annotations improve the present model's recovery of partly visible fronts? This follows the completed negative PR40 result but uses a different, explicitly frozen supervision delta. It is not a repetition of PR40's five-image omission extension or the earlier 16-edge box study.

## What will run

The [notebook](../notebooks/baomai_visible_extent.ipynb) prepares one matched pair from the [protocol](../experiments/visible-extent-2026-11-11.json).

| Outer role | Labels | Initialization and selection |
| --- | --- | --- |
| `old_reviewed` | Exact eight-file `annotations-2026-11-10` overlay used for PR35 | Fresh epoch 173; finalized epoch 30 `last.pt` |
| `extended_reviewed` | Those same eight files plus the frozen accepted visible-extent delta | Fresh epoch 173; finalized epoch 30 `last.pt` |

Both inner receipts remain truthfully `arm: candidate`. Neither is relabeled as an original-label reference. The new cumulative candidate is `visible-extent-2026-11-11`; the standalone 079 proposal alone is not the training candidate. The PR40 five-image extension is not inherited.

The review scope is 061/062/063/079/082/094. This is a review limit, not a requirement to edit every image. The final protocol separately lists accepted changed images and reviewed images left unchanged. Any uncertain material remains unresolved rather than being assigned a convenient identity or box.

Training retains seed 46, 30 epochs, batch 4, input size 1280, FP32, AdamW, learning rate 0.0001, bias warmup 0.0001 and the existing losses/augmentations. Original 137/20 assignments and original validation labels remain fixed. A four-batch trial is separate from both full initializations. The resource trial cannot become either selected model.

## Colab handoff

Open the final source-pinned notebook, select a CUDA GPU, Run all, approve its Drive mount and return **`baomai-visible-extent-seed46-results.zip`**. The notebook downloads and verifies the exact source, original images, epoch 173 bytes and both effective label variants. It preserves completed bundles and the ZIP under `MyDrive/baomai-visible-extent-seed46-20261111`.

A returned arm contains its checkpoint manifest, weights, training receipt, staging manifest, log and results CSV. Keep the bundle intact. Completed arms are restored, reverified and skipped. A complete compatible resource trial can also be reused. The current source/runtime, restored arms and resource trial must agree before any missing full arm runs. Incomplete local runs are preserved and rejected; after interruption, choose a fresh workspace and restore completed Drive artifacts.

The return packager uses the existing completion and six-artifact validators, verifies the two candidate roles and exact effective-label delta, then reads the ZIP back. It refuses swapped roles, wrong source/runtime, partial trials, altered original images, changed validation labels and undeclared label changes.

```bash
python -B scripts/package_visible_extent_results.py \
  --protocol experiments/visible-extent-2026-11-11.json \
  --old-reviewed /path/to/old_reviewed \
  --extended-reviewed /path/to/extended_reviewed \
  --output /path/to/baomai-visible-extent-seed46-results.zip
```

Add `--verify-existing` only to verify an existing ZIP without rewriting it. The command and both full training runs require the same clean pinned source.

## Fixed evaluation after return

Primary confidence is **0.13**; **0.15** is the only historical diagnostic. Keep 1280, NMS 0.70, endpoint cutoff 0.50, matching IoU 0.50 and the separately versioned 663-object reference. Keep original-reference results separate. Evaluate the current PR35 checkpoint in the same inference runtime as an operational comparator.

Report all missed/extra objects and every gained/lost identity, 0/1/2-locatable-corner groups, 5/10 px corner-pair successes, miss-free scenes, all three empty-image outputs and review workload. Inspect new proposals. Carry forward the inherited reference uncertainty and the separate 011:61 caveat. A gain confined to disputed reference rows is not established physical recovery. No threshold, seed or epoch search follows automatically.

The declared proposal criterion prioritizes supported additional recovery without losing supported existing matches or corner-pair successes, or introducing empty-scene detections. Additional scored extras need explicit review rather than automatic acceptance. Retain PR35 unless the measured result supports a separate documented change. Reused development scenes do not establish new-scene reliability.

## What the recipe can and cannot establish

The [recorded PR40 matched runs](partial-front-results-2026-11-11.md) completed 1,050 batches and 4,110 image presentations but 78 actual optimizer updates, because the pinned trainer accumulates gradients toward nominal batch 64 from batch 4. That is not evidence that training was frozen or that adding epochs would help. Keep the recipe unchanged so this comparison isolates the reviewed supervision. Correct annotation geometry and improved learned weights remain separate claims.

## Preparation checks

```bash
python -B scripts/check_package_visible_extent_results.py
python -B scripts/package_visible_extent_results.py --help
```

The focused checks use synthetic completed bundles for packaging and substituted notebook boundaries for recovery. They do not perform GPU training or establish accuracy. Candidate validation, source/protocol pins and actual-input preflight are complete; the notebook repeats the input checks in its clean downloaded source before training. The [verification record](visible-extent-preparation-verification-2026-11-11.json) preserves the preparation scope.

The six focused checks pass on the frozen candidate, including its real four-image delta and exact eight-file inheritance. The notebook's actual embedded input preflight also passes on all 157 hydrated originals and the 106,838,542-byte epoch173 checkpoint. It reports old-reviewed training totals 3,771 objects / 6,841 visible endpoints / 701 missing endpoints and extended-reviewed totals 3,772 / 6,841 / 703. Both original validation variants remain 664 / 1,227 / 101. This preparation ran in the working checkout before publication; it is not a clean published-source GPU run.
