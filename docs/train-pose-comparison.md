# Execute the controlled cassette training comparison

**Latest model comparison, 2026-11-07:** the [warmup comparison](warmup-comparison-2026-11-07.md) evaluates the audited models returned in [PR #16](https://github.com/Siuuuers/dl/pull/16). Lower bias warmup improves all five primary measures against M0, but its 37 missed objects exceed the fresh control's 36, so the required comparison against both controls does not pass. Retain epoch173 as the working model and preserve the warmup model as a promising candidate.

The [follow-up review](warmup-detection-review-2026-11-07.md) completes all nine lost and eight gained cases and records 320 recovered additional frames and labels with zero raw-hash overlap against the frozen 157 images. A three-image pilot without model predictions found that `frame_0160` rows 41–43 extend into the metal rack/base, violating the visible-front box policy. This unscored pool is not ready as supplied and is not an independent test. The next task is a separately versioned annotation adaptation, preserving source labels and the existing lower-front corner target. Record capture/session and prior-use limits and freeze reviewed images and adapted label bytes in a separate supplementary reference before scoring the fixed checkpoints.

This guide preserves the original reference/candidate recipe and historical CPU trial. The [first GPU comparison](gpu-comparison-2026-11-07.md) and [warmup reproduction handoff](gpu-warmup-handoff.md) retain their respective experiment records and commands. The current notebook fixes its reported setup syntax error; that repair does not require rerunning the returned models.

The user clarified on 2026-11-07 (Hong Kong time) that the current duty is object localization in the supplied baomai D405 images. Retain the existing two-corner target. The GPU handoffs preserve input and return procedures for reproducing the completed runs. Camera/robot calibration is outside this experiment.

Use the adopted [annotation policy](annotation-policy.md), the [dataset preparation](prepare-pose-training.md), and [PR #1's experiment plan](https://github.com/Siuuuers/dl/pull/1). Keep the supplied epoch173 checkpoint as M0. Train M1 with unchanged labels and M2 with the eleven-row deletion candidate. Each arm freshly initializes from the exact epoch173 bytes; it does not resume the original optimizer or an earlier trial.

## First run a bounded resource trial

Use the pinned runtime, including Ultralytics 8.4.182 and a compatible PyTorch installation. The existing working environment has CPU-only PyTorch. Inference timing does not establish training memory or duration.

From the repository containing this script:

```bash
python scripts/train_pose_comparison.py \
  --trial --arm reference --device cpu --threads 4 \
  --trial-batches 4 --max-seconds 600 --max-rss-mib 6000 \
  --output ../baomai-resource-trial-001
```

The output must be a new directory outside the repository. The runner verifies the original checkpoint and source data, stages a fresh complete pair inside that directory, and trains only the selected arm. Original labels, images, the prior prepared pair, and epoch173 remain unchanged. Loader caches are confined to this run's copies.

The trial constructs the actual training and validation loaders. It verifies the loaded image/object/visible-endpoint totals and the resolved image/label hashes. It then executes real forward passes, finite losses, backward passes and optimizer updates at the planned image size and batch size. Optimizer updates are counted at the optimizer method, separately from batches; gradient accumulation can make these counts differ. A recorded parameter change demonstrates that at least one update changed weights.

The trial retains the 30-epoch schedule and warmup configuration. It stops with a dedicated trial outcome after the requested number of batches, before treating the partial epoch as complete. It does not write an eligible checkpoint manifest or report a new model-accuracy result. The pinned pose trainer constructs validation batches of eight when training batch size is four. The early trial does not execute those validation batches or checkpoint finalization, so its peak memory and timings do not qualify those later phases or all possible training batches.

The parent process records wall time and samples process-tree resident memory every 0.2 seconds. It terminates the worker if the wall-time or RSS limit is exceeded, or available system memory falls below 512 MiB. These are monitored stop thresholds, not a hard kernel allocation limit. The RSS threshold measures host memory, not GPU memory. A budget termination is an incomplete trial and cannot become a completed fine-tune.

Inspect `training-receipt.json`, `training-progress.json` and `train.log`. A successful resource result has `purpose: resource-trial`, `status: trial-complete`, positive optimizer updates and a recorded parameter change. It has no completed training epochs or final checkpoint. Source drift or changed inputs invalidate its receipt even if some batches ran. Development trials may record a dirty checkout; full comparisons require a clean revision.

### Published execution finding

Clean published revision `53bc4f9a01655ada3755f37b41be858f644e2b44` was executed on 2026-11-07 (Hong Kong time). The [execution record](training-execution-verification-2026-11-07.json) and [preserved training log](training-resource-2026-11-07.log) contain the precise evidence.

Both real loaders verified 137/20 images, 3,772/664 objects and 6,839/1,227 visible endpoints. The first four-image training batch had tensor shape `[4, 3, 736, 1280]`, completed in approximately 6.75 seconds, produced finite loss/gradients and completed one optimizer update with an observed parameter change.

During the next batch, sampled process-tree RSS reached **6,050.04 MiB** and crossed the **6,000 MiB** stop threshold. The parent terminated the worker after approximately **17.64 seconds**. The outcome was `resource-trial` / `failed`, with one completed batch, zero completed epochs and no eligible checkpoint. The real stopped receipt was also rejected by the checkpoint-manifest writer.

This establishes a limit of the chosen execution budget; it does not prove that every CPU machine is incapable of this training. Steady-state throughput, later-batch memory, validation and checkpoint-finalization costs remain unqualified. Do not extrapolate a reliable full-run duration from this single completed batch or silently change the scientific recipe to force a pass.

All **66 focused checks** passed on that published revision: 36 evaluator, 12 quality, 13 checkpoint-bundle and five execution-budget/input checks. The bundle checks use synthetic completed artifacts; no real 30-epoch model existed at that CPU-trial milestone. Both saved-prediction replays reproduced all 60 image/threshold pairs exactly. New inference on one dense and one empty baseline image verified the actual loaded `[2,3]` head and reproduced the prior predictions exactly.

## Complete each arm on qualified compute

The default scientific recipe, `paired-finetune-v1`, is fixed in `scripts/pose_checkpoint.py` and follows PR #1: 30 epochs, batch 4, image size 1280, rectangular batches, AdamW, initial learning rate 0.0001, seed 42, FP32 training, and the recorded augmentations/loss gains. Omitting `--recipe` preserves this original selection. The separately named `warmup-bias-lr-1e-4-v1` changes only `warmup_bias_lr` from `0.1` to `0.0001` and is restricted to reference labels; use its own handoff and packer. New schema-2 receipts/manifests identify the recipe and its canonical SHA-256, while existing schema-1 bundles retain their original identity and validation behavior. The runner explicitly fixes the complete-data fraction to floating-point `1.0`. It disables image RAM caching, compilation, plots and optional logging integrations; it records resolved settings and uses zero data-loader workers. These choices are shared by both arms.

Choose the same explicitly named device and thread count for both runs. Do not silently reduce image size or batch size after an out-of-memory error; the runner rejects automatic batch changes. If the configuration is unsuitable, record a separately revised experiment before trying a different scientific recipe.

For example, after qualifying a single CUDA device and choosing a sufficient wall-time budget:

```bash
python scripts/train_pose_comparison.py \
  --train --arm reference --device 0 --threads 4 \
  --max-seconds 21600 --max-rss-mib 6000 \
  --output ../baomai-reference-001

python scripts/train_pose_comparison.py \
  --train --arm candidate --device 0 --threads 4 \
  --max-seconds 21600 --max-rss-mib 6000 \
  --output ../baomai-candidate-001
```

The six-hour value is an example execution budget, not a measured runtime estimate. Device `0` requires a CUDA-enabled runtime and an available accelerator. Use `cpu` only when its measured resource cost is acceptable. Run one arm at a time on the same qualified device to avoid competing resource use.

Each invocation stages its own verified pair. Its absolute dataset roots and staging-manifest hash therefore differ from other runs. The manifest binds every input image/label hash; the source reference, proposal identity and shared settings establish which experimental data differ. It must be the same declared candidate for both arm preparations.

## Preserve final-epoch evidence

A completed bundle contains:

- `checkpoint-manifest.json`: the eligible model identity for evaluation.
- `training-receipt.json`: the arm, initialization, source/code hashes, actual settings/runtime, true optimizer-step count and completed epoch sequence.
- `staging-manifest.json`: the precise input identities for both prepared variants.
- `fit/weights/last.pt`: the finalized checkpoint from the predeclared final epoch.
- `fit/results.csv` and `train.log`: hashed completion and training evidence.

The runner records an epoch only after every expected training batch and image completed and its checkpoint was saved. All epochs 1 through 30 must be present. It also requires training and checkpoint finalization to return successfully and rechecks the input bytes, source identity and initialization after execution. It writes the eligible manifest only after the worker has exited and its log is closed.

The pinned trainer's final console evaluation uses `best.pt`, and optimizer stripping changes the stored checkpoint epoch to `-1`. Neither is sufficient evidence for the predeclared final-epoch comparison. This runner hashes and identifies finalized `last.pt` after all finalization work. The comparison evaluator should load that file through its manifest. A resource trial, interrupted run, changed-input run, or incomplete epoch sequence cannot supply an eligible manifest.

Use the explicit `--checkpoint-manifest` options described in [evaluate-pose.md](evaluate-pose.md) for inference, replay and quality diagnostics. Keep the frozen validation identity, thresholds and matching rules unchanged across M0, M1 and M2. Evaluate detection misses/extras, visible-point coverage, both-corner success and unlocatable-side behavior before deciding whether the candidate improves on both controls.

Preserve the complete model bundle when moving results between machines. Its relative artifact references remain together; the recorded absolute training paths describe the original execution. To train again, create a fresh run instead of editing the old manifest or reusing its loader caches.

## Scope of this deliverable

This comparison measures image-space object and corner performance on development validation. Current work does not require camera mounting, robot/gripper details or calibration. Both the original label comparison and the later warmup comparison have returned models and measured results. Continue from the [completed detection review and intake](warmup-detection-review-2026-11-07.md) with the required separately versioned annotation adaptation before preparing and scoring a supplementary reference; retain the original default under the predeclared rule. Physical robot integration remains a separate possible later task.

Implementation references: pinned [training loop and finalization](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/engine/trainer.py), [model training entry point](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/engine/model.py), and [pose trainer](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/models/yolo/pose/train.py). The runner instruments those existing operations rather than replacing the optimization or loss algorithm.
