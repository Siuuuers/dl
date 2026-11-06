# Prepare the controlled cassette training comparison

The annotation convention is now adopted in [annotation-policy.md](annotation-policy.md), with [upstream-source evidence](annotation-sources-2026-11-06.json). The existing eleven-row correction remains a [deduplication-only candidate](../data/candidates/dedup-2026-11-06/README.md). This stage prepares complete reference and candidate datasets; it does not change the supplied model or start training.

**Executed on 2026-11-06:** published revision `1f6aeb60c48d72e8d1b9fdfbe02f8b67eaa4fbb6` built and verified both complete datasets with a clean working tree. The installed pinned resolver checked all 314 image/label pairings; seven invalid-input probes were rejected and a final verification passed after restoration. Read the [execution record](training-preparation-verification-2026-11-06.json) for code/data hashes, results and limits.

## Build and verify a fresh pair

Use a checkout containing `scripts/stage_pose_training.py`. Fetch original image bytes, not annotated previews, and run from the repository root:

```bash
git lfs pull --include='data/images/train/*.png,data/images/val/*.png' --exclude=''
python3 scripts/verify_pose_candidate.py
python3 scripts/stage_pose_training.py --output ../baomai-pair-001
python3 scripts/stage_pose_training.py --verify ../baomai-pair-001
```

`--output` must name a fresh directory. Preflight checks all 157 frozen reference labels, the declared candidate and every original PNG hash/dimension before creating it. The script copies full `images/train`, `images/val`, `labels/train` and `labels/val` trees into both `reference/` and `candidate/`, each with its own `data.yaml`. Only the five declared candidate training labels differ. All validation files remain byte-identical.

| Arm | Training images / objects | Visible / missing training endpoints | Validation images / objects |
| --- | ---: | ---: | ---: |
| Reference fine-tune data | 137 / 3,772 | 6,839 / 705 | 20 / 664 |
| Candidate fine-tune data | 137 / 3,761 | 6,839 / 683 | 20 / 664 |

Both arms retain thirteen empty training images and three empty validation images. The script enforces label schema, image-X ordering, canonical missing points, clipped boxes, point containment and strict in-frame visible points. The `1e-4 px` box-geometry allowance is decimal-serialization tolerance, not a model-accuracy target. It does not certify physical annotation correctness or full migration to the new policy.

Each YAML has an explicit absolute root for its own arm. Use the generated path on that machine. If the pair is moved, rebuild it in a fresh destination; do not manually rewrite the YAML or manifest to bypass verification. A complete pair contains approximately 282 MB of copied original images, plus small labels and provenance.

`staging-manifest.json` is written only after all copied files, YAMLs, counts and geometry checks pass. It records every source/output hash and the executing script/helper hashes. `--verify` reconstructs the expected result from the reference and proposal, then compares it with actual files; it does not trust a forged staging manifest.

The staging tree is checked as a fresh input artifact: extra files, symlinks and caches are rejected. Run verification before the first training-loader scan. Ultralytics may subsequently write label caches. Preserve the original staging receipt; use a fresh staged pair when repeating the immutable-input verification. A failed build may leave an incomplete directory without a complete manifest; choose another fresh output path after correcting the reported input problem.

## Verify the label paths used by the pinned runtime

Ultralytics **8.4.182** resolves split directories and derives label paths from the image path's final `images` component. A whole-directory symlink to the frozen images can resolve back to the source labels. Full copies avoid that ambiguity.

The integration check should call the installed `check_det_dataset(..., autodownload=False)`, `BaseDataset.get_img_files(..., fraction=1.0)` and `img2label_paths`, then verify all 314 image/label pairings across both arms. The reference arm must map to its reference labels; the candidate arm must map to its candidate labels. Also check the content hashes, not merely a filename or object count. This path check is separate from constructing a full training loader or performing a training step.

## What must be completed before the paired fine-tune

1. **Use explicit candidate-checkpoint identity in evaluation.** The evaluator, replay reader and quality diagnostic now accept `--checkpoint-manifest` for an eligible completed arm. Their default remains epoch173. The bundle binds the finalized checkpoint to training evidence while preserving frozen validation hashes; see [evaluate-pose.md](evaluate-pose.md). Do not overwrite the baseline model card or supply an arbitrary saved-prediction model hash.
2. **Run through the recorded training entry point.** [train-pose-comparison.md](train-pose-comparison.md) documents `scripts/train_pose_comparison.py`, which creates fresh staged inputs, records source/runtime/settings and actual completion, and writes an eligible checkpoint manifest only after a complete verified 30-epoch arm. Resource-trial outputs remain ineligible. Preserve the receipt, log and results with the final checkpoint.
3. **Check compute with a bounded real training resource trial.** The inspected environment has Python 3.12.14, Ultralytics 8.4.182 and Torch 2.10.0+cpu, with no CUDA device. The [published resource trial](training-execution-verification-2026-11-07.json) completed one real batch and then exceeded its 6,000 MiB host-memory budget. No epoch completed; later batches, validation and finalization remain unqualified. Inference latency cannot predict backward-pass memory or duration. Use the same suitable device and settings for both arms; record a change if resource limits require revising the recipe.
4. **Run [PR #1](https://github.com/Siuuuers/dl/pull/1)'s paired experiment.** Each arm freshly loads epoch173, with the same declared 30-epoch schedule, seed 42, batch 4, image size, augmentations and loss gains. Only the dataset YAML and run name differ. The initial M0 model is unchanged.
5. **Evaluate the predeclared final epoch from both arms.** Do not substitute independently selected `best.pt` files. The pinned trainer's `final_eval()` validates `best.pt` when present, while checkpoint stripping sets the stored epoch field to `-1`. Therefore, the last console validation and checkpoint epoch field alone do not establish the intended final-epoch comparison. Preserve the completed training receipt and hash the intended final `last.pt` after finalization.

Primary implementation references: [pinned data utilities](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/data/utils.py), [base dataset](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/data/base.py), [trainer](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/engine/trainer.py), and [checkpoint utilities](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/utils/torch_utils.py). These behaviors were inspected in the installed pinned package; no new model was trained for this preparation.

## Priorities after the first controlled result

The saved development evidence identifies distinct failure modes:

| Observed problem | Existing evidence | Next useful action |
| --- | --- | --- |
| Inconsistent instance annotations | Eleven reviewed duplicate rows in five training images | Test the isolated deletion candidate against an unchanged-data fine-tune. Do not assume a causal improvement before measuring it. |
| Wrong location despite high endpoint confidence | Reviewed `new_000008` row10 right endpoint is about 14.85 px from the retained reference, with confidence about 0.934 | Inspect comparable training corners and collect representative wrong-edge/occlusion examples if this persists. Do not move supported reference points to predictions. |
| Incorrect availability on unlocatable sides | 61/66 eligible reference-unlocatable sides still receive coordinates at endpoint cutoff 0.50 | Preserve meaningful missing-point training examples and report availability alongside visible-corner accuracy. Higher cutoffs already lose useful visible outputs. |
| Missing usable pairs and detection errors | At illustrative 5 px: 14 fully visible objects lack both returned points; 49 return both but miss the radius. Object baseline: 39 FN and 63 FP | Keep pair availability, localization and detection counts separate. Review box convention effects; do not weaken IoU merely to improve the score. |
| Limited independent evidence | Twenty validation images already used for epoch selection and subsequent development | Obtain genuinely untouched scene/session groups before a final generalization claim. Do not train on the reviewed validation examples. |

These are recorded in [model-quality.md](model-quality.md) and its source-hashed evidence. All 157 source originals are now available and verified, but downloading them does not mean all have received complete visual annotation review.

Compare each candidate against both M0 and the unchanged-data fine-tune. Retain PR #1's conservative no-regression rule and include the newer paired-corner and unlocatable-side diagnostics in the decision. The 2/5/10 px radii remain development illustrations, not approved acceptance limits. If performance trades one important error against another, retain the supplied default and report the tradeoff. A single seeded result is preliminary.

Keep broader box relabeling, new training images, augmentation/loss changes and architecture changes as separate experiments. If the isolated correction is inconclusive, use the actual failure examples to choose one next change. The user has confirmed robotic positioning/grasping as the intended use. The final application claim still requires calibrated geometry, a justified physical tolerance, acceptable misses/extras and availability behavior, action timing and representative independent testing; see [robot-grasp-readiness.md](robot-grasp-readiness.md).
