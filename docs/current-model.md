# Current model and training entry

22 November 2026, Asia/Hong_Kong.

**PR35 remains selected. The current training entry is the r19 notebook.** The [PR92 audit](../results/evaluation/pr92-r18-20261122/README.md) verifies the returned r18 runs. The [r19 handoff](front-review-r19.md) saves cassette-front boundaries and corner corrections after inspecting all 40 supplied nonempty photos in 081–130. No r19 GPU training or model promotion occurred during preparation.

## Run inference

Use branch codex/front-review-r19-20261122. Follow the [folder guide](predict-pose-folder.md#use-the-seed46-candidate-explicitly) to obtain the verified PR35 candidate bundle, retaining its manifest and companion files.

| Setting | Value |
| --- | --- |
| Checkpoint | PR35 seed 46 candidate |
| Checkpoint SHA-256 | 420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1 |
| Input size | 1280 |
| Box confidence | 0.13 |
| NMS IoU | 0.70 |
| Endpoint confidence | 0.50 |

Inspect complete original images, including regions without proposals. An endpoint confidence score does not certify its pixel accuracy or physical visibility.

## What PR92 established

On the frozen 666-front primary reference, PR35 and fresh control both have 18 misses / 80 unmatched outputs, versus r18's 34 / 103. Five-pixel pairs are 533/584 versus 522/584. The historical 663-front reference agrees with that ranking. PR35 and fresh control have exactly equal prediction arrays on the 20 photos.

On the pre-return 662-front r18 review, PR35/control have 36 misses / 102 unmatched outputs, while r18 has 15 / 88. Five-pixel pairs are 538/584 versus 551/584. All models return zero outputs on the three empty scenes. These exploratory results do not replace the fixed selection rule.

On identical r18 labels, the previous r17 candidate has one fewer miss and unmatched output, two fewer five-pixel successes, and the same ten-pixel successes. Training hardware also differs between generations. The audit separates this from the reference-only change at 090 row 43. [Full results, receipts and reproducible predictions](../results/evaluation/pr92-r18-20261122/README.md).

## Run the r19 comparison

The [complete r19 review](../data/reviews/fronts-2026-11-22-r19/manifest.json) contains all 157 annotation files. The [r19 training candidate](../data/candidates/front-review-2026-11-22-r19/manifest.json) selects eligible supervision. Saved changes comprise 460 row replacements across 34 photos; the only new visible point is 127 row 32's left front corner. Existing instance counts and row identities are preserved. [Annotation definition](annotation-boundaries-r19.md).

[Open r19 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r19-20261122/notebooks/baomai_front_review_r19.ipynb). Select GPU, choose Run all, and return **baomai-front-review-r19-results.zip**. Both arms initialize independently from original epoch 173: seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW, learning rate 0.0001. Selection uses the final epoch.

Both arms retain 137 training photos and 13 empty scenes. Reference / candidate contain 3,771 / 3,784 fronts and 6,841 / 6,838 locatable endpoints. Exactly 92 training-label files differ. Photo 015 and the separate 31 old-series treatment exclusions retain exact PR35 supervision in both arms; no new holds or uncertainty flags are added. [Verified inputs](../results/review/front-review-20261122-r19/verification.json).

Original 664-front trainer validation and frozen 666 / 663 scoring references remain unchanged. The 662-front r19 development review is separately versioned exploratory evidence. These amendments do not alter the PR92 comparison. [Notebook source and preflight receipt](../results/review/front-review-20261122-r19/notebook-verification.json).

## Evidence limits and future physical geometry

This experiment measures PR35 supervision against the complete eligible r19 set; it does not isolate this round. Repeated development scenes do not establish unseen-session reliability. A separate independently labeled capture session would supply that evidence.

The supplied images are RGB frames. Pixel radii do not establish gripper tolerance in millimetres. The [robot-grasp guide](robot-grasp-readiness.md) records calibration and physical measurements for later robot integration; they are not prerequisites for this annotation notebook.
