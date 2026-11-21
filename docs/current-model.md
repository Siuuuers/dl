# Current model and training entry

22 November 2026, Asia/Hong_Kong.

**PR35 remains selected. The current training entry is the r18 notebook.** The [PR90 audit](../results/evaluation/pr90-r17-20261122/README.md) verifies both returned r17 runs and reports their reference-dependent rankings. The [r18 handoff](front-review-r18.md) applies confident coordinate corrections from the requested 081–130 review. No r18 GPU training or model promotion occurred during preparation.

## Run inference

Use branch `codex/front-review-r18-20261122`. Follow the [folder guide](predict-pose-folder.md#use-the-seed46-candidate-explicitly) to obtain the verified PR35 candidate bundle, retaining its manifest and companion files.

| Setting | Value |
| --- | --- |
| Checkpoint | PR35 seed 46 candidate |
| Checkpoint SHA-256 | `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` |
| Input size | 1280 |
| Box confidence | 0.13 |
| NMS IoU | 0.70 |
| Endpoint confidence | 0.50 |

Inspect the complete original image, including regions without proposals. An endpoint confidence score does not certify its pixel accuracy or physical visibility.

## What PR90 established

On the frozen 666-front primary reference, PR35 / fresh control / r17 candidate have 18 / 20 / 34 misses and 80 / 83 / 103 unmatched outputs. Five-pixel corner-pair successes are 533 / 531 / 521 out of 584 eligible fronts. The historical 663-front comparison agrees with this ranking.

On the pre-return 662-front reviewed reference, misses are 35 / 38 / 15 and unmatched outputs are 101 / 105 / 88; five-pixel pairs are 538 / 534 / 549. This favorable exploratory result does not replace the frozen selection rule. Compared with r15 on identical reviewed labels, r17 gains one match but adds two unmatched outputs and one duplicate, with unchanged aggregate five- and ten-pixel corner-pair counts. [Measured results and reproducible evidence](../results/evaluation/pr90-r17-20261122/README.md).

## Run the r18 comparison

The [complete r18 review](../data/reviews/fronts-2026-11-22-r18/manifest.json) contains all 157 annotation files under its `labels/` directory. The [r18 training candidate](../data/candidates/front-review-2026-11-22-r18/manifest.json) explicitly selects eligible supervision. Original `data/labels/` bytes remain preserved.

[Open r18 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r18-20261122/notebooks/baomai_front_review_r18.ipynb). Select GPU, choose Run all, then return **`baomai-front-review-r18-results.zip`**. Both arms initialize independently from original epoch 173: seed 46, 30 epochs, batch 4, 1280 input, FP32, AdamW, learning rate 0.0001. The resource trial cannot initialize a full arm; final-epoch selection is fixed.

Both arms retain 137 training photos and 13 empty scenes. Reference / candidate contain 3,771 / 3,784 fronts. The notebook prints exact changed-file and endpoint counts before training. Photo 015 and the separate 31 old-series treatment exclusions retain exact PR35 supervision in both arms. No new holds or uncertainty flags are introduced. [Actual input verification](../results/review/front-review-20261122-r18/verification.json).

Original 664-front trainer validation and frozen 666 / 663 scoring references remain unchanged. The 662-front r18 development review is separately versioned exploratory evidence. Its corrections do not alter the PR90 audit. [Notebook source and preflight receipt](../results/review/front-review-20261122-r18/notebook-verification.json).

## Evidence limits and future physical geometry

The new comparison measures PR35 supervision against the complete eligible r18 set; it does not isolate this amendment round. Repeated development scenes do not establish reliability in an unseen capture session. A new independently labeled capture session would supply that missing evidence.

The supplied dataset and inspected camera paths provide RGB frames. Pixel radii do not establish a gripper tolerance in millimetres. The [robot-grasp guide](robot-grasp-readiness.md) records the calibration and physical measurements needed for later robot integration; they are not prerequisites for this annotation notebook.
