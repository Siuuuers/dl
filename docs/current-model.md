# Current model and training entry

22 November 2026, Asia/Hong_Kong.

**PR35 remains selected. The current training entry is the r17 notebook.** The [r17 handoff](front-review-r17.md) continues from the saved r16 labels and applies the next requests for photographs 067–075, 079 and 080. The earlier notebook handoff was interrupted before its source pin was finished; the labels were already saved. The [PR88 audit](../results/evaluation/pr88-r15-20261122/README.md) retains PR35. No r17 GPU training or promotion has occurred during preparation.

## Run inference

Use branch `codex/front-review-r17-20261122`. Follow the [folder guide](predict-pose-folder.md#use-the-seed46-candidate-explicitly) to obtain the verified PR35 candidate bundle, keeping its manifest and companion files together.

| Setting | Value |
| --- | --- |
| Checkpoint | PR35 seed 46 candidate |
| Checkpoint SHA-256 | `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` |
| Input size | 1280 |
| Box confidence | 0.13 |
| NMS IoU | 0.70 |
| Endpoint confidence | 0.50 |

Inspect the complete original image, including regions without proposals. A confident endpoint can still be inaccurate or lie behind an occluder.

## Run the r17 comparison

The [complete r17 review](../data/reviews/fronts-2026-11-22-r17/manifest.json) contains all 157 annotation files under its `labels/` directory. The [r17 training candidate](../data/candidates/front-review-2026-11-22-r17/manifest.json) applies eligible supervision explicitly. `data/labels/` remains the preserved original dataset.

[Open r17 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r17-20261122/notebooks/baomai_front_review_r17.ipynb). Select GPU, choose Run all, then return **`baomai-front-review-r17-results.zip`**. Both arms initialize independently from original epoch 173: seed 46, 30 epochs, batch 4, 1280 input, FP32, AdamW, learning rate 0.0001. The resource trial cannot initialize a full arm; final-epoch selection is fixed.

All 137 training photos and 13 empty scenes remain. Reference/candidate contain 3,771/3,784 fronts, with 92 differing training-label files; the notebook prints the endpoint counts before training. The prior 17 displayed-case approvals are inherited. Photo 015's undrawn possible-instance question remains the sole hold. It and the separate 31 old-series treatment exclusions use exact PR35 supervision in both arms. [Actual input verification](../results/review/front-review-20261122-r17/verification.json).

Original 664-front trainer validation and frozen 666/663 scoring references remain unchanged. The 662-front current development review is exploratory and keeps the same label bytes as r16. The [notebook preflight and source receipt](../results/review/front-review-20261122-r17/notebook-verification.json) records preparation checks; the returned r17 checkpoints will need their own evaluation.

## Evidence limits and future physical geometry

The [case ledger](../results/review/front-review-20261122-r17/review-status.json) retains photo 015's undrawn question. Its existing review bytes are saved but withheld from the training treatment. Amendment interpretations are explicit in the [request dispositions](../results/review/front-review-20261122-r17/request-coverage.json).

The PR88 report separates the frozen benchmark, pre-return r15 review and post-return r16 replay. A change in reference geometry is not new model learning. Repeated development scenes do not establish unseen-session reliability.

The supplied dataset and inspected camera paths provide RGB frames. Ten pixels is a per-corner image-space criterion, not a gripper tolerance in millimetres. The [robot-grasp guide](robot-grasp-readiness.md) records what is and is not established by the repository. Hardware details are not a gate for this notebook; a future robot integration needs actual gripper clearances, supported depth or plane geometry, and installation-specific calibration.
