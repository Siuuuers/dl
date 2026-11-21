# Current model and training entry

21 November 2026, Asia/Hong_Kong.

**PR35 remains selected. The new training entry is the r15 notebook.** The [r15 handoff](front-review-r15.md) saves 141 amended rows and one requested deletion across 24 photographs. Its 25-page PDF shows all 947 current labels on those photographs. The [completed PR86 audit](../results/evaluation/pr86-r14-20261121/README.md) finds a mixed result: the r14 candidate fits the reviewed geometry better but regresses on the frozen primary reference. No r15 GPU training or model promotion has occurred during preparation.

## Run inference

Use branch `codex/front-review-r15-20261121`. Follow the [folder guide](predict-pose-folder.md#use-the-seed46-candidate-explicitly) to obtain the verified PR35 candidate bundle, keeping its manifest and companion files together.

| Setting | Value |
| --- | --- |
| Checkpoint | PR35 seed46 candidate |
| Checkpoint SHA-256 | `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` |
| Input size | 1280 |
| Box confidence | 0.13 |
| NMS IoU | 0.70 |
| Endpoint confidence | 0.50 |

Inspect the complete original image, including regions without proposals. A confident endpoint can be an estimate behind an occluder; confidence does not prove physical visibility or coordinate accuracy.

## Run the r15 comparison

The [complete r15 review](../data/reviews/fronts-2026-11-21-r15/manifest.json) contains all 157 annotation files. The [r15 training candidate](../data/candidates/front-review-2026-11-21-r15/manifest.json) applies eligible supervision explicitly. The original `data/labels/` folder remains the preserved source dataset.

[Open r15 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r15-20261121/notebooks/baomai_front_review_r15.ipynb). Select GPU, Run all, then return **`baomai-front-review-r15-results.zip`**. Both arms start independently from original epoch 173 with seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW and learning rate 0.0001. The resource trial does not initialize a full arm; final-epoch selection is fixed.

All 137 training photos and 13 empty scenes remain. Reference/candidate contain 3,771/3,786 training fronts, with 92 differing training label files. The prior 17 displayed-case approvals are inherited, with no new release in this round. Photo015's undrawn possible-instance question is the sole hold. It and the separate 31 old-series treatment exclusions use exact PR35 supervision in both arms. [Actual input verification](../results/review/front-review-20261121-r15/verification.json).

Original 664-front trainer validation and fixed 666/663 scoring references stay unchanged. The 662-front r15 development review is a separately versioned exploratory diagnostic. The [notebook preflight and source receipt](../results/review/front-review-20261121-r15/notebook-verification.json) records the preparation checks; returned r15 models will need their own fixed evaluation.

## Remaining evidence limits

The [case ledger](../results/review/front-review-20261121-r15/review-status.json) retains photo015's unboxed question. Its supported existing-row amendments are saved in the review and PDF but withheld from training. Eight unsupported wording clauses and all other interpretations are explicit in the [request dispositions](../results/review/front-review-20261121-r15/request-coverage.json).

The PR86 report distinguishes frozen, pre-return r14 and post-return r15 scoring. A change in reference geometry is not new model learning. Repeated development scenes and image-pixel scores do not establish reliability on unseen camera sessions or millimetre robotic positioning.
