# Current cassette workflow

Updated 21 November 2026, Asia/Hong_Kong.

**PR35 remains selected. The new training entry is the r14 notebook.** The [r14 handoff](front-review-r14.md) saves 233 row amendments in 30 photos and a PDF containing exactly those changed photographs. The [completed PR84 audit](../results/evaluation/pr84-r13-20261121/README.md) found a mixed result: the r13 candidate fits the reviewed geometry better but regresses on the frozen references. Its GPU training is complete; no r14 GPU training or model promotion has occurred.

## Selected inference command

Use branch `codex/front-review-r14-20261121`. Follow the [folder guide](predict-pose-folder.md#use-the-seed46-candidate-explicitly) to obtain the verified PR35 candidate bundle, keeping its manifest and companion files together.

```bash
.venv/bin/python scripts/predict_pose_folder.py \
  --checkpoint-manifest ../seed46-results/candidate/checkpoint-manifest.json \
  --input /path/to/original-images \
  --output /path/to/new-review-results \
  --device cpu --conf 0.13 --nms-iou 0.70 --endpoint-conf 0.50
```

Selected checkpoint: PR35 seed 46, final epoch 30, SHA-256 `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1`. Input size 1280. The generic runtime default is still epoch 173 at confidence 0.25, so use the explicit manifest above. Inspect the complete original image, including areas without proposals. Display each endpoint independently at confidence 0.50 and connect the two when both qualify. Confidence does not establish physical visibility or accuracy.

## Current annotations and notebook

The [complete r14 review](../data/reviews/fronts-2026-11-21-r14/manifest.json) contains all 157 annotation files. The [r14 training candidate](../data/candidates/front-review-2026-11-21-r14/manifest.json) applies only eligible supervision. These are separately bound versions: the original `data/labels/` folder remains the immutable source dataset.

[Open r14 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r14-20261121/notebooks/baomai_front_review_r14.ipynb). Select GPU, Run all, then return **`baomai-front-review-r14-results.zip`**. Both arms independently start from original epoch 173 with seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW and learning rate 0.0001. The resource trial does not initialize a full arm. Final-epoch selection is fixed.

All 137 training photos and 13 empty scenes remain. The actual reference/candidate contains 3,771/3,787 training fronts, with 92 training label files differing. The user accepted 17 previously displayed amber cases, releasing 14 training photos. Only photo015's unboxed possible-instance question remains held. That photo and the separate 31 old-photo exclusions use exact PR35 labels in both arms. The [runner verification](../results/review/front-review-20261121-r14/runner-verification.json) checks the actual staged bytes.

Original 664-front training validation and the fixed 666/663 scoring references remain unchanged. The current 662-front r14 development review is exploratory. This compares the combined annotation amendments and newly approved training eligibility with PR35 supervision; it does not isolate those two effects. A separate evaluation of the returned models is needed before replacing the selected model.

## Physical questions and limits

The prior displayed amber labels, including photos074 and089, are accepted through the [user approval ledger](../results/review/front-review-20261121-r14/user-approval.json). Photo015's possible extra instance had no drawn row, so that distinct question remains unresolved. Its supported amendments are saved in the review and PDF but stay out of the treatment. See the [current case ledger](../results/review/front-review-20261121-r14/review-status.json) and [request dispositions](../results/review/front-review-20261121-r14/request-coverage.json).

A training run does not settle unclear front/rim ownership. Repeated development scenes do not establish accuracy on new camera sessions or robotic positioning. The [PR84 evaluation](../results/evaluation/pr84-r13-20261121/README.md) reports frozen, pre-return reviewed and post-return r14 results separately. The new r14 notebook's input preflight passed; GPU training is the next step.
