# Current cassette workflow

Updated 21 November 2026, Asia/Hong_Kong.

**PR35 remains selected. The new training entry is the r13 notebook.** The [r13 handoff](front-review-r13.md) saves the amendments and a PDF of all nonempty new-series photographs. The [completed PR82 audit](../results/evaluation/pr82-r12-20261121/README.md) found a mixed result: the r12 candidate fits the reviewed geometry better but regresses on the frozen references. No r13 training or model promotion has occurred.

## Selected inference command

Use branch `codex/front-review-r13-20261121`. Follow the [folder guide](predict-pose-folder.md#use-the-seed46-candidate-explicitly) to obtain the verified PR35 candidate bundle, keeping its manifest and companion files together.

```bash
.venv/bin/python scripts/predict_pose_folder.py \
  --checkpoint-manifest ../seed46-results/candidate/checkpoint-manifest.json \
  --input /path/to/original-images \
  --output /path/to/new-review-results \
  --device cpu --conf 0.13 --nms-iou 0.70 --endpoint-conf 0.50
```

Selected checkpoint: PR35 seed 46, final epoch 30, SHA-256 `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1`. Input size 1280. The generic runtime default is still epoch 173 at confidence 0.25, so use the explicit manifest above. Inspect the complete original image, including areas without proposals. Display each endpoint independently at confidence 0.50 and connect the two when both qualify. Confidence does not establish physical visibility or accuracy.

## Current annotations and notebook

The [complete r13 review](../data/reviews/fronts-2026-11-21-r13/manifest.json) contains all 157 annotation files. The [r13 training candidate](../data/candidates/front-review-2026-11-21-r13/manifest.json) applies only eligible supervision. These are separately bound versions: the original `data/labels/` folder remains the immutable source dataset.

[Open r13 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r13-20261121/notebooks/baomai_front_review_r13.ipynb). Select GPU, Run all, then return **`baomai-front-review-r13-results.zip`**. Both arms independently start from original epoch 173 with seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW and learning rate 0.0001. The resource trial does not initialize a full arm. Final-epoch selection is fixed.

All 137 training photos and 13 empty scenes remain. The actual reference/candidate contains 3,771/3,777 training fronts. Fifteen unresolved training photos and the separate 31 accepted old-photo exclusions have exact PR35 labels in both arms; 78 training label files differ. The [runner verification](../results/review/front-review-20261121-r13/runner-verification.json) checks the actual staged bytes.

Original 664-front training validation and the fixed 666/663 scoring references remain unchanged. The current 662-front r13 development review is exploratory. This compares the accumulated eligible annotation package with PR35 supervision. A separate evaluation of the returned models is needed before replacing the selected model.

## Physical questions and limits

Photo 124's former horizontal-direction question is resolved by the new xmax-only instruction. Photo 074's lower aperture ownership and the possible extra front below photo 089 row 47 remain open. See the [current case ledger](../results/review/front-review-20261121-r13/review-status.json) and [request dispositions](../results/review/front-review-20261121-r13/request-coverage.json).

A training run does not settle unclear front/rim ownership. Repeated development scenes do not establish accuracy on new camera sessions or robotic positioning. The [PR82 evaluation](../results/evaluation/pr82-r12-20261121/README.md) reports frozen and current-reference results separately. The new r13 notebook tests the amended labels under the same matched recipe; its input preflight passed, and GPU training is still the next step.
