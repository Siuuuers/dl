# Current cassette workflow

Updated 20 November 2026, Asia/Hong_Kong.

**PR35 remains selected. The new training entry is the r12 notebook.** The [r12 amendment](front-review-r12.md) saves the requested label changes, a changed-photo PDF and a separately pinned experiment. It has not been trained. [PR80](https://github.com/Siuuuers/dl/pull/80) contains the completed r11 return.

## Selected inference command

Use branch `codex/front-review-r12-20261120`. Follow the [folder guide](predict-pose-folder.md#use-the-seed 46-candidate-explicitly) to obtain the verified PR35 candidate bundle, keeping its manifest and companion files together.

```bash
.venv/bin/python scripts/predict_pose_folder.py \
  --checkpoint-manifest ../seed 46-results/candidate/checkpoint-manifest.json \
  --input /path/to/original-images \
  --output /path/to/new-review-results \
  --device cpu --conf 0.13 --nms-iou 0.70 --endpoint-conf 0.50
```

Selected checkpoint: PR35 seed 46, final epoch 30, SHA-256 `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1`. Input size 1280. The generic runtime default is still epoch 173 at confidence 0.25, so use the explicit manifest above. Inspect the complete original image, including areas without proposals. Display each endpoint independently at confidence 0.50 and connect the two when both qualify. Confidence does not establish physical visibility or accuracy.

## Current annotations and notebook

The [complete r12 review](../data/reviews/fronts-2026-11-20-r12/manifest.json) contains all 157 annotation files. The [r12 training candidate](../data/candidates/front-review-2026-11-20-r12/manifest.json) applies only eligible supervision. These are separately bound versions: the original `data/labels/` folder remains the immutable source dataset.

[Open r12 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r12-20261120/notebooks/baomai_front_review_r12.ipynb). Select GPU, Run all, then return **`baomai-front-review-r12-results.zip`**. Both arms independently start from original epoch 173 with seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW and learning rate0.0001. The resource trial does not initialize a full arm. Final-epoch selection is fixed.

All 137 training photos and 13 empty scenes remain. The actual reference/candidate contains 3,771/3,777 training fronts. Fifteen unresolved training photos and the separate 31 accepted old-photo exclusions have exact PR35 labels in both arms; 78 training label files differ. The [runner verification](../results/review/front-review-20261120-r12/runner-verification.json) checks the actual staged bytes.

Original 664-front training validation and the fixed 666/663 scoring references remain unchanged. The current 662-front r12 development review is exploratory. This compares the accumulated eligible annotation package with PR35 supervision. A separate evaluation of the returned models is needed before replacing the selected model.

## Physical questions and limits

Photo 124's former horizontal-direction question is resolved by the new xmax-only instruction. Photo 074's lower aperture ownership and the possible extra front below photo 089 row 47 remain open. See the [current case ledger](../results/review/front-review-20261120-r12/review-status.json) and [request dispositions](../results/review/front-review-20261120-r12/request-coverage.json).

A training run does not settle unclear front/rim ownership. Repeated development scenes do not establish accuracy on new camera sessions or robotic positioning. The [saved PR79 audit](../results/evaluation/pr79-r10-20261119/README.md) remains a historical evaluation; r12 preparation adds no fresh inference scores or model promotion.
