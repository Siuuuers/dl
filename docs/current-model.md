# Current cassette workflow

Updated 19 November 2026, Asia/Hong_Kong.

**Keep PR35 selected. The completed PR74 r7 comparison does not support replacing it.** The [r8 amendment](front-review-r8.md), complete 157-photo PDF and next notebook are prepared separately. R8 has not been trained.

## One selected inference command

Use branch `codex/front-review-r8-20261119`. Follow the [folder guide](predict-pose-folder.md#use-the-seed46-candidate-explicitly) to obtain the verified PR35 candidate bundle, keeping its manifest and companion files together.

```bash
.venv/bin/python scripts/predict_pose_folder.py \
  --checkpoint-manifest ../seed46-results/candidate/checkpoint-manifest.json \
  --input /path/to/original-images \
  --output /path/to/new-review-results \
  --device cpu --conf 0.13 --nms-iou 0.70 --endpoint-conf 0.50
```

Selected checkpoint: PR35 seed46, final epoch 30, SHA-256 `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1`. Input 1280. The generic runtime default remains epoch173 at confidence 0.25, so use the explicit manifest above. Inspect the entire original image, including areas without proposals. Display each endpoint independently at confidence 0.50; connect two qualifying endpoints. Confidence does not establish physical visibility or accuracy.

## Latest completed comparison: PR74

The [PR74 audit](../results/evaluation/pr74-20261119/README.md) verifies both completed 30-epoch A100 arms and fresh inference of PR35, control and candidate under identical settings and frozen references.

| Primary reference: 666 fronts | Matches | Scored misses | Unmatched outputs | Both corners ≤5 px | Both ≤10 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained PR35 | 648 | 18 | 80 | 533/584 | 582/584 |
| Fresh r7 control | 648 | 18 | 80 | 533/584 | 582/584 |
| PR74 r7 candidate | 639 | 27 | 95 | 537/584 | 580/584 |

The candidate loses nine matches and adds 15 unmatched outputs against its control, despite four more five-pixel pair successes. Historical 663-front scoring also worsens. Scored misses include extent failures: new010:15 has a confident proposal at IoU 0.4973. Original-pixel evidence shows rack included in the frozen new057:11 box. The score and frozen reference are retained; the limitations are reported separately.

PR35 and the fresh r7 control decoded prediction arrays are exactly equal on these 20 images at this operating point; checkpoint bytes differ. No result establishes equality on other scenes. The [PR72 audit](../results/evaluation/pr72-20261119/README.md) remains a separate historical comparison.

## Current amendment and next notebook

The [r8 review](front-review-r8.md) freshly inspects all 157 photographs and changes 1,618 rows across 93 photos. Every changed box records four original-pixel extrema. The [boundary guide](annotation-boundaries-r8.md) explains connected rack members, own rims and independent physical lower-corner decisions. The complete PDF highlights only changes from r7.

The complete review has 3,788 training fronts and 663 development fronts. The actual candidate/control has **3,770/3,771 training fronts**, with **59 training photos** differing in labels. **64 whole held training photos**, including all 31 old-series images, retain exact PR35 labels in both arms. The original `old_000013` file remains byte-identical.

[Open r8 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r8-20261119/notebooks/baomai_front_review_r8.ipynb). Select GPU and Run all, then return the final ZIP. Both full arms independently start from original epoch173 with seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW and learning rate 0.0001. The four-batch resource check does not initialize a full run. Final-epoch selection is fixed.

Original 664-front training validation and fixed 666/663 scoring references remain unchanged. The current development review is exploratory. This compares the accumulated eligible supervision package with PR35 labels; it does not isolate one amendment. No model is automatically promoted.

## Remaining physical questions

The old-series broad lower flange is connected to the cassette. The unresolved issue is the intended landmark at its outer turn versus the front-wall junction. Gray projections and gripper-adjacent strips can likewise retain ownership holds even after another edge is corrected. See the [explicit case ledger](../results/review/front-review-20261119-r8/review-status.json). A new training run does not settle those physical definitions, and repeated development scenes do not establish unseen-session reliability.
