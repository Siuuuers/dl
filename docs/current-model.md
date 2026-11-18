# Current cassette workflow

Updated 19 November 2026, Asia/Hong_Kong.

**Keep PR35 selected. The completed PR72 r5 comparison does not support replacing it.** The corrected r7 review, complete 157-photo PDF and next controlled notebook are prepared separately. R7 has not been trained.

## One selected inference command

Use branch `codex/front-review-r7-20261119`. Follow the [folder guide](predict-pose-folder.md#use-the-seed46-candidate-explicitly) to obtain the verified PR35 candidate bundle, keeping its manifest and companion files together.

```bash
.venv/bin/python scripts/predict_pose_folder.py \
  --checkpoint-manifest ../seed46-results/candidate/checkpoint-manifest.json \
  --input /path/to/original-images \
  --output /path/to/new-review-results \
  --device cpu --conf 0.13 --nms-iou 0.70 --endpoint-conf 0.50
```

Selected checkpoint: PR35 seed46, final epoch 30, SHA-256 `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1`. Input size 1280. The generic runtime default remains epoch173 at confidence 0.25, so use the explicit manifest above.

Review the entire original image, including places without proposals. Each endpoint is displayed independently when its score reaches 0.50 and its position is finite and in frame. A line connects two qualifying endpoints. Confidence does not establish physical visibility, identity or accuracy.

## Latest completed comparison: PR72

The [PR72 audit](../results/evaluation/pr72-20261119/README.md) verifies the returned archive and both completed 30-epoch L4 arms. PR35, the fresh control and r5 candidate were all freshly inferred in one CPU runtime, with identical operating settings and unchanged references.

| Primary reference: 666 fronts | Scored misses | Unmatched outputs | Both corners within 5 px | Both within 10 px |
| --- | ---: | ---: | ---: | ---: |
| Retained PR35 | 18 | 80 | 533/584 | 582/584 |
| Fresh r5 control | 20 | 83 | 531/584 | 583/584 |
| PR72 r5 candidate | 30 | 105 | 537/584 | 580/584 |

The candidate gains six five-pixel pair successes against its control but loses ten matches, produces 22 more unmatched outputs and loses three ten-pixel pair successes. The required historical 663-front reference also worsens: control/candidate misses 17/26, unmatched outputs 83/104. PR35 remains selected.

A scored miss means no qualifying one-to-one match at IoU 0.50. Strong proposals can fail because their extents disagree with the reference. Likewise, unmatched outputs are not all independently verified background errors. The 584 pair denominator includes eligible missed fronts as failures; the 666-front detection denominator includes zero- and one-corner fronts. See the saved traces and provenance in the complete audit.

Repeatedly used development images cannot establish unseen-session reliability. No model is promoted by either the audit or the new annotation pass. The [PR70 audit](../results/evaluation/pr70-20261119/README.md) remains a separate historical result.

## Current annotation pass and next notebook

The [r7 handoff](front-review-r7.md) follows six new marked examples plus the explicit `old_000013` original-box confirmation. Seven photos were directly re-inspected: 18 existing rows changed and one visible front was restored. The previous complete r6 review remains the baseline; the new PDF includes all 157 photos and highlights only r7 changes.

The complete review contains 3,788 training fronts and 663 development fronts. The actual candidate contains 3,780 training fronts because 44 whole held photos use exact PR35 labels in both arms. All 31 old-series training photos retain that hold; the user-confirmed `old_000013` review box is still restored exactly. Candidate/control counts are 3,780/3,771, with 79 training photos differing in label bytes. Both retain 137 training photos and 13 empty scenes.

[Open the r7 notebook in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r7-20261119/notebooks/baomai_front_review_r7.ipynb). Select GPU and Run all, then return the final results ZIP. The notebook pins an actual source commit and the exact [r7 protocol](../experiments/front-review-2026-11-19-r7.json). Both arms independently start from original epoch173, with seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW and learning rate 0.0001. Final-epoch selection is fixed before the run.

Original 664-front training validation and the fixed 666/663 scoring references remain unchanged. The current 663-front development review is separately versioned and exploratory, despite matching the historical count. This tests the accumulated eligible supervision package against the PR35-label control; it does not isolate one correction or directly compare r6 training against r7.

## Physical rule and remaining evidence limits

Use the [existing annotation policy](annotation-policy.md): visible front material and its own front rim define the box; actual lower-front corners define the points. Do not deliberately expand onto rack or cast shadow. An axis-aligned box can still enclose unavoidable background around a sloping or partly occluded front. An unlocatable corner is exactly `0 0 0`; a visible front can remain labeled with both points unavailable.

Descending gray projections in `new_000023`, pale fragments in `new_000017`, and the upper strip in `new_000114` remain explicitly held where originals do not resolve ownership. The next training run does not settle those physical questions. Any new reference amendment needs its own version and separate replay. Application-level acceptance and performance on new camera sessions remain unmeasured.
