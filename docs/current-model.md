# Current cassette workflow

Updated 19 November 2026, Asia/Hong_Kong.

**Use PR35 for reviewed image-space proposals. PR70's r4 candidate does not replace it.** The round 5 annotations and next notebook are prepared separately; r5 has not been trained.

## One selected command

Use branch `codex/front-review-r5-20261119`, which contains the latest review and comparison. Follow the [folder guide](predict-pose-folder.md#use-the-seed46-candidate-explicitly) to obtain the verified PR35 candidate bundle. Keep its manifest and companion files together.

```bash
.venv/bin/python scripts/predict_pose_folder.py \
  --checkpoint-manifest ../seed46-results/candidate/checkpoint-manifest.json \
  --input /path/to/original-images \
  --output /path/to/new-review-results \
  --device cpu --conf 0.13 --nms-iou 0.70 --endpoint-conf 0.50
```

The selected checkpoint is PR35 seed46, final epoch 30, SHA-256 `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1`. Input size is 1280. The generic runtime default remains epoch173 at confidence 0.25, so use the explicit manifest and command above.

Review the whole original image, including regions without proposals. The target is one visible cassette front and its actual lower-front corners, ordered by image X. Each predicted endpoint is displayed independently when its score is at least 0.50 and its position is finite and in frame; two qualifying points are connected. A displayed estimate can be obscured in the image. Its confidence does not prove visibility, object identity or coordinate accuracy.

## Latest completed comparison: PR70

The [PR70 audit](../results/evaluation/pr70-20261119/README.md) verifies both completed L4 runs and compares fresh PR35, control and r4-candidate predictions in one CPU runtime. Training source, starting weights, 30 completed epochs, all original images and the fixed scoring references were checked. No settings were tuned after viewing the result.

| Primary reference: 666 fronts | Scored misses | Unmatched outputs | Both corners within 5 px | Both within 10 px |
| --- | ---: | ---: | ---: | ---: |
| Retained PR35 | 18 | 80 | 533/584 | 582/584 |
| Fresh PR70 control | 20 | 83 | 531/584 | 583/584 |
| PR70 r4 candidate | 30 | 107 | 541/584 | 580/584 |

The pair denominator includes the 584 fronts with two locatable reference corners, including scored misses. It excludes the 82 fronts with zero or one locatable corner. The candidate improves the five-pixel pair count while losing matched-front coverage and adding unmatched outputs. The required 663-front historical comparison has 15/17/26 misses and 80/83/106 unmatched outputs in the same model order.

A scored miss means no match at the fixed IoU of 0.50, not necessarily no prediction. In `new_000010:15` and `new_000057:11`, strong proposals remain but their box extents cross below the matching threshold. Other cases need physical extent/ownership review. The report also shows a visible missed edge front and additional keyboard/rack proposals. These distinctions prevent either a high corner score or a matching failure from becoming a misleading general claim.

All three models produce no outputs on the three empty development images, but none yields an error-free nonempty scene under this association rule. These repeatedly used scenes do not establish new-session reliability or unattended complete-scene accuracy. **No model is promoted.** The [previous PR67 audit](../results/evaluation/sequence-fronts-pr67-20261118/README.md) remains a separate historical comparison.

## Current annotation pass and next notebook

The [r5 handoff](front-review-r5.md) records **135 edited rows, two requested deletions and one added front across 17 photographs**. It preserves immutable r4 base-row identities and a map to current row numbers. The focused PDF shows every current label on those photos, with changed rows in magenta and unresolved rows in amber. The [labeling guide](front-labeling-guide.md) explains the physical corner, visible envelope and inspection-filter rules.

`new_000023` base rows 24-26 retain an explicit physical-landmark question: the farther gray tips belong to the cassette, but may be lateral/projecting material instead of the required lower-front junction. Its entire photo keeps exact PR35 supervision in both next-run arms. The two specifically requested deletions in `new_000014` close its previous hold; its supported r5 labels are now included in the treatment.

The [r5 notebook](../notebooks/baomai_front_review_r5.ipynb) compares the accepted review against the PR35 control using the existing numerical recipe. Both arms independently start from epoch173, with seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW and learning rate 0.0001. The final epoch is selected in advance. There are 94 training photos with different label bytes between the arms. Original training validation and the fixed 666/663 scoring references remain unchanged.

This tests the disclosed accumulated supervision package and whole-photo hold, not one isolated correction. It is ready for a controlled experiment, with `full_dataset_certified: false`. Preparing it does not establish improvement. Earlier [r3 recovery](../data/reviews/fronts-2026-11-18/README.md) and [r4 review](front-review-r4.md) remain separate historical records.

## What would change the conclusion

First resolve target-surface ownership in the named ambiguous cases. A box may span an occluder when its separated visible pieces belong to one front; interior gripper coverage alone does not make it wrong. Keep an unlocatable natural-image corner as `0 0 0` and do not use model confidence as observed truth.

The next run should be judged on missed and extra fronts, partial-front coverage, scene-level failures and corner accuracy together. Any later reference amendment must receive a separate identity and replay, preserving the fixed comparisons above. The current ten-pixel corner criterion and miss-first priority are established; acceptable missed/extra fronts per scene still depend on the application. New-session performance remains unmeasured.
