# Current cassette workflow

Updated 18 November 2026, Asia/Hong_Kong.

**Use PR35 for reviewed image-space proposals. PR67 is evaluated and does not replace it.** The later label review is a separate artifact; it has not trained a new model.

## One selected command

Use branch `codex/pr67-and-r3-review-20261118`, which contains the evaluator and current review. Follow the [folder guide](predict-pose-folder.md#use-the-seed46-candidate-explicitly) to obtain the verified PR35 candidate bundle. Keep its manifest and companion files together.

```bash
.venv/bin/python scripts/predict_pose_folder.py \
  --checkpoint-manifest ../seed46-results/candidate/checkpoint-manifest.json \
  --input /path/to/original-images \
  --output /path/to/new-review-results \
  --device cpu --conf 0.13 --nms-iou 0.70 --endpoint-conf 0.50
```

The selected checkpoint is PR35 seed46, final epoch 30, SHA-256 `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1`. Input size is 1280. The generic runtime default remains epoch173 at confidence 0.25, so use the explicit manifest and command above.

Review the whole original image, including regions without proposals. The target is one visible cassette front and its two actual lower-front corners, ordered by image X. Each predicted endpoint is displayed independently when its score is at least 0.50 and its position is finite and in frame; both qualifying points are connected. A displayed estimate can be obscured in the image. Its confidence does not prove visibility, object identity, or pixel accuracy.

## Completed model comparison

The [PR67 report](../results/evaluation/sequence-fronts-pr67-20261118/README.md) contains fresh, same-runtime inference for all three models and both predefined references. Settings were fixed before scoring.

| Primary reference: 666 fronts | Missed | Unmatched proposals | Both corners within 5 px | Both within 10 px |
| --- | ---: | ---: | ---: | ---: |
| Retained PR35 | 18 | 80 | 533/584 | 582/584 |
| PR67 matched control | 20 | 83 | 531/584 | 583/584 |
| PR67 candidate | 25 | 105 | 532/584 | 584/584 |

The 584 denominator contains only fronts with two locatable reference corners. It excludes the partial/zero-corner fronts responsible for most scored misses. The candidate's extra 10 px success versus control comes from an existing proposal crossing matching IoU 0.50, not a newly discovered front. Several scored losses likewise involve box extent/ownership disagreements; others include a genuinely missing edge-front proposal and a new keyboard false proposal. The report preserves these distinctions.

The required historical 663-front comparison also retains PR35. A [separate exploratory replay](../results/evaluation/sequence-fronts-pr67-20261118/r3-exploratory/README.md) uses the recovered 662-front reference: PR35/control/candidate have 14/16/21 misses and 80/83/105 unmatched proposals. Each loses four scored misses because four previously unmatched reference rows were deleted. No matched physical identity is gained or lost by changing this reference.

These reused video frames support a development comparison. They do not establish unseen-session or unattended complete-scene accuracy. No new training run, threshold sweep, or model promotion follows from this report.

## Current label review

The [complete review bundle](../data/reviews/fronts-2026-11-18/README.md) contains 137 training-review files and 20 separately revised development-reference files. It preserves 3,789 training fronts and 662 development fronts, all 16 empty images, and the 11-field pose format.

The interrupted r3 geometry was reconstructed from the two saved vector PDFs, with independently checked before/current panels agreeing within 0.001 original-image pixel. This is rendering agreement, not physical annotation precision or recovery of the missing original authoring bytes. A separate original-pixel review additionally refines `new_000101` row 52 to `[485,543,550,571]`; both point tokens stay unchanged. The combined review has 889 replacements and 12 deletions from r2.

Only `new_000007` rows 7 and 14 retain recorded amber holds. Cyan means no recorded open hold, not certification. Review the updated change PDF and full atlas before merging this label revision. No training recipe currently selects it, and PR67 retains its original 664-front training validation plus its 666/663 predefined scoring references.

## Next bounded improvement

Inspect the named partial-front extent/ownership cases in the report before another fine-tune. A box may span an occluder if separated visible pieces belong to the same front; broad occluder coverage alone is not an annotation error. Preserve unlocatable natural-image corner slots as `0 0 0` and never turn a model estimate into observed truth.

Correcting a reference permits a separate saved-prediction replay without GPU training. Improving training labels requires a later, explicitly selected controlled experiment. Neither changes the fact that new-session generalization remains unmeasured. The current 10 px corner criterion and miss-first priority are established; the acceptable number of missed/extra fronts per scene remains an application acceptance decision.
