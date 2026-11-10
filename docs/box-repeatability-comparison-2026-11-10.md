# Reviewed boxes: the earlier detection gain did not repeat

**Decision, 2026-11-10: retain M0 epoch173, box confidence 0.25 and NMS IoU 0.70.** The two prospective seed pairs are complete and verified. At the primary setting, both corrected-label candidates miss one more reference object than their matched original-label controls. Both improve conditional p95 corner error slightly; extra detections and median error change in opposite directions between seeds. Neither candidate improves all five measures against its reference or M0.

**This study is complete; no GPU rerun is needed.** The next useful work is original-image review of persistent failures and training-label consistency. A [33-object review queue](../results/evaluation/baomai-box-repeatability-a100-001/failure-review-locations.json) preserves exact image IDs, original label rows, coordinates and model evidence. Begin with its 12 cases that have both reference corners locatable. These are review priorities, not newly accepted corrections.

## Question, inputs and fixed comparison

[PR #27](https://github.com/Siuuuers/dl/pull/27), pinned at [`265fe52`](https://github.com/Siuuuers/dl/commit/265fe52dc6e2d8721986fcfdf8944eb4ed33497f), returns the four GPU runs prepared in [PR #25](https://github.com/Siuuuers/dl/pull/25). The [pre-result study contract](https://github.com/Siuuuers/dl/blob/8ce9fe233dc3b4b9f3ccfc4491a952dca541a64b/docs/box-repeatability-2026-11-07.md) fixes seeds 43 and 44, paired original/corrected labels, all reported outcomes and descriptive interpretation without automatic promotion.

Every full run starts freshly from the same original epoch173 weights and completes 30 epochs. The numeric recipe, other than seed, is unchanged: batch 4, image size 1280, AdamW, nominal learning rate 0.0001, bias warmup learning rate 0.0001, AMP disabled and the existing loss/augmentation settings. The selected artifact is finalized epoch-30 `last.pt`.

Each arm uses 137 training images, 3,772 objects and 6,839 visible endpoints. The candidate changes the same 16 reviewed box edges in three training images, distributed 8/6/2 across `new_000030`, `new_000012` and `new_000128`. Object rows, classes and all endpoint tokens remain unchanged. The intervention retains the explicitly unresolved legacy annotation fields; it does not certify the whole dataset.

Fresh inference evaluates M0 and all four returned models in one common CPU runtime. The frozen reference contains 20 original images, 664 objects and 1,227 locatable endpoints; 584 objects have both corners locatable and three images are empty. Settings remain image size 1280, rectangular preprocessing, batch 1, FP32, ordinary NMS IoU 0.70, `max_det=300`, matching IoU 0.50 and endpoint confidence 0.50. Box confidence **0.25 is primary**; 0.15 and 0.40 are diagnostic. All 300 unique image/model/threshold records completed. M0 was inferred once and copied into the two seed evidence folders.

**Candidate versus its same-seed reference isolates the box-label intervention under the matched setup.** Candidate versus M0 combines additional training and changed labels. The larger differences from M0 cannot all be attributed to the corrected boxes.

## Primary measured results

Lower is better for the five measures below. Corner errors are conditional on a locatable reference endpoint being returned on a matched object. Missing endpoints are counted separately, including those lost through missed objects.

| Model | Missed objects | Extra detections | Missing visible endpoints | Median corner error, px | p95 corner error, px |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 epoch173 | 39 | 63 | 43 | 1.611790 | 5.161683 |
| Seed 43 original-label reference | 36 | 62 | 40 | 1.627241 | 5.229975 |
| Seed 43 corrected-label candidate | 37 | 61 | 41 | 1.636372 | 5.189023 |
| Seed 44 original-label reference | 38 | 69 | 41 | 1.663992 | 5.162338 |
| Seed 44 corrected-label candidate | 39 | 71 | 42 | 1.641763 | 5.137224 |

Neither new candidate passes the descriptive five-measure rule against either comparator. Seed 43 improves detection counts against M0 but worsens both conditional corner quantiles. Seed 44 ties M0's misses and improves missing endpoints/p95, but produces more extras and has a higher median. The study's decision remains `retain-M0`; no checkpoint is automatically eligible for adoption.

### All prospective paired changes

These are **candidate minus its matched reference**. Negative values are improvements. Means give the two seeds equal weight; no run is omitted or selected as a winner.

| Measure | Seed 43 | Seed 44 | Mean paired change | Range | Improved / tied / worsened |
| --- | ---: | ---: | ---: | --- | --- |
| Missed objects | +1 | +1 | +1.0 | +1 to +1 | 0 / 0 / 2 |
| Extra detections | -1 | +2 | +0.5 | -1 to +2 | 1 / 0 / 1 |
| Missing visible endpoints | +1 | +1 | +1.0 | +1 to +1 | 0 / 0 / 2 |
| Median error, px | +0.009131 | -0.022229 | -0.006549 | -0.022229 to +0.009131 | 1 / 0 / 1 |
| p95 error, px | -0.040953 | -0.025114 | -0.033033 | -0.040953 to -0.025114 | 2 / 0 / 0 |

The mean p95 change summarizes two seed-level percentile differences. It is **not a pooled endpoint p95**, a confidence interval or a significance result. The [comparison JSON](box-repeatability-comparison-2026-11-10.json) preserves every unrounded value and both seed outcomes.

### Historical seed 42 stays separate

The [completed seed-42 experiment](https://github.com/Siuuuers/dl/blob/5f2adb2a2d4aeecf87c29545406b50645e91ab5d/docs/box-supervision-comparison-2026-11-07.md) recorded candidate-minus-reference changes of -1 missed object, -1 extra detection, -1 missing endpoint, -0.007951 px median error and +0.092074 px p95 error. It improved detection and worsened the corner-error tail.

Both new seeds instead lose one match and improve conditional p95. **The earlier detection-gain/corner-tail pattern did not reproduce in these two specified pairs.** This shows that the single-seed result was not a stable basis for choosing a replacement under the tested conditions. It does not establish that every possible seed would behave this way or that the reviewed box corrections are physically wrong. Historical seed 42 is excluded from the prospective averages because its source and earlier session were not bound into this study.

## Why one object match disappeared in each pair

Each candidate retains all but one of its reference's matched objects and gains none. The lost object has one locatable reference corner, accounting for the additional missing endpoint. Both losses occur because confidence falls below the fixed primary gate:

| Seed | Frozen image / original label row | Reference confidence | Candidate confidence at diagnostic 0.15 | Reference IoU | Candidate IoU |
| --- | --- | ---: | ---: | ---: | ---: |
| 43 | `new_000058` / 19 | 0.252079 | 0.196345 | 0.651355 | 0.662684 |
| 44 | `new_000056` / 14 | 0.268810 | 0.214925 | 0.539100 | 0.541032 |

The candidate boxes match those reference rows at diagnostic confidence 0.15, with slightly higher overlap, but are absent at 0.25. These examples demonstrate a confidence-gate change rather than a failure to reach the matching-overlap criterion at 0.15. They do not establish exact physical box boundaries or justify changing the operating threshold after observing results.

### Diagnostic thresholds preserve the tradeoffs

| Seed | Box confidence | Reference FN / FP | Candidate FN / FP | Reference p95, px | Candidate p95, px |
| --- | ---: | --- | --- | ---: | ---: |
| 43 | 0.15 | 35 / 93 | 33 / 88 | 5.229432 | 5.250266 |
| 43 | **0.25** | **36 / 62** | **37 / 61** | **5.229975** | **5.189023** |
| 43 | 0.40 | 39 / 45 | 39 / 47 | 5.206477 | 5.193079 |
| 44 | 0.15 | 33 / 104 | 33 / 106 | 5.163622 | 5.176347 |
| 44 | **0.25** | **38 / 69** | **39 / 71** | **5.162338** | **5.137224** |
| 44 | 0.40 | 42 / 50 | 42 / 53 | 5.137132 | 5.107246 |

The favorable primary p95 direction is not uniform across the diagnostic settings. Those settings remain diagnostic; they do not replace the primary result or authorize an additional search for a favorable threshold.

## Corner improvements after controlling endpoint membership

Losing a high-error endpoint can lower a conditional percentile. Here the p95 improvement remains when both models are measured on exactly the same endpoint identities:

| Seed | Shared returned endpoints | Reference p95 on shared endpoints, px | Candidate p95 on shared endpoints, px | Endpoints better / worse |
| --- | ---: | ---: | ---: | --- |
| 43 | 1,186 | 5.220004 | 5.189023 | 587 / 599 |
| 44 | 1,185 | 5.154523 | 5.137224 | 640 / 545 |

The lost endpoint affects each conditional sample, but does not fully explain the p95 improvement. The mixed individual changes also show why a lower percentile does not mean every predicted corner improved. No coordinates are scored for the 101 reference-unlocatable endpoint slots.

The same-object diagnostic requires both corners to be returned within the stated radius. Its denominator includes all 584 objects with two locatable reference corners, including missed objects:

| Both corners within radius | M0 | Seed 43 reference | Seed 43 candidate | Seed 44 reference | Seed 44 candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 px | 263 | 273 | 269 | 278 | 283 |
| 5 px | 521 | 515 | 519 | 519 | 519 |
| 10 px | 568 | 570 | 571 | 570 | 570 |

At 5 px, seed 43 gains four object successes without losing any; seed 44 loses one and gains one. Both candidates remain at 519/584 versus M0's 521/584. These radii are illustrative and were not turned into application acceptance criteria.

All five models produce zero detections on the three empty images at all three thresholds. Every one of the 17 nonempty images still has at least one miss or extra detection for every model at confidence 0.25.

## Persistent failures give us a more useful next task

There are **33 identical reference-labelled objects missed by M0 and all four newly trained models** at the primary threshold. Across the models, 42 distinct reference objects are missed at least once. The persistent set contains nine objects with neither reference corner locatable, 12 with one and 12 with both.

For 30–32 of those 33 persistent misses per model, none of the returned predictions reaches matching IoU 0.50 even at diagnostic confidence 0.15. One persistent object has a qualifying box assigned to another reference object by the fixed matching procedure. Lowering confidence from 0.25 to 0.15 alone therefore cannot resolve most of this set.

The [review queue](../results/evaluation/baomai-box-repeatability-a100-001/failure-review-locations.json) contains every persistent row and the two candidate-only lost matches, with frozen source hashes and overlap evidence at each threshold. It is ordered by image ID and original label row. The visibility states describe the reference annotations; they are not a visual diagnosis of occlusion or physical cause.

The next bounded task is to inspect the original images for the 12 persistent cases with two locatable corners, then the remaining cases, and distinguish supported object identity, inconsistent visible-face box extent, matching competition and model error:

| Original validation image | Original one-based label rows to review first |
| --- | --- |
| `new_000054` | 17 |
| `new_000056` | 1, 4 |
| `new_000086` | 4, 15 |
| `new_000087` | 4 |
| `new_000088` | 4, 15 |
| `new_000089` | 4, 15 |
| `new_000090` | 4, 15 |

Inspect the image before treating either reference boxes or model outputs as truth. Use any supported findings to scope a small, prospective training-label review. Keep the completed evaluation reference frozen; any later reference revision needs its own version and comparison for all models.

The existing image viewer/editor and current model are sufficient for this task. This result supplies no evidence that a larger architecture, more epochs or another undirected seed search is the necessary next change. The 16 reviewed edges and all candidate artifacts remain available for later work.

## What was verified

The original return has **79,587,975 bytes**, SHA-256 `4e4d36aaee4bfe316a5517afa68a276e76eef07ed8680347bd9b52ffbe455beb`. Its 28 members have safe, unique paths and valid CRCs; all 27 payload hashes and sizes match the manifest. The four actual checkpoint files and original epoch173 bytes passed the existing identity/completion checks before inference.

Every full run records 30 completed epochs, 1,050 batches, 4,110 image presentations, 78 actual optimizer updates and exit code 0. Thirty finite CSV rows and the training/finalization logs corroborate completion. The four-batch resource trial is successful supplemental evidence and remains ineligible as a model.

All runs identify clean training source [`f087c35`](https://github.com/Siuuuers/dl/commit/f087c35f74bf304eb61998edcaa2172fe1cba49c), the same A100-SXM4-40GB runtime and matching non-seed settings. All seven recorded source hashes match the pinned Git blobs. The audit independently reconciled all 157 original images and labels, both label variants, the exact 8/6/2 changes, all five staging manifests and the remote-path YAML bytes reconstructed from pinned code.

Two provenance details matter. First, the trainer's terminal log validates its `best.pt`, while this study returns and evaluates finalized `last.pt`; terminal trainer metrics do not replace our fresh measurements. Second, none of the first four recorded batch samples in either seed contains a corrected image. The inventories and completed training are consistent, but this return does not directly sample a changed-label batch as the earlier seed-42 return did.

Fresh inference and the official saved-prediction summary ran from clean published evaluation source [`8ce9fe2`](https://github.com/Siuuuers/dl/commit/8ce9fe233dc3b4b9f3ccfc4491a952dca541a64b). The evaluator uses Python 3.12.14, Torch 2.10.0+cpu, torchvision 0.25.0+cpu, Ultralytics 8.4.182, NumPy 2.5.4, Pillow 13.0.0 and OpenCV 5.0.0.93. These recorded versions match the prior comparison. Fresh M0 reproduces all 60 historical prediction payloads exactly after excluding the two timing fields; this is output continuity, not a claim of identical training across studies.

An independent standard-library scorer separately parsed the labels and recomputed all 300 unique prediction/score records, 300 quality records, 15 scoring aggregates, 15 quality aggregates, paired diagnostics and study aggregation. Counts and match identities agree exactly. The largest floating arithmetic discrepancy is about 1.08e-14; the arithmetic verification allowance is not a new model-adoption tolerance.

The [verification record](../results/evaluation/baomai-box-repeatability-a100-001/verification.json), [evaluation capsule](../results/evaluation/baomai-box-repeatability-a100-001/evaluation.zip) and [checksum](../results/evaluation/baomai-box-repeatability-a100-001/evaluation.zip.sha256) retain raw predictions, quality records, small checkpoint bundles, audit scripts, commands, environment information, the summary and the failure queue. Returned weights stay in the original PR #27 archive. The capsule README gives the exact fresh-extraction and replay procedure.

```bash
python scripts/summarize_box_repeatability.py \
  --root /path/to/hydrated-dl \
  --evidence /path/to/extracted/evidence \
  --output /path/to/new-repeatability-summary.json
```

The replay uses the standard library and original validation PNGs/labels, without training, inference or model deserialization. The recorded Python version gives the closest floating-point reproduction.

## Scope of the conclusion

This is evidence from two specified new training seeds on a repeatedly used development split. It does not establish statistical significance, unseen-scene accuracy or a physical placement tolerance. Remote staged files and the GPU environment were not returned; recorded inventories were reconciled, not independently attested. The recorded GPU backend allows cuDNN TF32 and deterministic warning-only behavior, so bitwise GPU reproducibility is not claimed.

The supported workflow remains human-reviewed localization of supplied still images. Keep epoch173 as the working model, complete the targeted failure review, and define any next label intervention before another training experiment. The historical seed-42 decision, current thresholds and existing target definition remain intact.
