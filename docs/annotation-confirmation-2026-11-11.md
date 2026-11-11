# Reviewed annotations: completed seed46 confirmation

**At the fixed confidence of 0.25, the seed46 candidate ties its matched control at 24 missed fronts and reduces extras from 58 to 52 on the PR30 revised reference. The seed45 annotation-specific miss advantage does not repeat as a strict reduction. The primary extra-detection reduction is favorable in both experiments.**

The candidate also improves the chosen primary headline totals over epoch173: three fewer misses, two fewer extras, four fewer missing visible corners, and higher 5/10 px both-corner success. This is a useful experimental checkpoint. Keep epoch173 as the working default while the remaining cases receive focused review; do not dismiss the candidate or start another seed search.

Report date: **2026-11-11, Asia/Hong_Kong**. This completes the [seed46 handoff](training-annotations-seed46.md) using the returned archive in [PR35](https://github.com/Siuuuers/dl/pull/35). The [seed45 report](annotation-comparison-2026-11-11.md) remains unchanged historical evidence. Neither model promotion nor a merge is part of this report.

## Primary comparison

M0 is freshly inferred original epoch173. M1 is the original-label control; M2 is the reviewed-annotation candidate. Each trained arm starts independently from epoch173 and completes the same 30-epoch recipe with seed46. M2 versus M1 evaluates the annotation intervention within the pair; M2 versus M0 also includes additional training.

The primary box confidence is **0.25**, NMS IoU **0.70**, matching IoU **0.50**, and endpoint confidence **0.50**. All three use the same 1280 rectangular FP32 inference, batch 1 and `max_det=300`. Confidence 0.15 and 0.40 remain fixed diagnostics. For a both-corner success, each corresponding lower corner must lie within the given Euclidean distance in the original 1280×720 image. The denominator retains all eligible objects, including missed objects.

The revised reference has **20 images, 661 objects, 1,228 locatable corners and 584 objects with both corners locatable**. Three images are empty.

| Primary measure | M0: epoch173 | M1: control | M2: reviewed annotations |
| --- | ---: | ---: | ---: |
| Missed objects | 27 | 24 | **24** |
| Extra detections | 54 | 58 | **52** |
| Missing visible endpoints | 26 | 23 | **22** |
| Conditional median corner error, px | 1.6175 | 1.5730 | **1.5622** |
| Conditional 95th-percentile error, px | 5.2107 | 5.0924 | **5.1207** |
| Both corners within 5 px | 528/584 | 527/584 | **529/584** |
| Both corners within 10 px | 575/584 | 578/584 | **577/584** |

Against M1, M2 gains five and loses three individual 5 px successes, net +2. It loses one 10 px success and gains none, net −1. The median is lower but the 95th-percentile error is higher. On 1,205 identical returned revised-reference endpoints, the mean candidate-minus-control error is **+0.016158 px**, with 602 points worse and 603 better. A lower marginal median does not mean uniform localization improvement.

Against M0, M2 improves all seven measures displayed in this table. That favorable baseline comparison must remain visible. It does not cover every diagnostic: the 2 px both-corner result decreases from 265 to 261, common-returned mean error increases slightly, and the 10 px net gain comprises **three gains and one loss**. The three empty images have zero extras from all three models at the primary threshold.

The [condensed JSON](annotation-confirmation-2026-11-11.json) preserves both seeds and both references separately. Complete unrounded scores and per-image details are in the [evidence capsule](../results/evaluation/baomai-annotations-seed46-20261111/evaluation.zip).

## What repeated across the two experiments

These are candidate-minus-matched-control changes at confidence 0.25 on the same revised development reference. Negative miss/extra/missing-corner changes are favorable; positive success-count changes are favorable.

| Experiment | Training GPU | Misses | Extras | Missing visible corners | Both ≤5 px | Both ≤10 px |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Seed45 | L4 | −1 | −3 | −1 | −1 | 0 |
| Seed46 | A100 | 0 | −6 | −1 | +2 | −1 |

The extra-detection reduction repeats at the declared primary operating point. The annotation-specific miss reduction does not repeat as a strict gain. Corner changes remain mixed. Both candidate models have 24 revised-reference misses versus M0's 27, but the original-label controls also reduce misses after additional training. The comparison does not establish the label edits as the cause of all baseline-relative improvement.

The training GPU changed from **NVIDIA L4** in seed45 to **NVIDIA A100-SXM4-40GB** in seed46. Both arms within each experiment share their recorded runtime. Between experiments, the recorded runtime fields differ in GPU name and capability; numerical training settings differ in seed. Therefore cross-experiment variation cannot be attributed solely to the random seed. These are two matched label comparisons under their recorded environments, not a pure seed-only reproducibility experiment.

No result is averaged into a larger supposedly independent test set. The same 20 development images remain the same observations. Earlier box-only experiments are a different intervention and are not pooled here.

## The equal miss counts hide one gain and one loss

### A confidence-boundary loss: image 008, row 1

For the top-clipped front in [new_000008](https://github.com/Siuuuers/dl/blob/e4bb7019bcffd80b08cab28c4877dd562ee1d796/data/images/val/new_000008.png), row 1 on both references, the control's relevant confidence is **0.264444** and the candidate's is **0.232634**. Both match at 0.15; only the control passes 0.25. Its visible reference corner is not returned by either model, so the lost object match does not itself remove a returned visible corner.

### An overlap-boundary gain: image 090, revised row 43

For [new_000090](https://github.com/Siuuuers/dl/blob/e4bb7019bcffd80b08cab28c4877dd562ee1d796/data/images/val/new_000090.png), revised row 43, relevant confidence is high in both models: **0.76069→0.77012**. Box IoU changes from **0.499094 to 0.502481**, crossing the fixed 0.50 matching requirement.

PR30 shortened this reference box from a bottom coordinate of 717.12 to 700.00 while preserving its visible left corner. On the original box, all three models remain unmatched. That is why revised misses tie 24/24 while original misses are 35/36. The candidate's gain is valid under the declared revised-reference rule, but it is sensitive to that narrow overlap boundary. Neither the reference nor the matching threshold is changed after seeing the result.

### A baseline-relative miss can still have a strong nearby prediction

Both trained models lose M0's match for [new_000087](https://github.com/Siuuuers/dl/blob/e4bb7019bcffd80b08cab28c4877dd562ee1d796/data/images/val/new_000087.png), row 15, a narrow front in the upper-right stack. M0/control/candidate confidence is **0.9356/0.9163/0.9243**, while IoU is **0.528262/0.496214/0.493069**. The reference box begins at approximately y=121; the model proposals extend upward to approximately y=105.

The candidate's nearby proposal has corners about **3.162 and 1.179 px** from the reference. These are diagnostic distances from an **unmatched proposal**, not official matched-corner successes. The official miss remains. The example shows why a scored miss should not automatically be described as the network producing no useful front proposal. Lowering the confidence cutoff does not address this particular overlap failure.

M2 gains four revised-reference object matches and loses this one relative to M0, giving the net reduction of three misses.

## The 10 px loss involves competing proposals

The only candidate-versus-control 10 px pair loss is [new_000007](https://github.com/Siuuuers/dl/blob/e4bb7019bcffd80b08cab28c4877dd562ee1d796/data/images/val/new_000007.png), row 15, at the top image boundary. The control's officially matched corner errors are **5.760/1.508 px**; the candidate's are **14.596/14.356 px**.

Two candidate predictions survive the fixed inference procedure:

| Candidate proposal | Confidence | IoU with row 15 | Corner errors against row 15 | Official assignment |
| --- | ---: | ---: | --- | --- |
| Prediction 52 | 0.304644 | 0.589224 | 14.596/14.356 px | Matched to row 15 |
| Prediction 53 | 0.300398 | 0.850809 | 4.757/3.928 px | Unmatched extra |

Prediction indices here are zero-based within this saved image/threshold record. The fixed matcher processes confidence order; the slightly higher-confidence proposal takes the match. The higher-IoU proposal is not substituted into the official result. This case deserves a review of duplicate predictions, confidence ranking, target identity and box extent before concluding that the candidate cannot produce accurate corners or deciding to change the architecture.

The table is a diagnosis, not an alternative favorable score. No matcher, NMS, confidence threshold or corner reference is altered. Original pixels for 007/008/087/090 were inspected; the case records retain their hashes and the independent scoring traces.

There is also a single-corner issue outside the both-corner denominator: image 008 row 35 has a candidate left-corner error of **11.750 px**, compared with 11.589 px for control and 7.353 px for M0. The 10 px ceiling is therefore not a guarantee for every visible point.

## Remaining misses and the concrete review queue

| Locatable reference corners | Reference objects | M0 misses | Control misses | Candidate misses |
| --- | ---: | ---: | ---: | ---: |
| Zero | 17 | 8 | 8 | 8 |
| One | 60 | 12 | 10 | 10 |
| Two | 584 | 7 | 6 | 6 |
| **Total** | **661** | **27** | **24** | **24** |

Eighteen candidate misses are outside the two-corner denominator. **577/584≈98.8% is an eligible-two-corner success fraction, not overall system accuracy.** Hidden fronts and clipped fronts remain valid object targets when their observed identity satisfies the policy.

The [readable 26-object review queue](../results/evaluation/baomai-annotations-seed46-20261111/review-queue.md) and its [complete JSON](../results/evaluation/baomai-annotations-seed46-20261111/review-queue.json) contain every current candidate miss, the image 007 corner diagnostic, and the image 090 gained match. The image 008 loss is attached to its existing miss row. Each entry maps the PR30 row to the original source row and records misses for M0 plus both control/candidate pairs.

Of the 24 current misses, **21 are missed by all five compared model variants**. That is descriptive persistence, not five independent confirmations. Their earlier reference-review status is:

- **19** retain an explicitly unresolved disposition from the PR30 review; 18 of those are in the persistent group.
- **Two** were amended in PR30: image 008 row 33 and image 009 row 57.
- **Three** were outside the earlier 33-row queue: image 009 row 52, image 011 row 61 and image 087 row 15.

Repeated model disagreement does not resolve uncertain labels. Preserve the 19 unresolved cases without inventing hidden corners or editing labels to match predictions. Start the next focused pass with the five other current misses and the 007 competing-proposal case, using the original pixels and existing annotation policy. The two amended rows need diagnosis rather than an assumption that their reviewed labels are wrong. Keep 090 as a boundary-sensitive measurement example.

At the primary threshold, both control and candidate return 63 coordinates on the same 67 shared matched hidden sides. Their aggregate 63/68→64/68 change reflects membership: the control-only hidden side on 008 row 1 is not returned, while the candidate-only hidden side on 090 row 43 is returned. There is no visible reference coordinate with which to measure those hidden-side errors.

## Fixed-threshold diagnostics

| Box confidence | M0 misses/extras | Control misses/extras | Candidate misses/extras | Candidate−control misses/extras |
| --- | ---: | ---: | ---: | ---: |
| 0.15 | 22/82 | 22/77 | 21/78 | −1/+1 |
| **0.25, primary** | **27/54** | **24/58** | **24/52** | **0/−6** |
| 0.40 | 30/35 | 29/36 | 29/38 | 0/+2 |

The primary extra reduction is not uniform across the diagnostic thresholds. Lowering the candidate from 0.25 to 0.15 yields three fewer misses and 26 additional extras on these images. Preserve that tradeoff rather than changing the chosen operating point retrospectively. Local extra counts can also move against the aggregate: image 089 has four more candidate extras than M0. The exact saved unmatched predictions remain available for review.

## Original reference retained separately

The original reference has 664 objects and 1,227 visible corners on the same 20 images. Its both-corner denominator is also 584.

| Measure at 0.25 | M0 | Control | Candidate |
| --- | ---: | ---: | ---: |
| Missed objects | 39 | 35 | 36 |
| Extra detections | 63 | 66 | 61 |
| Missing visible endpoints | 43 | 39 | 39 |
| Conditional median error, px | 1.6118 | 1.5671 | 1.5561 |
| Conditional 95th-percentile error, px | 5.1617 | 5.0973 | 5.1306 |
| Both ≤5 px | 521/584 | 520/584 | 522/584 |
| Both ≤10 px | 568/584 | 571/584 | 570/584 |

Here the candidate has one more miss and five fewer extras than its control. Its comparison with M0 is favorable on all seven displayed measures. Both reference versions are retained, and the report never presents a changed reference as additional model learning.

## Return and execution verification

PR35 head is [`8e8bb99cabcb44ed0ec0acd14db911aa44013fd8`](https://github.com/Siuuuers/dl/tree/8e8bb99cabcb44ed0ec0acd14db911aa44013fd8). The 39,776,456-byte return ZIP has SHA-256 `df53683ccf6f266a605b2844f42a98e808f75686cfab09f14d680c9d061e8a91`. All 13 unique safe members, twelve indexed payload hashes/sizes and CRC verified. Both final checkpoint files are 21,704,814 bytes.

| Checkpoint | SHA-256 |
| --- | --- |
| M0 epoch173 | `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e` |
| Seed46 control | `ad99cff1bd8f252f7851fec371a9d5878244ca66a3b1865e98ae16687c22ec33` |
| Seed46 candidate | `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` |

Both receipts and raw logs identify **A100-SXM4-40GB**, Python 3.13.16, Torch 2.14.1+cu130, CUDA 13.0, torchvision 0.29.1+cu130 and Ultralytics 8.4.182. Each records 30 epochs, 1,050 batches, 4,110 image presentations and 78 actual optimizer updates; successful exit, no budget stop, parameter changes, completed finalization and input reverification. Both have `resume=false` and initialize from original epoch173. Monitored run wall times are 388.002 s and 388.011 s, about 12 min 56 s combined, excluding setup and the separate resource trial.

The exact training source is clean `23d7db97ca626f5591dbd26e699c0f23d318ee73`; candidate and recipe hashes match the handoff. All 157 original PNGs, original checkpoint bytes, candidate files and four recorded staged validation views verify. The selected loader counts are 3,772 objects/6,839 visible corners for control and 3,771/6,841 for the candidate.

[PR34](https://github.com/Siuuuers/dl/pull/34) changes the notebook to save and restore completed runs through Drive. Exact notebook-source comparison confirms that the training command, recipe, source pin, candidate hash, input checks and completion assertions remain unchanged. A completed restored arm returns before a new training subprocess starts. The returned receipts therefore describe the A100 training; the later L4 packaging session does not relabel it as L4 training. The Drive transport and packaging host are not independently attested by the returned files. Separate resource-trial records and prior failed attempts were not returned.

The prescribed finalized `last.pt` checkpoints were evaluated. The raw trainer logs end with validation of `best.pt`; those scores are not substituted for this comparison.

Fresh M0/M1/M2 inference uses clean [`e4bb7019bcffd80b08cab28c4877dd562ee1d796`](https://github.com/Siuuuers/dl/tree/e4bb7019bcffd80b08cab28c4877dd562ee1d796), with matching training-script bytes. The restored CPU runtime matches the earlier recorded core versions: Python 3.12.14, Torch 2.10.0+cpu, torchvision 0.25.0+cpu, Ultralytics 8.4.182, NumPy 2.5.4, Pillow 13.0.0 and OpenCV 5.0.0.93. CPU timings are not GPU deployment latency.

All seven evaluation/diagnostic/summary commands succeeded. **180 unique prediction records** produce **360 per-image reference evaluations and 18 aggregates**. The independent standard-library scorer agrees on every count and identity, with maximum floating-point difference 1.10×10⁻¹⁴.

Fresh M0 is not bitwise identical to the saved seed45 M0, despite matching recorded runtime headers. Maximum coordinate change is 0.0001221 px and maximum keypoint displacement is 0.0001365 px. Detection counts, original match identities, endpoint availability and radius-success counts remain unchanged; error quantiles differ slightly. No cause is asserted, and the new report preserves the actual fresh values.

The [capsule verification](../results/evaluation/baomai-annotations-seed46-20261111/verification.json) records complete payload readback and fresh extraction/replay from the exact clean source. The [README](../results/evaluation/baomai-annotations-seed46-20261111/README.md) gives the portable command and the precise timestamp/invocation/root-path normalization rules. The capsule includes raw predictions, unrounded results, small checkpoint evidence bundles, independent audit/scoring code, runtime records and the review queue. Original images and weights remain in their existing repository sources and PR35.

## Decision and next work

The planned confirmation is complete. **Keep the corrected annotations and both experimental results; retain epoch173 as the current default and seed46 M2 as a useful experimental candidate.** The absence of a repeated annotation-specific miss advantage does not erase the candidate's favorable baseline totals. Those totals also do not resolve the concrete competing-proposal and box-extent cases or establish unseen-scene accuracy.

The next bounded task is the original-pixel and supervision review identified in the 26-object queue, with the five current misses outside the unresolved group and the 007 corner case first. Separate actual missing proposals, low-confidence proposals, reference/box-overlap failures and duplicate/assignment competition. Use the existing policy and editor where useful; record unresolved evidence instead of guessing.

**No seed47 sweep, new training recipe, threshold retuning, architecture change or automatic promotion is proposed.** A further training experiment should follow a new, visually supported supervision change or new evidence, rather than repeat this unchanged intervention hoping for a better seed.
