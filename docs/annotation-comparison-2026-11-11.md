# Reviewed annotations: completed seed45 comparison

**The corrected-label model has a small favorable detection result against its matched control at confidence 0.25: one fewer missed front and three fewer extra detections. It ties the control at 10 px both-corner success and loses one success at 5 px. Keep epoch173 as the working default and run the prospectively reserved seed46 confirmation with the same annotations.**

Report date: **2026-11-11, Asia/Hong_Kong**. Execution records use UTC on 2026-11-10. This report completes the [seed45 training handoff](training-annotations-2026-11-10.md) using the actual return in [PR32](https://github.com/Siuuuers/dl/pull/32). It makes no automatic promotion or independent-test claim.

## What was compared

| Model | Meaning |
| --- | --- |
| M0 | Original epoch173, freshly evaluated after both returned runs finished |
| M1 | Original-label control, independently fine-tuned from epoch173 for 30 epochs with seed45 |
| M2 | Reviewed-annotation candidate, independently fine-tuned from epoch173 with the same recipe and seed45 |

M2 versus M1 measures the annotation intervention within this matched pair. M2 versus M0 includes the effect of further training as well as the changed labels. These comparisons answer different questions.

The candidate remains [annotations-2026-11-10](../data/candidates/annotations-2026-11-10/manifest.json): five added fronts, six removed rows, three retained-row amendments and two exact corner transfers across eight training images. Its existing 6,839 visible corner occurrences were preserved. Neither evaluation reference was changed after seeing these predictions.

Primary scoring uses box confidence **0.25**, NMS IoU**0.70**, object-matching IoU**0.50** and endpoint confidence **0.50**. All three models use the same 1280 rectangular FP32 inference settings, batch 1 and `max_det=300`. Confidence0.15 and0.40 are fixed diagnostics. For an eligible object, **both corresponding lower corners must each be within the stated Euclidean radius in the original 1280×720 image**. Missed eligible objects remain in the denominator.

## Primary results: PR30 revised reference

This is the current development interpretation: **20 images, 661 objects, 1,228 visible endpoints and 584 objects with both corners locatable**. Three images are empty.

| Measure at confidence 0.25 | M0: epoch173 | M1: control | M2: corrected labels |
| --- | ---: | ---: | ---: |
| Missed objects, lower is better | 27 | 25 | **24** |
| Extra detections, lower is better | 54 | 58 | **55** |
| Missing visible endpoints, lower is better | 26 | 22 | **21** |
| Conditional median corner error, px | 1.6175 | 1.6410 | 1.6321 |
| Conditional 95th-percentile error, px | 5.2107 | 5.1841 | 5.1633 |
| Both corners within 5 px | 528/584 | 531/584 | **530/584** |
| Both corners within 10 px | 575/584 | 579/584 | **579/584** |
| Extra detections on the three empty images | 0 | 0 | 0 |

**M2 versus M1:** one fewer miss, three fewer extras and one fewer missing visible endpoint. At5px, one eligible object changes from success to failure. At10px, exactly the same 579 object identities succeed in both models; there are no gained or lost 10 px successes.

**M2 versus M0:** three fewer misses, one additional extra detection and five fewer missing visible endpoints. Both-corner success increases by two objects at 5 px and four at 10 px. However, M1 also gains those same four 10 px successes over M0. This experiment therefore does not attribute that 10 px gain specifically to the corrected labels. M2's conditional median error is slightly higher than M0's.

The full values, denominators and all fixed thresholds are in the [condensed result JSON](annotation-comparison-2026-11-11.json). The complete unrounded per-image result is preserved in the [evidence capsule](../results/evaluation/baomai-annotations-seed45-20261111/evaluation.zip).

## Why the gain needs confirmation

### One recovered control miss crosses the confidence threshold

The only candidate-versus-control detection gain at 0.25 is **[new_000056](https://github.com/Siuuuers/dl/blob/ed4ee162738685bff93d6cad1402d95e128bceb4/data/images/val/new_000056.png), row14** on both references. It is the observed left part of a lower front beside the hand. One lower corner is locatable; the right side is hidden.

| Model | Matching detection confidence | Box IoU with revised row14 | Matched at 0.25? |
| --- | ---: | ---: | --- |
| M0 | 0.774575 | 0.512360 | Yes |
| M1 | 0.193755 | 0.522489 | No |
| M2 | 0.288164 | 0.519537 | Yes |

M1 and M2 both match it at 0.15. At0.25, only M2 passes the confidence cutoff. M0 already detects this object. This is a real gain relative to the matched control at the declared operating point, but it is not a newly recovered baseline miss or an improvement in this box's IoU. No control-detected object is lost at 0.25.

The reference box is approximately `[418.88,648.88,528.58,672.00]`; its visible left corner is `(418.88,665.54)`. These are reference coordinates, not new edits.

### The detection difference depends on the operating threshold

| Box confidence | M0 misses / extras | M1 misses / extras | M2 misses / extras | M2−M1 misses |
| --- | ---: | ---: | ---: | ---: |
| 0.15, diagnostic | 22 /82 | 21 /84 | 21 /83 | 0 |
| **0.25, primary** | **27 /54** | **25 /58** | **24 /55** | **−1** |
| 0.40, diagnostic | 30 /35 | 28 /37 | 29 /34 | +1 |

At0.40, the candidate loses **[new_000008](https://github.com/Siuuuers/dl/blob/ed4ee162738685bff93d6cad1402d95e128bceb4/data/images/val/new_000008.png), row34**, a front clipped by the right image boundary: control confidence is about 0.400510, candidate confidence about 0.389357, and both boxes have IoU about 0.579. Again the transition is around the confidence boundary.

Lowering M2 from 0.25 to 0.15 would recover three more targets while adding 28 extras on these images. That tradeoff is visible in the table; it does not justify choosing a new threshold after inspecting this result. Keep 0.25 primary for the confirmation. The user prioritizes misses, but has not specified an unlimited or numerical extra-detection budget.

At0.25, candidate-control extra counts decrease by one each in images 007, 054, 058 and 087, and increase by one in 090: net−3. These are reference-relative unmatched detections. A changed unmatched count alone does not establish the physical correctness of every affected box.

### One corner moves just outside the stricter5px radius

The sole lost 5 px pair is **[new_000055](https://github.com/Siuuuers/dl/blob/ed4ee162738685bff93d6cad1402d95e128bceb4/data/images/val/new_000055.png), row20**, a front clipped at the top of the image. Its left/right errors change from **2.01978 / 4.91668 px** in M1 to **2.16888 / 5.02027 px** in M2. Both corners remain within 10 px. The radius transition is small, but it remains recorded as a loss under the fixed 5 px diagnostic.

The marginal median and 95th-percentile errors are slightly lower for M2 than M1. That does not establish general localization improvement. On the **same 1,206 returned revised-reference endpoints**, the candidate's mean error is **0.001822px higher**:626 points worsen and 580 improve. The differences are tiny and mixed; the same-point analysis prevents a change in the returned population from being mistaken for uniform improvement.

Original full-resolution pixels were inspected for these three cases. No label, threshold or prediction was changed as a result of this inspection. The capsule contains the numerical case analysis and the original-image hash record.

## Remaining misses matter more than the attractive 10 px fraction

| Locatable reference corners | Reference objects | M0 misses | M1 misses | M2 misses |
| --- | ---: | ---: | ---: | ---: |
| Zero | 17 | 8 | 8 | 8 |
| One | 60 | 12 | 12 | 11 |
| Two | 584 | 7 | 5 | 5 |
| **All** | **661** | **27** | **25** | **24** |

**Nineteen of M2's 24 misses concern objects with fewer than two locatable corners.** They are valid detection targets under the visible-front policy, but they are outside the 584-object both-corner denominator. Therefore **579/584≈99.1% is a conditional two-corner result, not overall system accuracy**. The remaining difficult detection population includes occluded and clipped fronts; further original-pixel annotation review should focus there after the fixed confirmation, without changing this candidate midway.

M2 returns coordinates on 61 of 67 reference-unlocatable sides belonging to matched objects; M1 returns 60 of 66 and M0 returns 61 of 66. On the 66 shared matched hidden sides, both M1 and M2 return 60 outputs; M2's additional output is the newly matched image 056 row14 right side. The 60/66→61/67 change therefore reflects matched-object membership. This is availability disagreement where no visible reference coordinate exists. It cannot measure hidden-corner accuracy or prove that a hidden physical corner was observed. Endpoint confidence is not a reliable visibility guarantee in these examples.

The both-corner result also leaves out single-corner accuracy. For example, the visible left corner of image008 row34 has error **11.708 px** in M2, **11.799 px** in M1 and **3.087 px** in M0 at the primary threshold. This baseline-relative regression remains relevant even though the object is excluded from the two-corner denominator. The full visible-endpoint and same-point diagnostics preserve it.

The 10 px ceiling remains a provisional image-space diagnostic. It is not annotation slack or a calibrated physical tolerance. Supplied-image localization remains the task; no robot or camera-calibration change is required to complete this comparison.

## Original reference retained separately

The original frozen reference has **664 objects, 1,227 visible endpoints and 584 both-corner-eligible objects** on the same 20 images. These results preserve historical comparability.

| Measure at confidence 0.25 | M0 | M1 | M2 |
| --- | ---: | ---: | ---: |
| Missed objects | 39 | 36 | 35 |
| Extra detections | 63 | 66 | 63 |
| Missing visible endpoints | 43 | 37 | 36 |
| Conditional median error, px | 1.6118 | 1.6398 | 1.6300 |
| Conditional 95th-percentile error, px | 5.1617 | 5.1928 | 5.1808 |
| Both corners within 5 px | 521/584 | 525/584 | 524/584 |
| Both corners within 10 px | 568/584 | 573/584 | 573/584 |

The primary candidate-control detection difference has the same direction and magnitude on both references. Candidate-versus-baseline differs: the original reference gives four fewer misses with unchanged extras, whereas the revised reference gives three fewer misses and one extra. Changing a reference changes what counts as a match; that is a measurement effect, not additional model learning. Historical results remain unchanged.

## Returned-run and evaluation verification

PR32 head: [`074f4d28736fab8b82baf7e56e6ad551997ae3c6`](https://github.com/Siuuuers/dl/tree/074f4d28736fab8b82baf7e56e6ad551997ae3c6). Its 39,774,041-byte ZIP has SHA-256 `598c6e1177401690e0057fcaafa260a41e798b216d1c59309b652a94d518293e`. All 13 unique regular members, the twelve indexed payload hashes/sizes, archive CRC, checkpoint manifests, receipts and staged inventories were checked.

| Recorded property | Both training arms |
| --- | --- |
| GPU | NVIDIA L4 |
| Runtime | Python 3.13.16, Torch 2.14.1+cu130, CUDA 13.0, torchvision 0.29.1+cu130, Ultralytics 8.4.182 |
| Training source | Clean `1c5d2e48ab26c578c5f6660010d6ecfb33231eb8` |
| Recipe | `reviewed-annotations-seed45-v1`, SHA-256 `34b8eda9e9b986a2ba19d9ba07661eef1792027ef8d4465d10763a8eb2cd2850` |
| Completion | 30 epochs, 1,050 batches, 4,110 image presentations and 78 actual optimizer updates per arm |
| Checkpoint selection | Finalized `fit/weights/last.pt` from each completed run |
| Exit / budget | Successful / no budget stop |

The protocol did not require an A100. Both L4 runs use the same recorded runtime and settings, so the hardware name is not a reason to repeat seed45. Monitored run wall times were 463.460 s and 456.203 s, approximately 15 min 20 s in total; this excludes download and the separate resource trial and is not a promise for a future session. The separate trial receipt/log was not returned, so that trial was not independently reconstructed.

The loader provides a direct supervision witness: the second recorded batch contains the same images, including `new_000072`, while its object count changes 82→80, consistent with the two reviewed deletions. The first batch has matching image IDs, count and loss; losses diverge after changed supervision reaches the loader.

| Model | Evaluated checkpoint SHA-256 |
| --- | --- |
| M0 | `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e` |
| M1 | `111eb3edd8723fd7eccf5e9392e414cbefc64cf4e0dc2191a508d2f6bfa6c0cf` |
| M2 | `47cb4fc84260fdc0fa0c707e046dfb57e85702b63ce1a1798181a110884492f1` |

The trainer's last log section validates `best.pt`; this report instead evaluates the prescribed final `last.pt` files. Trainer mAP or loss is not substituted for this comparison.

All three fresh model runs used clean evaluation revision [`ed4ee162738685bff93d6cad1402d95e128bceb4`](https://github.com/Siuuuers/dl/tree/ed4ee162738685bff93d6cad1402d95e128bceb4), whose relevant training-script bytes match the training source. Inference used Python 3.12.14, Torch 2.10.0+cpu and the same Ultralytics 8.4.182 for all three models. Recorded CPU timings are not L4 deployment latency.

There are **180 unique image/model/threshold prediction records**, scored against two references to produce 360 per-image reference evaluations and 18 aggregates. A standalone standard-library implementation reproduced every match/count/identity and the reported common-endpoint comparisons; the maximum numerical difference was 1.11×10⁻¹⁴. Fresh M0 predictions match all 60 historical M0 record payloads exactly after excluding only timing fields; the fresh execution headers remain separate.

The [capsule verification](../results/evaluation/baomai-annotations-seed45-20261111/verification.json) records safe extraction, complete payload hashes and a fresh saved-prediction replay from the pinned clean source. The [capsule README](../results/evaluation/baomai-annotations-seed45-20261111/README.md) gives the replay command and exact path/timestamp comparison rules. It contains predictions, detailed scores, quality records, receipts, inventories, logs and independent audit code. The original checkpoint bytes remain in PR32.

These checks establish consistency and reproducibility of the recorded experiment. They do not certify all annotation identities, attest remote hardware independently or turn 20 repeatedly used development images into a new independent test set. Related frames and corners are correlated. Earlier box-only seeds 42–44 test a different intervention and are not pooled with seed45.

## Concrete next step

Use the new [seed46 confirmation notebook](../notebooks/baomai_reviewed_annotations_seed46.ipynb) and [short handoff](training-annotations-seed46.md). Seed46 was reserved in the prospective seed45 plan. It is one matched confirmation pair, with the same candidate hash, original epoch173 initialization, final-epoch selection and numerical settings except the seed. Both arms start from epoch173 independently. Prefer the same L4/runtime when available and record any difference.

After return, freshly evaluate M0 and both seed46 checkpoints under their matching pinned source. Report both references, primary 0.25 and the existing diagnostics. Keep seed45 and seed46 separately visible. A repeat favorable miss result with an acceptable companion-metric tradeoff would strengthen the case for adoption; disagreement would weaken it and should lead back to targeted supervision review. If the confirmed candidate still adds extras relative to M0, agree the acceptable extra-detection tradeoff before changing the default. Two favorable seeds would still not establish unseen-scene accuracy.

**Current state:** seed45 evaluation completed; epoch173 remains the working default; seed46 is prepared, not executed. No label changes, threshold changes or automatic promotion are part of this result.
