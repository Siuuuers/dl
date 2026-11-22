# PR92 r18 result audit

**Keep PR35 selected.** The returned r18 candidate has more misses and unmatched outputs than the fresh control on both frozen references. It agrees better with the separately pinned r18 exploratory review, but that favorable result does not replace the fixed selection rule. Against the preceding r17 candidate on identical r18 labels, it loses one match and gains two five-pixel corner-pair successes; ten-pixel successes are unchanged. Training hardware also changed from L4 to A100, so this is not an isolated test of the latest annotation amendments.

This audit uses the references pinned **before PR92 returned**. No r19 correction enters any score. [Summary](summary.json), [full fixed comparison](fixed-comparison.json), [return integrity](verified-return.json), [reference/model separation](reference-and-generation-comparison.json).

## Verified return and inputs

[PR92](https://github.com/Siuuuers/dl/pull/92) is pinned to head `f794bdd3d2842771e2cc29c624e8c2f26dba2ef8`. The actual [committed results archive](https://github.com/Siuuuers/dl/blob/f794bdd3d2842771e2cc29c624e8c2f26dba2ef8/results/gpu/baomai-front-review-r18-a100-001/baomai-front-review-r18-results.zip) is **39,783,562 bytes**, SHA-256 `167782e60b0a24d5ba041222a669aff2cb782ef684b3f64c7106cf9bcbb18445`. Its CRC, safe unique paths, exact inventory, and all 12 indexed members' hashes and sizes passed. The direct archive was sufficient; redundant transport chunks were not downloaded.

The exact training source is `b6dbcc75efd5994bfdb3de456862608b2cc3aa40`. A detached checkout of that commit isolates evaluation from r19 work. Every recorded training-source file was checked against its Git object and local bytes. Staged source/output image and label hashes, all 157 original PNGs and original labels, and both completed checkpoint bundles passed verification. The actual 106,838,542-byte epoch 173 initializer hashes to `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`.

Notebook commit `850fd7e36c2e283b798134056b0ecd6f3f534506` contains the expected notebook, SHA-256 `eb3a5f852e8d6db143197d4a808cf3e9c68b05725f6d224032f7a0abde9f0125`. Its embedded source and protocol pins match. Protocol SHA-256 is `788715d37250483c367513490e20945871cb77628437ed2ca27c72ea96587137`. [Notebook verification](transport-and-notebook-verification.json)

| Actual training input | Fresh control | r18 candidate |
| --- | ---: | ---: |
| Photos / empty photos | 137 / 13 | 137 / 13 |
| Fronts | 3,771 | 3,784 |
| Locatable endpoint slots | 6,841 | 6,837 |
| Missing endpoint slots | 701 | 731 |
| Original validation photos / fronts | 20 / 664 | 20 / 664 |

Exactly **92 training-label files differ** between arms. Photo 015 and the separate 31 old-series treatment exclusions use identical PR35 labels in both arms; they remain in training. Images and original validation labels are identical. The full review contains 3,786 training fronts, while eligible candidate training contains 3,784 because held-photo supervision follows the established control selection.

Both arms independently initialized from epoch 173 and completed **30 epochs, 1,050 batches, 4,110 image presentations, and 78 recorded optimizer steps**, with exit code 0 and the same NVIDIA A100-SXM4-40GB runtime fingerprint. The fixed seed 46 recipe and final-epoch selection are preserved. Recorded wall times were 383.40 and 383.70 seconds; these are completion records, not a controlled speed comparison with prior L4 runs.

| Model | Checkpoint SHA-256 |
| --- | --- |
| Selected PR35 | `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` |
| Fresh r18 control | `3ac9f977089edc9b4a6abde38466a21f93c7bdbfc8633212c92d856e7623fe6a` |
| Returned r18 candidate | `baf2c0eb852c1fbbae5dc6aaf1c68a0fd6ed7d75ed7be892557b9d01fda4d9a1` |

## Fixed inference and provenance

All three models received fresh CPU inference in one environment: Python 3.12.14, Torch 2.14.1+cpu, torchvision 0.29.1+cpu, Ultralytics 8.4.182, NumPy 2.4.6, Pillow 11.3.0, and OpenCV 5.0.0.93. Settings are input 1280, box confidence 0.13, endpoint confidence 0.50, NMS IoU 0.70, matching IoU 0.50, FP32, maximum 300 outputs, and four CPU threads. [Environment](inference-pip-freeze.txt), [commands](inference/commands.json), [scorer adaptation](measurement-adaptation.patch)

Training receipts record a clean source checkout. Local inference provenance records **`M weights/cassette_lower_epoch173.pt`**, because the LFS pointer was hydrated with verified original bytes. Evaluator, scoring, and checkpoint-reader hashes match the pinned source; all original data hashes were independently verified. The three inferred model weights are separately verified checkpoint artifacts. This audit does not describe the inference checkout as clean. [Provenance and pair checks](inference-provenance.json)

The fresh control and PR35 produce **exactly equal prediction arrays on all 20 photos**, with zero coordinate or confidence differences at this operating point. Their checkpoint artifact hashes differ. PR35's actual training receipt also names A100; its historical archive directory contains “l4”, so the directory name should not be treated as hardware evidence. Previous PR90 control/candidate receipts name L4. [Preserved receipts](checkpoint-receipts/)

Both-corner success requires a matched object, two locatable reference corners, adequate prediction confidence, coordinates inside the image, and both distances within the stated radius. Eligible missed objects remain failures in the denominator of 584. Zero- and one-corner fronts remain in detection results. Unmatched outputs are fixed-matching outcomes and have not all been independently classified as background false detections; duplicate counts are a subset.

## Frozen primary: 666 fronts

| Model | Matched | Misses | Unmatched outputs | Duplicate subset | Pairs within 5 px | Pairs within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PR35 | 648 | 18 | 80 | 32 | 533/584 | 582/584 |
| Fresh control | 648 | 18 | 80 | 32 | 533/584 | 582/584 |
| r18 candidate | 632 | 34 | 103 | 31 | 522/584 | 575/584 |

The candidate has **16 more misses and 23 more unmatched outputs** than control, with 11 fewer five-pixel and seven fewer ten-pixel pair successes. It gains three matched identities and loses 19. Its 34 misses comprise 13 zero-corner, 14 one-corner, and seven two-corner fronts.

## Historical: 663 fronts

| Model | Matched | Misses | Unmatched outputs |
| --- | ---: | ---: | ---: |
| PR35 | 648 | 15 | 80 |
| Fresh control | 648 | 15 | 80 |
| r18 candidate | 632 | 31 | 103 |

Duplicate and corner-pair counts equal the primary table. Both frozen references preserve their original bytes and existing uncertainty records.

## Pre-return r18 review: 662 fronts, exploratory

The review manifest is pinned to SHA-256 `85e48b568c5a36c9bbde0cf751fffd797754e513e29ccc9d1a724590a4800923`. It contains 20 development photos, 662 fronts, 1,226 locatable endpoint slots, 98 missing slots, and three empty scenes. Its development status has no unresolved cases. This is the r18 reference; no later amendment is substituted.

| Model | Matched | Misses | Unmatched outputs | Duplicate subset | Pairs within 5 px | Pairs within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PR35 | 626 | 36 | 102 | 21 | 538/584 | 573/584 |
| Fresh control | 626 | 36 | 102 | 21 | 538/584 | 573/584 |
| r18 candidate | 647 | 15 | 88 | 30 | 551/584 | 583/584 |

Compared with control, the candidate has 21 fewer misses, 14 fewer unmatched outputs, 13 more five-pixel successes, and ten more ten-pixel successes. It gains 24 matched identities and loses three, while its duplicate subset increases by nine. Its misses comprise seven zero-corner and eight one-corner fronts; all 584 two-corner fronts are matched. It is miss-free on 10 of 17 nonempty scenes, versus six for each baseline. All models return zero outputs on each of the three empty scenes.

The same predictions produce the frozen and exploratory tables. Their ranking conflict is reference sensitivity, not evidence that one table can replace the other. The fixed selection rule therefore leaves PR35 selected.

## Separate model changes from reference changes

Previous r17 outputs were replayed on the same r18 labels with the same scoring code and thresholds. The old frozen-reference results reproduce exactly. [Full previous-model replay](previous-r17-on-r18.json)

| Candidate, scored on identical r18 labels | Misses | Unmatched outputs | Duplicate subset | Pairs within 5 px | Pairs within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| PR90 r17, trained on L4 | 14 | 87 | 30 | 549/584 | 583/584 |
| PR92 r18, trained on A100 | 15 | 88 | 30 | 551/584 | 583/584 |

The new candidate loses **011 row 55**, with no gained matches. Its best box IoU changes from 0.505720 to 0.494516 across the fixed 0.50 boundary; confidence remains 0.774885. The front has only one locatable reference corner and does not enter the pair denominator. At 5 px, five identities gain success and three lose it. At 10 px, success identities are unchanged. **008 row 10** remains the only two-corner failure at 10 px, with right-corner error 12.119284 px. Exact identities, predictions, and errors are preserved in [case evidence](reference-and-generation-comparison.json).

There is also a separate **reference-only effect**. Leaving r17 predictions unchanged and moving from the old r16 review to the pre-return r18 review changes its candidate from 15 misses / 88 unmatched outputs to 14 / 87. The changed match is **090 row 43**. The same reference change removes PR35's match for that row, changing it from 35 / 101 to 36 / 102. Those score changes occur without a different model. Reporting PR90's old-reference score next to PR92's new-reference score would conceal this distinction.

The cross-generation comparison is descriptive: training supervision and GPU hardware both changed. The matched A100 control/candidate comparison is the appropriate within-run treatment comparison. Repeated development views, one fixed seed, and a few threshold crossings do not establish unseen-session reliability.

## Reproduction

Use source commit `b6dbcc75efd5994bfdb3de456862608b2cc3aa40`, the original images/initializer, and the verified PR92 archive. `audit_return.py` verifies the archive and staging against the source. `run_pr92_inference.py` accepts that source, the matching Python interpreter, a fresh output directory, and complete manifests for PR35/control/candidate. PR35 weights come from its [original archive](https://github.com/Siuuuers/dl/blob/8e8bb99cabcb44ed0ec0acd14db911aa44013fd8/results/gpu/baomai-reviewed-annotations-seed46-l4-001/baomai-reviewed-annotations-seed46-results.zip), SHA-256 `df53683ccf6f266a605b2844f42a98e808f75686cfab09f14d680c9d061e8a91`, candidate arm.

Replay the preserved evidence without loading model weights:

```bash
audit_source=/absolute/path/to/source-at-b6dbcc75
audit_evidence=/absolute/path/to/pr92-r18-20261122
python -B "$audit_evidence/measure_pr92.py" \
  --root "$audit_source" \
  --pr35 "$audit_evidence/inference/eval-pr35/predictions.jsonl" \
  --pr35-manifest "$audit_evidence/checkpoint-receipts/pr35/checkpoint-manifest.json" \
  --reference "$audit_evidence/inference/eval-reference/predictions.jsonl" \
  --reference-manifest "$audit_evidence/checkpoint-receipts/reference/checkpoint-manifest.json" \
  --candidate "$audit_evidence/inference/eval-candidate/predictions.jsonl" \
  --candidate-manifest "$audit_evidence/checkpoint-receipts/candidate/checkpoint-manifest.json" \
  --output /absolute/path/to/fresh-replay.json
```

For the previous-generation replay, add `--previous-r17` and supply the PR90 preserved predictions/receipts from `results/evaluation/pr90-r17-20261122` in that source. Run `compare_generations.py --root ... --audit ... --output ...` to reproduce the separate reference and model effects. Training logs are retained verbatim as `train-log.txt`; strict replay temporarily restores their original filename. [Delivery replay verification](replay-verification.json)

No GPU training or model promotion was performed during this audit. Pixel radii measure image-space agreement; no camera calibration, millimetre positioning accuracy, or robot grasp-success evidence is supplied here.
