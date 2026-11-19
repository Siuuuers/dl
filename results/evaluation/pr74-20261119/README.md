# PR74 r7 result audit

**Retain PR35 as the selected model.** Under the unchanged primary reference and operating point, the returned r7 candidate has **nine more scored misses and 15 more unmatched outputs** than either retained PR35 or the fresh r7 control. It gains four both-corner successes within five pixels but loses two within ten pixels. This result does not support promotion.

A scored miss means no one-to-one box match at IoU 0.50. It does not necessarily mean no proposal exists on the visible cassette. Several losses are disagreements about box extent, including a strong proposal at IoU 0.4973. The audit retains those outcomes and separately inspects the original pixels; it does not modify the reference to improve a model's score.

## Return and source verification

The [PR74 return](https://github.com/Siuuuers/dl/pull/74) was fetched at exact head **`942b4c83eb532a936a67cb4b0a3534913445c655`**. Its complete [results ZIP](https://raw.githubusercontent.com/Siuuuers/dl/942b4c83eb532a936a67cb4b0a3534913445c655/results/gpu/baomai-front-review-r7-a100-001/baomai-front-review-r7-seed46-v6-results.zip) is **39,782,036 bytes**, SHA-256 **`ec57673aba57306df2ec5a8e3efb3d06f5433cf53c968bce86b9560fe321f5f5`**, Git blob `9041d2c5d3d065339f2509c46ce3d2b2e45ede30`.

Verified checks include ZIP integrity and inventory, all 12 indexed member hashes and sizes, both completed checkpoint bundles, all recorded training-source hashes, candidate reconstruction, all 157 original image and label hashes, original epoch173 weights, retained PR35 weights, and unchanged primary/historical reference hashes. The independent checkout of [training source `826bc2061e7262378e482d73567b3c8b6f47f5c1`](https://github.com/Siuuuers/dl/tree/826bc2061e7262378e482d73567b3c8b6f47f5c1) was clean. The [notebook at `bb3b144372496781f8a892c758a64d1f0c27986d`](https://github.com/Siuuuers/dl/blob/bb3b144372496781f8a892c758a64d1f0c27986d/notebooks/baomai_front_review_r7.ipynb) pins that source and the exact r7 protocol SHA-256 `f3954ea32f24c58c0f71a296153428aca8fd6f7bcd28a6595ea1c3802d82907f`.

[verified-return.json](verified-return.json), [notebook-pin-verification.json](notebook-pin-verification.json) and the [original return index](returned-gpu-results-manifest.json) contain the evidence. Training was checked against the returned files and frozen validators; GPU training was not repeated.

Both arms completed **30 epochs, 1,050 batches, 4,110 image presentations and 78 recorded optimizer updates**, independently starting from the original epoch173 checkpoint. Both used NVIDIA A100-SXM4-40GB, seed 46, batch 4, image size 1280, FP32, AdamW and the same fixed learning-rate schedule. Final epoch 30 is selected. There was no epoch, seed or threshold search.

Each arm retains 137 training images, including 13 empty scenes. Control supervision contains **3,771 fronts and 6,841 locatable corners**; candidate supervision contains **3,780 fronts and 6,850 locatable corners**. Labels differ on **79 training photos**. The same **44 complete held photos** retain exact PR35 labels in both arms, including every old-camera photo. Holding inherited labels controls this comparison; it does not certify their physical correctness. Original validation remains 20 images and 664 fronts.

## Fresh inference and fixed scores

All three checkpoints were freshly inferred on the same 20 original PNGs in one CPU environment: Python 3.12.14, PyTorch 2.14.1+cpu and Ultralytics 8.4.182; see [package versions](inference-pip-freeze.txt). Settings remain **box confidence 0.13, endpoint confidence 0.50, match IoU 0.50, NMS IoU 0.70, input size 1280, ordinary NMS, FP32 and four CPU threads**. Pixel distances refer to the original images.

The scorer bytes are unchanged from the previous audit. [measurement-adaptation.patch](measurement-adaptation.patch) shows only protocol and checkpoint identity changes. [fixed-comparison.json](fixed-comparison.json) contains full denominators, per-image scores, matching traces and gain/loss identities for both references.

### Primary reference: 666 fronts

The unchanged reference has 20 images, 17 nonempty scenes, three empty scenes, 1,227 locatable endpoints and 105 missing endpoint slots. Manifest SHA-256: `0de64be3797dce2e43d4cb98d09620eab969bb95725ed9e75682dcfcd53c7f96`.

| Model | Matched fronts | Scored misses | Unmatched outputs | Duplicate subset | Both corners <=5 px | Both <=10 px | Miss-free nonempty scenes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained PR35 | 648/666 | 18 | 80 | 32 | 533/584 | 582/584 | 6/17 |
| Fresh r7 control | 648/666 | 18 | 80 | 32 | 533/584 | 582/584 | 6/17 |
| r7 candidate | 639/666 | 27 | 95 | 29 | 537/584 | 580/584 | 4/17 |

The full detection denominator stays **666**, including partial fronts and fronts without a locatable corner. The separate pair denominator stays **584 fronts with two locatable reference corners**, with missed eligible fronts counted as failures. It is not complete-scene accuracy.

| Locatable reference corners | Fronts | PR35/control misses | Candidate misses |
| --- | ---: | ---: | ---: |
| Zero | 23 | 10 | 13 |
| One | 59 | 7 | 11 |
| Two | 584 | 1 | 3 |

The candidate returns 1,210/1,227 qualifying locatable endpoints, versus 1,218 for both controls. Individual endpoint successes within five pixels fall from 1,149 to 1,143. A gain in the paired measure therefore does not mean all localization measures improve. All three models return zero outputs on the three empty scenes; none produces an error-free nonempty scene under the fixed matcher.

The **ordered decoded prediction arrays for PR35 and the fresh r7 control are exactly equal on all 20 images** at this operating point, verified in [prediction-equivalence.json](prediction-equivalence.json). Their checkpoint byte hashes differ. This observation does not establish equality of model tensors, behavior on other images, or other thresholds.

### Historical reference: 663 fronts

The unchanged historical manifest is `bbc1dca71f7a3e0fe2f54a4391962e562c3353a62533eb233b697819edcc64b1` (1,227 locatable endpoints, 99 missing slots).

| Model | Matches | Scored misses | Unmatched outputs | Miss-free nonempty scenes |
| --- | ---: | ---: | ---: | ---: |
| Retained PR35 | 648/663 | 15 | 80 | 7/17 |
| Fresh r7 control | 648/663 | 15 | 80 | 7/17 |
| r7 candidate | 640/663 | 23 | 94 | 4/17 |

The pair scores remain the same as in the primary table. The candidate adds eight misses and 14 unmatched outputs. These references share inherited labels, so agreement between them is not independent physical verification. No newer review labels were substituted into either score.

## Gained and lost fronts: what the pixels support

Versus either control, the candidate gains one primary match, **009:53**, and loses ten: **008:33, 009:56, 010:15, 010:31, 010:32, 011:61, 056:15, 057:11, 090:42 and 090:43**. Here `NNN:row` means `new_000NNN` and a one-based row in the frozen primary reference.

Every lost front's best candidate IoU is below 0.50, so these losses are not solely qualifying boxes consumed by other one-to-one assignments. They represent different situations:

| Front | Control best IoU | Candidate best IoU | Candidate confidence | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 009:53, gained | 0.31960 | 0.54964 | 0.24989 | An existing proposal changes extent and becomes a match. |
| 010:15, lost | 0.59836 | 0.49732 | 0.89210 | Strong overlapping proposal, just below the unchanged cutoff. |
| 010:32, lost | 0.69115 | 0.49217 | 0.19224 | Extent disagreement on an occluded region, below the cutoff. |
| 057:11, lost | 0.60205 | 0.48055 | 0.87528 | Strong proposal remains; gold extends into rack below the visible front. |
| 008:33, lost | 0.60516 | 0.07171 | 0.90206 | Best overlap is weak and belongs mainly to neighboring visible material. |
| 009:56, lost | 0.54474 | 0.00000 | Not applicable | No retained candidate output overlaps the frozen box. |

“Candidate confidence” describes the box with the largest overlap; it is not a probability that the intended front or its corners are correct. For `009:56`, the absence claim is restricted to the saved post-threshold output set.

[Original-pixel figures](figures), [figure provenance](figure-provenance.json) and [visual review](visual-review.json) separate those observations from mechanical scores. In particular, `057:11` makes rack inclusion visible, while several occluded cases have candidate extents closer to the visible white region and frozen boxes extending across an occluder. This supports further physical annotation review; it does not authorize changing the frozen answer key or claiming the candidate is better overall.

At five pixels, ten pair identities gain success and six lose it, for net +4. At ten pixels, `007:15` gains success and `008:10`, `010:15`, `057:11` lose it. The latter two also fail box association.

Total predictions increase from 728 to 734 while matches fall by nine, explaining the 15 additional unmatched outputs. The duplicate subset falls from 32 to 29; other unmatched outputs rise from 48 to 66. The candidate has 67 unmatched outputs at confidence >=0.25, versus 47 for control, so unmatched outputs are not confined to the weakest confidence band. This bin is descriptive; no alternative threshold was tested. Full output and scene traces are in [fixed-output-diagnostics.json](fixed-output-diagnostics.json).

The inspected device area in `new_000090` has three retained proposals in each arm despite no cassette front there. An unlabeled top-of-frame strip in `new_000089` has one versus two; its physical identity remains unresolved, so these counts are not asserted to be independently adjudicated background errors. Unmatched outputs as a whole must not be relabeled “background false positives” without such review.

## Original 664-front validation CSV

These returned epoch-30 metrics concern the original validation labels and aggregate metric procedure, rather than the fixed operating point above.

| Metric | Fresh r7 control | r7 candidate |
| --- | ---: | ---: |
| Box precision | 0.96820 | 0.95395 |
| Box recall | 0.91699 | 0.92018 |
| Box mAP50 | 0.93920 | 0.93538 |
| Box mAP50-95 | 0.57519 | 0.57799 |
| Pose precision | 0.98584 | 0.97492 |
| Pose recall | 0.93825 | 0.93976 |
| Pose mAP50 | 0.96043 | 0.96095 |
| Pose mAP50-95 | 0.96036 | 0.96080 |

Sources: [control CSV](checkpoint-receipts/reference/fit/results.csv), [candidate CSV](checkpoint-receipts/candidate/fit/results.csv). Small mAP increases do not override the fixed miss/unmatched comparison.

## Reproduce and practical limits

Saved-prediction replay uses Python 3.11+, Git and Git LFS, without loading weights or running training. Obtain the frozen source checkout with its original validation PNGs, set `pr74_source` to that checkout and `pr74_evidence` to this directory, then run:

```bash
python -B "$pr74_evidence/measure_pr74.py" \
  --root "$pr74_source" \
  --pr35 "$pr74_evidence/eval-pr35/predictions.jsonl" \
  --pr35-manifest "$pr74_evidence/checkpoint-receipts/pr35/checkpoint-manifest.json" \
  --reference "$pr74_evidence/eval-reference/predictions.jsonl" \
  --reference-manifest "$pr74_evidence/checkpoint-receipts/reference/checkpoint-manifest.json" \
  --candidate "$pr74_evidence/eval-candidate/predictions.jsonl" \
  --candidate-manifest "$pr74_evidence/checkpoint-receipts/candidate/checkpoint-manifest.json" \
  --output /path/to/new-fixed-comparison.json
cmp "$pr74_evidence/fixed-comparison.json" /path/to/new-fixed-comparison.json
```

The delivered script, receipts and predictions reproduce the comparison byte for byte; see [delivery verification](delivery-verification.json). Individual `eval-*` summaries use the original 664-front labels; use `fixed-comparison.json` for the 666/663 results.

For fresh inference, [run_pr74_inference.py](run_pr74_inference.py) accepts the frozen source, interpreter, a fresh output directory and checkpoint manifests. Get complete weights from the PR74 archive above and the [original PR35 archive](https://raw.githubusercontent.com/Siuuuers/dl/8e8bb99cabcb44ed0ec0acd14db911aa44013fd8/results/gpu/baomai-reviewed-annotations-seed46-l4-001/baomai-reviewed-annotations-seed46-results.zip), SHA-256 `df53683ccf6f266a605b2844f42a98e808f75686cfab09f14d680c9d061e8a91` (use its candidate arm). [Recorded commands](inference-commands.json) and logs preserve this audit's actual executions.

This is one seed and recipe on repeatedly used development images. The matched r7 arms share an A100 training environment; comparisons against prior L4 rounds cannot attribute every difference solely to annotations. No result here certifies all labels, unseen-camera reliability or robot positioning. Retaining PR35 preserves the established selection while physical annotation corrections and a separately pinned next experiment continue.
