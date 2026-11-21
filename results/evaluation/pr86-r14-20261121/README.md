# PR86 r14 result audit

**Keep PR35 selected.** The returned r14 candidate improves alignment with the accepted r14 review, but regresses on the frozen primary and historical references. Against the previous r13 candidate on the same r14 review, it has one fewer miss and five fewer unmatched outputs, while losing two ten-pixel corner-pair successes. The result remains mixed and reference-sensitive.

This report evaluates the completed r14 experiment before the newly requested r15 amendments. No training, threshold search, annotation changes or model promotion occur in this audit. [Summary](summary.json), [full scored comparison](fixed-comparison.json), [integrity audit](verified-return.json).

## Verified return

[PR86](https://github.com/Siuuuers/dl/pull/86) is pinned to head `218f10ce7e964c958d0fb26159b01334951e5d7a`. The directly downloaded archive has **39,781,759 bytes**, SHA-256 `8df7ef283d2b05a642fd22ae74b52d1758277ee06dba185bbf42f26243936f51`. ZIP CRC, unique safe member names, exact inventory and hashes/sizes of all 12 indexed members passed. Transport chunks were not separately downloaded because the complete archive was retrieved and verified directly.

Training source is `4d805995e41e3520b10d0b2f743b1219c8356ca9`; notebook revision is `a928f6beda7a797eeb21fc3961db7f1028555e4c`. The exact notebook bytes and embedded source pin match the published handoff. The r14 protocol SHA-256 is `a82e3b13e25ef9d4fdd405848bb4bafac16d658b92b9d1fadcc263ee3055cc12`. [Transport and notebook verification](transport-and-notebook-verification.json)

A separate checkout of the exact training source was used. All recorded source hashes, staged input hashes, original image/label bytes and completed checkpoint bundles passed. The 157 original PNGs and epoch173 initialization bytes were reverified. Later r15 work in the main checkout cannot change this audit source.

| Actual input | Fresh control | r14 candidate |
| --- | ---: | ---: |
| Training photos | 137 | 137 |
| Empty training photos | 13 | 13 |
| Front instances | 3,771 | 3,787 |
| Locatable endpoints | 6,841 | 6,838 |
| Missing endpoint slots | 701 | 736 |
| Original validation photos/fronts | 20/664 | 20/664 |

Exactly **92 training label files differ**. The held photo015 and the separate 31 old-series treatment exclusions have exact PR35 supervision in both arms. All photos remain in training.

Both arms independently started from original epoch173 and completed 30 epochs, 1,050 batches, 4,110 image presentations and 78 recorded optimizer steps on NVIDIA L4, with exit code 0. Seed 46, batch 4, input 1280, FP32, AdamW 0.0001 and final-epoch selection are unchanged. Receipt wall times are approximately 463.39 seconds per arm; they are not a controlled speed benchmark. [Preserved receipts](checkpoint-receipts/)

| Checkpoint | SHA-256 |
| --- | --- |
| Selected PR35 | `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` |
| Fresh r14 control | `befc8e56f8c1256526a719be63cff83e52b585823fa63c8564777990c29d59ce` |
| r14 candidate | `8cfe61eaed49e78d2d6dbc61c584e812ad970ab62506a59a27ceef67e6ff39ba` |

## Fixed inference

All three models were freshly inferred in one CPU environment: Python 3.12.14, Torch 2.14.1+cpu, torchvision 0.29.1+cpu, Ultralytics 8.4.182, NumPy 2.4.6, Pillow 11.3.0 and OpenCV 5.0.0.93. Settings remain box confidence 0.13, endpoint confidence 0.50, NMS IoU 0.70, matching IoU 0.50, input 1280, FP32, maximum 300 outputs and four CPU threads. [Environment](inference-pip-freeze.txt), [commands](inference/commands.json), [bounded script adaptation](measurement-adaptation.patch).

Compared with prior saved PR84 baseline arrays, both fresh baselines have tiny numerical differences on all 17 nonempty scenes; three empty scenes are identical. Maximum coordinate drift is 0.0001220703125 pixels; maximum confidence drift is approximately 0.000001639 for PR35 and 0.000001460 for the fresh control. Their matched and five/ten-pixel success identities on the same r14 review are unchanged. Raw arrays are preserved rather than rounded into equality. The cause of the tiny numerical drift is not asserted. [Prediction consistency](prediction-consistency.json)

## Frozen primary: 666 fronts

| Model | Matched | Misses | Unmatched outputs | Duplicate subset | Pairs within 5 px | Pairs within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PR35 | 648 | 18 | 80 | 32 | 533/584 | 582/584 |
| Fresh control | 646 | 20 | 83 | 29 | 531/584 | 583/584 |
| r14 candidate | 631 | 35 | 101 | 29 | 520/584 | 575/584 |

Against control, the candidate gains three matched identities and loses 18: 15 more misses and 18 more unmatched outputs. Of 35 candidate misses, 13 have zero locatable corners, 15 have one and 7 have two. Eligible missed detections remain failures in the 584-pair denominator.

## Historical: 663 fronts

| Model | Matched | Misses | Unmatched outputs |
| --- | ---: | ---: | ---: |
| PR35 | 648 | 15 | 80 |
| Fresh control | 646 | 17 | 83 |
| r14 candidate | 631 | 32 | 101 |

Corner-pair counts equal the primary table. These frozen references preserve their original annotation bytes and historical uncertainty records.

## Pre-return r14 review: 662 fronts, exploratory

This exact review was pinned before the returned training results. Manifest SHA-256: `95b3c8bd4746281f9474ecd787a063634ee1b6750d758f4454ddaaaa7479ef0f`. It has 20 development photos, 662 fronts, 1,226 locatable endpoints and 98 missing slots, including three empty photos. There are no unresolved development cases in this r14 review.

| Model | Matched | Misses | Unmatched outputs | Duplicate subset | Pairs within 5 px | Pairs within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PR35 | 628 | 34 | 100 | 21 | 540/584 | 575/584 |
| Fresh control | 624 | 38 | 105 | 16 | 535/584 | 574/584 |
| r14 candidate | 644 | 18 | 88 | 30 | 547/584 | 580/584 |

Against control, candidate has 20 fewer misses, 17 fewer unmatched outputs, 12 more five-pixel successes and 6 more ten-pixel successes. It gains 25 matched identities and loses 5. Its duplicate subset increases from 16 to 30 despite fewer unmatched outputs overall. Candidate is miss-free on 7 of 17 nonempty scenes; both baselines are miss-free on 6. Candidate misses comprise 7 zero-corner fronts, 10 one-corner fronts and 1 two-corner front. All three models emit zero outputs on all three empty scenes.

The same prediction arrays produce both frozen and reviewed tables. The change in ranking measures sensitivity to reference geometry; the favorable table does not silently replace the prospective selection rule. Reused development images do not establish generalization to new camera sessions or robotic positioning accuracy.

### Change since the previous r13 candidate

The previous candidate is scored using its saved arrays on this identical r14 review. The fresh baseline consistency check above preserves the distinction between raw numeric equality and unchanged scored outcomes.

| Candidate generation | Misses | Unmatched outputs | Duplicate subset | Pairs within 5 px | Pairs within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| PR84 r13 | 19 | 93 | 31 | 543/584 | 582/584 |
| PR86 r14 | 18 | 88 | 30 | 547/584 | 580/584 |

There are four gained and three lost matched identities. At 5 px, ten pairs gain success and six lose it. At 10 px, there are no gains and two losses: `new_000007:13` and `new_000008:20`. This supports a mixed result rather than an unconditional model improvement. Exact identities are retained in [summary.json](summary.json).

## Three bounded original-image checks

Original full-frame context and raw/overlaid crops were inspected for three newly lost reviewed matches:

- **new_000007:13:** both arms propose the top-edge front. Control/candidate best IoUs are 0.84395/0.49742. Candidate's shorter extent falls just below the fixed 0.50 match threshold; the scored miss is not an absent proposal.
- **new_000008:34:** control covers the right-edge partial front at IoU 0.68397. Candidate's best overlap is 0.06496 and mainly belongs to the tier below.
- **new_000089:49:** the gripper partly occludes the target. Candidate proposes a smaller exposed portion beside it, with IoU 0.45623 versus control 0.79523. A proposal remains, but it fails the fixed matching criterion.

These examples do not relabel the images or adjudicate every unmatched output. [Case evidence](representative-case-review.json)

## Reproduction and next step

Use the exact source commit above with original images and epoch173 bytes for `audit_return.py`. `run_pr86_inference.py` accepts that source, a compatible Python runtime, the original complete PR35 and returned r14 checkpoint manifests, and a fresh output directory. `measure_pr86.py` scores saved prediction arrays without loading weights.

The recorded comparison was replayed using the committed prediction files and preserved receipt-only folders, with **identical output bytes**. Original `train.log` bytes are saved as `train-log.txt` and restored temporarily by the strict receipt validator. [Replay verification](replay-verification.json), [log preservation](log-preservation.json).

The user's new r15 annotations are a later data revision. Any replay against those labels must be separately named as a post-return diagnostic; it adds no training evidence. The next controlled notebook can test the supported r15 changes while retaining the existing architecture, initialization, recipe and thresholds. This r14 run changed amendments and label eligibility together and cannot isolate their individual effects.

## Later r15 reference: cached diagnostic only

The separately saved [post-r15 diagnostic](post-r15-diagnostic.json) reuses the exact three prediction files above, with no inference, training or setting changes. Its reference manifest is `data/reviews/fronts-2026-11-21-r15/manifest.json`, SHA-256 `6012d09767851151312537c2046022b972bf6519f36d110675d3e82f74694e93`. Seven development photos changed: 007, 008, 009, 010, 054, 055 and 059. The reference still contains 20 photos, 662 fronts, 1,226 locatable endpoints, 98 missing slots and three empty scenes, with no unresolved development cases. Equal totals do not mean equal labels.

| Existing model, scored against r15 labels | Matched | Misses | Unmatched outputs | Duplicate subset | Pairs within 5 px | Pairs within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PR35 | 627 | 35 | 101 | 21 | 538/584 | 573/584 |
| Fresh r14 control | 623 | 39 | 106 | 16 | 533/584 | 572/584 |
| Returned r14 candidate | 645 | 17 | 87 | 30 | 549/584 | 583/584 |

Against its earlier r14-reference score, the unchanged returned candidate gains the match `new_000007:13`, two five-pixel successes and three ten-pixel successes. Both unchanged baselines lose `new_000007:15` and two successes at each radius. These differences result from the later reference revision, not model improvement. All three models still emit zero outputs on the empty scenes. This diagnostic cannot replace the frozen results or the pre-return r14 comparison; **PR35 remains selected**.

Reproduce this diagnostic from the saved arrays and receipt folders:

```sh
python results/evaluation/pr86-r14-20261121/measure_post_r15.py \
  --root . --output /path/to/fresh-post-r15-diagnostic.json
```
