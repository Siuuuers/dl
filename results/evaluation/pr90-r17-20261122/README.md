# PR90 r17 result audit

**Keep PR35 selected.** The returned r17 candidate improves agreement with the pre-return r16 review but regresses on the frozen primary and historical references. Against the previous r15 candidate on identical r16 labels, it gains one matched front while adding two unmatched outputs and one duplicate. Aggregate five- and ten-pixel corner-pair counts do not improve. These results do not establish improved performance on unseen capture sessions or a physical positioning tolerance.

This audit fixes the references before the new r18 amendments. The 666-front primary, 663-front historical, and 662-front r16 review are reported separately. No annotation edit, GPU training, threshold search, or model promotion occurs here. [Summary](summary.json), [full comparison](fixed-comparison.json), [integrity audit](verified-return.json).

## Verified return and training inputs

[PR90](https://github.com/Siuuuers/dl/pull/90) is pinned to head `ae390faccbf59731214933212fdce1a98868a630`. Its [direct committed archive](https://github.com/Siuuuers/dl/blob/ae390faccbf59731214933212fdce1a98868a630/results/gpu/baomai-front-review-r17-l4-001/baomai-front-review-r17-results.zip) is **39,782,260 bytes**, SHA-256 `339eec148cc5d56d9bd2351f2562f09658685dd9117073338c434af63ad4917f`. ZIP CRC, unique safe paths, exact inventory, and all 12 indexed members' hashes and sizes passed. Downloading the redundant transport chunks was unnecessary.

The exact training source is `08921b4ccd122798675293221145f2be41171db2`. Notebook revision `15869c7742e72ee1a11d0e5e2b668206f02c0348` contains the expected notebook bytes and embeds that source pin. Notebook SHA-256 is `8a1bfc4f548dfb3e67a23f07904a925bac14b9accd4378acf2fca730ef4c0da2`; protocol SHA-256 is `320d9e97805ddea968dd04b033037272c0fe41d77f1c149c2b05bc654f4ca12a`. [Notebook and transport verification](transport-and-notebook-verification.json)

A separate detached worktree at the training source isolates evaluation from r18 edits. Every recorded source file agrees with the Git object and local bytes. All staged image/label source and output hashes, all 157 original PNGs and original labels, and both complete checkpoint bundles passed verification. The actual 106,838,542-byte epoch 173 initializer was independently rehashed to `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`.

**Provenance distinction:** the returned training receipts record a clean training checkout. The local inference checkout is **not recorded as clean**: its saved provenance contains `M weights/cassette_lower_epoch173.pt` because the original LFS pointer was hydrated with the verified initializer bytes. The evaluator, scorer, checkpoint reader, protocol, original images, and labels retain their verified content hashes. The three inferred model weights are separately verified returned checkpoints. [Local inference provenance](inference-provenance.json)

| Actual training input | Fresh control | r17 candidate |
| --- | ---: | ---: |
| Photos | 137 | 137 |
| Empty photos | 13 | 13 |
| Fronts | 3,771 | 3,784 |
| Locatable endpoints | 6,841 | 6,837 |
| Missing endpoint slots | 701 | 731 |
| Original validation photos/fronts | 20/664 | 20/664 |

Exactly **92 training-label files differ**. Held photo 015 and the separate 31 old-series treatment exclusions retain identical PR35 supervision in both arms; these photos remain in training. Images and original validation labels are identical between arms.

Both arms independently initialized from epoch 173 and completed **30 epochs, 1,050 batches, 4,110 image presentations, and 78 recorded optimizer steps**, with exit code 0 and the same NVIDIA L4 runtime fingerprint. Seed 46, batch 4, input 1280, FP32, AdamW 0.0001, and final-epoch selection are unchanged. Receipt wall times were 460.77 seconds for control and 455.02 seconds for candidate; these are not a controlled speed benchmark. [Preserved receipts](checkpoint-receipts/)

| Model | Checkpoint SHA-256 |
| --- | --- |
| Selected PR35 | `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` |
| Fresh r17 control | `7474bfe23f5db6d9b63678059cf4b29e1a7c962bdf736940a240b4c2ddba8c42` |
| Returned r17 candidate | `bab1050d4a8adbaeda4fe371203b58b35581074c909d63328317acfd36457b2b` |

## Fixed inference and scoring

All three models were freshly inferred in the same CPU environment: Python 3.12.14, Torch 2.14.1+cpu, torchvision 0.29.1+cpu, Ultralytics 8.4.182, NumPy 2.4.6, Pillow 11.3.0, and OpenCV 5.0.0.93. Settings were box confidence 0.13, endpoint confidence 0.50, NMS IoU 0.70, matching IoU 0.50, input 1280, FP32, maximum 300 outputs, and four CPU threads. [Environment](inference-pip-freeze.txt), [actual commands](inference/commands.json), [script adaptation](measurement-adaptation.patch)

A pair succeeds only when its object is matched and **both locatable endpoints** pass the confidence, image-boundary, and pixel-distance criteria. Eligible missed objects remain failures in the denominator of 584. Zero- and one-corner objects remain in detection results. “Unmatched outputs” describes the fixed matching outcome; these have not all been independently adjudicated as background false detections. Duplicate counts are a subset of unmatched outputs.

## Frozen primary: 666 fronts

| Model | Matched | Misses | Unmatched outputs | Duplicate subset | Pairs within 5 px | Pairs within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PR35 | 648 | 18 | 80 | 32 | 533/584 | 582/584 |
| Fresh control | 646 | 20 | 83 | 29 | 531/584 | 583/584 |
| r17 candidate | 632 | 34 | 103 | 29 | 521/584 | 575/584 |

The candidate has **14 more misses and 20 more unmatched outputs than control**, with ten fewer five-pixel successes and eight fewer ten-pixel successes. Its 34 misses comprise 13 zero-corner, 14 one-corner, and 7 two-corner fronts. This does not meet the frozen comparison rule for selecting a replacement.

## Historical: 663 fronts

| Model | Matched | Misses | Unmatched outputs |
| --- | ---: | ---: | ---: |
| PR35 | 648 | 15 | 80 |
| Fresh control | 646 | 17 | 83 |
| r17 candidate | 632 | 31 | 103 |

Duplicate and corner-pair counts equal the primary table. Both frozen references preserve their original annotation bytes and existing uncertainty records.

## Pre-return r16 review: 662 fronts, exploratory

This reference was pinned before PR90 returned. Manifest SHA-256 is `1645f999fa8be08e9c8a317ffccd88e54285a81aac181a8e1ec8c9b328b81993`. It contains 20 development photos, 662 fronts, 1,226 locatable endpoints, 98 missing slots, and three empty scenes, with no unresolved development cases in that review. No r18 label is used in this audit.

| Model | Matched | Misses | Unmatched outputs | Duplicate subset | Pairs within 5 px | Pairs within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PR35 | 627 | 35 | 101 | 21 | 538/584 | 573/584 |
| Fresh control | 624 | 38 | 105 | 16 | 534/584 | 573/584 |
| r17 candidate | 647 | 15 | 88 | 30 | 549/584 | 583/584 |

Compared with control, the candidate has 23 fewer misses, 17 fewer unmatched outputs, 15 more five-pixel successes, and 10 more ten-pixel successes. It gains 26 matched identities and loses 3. Its duplicate subset rises from 16 to 30 despite fewer unmatched outputs overall. It is miss-free on 9 of 17 nonempty scenes, versus 6 for each baseline. Candidate misses comprise 7 zero-corner and 8 one-corner fronts; all 584 two-corner fronts are matched. All three models produce zero outputs on each of the three empty scenes.

The same predictions produce both the frozen and reviewed tables. This ranking conflict is reference sensitivity. The favorable exploratory result does not replace the frozen selection rule; repeated use of the same development scenes also limits claims about unseen sessions.

### Change from r15 on identical r16 labels

The prior PR88 predictions were replayed on exactly the same r16 reference. The prior r16 diagnostic reproduced **byte-for-byte**. [Previous-generation comparison and case evidence](prior-generation-comparison.json)

| Candidate generation | Misses | Unmatched outputs | Duplicate subset | Pairs within 5 px | Pairs within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| PR88 r15 | 16 | 86 | 29 | 549/584 | 583/584 |
| PR90 r17 | 15 | 88 | 30 | 549/584 | 583/584 |

The only gained matched identity is **new_000089 row 49**, with no lost matches. A new retained r17 box has confidence 0.137281, just above the fixed 0.13 threshold, and IoU 0.747528. The best retained r15 box had IoU 0.447477. This object has only one locatable reference corner and is outside the two-corner denominator. Full-frame original pixels were inspected: the gripper obstructs the front's left portion. The reference's unlocatable left corner is not assigned a coordinate here.

At 5 px, 090 rows 45/46 gain success while 059 row 11 and 090 row 19 lose it; at 10 px, success identities are identical. For 059 row 11, left-corner error changes from 4.963027 to 5.068042 px. For 090 row 19, right-corner error changes from 4.986816 to 5.067965 px. These boundary crossings explain why equal totals do not mean identical successful objects. The sole r16-review two-corner failure at 10 px remains 008 row 10, whose right-corner error is 11.929434 px despite confidence 0.909419. Original full-frame context was also inspected for that case; labels were unchanged.

Fresh PR35/control results have identical detection counts and matched/corner-success identities to the prior audit on all three references. The raw arrays are not byte-identical: maximum coordinate difference is 0.0001220703125 px and maximum confidence differences are 1.63913e-6 and 1.46031e-6, respectively. Recorded runtime versions match. The cross-generation comparison uses preserved r15 predictions and should not be mistaken for a new same-process inference of r15 weights. It also does not isolate the r16-to-r17 annotation change.

## Reproduction

Use a detached source checkout at `08921b4ccd122798675293221145f2be41171db2`, the verified original images/initializer, and the actual PR90 archive above. `audit_return.py` verifies the archive and staging against that source. `run_pr90_inference.py` takes the source, matching Python runtime, fresh output directory, and complete PR35/returned checkpoint manifests. PR35 weights come from the [original PR35 archive](https://github.com/Siuuuers/dl/blob/8e8bb99cabcb44ed0ec0acd14db911aa44013fd8/results/gpu/baomai-reviewed-annotations-seed46-l4-001/baomai-reviewed-annotations-seed46-results.zip), SHA-256 `df53683ccf6f266a605b2844f42a98e808f75686cfab09f14d680c9d061e8a91`, candidate arm.

The following replay uses the preserved evidence without loading model weights. Replace the two paths with the isolated source checkout and this evidence directory:

```bash
audit_source=/absolute/path/to/source-at-08921b4
audit_evidence=/absolute/path/to/pr90-r17-20261122
python -B "$audit_evidence/measure_pr90.py" \
  --root "$audit_source" \
  --pr35 "$audit_evidence/inference/eval-pr35/predictions.jsonl" \
  --pr35-manifest "$audit_evidence/checkpoint-receipts/pr35/checkpoint-manifest.json" \
  --reference "$audit_evidence/inference/eval-reference/predictions.jsonl" \
  --reference-manifest "$audit_evidence/checkpoint-receipts/reference/checkpoint-manifest.json" \
  --candidate "$audit_evidence/inference/eval-candidate/predictions.jsonl" \
  --candidate-manifest "$audit_evidence/checkpoint-receipts/candidate/checkpoint-manifest.json" \
  --output /absolute/path/to/fresh-replay.json
```

To reproduce the previous-generation comparison, first run the unchanged `results/evaluation/pr88-r15-20261122/measure_post_r16.py` from that source, then pass its output to `compare_previous.py --root ... --audit ... --previous-replay ... --output ...`. [Delivery replay verification](replay-verification.json) records the replay check. Training logs are retained verbatim as `train-log.txt`; strict replay restores the original filename in a temporary bundle.

Pixel radii are image-space criteria. No camera calibration, robot success rate, unseen-session validation, or measured millimetre tolerance is supplied by this audit. PR35 remains selected while the next prospectively pinned experiment is prepared.
