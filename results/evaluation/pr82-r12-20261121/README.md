# PR82 r12 result audit

**Keep PR35 selected. The returned r12 candidate is a mixed, reference-sensitive result.** It has more misses and unmatched outputs on both frozen historical references, while fitting the independently pinned r12 reviewed geometry better. The same prediction arrays produce both conclusions. This is evidence that annotation definition matters; it does not support silently choosing the most favorable reference or promoting the candidate.

The audit verified the return and reran all three models at the fixed operating point. No training, threshold, NMS, seed or epoch search was performed. [Summary](summary.json), [complete comparison](fixed-comparison.json), and [representative visual review](representative-case-review.json) preserve the measured evidence.

## Verified return

[PR82](https://github.com/Siuuuers/dl/pull/82) is pinned to head `1edab4635ff3dc052cc7a67d09d09c773361fb41`. Its raw ZIP contains **39,781,223 bytes**, SHA-256 `24b15853253ba20762939ae16e236a1ee99736df58f5ca96a06a57c5f996e572`. ZIP CRC, safe unique paths, exact archive inventory, and hashes and sizes of all 12 indexed members passed.

Training used source `4f4423a511b42bc804cbcbeb7a772823cef2eb3b` and [r12 protocol](../../../experiments/front-review-2026-11-20-r12.json), SHA-256 `6c23b00d5251b9cd98630a796a328dc9cc5574296ccb47bac83cbb1ffa3e7f0a`. A detached checkout of that exact source was used for this audit, so subsequent r13 staging changes cannot alter the checked training code. All recorded source files, staged data hashes, original images and labels, completed checkpoint bundles and original epoch173 bytes were verified. [Return audit](verified-return.json) records the checks.

| Actual training input | Fresh control | r12 candidate |
| --- | ---: | ---: |
| Training photographs | 137 | 137 |
| Empty training scenes | 13 | 13 |
| Fronts | 3,771 | 3,777 |
| Locatable endpoints | 6,841 | 6,845 |
| Missing endpoint slots | 701 | 709 |
| Original validation photographs / fronts | 20 / 664 | 20 / 664 |

The arms differ on **78 training label files**. All **15 held training photographs** and **31 separate old-photo exclusions** retain exact PR35 supervision in both arms. Original validation and original images are identical. Equal held labels control the treatment difference; they do not certify the inherited geometry.

Both arms independently started from epoch173, used seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW at 0.0001, and selected final epoch 30. Each completed 1,050 batches, 4,110 image presentations and 78 recorded optimizer steps on NVIDIA L4. Recorded training wall times are 458.21 seconds for control and 454.93 seconds for candidate; these receipts are not a controlled speed comparison. [Checkpoint receipts](checkpoint-receipts/) preserve the original manifests, logs and CSVs. Exact log bytes use `.txt` filenames so ordinary log ignores cannot omit them; [log identities](log-preservation.json) record the mapping.

## Fresh fixed inference

All three checkpoints ran in the same CPU environment: Torch 2.14.1+cpu, torchvision 0.29.1+cpu, Ultralytics 8.4.182, NumPy 2.4.6, Pillow 11.3.0 and OpenCV 5.0.0.93. [Package pins](inference-pip-freeze.txt) and [commands](inference/commands.json) record the runtime and exact invocation.

Settings remain box confidence 0.13, endpoint confidence 0.50, ordinary NMS IoU 0.70, matching IoU 0.50, input 1280, FP32, maximum 300 detections and four CPU threads. The scoring implementations are unchanged; [measurement adaptation](measurement-adaptation.patch) shows the experiment identity and separately identified current-review loading changes.

Fresh PR35 predictions exactly reproduce its prior PR79 audit on all 20 photographs. **The fresh r12 control is not prediction-identical to PR35:** only the three empty-scene arrays are equal, while all 17 nonempty scenes differ. Their distinct results are therefore reported separately. [Prediction consistency](prediction-consistency.json)

## Frozen primary reference: 666 fronts

| Model | Matched | Misses | Unmatched outputs | Duplicate subset | Both corners <=5px | Both <=10px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained PR35 | 648 | 18 | 80 | 32 | 533/584 | 582/584 |
| Fresh r12 control | 646 | 20 | 83 | 29 | 531/584 | 583/584 |
| r12 candidate | 637 | 29 | 97 | 31 | 528/584 | 581/584 |

Relative to its fresh control, candidate has **9 more misses, 14 more unmatched outputs, three fewer five-pixel pair successes, and two fewer ten-pixel pair successes**. It gains three primary identities and loses 12; every identity and best-overlap proposal is in the complete comparison.

| Locatable reference corners | Fronts | PR35 misses | Control misses | Candidate misses |
| --- | ---: | ---: | ---: | ---: |
| Zero | 23 | 10 | 10 | 13 |
| One | 59 | 7 | 9 | 14 |
| Two | 584 | 1 | 1 | 2 |

Thus 27 of the candidate's 29 primary misses belong to the partial/zero-corner groups. The pair denominator is the 584 reference fronts with two locatable corners; eligible missed detections remain failures. It is not a complete-scene accuracy denominator.

## Historical diagnostic: 663 fronts

| Model | Matched | Misses | Unmatched outputs |
| --- | ---: | ---: | ---: |
| Retained PR35 | 648 | 15 | 80 |
| Fresh r12 control | 646 | 17 | 83 |
| r12 candidate | 638 | 25 | 96 |

The historical diagnostic also disfavors candidate: eight more misses and 13 more unmatched outputs than control. Its pair-success counts match the primary table. The frozen references retain their exact manifest hashes and unresolved rows.

## Pre-return r12 review: 662 fronts, exploratory

The current r12 review was pinned in the training protocol **before the returned results**. Its manifest SHA-256 is `2ccac69fd2cff2c5748a6063c743271b5dd92c3f1f7493542486d1eddb31602f`. It contains 20 development photographs, 662 fronts, 1,226 locatable endpoints and 98 missing slots, including three empty scenes. Its unresolved photo 089 case remains explicitly recorded.

| Model | Matched | Misses | Unmatched outputs | Duplicate subset | Both corners <=5px | Both <=10px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained PR35 | 626 | 36 | 102 | 21 | 534/584 | 573/584 |
| Fresh r12 control | 621 | 41 | 108 | 17 | 530/584 | 571/584 |
| r12 candidate | 642 | 20 | 92 | 29 | 541/584 | 581/584 |

Here the candidate has **21 fewer misses, 16 fewer unmatched outputs, 11 additional five-pixel pair successes and 10 additional ten-pixel pair successes** versus its fresh control. Against PR35 it has 16 fewer misses, 10 fewer unmatched outputs, seven more five-pixel successes and eight more ten-pixel successes. Candidate has eight miss-free nonempty scenes out of 17, versus six for each comparator. It still has more scored duplicate outputs than either comparator.

The same saved candidate outputs are being compared with different label geometry. These are not two differently tuned models. The gain is meaningful as an alignment measurement against the reviewed target, but this reused development set is exploratory and does not overturn the prospective selection rule.

All three models have zero outputs on the three empty scenes under every reference. No model has an error-free nonempty scene on the frozen primary reference; each has one on the current r12 review.

## What the changed matches mean

The current r12 comparison gains 23 matches and loses two versus control. Both lost current-reference examples and four representative gains were inspected against original-pixel crops. This bounded review is saved in [representative-case-review.json](representative-case-review.json); it is not a physical adjudication of every unmatched output.

- `new_000008:33`: candidate proposals mainly cover adjacent tiers rather than the reviewed partial front; best candidate IoU is 0.08958. This loss remains in the frozen and current references.
- `new_000009:55`: no retained candidate output overlaps the small reviewed front fragment. Its geometry is identical to frozen-primary row 56; row numbers must always name their reference.
- `new_000007:6`: the candidate fits the narrower reviewed front extent. The control's best IoU is 0.49853, just below the unchanged 0.50 cutoff. A scored miss here does not mean no cassette proposal existed.
- `new_000056:16`: the candidate matches a thin front fragment at the bottom image edge beside the hand; the control mainly covers the tier above.
- `new_000090:41` and `new_000011:59`: candidate proposals fit narrower exposed front regions, while control proposals extend farther into surrounding or occluding regions.

Boxes, corners, instance identity and occlusion are related targets but cannot be substituted for one another. A corner can be accurate while its enclosing box fails the chosen IoU match, and an unmatched output can reflect extent disagreement or duplication rather than background hallucination. These findings support making the physical target definition consistent before interpreting aggregate scores as practical robot performance.

## Post-return r13 diagnostic

The later [r13 review](../../../data/reviews/fronts-2026-11-21-r13/manifest.json), SHA-256 `e76da567c1b6ca58b814afe3dc2e04710ce2a725b506f3b35f2fe17887caa4a5`, was amended after the PR82 return. Its [separate diagnostic](post-r13-diagnostic.json) reuses the exact same cached predictions. It is not a new trained model or a replacement primary reference.

Three development label files change from r12: **new_000009, new_000011 and new_000056**. Development counts remain 662 fronts. All three models retain exactly the same matched identities and 5/10-pixel pair-success identities as the pre-return r12 review. The only count change is one fresh-control unmatched output reclassified from duplicate to other: duplicate subset 17 to 16, while total unmatched outputs remain 108. PR35 and candidate retain the table's counts.

This is reported separately to preserve the distinction between a reference pinned before training results and a later annotation revision. No extra inference or training was performed for this replay.

## Returned trainer metrics

The trainer uses the original 664-front validation labels and its own aggregate metric procedure. It does not use the same fixed-point comparison as the tables above.

| Epoch 30 metric | Fresh control | Candidate |
| --- | ---: | ---: |
| Box precision | 0.96808 | 0.95552 |
| Box recall | 0.91365 | 0.92169 |
| Box mAP50 | 0.93881 | 0.94008 |
| Box mAP50-95 | 0.57249 | 0.51704 |
| Pose precision | 0.98807 | 0.98251 |
| Pose recall | 0.93825 | 0.93072 |
| Pose mAP50 | 0.96042 | 0.96073 |
| Pose mAP50-95 | 0.96021 | 0.96047 |

Small mAP50 gains coexist with a lower box mAP50-95 and frozen-reference regressions. These aggregate values alone cannot select a replacement.

## Next experiment and practical limits

Continue the supported annotation amendments with the same paired recipe and clearly versioned inputs. Keep the frozen comparisons and the current-review diagnostic separate. Do not change architecture, confidence or NMS simply to make this one result look favorable. A future change to the primary reference should be justified by independent physical annotation decisions and declared prospectively, with the older comparison retained for continuity.

Partial fronts and duplicate outputs remain concrete improvement targets. An independent set of new camera sessions is needed to determine whether reviewed-geometry gains generalize. These image-space results do not establish millimetre positioning accuracy or grasp success, and endpoint confidence is not a calibrated probability of physical visibility or pixel accuracy.

## Reproduction

`audit_return.py` verifies the exact PR82 archive and extracted bundles against the original r12 source revision. `run_pr82_inference.py` accepts the three original complete checkpoint bundles and a fresh output directory. `measure_pr82.py` replays the saved prediction files and receipts without loading weights, reporting the frozen primary, historical and pinned pre-return r12 review separately.

Use preserved source `4f4423a511b42bc804cbcbeb7a772823cef2eb3b` for the training-source audit. Use the matching source files and original PNGs for scoring. [Replay verification](replay-verification.json) records byte-identical reproduction of the full comparison. `measure_post_r13.py` separately replays the same cached outputs against the post-return r13 manifest. No new training or model promotion is part of this audit.
