# PR84 r13 result audit

**Keep PR35 selected. The r13 candidate improves alignment with the reviewed fronts but does not meet the existing frozen-reference selection rule.** Against its fresh control, it has 21 fewer misses on the pre-return r13 review, but eight more misses on the frozen primary reference. Relative to the previous r12 candidate on the same r13 review, it preserves all matched object identities and all ten-pixel pair-success identities, while unmatched outputs increase from 92 to 94.

This is a measured result, not a new training run or a model promotion. All three models were freshly inferred in one fixed CPU environment. The [complete comparison](fixed-comparison.json), [summary](summary.json), [integrity audit](verified-return.json), and [six physical case reviews](representative-case-review.json) preserve the evidence. Newly requested r14 annotation edits are not used to decide this result.

## Verified training return

[PR84](https://github.com/Siuuuers/dl/pull/84) is pinned to head `e881e38b63b8299dce22bae6b472134c131c1ff7`. The returned archive contains **39,781,828 bytes**, SHA-256 `6143530679a6c9a972119852b9f4922d8e7143bb01cfbd6edc52d0ecd8e70f4c`. ZIP CRC, unique safe member paths, exact inventory, and hashes and sizes of all 12 indexed members passed. Both transport chunk ZIPs and their raw parts also passed their published hashes; concatenating the raw parts reproduced the directly retrieved archive byte for byte. [Transport verification](transport-and-notebook-verification.json)

Training used source `5c2817f1c1eaa1bbce7060facb82732c3d249e10` and [r13 protocol](../../../experiments/front-review-2026-11-21-r13.json), SHA-256 `b77ab22c5655146b05a870458fcad080f4e6e9dcc890ae248564b86147a9f22d`. The notebook at commit `4795910cdeb37d23d4b99031eb480c94b883b478` matches its published SHA-256 `de0146574fff097da7edf45bbd8fdb591569add7dff513fdb63df5b970fbea96`.

A detached checkout of the exact training source was used. Every recorded source hash, staged image/label hash, completed checkpoint bundle, original PNG and original label was checked. The original epoch173 initialization bytes were downloaded and verified again. Subsequent r14 code or annotation changes cannot alter this audit's source.

| Actual training input | Fresh control | r13 candidate |
| --- | ---: | ---: |
| Training photos | 137 | 137 |
| Empty training photos | 13 | 13 |
| Front instances | 3,771 | 3,777 |
| Locatable endpoints | 6,841 | 6,845 |
| Missing endpoint slots | 701 | 709 |
| Original validation photos / fronts | 20 / 664 | 20 / 664 |

The arms differ on **78 training label files**. The 15 then-held training photos and the separate 31 old-series exclusions have exact PR35 supervision in both arms. This records the actual returned experiment; any later approval releasing held annotations belongs to a new experiment.

Both arms independently started from epoch173, used seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW at 0.0001, and selected final epoch 30. Each completed 1,050 batches, 4,110 image presentations and 78 recorded optimizer steps on NVIDIA L4, with exit code 0. Recorded training wall time is 454.80 seconds for control and 451.96 seconds for candidate; these receipts are not a controlled speed benchmark. [Preserved receipts](checkpoint-receipts/)

| Checkpoint | SHA-256 |
| --- | --- |
| Selected PR35 | `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` |
| Fresh r13 control | `200a791cacbb9511515b06f2216685d831cebd8adfa225800362ebd074dae9bd` |
| r13 candidate | `fef17fac7cf2b56435b655087740c5fe2a583ea4f2ca4531076df3d641942d0b` |

## Fixed inference and reproducibility

All three checkpoints were freshly inferred with Python 3.12.14, Torch 2.14.1+cpu, torchvision 0.29.1+cpu, Ultralytics 8.4.182, NumPy 2.4.6, Pillow 11.3.0 and OpenCV 5.0.0.93. [Package inventory](inference-pip-freeze.txt) and [actual commands](inference/commands.json) record the environment and completed runs.

Settings remain box confidence 0.13, endpoint confidence 0.50, ordinary NMS IoU 0.70, matching IoU 0.50, input 1280, FP32, maximum 300 outputs and four CPU threads. The evaluator and scoring implementations retain their hashes. [Measurement adaptation](measurement-adaptation.patch) changes experiment identities, the separately identified r13 review loader, and restoration of preserved log filenames for offline receipt verification; it does not change matching or scoring rules.

The fresh PR35 arrays are **not bit-identical** to the prior saved PR82 arrays. All 17 nonempty arrays contain small numerical differences; the three empty arrays are identical. The maximum coordinate difference is 0.0001220703125 pixels and maximum confidence difference is approximately 0.000001639. The fresh r13 control similarly differs numerically from the saved r12 control by at most 0.0001220703125 pixels and approximately 0.000001461 confidence. Their recorded main runtime versions match; the audit does not attribute the variation to an unverified cause.

These tiny differences leave each baseline's matched identities and five/ten-pixel success identities unchanged on the same r13 reference. The fresh r13 control is a distinct comparator from PR35 and is reported separately. Raw values are retained without rounding to force equality. [Prediction consistency](prediction-consistency.json)

## Frozen primary reference: 666 fronts

| Model | Matched | Misses | Unmatched outputs | Duplicate subset | Both corners <=5 px | Both <=10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected PR35 | 648 | 18 | 80 | 32 | 533/584 | 582/584 |
| Fresh r13 control | 646 | 20 | 83 | 29 | 531/584 | 583/584 |
| r13 candidate | 638 | 28 | 98 | 32 | 529/584 | 581/584 |

Candidate has eight more misses, 15 more unmatched outputs, two fewer five-pixel pair successes and two fewer ten-pixel pair successes than control. It gains four matched identities and loses 12. Relative to selected PR35 it has ten more misses and 18 more unmatched outputs.

| Locatable reference corners | Fronts | PR35 misses | Control misses | Candidate misses |
| --- | ---: | ---: | ---: | ---: |
| Zero | 23 | 10 | 10 | 13 |
| One | 59 | 7 | 9 | 13 |
| Two | 584 | 1 | 1 | 2 |

Thus 26 of 28 candidate misses belong to fronts with fewer than two locatable corners. The 584-pair denominator contains reference fronts with two locatable corners; eligible missed detections remain failures. It is not a complete-scene accuracy denominator or a measurement of robotic grasp success.

## Historical reference: 663 fronts

| Model | Matched | Misses | Unmatched outputs |
| --- | ---: | ---: | ---: |
| Selected PR35 | 648 | 15 | 80 |
| Fresh r13 control | 646 | 17 | 83 |
| r13 candidate | 639 | 24 | 97 |

Candidate has seven more misses and 14 more unmatched outputs than control. Its pair-success counts match the primary table. Both frozen references preserve their original bytes and recorded unresolved rows.

## Pre-return r13 review: 662 fronts, exploratory

This review was pinned in the training protocol before results returned. Its manifest SHA-256 is `e76da567c1b6ca58b814afe3dc2e04710ce2a725b506f3b35f2fe17887caa4a5`. It contains 20 development photos, 662 fronts, 1,226 locatable endpoints and 98 missing endpoint slots, including three empty photos. Its then-unresolved photo 089 case is preserved in this historical comparison.

| Model | Matched | Misses | Unmatched outputs | Duplicate subset | Both corners <=5 px | Both <=10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected PR35 | 626 | 36 | 102 | 21 | 534/584 | 573/584 |
| Fresh r13 control | 621 | 41 | 108 | 16 | 530/584 | 571/584 |
| r13 candidate | 642 | 20 | 94 | 31 | 541/584 | 581/584 |

Candidate has **21 fewer misses, 14 fewer unmatched outputs, 11 more five-pixel pair successes and ten more ten-pixel pair successes** than control. It gains 23 matched identities and loses two. Against PR35 it has 16 fewer misses, eight fewer unmatched outputs, seven more five-pixel pair successes and eight more ten-pixel pair successes.

Candidate has eight miss-free nonempty scenes out of 17; both comparators have six. Its 20 misses comprise eight zero-corner fronts, 11 one-corner fronts and one two-corner front. Its duplicate subset is larger than either comparator's, despite fewer unmatched outputs overall. All three models emit zero outputs on the three empty scenes.

The same fresh outputs produce the frozen and reviewed comparisons. The reversal measures sensitivity to annotation geometry. It supports improved alignment with the reviewed target while preserving the limits of this repeatedly reused development cohort. It does not justify silently replacing the prospective selection rule with whichever reference scores best.

### What changed since PR82 on identical r13 review geometry?

The [previous post-r13 replay](../pr82-r12-20261121/post-r13-diagnostic.json) lets the two candidate generations be compared with exactly the same reference.

| Candidate generation | Matched | Misses | Unmatched outputs | Duplicate subset | Pairs <=5 px | Pairs <=10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PR82 r12 | 642 | 20 | 92 | 29 | 541/584 | 581/584 |
| PR84 r13 | 642 | 20 | 94 | 31 | 541/584 | 581/584 |

All matched object identities and all ten-pixel pair-success identities are unchanged. At five pixels, `new_000090:46` gains success while `new_000008:15` loses success, leaving the total unchanged. The latest small annotation amendment therefore has not established a further model improvement over r12 on this target. Unmatched-output totals increase by two. The exact differences are recorded in [summary.json](summary.json).

## Original-image interpretation

Six bounded cases were inspected using original RGB crops and actual fresh predictions. This is not an adjudication of every unmatched output.

- **Current r13 `new_000008:33`:** the right-edge partial front has a control overlap of 0.60125; the best candidate overlap is only 0.09004 and mainly belongs to the tier below.
- **Current r13 `new_000009:55`:** control overlaps the small exposed front by 0.54728. No candidate output overlaps it at the fixed operating point. This maps to primary row56 with the same geometry.
- **Primary `new_000087:15`:** both models already propose the cassette tier. Control/candidate IoUs of 0.49516/0.50705 straddle the fixed 0.50 match threshold. This additional primary match is not evidence of discovering a previously absent physical cassette.
- **Current r13 `new_000007:6`:** control already emits a proposal, with IoU 0.49853. Candidate fits the reviewed face more tightly, reaching 0.74759; the control box includes more rack below.
- **Current r13 `new_000056:16`:** candidate overlaps a thin front fragment at the image edge beside the hand. Control's best overlap mainly belongs to the tier above.
- **Current r13 `new_000090:41`:** candidate better fits the narrow exposed front beside an upright occluder, while control includes more surrounding area. Neither predicted right-point coordinate certifies the unlocatable physical corner.

A scored miss can be an existing proposal below an IoU threshold, and an unmatched output can be a duplicate or an extent disagreement. Box extent, corner accuracy and physical object recovery must be interpreted together.

## Returned trainer metrics

The trainer uses original 664-front validation labels and its own aggregate metric procedure. These values are not the fixed-point tables above.

| Epoch30 metric | Fresh control | Candidate |
| --- | ---: | ---: |
| Box precision | 0.96808 | 0.95472 |
| Box recall | 0.91365 | 0.92082 |
| Box mAP50 | 0.93881 | 0.93503 |
| Box mAP50-95 | 0.57249 | 0.51241 |
| Pose precision | 0.98807 | 0.98251 |
| Pose recall | 0.93825 | 0.93058 |
| Pose mAP50 | 0.96042 | 0.96068 |
| Pose mAP50-95 | 0.96021 | 0.96041 |

The small pose aggregate gains do not establish a replacement model. The original [control CSV](checkpoint-receipts/reference/fit/results.csv) and [candidate CSV](checkpoint-receipts/candidate/fit/results.csv) retain all epochs.

## Next experiment

The requested physical annotation amendments and explicit approval of previously rendered amber fields change the supervision available for the next round. The [completed r14 package](../../../docs/front-review-r14.md) records 233 changed rows in 30 photos and releases 14 previously held training photos; the unboxed photo 015 question remains held. The next matched experiment tests these annotation and eligibility changes together. It cannot attribute any result to either component alone. The returned r13 candidate still used the older 15-photo hold; later approval cannot retroactively change its training inputs.

Keep architecture, initialization, training budget, seed, confidence and NMS fixed while testing the amended supervision. Partial fronts and duplicate proposals are concrete remaining targets. The evidence does not support a larger architecture or another threshold search as the immediate next step.

For generalization, eventually collect a separate set of camera sessions after the target definition is stable. This development cohort has been reused across many decisions. Any future change of primary evaluation target should be declared prospectively, based on physical annotation decisions, with the frozen historical comparison retained for continuity.

## Separate post-return r14 reference replay

The later r14 review changes development labels for photos 007, 008, 009, 010, 053, 054, 055 and 059. Its manifest is pinned separately at SHA-256 `95b3c8bd4746281f9474ecd787a063634ee1b6750d758f4454ddaaaa7479ef0f`. Object and endpoint counts remain 662 fronts, 1,226 locatable endpoints and 98 missing slots across the same 20 photos. The user's approval resolves the previously displayed photo 089 case without changing that photograph's label bytes. Equal counts do not mean equal reference geometry.

The [post-r14 diagnostic](post-r14-diagnostic.json) reuses the exact saved predictions above, with no inference, training, threshold or checkpoint change. This later target is exploratory only and does not enter the pre-return selection decision.

| Existing model on post-return r14 review | Matched | Misses | Unmatched outputs | Duplicate subset | Both corners <=5 px | Both <=10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected PR35 | 628 | 34 | 100 | 21 | 540/584 | 575/584 |
| Fresh r13 control | 624 | 38 | 105 | 16 | 535/584 | 574/584 |
| r13 candidate | 643 | 19 | 93 | 31 | 543/584 | 582/584 |

Relative to the pre-return r13 review, PR35 gains matches for photo 010 rows 26/27, control gains 010 rows 26/27/28, and candidate gains 010 row 26. No matched identity is lost. At 5 px the candidate gains 008 rows 3/15/17 and loses 008 row 24; at 10 px it gains 008 row 5 and 010 row 26 and loses 008 row 32. All models still emit zero outputs on the three empty scenes. These changes arise entirely from the target revision: model weights and prediction arrays are unchanged. This is a recorded baseline for later r14 comparisons, not new evidence that training improved a model.

## Reproduction

Use the exact training source `5c2817f1c1eaa1bbce7060facb82732c3d249e10` with original PNGs and epoch173 bytes for `audit_return.py`. The script verifies the returned archive, extracted checkpoint bundles, recorded code and staged inputs. `run_pr84_inference.py` accepts the original complete PR35 and returned r13 bundles, a compatible Python runtime, and a fresh output directory; no training runs.

`measure_pr84.py` scores the saved prediction arrays without loading weights. It also accepts the preserved receipt folders in this audit: the original byte-identical `train-log.txt` is restored temporarily as `train.log` for the unchanged strict receipt validator. This avoids changing the original receipt or losing logs to repository ignore rules. [Log-preservation record](log-preservation.json)

The final comparison was reproduced from both complete returned bundles and the saved receipt-only folders, with identical output bytes. [Replay verification](replay-verification.json) records that result. No later reference changes enter `fixed-comparison.json`. The separately named post-return reference replay can be reproduced with:

```bash
python3 -B results/evaluation/pr84-r13-20261121/measure_post_r14.py \
  --root . --output /path/to/fresh-post-r14-diagnostic.json
```

The replay verifies the exact r14 reference, all development-label hashes, cached prediction hashes, checkpoint receipt identities and unchanged scorer code. No new training, threshold search or model promotion is performed.
