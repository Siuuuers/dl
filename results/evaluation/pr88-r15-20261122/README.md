# PR88 r15 result audit

**Keep PR35 selected.** The returned r15 candidate improves agreement with the accepted r15 review, but still regresses on the frozen primary and historical references. Against the previous r14 candidate on identical r15 labels, it gains one match at the fixed IoU boundary. Aggregate five- and ten-pixel corner-pair counts are unchanged. These results do not establish improved performance on unseen camera sessions or a physical positioning tolerance.

This audit evaluates the completed r15 experiment **before the new r16 amendments**. No training, threshold search, annotation edit or model promotion occurs here. [Summary](summary.json), [full comparison](fixed-comparison.json), [integrity audit](verified-return.json).

## Verified return and inputs

[PR88](https://github.com/Siuuuers/dl/pull/88) is pinned to head `e05a985cb2ee21be82adc164c85c99a5ad15fa82`. The direct committed archive has **39,781,855 bytes**, SHA-256 `6a323797cc05eac71a61b5a8d328dfad1c242b4506cbac51eb7e91f84971d22a`. ZIP CRC, unique safe paths, exact inventory and all 12 indexed members' hashes/sizes passed. No redundant transport-chunk download was needed.

Training source is `fe7ae1095d24ebb6afea2c57194c88b369e86921`. Notebook revision `833a1ae37f9840b105b73f2e7f36952cb3946f7a` has the exact handoff notebook hash and embeds that source pin. Protocol SHA-256 is `3b3a7f2373fb3297ffcab4d193a4e9bb07fe8970508d228c602dc78a0fb3056b`. [Notebook verification](transport-and-notebook-verification.json)

A detached checkout of the exact training source isolates this audit from r16 edits. All recorded source files, staged source/output bytes, original image/label bytes and completed checkpoint bundles passed verification. The actual 157 original PNGs and epoch 173 initializer were also rehashed. The initializer hash is `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`.

| Actual training input | Fresh control | r15 candidate |
| --- | ---: | ---: |
| Photos | 137 | 137 |
| Empty photos | 13 | 13 |
| Fronts | 3,771 | 3,786 |
| Locatable endpoints | 6,841 | 6,838 |
| Missing endpoint slots | 701 | 734 |
| Original validation photos/fronts | 20/664 | 20/664 |

Exactly **92 training-label files differ**. Held photo 015 and the separate 31 old-series treatment exclusions use identical PR35 supervision in both arms; all remain in training. Images and original validation labels are identical between arms.

Both arms independently started from epoch 173 and completed **30 epochs, 1,050 batches, 4,110 image presentations and 78 recorded optimizer steps**, with exit code 0 and the same NVIDIA L4 runtime fingerprint. Seed 46, batch 4, input 1280, FP32, AdamW 0.0001 and final-epoch selection are unchanged. Receipt wall times were 470.89 seconds for control and 466.10 seconds for candidate; they are not a controlled speed benchmark. [Preserved receipts](checkpoint-receipts/)

| Model | Checkpoint SHA-256 |
| --- | --- |
| Selected PR35 | `420c00f73eeba24e587342400878b7d96371d3e745772113809aa705ec8163b1` |
| Fresh r15 control | `32d2e1ea714b73797937f428fcedffac61dd86af03040013524526160496f457` |
| Returned r15 candidate | `32783de8ca001cd959b3fa67e814061577537151f442c48977e81003704cada4` |

## Fixed inference and scoring

All three models were freshly inferred in one CPU environment: Python 3.12.14, Torch 2.14.1+cpu, torchvision 0.29.1+cpu, Ultralytics 8.4.182, NumPy 2.4.6, Pillow 11.3.0 and OpenCV 5.0.0.93. Settings were box confidence 0.13, endpoint confidence 0.50, NMS IoU 0.70, matching IoU 0.50, input 1280, FP32, maximum 300 outputs and four CPU threads. [Environment](inference-pip-freeze.txt), [commands](inference/commands.json), [script adaptation](measurement-adaptation.patch)

Both fresh baseline prediction arrays are **exactly equal on all 20 images** to the saved PR86 PR35/control arrays. The returned control checkpoint has a different artifact hash, but its observed arrays coincide at this operating point. File hashes differ because provenance/timing headers differ. Maximum coordinate and confidence differences are zero; matched and corner-success identities are unchanged. [Raw-array consistency](prediction-consistency.json)

A pair succeeds only when the object is matched and **both locatable endpoints** pass the confidence, image-boundary and pixel-distance criteria. Eligible missed objects remain failures in the denominator of 584. Zero- and one-corner objects are still included in detection results. Unmatched outputs are fixed-matching outcomes and have not all been independently judged to be background false detections.

## Frozen primary: 666 fronts

| Model | Matched | Misses | Unmatched outputs | Duplicate subset | Pairs within 5 px | Pairs within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PR35 | 648 | 18 | 80 | 32 | 533/584 | 582/584 |
| Fresh control | 646 | 20 | 83 | 29 | 531/584 | 583/584 |
| r15 candidate | 631 | 35 | 101 | 29 | 521/584 | 575/584 |

The candidate has 15 more misses and 18 more unmatched outputs than control. Its 35 misses comprise 13 zero-corner fronts, 15 one-corner fronts and 7 two-corner fronts. This fails to establish an improvement under the fixed reference.

## Historical: 663 fronts

| Model | Matched | Misses | Unmatched outputs |
| --- | ---: | ---: | ---: |
| PR35 | 648 | 15 | 80 |
| Fresh control | 646 | 17 | 83 |
| r15 candidate | 631 | 32 | 101 |

Corner-pair counts equal the primary table. These references preserve their original annotation bytes and uncertainty records.

## Pre-return r15 review: 662 fronts, exploratory

This reference was pinned before PR88 returned. Manifest SHA-256 is `6012d09767851151312537c2046022b972bf6519f36d110675d3e82f74694e93`. It contains 20 development photos, 662 fronts, 1,226 locatable endpoints, 98 missing slots and three empty scenes, with no unresolved development cases.

| Model | Matched | Misses | Unmatched outputs | Duplicate subset | Pairs within 5 px | Pairs within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PR35 | 627 | 35 | 101 | 21 | 538/584 | 573/584 |
| Fresh control | 623 | 39 | 106 | 16 | 533/584 | 572/584 |
| r15 candidate | 646 | 16 | 86 | 29 | 549/584 | 583/584 |

Compared with control, candidate has 23 fewer misses, 20 fewer unmatched outputs, 16 more five-pixel successes and 11 more ten-pixel successes. It gains 27 matched identities and loses 4; its duplicate subset rises from 16 to 29 despite fewer unmatched outputs overall. It is miss-free on 8 of 17 nonempty scenes, versus 6 for each baseline. Candidate misses comprise 7 zero-corner and 9 one-corner fronts; all 584 two-corner fronts are matched. All three models produce zero outputs on all three empty scenes.

The same prediction arrays produce the frozen and reviewed tables. This ranking change reflects reference sensitivity. The favorable exploratory table does not replace the fixed selection rule, and repeatedly reused development scenes do not establish unseen-session generalization.

### Change from r14 on identical r15 labels

The prior candidate's saved predictions were replayed on the same r15 reference; the prior diagnostic was reproduced byte-for-byte. [Prior-generation replay](prior-generation-replay-verification.json)

| Candidate generation | Misses | Unmatched outputs | Duplicate subset | Pairs within 5 px | Pairs within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| PR86 r14 | 17 | 87 | 30 | 549/584 | 583/584 |
| PR88 r15 | 16 | 86 | 29 | 549/584 | 583/584 |

The only gained match is `new_000011:56`, with no lost matches. At 5 px, `new_000090:19` gains success and `new_000090:46` loses it. At 10 px, success identities are identical. Aggregate equality at 5 px therefore does not mean identical successful objects.

## Four bounded image checks

Original full-frame context, unmodified crops and candidate overlays were inspected. [Review panel](representative-cases.png), [exact case evidence](representative-case-review.json)

- **011 row 56:** the gripper covers the front's right side. Both candidates propose nearly the same box; IoU moves from 0.496218 to 0.504338 across the fixed 0.50 boundary. This one-corner object does not enter the two-corner denominator. The predicted right endpoint has no verified reference coordinate.
- **090 row 19:** right-corner error moves from 5.025645 to 4.986816 px, just crossing the 5 px criterion while both boxes stay matched.
- **090 row 46:** right-corner error moves from 4.501719 to 5.104240 px. This loses 5 px success but preserves 10 px success.
- **008 row 10:** both boxes remain matched; right-corner error is 12.022668 then 12.107580 px, with confidence around 0.91. It remains the sole r15 two-corner failure at 10 px. Confidence does not guarantee coordinate accuracy.

These checks explain specific score changes without modifying annotations or adjudicating all outputs. Pixel radii are image-space evaluation criteria; they are not measured millimetre tolerances of a robot or gripper.

## Reproduction and next experiment

Use source commit `fe7ae1095d24ebb6afea2c57194c88b369e86921` with original photos and epoch 173 bytes for `audit_return.py`. `run_pr88_inference.py` takes that source, the fixed Python runtime, complete PR35/returned checkpoint manifests and a fresh output directory. `measure_pr88.py` replays saved predictions without loading model weights.

The recorded comparison was replayed from the saved arrays and preserved receipt-only folders with **identical output bytes**, SHA-256 `6231a16b09d81634d8d7323a0e5853b1256d4d0d6ce0b6ebfb1bd1a3c9b1a91a`. Original `train.log` bytes are saved as `train-log.txt`; the strict receipt validator temporarily restores their original filename. [Replay verification](replay-verification.json), [log preservation](log-preservation.json)

The new r16 corrections are a later data revision. Any replay against them must be separately named as a post-return diagnostic and adds no training evidence. A new pinned notebook can test r16 supervision with the existing architecture, initialization, recipe and thresholds. Keep PR35 selected until the prospective selection criteria support a change.

## Later r16 reference: cached diagnostic only

The separate [post-r16 diagnostic](post-r16-diagnostic.json) reuses the exact three PR88 prediction files above. **No inference, training, threshold change or checkpoint change occurred.** Its reference is `data/reviews/fronts-2026-11-22-r16/manifest.json`, SHA-256 `1645f999fa8be08e9c8a317ffccd88e54285a81aac181a8e1ec8c9b328b81993`.

Only development photos **009 and 010** changed in r16; requested photos 007 and 008 retain their r15 label bytes. The revised reference still has 20 photos, 662 fronts, 1,226 locatable endpoints, 98 missing slots and three empty scenes, with no unresolved development cases. Equal counts do not mean equal geometry.

| Existing model, scored against r16 labels | Matched | Misses | Unmatched outputs | Duplicate subset | Pairs within 5 px | Pairs within 10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PR35 | 627 | 35 | 101 | 21 | 538/584 | 573/584 |
| Fresh r15 control | 624 | 38 | 105 | 16 | 534/584 | 573/584 |
| Returned r15 candidate | 646 | 16 | 86 | 29 | 549/584 | 583/584 |

The unchanged control gains match `new_000009:19` and its five- and ten-pixel successes. Relative to its pre-return r15-reference score, that means one fewer miss, one fewer unmatched output and one additional pair success at each radius; its duplicate count is unchanged. PR35 and the returned r15 candidate have unchanged detection counts, matched identities and corner-success identities. All three models still emit zero outputs on all three empty scenes.

On this later reference, candidate has 22 fewer misses and 19 fewer unmatched outputs than control, with 15 more five-pixel successes and 10 more ten-pixel successes. These are scores of existing models against later labels; the difference from the previous table comes entirely from reference amendments. This diagnostic cannot replace the frozen comparison or the pre-return r15 score. **PR35 remains selected.**

The cached diagnostic was replayed with identical output bytes, SHA-256 `0d211366190641db13b6e8c702fa6d6e18688f1103242c58e793b911b87cfcc8`. The frozen comparison and all three prediction files were verified unchanged. [Replay verification](post-r16-replay-verification.json)

Reproduce from saved arrays and receipt folders:

```sh
python results/evaluation/pr88-r15-20261122/measure_post_r16.py \
  --root . --output /path/to/fresh-post-r16-diagnostic.json
```
