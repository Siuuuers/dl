# PR121 R36 fixed-inference review

The R36-label native candidate does not recover a missed front relative to its R35-label control on any preserved reference. Current R36 misses stay at 15, corner-error rows stay at 19, and unmatched outputs increase from 93 to 94. Frozen666 remains the primary historical comparison; PR35 remains selected. Current reviewed labels are a development diagnostic, not an independent holdout.

## Verified provenance

| Item | Exact identity |
|---|---|
| Returned artifact PR | https://github.com/Siuuuers/dl/pull/121 |
| Artifact commit | `f7ad5dd9297a837bd052995fdf842e2d73b0cd1d` |
| Published training source | `83aeca67264e1f98872abeca40debc212c0775e9` |
| Full archive bytes | 39797960 |
| Full archive SHA256 | `20714596f21eb9728f7a33d1535592ec18aa03980ff6108ec598e78ae96286ba` |
| GPU results manifest SHA256 | `f7e7f6f3d7b94b211d3fcbbfbb4b414b8539d08257bd46e9ca0a982ceab103e9` |
| Protocol SHA256 | `aa0f50570ac06b8b9da95938955ee53413f25087e53f53779fb3fc5f317c970b` |
| R35-label control last.pt SHA256 | `68b5bcc1c3e940b0db7630938391938b6fd51aeb4475739afd478ac96fbf1070` |
| R36-label candidate last.pt SHA256 | `f501bb80f7e77f6a35cf72630ca5ff2a3b2169690ee5d9852919d1cbd2661ba4` |
| Control prediction cache SHA256 | `d33484b9e4abf2cc4679cd7bafa0d3e7221820f3c6f4d64be2f940ef7f6e293d` |
| Candidate prediction cache SHA256 | `6f718c4344eb20b05d95703570d92d5d2112f166f5beaabb3cfb7e27273b3382` |
| R36 reference manifest SHA256 | `d0cdb9d56c043175ca12e538951cb28b1a8cd897039cce7a971e8f046407d416` |
| Original epoch-173 initializer SHA256 | `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e` |

The returned archive is stored as two wrapped chunks under `results/gpu/baomai-front-review-r36-l4-001` at the artifact commit. Unzip each wrapper, concatenate raw part00 then part01, and verify the full digest above. Wrapper hashes, raw part hashes, all 15 archive-member hashes, receipts, staging identities, and source-file hashes are recorded in `archive-audit.json`. ZIP integrity and safe member paths were checked before extraction.

Both arms passed the exact-source native verifier and paired-run checks: native Pose26 loss, seed 46, independent original epoch 173 initialization, final epoch 30, 1,050 batches, 4,110 image presentations, and 78 optimizer steps. Both use the same L4 runtime and numerical recipe. The accepted training difference is exactly two box-only rows on one photo, `new_000034:34` and `new_000034:35`. No target was added or removed; classes, points, and visibility stay unchanged. The source verification used a detached checkout of the recorded source commit, not the mutable R37 working tree. All 157 original PNGs and the actual initializer were materialized and hash-verified.

Fresh inference was run on both returned checkpoints in a common CPU environment: Python3.12.14, PyTorch2.10.0+cpu, torchvision0.25.0+cpu, Ultralytics8.4.182, NumPy2.3.5, Pillow13.0.0, and OpenCV5.0.0.93. The complete runtime fingerprint is in the cache headers. Original image geometry, FP32, size 1280, rect=True, batch1, max_det300, quantize32, one-to-many output with NMS IoU 0.70, box confidence 0.13, keypoint confidence 0.50, and scoring IoU 0.50 were preserved. No retraining, confidence sweep, NMS sweep, or latency comparison was performed.

PR121 control predictions are **exactly identical to PR119 candidate predictions on all 20 images**, despite different checkpoint file hashes. All indexwise box, point, and confidence differences are zero. Earlier PR119 control replay had tiny floating-point differences from its predecessor; that older fact must not be copied onto this round.

## Primary results

| Reference | Model | Missed fronts | Unmatched | Duplicate subset | Both corners <=5px | Both corners <=10px |
|---|---|---:|---:|---:|---:|---:|
| R36, 659 fronts | R35-label control | 15 | 93 | 31 | 570/583 | 582/583 |
| R36, 659 fronts | R36-label candidate | 15 | 94 | 31 | 570/583 | 582/583 |
| R36, 659 fronts | PR35 selected | 43 | 112 | 15 | 546/583 | 569/583 |
| Frozen666 | R35-label control | 36 | 107 | 29 | 517/584 | 573/584 |
| Frozen666 | R36-label candidate | 36 | 108 | 29 | 518/584 | 573/584 |
| Frozen666 | PR35 selected | 18 | 80 | 32 | 533/584 | 582/584 |
| Historical663 | R35-label control | 32 | 106 | 29 | 517/584 | 573/584 |
| Historical663 | R36-label candidate | 32 | 107 | 29 | 518/584 | 573/584 |
| Historical663 | PR35 selected | 15 | 80 | 32 | 533/584 | 582/584 |

The current candidate emits 738 outputs. All 20 images are included, including three empty scenes (`new_000146`, `new_000147`, `new_000148`); both arms emit no outputs in those scenes. All 583 two-corner fronts match. Partial fronts account for all 15 misses: three of 18 zero-corner fronts and twelve of 58 one-corner fronts. There are 19 matched rows with at least one available corner more than 5 original-image pixels away or unavailable at the fixed keypoint gate. The radii are review diagnostics, not calibrated robot tolerances.

No missed identities change in the paired experiment. Frozen666/historical663 gain one five-pixel pair, at row `new_000010:20`; this does not change the current-review corner-error ledger. Pair denominators include eligible missed objects.

### Every current miss, with geometric triage

| R36 reference identity | Area (px2) | Available corners | Best output IoU | Geometric description |
|---|---:|---:|---:|---|
| `new_000008:33` | 2278.0 | 1 | 0.0173 | very low overlap (<0.10) |
| `new_000008:34` | 2706.0 | 1 | 0.0364 | very low overlap (<0.10) |
| `new_000009:52` | 210.0 | 1 | 0.4438 | overlap below 0.50 gate |
| `new_000009:55` | 874.0 | 0 | 0.0000 | very low overlap (<0.10) |
| `new_000010:31` | 2088.0 | 1 | 0.3747 | overlap below 0.50 gate |
| `new_000010:34` | 792.0 | 0 | 0.0054 | very low overlap (<0.10) |
| `new_000011:55` | 558.0 | 1 | 0.4887 | overlap below 0.50 gate |
| `new_000011:56` | 320.0 | 1 | 0.4033 | overlap below 0.50 gate |
| `new_000011:57` | 186.0 | 1 | 0.3369 | overlap below 0.50 gate |
| `new_000011:60` | 312.0 | 0 | 0.0728 | very low overlap (<0.10) |
| `new_000053:1` | 470.0 | 1 | 0.2397 | overlap below 0.50 gate |
| `new_000086:57` | 184.0 | 1 | 0.0000 | very low overlap (<0.10) |
| `new_000088:51` | 655.8 | 1 | 0.4578 | overlap below 0.50 gate |
| `new_000088:57` | 216.0 | 1 | 0.0000 | very low overlap (<0.10) |
| `new_000090:39` | 1071.0 | 1 | 0.4927 | overlap below 0.50 gate |

Seven misses have best overlap below 0.10; eight have an output overlapping below the unchanged 0.50 matching gate. This is geometric attribution only. It does not prove whether an object was physically recognized, whether a reference is correct, or whether a learned confidence change would recover it. Zero-IoU ties have no arbitrarily assigned nearest pID.

## Size and corner availability

Bins were fixed descriptively before reading PR121 counts. All measurements are in original 1280x720 image pixels. Short side is min(width,height); height is ymax-ymin; area is width*height. Area<1,024px2 does not mean both dimensions are below32px.

| R36 reference area | Fronts | Misses | Matched | Duplicate outputs associated with these fronts |
|---|---:|---:|---:|---:|
| <1,024px2 | 29 | 11 | 18 | 15 |
| 1,024 to <4,096px2 | 495 | 4 | 491 | 9 |
| 4,096 to <16,384px2 | 135 | 0 | 135 | 7 |
| >=16,384px2 | 0 | 0 | 0 | 0 |

Thus 11/15 misses are small by this predeclared area bin, but 18/29 such references are already matched. All five small two-corner fronts are matched. The 15 small-associated duplicates do not explain the other 16 duplicates associated with larger references. Across all sizes, 11 duplicate outputs associate with two-corner fronts, 15 with one-corner fronts, and 5 with zero-corner fronts. Twenty references have short side<16px (7 misses); 18 have height<16px (5 misses). All dimension tables, cross-tabs, output-size tables, and exact associations are in `size-visibility-audit.json`.

For duplicates, association selects the highest-IoU same-class **already matched reference** at IoU>=0.50. For other-unmatched outputs, it selects the highest positive IoU reference, which is a proximity statistic rather than physical identity. Eight other-unmatched outputs overlap no reference and are counted separately. Size, perspective, corner availability, clipping, and occlusion are confounded; these counts establish association, not causation or a validated deletion policy.

### Reference-only deletion thought experiment

Hold all 738 candidate outputs fixed, remove reference rows with area strictly<1,024px2 **only in memory**, and rerun the unchanged matcher. This removes 29 R36 reference rows: 18 matched and 11 missed, including 5 two-corner-eligible targets. No retained-row assignment changes occur.

| R36 quantity | Preserved targets | Hypothetical exclusion |
|---|---:|---:|
| Reference fronts | 659 | 630 |
| Matched fronts | 644 | 626 |
| Missed fronts | 15 | 4 |
| Unmatched outputs | 94 | 112 |
| Duplicate subset | 31 | 16 |
| Other-unmatched | 63 | 96 |

The lower miss count comes from removing requirements. The lower duplicate count largely reclassifies outputs, while total unmatched rises. This is not a model improvement or a prediction of retraining behavior. Canonical references and training labels are unchanged. Frozen666 is reported separately in the JSON: 17 removed targets include 5 matched and 12 missed; unmatched rises108->113 and duplicates29->23.

### Reference-overlap screen

No R36 reference pair has IoU>=0.50, and no pair has equal box coordinates. Two pairs exceed90% containment of the smaller rectangle: `new_000010:29/30` (IoU0.212) and `new_000053:1/2` (IoU0.105). These are **physical review flags, not established duplicate annotations**. Sloping/clipped neighboring fronts can have contained axis-aligned rectangles. The original-pixel crops and prior top-clipped-front review do not establish duplicate objects automatically. Neither pair is deleted or relabelled by this audit.

## One-to-one head probe: kept separate

A single bounded probe selected the candidate checkpoint's already trained one-to-one branch with `nms=False`, leaving all other inference and scoring gates fixed. The helper used a fresh model instance and verified actual `backend_end2end=True`, CPU FP32, retained one-to-one modules before fusion, and learned tensor differences from epoch 173. The stored IoU 0.70 parameter is **inactive** on the NMS-free branch. This is not an NMS-threshold sweep or a new trained model.

| Reference | Head | Misses | Unmatched | Duplicate subset | Other-unmatched | Pairs <=5px | Pairs <=10px |
|---|---|---:|---:|---:|---:|---:|---:|
| R36 | Primary one-to-many | 15 | 94 | 31 | 63 | 570/583 | 582/583 |
| R36 | Probe one-to-one | 16 | 67 | 37 | 30 | 572/583 | 581/583 |
| Frozen666 | Primary one-to-many | 36 | 108 | 29 | 79 | 518/584 | 573/584 |
| Frozen666 | Probe one-to-one | 38 | 82 | 32 | 50 | 525/584 | 573/584 |
| Historical663 | Primary one-to-many | 32 | 107 | 29 | 78 | 518/584 | 573/584 |
| Historical663 | Probe one-to-one | 38 | 85 | 31 | 54 | 524/584 | 572/584 |

Current R36 recoveries: `new_000009:52`, `new_000010:34`, `new_000011:56`, `new_000011:57`, and `new_000088:51`. New misses: `new_000008:1`, `new_000008:7`, `new_000010:33`, `new_000011:58`, `new_000055:23`, and `new_000090:49`. All 583 two-corner fronts remain matched; matched corner-error rows change19->15, while the ten-pixel pair count decreases by one. Every changed miss is cropped in the PDF.

The probe emits 710 records, including one invalid zero-height stored box at `new_000010`, original probe p37: `[323.1650695800781,720,438.2831726074219,720]`, confidence0.179103285074234. This is consistent with native image-boundary clipping; raw pre-clipping coordinates were not captured. The strict production scorer rejects the raw probe set. The diagnostic preserves all 710 records and original pIDs: the unchanged scorer processes 709 positive-area outputs, and the invalid output is explicitly counted as other-unmatched/false output. It is never silently discarded or allowed to match. The helper makes no production-scorer change.

The head probe reduces other false outputs but increases duplicates and misses on the primary frozen reference and current review. It does not solve the duplicate problem. The existing primary head and configuration are preserved; there is no automatic promotion or latency claim. Probe and primary pIDs are separate cache index spaces.

## Matching and visibility definitions

A miss is an unmatched reference after same-class, confidence-ordered one-to-one matching at IoU 0.50. A duplicate is an **unmatched prediction** overlapping an already matched reference at IoU>=0.50. It is a subset of unmatched predictions, not an extra count and not evidence that annotation rows are duplicate objects. Other-unmatched can include misplaced boxes, excluded regions, or unadjudicated fronts; it is not automatically a hallucination.

A maximum-cardinality pass on the same eligible IoU graph finds zero additional possible matches for every checkpoint/reference/image, including the head probe. The official greedy scorer is unchanged. This rules out recoverable TP-count loss from greedy assignment in these replays, not all physical ambiguity among equal-cardinality assignments.

Visibility0 means that reference corner has no coordinate truth. It is not scored at(0,0), and a predicted hidden point is not proof that the physical corner is visible. Stored image y increases downwards: ymin is top, ymax is bottom. Row IDs refer to the named reference version; pIDs refer to the named untouched prediction cache. Earlier PR113/114/117/119 pIDs must not be transferred.

## R37 reference replay equivalence

The R37 review manifest SHA256 is `35a2c6b3ddc1691eaae4b7e416144b8f68f0ab960cbd3ca8b65edd647b619471`. All 20 development-label files have exactly the same bytes and label hashes as R36. Cached replay verifies every detection, point, pair, assignment, and category value is unchanged for all five checkpoint records and the one-to-one probe. `r37-reference-delta.json` contains zero changed development rows and zero assignment/category changes. R37 amendments affect training labels only. This is reference equivalence, not an R37 training or accuracy result.

The retained old-series and held `new_000015` supervision is not newly certified by this round. No full-dataset certification is claimed.

## Files and replay

- `archive-audit.json`: all archive/member/checkpoint/source identities and exact-source verification.
- `predictions/reference.jsonl`, `predictions/candidate.jsonl`: fresh unchanged primary inference.
- `comparison.json`: all preserved reference/model scores, every missed identity, complete current candidate error ledger, matching diagnostics, and control reproduction.
- `size-visibility-audit.json`: explicit bins, association rules, all reference-size and prediction-size tables, cross-tabs, current row/output inventories, overlap screen, and in-memory deletion thought experiment.
- `predictions/candidate-one2one.jsonl`, `head-probe-model-audit.json`: original probe outputs and actual model/backend evidence.
- `head-probe-comparison.json`: separate head comparison, invalid-output policy, all probe error identities, and changed-miss lists.
- `r37-reference-delta.json`, when present: separately verified reference replay, never an R37 learning result.
- `pdf-inventory.json`, `pdf-qa.json`: complete catalogue identities, source-image hashes, final PDF digest, and render validation.

Replay from the repository root after materializing original validation PNGs and installing the recorded environment:

```sh
python3 scripts/score_pr121_review.py
python3 scripts/render_pr121_failures.py --output /tmp/pr121-r36-misses-and-errors.pdf
```

The score replay consumes caches and does not run inference. To reproduce fresh primary inference, use `scripts/evaluate_pose.py` from the exact published source checkout with each extracted `checkpoint-manifest.json`, device CPU, threads 4, confidence 0.13, and overlays0. The one-to-one helper and its exact invocation are documented in `head-probe-model-audit.json`. R37 generic registry edits must not be substituted when verifying PR121 receipts/source hashes. ZIPs, weights, original images, rendered QA images, and PDFs stay outside this result directory.
