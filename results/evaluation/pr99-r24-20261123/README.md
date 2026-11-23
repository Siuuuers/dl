# PR99: R24 model comparison

23 November 2026, Asia/Hong_Kong.

**Keep PR35 as the selected model. R24 gives a small improvement over R23 on the same accepted R24 review, but it does not establish a replacement under the frozen evaluation. No automatic R25 notebook or unchanged training rerun is warranted.** The R24 annotations remain accepted and unchanged.

This report audits [PR99](https://github.com/Siuuuers/dl/pull/99), runs fresh CPU inference for PR35, the returned R24 control and candidate, and the prior R23 candidate, and scores all four against identical references. The numerical summary is [summary.json](summary.json).

## 1. Fixed-reference result

All models use input 1280, FP32, rectangular preprocessing, confidence 0.13, NMS 0.70, matching IoU 0.50, and endpoint confidence 0.50. The 584 denominator includes every reference front with two locatable corners, including eligible misses.

| Frozen 666-front reference | Matched targets | Unmatched targets | Unmatched outputs | Both corners ≤5 px | Both corners ≤10 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected PR35 | 648 | 18 | 80 | 533/584 | 582/584 |
| R24 matched control | 646 | 20 | 83 | 531/584 | 583/584 |
| R24 candidate | 631 | 35 | 105 | 518/584 | 573/584 |
| R23 candidate, freshly inferred | 630 | 36 | 110 | 521/584 | 575/584 |

The separate historical 663-front reference yields PR35/control/R24/R23 unmatched-target counts **15/17/31/32**, and unmatched-output counts **80/83/104/109**. All four models produce zero outputs on the three empty scenes. No model replacement is supported by this fixed comparison.

The original 20-image, 664-front validation used inside training remains unchanged. It is separate from both post-training scoring references and the review below.

## 2. Same accepted R24 reference, same runtime

The separately pinned R24 review has 662 fronts. It was saved before PR99 returned. Both R23 and R24 are evaluated against those same label bytes, including the four development-only point corrections in photo010.

| R24 exploratory reference | Matched targets | Unmatched targets | Unmatched outputs | Both corners ≤5 px | Both corners ≤10 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| PR35 | 625 | 37 | 103 | 544/584 | 573/584 |
| R24 matched control | 624 | 38 | 105 | 540/584 | 573/584 |
| R23 candidate | 645 | 17 | 95 | 557/584 | 583/584 |
| R24 candidate | 646 | 16 | 90 | 557/584 | 583/584 |

R24 therefore adds **one scored match and removes five unmatched outputs** relative to R23 on the same review. Miss-free nonempty scenes increase from 9/17 to 10/17. The 5 px and 10 px totals tie; their individual successes are not all identical. On 1,215 identical locatable returned endpoints, 594 errors decrease and 621 increase, with mean R24-minus-R23 error **+0.014118 px**. This is a modest mixed result.

Reference choice matters. With predictions held unchanged, **16 of R24's 35 frozen-primary misses become matches when the reference boxes change to R24 geometry**; six other old misses refer to deleted rows. The exact lineage and IoU trace is [reference-sensitivity.json](reference-sensitivity.json). An unmatched target is a scoring outcome, not proof that the network produced no nearby cassette proposal. The favorable exploratory ranking does not erase the frozen result.

## 3. What the latest point repairs changed

The actual returned staging inventories, source label bytes and training receipts establish:

- The two matched controls use identical inputs; their **879/879 loaded parameter/buffer tensors are identical**.
- R23-to-R24 candidate supervision differs only in **22 endpoints across 18 rows in three training files**: 012 has 6 rows/7 points, 014 has 7/7, and 016 has 5/8.
- Boxes, classes, visibility tokens, instance counts, every training image, and trainer validation are unchanged. The four corrected photo010 points remain development-only.
- Both rounds record the same L4 runtime, recipe hash, seed46, independent epoch173 initialization, 30 epochs, 1,050 batches, 78 optimizer updates, and final-epoch selection.
- Shared training-source changes only register the new round's verifier and include its source hash. They do not alter optimization logic.

[Crossround input verification](crossround-input-verification.json) preserves the exact delta and tensor comparison. The R23-to-R24 comparison tests this latest training-label delta in one matched seed/runtime setting. In contrast, PR99's candidate versus its PR35-label control tests **all accumulated eligible review changes**. Neither comparison establishes generalization to a new camera session.

## 4. Original-pixel case review

Original crops were inspected before annotated comparisons; accepted labels were preserved. [Case records](crossround/cases.json) retain image hashes, crops, coordinates and exact selected proposals. Rounded or blurred boundaries explain why small score differences should not trigger automatic relabeling.

- **007:14, left corner:** error improves from 2.240 to 0.190 px. The new point is closer to the accepted lower-front junction; both versions already meet 5 px. [Comparison](crossround/new_000007-row14-comparison.png).
- **059:5, left corner:** error increases from 0.075 to 3.048 px as the prediction moves outward toward the side/rack boundary. It still meets 5 px. [Comparison](crossround/new_000059-row05-comparison.png).
- **053:11, left corner:** error increases from 1.397 to 4.060 px, again moving outward from the accepted front-face point. [Comparison](crossround/new_000053-row11-comparison.png).
- **089:49, the only new R24-review match:** R23 already has a proposal at confidence 0.333 and IoU 0.448. R24 changes its extent, reaching IoU 0.719 at confidence 0.144. This is an overlap-matching gain on a partly occluded front, not a previously absent proposal appearing. [Comparison](crossround/new_000089-row49-comparison.png).
- **087:15:** the frozen-primary match is lost, while both models still match the same cassette under accepted R24 geometry. [R24-reference view](crossround/new_000087-row15-comparison.png).

The equal 557/584 five-pixel totals contain two gains and two losses, with decisive endpoint errors near the cutoff. The gains are 008:28 and 059:11; losses are 059:1 and 059:2, using R24 row IDs. They do not demonstrate a broad localization improvement.

## 5. Completed partial-front diagnosis and next useful work

The [complete sixteen-case diagnosis](partial-front-failure-queue.md) and [machine-readable queue](partial-front-failure-queue.json) now account for every remaining R24-review miss in seven photographs. Seven targets have no locatable corner and nine have one; all 584 two-locatable-corner targets are matched. Each case links its exact accepted row, original image and full R24 PDF page.

The retained predictions show **ten cases with an overlapping box below IoU 0.50, six with no retained overlap, and zero losses caused by a qualifying retained proposal being assigned elsewhere**. Some low overlaps belong to neighboring fronts, so overlap alone does not establish a target-associated proposal. These saved outputs are already filtered at confidence 0.13 and NMS 0.70; they cannot identify whether an absent raw proposal was confidence-gated, suppressed, or never decoded with qualifying geometry.

The clearest next supervision question is visible box extent. In **011:55–57**, the retained boxes extend beyond the accepted exposed fragments into the occluded region, while their available left endpoints are approximately **1.96, 3.36 and 1.43 px** from the accepted points. These are diagnostic distances on unmatched alternatives, not official corner successes. Another corner-only training round would not directly address those association failures. The queue also preserves three coverage checks already matched by PR35 and the control but missed by both recent candidates: **008:33, 008:34 and 009:55**.

The next bounded data action is a consistency review of comparable eligible training originals: the visible width/height of fronts exposed beside or through the jaw, and the completeness of clipped fronts. Keep the development photographs in their present role. Any further raw-proposal investigation should be tied to a specific unresolved case and mechanism before a new experiment is prepared.

Do not repeat the completed general NMS/threshold, partial-front-extension or synthetic-occlusion experiments without a new specific mechanism. Previous NMS reductions removed valid partial-front matches, and the existing 0.13 proposal setting already embodies the completed confidence review. There is no evidence here for a new architecture or a blanket brightness filter. Keep one selected checkpoint and the existing reviewed-image workflow; use the full R24 PDF for annotation review. A separately captured and independently labeled session remains the next useful generalization check. **No automatic R25 notebook, training rerun or model promotion follows this diagnosis.**

## 6. Provenance and replay

The exact PR99 archive is **39,783,117 bytes**, SHA-256 `4b2505711bce656f4b693a5ccff93bd23cbda370a5d5ec6933c86e48c0e4e719`. Source is `6307dae6a8db545a72446abb448f19a5dcd5d2b2`; notebook commit is `5feb60b8156c5aff6f9c5ebc3814cbe378ff5628`. [Return audit](return-audit.json) verifies every indexed payload, both completed arms, source/protocol/notebook pins, all 157 original images, initializer, staged labels, held/excluded supervision and unchanged trainer validation. No training was repeated.

Fresh inference covers four checkpoints ×20 images =80 image predictions, with one excluded zero-image warmup per model. Runtime is Python3.12.14, Torch2.14.1+cpu, torchvision0.29.1+cpu, Ultralytics8.4.182, NumPy2.4.6, Pillow11.3.0 and OpenCV5.0.0.93, using four CPU threads. Previous cache headers match, but small floating-point differences appear in fresh PR35/control arrays; this report uses fresh outputs throughout.

To replay the numerical reports without weights, inference or training, use a checkout containing the pinned scripts/references and hash-matching original validation PNGs:

```bash
python3 results/evaluation/pr99-r24-20261123/replay_evidence.py \
  --root /path/to/dl \
  --output /tmp/pr99-replay-verification.json
```

[Replay verification](replay-verification.json) records exact report-byte agreement for both scored comparisons and the sensitivity trace. Checkpoint receipts and training logs are included; archived `train-log.txt` is restored temporarily by the strict receipt adapter. Weight files, returned archives, full camera images and the Python environment remain external inputs and are not duplicated here.

For fresh inference, restore the verified PR35/PR97/PR99 archives from their existing result branches, use the pinned CPU packages in [runtime-requirements.txt](runtime-requirements.txt), and run [run_pr99_inference.py](run_pr99_inference.py) with the PR35 candidate and PR99 reference/candidate manifests. The R23 candidate uses the same project `evaluate_pose.py` command and operating point. [verify_crossround_inputs.py](verify_crossround_inputs.py) repeats staged-byte/tensor verification with the two restored bundles; [analyze_crossround.py](analyze_crossround.py) regenerates the same-reference diagnosis and case sheets.

The included changes are result evidence and reproducible adapters. No accepted annotation, trained weight, runtime default, training recipe or promotion rule is changed.
