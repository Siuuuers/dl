# Partial-front stages and one reflected-output union

This completed CPU replay uses immutable saved outputs. It runs no model or training and changes no annotation. The three focused restorations remain missed. The separate union experiment asks whether other missed fronts already have unused reflected-view detections under the existing thresholds.

## Three restored misses

Only the historical adapted original-view checkpoint has a complete decoded pool. Its focused stage trace is:

| Historical target | Stage | Best decoded IoU | Best-box confidence | Eligible decoded boxes |
|---|---|---:|---:|---:|
| r34/new_000009/R056 | confidence_gate | 0.608489 | 0.00215215 | 1 |
| r37/new_000009/R052 | no_iou_eligible_decoded_box | 0.409067 | 0.33002254 | 0 |
| r37/new_000053/R001 | confidence_gate | 0.690867 | 0.00009662 | 2 |

Two targets have eligible boxes below the confidence cutoff. The lower009 target has no IoU-eligible box anywhere in this fixed original-view tensor. Lowering a confidence cutoff cannot create a missing box geometry. The bound is limited to this checkpoint and view. Newer post-native-NMS caches do not provide equivalent stage attribution.

The QFL reflected view does contain an eligible lower009 box above 0.25. It is already mutually paired to a higher-scoring original box. Its corner-pair-unavailable status does not make it unpaired, and the original would also suppress it in an unrestricted union at 0.70. This is a counterexample to treating every additional view response as recoverable output.

## Fixed union, all fronts

Retain the completed original-view detections and their existing averaged corners. Append every reflected output unused by any existing mutual box link, then apply the same score-ranked postclip suppression at IoU strictly above 0.70. No reference is used to select additions; no coordinates, scores, weights, thresholds or caps are tuned. The preliminary r67 support gate was fixed before measuring this full comparison.

| Method | Variant | r67 misses | Extras | Duplicate subset | Pair ≤5 px | Pair ≤10 px |
|---|---|---:|---:|---:|---:|---:|
| historical_retained | unchanged | 21 | 50 | 5 | 569/583 | 583/583 |
| historical_retained | unpaired_reflection_union | 16 | 62 | 9 | 569/583 | 583/583 |
| pr157_native | unchanged | 21 | 48 | 4 | 568/583 | 583/583 |
| pr157_native | unpaired_reflection_union | 15 | 60 | 9 | 568/583 | 583/583 |
| pr157_qfl | unchanged | 15 | 82 | 17 | 570/583 | 582/583 |
| pr157_qfl | unpaired_reflection_union | 11 | 115 | 33 | 570/583 | 582/583 |
| pr158_original_labels | unchanged | 50 | 84 | 6 | 541/583 | 566/583 |
| pr158_original_labels | unpaired_reflection_union | 48 | 93 | 8 | 541/583 | 567/583 |
| pr158_reviewed_labels | unchanged | 18 | 50 | 13 | 572/583 | 583/583 |
| pr158_reviewed_labels | unpaired_reflection_union | 10 | 60 | 18 | 572/583 | 583/583 |

The complete 40-row matrix also includes frozen r50, supplied664 and frozen666 sensitivity references. The per-target ledger identifies gains, losses and rematches, with confidence-qualified errors for every annotated known corner. The output ledger distinguishes added reflected detections from original outputs removed by the new pooled suppression pass. A better miss count alone does not establish a preferable operating method.

## Training-label support

The unchanged PR157 training labels contain 3999 fronts, including 465 with exactly one known corner. At nominal 1280×720 image coordinates, the one-known-corner counts within the focused width/height bounds are:

| Width and height bound | Fronts | Images |
|---|---:|---:|
| ≤18 × ≤31 px | 10 | 8 |
| ≤15 × ≤14 px | 0 | 0 |
| ≤47 × ≤10 px | 0 | 0 |

These descriptive counts identify a possible data-support issue for a later controlled experiment. They do not show that a particular loss or augmentation will improve these targets. The restored development labels are not training-label additions.

## Limits

- All comparisons use repeatedly inspected development photographs. No model or threshold is promoted.
- Complete decoded pre-confidence/pre-NMS tensors exist only for the historical adapted original view. Newer caches are already native-NMS filtered; PR157 keeps scores >0.001 with a 300-output cap, while PR158 and r60 native caches keep scores >0.25.
- A missing current cached candidate cannot be attributed to raw confidence, geometry, native NMS or the cap without missing raw tensors. A cap-hit flag does not prove a relevant candidate was truncated.
- Raw historical tensors and common-runtime final records differ numerically; r63 continuity and focused r67 matching are checked without asserting byte-identical inference.
- The greatest-overlap saved box can describe a neighboring front. It is a comparison of boxes, not proof of the network's physical ownership intent.
- The union adds all unpaired reflected outputs using prediction-only associations. Reflected corners remain unaveraged, scores are unchanged, and added outputs may suppress original outputs or change evaluation matching.
- Native views each had max_det=300; the existing postclip union pass introduces no additional cap. Source-view identities distinguish additions, suppression and rematching.
- Finite-pool threshold support is an upper bound before suppression and matching, not a trained result, usable threshold recommendation or estimate of output precision.
- Training size bins are descriptive, chosen from the three development targets, and count nominal original-image labels before augmentation. They do not establish visual similarity, effective training exposure or an optimal loss.
- The three restored focus targets remain missed by this fixed union. No labels, detector tensors, boxes, scores or existing averaged corner coordinates are modified.
