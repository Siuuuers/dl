# R50: error types, evidence and next actions

R50 reviews the completed [PR137 run](https://github.com/Siuuuers/dl/pull/137) and the original development photographs. [PR140](https://github.com/Siuuuers/dl/pull/140) repairs notebook checkout; it contains no completed R49 training, pre-NMS diagnosis or Soft-NMS result. No model is promoted by this review. PR35 remains selected.

The 67-page PDF preserves every fixed-R38 error case: **15 missed targets, 20 matched fronts failing the visible-corner 5px check, and 84 unmatched predictions**. Each card contains original pixels, recorded reference/model overlays, an observed type, evidence, a conclusion, an existing-data next action, observation confidence and a separate current-R50 status. There are 119 cards because one physical cassette can appear in both a miss card and several unmatched-output cards.

The historical frozen666 reference remains the primary gate. R38 is the frozen reference for this detailed review. The current R50 proposal is replayed separately; it never overwrites either frozen reference.

## Fixed predictions, separate reference versions

The reference and candidate below are the two completed PR137 model arms. Both use the same confidence and matching settings. R50 inherits the nine R48 development-row amendments and changes only `new_000010/R029` bottom from y=720 to y=714, following the user's ownership clarification. Corner tokens remain unchanged. This annotation change is not a training result.

| Reference labels | Model arm | Matched | Missed | Unmatched | Formal duplicate subset | Both corners within 5px |
|---|---|---:|---:|---:|---:|---:|
| Frozen R38 | Reference | 641 | 13 | 90 | 18 | 569 / 583 |
| Frozen R38 | Candidate | 639 | 15 | 84 | 14 | 566 / 583 |
| Proposed R50 | Reference | 641 | 13 | 90 | 16 | 570 / 583 |
| Proposed R50 | Candidate | 639 | 15 | 84 | 13 | 566 / 583 |

All target-to-output assignments and all 119 reviewed case identities remain unchanged. Five unmatched diagnostic bins change between R38 and R50:

| PR137 candidate output | R38 category | R50 category |
|---|---|---|
| new_000087/P059 | Duplicate | Near gate |
| new_000087/P063 | Near gate | Duplicate |
| new_000089/P052 | Near gate | Positive overlap |
| new_000089/P063 | Positive overlap | Near gate |
| new_000090/P050 | Duplicate | Near gate |

Thus the duplicate subset falls by one **overall**; this is not one isolated output being corrected. The full replay is in [r50-reference-replay.json](../results/evaluation/front-errors-r50-20261127/r50-reference-replay.json), and [error-index.json](../results/evaluation/front-errors-r50-20261127/error-index.json) preserves the original and current case geometry.

## What original pixels support

All 119 crops were inspected with the full original frame available. All 15 original images containing cases were hash-verified; the remaining development images do not contribute cases. An independent reviewer also inspected the five full frames and exact/context crops containing all 12 zero-overlap outputs.

| Observed type | Cards | Meaning |
|---|---:|---|
| Extent crosses an occluder | 56 | The box crosses a gripper or hand, or differs from the visible fragment. |
| Corner placement disagreement | 20 | The box is matched, but an accepted endpoint is over 5px from its recorded corner. |
| Repeated front hypothesis | 14 | A separate output already owns the reference assignment. |
| Image-boundary extent | 7 | The clipped target or top-edge strip has incompatible retained extent. |
| Noncassette structure response | 7 | Five keyboard/peripheral outputs and two lower rack/support-band outputs, assessed from pixels. |
| Front extent / neighboring material | 5 | Cap, seam, lower-band or partial-width extent differs. |
| Instance ownership unresolved | 4 | Four top strips remain unproven as complete cassette fronts. |
| Explicitly excluded visible fragment | 3 | Three outputs cover the region of one user-excluded cassette-associated sliver. |
| Occluded front lacks a localized output | 3 | Three zero-corner targets lack a suitably localized retained box. |

These are **card counts, not unique physical cassette counts or causal estimates**. The scorer and observed type are separate fields. A physical repeated-front hypothesis need not remain a formal duplicate after reference geometry changes. The 56 occlusion-extent cards include nine miss cards and 47 unmatched-output cards.

### Zero overlap does not imply background

The 12 zero-R38-overlap outputs split into five supported keyboard/peripheral responses, three outputs associated with the same excluded visible cassette sliver, and four unresolved top-strip cases. The four top strips are `new_000055/P027`, `new_000089/P050`, `new_000089/P060`, and `new_000090/P059`. They remain explicit ownership holds and are not automatically mined as negatives.

The three sliver outputs are `new_000009/P057`, `P065` and `P066`. The recorded R38 review explicitly deletes old R37 row 52, with box `[600, 490, 615, 504]`, by user request. Original pixels support cassette-associated material; this deletion changes the selected reference population and does not establish that the region is background, a duplicate or below a general size cutoff. Preserve that decision. Their IoUs with the old rectangle are 0.3303, 0.4091 and 0.2206, so none is a recovered detection at the fixed 0.50 gate even under that earlier rectangle. See [the R38 row review](../results/review/front-review-20261126-r38/row-review.json) and [the independent pixel/identity review](../results/evaluation/front-errors-r50-20261127/zero-overlap-independent-review.json).

## Priorities using data already available

All 15 candidate misses have fewer than two locatable corners: 12 of 54 one-corner targets and three of 17 zero-corner targets. All 583 two-corner targets remain matched. The 172 supplementary training fronts added by PR137 contain 164 two-corner examples, eight one-corner examples and no zero-corner examples. This motivates diagnosis and review of existing training-side partial-front labels, but does not prove why the candidate changed. If diagnosis supports a new eligible training-label correction, test that label delta while keeping native loss, whole-image membership and the update budget fixed. Hold training if no new supported delta remains.

R29 partial-image sampling and R31 factor-two partial-box loss already ran. R31 was mixed and its seed47 screen failed; repeating sampling or loss weighting is not the default next step. See the audited prior-trials discussion in [project-direction-r50.md](project-direction-r50.md).

First settle visible contour, ownership and the recorded annotation contract. Then instrument inference to distinguish absent proposals, confidence removal, ordinary NMS, clipping and postclip suppression. The current cache starts after ordinary NMS, so a retained miss cannot identify an earlier rejection stage. Preserve development photographs for evaluation; use existing training-side occlusion examples and verified background regions for any new training treatment.

The existing postclip filter already removes nine candidate outputs without adding misses: unmatched 93 to 84, duplicates 22 to 14, misses unchanged at 15. Higher confidence has a recall cost: at confidence 0.25, candidate unmatched falls from 84 to 51 while misses rise from 15 to 18. The selected operating confidence remains 0.13. New Soft-NMS, enhancement and resolution benefits have not been measured. A simple deterministic postprocessor can be part of inference; it does not require the model to call a tool.

Image y increases downward: ymin is top and ymax is bottom. Scale does not authorize including another cassette, rack, gripper or a neighbor's shadow. Missed partial-box heights overlap matched heights, so size and dimness should be tested as separate hypotheses rather than treated as established causes.

## Measurement and thesis scope

A manageable proposed question is how consistent visible-front annotations and partial-front supervision affect detection and two-corner localization in crowded tissue-cassette images. Discuss the scope with the professor: the lecture supplies detection concepts, not a settled thesis specification.

Report TP/FN, all unmatched, duplicate subset, visibility strata, original-pixel corner errors and unconditional within-5/10px success together. Record runtime, seed, image exposure and actual optimizer updates. Compare one factor at a time. Existing development evidence can support a careful reproducibility/ablation study now; a later capture/session-separated holdout is needed for a stronger independent generalization claim.

The lecture at `upload/detection.pdf` defines IoU on page 5 and covers precision/recall, confidence ranking, the precision-recall curve and AP on pages 9-11. Fixed operating-point precision and recall are not AP: the R38 candidate has precision 639/723 = 88.38% and recall 639/654 = 97.71% at the recorded thresholds. The truncated retained cache is insufficient for a complete AP benchmark. One-to-one assignment and the formal-duplicate rule are repository-scorer definitions; see [pose_scoring.py](../scripts/pose_scoring.py).

## Reproduce and verify

```bash
python scripts/export_front_error_review_r50.py \
  --archive /path/to/baomai-additional-front-pilot-r47-results.zip \
  --images-dir /path/to/hash-verified-original-pngs \
  --output /path/to/r50-missed-errors-unmatched.pdf
```

The renderer audits the pinned PR137 archive before creating the PDF, validates original image hashes and the R50 label manifest, replays both model arms, verifies 119 unique case IDs and checks paragraph bounds. It never loads a model or writes labels. The human-readable analysis input is [case-observations.json](../results/evaluation/front-errors-r50-20261127/case-observations.json); the complete per-case output is [case-analysis.json](../results/evaluation/front-errors-r50-20261127/case-analysis.json). The observations are an agent visual review, not a claim that a person manually certified every label.

All 67 rendered pages were inspected as contact sheets, with detailed checks of summary, miss, exclusion, amendment, category-transition, hold and keyboard-response pages. Text bounds and all 119 Observed/Conclusion/Next sections passed. [verification.json](../results/evaluation/front-errors-r50-20261127/verification.json) records generation and replay checks; [visual-verification.json](../results/evaluation/front-errors-r50-20261127/visual-verification.json) binds the final PDF hash to the rendered-page inspection.

Pins: PR137 head `dcb43d7ad061fc3a43f1392398bb8794d9d7168d`; PR140 head `55a6f68993a63fcdf422014995fada33352720df`; archive SHA-256 `fc8c40357c83831d17f88b0bb5a5ecd2e0cb1a8128be697e37e80620e4981023`; R50 manifest SHA-256 `f639459bdd986f6553b9e00f0ecfdf1eee3c7f5344e95c300a4a17a66352c04f`.
