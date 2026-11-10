# Annotation corrections applied; keep epoch173 as the working model

**2026-11-10: the first bounded annotation correction is complete.** Under the user's authorization to use assistant-marked labels, the revised validation reference now contains **661 instances and 1,228 locatable endpoints** across the same 20 original images. Four source rows are removed, one previously omitted cassette is added, and 13 retained rows are modified. The accepted edits affect 11 label files. All existing visible-corner coordinates are preserved, including the supported right points transferred during duplicate consolidation.

The user has also prioritized **missed objects** and proposed **10 px**. Record 10 px as a provisional Euclidean tolerance per actual corner in the original 1280×720 image. Continue reporting 5 px as a stricter diagnostic and report misses/extras separately. This preference does not retroactively change the completed seed study's acceptance rule.

The [complete revised labels and editor instructions](../data/references/visible-front-2026-11-10/README.md), [exact amendment manifest](../data/references/visible-front-2026-11-10/manifest.json), [all-model summary](../results/evaluation/baomai-reference-amendment-20261110/summary.json), and [full replay capsule](../results/evaluation/baomai-reference-amendment-20261110/replay.zip) are the deliverables. This follows the [33-row original-image review](persistent-failure-review-2026-11-10.md) in PR #29 and the completed PR #27/#28 repeatability results.

## What was corrected

All row numbers below refer to the **original frozen label file**. The manifest maps retained rows into the revised files and separately identifies the added instance. Markings were selected from original pixels before evaluating the revised reference; predicted rectangles were not copied.

### Instance identity and an omitted cassette

| Original image / rows | Accepted decision |
| --- | --- |
| `new_000058`, row 15 | Remove the seam-straddling box. Rows 13 and 14 already represent the two visible fronts it crosses; the image does not support a distinct additional front there. |
| `new_000058`, row 19 | Retain the third central front and correct its visible box to `[635, 410, 790, 445]`. Preserve its existing right corner and missing left slot. |
| `new_000059`, row 16 | Remove the analogous seam-straddling box across already labeled fronts 14/15. |
| `new_000059`, added instance | Add the third central front below row 15. Its visible upper region connects to the exposed right strip beside the gripper. Box: `[637, 426, 794, 463]`; right lower corner: `(790.5, 460.5)`; left corner: `0 0 0`. No original row represents this object. |
| `new_000086`, rows 47/58 | Consolidate into retained row 47, using observed box `[653, 429, 736, 446]` and the exact supported right-corner tokens from row 58. Remove the redundant row 58. |
| `new_000086`, rows 48/59 | Consolidate into retained row 48, using observed box `[671, 448, 736, 465]` and the exact supported right-corner tokens from row 59. Remove redundant row 59. The left area around x653 is gripper material; the actual left corner remains hidden. |

Fresh box boundaries and the newly marked right corner have roughly 1–2 px of visual softness. These are reviewable pixel selections, not statistical uncertainty intervals or subpixel physical measurements. The added `new_000059` object has separate provenance from deleted row 16; it is not presented as a guess about that old row's intent.

### Ten bounded box-edge amendments

| Image | Source row | Edge | Original → revised, px |
| --- | ---: | --- | ---: |
| `new_000054` | 17 | Top | 362.94 → 392 |
| `new_000056` | 1 | Top | 258.99 → 285 |
| `new_000086` | 4 | Top | 132.12 → 122 |
| `new_000087` | 4 | Top | 111.24 → 103 |
| `new_000088` | 4 | Top | 128.53 → 120 |
| `new_000089` | 4 | Top | 151.76 → 143 |
| `new_000090` | 4 | Top | 185.72 → 173 |
| `new_000008` | 33 | Bottom | 480.06 → 456 |
| `new_000009` | 57 | Left | 748 → 810 |
| `new_000090` | 43 | Bottom | 717.12 → 700 |

The first two remove definite rack overhang above the front. The next five restore visible front material omitted by thin lower-band boxes, while excluding the projecting upper lip/roof. The final three remove neighboring-front, gripper or rack/background extent. Orthogonal box-coordinate tokens and all six corner tokens remain byte-for-byte unchanged for these ten rows. Their unmodified edges are outside the accepted correction scope and are not newly certified tight.

### Nineteen queue rows remain explicitly unresolved

The 33-row queue now has **14 amended dispositions and 19 retained unresolved dispositions**. A completed review is not a claim that every physical boundary can be determined from the available pixels.

| Images / original rows | Remaining issue |
| --- | --- |
| `new_000007` 7/14; `new_000011` 15; `new_000057` 1; `new_000059` 1 | Extremely shallow top strips: independent front identity versus adjacent front/lip remains unclear. |
| `new_000008` 1/36; `new_000009` 8; `new_000053` 1 | Frame-clipped fronts/fragments with uncertain extent or front/upper-material separation. Preserve independently supported endpoint visibility. |
| `new_000009` 53/56; `new_000055` 26; `new_000086` 57; `new_000088` 57 | Foreground occlusion or uncertain fragment identity/extent. Row 56's pale patch may be linkage or a target fragment; deletion would assert an unsupported identity. |
| `new_000056` 4; row 15 in `new_000086`, `new_000088`, `new_000089`, `new_000090` | Blurred or shaded front-to-roof/lip transitions do not support one confident exact upper edge. |

For `new_000055` row 26, the source clearly omits part of a visible front, but the exact replacement top/right bounds remain unresolved. Ambiguous rows are retained as disclosed legacy annotations in this bounded reference. They are not certified positives, newly approved negatives, or fully corrected examples. A future complete policy migration must resolve or explicitly quarantine affected images rather than silently ignore this issue.

## What the same predictions score now

The replay verifies original input identity first, then uses the same stored boxes and points against each reference. It evaluates five unique models at confidence 0.15, 0.25 and 0.40, including all three empty images. Matching IoU remains 0.50, NMS 0.70, and endpoint confidence 0.50.

### Epoch173 at confidence 0.25

| Measure | Frozen reference | Revised reference |
| --- | ---: | ---: |
| Reference instances | 664 | 661 |
| Matched instances | 625 | 634 |
| Missed instances | 39 | **27** |
| Extra detections | 63 | **54** |
| Locatable reference endpoints | 1,227 | 1,228 |
| Missing locatable endpoints | 43 | **26** |
| Both actual corners within 5 px | 521 / 584 (89.21%) | **528 / 584 (90.41%)** |
| Both actual corners within 10 px | 568 / 584 (97.26%) | **575 / 584 (98.46%)** |
| Conditional median corner error | 1.611790 px | 1.617458 px |
| Conditional p95 corner error | 5.161683 px | 5.210696 px |

These are **reference effects, not model improvements**. The model returns the same 688 detections at this threshold. Seven corrected top edges recover seven existing same-front matches; the corrected third-front box in `new_000058` recovers another; deleting two invalid rows and consolidating two duplicate pairs removes four reference misses. The newly annotated cassette in `new_000059` is already detected, converting an extra prediction into a match. Together these produce 12 fewer misses and nine fewer extras with a net three fewer reference instances.

The median/p95 increase slightly because the set of matched, scoreable endpoints changes. Existing predicted coordinates and existing visible reference coordinates did not move. Do not compare the two columns as if a retrained model had improved recall and worsened its corners.

An independent calculation confirms that all 1,184 previously scored M0 endpoint errors remain bit-identical and 18 additional endpoints enter the calculation. Two of those additions exceed 5 px: the right corner of `new_000058` source row 19 at 6.203322 px and the newly annotated right corner in `new_000059` at 9.211098 px. Both pass a 10 px ceiling, illustrating why that ceiling is less demanding than precise corner placement.

### All five models on the revised reference

At confidence 0.25, all rows below share the same 661 instances, 1,228 locatable endpoints and 584 two-locatable-corner objects.

| Model | Misses | Extras | Missing locatable endpoints | Conditional median / p95, px | Both corners ≤5 px | Both corners ≤10 px |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M0: epoch173 | 27 | 54 | 26 | 1.617 / 5.211 | 528/584 | 575/584 |
| Seed 43 original-label control | 24 | 53 | 23 | 1.628 / 5.231 | 522/584 | 577/584 |
| Seed 43 corrected-box candidate | 24 | 51 | 23 | 1.640 / 5.212 | 526/584 | 578/584 |
| Seed 44 original-label control | 26 | 60 | 24 | 1.668 / 5.164 | 526/584 | 577/584 |
| Seed 44 corrected-box candidate | 27 | 62 | 25 | 1.647 / 5.151 | 526/584 | 577/584 |

On this revised reference, the seed-43 candidate ties its same-seed control for misses; the seed-44 candidate still misses one more. The corrected-box training intervention therefore does not show repeatable miss reduction. Selecting the most favorable individual seed would not establish that the annotation intervention caused its result. Keep epoch173 as the working default and preserve the frozen-reference study as its own historical comparison.

## Is 10 px acceptable?

**Use it as a provisional ceiling for reviewed image-space localization.** For each corresponding returned corner, evaluate `sqrt(dx² + dy²) <= 10` in the original 1280×720 coordinates. Requiring ±10 independently on both axes would allow a 14.14 px diagonal error and is a different rule.

The 584 two-locatable-corner references retain exactly the same point pairs in both versions. Their corner-to-corner span ranges from **50.36 to 218.27 px**, with median **90.36 px**. A 10 px radius is therefore about **11.07% of the median span**, ranging from 4.58% to 19.86% across objects. It exceeds 10% of span for 425/584 objects (72.77%). This is fairly loose relative to small cassettes, so precise original-pixel annotation and the 5 px diagnostic remain useful. A numerical allowance never licenses placing a label arbitrarily within ten pixels.

The revised M0 10 px result is 575/584, but it excludes **77 objects with one or zero locatable corners**. Its 27 detection misses comprise eight zero-corner, twelve one-corner and seven two-corner objects. Thus **20 of the 27 misses are outside the two-corner metric's population**. High 10 px success cannot establish the user's missed-object goal by itself. Seven eligible two-corner objects lack a returned pair; two more have a returned pair exceeding 10 px.

There is no physical unit conversion or independent-scene reliability claim here. Image-space tolerance is sufficient to continue this still-image task; downstream requirements can be specified if that scope later changes.

## Miss-priority operating choice

| M0 box confidence | Misses / 661 | Extras | Object recall | Both corners ≤10 px |
| --- | ---: | ---: | ---: | ---: |
| 0.15 | **22** | 82 | 96.67% | 575/584 |
| 0.25 | 27 | 54 | 95.92% | 575/584 |
| 0.40 | 30 | **35** | 95.46% | 575/584 |

For a review queue, **0.15 is the first stored setting to try when catching more objects matters most**: it recovers five benchmark matches at the cost of 28 additional extra detections compared with 0.25. Its 5 px and 10 px both-corner counts are unchanged, so those recoveries do not translate into more successful complete corner pairs at these radii. Confidence 0.25 remains the recorded primary/default setting, allowing comparisons to retain their meaning. The table makes the review-workload tradeoff explicit without inventing an acceptable extra-detection rate.

Endpoint confidence also needs care: M0 returns a finite in-frame point at confidence ≥0.50 for **61 of 66 reference-unlocatable endpoint slots on matched objects** at the primary threshold. These are diagnostic outputs where a true target coordinate is unavailable; they are not measurable geometric errors or proof of a correct hidden corner. The labels retain missing slots instead of inventing geometry.

At confidence 0.25, 16 of the 17 nonempty images still have at least one reference-matching miss or extra; `new_000058` is now the one clean nonempty image. All three empty images remain free of extras for all five models at this threshold. Remaining reference ambiguities limit how directly these counts can be interpreted as physical model failures, and the small reused image set does not establish future-scene reliability.

## Verification and the next useful work

The [verification record](../results/evaluation/baomai-reference-amendment-20261110/verification.json) binds the revised manifest, labels, code, full replay and capsule. Original PNGs and labels were hash-checked. Original replay identity is verified before amendment, rather than rewriting prediction headers to accept revised label hashes. All 15 original aggregate score and quality results reproduce the published evidence; the same scorer then produces all 15 revised aggregates and 300 per-image comparisons.

Mechanical checks cover the exact 20-file set, 11-field schema, finite normalized values, canonical missing states, image-X order, positive boxes, image bounds and visible-point containment. A separate assistant reviewed row mappings, before/after text, preserved corner tokens and instance decisions. These checks and assistant reviews establish a reproducible bounded correction, not human certification of every label.

The existing HTML editor can open a working copy of these labels; the [reference README](../data/references/visible-front-2026-11-10/README.md) supplies the exact preparation command. Native browser input failed in this session, so correction used original-image coordinate review and direct label writes; no GUI save round-trip is claimed.

The next model-improvement task is to review the **analogous training annotations** in the existing 137 training images: omitted visible fronts, duplicate identities with inconsistent visibility, rack/occluder overhangs, and thin-band boxes that omit visible front material. Form one separately identified, image-supported training-label intervention before comparing a matched control and candidate. The present validation amendment improves measurement; model learning requires a subsequent training intervention. Keep uncertain cases explicit, preserve the split, and use the completed GPU study rather than rerunning it unchanged.
