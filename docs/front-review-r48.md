# R48: original-image amendments and PR137 interpretation

27 November 2026, Asia/Hong_Kong.

The [PR137 return](https://github.com/Siuuuers/dl/pull/137) contains completed R47 training. The [result audit](additional-front-pilot-r47-results.md) reproduces its saved scores and explains the mixed outcome. This separate annotation review addresses the latest original-image feedback and continues from [R47](front-review-r47.md). It does not change a returned prediction or overwrite a frozen reference.

Two PDF documents are delivered with this review: the amendment atlas and the [complete missed/error/unmatched atlas](front-error-review-r48.md). Repository files contain the reproducible programs, text evidence and versioned labels. Original images, PDFs, archives and weights are not added to this source change. The existing HTML remains its earlier review snapshot.

## 1. Scope and interpretation

- **Supplementary photographs:** inspect all 12 originals and all 507 current rows. Propose 44 side-boundary amendments; preserve every corner and availability token exactly. Resolve one previously held boundary. Preserve the historical deleted row as a deletion.
- **Development photographs:** map 15 user hints to ten existing R38 targets in seven originals. Propose amendments on nine rows and retain two explicit field holds. Export all 20 development TXT files as a separate proposal, with unchanged 654-instance membership and corner-availability population. The other rows are inherited, not newly certified.
- **Reviewer status:** these are assistant visual proposals. The review contains no independent human sign-off and makes no subpixel-accuracy claim. The nominal uncertainty of the new visual coordinates is approximately 2 original-image pixels; that is an explanatory estimate, not a measured confidence interval.

Original images are 1280 by 720. X increases right and Y increases down. Stored `ymin` is the **top**, and stored `ymax` is the **bottom**. The user's physical descriptions take precedence over the reversed variable names in the feedback: "ymin (bottom)" is reviewed as the physical bottom and written, when justified, to stored `ymax`.

The user's P numbers come from the earlier R47 error PDF: **PR133 candidate** postclip outputs. The [identity replay](../results/evaluation/front-errors-r48-20261127/prior-reference-identity.json) proves that these saved outputs equal **PR137 reference** records across all 20 photographs. They do not identify the new PR137 candidate outputs. R numbers identify one-based lines of the R38 development labels. Supplementary `source` identities remain independent of output line numbering.

## 2. Development amendment decisions

All numbers below are original-image pixels, rounded for readability. Exact before/after TXT and frozen cue objects are in the [development ledger](../data/reviews/fronts-2026-11-27-r48/audit-ledger.json).

| User's old cue | Existing R38 target | Proposed decision | Original-image reason |
| --- | --- | --- | --- |
| `new_000010 / P029` | R029 | **Hold bottom; retain 720** | Old P029 top is about 711.82, below which the retained visible left corner is at y714. The material on the right continues to the frame edge. A full-front contraction to 712 or 715 is not established by the single image. |
| `new_000090 / R047` | R047 | Left point `(678.5,174.86)` to `(681,174)`; right `(785.48,179.81)` to `(790,181)` | The outward lower turns in the original support the direction suggested by old P032. Keep the box and both availability flags. |
| `new_000008 / P035, P036`; P038 as context | R017 | Top `457.69` to `462` | The visible left seam starts lower. The prediction bands below about y508 extend through gripper and rack; they do not establish another cassette. Keep both existing lower corners. |
| `new_000053 / R014` | R014 | Bottom `319.48` to `321` | The front's lower-left transition reaches about y321, near old P002's left-point height. Keep the recorded corner coordinates. |
| `new_000087 / P058, P062` | R043 | Top `426.18` to `433` | The owned narrow front begins near y433. The dotted assigned P053 top at 428.51 still includes the preceding surface. Both cues refer to the same existing front. |
| `new_000089 / P052, P055` | R042 | Right `620` to `614` | The upper rim reaches roughly x612–615 before the jaw. The predictions' right bounds around x656–658 include substantial gripper. The hidden right corner stays missing. |
| `new_000089 / P059` | R041 | Top `432.69` to `436`; **hold right at 664.14** | The frontal seam supports the top change. A thin strip above the jaw at roughly x630–662/y440–443 has uncertain ownership. Cutting the full right bound to the jaw's left edge could remove valid target material. |
| `new_000089 / P064, P058` | R049 | Top `464.63` to `473` | The lower partial front starts after the neighboring band. The dotted assigned P057 top at 472.89 supports this nominal seam. |
| `new_000059 / P020, P019` | R018 | Top `426` to `418` | Include the owned molded frontal cap, separated from the preceding cassette and continuous with this front. Copying P020's top near 434 would cut the cap. |
| `new_000090 / P050` | R040 | Top `465.21` to `469` | The owned upper rim follows the separator near 469. The dotted assigned P047 top at 468.22 is a nearby cue. |

Nine rows change: eight box fields and two visible corner positions. No instance is added or deleted; no corner gains or loses availability. Two corners changed in one row do not mean two changed rows. The R029 bottom hold and R041 right-boundary hold remain explicit even though the latter row has a supported top correction.

The particularly important ownership finding is `new_000008 / R017`: repeated predictions do not establish repeated physical objects. Keep one row for the already identified cassette. The model's excess bands are documented as evidence; they are not copied into the annotation set.

## 3. Supplementary side-boundary pass

The [507-row ledger](../data/reviews/additional-fronts-2026-11-27-r48/audit-ledger.json) anchors every decision to the exact R47 before-line and original-image hash. The audit uses all full originals, 47 original/R47 row sheets, additional magnified pixel strips, and before/after overlays for every proposed side edit.

| Photograph | Current rows amended | Fields |
| --- | ---: | --- |
| `frame_0001` | 36 | Left/right sides, predominantly right sides excluding continuing rack |
| `frame_0074` | 4 | Right sides of source008, 024, 038, 041 |
| `frame_0301` | 4 | Left sides of source031, 032, 034, 037 |
| Other nine reviewed photographs | 0 | Retained where an additional displacement was not supported beyond local edge softness |

The named `frame_0001` corrections are:

| Source row | Field | Previous nominal value | R48 nominal value |
| --- | --- | ---: | ---: |
| 006 | Left / `xmin` | 577.01 | 579 |
| 007 | Right / `xmax` | 730.40 | 728.5 |
| 008 | Right / `xmax` | 807 | 805 |
| 009 | Right / `xmax` | 973.66 | 971 |
| 010 | Right / `xmax` | 807.63 | 805 |
| 011 | Right / `xmax` | 806.35 | 804 |
| 043 | Left / `xmin` | 776.24 | 779 |

For source043, the continuing rack occupies approximately x774–778 while the visible owned gray front begins around x779–781 above the gripper. Independent assistant review of the coordinate grid and original crop supports nominal xmin779 with about 2 px uncertainty. Resolve only this boundary hold. Keep the hidden left corner exactly `0 0 0`; the diagnostic curve stops at the gripper.

This makes six annotation-complete supplementary photographs: `frame_0001`, `frame_0146`, `frame_0160`, `frame_0178`, `frame_0371`, and `frame_0383`. The other six photographs still contain **12 held rows / 31 unresolved fields** and are excluded as whole photographs from complete-candidate export. The newly complete `frame_0001` supplies 43 fronts: 41 with two locatable corners and two with one. Completing that photograph does not mean it has already been trained.

All 1,014 supplementary corner slots, including 66 unavailable slots, retain their exact R47 coordinate/visibility tokens. No top or bottom boundaries change in this supplementary pass. The eleven supplied open side traces are diagnostic explanations, not segmentation labels or inferred hidden contours.

## 4. Shadows, caps and small fronts

Keep the same physical target at every size:

- A shadow cast onto the target's own front remains part of that surface.
- An owned molded frontal rim or cap remains included even when its illumination is gray.
- Rack, gripper, a neighboring cassette or its shadow must not determine the target's outer box bounds.
- An axis-aligned rectangle may contain some background or an occluder between supported visible extrema. That does not justify extending the extrema to foreign material.

The input already contains neighboring context. If a future crop-based method needs more context, increase its input crop and keep the label geometry consistent. Do not grow a small target's annotation to make matching easier. The [annotation policy](annotation-policy.md) now records this distinction explicitly.

## 5. How to use this release

The development proposal and supplementary training candidates are separate. The development directory contains only the 20 validation files; it is not a replacement for a complete 157-photo training dataset. The supplementary directory contains 12 proposals and only the six eligible complete-photo exports. No training notebook is retargeted to these files automatically.

For the next experiment, the strongest measured lead is partial-front supervision: PR137 added 164 two-corner examples, eight one-corner examples and no zero-corner examples, while every remaining R38 miss is in a one- or zero-corner target. R48's newly complete photograph is also mostly two-corner. Therefore its release alone is not evidence that another expanded-data run will solve partial-front misses. Use [the thesis and experiment plan](thesis-plan-r48.md) to select one supported intervention and a genuinely independent capture evaluation.

The two development holds need better material-ownership evidence, such as an adjacent unoccluded view or a human adjudication of the same physical rim. Retaining the prior coordinate preserves uncertainty; it is not a declaration that the old boundary is correct.

## 6. Reproduce the amendment package

Install the existing Python image/PDF dependencies. Obtain the pinned PR137 result archive, rebuild the exact R47 review using its documented R44 inputs, and provide the seven pinned original development PNGs. The original hashes are in the ledgers and frozen manifest; the result archive hash is checked before reading predictions.

```bash
python scripts/build_front_review_r48.py \
  --r47 /absolute/path/to/r47-review \
  --archive /absolute/path/to/baomai-additional-front-pilot-r47-results.zip \
  --originals /absolute/path/to/development-originals \
  --output /absolute/path/to/r48-review \
  --pdf /absolute/path/to/r48-annotation-amendments.pdf

python scripts/build_front_review_r48.py \
  --check-output /absolute/path/to/r48-review
```

Run from the repository's scripts/dependencies; the copied renderer/builder in the review package are source evidence, not a claim of a standalone Python dependency closure. The builder verifies committed exports, exact parent hashes, deterministic coordinate operations, every original image, cue identity, whole-photo eligibility and strict file membership. It refuses stale unrelated output files rather than incorporating them into a new inventory. Store the PDF outside the generated review directory.

The [complete error atlas](front-error-review-r48.md) deliberately retains R38 so the published model result stays interpretable. The completed [R48 label-sensitivity replay](front-reference-sensitivity-r48.md) compares both unchanged prediction caches under both references. All target assignments, misses and total unmatched counts stay unchanged. Formal duplicates change 18/14 to 16/13 for the reference/candidate arms; these are category changes within the same unmatched outputs. A score change caused by new labels is not a model improvement.

The amendment PDF contains 53 pages: summary/rules/request map on 1–3, ten development-target pages on 4–13, twelve full-frame supplementary views and complete side indexes on 14–25, and 56 detail cards on 26–53. Every one of the fifteen development requests and 507 current supplementary identities is indexed. The [verification record](../results/review/front-review-r48-20261127/verification.json) binds the PDF hash, page index, visual coverage, label replay and focused guard checks.
