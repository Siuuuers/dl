# R33 front review: owned material, occluders, and honest coverage

Accepted 25 November 2026, Asia/Hong_Kong. R33 applies the existing [R32 boundary standard](annotation-boundaries-r32.md) across every manifested new-series photograph. It preserves original pixels, stable one-based label rows, earlier review rounds, original trainer-validation labels and the frozen evaluation references.

## Coverage means screening, not pixel-perfect certification

The review covers all 126 `new_*` photographs: 110 nonempty photographs with 4,178 labeled fronts, plus 16 empty scenes. Every photograph received a full-context visual screen. Every labeled row received read-only numerical geometry checks. Suspicious and specifically requested rows received separate original-pixel crops; their exact inspection level is recorded in `results/review/front-review-20261125-r33/coverage.json`.

This does not certify every coordinate to subpixel precision. The 31 old-series photographs are outside this new-photo request. The held training photograph `new_000015` was screened, but its prior training supervision remains held. Ambiguous observations are recorded as retained or unresolved, rather than silently converted into new ground truth.

## Preserve the established target

Box the observed frontal face, including its own visible molded lip, rim and short protrusion. Exclude the broad perforated upper plane, neighboring fronts, rack, jaw and hand. Use contour continuity and occlusion junctions to determine ownership; brightness and color are supporting evidence from that exposure, not a universal material classifier.

The physical upper and lower edge language remains explicit. Stored image y increases downward: top is stored `ymin`; bottom is stored `ymax`. The user's earlier `ymax` and `ymin` wording referred to physical top and bottom respectively. Review prose uses top/bottom to avoid reversing the intended amendment.

The unchanged reference `new_000058`, row 7, remains the own-lip example. Its box and corner tokens are byte-identical to R32.

## Remove all-occluder boundary strips; retain owned extrema

A boundary should move inward when the entire removed strip contains no required observed target material. In `new_000088`, row 51, the old left edge at x749 lies on the jaw. The observed right-side fragment begins around x802 at its upper seam and around x810 lower down. R33 sets the left edge to x802 while preserving the right edge and the visible right landmark. The unavailable left landmark remains `0 0 0`.

The contrasting case is `new_000088`, row 43. A thin owned upper strip around y442-444 extends across the jaw to about x661. Trimming the right side to the broad lower face would cut that strip, so its right edge is retained. Likewise, `new_000066`, row 14, retains right edge x519 because its own upper lip reaches it, even though the hand occupies much of the box interior. Only that row's standalone rack strip on the left is removed.

An axis-aligned box can therefore legitimately include an internal occluder or neighboring pixels around a tilted face. It is not a segmentation mask. A box can also begin at an occlusion boundary even when the physical cassette corner is hidden; that box boundary must not be turned into a visible corner annotation.

## Accepted coordinate changes

Values below are original-image pixels, rounded only for display. The canonical ledger preserves the exact before/after TXT tokens and decoded geometry. Other coordinates and all visibility/class tokens are preserved.

| Photograph | Row | Field | R32 | R33 | Basis |
|---|---:|---|---:|---:|---|
| `new_000003` | 61 | Bottom, stored `ymax` | 579 | 574 | Remove a hand/rack-only lower strip. |
| `new_000006` | 53 | Right, stored `xmax` | 519 | 509 | Remove finger-only extent beyond the visible frontal fragment. |
| `new_000013` | 36 | Left, stored `xmin` | 822 | 828 | Remove the jaw-only strip left of the observed face. |
| `new_000017` | 59 | Right, stored `xmax` | 620 | 617 | Follow the visible taper before the jaw. |
| `new_000034` | 43 | Right, stored `xmax` | 635 | 622 | Exclude the jaw and its vertical slot. |
| `new_000040` | 34 | Right, stored `xmax` | 638 | 620 | Exclude the jaw beyond the owned upper extent. |
| `new_000040` | 35 | Right, stored `xmax` | 636 | 617 | Follow the next, narrower observed fragment. |
| `new_000059` | 2 | Left / right | 804 / 982 | 810 / 981 | Tighten to the original-pixel face contours; preserve top/bottom and points. |
| `new_000066` | 14 | Left, stored `xmin` | 443 | 448 | Remove a standalone rack strip; retain the own upper lip at right. |
| `new_000088` | 49 | Bottom, stored `ymax` | 449.35 | 439 | Stop at the target's lower seam before the next front. |
| `new_000088` | 51 | Left, stored `xmin` | 749 | 802 | Box the observed fragment instead of the gripper. |
| `new_000090` | 40 | Bottom, stored `ymax` | 495.74 | 491 | Remove a lower strip from the following front. |
| `new_000090` | 41 | Bottom, stored `ymax` | 514.98 | 510 | Stop at the owned lower seam. |
| `new_000090` | 46 | Right corner x | 785.09 | 790 | Use the actual lower-right turn; preserve corner y and the box. |

These are 14 amended rows across ten photographs: eight training rows across seven photographs and six development rows across three photographs. All are independently inspected against unmarked original pixels. Pixel uncertainty is generally about 1-3 pixels and is recorded per row; it is a visual ambiguity interval, not a statistical confidence interval.

## Predictions are cues with exact provenance

For `new_000059`, row 2, PR113 candidate output p17 has left/right edges about809.84/980.58. The accepted edges810/981 come from the source contours, not from copying its top, bottom or corner estimates. For `new_000090`, row 46, PR113 p15 and the prior PR110 p25 independently cue a right turn near x790. Their output numbers refer to their own exact ordered prediction caches and are not interchangeable between checkpoints.

PR113 p54 for `new_000088` predicts a hidden left corner around(755.20,466.98). Such model estimates remain permitted by the established project design. They do not establish physical visibility and do not replace an unavailable annotation. A predicted coordinate in a `v=0` reference slot is unscorable for observed-corner error, rather than automatically a false positive.

## Read-only text flags and unresolved cases

The repeatable command `python scripts/triage_front_geometry_r33.py` checks every new-series row. It found zero nonfinite/out-of-image/nonpositive boxes, malformed unavailable points or observed points more than one pixel outside their own boxes. It also records advisory flags: bottom-minus-lowest-visible-corner gaps, small/tall fragments, clipping proximity and neighbor points inside an AABB.

The larger-gap threshold is `max(6 pixels, 0.25 * box height)`. There are 101 such flags. The separate fixed five-pixel screen marks 427 rows, and 976 rows contain a neighbor's visible point more than one pixel inside their box. These counts overlap and are not error counts. All 101 large-gap cases received focused crop screening. One visible corner, tilt, a rim or occlusion can explain a gap; some new lower-boundary corrections emerged only after checking ownership in the pixels.

For example, the precise right extent of `new_000010`, row31, remains unresolved where gray plastic, finger and nail blur together. Several clipped or oblique faces have an uncertain division between an owned lower rim and the next front's upper lip. Those coordinates are retained with explicit review notes. No rule snaps a bottom edge to a corner y coordinate, and no color threshold makes automatic amendments.

## Training and evaluation remain separate

The next native-loss comparison uses the previous eligible R31 labels as its reference and the eight eligible R33 training-boundary corrections as its candidate. Original trainer validation is unchanged. The development amendments are only a separately identified exploratory scoring reference. Earlier fixed references remain available for historical comparison; freezing a reference does not assert that every historical boundary is physically perfect.

After narrowing `new_000088`, row51, a broad historical prediction can fail the IoU matching threshold because it includes the jaw. Report that reference-driven localization change separately from a new model failure to propose any cassette. Review the same raw outputs against preserved R32 and amended R33 before making a performance claim.
