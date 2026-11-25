# R32 cassette front standard: upper lip, corners, and ownership

Accepted 25 November 2026, Asia/Hong_Kong. This standard uses verified original PNGs and the immutable R31 labels. The R32 review covers 25 development rows: 23 amended and 2 retained. A related upper-lip sweep screened 113 rows across four photographs. These bounded checks do not certify every annotation in the dataset.

## Coordinate language

The user confirmed that their `ymax` means the **upper** edge and their `ymin` means the **lower** edge. Stored image coordinates increase downward, so the stored field names are reversed relative to that terminology:

| Physical instruction | Stored field | Direction of change |
|---|---|---|
| Upper edge, user `ymax` | `ymin` | Up decreases the value; down increases it. |
| Lower edge, user `ymin` | `ymax` | Down increases the value; up decreases it. |
| Left edge | `xmin` | Left decreases the value. |
| Right edge | `xmax` | Right increases the value. |

Review prose uses **upper**, **lower**, **left**, and **right**. The change ledger also records the stored field and exact numerical direction. Physical intent controls interpretation when a sentence combines an edge name with an inconsistent sign.

Several earlier reviews inspected or retained the lower edge when the user was asking about the upper edge. Those decisions did not answer the intended request. R32 corrects this mapping and reviews the relevant upper boundaries again.

## Which components belong in the box

Box the observed frontal face **including its own visible molded frontal lip, rim, and short protrusion**. A shaded part belongs in the target when contour continuity and the stacking pattern support that ownership. Exclude the broad perforated horizontal upper plane, the rear or whole three-dimensional silhouette, neighboring cassettes, the rack, and the gripper.

The earlier instruction to exclude “upper surfaces/projecting tray” does not exclude every component above the broad front. The distinction is between the front's own short molded rim and the broader upper plane or another object's structure.

The reference `new_000058`, row 7, illustrates this standard. Its unchanged box is `(801, 75)–(970, 116)`. The small left lip lies around x802–839, y78–88, while most of the broad pale front starts near y90. The box therefore includes the front's own lip above its main face. The following cassette has a separate lip near y119–128, which does not belong to row 7.

In `new_000010`, row 25, the raised frontal lip near y340–350 is the analogous included component. The perforated plane above it remains excluded. The accepted upper edge changes from stored `ymin = 355.34` to `340`.

## Determine ownership before using color

1. Locate the broad frontal plate and the visible turns of its lower corners.
2. Follow its sides upward to its own short frontal lip or protrusion.
3. Use repeated cassette geometry and occlusion junctions to separate that lip from the upper cassette's shaded side face.
4. Place the box at the extrema of the observed components belonging to the target front.
5. Confirm the seam in the original pixels with prediction overlays hidden.

Brightness is a supporting cue. Cassette plastic can appear gray in shadow, and metal can be bright under reflection. A gray triangle may be a cassette side face, a cast shadow, or rack structure. A global rule that equates white with cassette and gray with rack would give inconsistent results. Any recorded color example must retain its source image, coordinates, and surrounding geometry.

The box is an axis-aligned rectangle, so it can contain some pixels from a neighboring object at a corner when the target is slanted. Remove an upper, lower, or side strip when it contains no required part of the target. Shrinking until every interior pixel belongs to the target would cut valid slanted fronts and protrusions.

## Rack reference

In `new_000090`, rows 18–20, the darker strip on the right continues vertically across cassette gaps and connects to the repeated rack lattice. Each front has its own rounded or slanted termination before that strip. The accepted right box edges are x778, x775, and x773 respectively. The right landmarks remain at x775, x773, and x771.

The source hashes and sampled pixel values are recorded in `results/review/front-review-20261125-r32/local-color-reference.json`.

| Local sample | Center (x, y) | Median RGB |
|---|---|---|
| Rack beside row 18 | (784, 278) | (159, 161, 161) |
| Rack beside row 19 | (780, 303) | (107, 106, 104) |
| Rack beside row 20 | (777, 330) | (68, 65, 68) |
| Front of row 18 | (720, 278) | (192, 192, 191) |
| Front of row 19 | (720, 304) | (189, 187, 189) |
| Front of row 20 | (720, 330) | (181, 183, 183) |
| Shaded frontal rim | (740, 291) | (171, 170, 163) |

These are examples from one exposure. They do not measure material reflectance and do not define a reusable color classifier. In particular, the rack examples themselves span a wide range of brightness. Shape continuity and component ownership remain necessary.

## Keypoints mark the lower frontal corners

Each keypoint marks the visible physical turn of the frontal lower edge into its side. It does not automatically coincide with a box corner or with the outer end of an upper cap. A rounded or blurred corner warrants a small review interval rather than a claim of subpixel certainty.

For `new_000008`, row 7, the R31 left point `(1217, 22)` lies on the lower gray seam. The accepted point `(1221, 20)` follows the brighter face turn, within the visually plausible region x1219–1223, y19–21. Moving only x while holding y22 would leave the point on the seam. The unavailable right point remains `(0, 0, 0)`.

The user's suggested comparison between the lower boundary and the lower-corner y coordinates is useful as a review flag after the naming correction. In stored coordinates, compare `ymax` with the y values of the visible lower corners. Perspective, a visible rim, occlusion, and rounding can produce valid gaps. A text-only check should request visual inspection rather than automatically move a boundary. The upper edge has no corresponding requirement to be close to the lower corners.

The requested removal of the clipped left point of PR109 output p54 maps to `new_000009`, row 12. Its left visibility changes from 2 to 0 and its left coordinates become `(0, 0)`. This reduces visible development endpoints by one and two-corner pairs by one; it does not remove an object.

## Accepted upper-edge changes

All values below are original-image pixels. Each change affects stored `ymin`, which is the physical upper edge. The complete ledger also records lower-edge, horizontal-edge, landmark, and visibility changes where applicable.

| Image and row | R31 upper | R32 upper | Reason |
|---|---:|---:|---|
| `new_000008`, row 4 | 19.22 | 12 | Include the front's own left lip. |
| `new_000008`, row 6 | 21 | 18 | Include the upper turn of the own lip. |
| `new_000008`, row 10 | 411.41 | 421 | Exclude the preceding front's lower strip. |
| `new_000008`, row 12 | 263.11 | 257 | Include the raised frontal rim below the perforated plane. |
| `new_000008`, row 19 | 152.02 | 142 | Include the raised left lip. |
| `new_000008`, row 20 | 202.54 | 194 | Include the own left lip found in the consistency sweep. |
| `new_000008`, row 29 | 373.97 | 370 | Include the own lip above the broad front. |
| `new_000010`, row 25 | 355.34 | 340 | Include the raised frontal rim and exclude the perforated plane. |
| `new_000055`, row 21 | 14 | 21 | Exclude upper-cassette material while retaining the own lip. |
| `new_000086`, row 48 | 448 | 445 | Reach the shaded upper seam of the target. |
| `new_000089`, row 43 | 468.31 | 474 | Exclude the preceding front above the seam. |
| `new_000089`, row 48 | 437.70 | 442 | Follow the own frontal rim below the perforated plane. |
| `new_000090`, row 39 | 443.41 | 449 | Exclude the preceding front and retain the target's upper seam. |

The optical edge locations have approximately 1–3 native pixels of uncertainty. The independent review files record these intervals where assessed. They describe blur and visual ambiguity, not statistical confidence intervals.

## Evidence and scope

The canonical change ledger is `results/review/front-review-20261125-r32/operations.json`. Independent surface decisions, additional sweep findings, and corroborating review are recorded in `independent-surface.json`, `independent-surface-extra.json`, and `independent-root-endorsement.json` in the same directory. Original-pixel crops with coordinate rulers and numbered anatomy examples accompany the amendment report.

The two recovered phone screenshots show `new_000016`, rows 26–27, and `new_000056`, row 12. The referenced yellow-marked `new_000008`, row 4, photograph was not recovered. The accepted row 4 boundary is grounded in the original source pixels and the confirmed upper/lower convention; that marked photograph remains an optional cross-check if the user intends a different seam.

The bounded upper-lip sweep screened 113 rows: 35 in `new_000008`, 34 in `new_000010`, 26 in `new_000055`, and 18 in `new_000058`. It identified two further upper-only corrections, `new_000008` rows 12 and 20. It was a visual screen followed by detailed inspection of convincing findings, not a certification that all 113 rows are perfect.

Legacy protected examples `new_000008`, rows 19, 20, and 29, are revised only at their upper boundaries under the clarified own-lip standard. R31 remains immutable, and every other field of those standards is preserved. The reference `new_000058`, row 7, remains byte-identical.

All 25 reviewed rows are development images: 23 are amended and 2 retained. The 23 changed rows span nine photographs. No training image, original trainer-validation label, frozen 666-object reference, or historical 663-object reference changes. R32 contains 661 development objects, 1,225 visible endpoints, and 97 missing endpoints. The training supervision remains the same as R31.
