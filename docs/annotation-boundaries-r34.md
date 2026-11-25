# R34: focused boundary and corner amendments

Accepted 25 November 2026, Asia/Hong_Kong. This follows the accepted R33 snapshot with 17 user-named row decisions: 16 amendments and one retention. Seven amended rows in six training photos enter the eligible training overlay. Nine amended development rows remain evaluation-only. The earlier full new-photo screen is inherited; this round does not claim another whole-dataset review.

## Target and coordinate convention

Keep one `cassette_front` class and two actual lower-front turns, ordered by image x. The box encloses observed owned frontal material, including the same front's molded lip and rim. The box is not a segmentation mask. An unavailable corner remains `0 0 0`; a model's estimate of a hidden corner does not create observed annotation truth.

Stored image coordinates have x rightward and y downward. Top is stored `ymin`; bottom is stored `ymax`. The user's physical top/bottom wording is translated explicitly in the ledger. Moving a point upward decreases stored y.

## Include same-instance material through a gripper opening

Visible frontal fragments belonging to the same cassette are one object, including fragments seen through a gripper slot. Enclose all their observed extrema in one axis-aligned box. Gripper pixels between those fragments can occur inside that rectangle; they do not determine its outer boundary. Do not extend the box to an imagined hidden edge or mark the gripper edge as a physical corner.

This is already consistent with the R33 retained decision for `new_000097:39`. The user's ownership observation and original-pixel review resolve R33's mistaken jaw-only interpretation for `new_000034:43` and `new_000040:34,35`. Restore their exact R32 TXT rows and right extents of approximately 635, 638 and 636 pixels. The hidden right points stay unavailable. Preserve the historical R33 record and record the correction here.

## Requested amendments

Values below are original-image pixels, rounded for readability. Exact TXT tokens and original image hashes are in `results/review/front-review-20261125-r34/row-review.json`.

| Photograph | Row | Change from R33 to R34 |
|---|---:|---|
| new_000003 | 61 | Right edge 872 to 868; remove rack-only strip. |
| new_000006 | 53 | Left edge 459 to 464; preserve the visible left corner. |
| new_000013 | 36 | Right edge 868 to 865; right corner (865,508) to (863,503). |
| new_000034 | 43 | Restore R32 right edge 635, including the slot fragment. |
| new_000040 | 34 | Restore R32 right edge 638, including the slot fragment. |
| new_000040 | 35 | Restore R32 right edge 636, including the slot fragment. |
| new_000059 | 2 | Right corner y37.77 to y35; x and visibility unchanged. |
| new_000066 | 14 | Restore observed left corner (451,673), visibility 0 to 2. Right remains unavailable. |
| new_000088 | 51 | Bottom 477.85 to 475; exclude the background-only strip. |
| new_000088 | 43 | Top 438.75 to 443; bottom 463.94 to 460. Retain the owned upper-rim extent across the gripper. |
| new_000089 | 42 | Top 451.32 to 456; retain the soft upper rim and remove the preceding face. |
| new_000007 | 10 | Bottom 18.23 to 9.5, near the physical lower corners. |
| new_000009 | 12 | Bottom 14.49 to 11, near the observed right corner. |
| new_000055 | 23 | Left edge 530 to 528, including the visible side rim. |
| new_000010 | 24 | Top 459 to 464, excluding the preceding face. |
| new_000057 | 1 | Right edge 878.12 to 881, including the outer rim. |
| new_000009 | 51 | Retain top 474. The cited prediction top 473.932 differs by only 0.068 pixels. |

Typical coordinate ambiguity is 1-2 original pixels, with per-row values in the ledger. These are reviewed coordinates, not subpixel metrology. Only explicitly named fields change; nearby labels are not moved by a global offset or by a white-pixel rule.

## Prediction identities are bound to their source run

The user's magenta prediction cues are from the PR113 candidate (seed 47, factor-two box loss, NMS 0.70), not the PR114 candidate. A pID is its one-based position in the saved prediction cache and is not a TXT row number.

| Image / PR113 prediction | Reviewed TXT row | Field |
|---|---:|---|
| new_000007 / p54 | 10 | Bottom |
| new_000009 / p56 | 12 | Bottom |
| new_000009 / p58 | 51 | Top, retained |
| new_000010 / p31 | 24 | Top |
| new_000055 / p27 | 23 | Left |
| new_000057 / p22 | 1 | Right |
| new_000089 / p52 | 42 | Top |

A duplicate prediction may have a useful boundary without its full box or estimated hidden corner becoming annotation truth. For the two clipped top-row cases, the user asks to bring the bottom toward the observed corner; the accepted bounds are tighter than those duplicate predictions. The PR113 cues and exact replayed coordinates are retained separately for provenance.

## Keep annotation changes separate from model improvement

All object rows, original images, splits, original trainer-validation labels, and frozen 666/663 scoring references remain unchanged. There is one explicitly observed corner restoration, so R34 is not a box-only treatment. The next fixed pair compares R33 versus eligible R34 supervision under the same native loss and training recipe. PR35 remains selected; the revised development reference is a separately named diagnostic. A rescore after changing labels is not a new model result.
