# R35: shaded seams, owned fragments, and explicit exclusions

Accepted 25 November 2026, Asia/Hong_Kong. This focused follow-up to the accepted R34 snapshot resolves 11 user-named rows: nine replacements and two explicit instance exclusions. Five replaced rows in three training photographs enter the training treatment. Four replacements and both deletions are development-only. Earlier screening remains part of the history; this round does not certify the complete dataset.

## Ownership and coordinate convention

The target remains one observed cassette front with its own molded frontal lip, rim, short protrusion, and shaded supporting portions. Follow contour continuity, stacking geometry and occlusion junctions before interpreting colour. A shaded plastic rim can belong to the target; a bright metal strip need not. Exclude another cassette, standalone rack or gripper, and the broad perforated upper plane.

The box spans the extrema of owned observed material. The same cassette can appear in separated pieces through a gripper opening. Its rectangular box may include the intervening occluder. Do not extend it to a different column merely because pale material is visible in another opening.

Stored x increases rightward and stored y downward. Top is `ymin`; bottom is `ymax`. The user's upward-positive wording means decreasing stored y. Reviews use top/bottom and state numerical directions explicitly. Keypoints are actual lower-front turns, not arbitrary box corners; unavailable points remain `0 0 0`.

## Accepted changes

Values below are original-image pixels, rounded for reading. Exact before/after label tokens and image hashes are in `results/review/front-review-20261125-r35/row-review.json`.

| Image | R34 row | R35 decision |
|---|---:|---|
| new_000006 | 53 | Right edge 509 to 503; exclude the finger and neighboring-only strip. |
| new_000013 | 36 | Left edge 828 to 833 and bottom 512 to 505; retain the observed right point at (863,503). |
| new_000034 | 43 | Right edge 635 to 640; include the full owned fragment in the narrow left-jaw slot. The central opening near x700 belongs to another column. |
| new_000034 | 34 | Left 427 to 426, top 505.05 to 514, bottom 547 to 546; follow the shaded transitions of the lowest front. |
| new_000034 | 35 | Left 511 to 509, top 255 to 258, bottom 298 to 300; include the own shaded rim. |
| new_000009 | 51 | Top 474 to 472 and bottom 490 to 488; follow the two shaded transitions. |
| new_000007 | 10 | Right edge 879.53 to 874; remove the rack/neighbor strip. |
| new_000010 | 24 | Right edge 911 to 910; right point x911 to x909 at the same y491 and visibility 2. |
| new_000057 | 1 | Right edge 881 to 883; include the own rim and supporting shaded portion. |
| new_000009 | 56 | Delete the actual R34 label row, as explicitly requested. |
| new_000009 | 57 | Delete the actual R34 label row, as explicitly requested. |

The requested rows 34 and 35 are in `new_000034`, distinct from the `new_000040` rows amended in R34. The phrase "new_000057 row 1 required positive" is interpreted as continuation of the preceding right-edge request: increase `xmax`. The ledger records that interpretation.

The `new_000010:24` point adjustment is a necessary local consistency correction supported by the observed lower-right turn. The full review therefore changes a corner as well as boxes, while the eligible training treatment remains five box-only changes. No retained point changes visibility.

Visual boundary ambiguity is approximately 1.5-2.5 original pixels, recorded per row. Soft shaded seams are not subpixel measurements, and these intervals are not statistical confidence intervals. Unrequested neighboring rows are preserved.

## Deletions preserve their historical interpretation

The latest user instruction excludes `new_000009` rows 56 and 57 from the revised development reference. These are label-row identities, not prediction IDs. Row 56 descends from original row 57, previously narrowed around the right-jaw fragment. Row 57 descends from the lower fragment added during the November 18 sequence split. The independent review records those source documents.

The current pixels do not by themselves prove that either patch cannot be cassette material. Record these as explicit user-directed instance exclusions, not confirmed duplicate annotations or a newly proven physical classification. Their prior labels and interpretations remain in R34 and earlier snapshots.

Both rows are at the end of the R34 file. The revised file has 55 rows, with identities 1-55 unchanged and parent rows 56/57 mapped to deleted. The reviewed development reference changes from 661 to 659 fronts, while the two-locatable-corner denominator stays 583. Deletion effects must be separated from model learning in every comparison.

## Controlled next comparison

R35 compares eligible R34 versus eligible R35 training labels under the same native Pose26 recipe and independent epoch-173 initialization. Original trainer-validation labels and the frozen 666 / historical 663 evaluation references are separate. Preserve completed PR117 results against R34; identify rescoring against R35 as a reference-only change. No automatic model promotion follows from an annotation edit.
