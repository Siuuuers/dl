# R26: visible cassette boundaries and partial fronts

23 November 2026, Asia/Hong_Kong.

This bounded review starts from immutable R25 labels at
`ace99d4aa71e5c20d7c5d2445f18fdab58d63496`. Row numbers refer to that review.
All 22 user-named rows in eight development images were inspected against the
original 1280x720 pixels. A bounded training-analogue pass adds three corrected
rows in photo013. This is not a new full-dataset certification.

## Physical annotation rule

Box the visible cassette front and its visible rim. Exclude rack-only and
gripper-only strips, but accept that a rectangular box around a face occluded
diagonally necessarily contains some occluder pixels. A box is not a mask.
Place each locatable landmark at the visible lower front-face edge turning
into that face's side. Brightness and curved shadow help trace the material
boundary; neither white pixels alone nor a fixed inward offset defines it.
Preserve unavailable corners as exact zero-valued triplets. Preserve the
eleven accepted standard rows from R25 byte for byte.

Root and separate original-pixel reviews inspected scenes, enlarged unmarked
crops, and label overlays. Model predictions did not choose the amendments.
Small adjustments are practical coordinates for blurred images, not claims
of subpixel physical precision. Typical local point uncertainty remains
roughly one or two original pixels; these are qualitative review estimates.

## Decisions and wording interpretations

The [row ledger](../results/review/front-review-20261123-r26/row-review.json)
records all exact before/after lines, source hashes, reasons and retained rows.

| R25 identity | Decision |
| --- | --- |
| 007:4 | Move xmin right by1px, from644.5 to645.5. Interpreting the wording as row1's xmin564 would move left and include an unrelated stack. |
| 007:8,9,14,15 | Small right-edge trims agree with the visible local rims and the suggested upper-row right-point x values. |
| 008:5 | Right point to(1205.5,22), xmax to1207, following the actual front turn. |
| 008:36 | Delete the blurred bottom patch: a separate cassette front is not established. |
| 009:48-52 | Move xmin off the gray rail and each left point to its own front turn. Hidden right points stay missing. |
| 010:24 | xmin797 to810, excluding the gripper-only strip. |
| 011:54 | Retain: the suggested neighboring-coordinate clamp would cut away the visible face. |
| 053:1 | xmax approximately710 to686, at the visible top-clipped transition; the width ratio alone is not the evidence. |
| 089:48,49 | xmin to798 and811 respectively, at the visible cassette/gripper intersections. |
| 090:39-42 | Retain: left landmarks are lower corners, not suitable top-box boundaries. |
| 090:43 | Move **ymin**677.38 to670 to include the upper cap. The requested feature lies above the old box; increasing ymax would add support below it. |
| 013:29-31, training | Three left-corner corrections; rows29 and30 also trim xmin by1px. |

The proposed universal neighboring-cassette constraint is contradicted by
011:54 itself: its left/right turns are at x603/x668, while row53's left/right
turns are at x602/x666. A lower row can project farther right. A landmark on
one front cannot dictate another front's box, or even its own rim extreme.
The TXT scan supplies review candidates only; it cannot identify material
ownership without the pixels. No global coordinate clamp is implemented.

## Small and unresolved objects

Keep an identifiable cassette front even if it is small, partially occluded,
or difficult for the model. An unlocatable corner does not invalidate the
whole instance. The sole deletion here is a reviewed target-identity decision,
not a minimum-size rule or an attempt to improve recall by removing hard cases.
Do not silently remove real objects from an otherwise retained training image:
the ordinary YOLO object-row format does not make that region an ignore mask.
Keep genuinely ambiguous cases in the review ledger and preserve the existing
whole-photo hold for training photo015. Future ignore-region handling would
require an explicit training-policy change.

The bounded analogue checks retain006 rows50-52/60,012 rows14/25-27/35-38/40,
and013 rows28/32/35-36. Row013:32 remains too soft near the jaw for a better
coordinate to be established. The preserved current label is not a claim of
exact physical certainty, and this pass does not lift any inherited hold.

## Training and evaluation separation

R26 makes19 replacements and one deletion across eight changed photographs:
three replacements in training,16 replacements and one deletion in development.
There are18 box changes and nine corner changes in total. Training changes are
only three left corners and two left box edges in013. The review has661
development objects; deleting008:36 removes a zero-corner row and changes no
two-corner denominator. Original trainer validation still has664 objects;
frozen666/663 scoring references are untouched.

The next notebook retains the two-point model and fixed paired recipe. The
R25 result provides the saved predecessor comparison; PR35 remains selected.
Changes in matching caused by the new reference boxes must be described as
reference changes, never as learned model improvement. The requested PDF is
an error review of retained PR35 predictions, not an amendment atlas or a
missed-object report.

Format reference: [Ultralytics pose dataset documentation](https://docs.ultralytics.com/datasets/pose/).
