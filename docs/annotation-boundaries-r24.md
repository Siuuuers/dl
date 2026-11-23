# R24: front-face corner amendment

23 November 2026, Asia/Hong_Kong.

The immutable parent is the [r23 review](annotation-boundaries-r23.md). Row
numbers refer to that version. This focused pass inspects both endpoints of
21 named rows: `010:15,17–19`, `012:4,6–9`, `014:21–25,31,32` and
`016:18,19,25–27`. The accepted standards `010:16` and `016:24` remain exactly
unchanged. The intervening `012:5` has the same visible right-point problem
and is the single additional adjacent row in this pass. It is identified
separately from the user's 21 named examples.

The landmark remains the actual visible lower corner of the cassette's
**front face**, ordered left to right by image X. Trace the lower front edge
until it turns into the side of that face. A projecting side wall, upper lip,
shadow on the rack or bounding-box extreme does not define this point.
Boxes retain the r23 material-extent convention and all their existing tokens.

The user's shadow-to-bright transition is useful supporting evidence. Inspect
the local brightness change together with the continuation of the bottom edge,
the side edge and neighboring cassette geometry. A white cassette can appear
grey under shadow, and a rack can have a bright reflection. No fixed colour
threshold, fixed inward offset or brightest-pixel rule is used. Rounded molding
and image blur leave approximately one or two original pixels of placement
uncertainty in most crops. The shadowed right corners `016:25–27` have a softer
transition, with approximately two or three pixels of visual ambiguity. These
are qualitative estimates, not calibrated error bounds; the selected integer
positions do not claim subpixel physical accuracy.

Each endpoint receives an explicit corrected or retained decision. Good points
are kept even when the opposite point needs correction. All 44 reviewed endpoints
are locatable and retain visibility `2`; class, boxes, instance identities,
row counts and image splits are unchanged. Missing points elsewhere retain
their existing zero representation. No hidden corner is inferred.

Photo `new_000010` is in the development split. Its r24 review coordinates are
saved for inspection and the separately pinned exploratory reference. They
never enter training or replace the original 664-front trainer validation or
the frozen 666-front / 663-front scoring references. The other three named
photos remain training photos. The r22 PDF and all earlier review and evaluation
artifacts remain their own immutable snapshots.

The exact per-endpoint reasons and saved coordinates are in
[point-review.json](../results/review/front-review-20261123-r24/point-review.json).
The new notebook tests the accumulated eligible supervision with the unchanged
two-point architecture and paired recipe. Annotation correction itself is not
evidence that a trained model has improved.
