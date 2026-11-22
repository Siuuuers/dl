# R22: colour-assisted contour review and actual lower corners

R22 continues the visible-front definition in [r21](annotation-boundaries-r21.md).
The immutable input is the r21 review in commit
`41dd2be835f5b0e30b10e15431c6795cb00f7388`.

## Material ownership

Enclose this cassette's visible front, its own molded rim and its visible rounded
or sloping side contour. Include an identifiable visible patch of that front
through an opening. Exclude unsupported outer margins belonging to the rack or
another cassette. Do not infer a hidden three-dimensional body.

White-versus-grey appearance is a useful inspection cue. R22 compares local
edge brightness with nearby front material and inspects continuous grey strips.
It does not apply an automatic colour threshold to label geometry. Shadowed
cassette material can be grey, rack reflections can be bright, and blur blends
the two at their boundary. A flagged edge needs original-pixel contour review;
an unflagged edge is not certified correct.

Inspect a suspicious boundary with the original image, the current annotation,
and surrounding rows visible. Trace which material belongs to the current
cassette. Preserve its projecting rounded wall; do not borrow the wider contour
of the cassette above it. Use the existing accepted standards 094:31-32,
086:3 and 087:5 as unchanged references.

## Independent landmarks

The two points locate the actual lower-front corners, ordered by image X.
They need not coincide with horizontal box extrema. A box correction must not
silently snap a point to the new edge or impose a constant point-to-box margin.

The user authorizes corrections to points on the rack or away from the actual
corner. R22 therefore permits independently reviewed X and Y repositioning.
Each changed point records its old and new coordinates, visibility, reason,
original-image evidence region and a separate review. Unrelated point fields
remain byte-identical.

A corner that cannot be located is represented by the exact normalized triple
`0 0 0`. Visible corners use visibility 2. Removing a point does not remove the
cassette row. The explicit right-point removals on 096:37 and 096:38 retain both
object identities and all their other fields unless separately adjudicated.

## Scope and vertical-coordinate interpretation

Finish the 40 supplied photos in 081-130, then perform the colour-assisted pass
over the 70 supplied photos in 001-080. The 20 absent IDs are recorded in the
coverage ledger. No old-series labels or empty scenes are edited in this round.

Horizontal box bounds may move inward or outward when supported by visible
material. The only requested vertical-box case is 094:23. Image Y increases
downward. In r21 its bottom is approximately 656 px, while row 30's bottom is
approximately 635 px; the wording "increase ymax" and that numeric comparison
cannot both be applied literally. Original-pixel review shows its own upper lip
and part of its face were clipped. Restore that visible material by decreasing
`ymin` to 605 px, preserving `ymax` at approximately 655.99 px, and extending
`xmin` to 342.5 px for the own visible notch. This is an upward expansion, not a
literal increase of the bottom coordinate. The audit records the independent
review and this interpretation of the requested vertical extent.

All classes, row identities, instance counts and split assignments are preserved.
Every change must replay exactly from the immutable r21 line. Geometry checks
establish valid serialization and consistency; they do not prove material
ownership or certify the full dataset.

## PDF and training are separate views

The full PDF displays every photograph with nonempty current labels, including
unchanged and old-series photographs. It shows all current rows on each included
photo. Cyan marks retained fields and magenta marks changed fields. Missing
points have no marker; removed-point identities are recorded in the PDF QA data.
Original pixels and annotation coordinates are preserved. Tag placement may move
for readability without moving any annotation.

The training candidate keeps the established cohort and eligibility decisions.
Photo 015 and the 31 old-series photos retain exact PR35 supervision in both
training arms. Development-label corrections remain outside training. Original
trainer validation and frozen 666/663 scoring references remain unchanged; the
current review is separately identified as exploratory.

PR96 is evaluated against its pre-return r21 and frozen references. Later r22
annotations do not rewrite that evaluation. Preparing the next notebook does
not establish a better model or select a checkpoint.
