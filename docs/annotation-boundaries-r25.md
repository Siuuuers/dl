# R25: small front-face corner amendments

23 November 2026, Asia/Hong_Kong.

The immutable parent is the [R24 review](annotation-boundaries-r24.md) at
`47a1498f53b3bcd9cc575d0f7c6e4e4de8e2d4b8`. All row identities in this pass
refer to that saved review. The user named `008:35`, `010:18-19`, and
`012:3,28`, and supplied nine accepted standards: `008:18-21,28-32`.
Those nine complete lines and the earlier `010:16` and `016:24` standards
are preserved exactly.

## Landmark and scope

Trace the visible lower edge of the cassette front face to its turn into the
side of that face. The projecting lip, side-wall silhouette, box extreme and
shadow cast on neighboring material do not define the landmark. Shadow and
brightness help interpret the local edge, but neither a brightest-pixel rule
nor a uniform inward offset is used.

Both endpoint slots of the five named rows were inspected using the original
1280x720 pixels. Three adjacent rows show the same clearly identifiable local
problem: `010:20` and `012:1-2`. Their own visible front-face junctions were
reviewed separately. All eight accepted changes are left-corner XY changes;
right-point coordinates are retained. This is a bounded amendment of three
photographs, not a claim that the entire dataset has been reviewed again.

Full original scenes were inspected before local crops and saved-point
overlays. Root and a separate original-pixel review reconciled the decisions;
model predictions were not annotation targets. Rounded molding and blur leave
about one or two pixels of qualitative placement ambiguity at most changed
corners, and about two or three at the shadowed `012:28` left corner. These
estimates are not calibrated error bounds.

## Missing and ambiguous points

`008:35` has only a locatable left corner. Its right physical corner is outside
the image, so its exact missing-point tokens remain unchanged. Reviewing both
endpoint slots does not require inventing both coordinates.

The right corner of `012:28` remains at its saved coordinate. The front face,
shadow and neighboring cassette layer merge into a soft boundary; an upward
alternative near `(329,700)` is plausible but cannot be established as a more
accurate material-edge turn from the original pixels. Two separate local
reviews support retention with this limitation. This is insufficient evidence
to amend the existing locatable point, not confirmation of exact physical
placement. Its visibility remains `2`; no hidden point or sharper synthetic
edge is introduced.

Class, boxes, visibility, row identities, object counts and image splits are
unchanged. The [point ledger](../results/review/front-review-20261123-r25/point-review.json)
records the before/after lines, both endpoint decisions, evidence and the
retained soft-boundary uncertainty.

## Training and evaluation

`new_000008` and `new_000010` are development images. Their four corrected
points belong only to the separately versioned R25 review. Four corrected
points in training photograph `new_000012` enter the candidate. The original
20-image, 664-front trainer validation and frozen 666/663 scoring references
remain unchanged. Inherited held/excluded training-photo handling is retained.

The user has authorized the next notebook after these amendments. It keeps
the existing two-point architecture and matched training recipe, testing the
updated supervision. These eight point amendments neither constitute new
camera observations nor establish that the next trained model will improve.
