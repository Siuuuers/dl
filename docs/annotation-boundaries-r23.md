# R23: lower corners of the front face

The completed [r22 review](annotation-boundaries-r22.md) is the immutable parent.
R23 applies the user's four named right-point corrections against its exact rows:
`new_000001:27` and `new_000002:25–27`. The accepted standards are
`new_000001:1,19,25` and `new_000002:28`; they remain byte-identical.

The landmark is the visible lower corner of the cassette's **front face**.
Follow the lower front edge to where it turns into the side wall. A projecting
side wall, rounded outer silhouette, upper lip or rack edge is not a substitute.
The box still encloses the cassette's visible material under the r22 rule;
the landmark therefore need not coincide with the box's horizontal extreme.

Use the original camera pixels and neighboring cassette geometry. White versus
grey is supporting evidence: shadow can darken a cassette and reflection can
brighten a rack. There is no automatic colour threshold or fixed inward offset.
The four chosen movements range from five to seven original pixels in X and
are individually judged, including the different Y adjustment of `002:27`.
Blur and rounded molding leave approximately one or two pixels of visual
placement ambiguity; the coordinates are annotations, not subpixel measurements.

All four points stay visible (`2`). The two points retain image-X ordering.
If a corner cannot be located in another case, its established representation
remains `0 0 0`; no hidden corner is guessed. This amendment changes no box,
left point, visibility value, class, row identity, instance count or split.
Only the four named points and their nearby context were newly inspected.

The r22 PDF remains the completed r22 snapshot. The r23 comparison image reads
the new saved labels. All 155 other label files, all development labels and
the frozen scoring references retain their previous bytes. The new training
candidate includes r22 plus these corrections, using the existing two-point
architecture and paired recipe. Preparing it does not establish a model gain.
