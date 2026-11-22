# Round 21: visible cassette contours

The immutable parent is the saved r20 review. The user identified both remaining rack strips and clipped cassette material and supplied **new_000094 rows 31 and 32** as the new comparison standard. This round corrects the horizontal extent of the 40 supplied nonempty photographs in 081–130. IDs 091–093 and 104–110 are absent from the prepared source.

## What belongs to one front

Enclose the visible cassette front, its connected molded rim, and its visible rounded or sloping side contour. Follow material ownership in the original camera image: the contour must belong to that cassette, rather than to a vertical rack rail, a finger, another cassette, or a background gap. Gray or dark pixels are not automatically rack; a shaded plastic wall remains cassette material when its local contour and neighboring context establish that ownership.

The target remains the visible front of one cassette. Include an identifiable visible patch of the same front through a gripper opening as well as its main exposed portion; an occluder can separate those visible patches. This is not an inferred rectangle around the hidden three-dimensional cassette. Do not invent concealed extent, copy the width of a neighboring object, or include a neighbor merely because it appears inside the current vertical box interval.

For the visible target region F, the horizontal bounds conceptually follow min(x in F) and max(x in F). These extrema can occur above the lower corners. Rounded walls, a beveled rim, perspective, and partial occlusion all make endpoint X an unreliable substitute for the silhouette extrema. No segmentation mask is claimed or added by this definition; the saved target is still an axis-aligned box and two lower-front keypoints.

The user's reference rows 094:31–32 are preserved exactly. Earlier accepted examples 087:5 and 086:3 are also preserved. Their numerical point-to-box margins are not a universal padding rule. Inspect the current object's own pixels. A rectangular box around a sloping contour can contain some background at a corner even when its extrema are correct; removing that incidental background would require a different target representation.

## What went wrong in r20

The earlier text distinguished outer contours from lower corners, but the implementation allowed only shrinking boxes and the visual judgments did not consistently honor that distinction. Some boxes were tightened to lower points while a rounded wall extended farther out. Other boxes and a few existing corner points remained on a continuous rack rail. Numerical validity, exact replay and point containment did not establish physical correctness.

This round therefore permits both inward trims of rack-facing margins and outward restoration of clipped visible material. Every one of the user's 46 named edges has a recorded before/after disposition. Wider review uses the same visual rule; it is not a claim that the full dataset is certified.

## Points and boxes have separate meanings

The two points remain the locatable actual lower-left and lower-right front corners, ordered by image X. Preserve their coordinates unless a point itself is visibly on the rack and conflicts with a physically supported box correction. Such a repair must have an independently reviewed original-image crop, a specific physical reason, exact before/after coordinates, and its own ledger entry. Do not automatically move a point to a new box edge. A real lower corner may lie inside the box's horizontal extent.

This pass permits only separately supported endpoint-X corrections. All Y coordinates, visibility values, classes, row identities and object counts remain unchanged. Missing points remain the exact inherited sentinel. If the image does not independently support a new corner location, numerical convenience does not justify inventing one.

Review the native image and enlarged crops together. Enlargement exposes existing pixels; it does not recover lost detail. Soft edges may have roughly pixel-scale uncertainty, so long decimal serialization is reproducibility rather than subpixel measurement accuracy.

## Review, training and scoring

Save exact parent-to-current replacements and bind their image, label and audit hashes. The PDF renders the saved annotations without moving their coordinates. Magenta indicates changed fields; cyan indicates retained fields. Endpoint repairs are highlighted separately from boxes.

Keep the established two-point Baomai architecture and paired recipe: original epoch 173 independently initializes both arms; seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW and initial learning rate 0.0001; select final-epoch checkpoints. The resource trial never initializes either full run.

Both arms keep 137 training images, including 13 empty scenes. Photo 015 and the 31 old-series photos keep identical PR35 supervision in both arms. The five reviewed development images 086–090 do not enter training. Original trainer validation stays at 20 photos / 664 fronts. The frozen 666-front primary and 663-front historical references remain immutable; the separately versioned current development review is exploratory.

PR94 is measured against references that existed before its returned checkpoints. New r21 annotations cannot retroactively improve that result. The next experiment measures accumulated eligible r21 supervision against PR35 supervision; it does not isolate the causal effect of the latest boundary changes. Preparing annotations or a notebook does not establish improved model accuracy or select a replacement model.
