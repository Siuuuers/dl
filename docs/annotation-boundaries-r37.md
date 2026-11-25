# R37: supplied markings resolve two retained requests

Reviewed 26 November 2026, Asia/Hong_Kong, after PR121 returned the R36 run. This round resolves two current user-marked training rows in two original photographs. One right box edge and one observed left-corner coordinate change. The earlier `new_000010:24` landmark question is carried forward unchanged as an inherited pending request, not a third current amendment.

The two JPEGs are actual attachments supplied in this conversation. Their bytes are preserved under `results/review/front-review-20261126-r37/user-markup/`. The ledger binds both original JPEG hashes, the original PNG and label hashes, the screenshot registration, and the independent pixel review. R36 correctly recorded that these markings were unavailable at that time; R37 records their subsequent receipt and adjudication.

## Physical target and coordinate convention

The target remains the observed cassette front with its own molded lip, rim, short protrusion, and shaded supporting material. Follow contour continuity, stacking and occlusion boundaries. Colour supports ownership; it cannot identify the target by itself. Exclude separate rack, gripper, and material above or beside the selected frontal surface. An axis-aligned box can contain internal finger pixels while enclosing the extrema of owned visible material.

Stored x grows rightward and y downward in the original 1280 x 720 image. Physical top is `ymin`; bottom is `ymax`. A point marks an observed lower-front turn, not an arbitrary box corner or occluder edge. Unavailable corners retain their original zero triplets. Screenshot scaling and hand-drawn marker placement do not provide subpixel physical accuracy; the two selected coordinates have approximately 2 original pixels of visual ambiguity.

## Two accepted amendments

| Image | Row | R36 to R37 | Preserved fields |
|---|---:|---|---|
| new_000006 | 53 | Right edge `xmax` 503 to 495 | Left 464, top 563, bottom 586; observed left point (465,580), v2; unavailable right point. |
| new_000034 | 35 | Left point (518.35,297.09) to (517,294), v2 | Entire box (511,255,637,300), right point (635,284), class and visibility. |

Displayed coordinates are rounded for reading. Exact before/after tokens are in `results/review/front-review-20261126-r37/row-review.json`.

### new_000006 row 53: follow the selected sloping front

The saved teal box in the first attachment has original bounds (464,563,503,586). Registering its visible edges gives about 26.69 screenshot pixels per original x pixel and 26.70 per y pixel. The yellow mark maps to roughly x493.3-495.8 and y570.1-574.3, centered near (494.6,572.1). The integer choice `xmax=495` follows both the marker and the original contour.

The pink upper outline selects the lower sloping front: its upper seam begins around (472,564) and reaches the finger around (495,572). Pale material continuing toward the old x503 lies above that selected seam, outside this reviewed front. The rectangular box may still contain internal finger pixels; eliminating every occluder pixel would cut the front.

The lower pink stroke reaches approximately y587.6 near the finger, but it is thick and hand drawn. That roughly 1-2 pixel difference from the saved bottom 586 is within marker placement and raw-seam ambiguity. It does not justify changing the bottom. The other edges and both point triplets remain exact.

### new_000034 row 35: the orange mark identifies the turn

The second attachment is a user-edited view of the earlier A/B pixel grid. A=(511,297.09) and B=(518.35,297.09) were analytical anchors in the prior report; the orange and pink markings are newly supplied user evidence. A/B registration and eleven horizontal grid lines give scales of about 64.75 and 64.78 screenshot pixels per original pixel.

The orange region maps to roughly x515.8-518.7 and y292.8-295.2, centered near (517.2,294.0). It identifies the soft lower-left turn of the frontal plate above the broader shaded rim. The accepted point is the integer coordinate (517,294), with approximately 2 pixel visual ambiguity. The change is modestly leftward and mainly upward from the previous point. It does not place the point at the projected A location or force a landmark to share the box's xmin.

Keep the existing box: the own molded lip and shaded supporting rim remain included even though the observed turn lies above the bottom extent. Visibility remains 2 and the right point is unchanged.

## Inherited pending landmark

`new_000010:24` is unchanged in full. Its left point is unavailable behind the jaw; its observed right point remains (909,491). Neither new attachment identifies this case. Preserve both triplets and the R36 box (810,464,913,495) until the requested physical point is clarified. The ledger distinguishes this inherited item from the two newly supplied and resolved requests.

## Scope and next comparison

Exactly two training rows change in two photographs: one box and one left-point coordinate pair. No class, visibility, instance, row identity, or unrelated field changes. All 20 development label files remain byte-identical to R36, including the 55 retained `new_000009` rows and the earlier exclusions. The review still contains 137 training images / 3,786 fronts and 20 development images / 659 fronts, with 583 development objects having two locatable corners. Inherited training exclusions and the held `new_000015` photograph are preserved.

Compare eligible R36 versus eligible R37 training labels under the same native recipe and budget. The treatment now includes an observed-corner correction as well as a box correction. The R37 development reference is byte-identical to R36, so there is no reference-change explanation for a new evaluation result. Preserve PR121 results against R36. This focused review does not certify the full dataset or claim a model improvement.

No blanket tiny-label deletion is introduced. A small box can describe a real clipped or occluded front; size alone does not resolve ownership. Advice about size and duplicate outputs must remain separate from label removal and from duplicate-annotation claims.
