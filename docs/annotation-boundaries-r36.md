# R36: owned white rims and explicit landmark questions

Reviewed 26 November 2026, Asia/Hong_Kong, after PR119. This is a verified subset of five user-named rows across four original photographs. Four rows receive box amendments; one row is retained because the referenced marked crop is unavailable. Two of the amended rows also have unresolved landmark requests. All point coordinates, class tokens, visibility tokens and object identities remain unchanged.

## Physical interpretation

The target remains the observed cassette front, including its own molded lip, rim and shaded supporting material. Trace continuity of the frontal plate, changes of plane, stack order and occlusion junctions. Colour supports that interpretation; neither white nor dark pixels alone establish ownership. A bright rack can be outside the target and a shaded plastic rim can be inside it.

A bounding box describes the extrema of owned visible material. A lower-front corner is a separate physical landmark and need not share the box's xmin or xmax. A sloping front can have its leftmost material near the top while its lower-left turn is farther right. An axis-aligned rectangle around an occluded front can contain internal gripper or finger pixels; trimming every such pixel would cut real cassette material. Unavailable corners remain the original `0 0 0` triplets.

Stored image coordinates are x right and y down, on the original 1280 x 720 pixels. Physical top is stored `ymin`; physical bottom is stored `ymax`. The user's top/bottom wording is translated explicitly before editing.

## Applied subset

| Image | Row | R35 to R36 | Basis |
|---|---:|---|---|
| new_000034 | 34 | xmin 426 to 435 | The own left contour is approximately x434-436. Remove the rack strip while retaining the visible left turn at (436,545). |
| new_000034 | 35 | xmin 509 to 511; physical top 258 to 255 | Restore the requested R34 edges. Preserve right 637 and bottom 300. |
| new_000010 | 24 | xmax 910 to 913 | Include the owned white rim through x912 before the dark separator around913-914. |
| new_000057 | 1 | xmin 777.00012 to 782 | Tighten toward the left contour while keeping the observed point (782.43,20.62) inside. |

For new_000034:34, the request calls the nearby landmark the "right point". The relevant saved point is the **left** point at (436,545). Literal xmin 544.76 would reduce a 119-pixel front box to 0.24 pixels and exclude the left corner. R36 applies the observed physical contour and stated rack-exclusion intent, with this wording correction recorded. Neither point moves.

The approximate visual ambiguity is 2-2.5 original pixels, recorded by row. These ranges describe image-reading limits, not statistical intervals or calibrated physical tolerances. Rounded displayed values are explanatory; exact unchanged and changed tokens are bound in the ledger.

## Unresolved requests, preserved fields

1. **new_000006:53:** the referenced yellow xmax marker and red outline were not present in the latest message, PR119, or available recent uploaded images. Retain the complete R35 row, including xmax 503. The absent target is not reconstructed or claimed as applied.
2. **new_000034:35:** at the current corner height around y297, x511 lies on the dark rack/edge transition. The observed pale lower-front turn is approximately x516-519,y296-299, consistent with the saved (518.35,297.09). Restore the two box fields, but preserve the corner pending a marked intended physical feature. The box extremum and lower corner are different features.
3. **new_000010:24:** the saved left corner is unavailable behind the jaw; the only observed point is the right corner at (909,491). Correct the box independently. Preserve both point triplets until "left point" is mapped to a visible physical landmark. No hidden corner is invented.

`held_request_rows` records these pending fields separately from inherited whole-image training exclusions. Fully held rows remain byte-identical. A partially verified row may change only fields outside its pending list. Pending questions are not represented as resolved merely because the notebook can run.

## Why the white rim in new_000010:24 belongs

The broad pale frontal plate continues into the small pale strip at x911-912. Its rounded lower contour joins the same plate. The darker separator farther right continues down into the rack, with another front beyond that separator. Those connected shapes support extending the box to x913. The left jaw cuts across the target and hides its actual left turn; the jaw's edge is not a cassette landmark. This corrects the earlier x910 boundary without treating whiteness itself as a material detector.

## Controlled next comparison

The only eligible training differences are two box-only replacements in one photograph, new_000034 rows 34 and 35. Compare accepted eligible R35 labels against this verified R36 subset using the same native Pose26 recipe, seed 46, independently initialized epoch 173 models and fixed training budget. Development amendments do not enter training supervision or original trainer validation.

The complete review remains 137 training images / 3,786 fronts and 20 development images / 659 fronts, including 583 fronts with two locatable corners. The eligible training overlay and original trainer validation remain separate inventories. Preserve PR119 scoring against R35 before reporting any R36 reference-only rescore. Frozen 666 is primary, 663 historical, and PR35 remains selected unless a later explicit evaluation justifies a change. The two earlier new_000009 exclusions remain in place; no instance is added or removed in R36.
