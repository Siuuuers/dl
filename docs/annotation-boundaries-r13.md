# Round 13: front amendments and the complete nonempty new-photo atlas

Round 13 starts from the saved r12 review at `eb8284dcab71074275d9d22fc0fd974839d66863`. The new text requests changes in photos 005, 009, 011, 012, 026, 043, 053 and 056. Exact source rows, original-image hashes and request dispositions are recorded before replay. Every unrequested review file and every old-series review file retains its r12 bytes.

## Physical target

Keep the [r12 physical contract](annotation-boundaries-r12.md): one visible cassette front and its owned frontal rim per box, with the two actual lower front corners ordered by image X. Observe the enclosing box and the lower front/side junctions independently. A projecting rim can place a corner inside the box. The rack, hand and gripper do not establish a front edge. Hidden or unlocatable points remain exactly `0 0 0` independently on each side; locatable points use visibility 2.

Image X increases rightward and image Y downward. Feedback that names the physical upper or lower edge is translated to pixel `ymin` or `ymax` only when the original image supports that interpretation. Never invert or collapse a front to satisfy a word literally. Cross-row anchors use immutable r12 coordinates, preserving the target point's other coordinate and visibility. This prevents sequential changes from propagating to unrelated rows.

Perspective can make stacked edges taper. A reference in the same column does not justify a common X when that coordinate lies in the rack or inside a different visible rim. Every unsupported or direction-conflicting clause receives an explicit disposition. In photo 005 row 49, the explicit target is pixel Y528, an upward move from532.5 despite the written positive sign; the original image supports the target. Incomplete wording alone does not create a physical training hold.

## PDF scope and highlights

The PDF includes every current `new_*` photograph with at least one annotation. All `old_*` photographs and all empty-label photographs are excluded from the PDF. Every current object on each included photo is displayed. Magenta marks fields changed relative to r12; cyan marks retained fields; amber identifies existing unresolved physical cases. An unchanged photo remains included when it has labels. Original RGB pixels and current row identities are preserved.

The full annotation package still contains all157 label files, including empty scenes and old-series photos. Excluding a photo from the PDF does not remove it from the dataset or training.

## Experiment

Unresolved training photos and the separate31 old-series treatment exclusions retain exact PR35 labels in both arms. Original images, split membership, empty training scenes, original training-validation labels and frozen666/663 scoring references stay fixed. The r12 current review was pinned before PR82 returned; the later r13 development review is separately identified as an exploratory diagnostic.

Both fresh arms independently initialize from original epoch173 and use the established seed46,30-epoch,batch4,1280-input,FP32,AdamW0.0001 recipe with final-epoch selection. Preparing r13 runs no training and makes no promotion claim. PR82's mixed reference-dependent results must remain visible; PR35 remains selected under the existing evaluation rule.
