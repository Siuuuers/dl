# Round 10: visible fronts and front-face corners

This review continues the saved round 9 reconstruction at commit `848242c6aedc0a521daf0471e1eda514431f5684`. The user requested a fresh pass over **all 126 `new_*` originals**, a complete illustrated PDF, and a new controlled-training notebook. All 31 `old_*` review label files are preserved byte for byte at the user's direction.

The upstream [baomai README](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/README.md) identifies the two actual lower-front corners, ordered by image X. The [source and policy audit](annotation-policy.md) separates that upstream landmark contract from this project's visible-front box convention. The separately linked `yolo-pose` branch uses a different five-point task; its whole-object box and corner rules do not replace the baomai contract here.

## Box and landmarks answer different questions

The box encloses the identifiable **visible cassette front and its own frontal rim**. Locate a supported leftmost, highest, rightmost and lowest point of that material. Their enclosing axis-aligned rectangle is the box. Exclude the rack, rear perforated plane, hand, gripper and neighboring cassette. A connected shaded rim still belongs to the target; brightness alone is not a material classifier.

The landmarks identify the **actual lower corners of the front face**. A visible side face, flange or curved rim can extend beyond the landmark. Consequently, a correct point need not be the silhouette extreme or an axis-aligned box corner. Do not snap points to the box or apply a fixed inward offset. Inspect each local front/side junction in the original PNG. Keep `v=2` only for a reliably locatable junction; an obscured or unlocatable corner is canonical `0 0 0`, independently for each side. A valid missing point does not by itself make an instance unresolved.

Use the rack's continuation through exposed gaps, its connections to the mesh, occlusion order and local perspective to decide ownership. A dark line can also be a shadow or shaded cassette rim. Use all of those cues together.

## Occlusion and instance identity

Bound the visible owned material. If parts of one front remain visible on both sides of a gripper, its box necessarily encloses the gap between them. If only a small fragment is visible, do not extend its box to an inferred hidden corner. Keep separate fronts separate when an actual separator is visible; neighboring stack spacing alone does not prove another instance.

The user explicitly removed round 9 rows **7 and 14 in `new_000007`**. This is a photo-specific decision for two extremely thin frame slivers, not a new rule that deletes all clipped targets. All other edits are bound to exact round 9 before-lines. `row-maps.json` records deletions, additions and every surviving object's new row. Cases retain round 9 `base_rows` and separately list current rows.

## How to use the marked examples

The user said that amber cases before `new_000044` are generally acceptable and that original cyan labels from `new_000044` onward are generally correct. **Original cyan means the baomai labels in `data/labels`, not every cyan field in the latest review and not the separate PR35 control overlay.** Compare physical objects across versions before restoring fields; row order and counts can differ. Preserve supported additions, duplicate removals and boundary corrections. Record a concrete pixel-based reason for exceptions to the broad preference.

The uploaded blue rectangles and red/orange points communicate intent. Their thickness, resizing and compression do not establish exact camera coordinates. Original 1280 x 720 PNGs remain the coordinate authority. Coordinates are nominal pixel estimates under blur; normalized decimals preserve serialization rather than claim subpixel physical accuracy. Zoom and contrast can aid inspection without changing training images or recovering hidden geometry.

One marked screenshot, `04-427838.jpg`, visually matches `new_000013`, despite that identifier being absent from the user's abbreviated list. The review follows the image content.

## Deliverables and training scope

The PDF contains every new photo, including unchanged accepted photos and true empty scenes. Magenta marks only changed round 10 fields; cyan denotes inherited fields; amber denotes a remaining unresolved field. Additions and deletions have explicit notes. The complete ledger preserves before/after tokens, source hashes, original-label mappings where used, and review evidence. A completed visual pass is not a claim of infallible physical ground truth.

Training uses the accepted round 10 training labels. Any unresolved training photograph retains exact PR35 labels in both arms. The 31 user-confirmed old photographs are **excluded from this round's treatment**, separately from unresolved holds, and also use exact PR35 labels in both arms. Original PNGs, original training-validation labels, negative photos, split membership, and the frozen 666-front primary and 663-front diagnostic scoring references stay unchanged. Development-review corrections are separately versioned exploratory labels.

Use the established epoch-173 initialization independently for both arms, seed 46, 30 epochs, batch 4, image size 1280, FP32, AdamW with learning rate 0.0001, and final-epoch checkpoint selection. PR35 remains selected until actual new results justify a separate model decision. Preparing this notebook does not execute GPU training or promote a model.
