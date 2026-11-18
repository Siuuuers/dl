# Where the box and two points belong

This guide explains the existing [visible-front annotation policy](annotation-policy.md). It does not change the task or create a new landmark definition. The round 5 examples and exact edits are in the [review record](front-review-r5.md).

## Identify the surface before placing a mark

One row means one identifiable **cassette front face, including that front's own visible rim**. It does not mean the cassette's entire three-dimensional silhouette. The upper surface, projecting tray, back, neighboring cassette, metal rack, gripper and hand do not define the target's outer limits.

First follow the same front through the original photograph and its immediate neighbors in the stack. Decide which material belongs to that instance. Then place its box and judge each corner separately. Adjacent frames can help explain the mechanism, but cannot supply a hidden target coordinate in this frame.

## The box encloses visible target material

Use the smallest axis-aligned rectangle containing the identifiable visible material of that front. Its own rim counts. A clipped face ends at the image boundary; a partly hidden face ends at its observed extent, without estimating its hidden outline.

An axis-aligned rectangle cannot trace a sloping or curved outline exactly. Some background can therefore lie inside a correctly fitted box. If two visible patches belong to one front, the box can also span a gripper or rack between them. Those interior pixels do not themselves make the box incorrect: the box's outer extents must come from the target, and the patches must belong to the same instance.

Do not split one cassette into two labels solely to remove an occluder from its rectangle. A pixel mask could describe the visible surface more precisely, but that would introduce a different output and annotation task. The present box-and-two-corner task is retained.

## The points mark physical lower-front corners

The left point marks the actual junction of this front's lower edge and its image-left side. The right point marks the corresponding image-right junction. They follow image X order, independently of which physical end of the cassette is closest to the camera.

These points are not automatically the rectangle's lower-left and lower-right corners. Perspective, sloped sides and a rounded contour can place a physical junction elsewhere within the axis-aligned envelope. They are also not an inset groove, a bright stripe, an upper projecting ledge, or the edge of an occluding gripper.

That geometric distinction does not excuse a misplaced annotation. Round 5 found real examples where a point sat inside the front or on the wrong part of its contour. Those coordinates are corrected from the source pixels. Where a box also cuts off visible front material, it receives its own separate correction.

| Evidence in the original frame | Label decision |
| --- | --- |
| A lower-front corner is visible and its junction can be located | Use the observed coordinate with visibility 2. |
| The front is identifiable but one corner is covered, clipped or materially ambiguous | Keep the front; encode that side as `0 0 0`. |
| Both corners are unavailable but front material is identifiable | Keep the front box; both slots are `0 0 0`. |
| An entirely hidden cassette is suggested only by spacing or another frame | Do not create a target in this frame. |
| Visible material cannot be assigned confidently to a front or its neighbor | Record the uncertainty; do not resolve it through a numeric snap rule. |

Mild blur or rounded molding does not automatically make a point missing. The question is whether the physical junction is locatable in the image. Coordinates on soft contours are nominal estimates, not proof of subpixel physical accuracy.

## Use filters as a viewing aid

The review uses the original crop beside an optional grayscale, zero-cutoff autocontrast view. This remaps intensities without moving their coordinates. It can make a faint existing transition easier to inspect; it cannot reveal a surface covered by a gripper or recover a unique corner from missing detail.

Pillow documents autocontrast as a histogram-based mapping of the remaining intensity range to black and white. At cutoff zero this pass does not deliberately discard a percentage of bright or dark pixels. See [Pillow ImageOps documentation](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html).

The source PNG is always the final evidence. This pass uses no generative restoration, inpainting or super-resolution to manufacture annotation detail. Filtered views are not substituted for the original training or inference images. Adding a permanent preprocessing filter would be a separate experiment because it changes the model's input.

## Keep physical review separate from automatic checks

Physical review decides instance identity, edge ownership and corner availability. A second reviewer can inspect the same original context and disagree with a proposed coordinate. Software then checks the mechanical conditions: valid format, original image identity, bounds, nonempty boxes, keypoint ordering, canonical missing slots, exact operation replay and stable row mappings.

Passing those checks proves that the saved labels match the recorded decisions. It cannot prove that a chosen edge belongs to the cassette. In particular, never move a point onto a box edge just to satisfy a containment check, or enlarge a box just to contain a point that was placed on the wrong object.

Requests for a round use the previous round's immutable row numbers. Deletions and additions are applied after all edits are keyed to those base identities. The focused PDF shows current row numbers, with explicit before-to-current mappings for count changes. Magenta identifies a row changed in that pass; the operation ledger identifies exactly which fields changed.

The saved pose format is one object per row: class, normalized box center and size, then two normalized coordinate/visibility triplets. This follows the [Ultralytics pose format](https://docs.ultralytics.com/datasets/pose/); the physical meaning of the two landmarks comes from this project's annotation policy, not from the generic file format.
