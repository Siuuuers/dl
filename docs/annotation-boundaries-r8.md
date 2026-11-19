# Cassette boundaries: the round 8 review rule

Date: **2026-11-19**. This note records the user's five new marked examples and the original-image review after PR #74. It supplements the preserved [annotation policy](annotation-policy.md); it does not rewrite earlier annotation versions or their pinned hashes.

## The target and its four extremes

The target remains the **identifiable visible cassette front, including its own frontal rim**. The rear perforated upper plane, rack, hand, gripper and neighboring cassette are outside that target. Gray or shadowed material can still belong to the front. Brightness alone does not establish ownership.

Find four supported points on the visible target: a leftmost, highest, rightmost and lowest point. The axis-aligned box is their enclosing rectangle:

```text
x_min = left witness.x       y_min = top witness.y
x_max = right witness.x      y_max = bottom witness.y
```

The witnesses need not be the four corners of the object, nor four different pixels. Image Y increases downwards, so the highest point has the smallest Y. A sloping or rounded face can require background wedges inside this rectangle. Those wedges do not justify moving an outer boundary to a rack rail, or cutting off target material to remove every background pixel.

Every changed box in round 8 records these four original-pixel witnesses in `operations.json`. They are review evidence only: **the training row still contains one box and two lower-front landmarks**. Nine decimal places in normalized text preserve serialization; they do not imply subpixel physical accuracy. Blurred transitions have nominal pixel-scale uncertainty.

## How to distinguish a rack rail from a cassette rim

Use the full scene as well as a crop. A rack member can often be followed through exposed gaps, past several cassettes and into the surrounding mesh. It has structural continuity with that mesh. A cassette's rim follows its own front, turns with that cassette, and terminates at its side. A continuous dark band is not automatically rack: cast shadow and a shaded frontal rim also look dark.

For each suspected boundary, check the following evidence together:

1. **Connection:** does the strip join the rack mesh or return around this cassette?
2. **Termination:** does it continue beyond this object, or end at this object's side?
3. **Occlusion:** which surface ends against the other? An occluder boundary is not a hidden physical corner.
4. **Perspective:** does the proposed edge follow the tilted front and its own rim?

Neighboring cassettes help interpret the physical structure. Their spacing does not supply coordinates for an unseen corner or an entirely hidden cassette. The marked blue lines are useful structural hints, but their hand-drawn thickness and screenshot scale do not specify exact camera-pixel boundaries.

## Where top and bottom belong

**Top:** the highest visible point of the included front/rim. This can lie above the bright main face, especially on a sloping upper lip. A broad rear perforated plane remains outside the front task. Do not raise the box to the rack or lower it merely to the brightest stripe.

**Bottom:** the lowest visible point belonging to that same front/rim. Follow the transition into the support or gap below. Do not include the continuous base rail, the next cassette, or a cast shadow on another surface. If a projection's physical ownership remains ambiguous, keep that uncertainty in the case ledger.

Bounds are chosen per cassette. Round 8 includes both expansions to recover an omitted own rim and contractions to remove rack margins. There is no shared pixel inset, fixed percentage shrink or stack-spacing replacement rule.

## The two landmarks are separate physical decisions

Keep the baomai task's actual lower-front corner pair, ordered by image X. A landmark is a physical turn of the front contour; it is not automatically an axis-aligned box corner. Identify the turn in the original pixels, then decide its coordinates and availability independently.

If a point falls outside a proposed box, reopen both judgments. Correct an unsupported point on a rack rail instead of padding the box to retain it. Conversely, retain a supported point and recover target material when the proposed box is too small. Never snap every point to the rectangle.

Keep `v=2` for a reliably locatable visible junction and canonical `0 0 0` for an unlocatable one. Retain the other endpoint independently. An identifiable front can remain with one or both points missing. The [pinned upstream baomai README](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/README.md) supplies the one-class/two-point task and missing-point convention; the box ownership details above are the project's explicit review convention.

## What remains unresolved

In the older upward-looking images, the broad pale lower flange is connected to the cassette. The remaining question concerns the intended landmark on that shape: the outer flange turn or the front-wall junction. This is not evidence that the flange is rack. Those inherited landmark/extent cases remain disclosed, with exact PR35 supervision on every held training photograph in both arms.

The user-confirmed original `old_000013` label file is preserved byte for byte. A new visual pass does not silently clear its separate landmark-definition hold, nor the other explicit blur, clipping or occlusion holds. Original empty-rack photographs remain genuine negatives; an uncertain photograph is never converted into a negative.

## Inspection and experiment boundaries

Original 1280×720 PNGs are the coordinate authority. Zoom and optional grayscale/contrast views expose existing transitions; they cannot recover hidden geometry or establish material ownership. Training uses the unchanged original PNGs.

The complete 157-photo atlas displays current review labels. Magenta highlights only field changes from round 7; cyan denotes an unchanged field; amber identifies an unresolved field. The row map and exact before/after token operations preserve identity. Per-image coverage distinguishes original-scene inspection and detailed crops from physical certification.

The separately composed experiment uses eligible round 8 training labels and exact PR35 labels on every held training photograph in both arms. Original training validation and the frozen 666-target and 663-target scoring references remain unchanged. Development-review corrections are exploratory and independently versioned. PR #74's outcome does not establish an improvement; PR35 remains selected pending a separate model decision.
