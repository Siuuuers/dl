# Visible-front adjudication reference, 2026-11-12

This is the latest complete, assistant-reviewed **20-image development reference** on this working branch. It inherits the six coordinate refinements in `visible-front-corners-2026-11-12` and makes two further endpoint amendments in `new_000010` under the existing lower-front annotation rule.

| One-based row | Slot | Parent reference | This reference | Reason |
| --- | --- | --- | --- | --- |
| 28 | Left | `(918.92,493.90,v2)` | **`(923,492,v2)`** | Move the coordinate onto the exposed wax turn; the saved parent point was below-left of that contour. |
| 32 | Left | `(453.21,660.70,v2)` | **`0 0 0`** | The old coordinate is unsupported. A replacement beside the hand is not reliably locatable; the reviewers' disagreement is preserved. |

Coordinates above use the original **1280×720** image. The integer representative for row 28 was recorded by a fresh image-only reviewer before label comparison. Eight-decimal normalized serialization does not imply subpixel physical precision. Row 32's missing state records insufficient support for a reliable coordinate, not proof that the physical point is hidden or absent. Its object and box are retained, and its right slot was already missing.

The complete set contains **663 objects, 1,227 locatable endpoints, 99 missing slots and 3 empty images**. Objects with zero, one and two locatable corners number **20, 59 and 584**, respectively. Exactly two endpoint slots changed from the immediate parent. Classes, boxes, instance identities, row order, right endpoints and the other 19 label files are unchanged. Original baomai labels and every earlier reference remain intact.

The review targeted six previously uncertain left points; it did not freshly certify all 1,227 coordinates. Rows 25, 26, 27 and 31 remain supported at their existing placements. Other inherited box/identity uncertainties remain documented. This is a reused development reference, not an independent test set or a training-label overlay.

- [Manifest and inherited provenance](manifest.json)
- [Revised new_000010 labels](labels/val/new_000010.txt)
- [Original-image adjudication report](../../../docs/corner-adjudication-2026-11-12.md)
- [Frozen decisions and reviewer dissent](../../../results/review/corner-adjudication-20261112/review.json)
- [Saved-prediction replay](../../../results/review/corner-adjudication-20261112/replay.json)
- [Immediate parent reference](../visible-front-corners-2026-11-12/README.md)

For editor review, pair the unchanged originals under `data/images/val` with the corresponding files in this directory's `labels/val`. For a new model comparison, name this reference explicitly and keep the old reference available for reconciliation. Do not silently substitute it into historical result files. The replay shows that **PR35 remains selected**; changing reference labels did not train or improve any model.

Manifest SHA-256: `75e8689f65f7f32ca7f4657f27d214aa1ba371347812de780c6e5e6ac21af61e`.
