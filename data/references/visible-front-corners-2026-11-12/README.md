# Visible-front corner reference, 2026-11-12

This complete 20-image development reference inherits `visible-front-2026-11-11` and refines six existing visible endpoint coordinates in `new_000010`. It follows the actual lower-front target in `baomai-visible-front-v1`.

| Row in new_000010 | Endpoint | Reviewed original-image coordinate |
| --- | --- | --- |
| 15 | Right | `(1008,138)` |
| 16 | Right | `(996,187)` |
| 22 | Right | `(926,428)` |
| 23 | Right | `(919,460)` |
| 24 | Right | `(911,491)` |
| 31 | Left | `(457,641)` |

The original images, boxes, visibility states, instance identities and row order are preserved. The other 19 label files are exact copies of the parent. Counts remain 663 objects, 1,228 visible/locatable endpoints, 98 missing slots and 584 objects with both endpoints locatable. All original and earlier reference files remain intact.

These are assistant-reviewed visual refinements with moderate pixel-placement certainty. They are not human certification, subpixel measurements, hidden-corner estimates, training labels, or an independent test set. Several material-boundary and occlusion questions remain unresolved; see the review record.

- [Complete manifest](manifest.json)
- [Revised new_000010 labels](labels/val/new_000010.txt)
- [Original-image report and open questions](../../../docs/visible-corner-review-2026-11-12.md)
- [Exact review decisions](../../../results/review/visible-corners-20261112/review.json)
- [All-three-model replay](../../../results/review/visible-corners-20261112/replay.json)

Use these labels explicitly when reviewing this revision with the existing editor or comparing models. The editor needs the unchanged `data/images/val` originals and the corresponding files under this directory's `labels/val`. Its row numbers remain compatible with the parent's original-to-reference mapping.

Historical reports and experimental protocols retain their earlier reference pins. A new score must identify this reference ID and manifest hash; do not compare one model on these labels with another on older labels. No change to runtime model selection or inference settings is required.

The source manifest still identifies the original `data/manifest.json`; `parent_reference` separately identifies the 2026-11-11 amendment. Full replay and verification instructions are in the linked report.
