# Native augmentation and label-survival audit

This CPU diagnostic runs the existing augmentation. It does not train or score a model, change annotations, or replay historical training draws.

Three fixed passes contain 426 image presentations and 11997 source-object presentations.
Native filtering retains 11880 and removes 117; 368 retained boxes are clipped.

Of the removed presentations, 108 have zero clipped box area. The remaining 9 retain positive box area, with a largest short side of 4.047 pixels. These are box measurements, not physical object-visibility measurements.

| Source known corners | Presentations | Retained | Removed | Box clipped | Known corners outside canvas |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 411 | 397 | 14 | 26 | 0 |
| 1 | 1395 | 1370 | 25 | 83 | 39 |
| 2 | 10191 | 10113 | 78 | 376 | 279 |

## Interpretation

Native transformed-label arithmetic and row lineage are verified. A removed row or changed visible-point count is not by itself an error. Box area ratios do not measure physical visible material. Human image review is required to judge recognizability after cropping; this audit cannot prove which augmentation would improve a detector.

Source `v=0` means an unavailable annotation, not a negative physical existence label. A newly cropped-out known corner becomes unavailable in the transformed sample. Native internal coordinates for unavailable slots need not remain `0 0`; those coordinates are masked by the visibility flag. Canonical source labels remain unchanged.

The native box filter uses width and height > 2 pixels, area ratio > 0.10 and aspect ratio < 100. Reasons may overlap. The area ratio compares the clipped transformed axis-aligned box to the scaled source box; it is not an occlusion fraction.

## Review panels

Red marks source rows removed by the native filter; orange marks retained rows affected by clipping or lost known corners. These colors flag review cases, not annotation verdicts. Row numbers refer to the frozen training label file. Cyan/magenta dots show known left/right corner slots; unavailable points are not drawn. Panels are high-quality JPEG display copies; all measurement uses verified original images and numerical target geometry.

- [frame_0178/pass01](panels/01-frame_0178-pass01.jpg) — removed.
- [new_000002/pass01](panels/02-new_000002-pass01.jpg) — clipped_or_visibility.
- [new_000003/pass01](panels/03-new_000003-pass01.jpg) — clipped_or_visibility.
- [new_000004/pass01](panels/04-new_000004-pass01.jpg) — removed.
- [new_000005/pass01](panels/05-new_000005-pass01.jpg) — clipped_or_visibility.
- [new_000012/pass01](panels/06-new_000012-pass01.jpg) — removed.
