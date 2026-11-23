# PR99: the sixteen remaining R24 partial-front misses

This completed diagnostic covers **all 16 unmatched targets in seven photographs** for the R24 candidate on the accepted **662-object R24 exploratory reference**. Seven have no locatable corner and nine have one. All 584 targets with two locatable corners are matched. The frozen 666-object primary comparison remains separate.

**None of these targets has a retained proposal reaching the fixed IoU 0.50 matching threshold.** Ten have an overlapping rectangle below the threshold; six have no overlapping retained rectangle. **No case is lost because a qualifying retained proposal was assigned to another target.** The saved outputs are already filtered at confidence 0.13 and NMS IoU 0.70. They cannot establish whether a missing output was confidence-gated, NMS-suppressed or absent from the decoded network geometry.

The accepted R24 labels, checkpoint, predictions, thresholds and scorer are unchanged. No new inference or training was run. Original full-scene context and bounded pixel crops were inspected for each of the seven images.

## Navigate the complete PDF

Use **`front-review-r24-all-nonempty-new.pdf`**, 116 pages; PDF SHA-256 `6ff2a34971900b497a58fd52e8be3b3ef2dc6b213cfc8205b1665b2dc134a0cb`. Page numbers below come directly from the verified atlas `photo_pages` mapping. Rows are the current, one-based R24 rows, not historical label positions.

| Photograph | PDF page | Missed R24 rows | Exact source |
|---|---:|---|---|
| `new_000008` | 10 | 33, 34, 36 | [Original](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000008.png) · [R24 labels](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt) |
| `new_000009` | 11 | 55, 56, 57 | [Original](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000009.png) · [R24 labels](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000009.txt) |
| `new_000010` | 12 | 14, 33, 34 | [Original](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000010.png) · [R24 labels](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000010.txt) |
| `new_000011` | 13 | 55, 56, 57, 60 | [Original](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000011.png) · [R24 labels](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000011.txt) |
| `new_000053` | 49 | 1 | [Original](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000053.png) · [R24 labels](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000053.txt) |
| `new_000086` | 78 | 57 | [Original](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000086.png) · [R24 labels](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000086.txt) |
| `new_000088` | 80 | 57 | [Original](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000088.png) · [R24 labels](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000088.txt) |

## Complete case ledger

Prediction indices are **zero-based** within the saved candidate array. IoUs and confidence scores are shown to six decimals; full precision, exact box coordinates, keypoints, source hashes and current-to-primary row lineage are in [the companion JSON](partial-front-failure-queue.json). “Geometry” below means an overlapping retained rectangle fails the fixed box criterion; it does not automatically mean that rectangle depicts the same physical front.

| R24 target | PDF page | Locatable corners | Maximum retained IoU | Best overlapping output: index / confidence | Diagnostic |
|---|---:|---:|---:|---|---|
| `new_000008:33` | 10 | 1 | 0.074066 | 11 / 0.922946 | Geometry below threshold |
| `new_000008:34` | 10 | 1 | 0.065351 | 29 / 0.764837 | Geometry below threshold |
| `new_000008:36` | 10 | 0 | 0.000000 | None | No retained overlap |
| `new_000009:55` | 11 | 0 | 0.000000 | None | No retained overlap |
| `new_000009:56` | 11 | 1 | 0.000000 | None | No retained overlap |
| `new_000009:57` | 11 | 0 | 0.000000 | None | No retained overlap |
| `new_000010:14` | 12 | 0 | 0.385671 | 35 / 0.165105 | Geometry below threshold |
| `new_000010:33` | 12 | 0 | 0.220174 | 31 / 0.247091 | Geometry below threshold |
| `new_000010:34` | 12 | 0 | 0.007614 | 31 / 0.247091 | Geometry below threshold |
| `new_000011:55` | 13 | 1 | 0.482170 | 55 / 0.786640 | Geometry below threshold |
| `new_000011:56` | 13 | 1 | 0.453501 | 62 / 0.261832 | Geometry below threshold |
| `new_000011:57` | 13 | 1 | 0.336395 | 60 / 0.337293 | Geometry below threshold |
| `new_000011:60` | 13 | 0 | 0.072347 | 67 / 0.179956 | Geometry below threshold |
| `new_000053:1` | 49 | 1 | 0.314293 | 21 / 0.137279 | Geometry below threshold |
| `new_000086:57` | 78 | 1 | 0.000000 | None | No retained overlap |
| `new_000088:57` | 80 | 1 | 0.000000 | None | No retained overlap |

### A useful distinction before another training run

The strongest concrete geometry cases are **`011:55`, `011:56` and `011:57`**. Their best retained boxes extend beyond the accepted visible fragments into the occluded region. Although those boxes fail matching, their locatable left endpoints lie approximately **1.96, 3.36 and 1.43 pixels** from the accepted points. These are diagnostic distances on unmatched alternatives, not official corner successes. They identify a visible-box supervision question; another corner-adjustment run would not directly address it.

Three current misses—**`008:33`, `008:34` and `009:55`**—are already matched by both PR35 and the R24 control on this exact same reference. Both R23 and R24 candidates miss them. Preserve these three cases as coverage checks for any later candidate. All sixteen current misses are also misses for the freshly evaluated R23 candidate.

## Per-case observations and next evidence

Boxes below are `(xmin, ymin, xmax, ymax)` in original 1280×720 pixel coordinates. They are file values, not a claim of subpixel annotation certainty. A nearest rectangle listed for a zero-overlap case is spatial context only; it is not a recovered or associated detection.

### new_000008:33 — PDF page 10, R24 row 33

[Original photograph](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000008.png) · [Exact accepted label row](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt#L33). Accepted box **(1211.84, 409.44, 1280.00, 456.00)**; 1 locatable corner(s).

Rightmost stack, upper of the two missed right-clipped fronts. The maximum-overlap R24 output belongs to neighboring row 30; the overlap is incidental, not a qualifying target box.

Best overlapping retained output **11**: IoU **0.074066**, confidence **0.922946**, box **(1059.95, 407.18, 1229.48, 458.71)**; assigned to R24 row 30. None of the retained outputs qualifies for this target.

**Next evidence:** Compare the already saved PR35/control target boxes with R23/R24 at this exact front. If mechanism diagnosis is pursued later, bind a raw R24 proposal trace to this row before deciding whether a score change or geometry supervision is relevant. Do not repeat the historical PR35 confidence sweep.

### new_000008:34 — PDF page 10, R24 row 34

[Original photograph](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000008.png) · [Exact accepted label row](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt#L34). Accepted box **(1205.00, 449.00, 1280.00, 492.00)**; 1 locatable corner(s).

Rightmost stack, middle missed clipped front. The maximum-overlap output is assigned to the next lower front, row 35; it does not cover this target.

Best overlapping retained output **29**: IoU **0.065351**, confidence **0.764837**, box **(1201.99, 486.73, 1280.00, 527.92)**; assigned to R24 row 35. None of the retained outputs qualifies for this target.

**Next evidence:** Audit existing eligible training examples of consecutive right-clipped fronts for consistent instance completeness. Compare with the saved PR35/control match; keep image008 development-only and do not transfer its labels into training.

### new_000008:36 — PDF page 10, R24 row 36

[Original photograph](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000008.png) · [Exact accepted label row](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000008.txt#L36). Accepted box **(996.00, 702.00, 1078.00, 720.00)**; 0 locatable corner(s).

A small face segment at the bottom image boundary near x996–1078; both lower corners are outside the locatable portion. No retained prediction overlaps its box.

Maximum retained IoU is **0**; there is no overlapping output and therefore no meaningful best associated proposal.
Nearest rectangle for orientation only: output **22**, confidence **0.909957**, box **(1040.24, 476.01, 1197.63, 523.02)**, nearest-edge gap **178.98 px**. It has zero IoU with the target.

**Next evidence:** Look for clearly supported bottom-boundary front examples in existing eligible material and record their capture family and complete-image coverage. If missing-output mechanism becomes necessary, obtain a separate raw R24 trace; current outputs do not distinguish scoring from suppression or absent geometry.

### new_000009:55 — PDF page 11, R24 row 55

[Original photograph](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000009.png) · [Exact accepted label row](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000009.txt#L55). Accepted box **(682.00, 471.00, 720.00, 494.00)**; 0 locatable corner(s).

The front fragment visible in the gap between the gripper jaws, above the smaller row 57 fragment. Both corners remain unavailable. PR35/control match this accepted identity, but R23/R24 have no overlapping retained output.

Maximum retained IoU is **0**; there is no overlapping output and therefore no meaningful best associated proposal.
Nearest rectangle for orientation only: output **57**, confidence **0.541150**, box **(599.66, 433.41, 672.21, 459.35)**, nearest-edge gap **15.21 px**. It has zero IoU with the target.

**Next evidence:** Compare the existing PR35/control proposal and R24 absence for the same stable identity, then inspect supervision of fronts exposed inside the jaw gap. Preserve both missing-corner slots; a future mechanism test must use raw candidate proposals rather than assume a confidence cause.

### new_000009:56 — PDF page 11, R24 row 56

[Original photograph](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000009.png) · [Exact accepted label row](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000009.txt#L56). Accepted box **(810.00, 490.00, 828.00, 521.00)**; 1 locatable corner(s).

Narrow exposed front at the right side of the right jaw; only the accepted right corner is locatable. No retained R24 output overlaps it.

Maximum retained IoU is **0**; there is no overlapping output and therefore no meaningful best associated proposal.
Nearest rectangle for orientation only: output **39**, confidence **0.903639**, box **(599.01, 411.66, 677.83, 441.05)**, nearest-edge gap **140.94 px**. It has zero IoU with the target.

**Next evidence:** Find eligible examples of narrow jaw-side fronts with one locatable corner; review each complete image and the distinction from bright gripper/rack surfaces. Keep the hidden left corner missing. Raw proposal evidence would be needed before a filtering hypothesis.

### new_000009:57 — PDF page 11, R24 row 57

[Original photograph](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000009.png) · [Exact accepted label row](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000009.txt#L57). Accepted box **(702.00, 493.00, 717.00, 509.00)**; 0 locatable corner(s).

Tiny front fragment lower in the gap between the jaws; neither corner is locatable. No retained output overlaps the accepted box.

Maximum retained IoU is **0**; there is no overlapping output and therefore no meaningful best associated proposal.
Nearest rectangle for orientation only: output **68**, confidence **0.241717**, box **(597.99, 451.94, 662.46, 477.61)**, nearest-edge gap **42.43 px**. It has zero IoU with the target.

**Next evidence:** Seek a clearer existing view of this fragment type to support supervision and instance completeness, with capture relationships recorded. Relate any future raw R24 trace by image and geometry to the archived PR35 trace; historical row numbers alone are not a stable identity across reference versions.

### new_000010:14 — PDF page 12, R24 row 14

[Original photograph](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000010.png) · [Exact accepted label row](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000010.txt#L14). Accepted box **(668.00, 463.00, 747.00, 496.00)**; 0 locatable corner(s).

Front material is exposed in the upper central jaw gap, with both corners occluded. The best-overlap prediction spans much farther left across the jaw/occlusion region than the accepted visible extent.

Best overlapping retained output **35**: IoU **0.385671**, confidence **0.165105**, box **(597.00, 463.25, 737.83, 503.08)**; unmatched output. None of the retained outputs qualifies for this target.

**Next evidence:** Prioritize existing training examples that teach the visible extent of a front through the jaw gap, with full-image coverage. Contrast the accepted target box with this saved over-wide proposal; no corner relocation or hidden-corner invention is indicated.

### new_000010:33 — PDF page 12, R24 row 33

[Original photograph](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000010.png) · [Exact accepted label row](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000010.txt#L33). Accepted box **(695.00, 497.00, 727.00, 533.00)**; 0 locatable corner(s).

A narrow front strip is visible lower between the jaws. The best retained proposal covers its vertical band but extends approximately 96 pixels left of the accepted visible box.

Best overlapping retained output **31**: IoU **0.220174**, confidence **0.247091**, box **(599.15, 494.95, 728.51, 535.40)**; unmatched output. None of the retained outputs qualifies for this target.

**Next evidence:** Review comparable eligible jaw-gap fronts for consistent visible-box extent and separation of adjacent layers. Preserve missing endpoints; use these retained over-wide boxes as a diagnostic question, not annotation truth.

### new_000010:34 — PDF page 12, R24 row 34

[Original photograph](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000010.png) · [Exact accepted label row](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000010.txt#L34). Accepted box **(696.00, 534.00, 729.00, 558.00)**; 0 locatable corner(s).

Lower front strip at the bottom of the jaw gap, just above the hand. Its only overlapping output is the over-wide proposal associated spatially with the layer above, with IoU only 0.0076.

Best overlapping retained output **31**: IoU **0.007614**, confidence **0.247091**, box **(599.15, 494.95, 728.51, 535.40)**; unmatched output. None of the retained outputs qualifies for this target.

**Next evidence:** Inspect eligible examples of the last visible layer above a hand or gripper for omission and layer separation. The incidental overlap provides no usable target box; a raw proposal trace would be needed to classify any filtering cause.

### new_000011:55 — PDF page 13, R24 row 55

[Original photograph](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000011.png) · [Exact accepted label row](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000011.txt#L55). Accepted box **(604.00, 435.00, 635.00, 453.00)**; 1 locatable corner(s).

Upper of three exposed strips on the left side of the jaw. A strong retained proposal covers the front but extends right into the occluded region; its available left corner is close to the accepted point.

Best overlapping retained output **55**: IoU **0.482170**, confidence **0.786640**, box **(602.77, 435.12, 654.46, 457.29)**; unmatched output. None of the retained outputs qualifies for this target.

**Next evidence:** Use this case to define a visible-box-extent supervision audit on comparable training originals: inspect the right edge at the jaw boundary and neighboring-layer separation. Keep the accepted corner; its diagnostic distance is about 1.96pixels even though the box fails IoU.

### new_000011:56 — PDF page 13, R24 row 56

[Original photograph](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000011.png) · [Exact accepted label row](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000011.txt#L56). Accepted box **(606.00, 452.00, 626.00, 470.00)**; 1 locatable corner(s).

Middle exposed strip beside the left jaw. The best-overlap output extends right beyond the accepted visible width and slightly lower; its left endpoint remains close to the accepted point.

Best overlapping retained output **62**: IoU **0.453501**, confidence **0.261832**, box **(601.26, 452.96, 637.19, 473.34)**; unmatched output. None of the retained outputs qualifies for this target.

**Next evidence:** Review the same bounded visible-box-extent question on eligible training images, preserving the unavailable right corner. The existing left coordinate is about 3.36pixels from the reference, so another corner-tuning run does not address the measured association failure.

### new_000011:57 — PDF page 13, R24 row 57

[Original photograph](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000011.png) · [Exact accepted label row](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000011.txt#L57). Accepted box **(606.50, 468.00, 622.00, 480.00)**; 1 locatable corner(s).

Lowest exposed strip beside the left jaw. The retained output is wider and taller than the accepted visible fragment, while its available left endpoint remains close.

Best overlapping retained output **60**: IoU **0.336395**, confidence **0.337293**, box **(601.29, 467.95, 632.02, 485.94)**; unmatched output. None of the retained outputs qualifies for this target.

**Next evidence:** Review width/height targets for tapered partial fronts near a jaw, along with complete-image coverage. The retained left point is about 1.43pixels from the accepted point; evaluate geometry supervision before any new corner intervention.

### new_000011:60 — PDF page 13, R24 row 60

[Original photograph](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000011.png) · [Exact accepted label row](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000011.txt#L60). Accepted box **(686.00, 463.00, 712.00, 475.00)**; 0 locatable corner(s).

A small lower segment in the central jaw opening. The maximum-overlap retained output lies mostly above this target; its overlap does not establish a separately recovered lower fragment.

Best overlapping retained output **67**: IoU **0.072347**, confidence **0.179956**, box **(678.94, 437.25, 721.86, 467.14)**; unmatched output. None of the retained outputs qualifies for this target.

**Next evidence:** Look for eligible jaw-gap examples with two separately visible layers and distinguish their instance extents from the foreground jaw. The accepted target stays unchanged; do not treat the upper neighboring proposal as a recovered lower front.

### new_000053:1 — PDF page 49, R24 row 1

[Original photograph](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000053.png) · [Exact accepted label row](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000053.txt#L1). Accepted box **(639.00, -0.00, 710.00, 10.00)**; 1 locatable corner(s).

Very thin front segment clipped by the upper image boundary. The overlapping proposal is only about 4 pixels tall and spans wider than the accepted 10-pixel-tall fragment.

Best overlapping retained output **21**: IoU **0.314293**, confidence **0.137279**, box **(636.67, 0.00, 757.79, 4.04)**; unmatched output. None of the retained outputs qualifies for this target.

**Next evidence:** Review eligible top-border examples for visible extent and one-corner labels. A retained box already exists, so lowering confidence alone cannot repair its geometry. Keep the original image resolution and the prior negative 1920 result in view.

### new_000086:57 — PDF page 78, R24 row 57

[Original photograph](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000086.png) · [Exact accepted label row](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000086.txt#L57). Accepted box **(503.00, 596.00, 511.00, 619.00)**; 1 locatable corner(s).

An approximately 8-pixel-wide exposed strip beside the thumb at the lower rack; only the left corner is accepted as locatable. No retained output overlaps its box.

Maximum retained IoU is **0**; there is no overlapping output and therefore no meaningful best associated proposal.
Nearest rectangle for orientation only: output **51**, confidence **0.891306**, box **(431.82, 589.07, 498.48, 620.42)**, nearest-edge gap **4.52 px**. It has zero IoU with the target.

**Next evidence:** Review eligible narrow hand-adjacent fronts and distinguish material from finger/rack highlights. Treat 086 and 088 as a possible capture family until provenance is checked; they are not two demonstrated independent samples.

### new_000088:57 — PDF page 80, R24 row 57

[Original photograph](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/images/val/new_000088.png) · [Exact accepted label row](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/data/reviews/fronts-2026-11-23-r24/labels/val/new_000088.txt#L57). Accepted box **(505.00, 592.00, 514.00, 616.00)**; 1 locatable corner(s).

An approximately 9-pixel-wide front strip next to the thumb in a visually similar rack/hand arrangement to 086. No retained output overlaps the accepted box.

Maximum retained IoU is **0**; there is no overlapping output and therefore no meaningful best associated proposal.
Nearest rectangle for orientation only: output **51**, confidence **0.860107**, box **(434.51, 581.61, 502.10, 615.70)**, nearest-edge gap **2.90 px**. It has zero IoU with the target.

**Next evidence:** Pair with 086 for a provenance and supervision audit of this sliver pattern, while preserving separate recorded image identities. Avoid counting adjacent-looking frames as independent evidence; do not extrapolate the hidden right corner.

## Bound the next experiment with the existing history

- [postprocess-experiment-2026-11-07.md](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/docs/postprocess-experiment-2026-11-07.md): Epoch173 already tested NMS .70/.60/.68/.50 with confidence .15/.25/.40. At .25, .60 and .50 suppressed valid partial-front matches. The .68 improvement reduced other extras, not duplicate count. It is not evidence for stronger NMS on R24.
- [proposal-filter-review-2026-11-11.md](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/docs/proposal-filter-review-2026-11-11.md): PR35 pre-NMS trace already distinguished a confidence-gated clipped-front proposal from a tiny fragment without qualifying decoded geometry. Confidence .13 was accepted for reviewed proposals. These older model-specific mechanisms must not be imputed to R24 or blindly repeated.
- [partial-front-results-2026-11-11.md](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/docs/partial-front-results-2026-11-11.md): PR40 five-image partial-front label extension recovered neither prespecified target; it is a completed negative result. New supervision must be materially new and supported, rather than the same extension rerun.
- [occlusion-results-2026-11-14.md](https://github.com/Siuuuers/dl/blob/5feb60b8156c5aff6f9c5ebc3814cbe378ff5628/docs/occlusion-results-2026-11-14.md): PR62 synthetic occlusion improved its synthetic probe but failed clean preservation, including a new empty-scene output. Do not simply rerun that recipe.

A defensible next data action is a bounded consistency review of **comparable eligible training originals**, particularly the visible width/height of fronts exposed at the jaw and the completeness of clipped fronts. Keep development photos in their current role. Any additional upstream frame pool needs its current review status, earlier model exposure and capture relationships checked before use; historical counts of unreviewed images are not asserted here.

Freeze a genuinely new, evidence-backed change before preparing another matched notebook. This queue does not authorize automatic R25 training, a threshold sweep or a model promotion.

## Evidence and verification

- [Fixed PR99 comparison](fixed-comparison.json): exact unmatched-target selection and same-reference PR35/control statuses.
- [Fresh R23 comparison on the same R24 reference](r23-same-runtime-comparison.json): all sixteen targets were also missed by R23.
- [Saved R24 candidate predictions](inference/eval-candidate/predictions.jsonl): original post-filter arrays and provenance.
- [Reference sensitivity](reference-sensitivity.json): stable current-to-primary lineage; no nearest-prediction identity matching.
- [Machine-readable queue](partial-front-failure-queue.json): all 16 targets, exact labels, coordinates, hashes, categories and per-case next evidence.

The unchanged scorer reproduced the complete saved match/unmatched/duplicate traces for all seven relevant images. All image and R24 label hashes matched their manifests; queue completeness, zero/one-corner counts, overlap categories and comparator statuses were asserted. The seven originals and recorded crops were inspected without changing labels.

This is a diagnostic on reused development images. It does not establish unseen-session accuracy, physical visibility from model confidence, or a cause for absent raw network proposals.
