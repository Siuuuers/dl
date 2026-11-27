# R52: four-side supplementary annotation review

R52 revisits all 507 current cassette annotations in the 12 supplementary camera photographs. It applies **11 side-boundary refinements in 10 rows of `frame_0160`**, including all eight rows named in the latest feedback. The other 497 TXT lines remain byte-for-byte unchanged from R48. This is an agent-assisted visual review of the original JPEGs, with no independent human adjudication and no new model training.

The changes are small: 1.829–2.506 original-image pixels. Their recorded nominal visual uncertainty is 2 pixels, reflecting blur and the soft transition between the gray cassette rim and the dark separator. The decimal values in a normalized TXT export are serialization precision; they are not a claim that the camera establishes subpixel physical truth. These refinements do not establish that training accuracy will improve.

## What was inspected

The source is the immutable R48 supplementary release, whose generated manifest SHA-256 is `184abbe67d6c8095b709f69d988665435e0a34680c5827bc6c20dc4947fe06ab`. The parent ledger and every proposal TXT are checked before replay. Original JPEGs come from the pinned R44 capsule, inside `history/pr131-r43.zip`; all 12 JPEG hashes were matched against the R48 identities.

Review covered every full original, every raw/current row crop, all four box sides, and the geometry cues described below. The 24 row sheets contain all 507 rows. Targeted coordinate strips were inspected for the proposed edits. Displays use nearest-neighbor enlargement of the recorded pixels; no super-resolution, enhancement, generated image, or model prediction supplies a boundary.

Source identity is kept separate from current TXT position. This matters because the inherited review contains a historical deletion and one addition. The ledger records `stable_id`, `source_row`, and `output_row`; it does not silently reinterpret a source number as a current TXT index.

| Photograph | Current rows | Amended rows | Held rows | Held fields | Complete-photo export |
| --- | ---: | ---: | ---: | ---: | --- |
| frame_0001 | 43 | 0 | 0 | 0 | Eligible |
| frame_0038 | 35 | 0 | 1 | 1 | Held |
| frame_0074 | 43 | 0 | 2 | 2 | Held |
| frame_0128 | 43 | 0 | 1 | 1 | Held |
| frame_0146 | 43 | 0 | 0 | 0 | Eligible |
| frame_0160 | 43 | 10 | 0 | 0 | Eligible |
| frame_0178 | 43 | 0 | 0 | 0 | Eligible |
| frame_0225 | 42 | 0 | 5 | 22 | Held |
| frame_0280 | 43 | 0 | 1 | 1 | Held |
| frame_0301 | 43 | 0 | 2 | 4 | Held |
| frame_0371 | 43 | 0 | 0 | 0 | Eligible |
| frame_0383 | 43 | 0 | 0 | 0 | Eligible |
| **Total** | **507** | **10** | **12** | **31** | **6 eligible / 6 held** |

Eligibility is the inherited whole-photo annotation gate, not human sign-off or a training split assignment. R52 assigns no new photo to training. In particular, exporting the complete `frame_0001` label does not establish that this image was included in a previous training run.

## The 11 recorded side changes

Coordinates refer to the original 1280 × 720 image: x increases right; y increases down. `xmin` is the left boundary and `xmax` is the right boundary. The table reports the old encoded coordinate and the new nominal visual coordinate; the ledger also records the exact re-encoded coordinate.

| frame_0160 source row | Side | R48 coordinate | R52 nominal coordinate | Reason |
| --- | --- | ---: | ---: | --- |
| 015 | xmax | 602.088 | 600.0 | Exclude the external dark separator while retaining the gray own rim. |
| 016 | xmax | 602.506 | 600.0 | Follow the visible front-to-separator transition. |
| 018 | xmax | 602.829 | 601.0 | Apply the same observed local ownership correction in the next row. |
| 019 | xmax | 540.124 | 538.0 | Exclude the separator toward the adjacent cassette. |
| 020 | xmax | 603.068 | 601.0 | Retain the owned shaded edge; remove the external dark strip. |
| 022 | xmax | 603.344 | 601.0 | Follow the lower row's own gray-to-dark transition. |
| 024 | xmin | 414.000 | 416.0 | Exclude the left separator/background band. |
| 025 | xmin | 353.867 | 356.0 | Retain the cassette turn and remove the external left band. |
| 029 | xmin | 410.372 | 412.5 | Follow the owned front turn beside source row 030. |
| 030 | xmin | 351.547 | 353.5 | Retain its gray left rim and exclude the external strip. |
| 030 | xmax | 417.120 | 415.0 | Preserve the slanted upper right extent while excluding the separator toward source row 029. |

Rows 018 and 022 are the two additional amended rows. Row 030 also receives a right-side refinement in addition to the requested left-side amendment. The other 11 photographs were reviewed and retained: their images did not support a further displacement in this pass.

Every class token, every y-center and height token, all 1,014 corner triplets, and all visibility values remain exact. This includes 948 locatable corners and 66 unavailable slots. The 31 held fields remain unchanged. No row was added, deleted, resurrected, or reordered. The existing strict geometry verifier confirms that every locatable corner remains inside its own exported box; the new two-pixel screening tolerance is not used to relax that export check.

## Reusing the neighbor-corner checker safely

`scripts/audit_front_geometry_r52.py` extends the R28 label-geometry screen in `scripts/rank_front_geometry_candidates.py`. It checks every current supplementary row and preserves the earlier strict-interior and lower-edge-gap definitions.

The screen records three separate cues:

1. **Strict foreign interior:** another cassette's locatable corner lies more than 2 pixels inside all four sides of this box.
2. **Edge contact:** another locatable corner lies within ±2 pixels of a left, top, right, or bottom edge, and within the other axis's span expanded by 2 pixels. A corner contact can belong to two edges.
3. **Large lower gap:** `ymax - max(locatable lower-corner y)` exceeds `max(5 pixels, 0.25 × box height)`, as in R28.

The screen also detects an own locatable corner more than 2 pixels outside the box. Such violations are ranked first, followed by strict foreign interior, lower-gap cues, and ordinary edge contacts. Unavailable `0 0 0` corners supply no position. The distance from a lower-front point to `ymin` is not required to be near zero: that distance naturally includes the front's height.

| Diagnostic count | R48 before | R52 after |
| --- | ---: | ---: |
| Rows with any cue | 370 | 368 |
| Boxes with a strict foreign interior corner | 38 | 36 |
| Strict foreign interior corner events | 42 | 40 |
| Boxes with an edge contact | 354 | 353 |
| Edge-contact corner events | 708 | 708 |
| Rows with a large lower gap | 11 | 11 |
| Rows with an own corner more than 2 pixels outside | 0 | 0 |

These rows overlap between cue categories, so the category counts must not be added to obtain the total. The unchanged 708 contact events are expected: a small correction can turn an interior event into an edge contact while removing a different contact. The target is consistent material ownership, not a zero-flag dataset.

**A neighbor's point inside an axis-aligned box is not physically impossible.** A tilted front has a rectangular envelope that can contain background, another cassette's projected point, or part of an occluder between the object's own visible extrema. Repeated molded lips and closely stacked fronts create many legitimate contacts. For example, retained `frame_0280:source:006` has projecting lips and a recessed side; shrinking to the recess would exclude cassette material. Retained `frame_0178:source:005` illustrates an envelope intersecting the adjoining tilted upper front.

The 38 original strict-interior flags and 354 contact flags are review workload, not measured annotation-error counts, precision, false-positive rates, or model metrics. Two strict-interior boxes were amended, but no independent adjudicated reference or timed review comparison was collected. These observations do not measure diagnostic precision or review-efficiency gains.

## Shadow and context policy

Include visible material belonging to the target cassette even when it is gray or shadowed, regardless of which object casts the shadow. Do not enlarge the box because a shadow falls on the rack or background. Likewise, the neighboring cassette, gripper, and rack must not determine the target's extrema.

An axis-aligned rectangle can still enclose some of those non-target pixels between valid extrema. Requiring every interior pixel to belong to the cassette would confuse a bounding box with a segmentation mask. Full-frame training already presents surrounding context to the detector. If an auxiliary classifier needs a wider crop, expand that input crop explicitly while keeping the target annotation policy consistent.

## Reproduce and verify

The canonical decision data and generated outputs are under `data/reviews/additional-fronts-2026-11-27-r52/`. The audit ledger contains every retained and amended row, inspection-sheet references, reasons, exact before/after TXT lines, unchanged holds, and nominal uncertainty. `manifest.json` contains the row map, side-change list, parent hashes, whole-photo gates, and all generated label/screen hashes.

From the repository root:

```bash
python3 scripts/build_front_review_r52.py \
  --originals /path/to/verified/12-jpegs \
  --output-dir /path/to/r52-replay \
  --evidence-dir /path/to/r52-evidence

python3 scripts/build_front_review_r52.py --check

python3 -m unittest discover -s tests -p test_front_geometry_r52.py -v
```

The builder independently replays the recorded amendments, protects class/y/corner tokens, preserves holds and full-photo gates, reruns both geometry screens, and rejects changed parent inputs. It produces no training run and does not alter returned predictions or evaluation references. The six targeted tests cover four-side contacts, strict interior distinction, two-edge corner contact, non-mutation, unavailable corners, photo isolation, protected tokens, and full-ledger replay.

To regenerate every inspection sheet, run:

```bash
python3 scripts/audit_front_geometry_r52.py \
  --output /path/to/r48-before-screen.json \
  --originals /path/to/verified/12-jpegs \
  --sheets /path/to/r52-inspection-sheets
```

`build_front_review_r52.restore_originals()` extracts only the 12 named JPEGs from the pinned R44 capsule and verifies both the capsule and individual image hashes. The builder exposes the source URL, byte count, and SHA-256 for notebook use.

For a later annotation-only experiment, hold the image inventory, split, training configuration, seeds, and evaluation references fixed. Compare the label versions on the same existing images. Adding a newly complete photograph at the same time would mix a data-volume effect with the annotation change. The present review is preparation for that test, not its result.
