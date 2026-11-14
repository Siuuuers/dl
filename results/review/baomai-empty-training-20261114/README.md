# Ten empty-label training originals reviewed

Date: 2026-11-14 (Asia/Hong_Kong).

**All ten inspected originals support retaining their empty labels.** No omitted visible white-wax front face was found, and no label, training overlay, evaluation reference, model or inference setting was changed. This is a new visual review of the original images, not a recovery of an unverified earlier claim.

## Scope and provenance

The existing [remaining-review queue](../border-fronts-20261113/remaining-review-queue.json) contains 67 training images outside the recorded review passes, including ten with zero frozen label rows. Those ten IDs were fixed before this pass. Each restored original PNG was verified against [data/manifest.json](../../../data/manifest.json), opened individually at its complete 1280x720 extent, and inspected for omitted target faces across the full frame. The existing zero-byte label files also match their frozen SHA-256 values. No prediction overlay or annotated preview was used to decide that an image should remain empty.

The target is the actual lower front face of the white wax block. The common large white two-prong foreground fixture, rack lattice, grid openings, background desk items and reflections are not that target. The review is limited to what the image supports; it does not infer whether objects exist behind occluders.

Exact image/label hashes, source-record hashes, per-image decisions and the coverage derivation are in [review.json](review.json).

## Individual observations

| Original | Observation and decision |
| --- | --- |
| `new_000134` | Close view of the empty lattice rack and its partition shelves. The large white split fixture enters from the bottom edge; pale squares belong to lattice openings and background. No visible white wax front face is supported anywhere in the frame. Retain the empty label. |
| `new_000135` | Wider view of the empty rack with exposed shelf grids and glare in the lower-right compartment. The white split fixture remains below the rack. No cassette front face or omitted target is visible in the rack or surrounding image. Retain the empty label. |
| `new_000136` | Tall empty rack on the desk; its upper, middle and lower compartments are visible. The white foreground fixture, printed card at right and surrounding desk items do not form a target wax front. No omitted target face is visible. Retain the empty label. |
| `new_000139` | Close, partly frame-cropped rack with empty lattice compartments. Bright patches in the lower-right grid follow openings and reflections; the large white fixture occupies the lower centre. No supported wax-front target is visible. Retain the empty label. |
| `new_000140` | Empty rack with some image blur, visible partitions and open lattice throughout the frame. The large white two-prong fixture at lower centre is a non-target structure. No supported omitted cassette front is visible; the inspection does not infer anything behind the fixture. Retain the empty label. |
| `new_000141` | Empty rack seen close up, with the white two-prong fixture, its round base and a hand in the lower foreground. All exposed compartments and image boundaries were inspected; no visible cassette wax front is supported. Retain the empty label. |
| `new_000142` | Similar empty-rack view with the foreground fixture and hand covering part of the lower centre. The exposed grid cells and shelves contain no visible target face. No label is added for the non-target white fixture or unseen material behind it. Retain the empty label. |
| `new_000143` | Wider desk scene showing an empty, narrow rack behind the white foreground fixture. Bright grid cells and glare, cups, paper items and a printed card are background or non-target structures. No supported cassette wax-front target is visible. Retain the empty label. |
| `new_000144` | Close oblique view of empty rack compartments and cross-shelves, with the fixture covering the lower centre. Openings, reflections and the white fixture were inspected as possible distractors; no omitted wax front is supported. Retain the empty label. |
| `new_000150` | Empty rack with exposed upper and lower lattice shelves and glare in the right compartment. The two-prong white fixture and hand occupy the lower foreground. No visible cassette wax front or target corner pair is supported. Retain the empty label. |

## What the coverage count means

The [2026-11-12 inventory](../baomai-review-coverage-20261112/coverage.json) records 64 distinct training IDs. The later six-image pilot adds `new_000001`, `new_000013`, `new_000027`, `new_000133`, `old_000004` and `old_000010`, making 70. This pass adds ten different IDs, bringing documented targeted review membership to **80/137**.

The other three empty-label training images, `new_000131`, `new_000132` and `new_000133`, already occurred in prior passes and were not newly reviewed here. All 13 empty-label training IDs now occur in some documented targeted pass. **This does not certify every label or every detail in all 80 images.**

There are still **57 nonempty training images** outside those recorded passes, containing **1,590 frozen object rows and 203 missing endpoint slots**. These counts use the original frozen manifest, not a merged candidate overlay. The historical queue is retained as its dated snapshot; the current subtraction and ID union are recorded in this review JSON.

## Limits and next step

This was one assistant pass over a targeted existing training cohort. It neither supplies independent test images nor measures a model improvement. No inference or training was run, and retaining these ten empty files does not establish that a future model will avoid false positives on them or on other backgrounds. The result removes a specific annotation uncertainty: this pass found no supported reason to add a target to these ten labels.

Continue original-image review of the remaining nonempty images and adjudicate existing held cases before choosing a further training revision. Preserve this evidence alongside the model-result comparison, while keeping review coverage separate from acceptance or accuracy claims.
