# Recovered cassette-front review

This is the current **review-only** annotation bundle, recovered from the saved 18 November r3 PDFs and followed by one newly inspected correction. It does not select a training recipe or replace the references used by PR67.

| Directory | Role | Images | Fronts | Locatable corners | Missing corners | Empty images |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `labels/train` | Training annotation review | 137 | 3,789 | 6,856 | 722 | 13 |
| `labels/val` | Separate development-reference review | 20 | 662 | 1,226 | 98 | 3 |

`labels/val` is **not** the original training-validation dataset. The original 664-front training validation and PR67's predefined 666/663-front references remain unchanged.

All rows use one `cassette_front` class and two actual lower-front corners:

```text
0 cx cy width height left_x left_y left_v right_x right_y right_v
```

Coordinates are normalized; `kpt_shape: [2,3]`, `flip_idx: [1,0]`. A locatable observed corner has `v=2`; an unlocatable slot is `0 0 0`. Boxes describe observed front material, including its own raised lip. Nearby frames inform ownership but do not supply hidden target coordinates. Retain identifiable partial fronts even when neither lower corner can be located.

## What was recovered

The original TXT authoring files from the interrupted r3 work were not found on a reachable GitHub branch. The two saved PDFs retain vector rectangles and corner circles. The [recovery evidence](../../../results/review/r3-recovery-20261118/) pins those PDFs, verifies every parent label, uses explicit deletion/renumbering records, and checks the complete 157-image inventory. No screenshot measurement or prediction-driven relabeling is used.

The reconstructed r3 has **888 replacements and 12 deletions**. Its 1,817 before/current panel-row comparisons agree with the full-atlas geometry within **0.000719 original-image pixel**. Exact parent tokens are retained for unchanged groups. Newly serialized changed groups reproduce rendered geometry, not the original missing authoring byte strings. This numerical agreement is not a claim of subpixel physical annotation accuracy.

A separate [original-pixel follow-up](../../../results/review/r3-recovery-20261118/new101-row52-followup.json) completes the requested `new_000101` row 52 correction: box `[481,543,550,574]` becomes nominal `[485,543,550,571]`, trimming rack/support margins. The actual left corner and missing right slot are unchanged. The soft contour has a few pixels of uncertainty. The combined review therefore has **889 replacements and 12 deletions** from r2, with no additions.

Only development rows `new_000007:7,14` retain recorded amber holds. Absence of a hold is not full physical certification. The [manifest](manifest.json) binds all current label files, source images, the recovery receipt and the follow-up operation. Review the updated two PDFs before merging this revision.

## Reproduce the recovery

Use the original r3 PDFs whose hashes are recorded in the recovery manifest, not the updated PDFs that include the follow-up. Install PyMuPDF 1.26.6 in a separate Python environment if needed, then run from the repository root:

```bash
python scripts/recover_pdf_review.py \
  --root . \
  --atlas /path/to/cassette-label-atlas-2026-11-18-r3.pdf \
  --changes /path/to/cassette-front-changes-2026-11-18-r3.pdf \
  --output /path/to/new-recovery-directory
patch -p1 -d /path/to/new-recovery-directory \
  < results/review/r3-recovery-20261118/new101-row52.patch
```

The immutable recovery receipt describes the r3 geometry before the follow-up. Current file hashes are in this directory's manifest. The one-row patch is a separate operation and must be applied exactly once. Neither command starts training or changes the selected model.
