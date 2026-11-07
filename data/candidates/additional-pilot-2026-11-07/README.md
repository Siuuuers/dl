# Additional-frame annotation pilot — 2026-11-07

**Status: reviewed partial corrections; all three images quarantined.** This is a
separately versioned annotation proposal for `frame_0001`, `frame_0160` and
`frame_0383`. It is not released for model scoring or training.

The records describe assistant visual review and independent second checks;
they do not claim human approval or that every box boundary is resolved.

The [illustrated report](../../../docs/additional-annotation-pilot-2026-11-07.md)
explains the corrections, exact corner changes and unresolved physical boundaries.
This work follows [PR #18](https://github.com/Siuuuers/dl/pull/18) and applies the
unchanged [baomai-visible-front-v1 policy](../../../docs/annotation-policy.md).

## Scope and disposition

| Image | Objects retained | Box rows changed | Corner slots changed | Unresolved rows |
| --- | ---: | ---: | ---: | ---: |
| `frame_0001` | 43 | 6 | 1 | 13 |
| `frame_0160` | 43 | 39 | 4 | 11 |
| `frame_0383` | 43 | 42 | 0 | 12 |
| **Total** | **129** | **87** | **5** | **36** |

There are no added or removed objects and no visibility changes. The labels retain
256 visible endpoints and two canonical missing endpoints. Source row order is
preserved. Five coordinate slots change across four rows; those rows are included
in the 87 box rows, not an additional object count.

The unresolved fields are 35 `ymin` values on exposed stepped faces and
`frame_0001` row 43's `xmin` beside the gripper. Source values for these fields remain
unaccepted placeholders. Other supported corrections do not make an image eligible.
`eligible_image_ids` is empty; each image's scoring and training eligibility is
false in the [manifest](manifest.json).

This is a selected three-image pilot from 320 recovered originals. The other 317
have not received complete annotation review. Two further originals were viewed
for context only. No model predictions, new inference or training were used, and
the frozen 137/20 development reference remains separate.

## Source and target

Source: `an94-doll/embedding_vision` at
`9a6bf2432582af517d789c94eab2ac27e03a304b`, under
[`box_in_basket/white_box_pose`](https://github.com/an94-doll/embedding_vision/tree/9a6bf2432582af517d789c94eab2ac27e03a304b/box_in_basket/white_box_pose).
The manifest pins the archive, source manifest, original images, original labels,
candidate labels, policy and review records by hash.

The proposal uses class 0 `cassette_front`, two actual lower-front corners ordered
by image X, `kpt_shape=[2, 3]`, and `flip_idx=[1, 0]`. The source name `white_box` is
mapped only for the individually reviewed identifiable fronts. This does not
approve the remaining pool's semantics. Gray front material is retained; a raised
gray roof/rim is not classified as front merely because it touches the cassette.

All coordinates are normalized by the original 1280×720 dimensions. Proposed manual
locations and serialized decimal precision do not imply subpixel accuracy.

## Files

- [manifest.json](manifest.json): source identity, exact changed rows, counts,
  unresolved fields and release gates.
- `labels/`: quarantined provisional TXT files.
- `source-labels/`: original TXT bytes, preserved separately.
- `reviews/`: one normalized review covering every row of each image, including identity,
  independent box/point judgments, alternatives and whole-image assessment.
- [verify.py](verify.py): provenance, accounting, structural geometry and
  quarantine checks.
- [render_comparison.py](render_comparison.py): raw/source/provisional comparisons
  from original pixels, with no model output.

The [review capsule](../../../results/review/baomai-additional-annotation-20261107/review.zip)
contains this candidate under `candidate/`, including nine plots in
`candidate/figures/`. `source/` contains the pinned manifest and the three original
JPEG/label pairs. `editor/` contains lossless decoded-RGB PNG exports and same-stem
TXT files, without resizing. The exports remain quarantined. Reviewer records
preserve the physical judgments; `verification.json` holds candidate consistency
checks, `PNG-export.json` records the image-export relationship, and `inventory.json`
lists capsule files. Use the [structured verification record](../../../docs/additional-annotation-pilot-2026-11-07.json)
for final archive identity, extraction and rendering readback results.

## Verification and continuation

After unpacking the capsule, run from the repository root:

```bash
python data/candidates/additional-pilot-2026-11-07/verify.py \
  --root . \
  --candidate /path/unpacked/candidate \
  --source-root /path/unpacked/source
```

Mechanical checks do not resolve physical surface orientation. Before freezing
eligible labels, adjudicate the 35 upper boundaries using an unequivocal view or
part reference, and resolve the observed left extent of `frame_0001` row 43.
Continue quarantine wherever the evidence remains ambiguous. Review complete
images after correction; do not obtain a score by selecting only resolved rows.
Capture correlation and prior upstream use still limit any later supplementary
evaluation.
