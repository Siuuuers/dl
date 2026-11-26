# R46: physical corner and cassette-front ownership review

27 November 2026, Asia/Hong_Kong.

**All 507 current fronts were inspected. This round amends 29 rows: 27 corner
positions and 17 box boundaries.** All 17 user-named rows have explicit decisions;
twelve further rows have supported corrections for the same problems. There are
no additions, deletions or visibility changes in this round. The 508 stable
historical identities, including the earlier deletion, remain intact.

[PR133](https://github.com/Siuuuers/dl/pull/133) is completed R45 training, including
`frame_0383` in its candidate. Its [verified outcome is mixed](additional-front-pilot-r45-results.md).
R46 changes only held photographs; all 43 `frame_0383` label rows are byte
identical to the completed experiment. A repeat GPU run would not test these
amendments. The new [four-cell CPU notebook](../notebooks/baomai_front_review_r46.ipynb)
checks annotations, makes the review PDF, replays PR133 and exports a usable ZIP.

## Physical guidance recorded

The [working annotation policy](annotation-policy.md) now explicitly records the
user's observation: a lower-front corner often has a relatively bright turn,
darker material or shadow underneath, and a lateral gray/shadowed region. This
supports a contour judgment; it is not an illumination-invariant rule. The
earlier [R25 guidance](annotation-boundaries-r25.md) already used brightness and
shadow as supporting evidence.

Identify one front, follow its lower edge to its own side turn, and check the
unmarked original crop before placing a point. A rack edge, neighbor, internal
stripe or cast shadow can mimic a corner. Preserve shaded material owned by the
front, including its frontal rim or small tab. Exclude the broad receding upper
plane, rack, gripper and neighboring cassette from the box's extrema. An
axis-aligned box can still contain background beside a slanted contour.

Image X increases rightward and image Y increases downward. **Moving a point up
decreases Y.** Moving the top boundary down increases `ymin`. The physical
corners need not coincide with rectangle corners. Each missing slot retains its
exact inherited tokens and numerical `0 0 0` value; no hidden point is invented.

## Requested rows and decisions

Coordinates below are original-image pixels. Rounded values make the physical
decision readable; the ledger retains exact before/after text. Nominal visual
uncertainty is usually about 1-3 pixels, not a calibrated confidence interval.

| Frame / immutable source row | R46 decision |
| --- | --- |
| `frame_0001 / 001` | Right point about `(654.21,213.45)` to `(654,211)`, upward onto the lower-front turn. |
| `frame_0001 / 008` | Left point to `(732,253)`; retain its upper-boundary hold. |
| `frame_0001 / 015` | Right point to `(876,362)`. |
| `frame_0001 / 023` | Left point to `(640,414)`. |
| `frame_0001 / 030` | Left point to `(877,457)` and `xmin=875`, excluding the continuing rack strip. |
| `frame_0160 / 005` | Left point to `(372,322)` and `xmin=371`. |
| `frame_0160 / 039` | Restore the earlier `xmin` about `468.824`. The left point was identical in source/R41/R43/R44 and is retained exactly; keep the independently supported R44 right-side correction. |
| `frame_0038 / 008` | Left point to `(682,385)`. |
| `frame_0038 / 009` | Left point to `(820,387)`. |
| `frame_0074 / 027` | Left point to `(425,515)`. |
| `frame_0074 / 043` | Left point `(497,690)`, `ymax=691`, `xmax=566`; exclude the bottom base strip and right upright. |
| `frame_0128 / 029` | `xmax=637`, before the continuing rack seam/neighbor. This one-pixel nominal refinement is smaller than local blur. |
| `frame_0128 / 039` | `xmin=712`, left point `(714,672)`; preserve the missing right point and top hold. |
| `frame_0128 / 041` | Restore R43 `xmax` about `710.673` and right point about `(707.375,675.236)`; retain the missing left point. |
| `frame_0146 / 025` | Tighten `xmax` to `511`, preserving the owned shaded frontal side. |
| `frame_0178 / 006` | Restore prior `xmin` about `707.773` and expand `xmax` to `780`; retain its physical corners. |
| `frame_0225 / 003` | Move `ymin` down to `205` and left point to `(463,228)`. This is a partial refinement; its upper-plane ownership hold remains. |

The additional twelve amended rows are `frame_0160:001,003`,
`frame_0074:001,009`, `frame_0280:014,024,025,031`, and
`frame_0301:015,024,025,026`. The last groups include points roughly 3-4 pixels
below the visible lower-front turn in the darker underside. Every current row,
including the 478 retained rows, has a recorded disposition.

## Contours, whole-image gates and training consequences

Nine manually interpreted diagnostic traces are supplied: seven closed outlines
and two deliberately open lower contours (`frame_0074:043` and `frame_0280:025`).
The open traces do not invent an unresolved top edge. No polygon is generated
from a bounding rectangle, and none is a segmentation training label.

The 34-page amendment PDF shows all 29 comparison cases, all 17 requested
identities with their current TXT line mapping, the nine traces, and all 507
current identities across twelve full-frame overviews. The existing HTML is
left as the historical R44 snapshot; R46 decisions live in the new PDF, ledger
and TXT exports.

The whole-image gate is unchanged: **328 source rows retain field holds and
eleven photographs remain excluded from the complete-candidate export.** Only
`frame_0383` is complete. Its label SHA-256 is
`1848d4dad3185444726de644f5303ee548fc74e08d4361d11c8bd8220fece7b0`,
exactly the PR133 training input. A useful partial correction does not turn all
remaining unlabeled or uncertain regions of that photo into valid background.

`frame_0160:037` remains the single held row in that 43-front photograph. The gray
upper-plane/tab transition around `y=592-596` is still ambiguous in the original
pixels. Its inherited `ymin=588.091` is a held placeholder, not a certified
contour. Resolving that one field from physical evidence is a focused way to
qualify another whole photograph. Other holds must be reviewed on their own
images; do not extrapolate a universal gray/white boundary rule.

No frozen training/development label, split, inference threshold, model
architecture or selected checkpoint is changed. The next GPU comparison should
first freeze a genuinely changed eligible whole-photo treatment. More photos
should add useful capture/occlusion variation; adjacent frame names alone do not
prove independent capture sessions or create an independent test set.

## Files and one-step use

The [review directory](../data/reviews/additional-fronts-2026-11-27-r46/) contains
the full 507-row audit, outlines, twelve proposal TXT files, the sole complete
candidate TXT, and a compact release status with all hashes. The complete
historical manifest is rebuilt deterministically from the pinned R44 archive
and ledger, avoiding another duplicate copy of its large history in Git.

Open the [R46 notebook](../notebooks/baomai_front_review_r46.ipynb), select a CPU
runtime, and run all four code cells. It uses one published source commit,
checks fixed archive hashes, exports the review, regenerates the PDFs and
replays all twelve returned score reports without training or inference. The
two PDFs and one ZIP are downloaded at the end. Reproducing identical files
does not require another result PR.

For local use:

```bash
python scripts/build_front_review_r46.py \
  --r44-archive /tmp/baomai-r46/r44-review.zip --download-r44 \
  --output /tmp/baomai-r46/review \
  --pdf /tmp/baomai-r46/review/r46-annotation-amendments.pdf \
  --zip /tmp/baomai-r46/baomai-front-review-r46-results.zip
```

The ZIP includes the small verification dependency set. From its extracted
directory, run `python scripts/build_front_review_r46.py --check-output .`.
Pillow and ReportLab are needed only for rendering; annotation replay itself
uses the standard library.

To quickly check a JPEG and an existing annotation TXT:

```bash
python scripts/preview_pose_labels.py \
  --image /tmp/baomai-r46/review/originals/frame_0001.jpg \
  --labels /tmp/baomai-r46/review/proposals/frame_0001.txt \
  --row 1 --scale 6 --output /tmp/baomai-r46/row001.png
```

Omit `--row` for the entire photograph. Preview numbers are one-based TXT lines;
use the manifest's stable-ID/output-row mapping after an earlier deletion. The
existing [HTML labeler](../tools/cassette_labeler/README.md) also supports manual
TXT editing, but its current folder importer expects PNG files. The quick
preview above accepts the original supplementary JPEGs without converting them
or modifying the HTML.

## Verification and limits

The [verification record](../results/review/front-review-r46-20261127/verification.json)
checks exact replay, original raster hashes, schema/geometry, all 508 historical
identities, all missing/visibility tokens, and the whole-image export set.
A separate extraction ran the bundled verifier and label preview successfully.
Adding a held photo to the complete-candidate directory was rejected. The
archive cannot be written inside its own input folder; generated bytecode is
excluded. The notebook's four Python cells compile.

The amendment PDF was rendered and inspected, with text-bound and identity
coverage checks. The [PR133 result audit](additional-front-pilot-r45-results.md)
reproduces every recorded score report from saved predictions. No new GPU run
or fresh model inference occurred in this review. Checksums establish file
consistency; the pixel judgments remain reviewable assistant proposals.
