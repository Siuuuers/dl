# R47: neighboring contours and cassette/rack boundaries

27 November 2026, Asia/Hong_Kong.

**All 507 current fronts in the twelve reviewed original photographs were
inspected. R47 refines 412 box boundaries across 320 rows and resolves 338
field holds.** All 1,014 corner slots retain their exact R46 text, including
the 66 unavailable slots. There are no new instances, deletions, corner
movements or visibility changes. All 508 historical identities remain present.

This is the next annotation revision after
[PR134 / R46](https://github.com/Siuuuers/dl/pull/134), whose source head is
`d49a55d7c2240bbb18565e643cf01e691d39a7a3`. PR134 is a CPU review and replay
package. The latest measured training remains
[PR133 / R45](additional-front-pilot-r45-results.md); these R47 changes have
not yet produced a new measured model result.

## The requested physical corrections

Coordinates below are original 1280 x 720 image pixels. Nominal uncertainty
is generally about 1-3 pixels, not a calibrated confidence interval or a claim
of subpixel physical accuracy.

| Stable source identity | R47 decision | Physical reason |
| --- | --- | --- |
| `frame_0160 / 001` | `xmin` about 486.778 -> **484** | Include the leftward molded, shaded frontal wing. Its contour joins this cassette's rim; the continuing rack upright is farther left. |
| `frame_0160 / 037` | `ymin` about 588.091 -> **595**; `xmax` about 537.982 -> **535** | Follow the exposed small tab and continuous upper frontal rim. Row 034's left corner near `(537,597.62)` clarifies the level and repeated shape; its coordinate is not copied. The old upper strip belongs to rack/receding context, while moving down to the bright inner band would cut owned rim. |
| `frame_0001 / 002` | `xmin` about 732.009 -> **733** | Move the left edge of the right-hand cassette off the rack strip. |
| `frame_0001 / 005` | `xmin` about 730.202 -> **731.5**; `ymin=219` | Preserve the rounded outer rim and the shaded front below the preceding cassette's lower seam. |
| `frame_0001 / 008` | `xmin` about 728.341 -> **730**; `ymin=237` | Keep the target's gray frontal body and its previously corrected lower corner, excluding rack from the outer bound. |
| `frame_0001 / 030` | `xmin=876` | A further local side refinement; the actual lower-left point remains at its R46 coordinate. |
| `frame_0001 / 042` | `xmin=697`; `ymin=619` | Follow the exposed front beside the rack and below row 039; retain the unavailable right corner. |

The broad `frame_0001` wording was treated as a review request for source rows
002-030 and 034-042, including the named 002/005/008 examples. All other current
rows were also inspected. A requested range does not imply that every bound
must move. The direction depends on which cassette owns the visible material.

For a cassette on the **right** of a rack strip, moving its **left** edge right
means increasing `xmin`. This is why "more right to exclude the rack" need not
mean enlarging `xmax`. Conversely, the left edge of `0160:001` moves left to
retain its own molded wing. No global inset or brightness threshold is used.

The [annotation policy](annotation-policy.md) now spells out this use of
neighboring contours. The bright-turn/darker-underside/lateral-shadow cue from
R46 remains supporting evidence. Ordinary blur can carry a nominal boundary
and uncertainty; a field hold is reserved for materially competing surface or
instance interpretations. An axis-aligned box can contain incidental neighbor
or occluder pixels internally, particularly around slanted fronts. Its extrema
must still come from the target's visible material.

## All-frame outcome

| Photograph | Current fronts | Rows with coordinate edits | Remaining held rows | Whole-photo gate |
| --- | ---: | ---: | ---: | --- |
| `frame_0001` | 43 | 38 | 1 | Held |
| `frame_0038` | 35 | 28 | 1 | Held |
| `frame_0074` | 43 | 31 | 2 | Held |
| `frame_0128` | 43 | 31 | 1 | Held |
| `frame_0146` | 43 | 31 | 0 | Complete candidate |
| `frame_0160` | 43 | 6 | 0 | Complete candidate |
| `frame_0178` | 43 | 33 | 0 | Complete candidate |
| `frame_0225` | 42 | 33 | 5 | Held |
| `frame_0280` | 43 | 29 | 1 | Held |
| `frame_0301` | 43 | 29 | 2 | Held |
| `frame_0371` | 43 | 31 | 0 | Complete candidate |
| `frame_0383` | 43 | 0 | 0 | Complete candidate; exact PR133 labels |
| **Total** | **507** | **320** | **13** | **5 complete; 7 held** |

The 338 resolved fields belong to 316 rows; some rows have more than one held
field, and a partial resolution can leave another field open. The remaining
13 rows contain 32 open fields. These counts refer to this round. Older
`counts` in the replay verification summarize cumulative annotation history;
`latest_counts` contains R47-only changes.

Further supported corrections include gripper-only spans in
`0225:033/034/035/037/038/040` and `0301:035/038/039/041/042`, as well as
broad receding roofs above `0280:034/036` and `0301:031`. These are visible
front extents, not guesses at a whole cassette behind the gripper. Missing
corner slots remain missing. Some upper boundaries are retained numerically
while their holds are resolved, including `0178:025` and `0178:032`, because a
further contraction would cut their owned gray rim.

## Remaining physical questions

| Photograph / source row | Remaining field(s) | Competing interpretation |
| --- | --- | --- |
| `0001 / 043` | `xmin` | The left gray sliver may be cassette material or gripper shadow. The exposed top/right contour is usable. |
| `0038 / 035` | Front identity | Bottom-clipped pixels may be another front or an underside/rack continuation. |
| `0074 / 033,036` | Top | The gripper hides portions needed to distinguish the frontal rim from the receding upper plane. |
| `0128 / 035` | Top | The obscured upper-left region leaves frontal-rim versus upper-plane ownership unresolved. |
| `0225 / 031` | Top, right | The upper-right strip blends into the gripper. The confirmed roof excess was refined, but the residual limits remain held. |
| `0225 / 036,039,041,042` | Instance identity and box bounds | Nearly hidden or bottom-clipped fragments do not establish separate fronts and their visible limits. |
| `0280 / 001` | Top | Raised prongs/upper surface versus frontal-rim ownership remains uncertain. |
| `0301 / 036` | Top, right | The white piece between the jaws may belong to the cassette rim or gripper. |
| `0301 / 043` | Identity, top | A bottom gray strip may be another clipped front or the preceding object's underside. |

All seven affected photographs remain outside the candidate training export.
A local correction does not turn the remaining uncertain material into
background. The next focused review can address these thirteen identities
without redoing the entire annotation history.

## PDFs and exact evidence

The 150-page amendment PDF includes:

- Summary and whole-photo status first.
- 350 comparison cards covering changed rows, resolved fields, requested rows
  and remaining holds, using the same original crop in all three panels.
- Fourteen manual diagnostic traces, eleven closed and three deliberately
  open. Row 037's neighboring row 034 point is labelled as context.
- All twelve full photographs with every current source identity.
- A complete 507-row audit index mapping stable source IDs to current TXT
  lines and comparison pages.

The separate [44-page error review](front-error-review-r47.md) covers all 13
PR133 candidate misses, all 90 unmatched outputs and all 18 matched corner
failures under current R38. These frozen development predictions are kept
separate from the supplementary annotation changes. Both PDFs are delivered
in the conversation; binary reports are not committed to Git. The old HTML
remains the R44 historical snapshot.

The [R47 review directory](../data/reviews/additional-fronts-2026-11-27-r47/)
contains the full decision ledger, original-image hashes, exact before/after
TXT, traces, all twelve proposal files and the five complete-photo files.
The historical full manifest is rebuilt from the pinned original R44 archive
and exact R46 ledger, avoiding another redundant history copy in Git.

```bash
python scripts/build_front_review_r47.py \
  --r44-archive work/r44-review.zip --download-r44 \
  --output work/r47-review \
  --pdf work/r47-review/r47-annotation-amendments.pdf \
  --zip work/baomai-front-review-r47-results.zip
```

After extraction, `python scripts/build_front_review_r47.py --check-output .`
verifies the package offline. The verifier checks exact predecessor replay,
all current identities, missing-point text, strict geometry, whole-photo
membership and inventory. Unknown old PDFs/HTML/scripts are rejected. An
in-folder amendment PDF has a receipt bound to the current manifest and render
sources; a stale report cannot acquire a new inventory silently.

Use the existing JPEG + TXT preview for a quick row check:

```bash
python scripts/preview_pose_labels.py \
  --image work/r47-review/originals/frame_0160.jpg \
  --labels work/r47-review/proposals/frame_0160.txt \
  --row 37 --scale 6 --output work/row037.png
```

`--row` is the current TXT line, not universally the immutable source row.
The index and manifest preserve the distinction after historical deletion or
addition. In particular, `0128:source:043` remains deleted and the separately
added identity occupies TXT line 43.

## Why a new training notebook is useful now

The five complete photographs provide 215 supplementary fronts, including
eight unavailable corner slots in `frame_0178`. Four photos are newly eligible;
all 43 `frame_0383` labels remain byte-identical to the PR133 input. The
[new expanded-data pilot](additional-front-pilot-r47.md) therefore tests a
changed eligible dataset, unlike repeating the R46 CPU export.

The candidate uses all five complete photos with the original 137 training
photos. The reference uses the previous 138-photo dataset. The notebook
records extra exposure under the same 30-epoch recipe and keeps every original
development image outside training. These related camera views are useful
training examples, not an independent test set. No model promotion or benefit
is claimed before the returned results are reviewed.
