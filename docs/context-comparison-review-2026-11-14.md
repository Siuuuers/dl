# Context comparison and the next notebook run

**Run the existing accepted-fronts notebook for the controlled comparison.** The second marked photo already has three distinct thumb-side fronts in that dataset. The first marked photo supplies stronger ownership evidence and a useful upper-bound proposal, but its remaining instance question does not justify changing the pinned experiment.

[Open the pinned notebook in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/cf5957b13abef6af6ea05bfa9f7c1afcbf4575f3/notebooks/baomai_accepted_fronts.ipynb). Its source pin, label candidate and protocol remain unchanged. This is permission to perform the specified experiment, not a declaration that every annotation or the resulting model is ready for deployment.

The [context review record](../results/review/context-comparison-20261114/review.json) binds the original images, exact saved labels, user attachments and current handoff. No label operation or case closure is applied by this follow-up. The four-page visual addendum can be reproduced with `scripts/render_context_comparison_review.py`; its audit records the PDF and source identities. The earlier 166-page atlas remains the full-review snapshot.

## The second marked image already has three fronts

The attachment matches **new_000038**, using the nail crease, thumb contour, gripper edge, rack pattern and stepped cassette contours. It is a different photo from **new_000037**, whose two-front correction appears on page 5 of the earlier atlas.

All coordinates below refer to the original 1280 by 720 image. These are the existing saved annotations; the user's confirmation establishes count and identity, not approval of individual pixel coordinates.

| Thumb-side front | Current row | Existing observed box `[left, top, right, bottom]` | Actual lower corners |
| --- | ---: | --- | --- |
| Upper | 1 | `[488, 592, 529, 615]` | Left `(490, 612)` locatable; right missing |
| Middle | 16 | `[490, 614, 518, 632]` | Left `(491, 629)` locatable; right missing |
| Lowest | 17 | `[492, 631, 513, 650]` | Left `(493, 647)` locatable; right missing |

These rows are identical in the full-reviewed file and limited training treatment. Adding another row for the three-front clarification would duplicate an existing object. Counting a cassette does not make its hidden right corner locatable. The separate row 18 at the bottom image border is elsewhere in the photo, not a fourth thumb-side front.

## Question 2: the shoulder helps assign the upper diamond patch

The first attachment matches **new_000030**, approximately the original crop `[624, 424, 805, 525]`. The orange ink identifies an exposed shoulder beside the gripper. Comparing this shoulder with row 11 above, the central rack divider and neighboring stack tiers supports the user's proposed association: **the marked shoulder and upper gray aperture patch belong to the same existing front, row 12**. This is a contextual inference corroborating the user's proposal; a fully exposed front is not required for recognition.

There is a concrete extent problem. The current row 12 box is approximately `[623.42, 455.72, 796.83, 502.40]`. Its top omits the associated upper material. A **nominal top at y443**, with roughly two to three original-image pixels of edge uncertainty, is a supported review proposal. Both actual corners remain missing. The proposal does not certify the inherited left, right or lower bounds.

The lower dark triangle continues to about y500–502, with an internal transition near y483. A further cassette tier is compatible with the neighboring fronts. However, the orange stroke's endpoint is a hand-drawn cue, not a measured physical seam. The current evidence does not uniquely establish whether the lower triangle belongs to the same front, another cassette front or a different surface.

**Decision:** preserve the stronger upper-patch association and prospective top correction in the review record. Do not add a second instance or apply a partially certified replacement to the training candidate. The lower-triangle ownership/count distinction still prevents a complete replacement; the inherited horizontal and lower bounds are not newly certified. The unchanged notebook still uses exact legacy labels for this entire image in both arms.

## Question 3: compare both the stack and its molded transitions

For **new_000117**, row 23 ends above the neighboring bottom-front/rack level. A lower cassette under the hand is therefore plausible. This is useful context, not an irrelevant guess.

The same image also shows the competing explanation. The transitions between rows 20 and 21, and between rows 21 and 22, contain narrow bands before the next broad front appears. Terminal row 25 has a narrow lower rim without a separately exposed broad front beneath it. A band in the predicted tier can therefore be a front fragment, a roof or a molded edge. Its location alone does not settle which target surface owns it.

The full-reviewed row 24 remains `[815, 491, 902, 503]`, with both corners missing. The limited training experiment retains the older PR35 geometry for the whole photo; that older geometry is not newly certified by this comparison. Row 11's top-strip question is also still open in this photo.

**Decision:** hold the surface-ownership question; apply no addition, deletion or inferred corner. Small height is not a reason to remove an identifiable target.

## Question 4: the frame removes the distinguishing context

For **new_000128**, compare the top regions around y0–35 with the fully visible transitions around y48–85 in the same columns. The shallow top strips resemble the molded or roof transitions leading into rows 2, 7 and 10. That makes a shared-owner interpretation plausible.

Clipping removes the region where a preceding broad front would be visible and enough of the transition that a remnant of a preceding cassette remains plausible too. The three columns are not equally exposed. The evidence does not support deleting rows 1, 6 and 9 together merely because their boxes are about five pixels high.

**Decision:** preserve those three holds and all missing corners. Equal physical cassette dimensions help check consistency, but perspective and occlusion prevent directly copying neighboring pixel heights or hidden corner coordinates.

## What is ready, and what is still unknown

| Item | Current conclusion |
| --- | --- |
| Three thumb-side fronts in new038 | Already represented; user clarification corroborates the count |
| Upper shoulder/diamond association in new030 | Supported contextual interpretation; nominal top correction recorded for further review |
| Complete ownership of the lower diamond triangle | Still pending |
| Thin band in new117 and clipped strips in new128 | Competing surface interpretations remain; no supported dataset edit |
| Existing limited notebook | Ready for its controlled comparison, with no new dataset version needed |
| Every label certified / model improved / deployment ready | Not established by this review |

The treatment still has **137 training photos, 3,788 objects, 6,858 locatable corner slots, 718 missing slots and 13 empty photos**. It changes label bytes on 71 photos relative to the fresh PR35-label control. All 21 unresolved training photos retain exact legacy labels in both arms. There are still 24 pending training case groups and 13 held development-reference rows.

Preserving legacy supervision equally defines what the comparison changes; it does not guarantee the effects of those imperfect labels cancel. The reused development photos also do not become an independent deployment test. The [training guide](accepted-front-corrections-2026-11-14.md) specifies the two fixed evaluation references and the existing comparison thresholds.

## External run

1. Open the pinned accepted-fronts notebook in a fresh GPU runtime and run its cells in order.
2. Keep the existing pins, verification checks and numerical settings. Allow the resource trial to complete before the two full runs.
3. Let both arms complete the specified 30 epochs. Each independently initializes from the original epoch173 checkpoint; the final epoch is preselected.
4. Return `baomai-accepted-fronts-seed46-results.zip` for the paired analysis. If the resource trial or another check fails, preserve the error and receipt instead of changing the experiment to get past the check.

No GPU training has been performed in this review. Existing model weights and the selected PR35 comparator remain unchanged.
