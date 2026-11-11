# Focused review after the seed46 confirmation

**The five current misses outside the previously unresolved group and two related diagnostics have been reviewed against their original pixels. No label change is accepted.** The existing nineteen unresolved dispositions remain unchanged. This closes the bounded inspection; it does not certify every retained box or supply a new training-label intervention.

The [machine-readable record](../results/verification/baomai-readiness-20261111/focused-case-review.json) preserves image and label hashes, original/revised row identities, crop coordinates, observations and limitations. Six original 1280×720 PNGs were viewed in full, followed by seven nearest-neighbor enlarged crops. Separate inspection copies carried selected reference rectangles. Original pixels and labels were unchanged; no inference or training ran for this review.

## Original-pixel findings

All row numbers below are one-based and are identical in the original and PR30 revised references for these seven cases.

| Case | Observation and decision |
| --- | --- |
| `008:33` | The right-clipped front has a visible left lower junction; its right corner is outside the image. Retain the PR30 lower-edge correction, which separates it from the front below, and retain the missing right point. |
| `009:57` | A small front/right-end fragment remains beside the gripper jaw. Retain the PR30 visible-fragment box and missing left point; do not expand into the hidden face. |
| `009:52` | A short exposed front segment and its left lower junction are visible beside the gripper. Preserve the instance and missing right point. Changed matches between models do not establish a reference defect. |
| `011:61` | A pale central fragment is exposed between the jaws; both actual outer lower corners are unavailable. The existing rectangle spans tool/background beyond that fragment. Its exact extent and relationship to neighboring row 59/front-versus-lip structure remain uncertain, so no replacement rectangle or point is selected. **This is an open boundary/identity caveat, not certification that the box is tight.** |
| `087:15` | A narrow front has two observable lower ends; the upper-face transition is soft. Preserve its reference. A proposal reaching farther upward can fail box overlap despite nearby useful corners; model agreement does not adjudicate the upper edge. |
| `007:15` | Top clipping and closely spaced horizontal structures accompany two competing candidate proposals. Preserve the reference and the official match; the alternative proposal remains diagnostic evidence only. |
| `090:43` | The left lower junction is visible while the hand hides the right side. Retain the PR30 lower-edge amendment and missing right point. A returned coordinate on the hand is not an observed hidden corner. |

The numeric context comes from the [unchanged PR36 queue](../results/evaluation/baomai-annotations-seed46-20261111/review-queue.md) and [completed report](annotation-confirmation-2026-11-11.md): `087:15` remains a miss at IoU 0.493069; `007:15` has a more accurate lower-confidence alternative; `090:43` crosses the fixed IoU 0.50 boundary narrowly. This review changes none of those scores or assignments.

## Consequence for the next step

Keep the [visible-front policy](annotation-policy.md), including identifiable partial fronts and canonical missing points. The `011:61` caveat is recorded separately without silently changing the frozen nineteen-row unresolved registry. A future supervision pass may investigate an analogous training case, but the present inspection does not justify another label overlay or training run.

Use the candidate through the reviewed folder workflow and judge it with object, scene and corner results together. This is a new bounded review; the earlier interrupted conversation's unpublished review and resolution trial are not treated as recovered evidence. No independent-test or supervisor-approval claim is made.
