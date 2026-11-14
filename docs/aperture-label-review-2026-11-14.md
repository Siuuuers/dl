# Cassette material, observed fronts, and all-photo label atlas

> Historical review snapshot. The [complete visible-front sweep and prepared handoff](complete-label-review-2026-11-14.md) supersedes the current-status counts in this report; its original evidence remains preserved.

**14 November 2026 — draft review candidate, not a training release.**

The user's confirmation establishes that cassette material is present behind the always-see-through opening in `new_000030`. It does not by itself determine whether the internal transition near y482 divides two front faces. That row is retained, its two unavailable actual corners remain missing, and its count/extent questions are visible in the atlas.

This pass processes all **87 previously open case groups across 64 training images**. The accepted incremental changes cover **42 images**: **182 row refinements**, comprising **376 box-bound changes** and six explicitly reviewed endpoint-row changes, **4 observed-front additions**, and **7 removals of non-target or unsupported duplicate identities**. Every replacement includes a box correction; the six endpoint edits are additional fields in those replacements, not six extra objects.

The complete candidate inherits all compatible earlier work and contains **71 training label files**. Original photos/labels, original validation, the separate development reference, the frozen annotation policy, historical candidates, and selected weights are unchanged. There was no training or inference.

[Candidate manifest](../data/candidates/aperture-reviewed-labels-2026-11-14/manifest.json) · [Exact accepted operations](../results/review/aperture-labels-20261114/accepted-review.json) · [Case status and questions](../results/review/aperture-labels-20261114/review-status.json) · [Selected original-pixel evidence](../results/review/aperture-labels-20261114/evidence-index.json)

## What the physical facts establish

Equal physical dimensions, a stationary rack arrangement, and common rack widths are useful correspondence constraints. Perspective still changes projected pixel widths and heights with depth and position. These constraints can help identify a likely slot or notice an inconsistent label; they do not measure an unseen corner.

A stationary cassette requires support. The supporter may be another cassette, a rack member, a ledge, or a gripper. The existence of support does not prove a visible lower-front instance at every presumed tier. An open hole with no visible rack shows that something blocks the line of sight, but absence of visible grid alone does not uniquely establish its material, surface plane, or instance count. For `new_000030`, the user's direct physical identification supplies the material evidence; the unresolved issue is its internal seam and precise visible front ownership.

The vertices of the diamond-shaped opening are **occlusion boundaries**. They are not substitutes for the cassette's actual lower-left/lower-right front corners.

## Target and source requirements

The current two-point source asks for the actual two lower-front corners, ordered by image x, with each unlocatable point independently missing. Its README does not impose a fully-visible-only object filter or require guessed full hidden boxes. The observed-front box convention was adopted here with support from the historical manual editor's front-face instructions. The upstream five-point/front-back task has different whole-object/side-plane requirements and must not silently redefine this two-point dataset.

Primary sources:

- [Pinned baomai two-point data README](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/README.md#L31-L39).
- [Pinned historical annotation editor](https://github.com/an94-doll/embedding_vision/blob/6f7112a553e1759fc755bda1f18ee56cb34dd315/detect/data/data_last/picture/work/ANNOTATION_EDITOR.md#L19-L20).
- [Separate yolo-pose five-point task](https://github.com/an94-doll/embedding_vision/blob/72a99dafa1b14bfabd101e639b9ddb5f5aa38d52/annotations/README.md#L34-L40).
- [Repository annotation policy](annotation-policy.md) and [earlier requirement comparison](visible-and-hidden-2026-11-14.md).

A visible partial front remains a positive object even when both actual corners are hidden or outside the frame. Its missing endpoint triples remain canonical `0.00000000 0.00000000 0`. A box may enclose the occluder between disconnected visible pieces belonging to the same single front. Fully hidden instances are not added from regular spacing alone.

## Material corrections

All row numbers in this table are **stable original row IDs**. PDF labels use **current output row numbers**, because deletions can shift later rows. The manifest preserves original→parent→current mappings.

| Image / original row | Accepted correction | Remaining limit |
| --- | --- | --- |
| new006 A59 / B53 | A top547/right526; B top563. Preserve the two user-confirmed identities and existing points. | Remaining blurred front/lip/hand boundaries. |
| new024 2/4/5 | Remove labels on metal rack/rail rather than cassette fronts. Row2's alleged visible point is explicitly identified as metal. | Thin real white fragment3 remains held; do not delete all small border labels. |
| new027 | Add two distinct top-border front fragments; three actual lower corners are visible across the two additions. | The second right corner remains missing at the frame edge. |
| new065 | Add the clearly visible bottom-clipped front. | Both lower corners are outside the image. |
| new061 1; new066 15; new074 14 | Remove roof-only regions without an observed vertical front before the frame boundary. | Does not assert no physical cassette exists outside the view. |
| new094 23/24/28 | Retain the bottom strip23 with a smaller observed box; remove unsupported third identity24; expand clipped front28 and trim hand. | Exact strip23 ownership and the separate aperture question remain held. |
| new097 36 | Restore observed right corner `(656,441,v2)` and extend bottom443, combining with the accepted top correction. | A separate possible extra front below40 remains unlabelled pending review. |
| new115 22 | Correct upper-front box, restore observed left corner `(607,474,v2)`, add a separate lower front with both corners missing. | Manual boundary placement spans a few pixels. |
| new116 22 | Correct box and restore observed left corner `(602,481,v2)`. | Lower dark material is held; do not copy new115's count into this frame. |
| new117 22/26 | Transfer the exact existing `(605.25,472.93)` point tokens from lower26 to upper22; lower26 becomes both-missing. Both fronts remain. | No coordinate comes from a different image or periodic spacing. |
| new117 23/24 | Correct23 left/top443/bottom493; trim retained24 to top491/bottom503 with the reviewed side bounds. | Band24 may be a separate front or underside/lip; identity is not certified. |
| new124 / new127 | Hold all four proposed aperture additions. New12439 right bound and new12710 clipped front extent are separately corrected. | Internal intensity changes do not yet establish the proposed two-front splits. |
| new130 54–57 | Refine observed narrow-strip boxes; preserve the already-labelled lower-right front unchanged. Reject its duplicate new proposal. | Proposed right corners54/55 remain missing: physical corner ownership is not sufficiently established. |
| new002 45 | Move the existing visible right point from about `(879.91,564.43)` to `(875,574)`. | This is an observed existing-v2 coordinate correction, not hidden-corner recovery. |

The other accepted box edits remove rack, roof, jaw, hand or neighboring-front overreach, or restore omitted visible fascia. They do not constitute whole-dataset certification of every surviving point or box.

## Dataset counts and model provenance

| Supervision | Images | Objects | Locatable point slots | Missing point slots | Empty images |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actual PR35 training inputs | 137 | 3,771 | 6,841 | 701 | 13 |
| Frozen fixed-geometry parent | 137 | 3,773 | 6,849 | 697 | 13 |
| This reviewed training candidate | 137 | 3,770 | 6,854 | 686 | 13 |
| Original validation, unchanged | 20 | 664 | 1,227 | 101 | 3 |
| Separate validation/development reference, unchanged | 20 | 663 | 1,227 | 99 | 3 |

The candidate's net three-object reduction is four additions minus seven removals. Point count changes include three newly observed points on retained fronts, three visible points on added border fronts, one removed false point on metal, and the count-neutral point transfer. The existing-v2 coordinate refinement changes accuracy of that annotation without changing the count.

**Existing model weights still reflect older supervision.** The PR35 return archive and training receipts establish the earlier eight-file overlay, including original new038 defects absent from later repairs. See the [PR35 provenance audit](label-refinement-2026-11-14.md) and its [machine-readable record](../results/review/label-refinement-20261114/provenance/audit.json).

PR35 remains selected. The previous development evaluation had20 missed objects and85 extra proposals; PR62's synthetic occlusion gains did not resolve the clean misses and introduced regressions. This label work is not a new model result. See [PR62 evaluation](occlusion-results-2026-11-14.md).

## What remains before another GPU run

Of the87 incoming case groups, **28 are resolved within their stated scope**, **39 have accepted progress with a remaining question**, and **20 remain held**. Remaining baseline cases concern **46 images**. One additional possible lower-front case in new096 is recorded separately, rather than changing the original denominator. These are documentary case counts, not a label-accuracy percentage. Soft-placement ranges are visual review ranges, not calibrated probabilities.

The highest-value user decision is whether the transition inside `new_000030` near y482 separates two cassette fronts or is shading within one front. Similar split questions remain in new124/new127. Additional physical-surface questions concern narrow lips/undersides, distant shaded fascia, and several clipped or hand-covered fragments. All are retained in the status file, with exact image/row/region information.

This candidate has `release_ready:false` and `training_recipe_id:null`. To create a training release, resolve those material decisions or explicitly quarantine affected images and document the resulting split. Do not silently present retained legacy uncertain rows as certified negatives/positives. Keep the development reference fixed for the comparison. Only then choose one controlled training hypothesis with fixed evaluation settings and a documented dataset identity.

## All-photo PDF

The **163-page atlas** contains a cover, reading guide, two clickable index pages, focused new030/new006 crops, and **one page for each of the157 working photos**:137 train and20 validation. All16 empty-label photos are included. It does not include excluded final-test/gap/unaccepted upstream images outside this frozen inventory.

The PDF displays the new training candidate and the separate663-object development reference. Cyan boxes are current annotations; amber dashed boxes identify recorded unresolved rows. Green/pink points are saved locatable left/right corners. A green segment is drawn only when both saved points have v2. This annotation preview has no model confidence scores; confidence-gated inference is a separate existing runtime behavior. High model confidence does not prove physical visibility or correct location.

Photos retain native1280×720 resolution with high-quality JPEG encoding for the review document. Overlays are vectors; focused original PNG crops are lossless. Original repository PNGs remain authoritative for pixel decisions. The [atlas audit](../results/review/aperture-labels-20261114/atlas-audit.json) records every page's source image hash, label source/hash, current held rows, full notes, and the PDF's checksum. Long footer notes are visibly abbreviated and retained in full in that audit.

## Reproduce and verify

```bash
python -B scripts/compose_observed_pose_review.py --root . verify \
  --candidate data/candidates/aperture-reviewed-labels-2026-11-14

python -B scripts/render_label_atlas.py --root . \
  --candidate data/candidates/aperture-reviewed-labels-2026-11-14 \
  --status results/review/aperture-labels-20261114/review-status.json \
  --output /tmp/cassette-label-atlas-2026-11-14.pdf \
  --audit /tmp/cassette-label-atlas-audit.json
```

The composer first reconstructs and verifies the historical parent with its unchanged verifier, then enforces exact parent rows, stable identities, declared changed fields, unchanged token preservation, original-image evidence for restored visible points, explicit non-target evidence for deleting a row with a visible point, and a bijective current row map. It refuses validation edits or a release-ready claim. Canonical literal visibility values prevent malformed encodings from bypassing point evidence checks. An addition whose box overlaps a complete parent instance by IoU0.75 or more requires explicit distinct-instance adjudication. This is a review trigger, not proof of identity: the final semantic audit caught and rejected the proposed new130 addition because the inherited partial-130-add-lowest-front already labels that physical front. All four accepted additions have zero box overlap with every parent row.

[Independent reconstruction](../results/review/aperture-labels-20261114/independent-verification.json) · [Verification receipt](../results/review/aperture-labels-20261114/verification.json) · [PDF QA](../results/review/aperture-labels-20261114/atlas-qa.json).
