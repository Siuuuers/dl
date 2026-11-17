# Cassette label corrections and model replay: 18 November 2026

The user's newly reported rows are corrected in a fresh review version:
**34 existing rows refined and six visible fronts added** across the training
and separately versioned development sets. The requested PDF review before merge
remains outstanding. PR65 stays the review PR, stacked on PR63.

This pass also re-scores the retained PR35 model and the previously trained PR64
control/candidate on the revised development labels using their saved prediction
arrays. **PR35 remains selected.** No model has been trained on this new label
version, and no model is promoted by changing annotation files.

## Review files and exact scope

| Set | Photos | Instances | Locatable endpoints | Missing endpoints | Empty photos | Changes from its parent |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Training review | 137 | 3,798 | 6,860 | 736 | 13 | 26 row replacements, 3 additions; 9 photos |
| Development reference | 20 | 666 | 1,227 | 105 | 3 | 8 row replacements, 3 additions; 4 photos |

The [training manifest](../data/candidates/sequence-refined-fronts-2026-11-18/manifest.json)
contains every training label file, inheriting the exact 17 November bytes except
for the listed operations. The [development manifest](../data/references/sequence-refined-reference-2026-11-18/manifest.json)
inherits the 14 November completion reference. The original training-validation
labels remain 664 instances and are not replaced by this review reference.

- Training manifest SHA256: `fb52d6946e598e977289ca025ac386c5044e9f15e9461b6e55f7c00a6b9c9171`.
- Development manifest SHA256: `0de64be3797dce2e43d4cb98d09620eab969bb95725ed9e75682dcfcd53c7f96`.
- [Exact training operations](../results/review/sequence-refinement-20261118/accepted-review.json).
- [Exact development operations and hold resolutions](../results/review/sequence-refinement-20261118/development-reference-review.json).

The delivered focused PDF has **15 pages**: two introductory/index pages and 13
enlarged comparisons. The full atlas has **174 pages**, including a separate page
for every one of the 157 working photos. The actual photo pages display the new
saved label files. Original crops in the focused comparisons are lossless; full
photo pages use native-resolution JPEG review copies with vector annotations.

| Topic | Focused PDF page | Full atlas focus page | Full photo page |
| --- | ---: | ---: | ---: |
| new034 right corners, rows 23-26 | 3 | 5 | 44 |
| new066 rows 8/12 | 4 | 6 | 64 |
| new070 rows 3/5/7/9 | 5 | 7 | 68 |
| new100 rows 4/15 | 6 | 8 | 87 |
| new001 rows 17-24 | 7 | 9 | 18 |
| new002 rows 21-24 | 8 | 10 | 19 |
| new026 three missing fronts | 9 | 11 | 36 |
| new009 split row 56 | 10 | 12 | 157 |
| new010 central column | 11 | 13 | 158 |
| new030 marked lower face | 12 | 14 | 40 |
| new041 marked lower face | 13 | 15 | 51 |
| new086 row 57 | 14 | 16 | 167 |
| new088 row 57 | 15 | 17 | 169 |

## What changed and why

### Correct the physical corner independently of the box

In **new034 rows 23-26**, the previous right endpoints stopped inside the front
faces rather than at the visible outer lower-front junctions. The revised right
points are approximately `(891,128)`, `(893,160)`, `(890,191)`, and `(886,222)` in
the original 1280x720 image. Their left points remain supported. Each box is
independently tightened to the observed slanted front; it is not simply stretched
to accommodate a new point. These are the only four existing endpoint-coordinate
changes in the pass. No existing visibility state changes.

Neighboring033/035 views establish the four-front stack, but the revised point
coordinates come from034 itself. Rounded, soft native contours support manual
whole-pixel judgments rather than subpixel measurement guarantees.

### Some boxes were too large; others were too small

| Image and rows | Correction |
| --- | --- |
| new001 rows 21-24 | Remove excess left divider/rack area and material below each front. The old bottoms were about 131,166,205,237 px; new bottoms are108,150,190,228 px. Preserve the supported lower-corner points. |
| new001 rows 17-20 | Correct the same clear excess right/bottom margin in the adjacent stack. |
| new002 rows 21-24 | Apply the same front-face rule using002's own visible contours. |
| new070 row 3 | Expand upward to include the missing upper part of its front and trim the unnecessary lower margin. |
| new070 rows 5/7/9 | Tighten each partly hand-occluded face; retain individually supported right points and missing left points. |
| new100 rows 4/15 | Restore the omitted upper halves of two visible fronts. Keep the broad ribbed roof out and preserve the existing lower-corner points. |

The additional check is limited to clear analogous defects in these nearby
stacks. It is not a blanket shrink operation or a claim that every other cyan
box is now ideal. Equal physical cassette size also does not require equal pixel
height under perspective.

### Separate front ownership from illumination

In **new066**, clearer064/068 views and neighboring065/067 identify a four-front
middle column. Row 8 is its third front; row 12 is its fourth. Revised boxes are
`[617,409,779,446]` and `[678,444,730,484]`. The internal dark-to-light change near
y460 belongs within the fourth front under this sequence-supported
interpretation. Both true corner pairs remain hidden.

In **new026**, three separate visible fragments below row 48 were missing. They
are added as current rows 54-56:

| New row | Visible-front box in source pixels | Pose points |
| --- | --- | --- |
| 54 | `[691,471,721,488]` | Both missing |
| 55 | `[697,488,720,506]` | Both missing |
| 56 | `[703,506,715,521]` | Both missing |

Each has visible target-frame front material and a distinguishable transverse
boundary. The narrowing edges of the gripper opening are not cassette corners.

In **new009**, old row 56 combined two fronts and included excessive gripper
area. It is tightened to `[682,471,720,494]`, and new row 58 describes the lower
front at `[702,493,717,509]`. Both corner pairs remain missing.

In **new010**, rows 10-12 are corrected to their complete clear front faces.
Row 13 follows the upper shoulder/aperture face at `[594,430,760,462]`; row 14
follows only the next dark aperture face at `[668,463,747,496]`. Two further
visible stem fronts are added as row 33 `[695,497,727,533]` and row 34
`[696,534,729,558]`. The wider012 view directly exposes the continuation of this
same central column below its first five fronts. Neither added object's actual
corners are visible in010. The remaining010 row 30 question is a separate
bottom-clipped fragment.

### The orange marks exposed a mistake in the earlier explanation

For **new030 and new041**, the preceding face's bottom and the next face's top
do not have to coincide. The unobscured neighboring views show a molded
transition between successive front faces. The lower boxes previously began at
an estimate of the preceding lower edge instead of at the observed next-face
onset.

- new030 row 12 still ends near y474; lower row 30 is tightened to `[686,483,722,501]`.
- new041 row 9 still ends near y460; lower row 15 is tightened to `[685,472,719,491]`.

This explicitly supersedes the earlier assertion that030's change near483 was
merely an interior tonal change within the lower face. The user-assisted surface
review and neighboring views support the revised lower-face onset. The gap is
not forced into either front box, and no hidden pose point becomes visible.
The native boundary is soft by a few pixels; the orange stroke is an identifying
mark, not a subpixel measurement.

The four supplied marked crops map to030,010,030,041. Their exact bytes and hashes
are retained in the review evidence. None directly depicts009; that correction
uses the explicit two-front identification plus original009/011 evidence.
Rechecks of 042 row 16,049 row 27, and 074 row 20 did not support the same onset change.

### Keep the small real fronts in086 and 088

**Row 57 remains in both images and is now cyan.** The narrow pale wedge is a
distinct visible front inside the second bottom compartment, separated from
row 52 by the dark metal divider. The old large boxes extended under the thumb.
They are tightened to `[503,596,511,619]` in086 and `[505,592,514,616]` in088.
The supported left corner stays; the right remains missing.

The front/roof transition is soft, but the small target itself has sufficient
physical support. Size is not the reason to delete a real front. Frames 086-090
provide compartment context. There is an arrangement/view change between085 and
086, so consecutive filenames alone are not proof of the same arrangement.

## Remaining review questions

The targeted corrections above do not certify all other annotations. Training
still has **nine case groups on eight photos**, covering nine distinct rows.
The two034 groups concern the same row 15. The development reference has
**eight held rows on seven photos**, down from13 rows on nine photos.

| Training image | Current held rows | Full atlas page |
| --- | --- | ---: |
| new005 | 15 | 22 |
| new034 | 15 | 44 |
| new049 | 18 | 56 |
| new094 | 23 | 81 |
| new115 | 11 | 95 |
| new117 | 11 | 97 |
| new121 | 53 | 101 |
| new128 | 5,8 | 108 |

These are boundary ownership or corner questions at the image border. The
[current status ledger](../results/review/sequence-refinement-20261118/review-status.json)
records their individual evidence and limits. They are not removed just for
being small.

| Development image | Current held rows | Full atlas page |
| --- | --- | ---: |
| new007 | 7,14 | 155 |
| new008 | 36 | 156 |
| new009 | 8 | 157 |
| new010 | 30 | 158 |
| new011 | 15 | 159 |
| new057 | 1 | 164 |
| new059 | 1 | 166 |

## What the existing models do on the revised labels

The [replay package](../results/review/sequence-refinement-20261118/model-replay/README.md)
contains immutable saved predictions and a reproducible CPU-only scorer. The
historical 14 November completion-reference scores reproduce exactly: 3,951 scalar
checks, maximum difference zero. This historical reference is separate from
PR66's 12 November primary adjudication reference.

The following table scores every model against the same new 666-object reference,
with box confidence 0.13, NMS IoU 0.7, matching IoU 0.5, endpoint confidence 0.5, and
1280-pixel inference size as recorded in the saved predictions.

| Existing model | Matched targets | Unmatched targets | Unmatched proposals | Duplicate proposals | Both corners within 5 px /584 | Both within 10 px /584 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PR35, retained model | 648 | 18 | 80 | 32 | 533 | 582 |
| PR64 matched control | 646 | 20 | 83 | 29 | 531 | 583 |
| PR64 previous candidate | 642 | 24 | 104 | 30 | 532 | 583 |

All three produce zero outputs on the three empty photos. The prior candidate
has **four more unmatched targets and 21 more unmatched proposals than its
matched control**, with almost identical corner results. This still does not
support promotion. The `candidate` in this replay is the older model trained on
3,788 objects, not a model trained on the new 3,798-object review.

Fixed predictions change scores when reference boxes or instance counts change.
That is an annotation effect, not model learning. An unmatched target can be a
localization mismatch: new010 row 14 now has best box IoUs of approximately 0.251,
0.250,0.337 for PR35/control/candidate, below the fixed 0.5 association threshold.
All three also miss the newly added009 row 58 and 010 rows 33/34, and the retained
small086/088 row 57. The candidate's earlier010 row 31/32 misses remain; the new
aperture corrections do not explain those away.

Unmatched proposals have not all been individually adjudicated as background
false positives. These repeatedly used same-video development scenes also do not
establish performance on an independent camera session or an application-level
physical tolerance.

## Notebook decision

The new review is ready for the requested visual check; it is not a new trained
model or a fully certified dataset release. The old accepted-fronts notebook
already ran in PR64 and pins its own 3,788-object candidate and protocol. Changing
one notebook version string cannot select this complete review: the existing
training verifier enforces the older candidate contract.

After the PDF review, the smallest next controlled experiment can reuse that
recipe with a separately versioned candidate, protocol and notebook: exact PR35
control labels versus the latest accepted review on photos without a hold, with
exact PR35 labels in both arms on the eight held photos. Preserve all 137 training
photos and 13 negatives, independent initialization, identical settings, and the
original training-validation files. Score the resulting models on both the
corrected and historical development references.

This existing whole-photo rule has a concrete consequence:034's unrelated row 15
hold would also keep its newly corrected rows 23-26 out of the treatment. That
limitation must be disclosed if reused; a row-level experiment would need a
different, explicit selection contract. No such contract is silently introduced
in this label-review PR.

There is no reason to mistake an indefinitely polished atlas for model evidence.
The next useful GPU run should test a pinned, limited label hypothesis against a
matched control, followed by the same failure-case inspection. This pass has
identified those cases and made the labels and comparisons reproducible.

## Verification and reproduction

Independent reconstruction checks every output directly from the parent and
operation ledger, without importing the composers. All 157 original PNG and
original label hashes, all 157 current row maps, schema, canonical missing states,
clipping, point order and containment pass. All 3,769 unchanged training rows and
655 unchanged development rows retain exact parent bytes. The only four existing
point-coordinate changes are034's right points; no existing visibility state
changes.

See the [independent verification](../results/review/sequence-refinement-20261118/independent-verification.json),
[atlas audit](../results/review/sequence-refinement-20261118/atlas-audit.json),
[focused PDF audit](../results/review/sequence-refinement-20261118/digest-audit.json),
and [PDF QA report](../results/review/sequence-refinement-20261118/qa-report.json).

```bash
python scripts/compose_sequence_reference.py --review results/review/sequence-refinement-20261118/development-reference-review.json --verify-only
python scripts/compose_sequence_front_review.py --review results/review/sequence-refinement-20261118/accepted-review.json --verify-only
python scripts/replay_sequence_reference.py --root . --verify-only
```

The full atlas and focused digest are generated by `render_label_atlas.py` and
`render_label_review_digest.py`; their audit files bind the exact source manifests
and PDF hashes. The retained prediction ZIP allows model-score reproduction
after the original Actions artifact expires, without weights, network, training
or inference.
