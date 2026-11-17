# Front-boundary review: 18 November 2026, round 2

The requested refinement pass is saved as
[`sequence-refined-fronts-2026-11-18-r2`](../../../data/candidates/sequence-refined-fronts-2026-11-18-r2).
It changes **387 existing bounding boxes** and deletes **one user-adjudicated
row**, across **28 photos**. All retained keypoint coordinates and visibility
tokens are unchanged. The complete review now has **3,797 fronts in 137 photos**,
including all 13 empty training photos.

## What changed

| Scope | Box refinements | Decision |
| --- | ---: | --- |
| `001–006` | 213 | Refine supported front envelopes, including the named `001` rows 1, 26, 27, 49, 50 and 51. Preserve the previously corrected `001` rows 17–24 and `002` rows 21–24. |
| `028–035`, `040–041`, `043`, `050–051`, `073` | 158 | Correct excess neighboring-front/rack extent and recover clipped upper front material. Includes the bounded follow-up to row 1 in `031/034`. |
| `094–100` | 16 | Recover the full upper front/rim in `094–099` rows 4/15; remove excess roof/rail from `100` rows 53–56. |
| `049` row 18 | 0 | Delete the specific clipped row as the user confirmed. No other instance is removed. |

The exact old/new 11-field label lines and target-frame evidence are in
[`accepted-review.json`](accepted-review.json). The source proposal ledgers are
retained with their original hashes. Their temporary preview paths document
the review process; the original repository images and the final PDF audit are
the preserved visual sources. A clarification in the accepted ledger explains
that the unchanged clipped `050` rows 23/28 were not newly adjudicated; no
additional physical hold was established for them.

The `073` correction keeps four front instances in the central column. Its
interior dark-to-bright shadow transition does not create another front. In
`030` and `041`, the supported lower aperture fronts remain unchanged. The
upper boxes are reduced to definite aperture material, but ownership of the
pale shoulders outside the aperture remains unresolved, so those upper rows
are amber again.

## One focused PDF

The conversation deliverable is
`cassette-front-focused-review-2026-11-18-r2.pdf` (**46 pages**):

| Pages | Content |
| --- | --- |
| 1–2 | Scope, annotation convention and clickable page index |
| 3–30 | Before/current comparisons for all 28 photos changed in this pass |
| 31–37 | Every remaining development-reference amber row, with an original crop |
| 38–46 | Every remaining training amber row, with an original crop |

Useful first checks are `030` on page 11, `041` on page 18, the `049` deletion
on page 20, `073` on page 23, and `094–100` on pages 24–30. Only selected rows
are drawn on comparison pages; other annotations remain in the saved dataset.
The PDF shows annotations, not model predictions. Cyan means no recorded open
hold, not a correctness certificate.

The [PDF audit](focused-pdf-audit.json) binds every included original image,
label file, crop, row selection and generator hash. Recreate it from the repo:

```bash
python scripts/render_front_refinement_review.py --root . \
  --candidate data/candidates/sequence-refined-fronts-2026-11-18-r2/manifest.json \
  --output /tmp/cassette-front-focused-review-2026-11-18-r2.pdf \
  --audit /tmp/cassette-front-focused-review-audit.json
```

No new full atlas is produced for this pass.

## What the box means

The target remains the **visible cassette front face**, including its raised
front rim. The [annotation policy](../../../docs/annotation-policy.md) is
unchanged. A box encloses the visible front material; it does not deliberately
include the ribbed roof, wall, rack or gripper as target material. An axis-aligned
box around a tilted front, or around visible pieces separated by an occluder,
can necessarily contain background inside the rectangle.

The two pose points are the actual lower-front corners when locatable. They
are not the bounding-box corners and are not placed on the gripper aperture.
If a target-frame corner is unlocatable it stays `0 0 0`. Neighboring frames
help identify surfaces and seams; they cannot make a hidden corner visible.
Fully hidden cassettes do not require memory, tracking or a target annotation
for this model. Small visible fronts are retained when supported; no arbitrary
minimum-height deletion rule is used. Soft boundaries retain about 2–3 native
pixels of manual uncertainty.

## Remaining questions and the notebook

The [training status ledger](review-status.json) contains 10 case groups on
9 photos, covering 10 distinct current rows:

| Training photo | Current rows |
| --- | --- |
| `new_000005` | 15 |
| `new_000030` | 12 |
| `new_000034` | 15, recorded by two inherited case groups |
| `new_000041` | 9 |
| `new_000094` | 23 |
| `new_000115` | 11 |
| `new_000117` | 11 |
| `new_000121` | 53 |
| `new_000128` | 5 and 8 |

The unchanged development reference separately retains 8 held rows on
7 photos. Both sets of remaining rows appear in the one focused PDF.

The new controlled notebook uses a separate training candidate. It retains
the exact PR35-label control, applies this review on photos without holds, and
keeps **every held photo byte-identical between control and treatment**. This
allows a controlled run without silently certifying the remaining questions.
It also means supported changes on a held photo, including `034`'s corrected
endpoints from the previous pass, do not enter this treatment yet.

The complete review has **3,797 fronts**; the bounded treatment has **3,791**.
The original training-validation labels remain **664 fronts**. The current
development scoring reference is separately pinned at **666 fronts**, and the
historical 14 November reference remains a separate diagnostic. These sets
serve different purposes and should not be combined into one accuracy number.

Preparing the notebook is authorized. The draft still awaits the requested PDF
review before merge. PR35 remains the selected model, and no new trained model
or improvement claim is created by a label correction. The prior
[saved-prediction replay](../sequence-refinement-20261118/model-replay/README.md)
remains applicable because the development reference did not change. The
same-video development split does not establish performance on another capture
session.

## Verification

[`verification.json`](verification.json) records an independent reconstruction
of all 137 training files without importing the composer, project label parser
or renderer. It confirms all 157 original image and label hashes, exact before
lines, 387 bbox-only replacements, the single authorized deletion, all row
maps, counts, point geometry, and the current hold identities. The 109 untouched
files and 3,410 untouched retained training rows remain byte-identical.

```bash
python scripts/check_front_refinement_review.py --root .
python scripts/compose_sequence_front_review.py \
  --review results/review/front-refinement-20261118-r2/accepted-review.json \
  --verify-only
```

These checks establish faithful composition and provenance. They do not prove
physical perfection or model accuracy. The visual review inspected the changed
photos and representative native-pixel crops; unmodified annotations are
inherited, not newly certified.
