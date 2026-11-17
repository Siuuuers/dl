# Video sequence label review — 17 November 2026

PR65 now contains one complete training-label review directory: [sequence-refined-fronts-2026-11-17](../data/candidates/sequence-refined-fronts-2026-11-17/README.md). It combines the full 14 November review with the supported corrections from adjacent video frames. The new atlas displays these corrected labels on the actual photo pages. The earlier two-file preview is retained as history.

## Result and scope

| Measure | Full 14 November review | Current sequence review |
| --- | ---: | ---: |
| Training photos | 137 | 137 |
| Empty training photos | 13 | 13 |
| Front instances | 3,791 | 3,795 |
| Locatable lower corners | 6,861 | 6,860 |
| Missing lower corners | 721 | 730 |
| Pending case groups after the two earlier wall/fixture closures | 24 | 9 |
| Training photos with pending questions | 21 | 8 |

This pass changes **17 photos**, with **20 row replacements, 9 additions, and 5 deletions**. One existing point in124 changes from locatable to missing: it was on the face above the gripper, while the actual lower corner is occluded. No hidden corner is filled from a neighboring frame. These counts describe the annotation revision, not measured model accuracy.

All137 training photos were screened in overview sheets. Changed cases were examined using original-resolution crops, physical tier correspondence, and neighboring original frames. This is not a fresh certification of every unchanged box and corner. Nominal boundaries in blurred crops have a few pixels of manual uncertainty;030's internal division has about4px uncertainty recorded separately from its stronger instance identity.

## Main physical decisions

The row numbers in this table refer to the **full 14 November review** unless a current number is stated. Stable identities in the manifest preserve the correspondence after deletions.

| Image | Decision |
| --- | --- |
| 030 | Split the old combined row12 into fourth/fifth central fronts. The nominal division follows tier context near474, not the darker tonal step near483. |
| 041 and042 | Add the missing fifth central front.042's front ends above the dark aperture tip, so that tip is excluded. |
| 044 | Refine row3 to include its upper aperture continuation; row4 already represents the next front. No extra instance. |
| 049 | Separate the upper/lower aperture fronts; retain and tighten clipped row21. Only the unrelated upper-left row18 remains held. |
| 062 | Resolve the lower patch as part of existing row9. No label-byte change is needed. |
| 074 | Separate third/fourth central fronts using075's exposed column. |
| 116 and117 | Delete row24: it duplicates lower molding over the hand, not a second front.116 was also explicitly confirmed by the user. |
| 123 | Retain and tighten row24 as the hand-occluded fourth front.122/124 show four cassettes in this slot. Extend row19 to include its own aperture continuation. |
| 124 | Existing19 owns the upper triangle. Add exactly one lower aperture front to restore the ten-front column; two additions would duplicate the upper instance. Mark19's unlocatable left corner missing. |
| 127 and128 | The two fronts identified as127 rows37/38 persist in128. Refine their visible bounds and the preceding upper-triangle owner. After deletion of the separate top-row duplicate, these two fronts are current rows36/37 in both frames. |
| 094 | Add the real fourth right-hand front fragment, mostly hidden by the hand. Keep the separate bottom-border question pending. |
| 001 | Adjacent views establish the tiny left fragment as cassette material behind the jaw. Retain one box spanning its two visible fragments and tighten excess. |
| 024,127,128 | Delete duplicate top-molding rows:024 current-baseline2,127/128 baseline1. |
| 033 | Retain the real clipped first front and tighten its rack-inclusive bounds. |
| 027,030,041 | Tighten clearly excessive cyan-box padding without imposing common image heights or changing valid corners. |

## What the box means

The target is the cassette's **visible front face**, not its whole body, roof, rack support, gripper, or an estimated hidden rectangle. A single axis-aligned box may contain an intervening occluder when two visible fragments belong to the same front. Its outer limits follow observed front material. Sloping adjacent fronts can therefore have overlapping rectangles without being duplicates.

Video continuity is useful for physical identity and surface ownership. It does not make an occluded target-frame corner visible, and it does not justify labeling a completely hidden cassette. Small fronts are not deleted by a height threshold. The contrast between117's molding and123's real thin fourth front is an explicit example.

The [annotation policy](annotation-policy.md) remains `baomai-visible-front-v1`: one class, two actual lower-front corners ordered by image x, and canonical `0 0 0` for unlocatable corners. Cyan means no recorded hold; amber means a recorded question remains. Neither colour is model confidence.

## Remaining review questions

There are **nine groups on eight photos**, corresponding to **nine distinct training rows**.034 has two questions about the same row. All are tiny frame-border ownership, extent, or corner questions; the previously held aperture cases are now resolved.

| Image | Current row(s) | Full atlas page | Remaining distinction |
| --- | --- | ---: | --- |
| new_000005 | 15 | 21 | Possible preceding front remnant versus molding; current extent/point is not certified. |
| new_000034 | 15 | 43 | Very small clipped front extent, plus whether the saved point is its actual corner. |
| new_000049 | 18 | 55 | Upper-left border remnant versus molding of the next front. |
| new_000094 | 23 | 80 | Bottom strip of the next front versus the preceding front's lower lip. |
| new_000115 | 11 | 94 | Top strip of a preceding front versus next-front molding. |
| new_000117 | 11 | 96 | Same top-border distinction; independent from the deleted lower row24. |
| new_000121 | 53 | 100 | Blurred top-border fragment ownership. |
| new_000128 | 5 and8 | 107 | Original rows6/9 remain ambiguous; deleting the separate left-column strip does not settle these other columns. |

Neighboring frames were checked for these cases too. Where the projected contour is within a few pixels of the border, context does not reliably decide which front pixels are actually in the target image. The ledger retains current questions and separately preserves the pre-sequence questions as history. Pending label rows are not silently treated as background.

The separate twenty-photo development reference remains unchanged:663 objects,1,227 locatable corners,99 missing corners,3 empty photos. It still has its own thirteen held rows on nine photos. The original in-training validation revision is a different set of labels with664 objects; this review does not substitute one for the other.

## Review files and reproducibility

The complete [operation ledger](../results/review/sequence-refinement-20261117/accepted-review.json) binds each deletion/replacement to an exact baseline row and records evidence frames. The [manifest](../data/candidates/sequence-refined-fronts-2026-11-17/manifest.json) contains all137 file hashes and stable row identities. The [current review status](../results/review/sequence-refinement-20261117/review-status.json) contains the narrowed holds. Five supporting visual-review records preserve the physical reasoning. The [atlas audit](../results/review/sequence-refinement-20261117/atlas-audit.json) maps every PDF photo page to its exact source image and label hashes.

The full PDF has173 pages:cover and guide, two linked index pages, twelve focused pages, and all157 working photos. Start with pages5–16. A short review copy contains the cover, guide, and these twelve focused pages; original full-atlas page numbers remain printed for cross-reference.

Reproduce the files and PDF with the commands in the candidate README. The composer verifies original image bytes, the frozen source and policy hashes, exact baseline rows, complete inventory, corner evidence, geometry, and deterministic output bytes. A second independent reconstruction matched every TXT file. PDF QA covers the cover/guide, all focused pages, and all17 changed-photo pages.

## Model and next notebook

The earlier accepted-fronts experiment is already completed in [PR64](https://github.com/Siuuuers/dl/pull/64). Its [PR66 result report](https://github.com/Siuuuers/dl/blob/d71303a2a3e2c6abed278fe91ba0f0a910fea63d/docs/accepted-fronts-results-2026-11-15.md) retains PR35. On the same663-object reference, PR35/control/candidate had20/22/24 misses and85/88/107 unmatched proposals. The last candidate did not meet promotion criteria. Unmatched proposals are association outcomes, not automatically confirmed background hallucinations.

This review changes labels, not model weights. Do not rerun the old `baomai_accepted_fronts.ipynb` expecting it to use this3795-object set: it is pinned to the earlier3788-object experiment. After the requested PDF review, merge the reviewed annotation work and specify the next controlled experiment against its exact candidate version, including treatment of the remaining ambiguous rows. This directory deliberately has no new training recipe.

The user's video-origin clarification also limits interpretation of evaluation: nearby train/validation frames are correlated, so the existing split is useful for development comparisons but does not establish performance on an independent recording session. Temporal-assisted labeling is appropriate here; source-frame identities and unchanged validation labels remain explicit. No new independent-session accuracy claim is made.
