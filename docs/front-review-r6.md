# Round 6: every original photo, one physical definition

**Historical baseline, preserved before the follow-up.** The interrupted r6 notebook was not generated or run. [Round 7](front-review-r7.md) records the subsequent six marked examples, corrects a mistaken deletion, restores the user-confirmed original box and supplies the current notebook. The figures below describe r6 at its saved revision.

The complete pass inspected **all 157 original photographs and all 4,450 inherited r5 fronts**, including the 16 empty scenes. It changes **50 photographs**: **223 row replacements, one deletion and one addition**. The complete review still contains 4,450 fronts: **3,787 in training photographs and 663 in development photographs**. No training or model replacement was performed. **PR35 remains selected.**

This round uses the immutable r5 / PR71 row numbers. The [operations](../results/review/front-review-20261119-r6/operations.json), [current row maps](../results/review/front-review-20261119-r6/row-maps.json), [all-photo coverage](../results/review/front-review-20261119-r6/request-coverage.json) and [uncertainty ledger](../results/review/front-review-20261119-r6/review-status.json) record the decisions. The [review manifest](../data/reviews/fronts-2026-11-19-r6/manifest.json) binds the original pixels, labels and evidence by hash.

## The user's corner interpretation is supported

The original [baomai data README](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/README.md) identifies the two actual corners along the front's lower edge, ordered by image X. It explicitly distinguishes these from the upper lip or projecting support. A reliably locatable point uses visibility 2; an unavailable one is exactly `0 0 0`. The [source-definition record](../results/review/front-review-20261119-r6/source-definition.json) preserves the source revision and interpretation.

The qualification is **which physical surface owns the corner**. A farther gray tip may belong to the side or support, and an inner white transition may be a groove rather than the corner. Neither "outermost pixel" nor "brightest part" defines the landmark. Several supported corrections move inward; others move outward. There is no universal inset, outward shift or snap to the box.

The existing [annotation policy](annotation-policy.md) is unchanged. The box encloses the identifiable visible front and its own frontal rim. Rack, cast shadow, receding top plane, projecting support, fingers and neighboring cassettes do not set its outer extent. A sloping front can require background inside an axis-aligned rectangle. The rectangle can also span an occluder between two visible pieces of the same front. These unavoidable interior pixels differ from extending a box onto a rack merely to provide context.

The [labeling guide](front-labeling-guide.md) gives the working rules. The nine user screenshots are preserved with their [source-image and r5-row mapping](../results/review/front-review-20261119-r6/example-mapping.json). Their colored marks were proposals checked against originals, not automatically accepted coordinates.

## What changed

| Item | Result in this pass |
| --- | ---: |
| Original photographs inspected | 157 |
| Inherited fronts inspected | 4,450 |
| Photographs with changed labels | 50: 34 training, 16 development |
| Existing rows replaced | 223 |
| Box fields changed in replacements | 184 |
| Left-point fields changed in replacements | 17 |
| Right-point fields changed in replacements | 31 |
| Rows deleted / fronts added | 1 / 1 |
| Empty photographs retained | 16: 13 training, 3 development |

Field counts overlap because one replacement can change more than one field. Addition and deletion are counted separately.

Concrete decisions illustrate the distinction:

- **new003 BASE56:** deleted the continuous rack-rail false instance below the neighboring front. Its old row identity remains in the operation ledger and before/detail panel.
- **new056, below BASE15:** added a separately visible front with its own upper seam at the lower frame boundary. Its visible box is `[418, 704, 466, 720]` in original pixels. Neither lower corner is exposed, so both remain `0 0 0`.
- **new073 BASE10 right:** moved to the observed lower-front turn near `(290, 454)`, rather than the farther side projection. The box and left point are independently retained.
- **new002 BASE1-9 right:** follows lower-front terminal turns. Some box bottoms are also tightened where they reached the next front.
- **new010 BASE15:** recovers front material cut by the old top while excluding the neighboring front below. This development correction does not alter training validation.
- **new014 BASE36-38:** the left turns are supported. The tapered right-side material remains an explicit ownership question.
- **new095 and new096:** some right points move inward from a dark strip; other inset points move outward to the actual front turn. This directly challenges a single offset rule.
- **new015 BASE31 and new025 BASE44/48:** the final PDF comparison challenged an initially overconfident retention judgment. Fresh independent source checks supported bottom31 `436→434`, top44 `242→250`, and bottom48 `355→352`, respectively. These box-only changes remove rack or receding top surface; both photos retain their separate whole-photo holds.
- **new032 BASE3/9/24/31 and new130 BASE25/28/44/49:** box-only trims remove definite rack margins or receding top surface while retaining independently justified points.

Original camera PNGs are unchanged. Coordinates are nominal estimates on soft source contours; normalization precision is serialization, not a claim of subpixel physical certainty. Original examples guide the semantics but are not treated as perfect independent ground truth.

## What remains uncertain

There are **51 recorded cases across 49 photographs**. They are specific evidence limits, not a failure to inspect the photos. **44 are training photographs; five are development photographs.**

All **31 old-series training photographs** share an unresolved distinction: a broad bright band may be part of the front rim or a projecting support. The originals and source wording do not establish ownership confidently. Their inherited review geometry is preserved and flagged. A filter cannot supply that missing physical identification.

The other held training photographs are:

`new_000014`, `new_000015`, `new_000016`, `new_000017`, `new_000018`, `new_000020`, `new_000023`, `new_000025`, `new_000028`, `new_000030`, `new_000114`, `new_000122`, `new_000127`.

The five development photographs with unresolved cases are `new_000007`, `new_000008`, `new_000058`, `new_000059` and `new_000089`. The thin strip between fronts 13 and 14 in 058/059 has uncertain ownership; the independently visible top of 13 can still be corrected. In 017, 015, 020 and 089, isolated pale patches do not securely establish extra instances. Those patches are recorded without inventing new boxes from stack spacing.

For a physically identified front, unavailable corners remain ordinary missing slots. A held physical interpretation is different: its inherited geometry remains reviewable but is not newly certified. In the controlled experiment, **every whole held training photograph retains exact PR35 labels in both arms**, including any independently supported edits elsewhere on that photograph. No photo is removed or converted into an empty negative.

## One PDF, including every photograph

The separately delivered **cassette-front-review-r6.pdf** contains **177 pages**:

- A cover, physical-definition page and linked photo index.
- **157 complete-photo pages**, showing every current label, including unchanged and empty photos.
- **17 enlarged original / before / current comparisons** for consequential decisions and unresolved ownership.

The cover explains the colors. **Magenta marks only the box, left point or right point changed in this pass.** Cyan marks inherited fields; amber flags unresolved fields. Circle and square distinguish image-left and image-right points. Missing slots have no point marker. Deleted base rows are red dashed in before/detail panels and absent from the current labels. A changed field takes magenta priority, while the header and caption also disclose any unresolved case on that photo.

Full-photo pages use current r6 row numbers; operations use immutable r5 base rows. The row map connects them after the deletion and addition. The PDF's [payload audit](../results/review/front-review-20261119-r6/pdf-audit.json) verifies all current row identities and image/label hashes. The [rendered-page QA](../results/review/front-review-20261119-r6/pdf-visual-qa.json) records actual layout inspection.

All 157 originals received a visual pass. Selected difficult decisions received an independent cross-check; the [receipt](../results/review/front-review-20261119-r6/root-cross-review.json) names that subset rather than claiming a second review of every row. The [view catalog](../results/review/front-review-20261119-r6/view-catalog.json) records inspection-render hashes. Selected source crops are retained; other crop renders are reproducible displays of the preserved originals and labels, not new source images.

## The next notebook

The following is the archived planned [r6 protocol](../experiments/front-review-2026-11-19-r6.json). Its notebook was not generated before the interruption. Use the [current r7 notebook](../notebooks/baomai_front_review_r7.ipynb) and its separately pinned source/protocol for the next run.

| Input or decision | Reference arm | Candidate arm |
| --- | --- | --- |
| Training photos / empty training scenes | 137 / 13 | 137 / 13 |
| Training fronts | 3,771 | 3,779 |
| Supervision | Exact PR35 | Eligible r6 review, exact PR35 on all 44 held photos |
| Initial weights | Original epoch173 | Original epoch173, independently initialized |
| Numerical recipe | Seed46, 30 epochs, batch4, input1280, FP32, AdamW, lr0.0001 | Identical |
| Checkpoint selection | Final epoch | Final epoch |
| Training-validation labels | Original 20-photo / 664-front labels | Exactly the same bytes |

There are **79 training photos with different label bytes between arms**. The candidate's 3,779 training fronts differ from the complete review's 3,787 because held photos use PR35 supervision. This tests the accumulated eligible supervision package, including the disclosed holds. It does not isolate one correction, directly compare r5 versus r6, or continue a previous candidate's weights. The resource trial initializes neither full run.

Keep three evaluation identities distinct:

| Reference | Role |
| --- | --- |
| Fixed 666-front reference | Primary comparison, unchanged |
| Fixed historical 663-front reference | Required diagnostic comparison, unchanged |
| Current r6 663-front development review | Separately hashed, exploratory only |

The two 663 counts happen to be equal; **their label bytes and identities are different**. Neither reviewed development labels nor frozen scoring labels replace original training validation.

The [PR70 result](../results/evaluation/pr70-20261119/README.md) still supports retaining PR35: the r4 candidate gained some five-pixel corner pairs while worsening matched-front coverage and unmatched outputs. A scored miss can also be a box-extent disagreement at the fixed IoU threshold. Compare missed and extra fronts, duplicates, partial-front groups, whole-scene failures and corner success together, with eligible misses retained in denominators. Do not tune a threshold or reference after looking at the outcome to make the result favorable.

Repeatedly reused development scenes cannot establish reliability on a new camera session. Relabeling is a controlled hypothesis, not a guaranteed improvement. The next run does not resolve the held physical ambiguities and does not automatically promote a model.

## Reproduce the checks

```bash
python scripts/verify_front_review_r6.py
python scripts/check_front_review_r6.py --output /tmp/front-review-r6-verification
python scripts/build_front_review_r6_pdf.py \
  --output /path/to/cassette-front-review-r6.pdf \
  --audit-output /path/to/pdf-audit.json
```

The [verification artifacts](../results/verification/front-review-r6-20261119/) record actual two-arm staging, unchanged original validation and both fixed references, exact PR35 bytes on all 44 held train photos, r4/r5 regression verification, and rejection of a rehashed candidate-only alteration. The r6 verifier also checks all-photo visual coverage, valid uncertainty identities/regions and exact field-change metadata, so PDF coloring cannot silently diverge from the labels. The notebook's embedded input preflight is exercised separately without launching GPU training.

`release_ready` and `full_dataset_certified` remain false. Mechanical verification establishes that the saved files reproduce the recorded decisions; it does not establish physical certainty or model improvement.
