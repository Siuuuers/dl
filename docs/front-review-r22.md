# Completed r22 review and PR96 results

22 November 2026, Asia/Hong_Kong.

The interrupted session's completed r22 annotations, PDF receipts and PR96
evaluation are preserved in commit `c1d68bc6ed388077f238c09b74f6a8e15a1375a5`.
The next training entry is [r23](front-review-r23.md), which carries this review
forward with four additional front-face corner corrections.

The r22 review changed 357 existing rows across 66 photographs: 352 boxes and
60 point fields (58 relocations and two removals). It added or removed no
object rows. The complete review contains 141 nonempty photographs and 4,448
fronts. [Saved review](../data/reviews/fronts-2026-11-22-r22/manifest.json);
[amendment counts](../results/review/front-review-20261122-r22/amendment-summary.json).

The [completed PDF](https://chatgpt.com/library?file_id=libfile_fde3_f-17ab40c44db48191b0a46c315807fe42_fddf6d9928348191b4ce3e0bd87531d7&account_id=personal)
has four index pages and 141 photo pages, including unchanged and old-series
photographs. Its saved coordinates and source pixels were checked in the
completed session. [Delivery receipt](../results/review/front-review-20261122-r22/pdf-delivery.json).
This remains the r22 snapshot; the r23 comparison separately shows its four
later edits. No repeat of the completed full review or model inference was
needed to prepare the continuation.

[PR96's completed audit](../results/evaluation/pr96-r21-20261122/README.md)
evaluated the returned r21 models against their pre-return references. On the
fixed 666-front primary reference, PR35 has 648 matched targets, 18 unmatched
targets and 80 unmatched outputs; the r21 candidate has 631, 35 and 100.
The candidate's ranking changes on the separate evolving review reference.
PR35 remains selected. R22 labels do not rewrite those results.

The [r22 boundary policy](annotation-boundaries-r22.md) records the completed
colour-assisted contour review, the independent landmark changes and the
preserved training exclusions. The r23 notebook uses these prepared labels
plus its named corner corrections with the established paired recipe.
