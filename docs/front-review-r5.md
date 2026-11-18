# Round 5: physical corners, focused amendments and PR70

The 17 requested photographs have been reviewed against the original PNGs. This pass contains **135 row replacements, two explicit deletions and one new visible front**. Among replaced rows, **41 box groups, 84 left-point groups and 40 right-point groups** change. No training or model promotion is performed by this annotation pass.

The [coverage ledger](../results/review/front-review-20261119-r5/request-coverage.json) accounts for all **144 requested field/row decisions**: 140 changed, one reviewed without change, and three held at `new_000023:24-26`. The [operation ledger](../results/review/front-review-20261119-r5/operations.json) preserves exact before/after text and reasons. Requests always use immutable **r4 / PR69 base rows**.

## What was corrected

Several points were inside the face or on the wrong contour, and some boxes cut off visible material or included support. Perspective explains why a physical corner need not coincide with an axis-aligned rectangle corner, but it does not excuse those errors. The [labeling guide](front-labeling-guide.md) explains the existing target definition without changing the model schema.

| Case | Decision |
| --- | --- |
| `new_000014:11,12` | Delete the two specifically requested rows. Surviving base 13 becomes current 11. This closes the former base 11 hold through explicit user adjudication, not proof of absence or a general size filter. |
| `new_000014:13` | Refit the visible border fragment to `[1260,93,1280,140]`; its left point becomes missing because the proposed junction merges with the divider/support. Right remains missing. |
| `new_000016:40` | Use `[912,613,995,665]`. Independent wider-context review found its own rim and smooth front strip above the first proposed crop. Point tokens are preserved. |
| `new_000018:14` | Right becomes `0 0 0`: the side strip is visible but its lower junction disappears into the thumb. The front remains. |
| `new_000023:22` | Exact row retained after review. Its aperture patch and narrow side strip belong to one front; the enclosing rectangle spans gripper pixels. Both corners remain missing. |
| `new_000023:24-26` | Physical-landmark identity remains pending. Exact r4 rows remain amber for review; neither old inset points nor proposed farther tips are certified. |
| `new_000023:27` | Supported right-point correction to `(952,505)`. |
| `new_000025:55` | Requested right point removed; the front and its other field groups remain. |
| `new_000095` below base 19 | Add the directly visible front at `[628,415,770,447]`. Both lower outer corners are covered. New current 20 is the addition; former base 20 becomes current 21. |
| `new_000096` | Interpret “28 to 20” as inclusive 20-28. Also correct the same visible inset-point error on context rows 18,19,29,30, explicitly recorded as additional scope. |

All other named point and box requests are accounted for individually. Companion box repairs follow observed material, not automatic expansion around a moved point. Additions and deletions are applied after all edits are keyed to immutable base rows.

The complete training review has **3,788 fronts, 6,849 locatable corners and 727 missing corner slots** across 137 photographs, including the same 13 empty training scenes. The separate development review remains exactly **20 photos, 662 fronts, 1,226 locatable and 98 missing corner slots**, with its inherited two amber rows at `new_000007:7,14`.

## One unresolved physical-feature question

For `new_000023` base 24-26, the proposed farther gray tips are approximately 18-20 pixels from the inherited points. Independent review supports their association with the cassette, but cannot establish whether they are the intended lower-front corners or lateral/underbody/projecting material. This is a material landmark choice, not rounding.

The [comparison image](../results/review/front-review-20261119-r5/new_000023-landmark-question.png) and [independent record](../results/review/front-review-20261119-r5/independent-corner-review.json) preserve both alternatives. Row 27, also shown in that original comparison, was subsequently accepted at its observed rounded termination; the final PDF's proposal panel is limited to pending rows 24-26. The [root review](../results/review/front-review-20261119-r5/root-cross-review.json) reconciles the other difficult cases. Contrast cannot settle physical surface ownership.

The experiment therefore retains **the whole `new_000023` photograph's exact PR35 labels in both arms**. Supported r5 edits on that photo remain in the review PDF but do not enter only the treatment arm. This does not certify the legacy points; it keeps an unresolved supervision decision identical between arms.

## Focused PDF

The separately delivered `cassette-front-changes-r5-2026-11-19.pdf` has **30 pages**: a linked index, every label on the 17 changed photographs, a physical-rule illustration, ten focused original/before/current details, and the explicit landmark question.

| Display | Meaning |
| --- | --- |
| Cyan | Inherited row |
| Magenta and `*` | Row edited or added in this pass; exact fields are recorded in the ledger |
| Amber | Unresolved row retained for review |
| Circle / square | Image-left / image-right actual lower-front point |
| No marker | Unavailable corner; canonical `0 0 0` |
| Red dashed box in a before panel | Removed base row |
| Orange on the final question page | Unaccepted right-point alternatives for rows 24-26 |

Full-photo pages show **current rows**. The [row maps](../results/review/front-review-20261119-r5/row-maps.json) preserve the base identities. Original crops were checked beside optional grayscale zero-cutoff autocontrast views. No generated restoration or replacement image was used. Training PNGs and preprocessing are unchanged. Soft contours remain nominal estimates; a clean PDF or tiny serialization error does not certify physical precision.

## PR70 outcome

The [complete result audit](../results/evaluation/pr70-20261119/README.md) verifies the returned archive, both 30-epoch L4 runs, source/recipe identities and original data. It then freshly infers PR35, the returned control and candidate in one fixed CPU runtime.

| 666-front primary reference | PR35 | Fresh control | r4 candidate |
| --- | ---: | ---: | ---: |
| Scored misses | 18 | 20 | 30 |
| Unmatched outputs | 80 | 83 | 107 |
| Both corners within 5 px | 533/584 | 531/584 | 541/584 |
| Both corners within 10 px | 582/584 | 583/584 | 580/584 |
| Miss-free nonempty scenes | 6/17 | 4/17 | 3/17 |

The candidate gains ten five-pixel pair successes versus control, but loses ten matched fronts and adds 24 unmatched outputs. The required 663-front historical comparison also has worse matched coverage. These results do not support promotion; **PR35 remains selected**.

A scored miss is no match at IoU 0.50, not necessarily no proposal. The report separates high-confidence extent disagreements at 010:15 and 057:11 from a visually supported missed front and background keyboard/rack detections. The 584 pair denominator contains only the two-locatable-corner subset, with eligible misses retained. This is a measured tradeoff on reused scenes, not proof that every r4 change is worse or that PR35 is ready for unattended deployment.

## Next notebook

[Open the r5 notebook in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r5-20261119/notebooks/baomai_front_review_r5.ipynb), choose **Runtime → Change runtime type → GPU**, then **Runtime → Run all**. Return the ZIP, including unfavorable results. The final notebook pins its source commit and the [new protocol](../experiments/front-review-2026-11-19-r5.json).

| Item | Reference | Candidate |
| --- | --- | --- |
| Training photos / empty scenes | 137 / 13 | 137 / 13 |
| Training fronts | 3,771 | 3,788 |
| Locatable / missing corners | 6,841 / 701 | 6,849 / 727 |
| Supervision | Exact PR35 labels | Accepted r5 review; exact PR35 on held 023 |
| Initialization | Original epoch173 | Original epoch173, independently |
| Recipe | Seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW, lr 0.0001 | Identical |
| Selected checkpoint | Final epoch | Final epoch |

There are **94 photographs with different label bytes** between arms. Complete review and candidate happen to have equal aggregate counts; the held photo still uses different legacy bytes. This tests the accumulated eligible supervision package, including the disclosed hold change from 014 to 023. It does not isolate one point edit or continue PR70's candidate.

The four-batch resource trial initializes neither full run. The notebook verifies and stages inputs, checks identities, backs up completed bundles to Drive and packages one ZIP. Original training validation stays at 664 fronts; fixed 666/663 scoring references stay separate. The 662-front review remains exploratory. No threshold, seed, epoch or NMS search and no automatic promotion are included.

## Verification and reproduction

The r4/r5 verification logic shares one configuration-driven helper. Each round retains independently pinned parent, review, candidate and protocol identities. Historical r4 artifacts remain byte-identical. Both wrappers and the shared helper are included in training and staging code hashes.

The [verification records](../results/verification/front-review-r5-20261119/) include actual two-arm staging, exact row-map checks and rejection of a candidate-only coordinate edit even after its file and protocol hashes are updated. Historical r4 tests preserve held-photo, reference-substitution and no-promotion protections. The notebook's embedded input preflight ran in an isolated source copy with all original images and epoch173. These checks establish reproducibility, not physical annotation certainty.

```bash
python scripts/verify_front_review_r5.py
python scripts/check_front_review_r5.py
python scripts/build_front_review_r5_pdf.py \
  --output /path/to/cassette-front-changes-r5-2026-11-19.pdf \
  --audit-output /path/to/pdf-audit.json
```

The PDF renderer uses Pillow, ReportLab and DejaVu Sans, drawing verified geometry without choosing coordinates. Its recorded hash identifies the delivered artifact; regenerated metadata may change bytes without changing geometry. The PR70 report separately provides a saved-prediction replay requiring no GPU or model training.
