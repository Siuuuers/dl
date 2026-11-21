# Round 14 labels, changed-photo PDF and the next paired experiment

21 November 2026, Asia/Hong_Kong.

**The amendments are saved as actual YOLO labels.** This pass inspected all 33 requested photographs and changed **233 existing rows in 30 photographs**: 213 box groups, 24 left-point groups and 33 right-point groups. A row can change more than one group. No cassette instance was added or deleted. The [complete review](../data/reviews/fronts-2026-11-21-r14/manifest.json) still contains all 157 label files; the [amendment summary](../results/review/front-review-20261121-r14/amendment-summary.json) binds the counts and exact changed-photo inventory.

**PR35 remains the selected model.** The returned r13 candidate improves alignment with the reviewed target but regresses on the frozen primary reference. The new notebook tests the r14 amendments together with the newly approved training photos, using the existing architecture and matched training recipe.

## What PR84 actually measured

[PR84](https://github.com/Siuuuers/dl/pull/84), at head [`e881e38b63b8299dce22bae6b472134c131c1ff7`](https://github.com/Siuuuers/dl/tree/e881e38b63b8299dce22bae6b472134c131c1ff7), contains the completed r13 return. The archive, transport parts, both checkpoint bundles, every recorded training-source/input hash and original epoch 173 initialization passed the [return audit](../results/evaluation/pr84-r13-20261121/verified-return.json). Both arms independently completed 30 epochs, 1,050 batches and 78 optimizer steps on NVIDIA L4. The source used for training was [`5c2817f1c1eaa1bbce7060facb82732c3d249e10`](https://github.com/Siuuuers/dl/tree/5c2817f1c1eaa1bbce7060facb82732c3d249e10), not this later r14 amendment.

PR35, the fresh control and the returned candidate were freshly evaluated in one CPU environment at the same fixed operating point. These are matching outcomes, not independent physical counts of cassette presence or absence. The [full measured report](../results/evaluation/pr84-r13-20261121/README.md) preserves predictions, receipts, reference identities, reproducible scoring and six bounded physical case inspections.

| Scoring reference | Model | Misses | Unmatched outputs | Both corners within 5 px | Both within 10 px |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen primary, 666 fronts | PR35 | 18 | 80 | 533/584 | 582/584 |
| Frozen primary, 666 fronts | Fresh r13 control | 20 | 83 | 531/584 | 583/584 |
| Frozen primary, 666 fronts | r13 candidate | 28 | 98 | 529/584 | 581/584 |
| Historical reference, 663 fronts | PR35 | 15 | 80 | 533/584 | 582/584 |
| Historical reference, 663 fronts | Fresh r13 control | 17 | 83 | 531/584 | 583/584 |
| Historical reference, 663 fronts | r13 candidate | 24 | 97 | 529/584 | 581/584 |
| Pre-return r13 review, 662 fronts | PR35 | 36 | 102 | 534/584 | 573/584 |
| Pre-return r13 review, 662 fronts | Fresh r13 control | 41 | 108 | 530/584 | 571/584 |
| Pre-return r13 review, 662 fronts | r13 candidate | 20 | 94 | 541/584 | 581/584 |

On the frozen primary reference, the candidate has eight more misses and 15 more unmatched outputs than the fresh control. On the separately pinned pre-return r13 review, it has 21 fewer misses and 14 fewer unmatched outputs. Both tables use the same prediction arrays and thresholds. This disagreement supports continuing the physical annotation work; it does not justify selecting whichever reference gives the more favorable result.

The previous PR82 candidate, replayed on the same r13 review, has the same matched identities and the same aggregate 5/10 px pair-success totals as the PR84 candidate. Unmatched outputs rise from 92 to 94, with duplicates rising from 29 to 31. One 5 px success identity is gained and another lost. The small r13 amendment therefore has not established an additional model improvement. Tiny numerical differences also exist between freshly repeated baseline arrays and the prior saved arrays; they do not change baseline match or 5/10 px success identities. Exact comparisons are preserved in the [prediction-consistency record](../results/evaluation/pr84-r13-20261121/prediction-consistency.json).

## Saved annotations and physical interpretation

Use [the r14 review labels](../data/reviews/fronts-2026-11-21-r14/labels/) for the current annotation package. The original `data/labels/` directory remains the preserved source dataset. The notebook selects the separately composed, eligible r14 training candidate explicitly.

| Changed cohort | Photographs | Amended rows |
| --- | ---: | ---: |
| Training review | 22 | 171 |
| Development review | 8 | 62 |
| Total | 30 | 233 |

All 124 unrequested review files, including all 31 old-series files, retain exact r13 bytes. Requested photos 012, 024 and 026 were inspected and remain unchanged. The [request ledger](../results/review/front-review-20261121-r14/request-coverage.json) records 122 grouped dispositions: 54 applied, 52 interpreted, 15 unsupported as written and one already satisfied. Every changed row has exact before/after bytes in the [operations](../results/review/front-review-20261121-r14/operations.json); 29 changed proposals also have separately bound physical reviews.

The [boundary policy](annotation-boundaries-r14.md) keeps a visible frontal rim, the bounding box and the actual lower-front corner distinct. Image Y increases downward. Cross-row references use immutable r13 anchors, preserving the target landmark's own tier and the coordinate not being changed. Ambiguous physical-top/bottom wording is translated explicitly where the visible seam supports it.

Some requested equalities would move a corner onto the rack, clip a visible rim or include an adjacent cassette. Representative retained decisions are:

| Request | Pixel-based decision |
| --- | --- |
| 001 row49 right point to xmax | Keep the actual lower-front turn at X1016.02; the box extends to X1021 to include projecting material. |
| 001 row50 right point to xmax; xmin to row46 xmax | Keep right X1004 and xmin887. X1015 is on the projecting side; xmin884 enters the dark intercolumn strip. |
| 003 rows1/3 left point and xmin to row2 left point | Keep row1's left fields and both original left box extents; only row3's left point moves to X424 at its own Y. A common xmin424 would clip visible rims. |
| 002 row45 add a left point at the specified cross-row coordinate | Keep the left point missing. The proposed coordinate is on an occluding/rack sliver, while the separate supported box trim is saved. |
| 012 row3 xmax to row8 right point | Keep row3's own extent; the requested anchor belongs to another column and would enclose an adjacent cassette. |
| 024/026 row27 xmax and right point to row26 xmax | Keep each target's tapered edge. The upper-row X lies in the dark rack at the target's own lower-corner Y. |

The full ledger contains the remaining interpretations and retained clauses. These decisions concern visible physical evidence; an unapplied wording clause does not automatically create a new whole-photo hold.

## What “all amber are correct” changes

The [approval ledger](../results/review/front-review-20261121-r14/user-approval.json) binds the user's statement to the exact prior PDF and case records. It resolves **17 previously drawn amber cases across 15 photos**: 14 training photos and development photo089. The 14 training photos now become eligible for reviewed supervision. This is a recorded user confirmation of the displayed annotations, separate from this pass's fresh visual inspection of the 33 requested photos.

**Photo015 remains the sole held training photo.** Its unresolved case asks whether an additional front exists and has no drawn row (`current_rows: []`). Approval of displayed amber labels does not answer whether that unboxed object is absent. Its supported amendments are nevertheless saved in the complete r14 review and shown in the PDF; its training supervision remains exact PR35 in both arms. The current development case ledger has no unresolved cases. The frozen evaluation references and their historical uncertainty records remain unchanged.

## Changed-photo PDF

The new PDF contains **exactly the 30 nonempty new-series photos whose annotation bytes changed in this pass**, displaying all 1,118 current objects. One clickable index page brings the total to 31 pages. Each included photograph shows all current rows. Magenta marks changed fields and cyan marks retained fields; an unchanged field is not a new physical certification. Photos 012, 024 and 026, all old-series photos and empty-label photos are absent from this PDF. The complete annotation package still contains all 157 files.

The exact photo inventory is pinned by the review manifest. [PDF delivery and validation](../results/review/front-review-20261121-r14/pdf-delivery.json) records the finished artifact and rendering checks.

## Separate post-return r14 diagnostic

Eight development label files change after PR84 returns: 007, 008, 009, 010, 053, 054, 055 and 059. The same cached PR35/control/r13-candidate predictions were therefore replayed against the new r14 review, separately from the completed pre-return comparison. **This adds no inference or training and cannot establish model improvement.**

| Existing model on the post-return r14 review | Misses | Unmatched outputs | Both corners within 5 px | Both within 10 px |
| --- | ---: | ---: | ---: | ---: |
| PR35 | 34 | 100 | 540/584 | 575/584 |
| Fresh r13 control | 38 | 105 | 535/584 | 574/584 |
| r13 candidate | 19 | 93 | 543/584 | 582/584 |

Compared with those models' pre-return r13 scores, their match counts increase by two, three and one respectively, all through the changed photo010 reference. Endpoint-success identities also change, including gains and losses. The model bytes and predictions did not change; the scoring target did. This [post-return diagnostic](../results/evaluation/pr84-r13-20261121/post-r14-diagnostic.json), with its [reproduction script](../results/evaluation/pr84-r13-20261121/measure_post_r14.py), establishes an explicit baseline for future r14 comparisons and never replaces the frozen primary result.

## Run the r14 notebook

[Open r14 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r14-20261121/notebooks/baomai_front_review_r14.ipynb).

1. Choose **Runtime → Change runtime type → GPU**.
2. Choose **Runtime → Run all** and connect Drive when requested.
3. Let both independently initialized arms finish. The fixed recipe is seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW and learning rate 0.0001, with final-epoch selection.
4. Return **`baomai-front-review-r14-results.zip`**. Verified completed arms and the ZIP are backed up to Drive; compatible completed work can be reused.

| Actual training input | Reference | Candidate |
| --- | ---: | ---: |
| Photographs | 137 | 137 |
| Empty scenes | 13 | 13 |
| Fronts | 3,771 | 3,787 |
| Locatable endpoint labels | 6,841 | 6,838 |
| Missing endpoint slots | 701 | 736 |
| Original validation photographs/fronts | 20/664 | 20/664 |

Exactly **92 training label files differ between the arms**. The complete review has 3,789 training fronts, while the actual candidate has 3,787 because the declared retained cohorts use their exact PR35 labels. The [materialized input verification](../results/review/front-review-20261121-r14/verification.json) checks actual files, hashes, counts and the separate held/old-series rules. The notebook's exact embedded input preflight has also run successfully against all 157 original PNGs and the epoch173 checkpoint. Its seven code cells compile; source-commit handoff validation is recorded separately in [notebook verification](../results/review/front-review-20261121-r14/notebook-verification.json).

The experiment changes **annotations and training eligibility together**. A later improvement would measure their combined effect; it would not identify which part caused it. Both arms keep the same architecture, initialization, seed, duration, image size and optimizer. No r14 GPU training or model replacement occurs during preparation.

## Where to improve next

The immediate useful experiment is this controlled supervision update. PR84's primary-reference misses remain concentrated in fronts with zero or one locatable corner: 26 of its 28 misses are in those groups. However, some scoring losses are box-overlap disagreements rather than absent proposals, and the reviewed-reference result has the opposite direction. Inspecting these identities is more informative at this stage than changing architecture or searching confidence/NMS thresholds.

For a later data collection, reserve whole new camera sessions before training, with deliberate coverage of partial fronts, hand/gripper occlusion and rack-only regions. That would address the largest remaining evidence gap: every current comparison reuses development scenes. Image-pixel corner success alone does not establish millimetre positioning accuracy or grasp success.

## Reproduce or edit

```bash
python3 -B scripts/verify_front_review_r14.py
python3 -B scripts/check_front_review_r14.py --output /path/to/fresh-r14-check
python3 -B scripts/build_front_review_r14_pdf.py --output /path/to/r14-changed-photos.pdf
python3 -B results/evaluation/pr84-r13-20261121/measure_post_r14.py \
  --root . --output /path/to/fresh-post-r14-diagnostic.json
```

For manual editing, open a fresh working folder containing `data/images/` copied as `images/` and the r14 review's `labels/` in the [HTML labeler](../tools/cassette_labeler/README.md). Preserve the pinned review and collect later corrections in a new revision.
