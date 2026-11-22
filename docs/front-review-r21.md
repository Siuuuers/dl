# Round 21: visible cassette contours and the next notebook

22 November 2026, Asia/Hong_Kong.

**The r21 notebook is the new training entry. PR35 remains the selected model.** This amendment corrects both remaining rack margins and clipped cassette material. It changes **261 boxes in 35 photos**, with **56 separately reviewed endpoint-X repairs** where an inherited point itself lay on the rack. The immutable r20 annotations, notebook and PR94 return remain available for reproduction.

## Run the notebook

[Open r21 in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/front-review-r21-20261122/notebooks/baomai_front_review_r21.ipynb), or download [baomai_front_review_r21.ipynb](../notebooks/baomai_front_review_r21.ipynb).

1. Choose a GPU runtime.
2. Select **Run all**, connecting Drive when offered.
3. Let input verification, the four-batch resource trial, both training arms and packaging finish.
4. Return **baomai-front-review-r21-results.zip**. Verified completed bundles and the ZIP are also retained in the notebook's Drive folder when connected.

The notebook pins training source `c4fb76afe280552ed13ed3c9b78608deceb7b3c4` and the exact r21 protocol hash. [Notebook verification](../results/review/front-review-20261122-r21/notebook-verification.json) records the delivered notebook and pins. The resource trial never initializes a full run. Completed arms are reused only after their source, protocol, staged inputs, checkpoint and runtime receipts pass verification.

## The corrected object definition

Enclose this cassette's visible front, its own molded rim, and its visible rounded or sloping side contour. Include identifiable visible patches through an opening when they belong to the same front. Exclude an adjoining rack rail or a neighboring cassette from unsupported outer margins. Do not infer the hidden three-dimensional cassette extent.

The lower points are local front-corner landmarks. The horizontal silhouette extrema can occur higher up, so neither exact endpoint equality nor a fixed point-to-box margin is a valid universal rule. The accepted examples **094:31–32** remain byte-identical; earlier standards **086:3 and 087:5** also remain unchanged.

R20's shrink-only rule could not repair a box that was already too narrow. Its saved results also show mistakes in distinguishing a cassette's own contour from lower points, continuous rack strips and the wider front immediately above. R21 permits both inward and outward corrections. Original-pixel review sets the geometry; mechanical validity alone does not establish material ownership. See the [boundary policy](annotation-boundaries-r21.md).

| Saved example | r20 bound | r21 bound | Physical reason |
| --- | ---: | ---: | --- |
| 084:20 xmax | 516.91 | 514.50 | Remove the external separator margin |
| 086:24 xmax | 844.74 | 838.50 | Exclude the continuous rack; right-point X separately changes 842.61 to 837.00 |
| 087:3 xmax | 522.10 | 526.00 | Restore the projecting cassette wall |
| 087:53 xmax | 827.58 | 835.00 | Restore the rounded right wall |
| 089:39 xmin | 581.47 | 578.00 | Restore the visible left contour |
| 090:44 xmax | 788.28 | 798.00 | Restore the broad rounded right wall |

These are original-image pixel coordinates. Decimal serialization preserves the chosen values; it does not imply subpixel physical certainty in blurred images. A rectangular box can still contain some background around a sloping contour or between visible patches separated by an occluder.

## Scope and exact changes

All **40 supplied photos / 1,715 current rows** in 081–130 were reviewed. IDs 091–093 and 104–110 are absent from the prepared dataset. The corrections include **all 46 named edges**, with the requested direction verified in [request coverage](../results/review/front-review-20261122-r21/request-coverage.json).

| Amendment | Training split | Development split | Total |
| --- | ---: | ---: | ---: |
| Changed existing box rows | 137 | 124 | 261 |
| Separately reviewed point-X repairs | 26 | 30 | 56 |

There are **112 inward edge corrections and 169 outward restorations** across those boxes. The 56 point repairs consist of 9 left and 47 right points. Every Y coordinate, visibility value, class, instance count, row identity and point token outside these explicit X repairs is preserved. No objects are added or removed.

Photos **081, 116, 117, 122 and 125** retain their complete r20 labels. All 117 photos outside this review range also retain their r20 labels. Source fingerprints verify **606 original, r20 and reference files unchanged**. [Exact operations](../results/review/front-review-20261122-r21/operations.json), [visual audit](../results/review/front-review-20261122-r21/horizontal-audit.json), [review verification](../results/review/front-review-20261122-r21/verification.json) and [source preservation](../results/review/front-review-20261122-r21/source-preservation.json) retain the evidence.

Three limited observations remain explicitly recorded: glare prevents a defensible new right-corner localization on 083:1–2; 086:15's small proposed shift was rejected at a soft boundary; and 100:48's right glare may involve Y as well as X, so only its independently supported left-box restoration was applied. These retain accepted coordinates rather than inventing new ones. They are not new hard holds, and the full dataset is not certified.

## Saved annotations and PDF

The [complete r21 review](../data/reviews/fronts-2026-11-22-r21/manifest.json) contains all 157 label files. The [training candidate](../data/candidates/front-review-2026-11-22-r21/manifest.json) explicitly chooses eligible supervision.

The PDF has **36 pages: one linked index page and 35 changed photos**, showing all **1,531 current row tags** on those photos. Magenta marks changed fields; cyan marks retained fields. Changed point markers and boxes are tracked separately. The renderer preserves annotation coordinates and original image pixels; moving a number tag for readability does not move the annotation. [PDF delivery](../results/review/front-review-20261122-r21/pdf-delivery.json), [renderer QA](../results/review/front-review-20261122-r21/pdf-qa.json) and [independent PDF inspection](../results/review/front-review-20261122-r21/pdf-visual-review.json) identify the artifact and checks.

## PR94: valid return, reference-sensitive results

The [PR94 audit](../results/evaluation/pr94-r20-20261122/README.md) verifies the exact returned ZIP, source, notebook, protocol, two completed 30-epoch L4 runs and both checkpoints. All three models were measured with the same fixed inference settings. No r21 annotations entered that evaluation.

| Scoring reference | Model | Matched fronts | Unmatched targets | Unmatched outputs | Both corners within 10 px |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen 666 primary | PR35 | 648 | 18 | 80 | 582/584 |
| Frozen 666 primary | Fresh r20 control | 646 | 20 | 83 | 583/584 |
| Frozen 666 primary | r20 candidate | 631 | 35 | 103 | 574/584 |
| Pre-return r20 / 662 exploratory | PR35 | 625 | 37 | 103 | 573/584 |
| Pre-return r20 / 662 exploratory | Fresh r20 control | 624 | 38 | 105 | 573/584 |
| Pre-return r20 / 662 exploratory | r20 candidate | 648 | 14 | 86 | 583/584 |

The candidate's ranking reverses. Exact identity tracing shows that **16 of its 35 frozen-primary misses become matches for the same prediction** solely because reference-box overlap crosses the 0.5 matching threshold. Six primary identities were deleted in the intervening review; 13 retained identities remain unmatched. One former match becomes unmatched and two added identities match. Original-pixel diagnostics also show residual rack spill despite some favorable matches.

Thus an unmatched target is not automatically a physically absent detection, and an unmatched output is not automatically a confirmed background false positive. The fixed comparison does not establish a replacement for PR35. The historical 663-front results, both 5-pixel and 10-pixel point metrics, duplicated-output counts and representative images are preserved in the audit. All three models produced no outputs on the three empty development scenes; that small result does not certify unseen scenes.

## Training and validation

The established two-point model and paired recipe remain: both arms independently start from original epoch 173, with seed 46, 30 epochs, batch 4, input 1280, FP32, AdamW and initial learning rate 0.0001. Use final-epoch checkpoints and the same runtime for both arms.

| Actual training input | PR35 reference | r21 candidate |
| --- | ---: | ---: |
| Images | 137 | 137 |
| Empty scenes | 13 | 13 |
| Front instances | 3,771 | 3,784 |
| Locatable endpoint slots | 6,841 | 6,838 |
| Missing endpoint slots | 701 | 730 |

Exactly **92 training-label files differ** between arms. Photo 015 and the 31 old-series photos retain exact PR35 supervision in both arms. The complete review has 3,786 training fronts; the eligible candidate has 3,784 because the inherited photo-015 supervision remains.

Photos 086–090 are development images. Their amended labels improve the versioned review and its exploratory scoring reference; they are not training inputs. Original trainer validation remains **20 photos / 664 fronts**. The frozen **666-front primary** and **663-front historical** references remain unchanged. The r21 development review has **662 fronts** and is reported separately.

The notebook's actual embedded input preflight passed, and staging copied and independently verified **314 images and 314 labels** across both arms. [Preflight](../results/review/front-review-20261122-r21/input-preflight.json) and [staging readback](../results/review/front-review-20261122-r21/staging-verification.json) record the checks. GPU execution, package installation and Drive behavior for this new notebook remain to be exercised in Colab; preparation itself does not train or promote a model.

The next run compares accumulated eligible r21 supervision against PR35 supervision. It does not isolate this round's causal contribution. Reused development scenes and repeatedly adjusted labels do not replace a fresh capture session with labels frozen before evaluating the next model.
