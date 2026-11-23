# Improving the model after R24, using the data already available

**The latest model learned different weights, but the training settings did not change. R24 made a small box-matching improvement over R23 on the same accepted R24 labels, with mixed corner results; it did not establish a reliable replacement for PR35. We can still improve the project with the existing images. The R25 annotation amendments are complete and its matched training notebook is ready.**

This report explains the completed PR99 result and separates it from the new R25 work. The [complete failure ledger](../results/review/front-review-20261123-r25/model-failure-inventory.md) lists every missed R24 target, every failed two-corner pair at 5 px, the additional one-corner errors, and all duplicate outputs. Its [machine-readable record](../results/review/front-review-20261123-r25/model-failure-inventory.json) retains exact coordinates, confidence values, source hashes and verification. It uses saved predictions; no additional inference or training was needed.

## What PR35 improved over the original baomai model

PR35 preserves the original one-class, two-corner model architecture. Its initializer is the original epoch173 checkpoint from historical [baomai revision `351eb3cf`](https://github.com/an94-doll/embedding_vision/tree/351eb3cf960ccc34829e2bc72e1fa4645acc6295), SHA-256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e`. The saved `PR35_vs_original_baomai_report.md` records this same-threshold comparison on its main 2026-11-12 development reference: 663 fronts, including 584 eligible for two-corner scoring.

| Confidence 0.13; identical reference | Original epoch173 | PR35 |
| --- | ---: | ---: |
| Matched fronts / 663 | 642 | 643 |
| Missed fronts | 21 | 20 |
| Unmatched outputs | 86 | 85 |
| Duplicate outputs | 27 | 31 |
| Both corners within 5 px / 584 | 529 | 529 |
| Both corners within 10 px / 584 | 575 | 577 |
| Outputs on three empty images | 0 | 0 |

This is a modest, mixed improvement: one fewer miss, one fewer unmatched output and two additional ten-pixel pairs, with unchanged five-pixel success and four more duplicates. The [completed proposal-filter review](proposal-filter-review-2026-11-11.md) preserves the PR35 results and the reason for choosing 0.13 for assisted review. Lowering a confidence threshold changes the returned proposals; it does not improve the learned weights. This historical 663-front reference must not be combined with later 663-, 666- or 662-front reference versions merely because their counts are similar.

The [seed46 training record](training-annotations-seed46.md#fixed-inputs-and-recipe) establishes the intervention: 30 additional epochs from epoch173, seed 46, batch 4, input 1280, FP32, rectangular loading and AdamW at initial learning rate 0.0001, with final-epoch checkpoint selection. The eight changed training label files contain five added fronts, six deletions and three retained-row modifications. Both arms use the same 137 training images and original 20-image trainer validation. The [returned-input audit](label-refinement-2026-11-14.md#1-what-the-selected-model-actually-learned-from) records 3,771 objects and 6,841 locatable endpoint slots for PR35, versus 3,772 objects and 6,839 slots for the original-label control. Later accepted label repairs were not retroactively included in PR35's weights.

The separate [original seed46 matched-control study](annotation-confirmation-2026-11-11.md#primary-comparison) uses **confidence 0.25 and the 661-front PR30 reference**, so its counts stay in their own comparison:

| Historical seed46 study | Original epoch173 | Original-label control | PR35 candidate |
| --- | ---: | ---: | ---: |
| Missed fronts | 27 | 24 | 24 |
| Unmatched outputs | 54 | 58 | 52 |
| Both corners within 5 px / 584 | 528 | 527 | 529 |
| Both corners within 10 px / 584 | 575 | 578 | 577 |

The control also recovers three baseline misses after additional training. Therefore the baseline-relative improvement cannot all be assigned to the label edits: within the matched pair, misses tie, extras decrease by six, five-pixel pairs increase by two and ten-pixel pairs decrease by one. The [receipts and PR35 archive](annotation-confirmation-2026-11-11.md#return-and-execution-verification) identify A100 training; a later L4 session restored and packaged those completed runs. Cross-round differences do not establish the GPU, seed or learning rate as the cause of a model advantage.

Finally, “selected” is the current evaluation decision, not proof that PR35 is physically best under every accepted reference. The [completed PR99 comparison](../results/evaluation/pr99-r24-20261123/README.md) favors PR35 on the frozen 666-front reference but favors R24 on the accepted R24 review, with predictions and scoring settings held fixed. Section 2 preserves both rankings and the reference-geometry explanation. Accepted corrections are not invalidated by an older benchmark's ranking, and repeated use of these development photographs does not establish performance on a new camera session.

## 1. What “new parameters” means in this experiment

There are two different things that people call parameters:

| Meaning | What happened from R23 to R24 |
| --- | --- |
| **Learned weights:** the numbers the network changes while fitting examples | These changed because some training corner coordinates changed. The returned R23 and R24 candidate checkpoints are different. |
| **Training settings:** learning rate, optimizer, epochs, input size, initialization, seed and related choices | These were held fixed: AdamW, initial learning rate 0.0001, 30 epochs, input 1280, seed 46, the same epoch173 initializer and the same recorded L4 runtime. |

R24 did **not** continue training the returned R23 weights. Both candidate runs started independently from the same original initializer. The actual latest supervision change was **22 corner locations in 18 rows of three training photos**: 012, 014 and 016. The candidate has 6,836 locatable training endpoints, so approximately **0.32% of those endpoint locations changed**. Boxes, visibility, classes, images and instance counts stayed the same. The four repaired points in photo010 belong to development data and did not enter training. These facts are verified from the actual staged files and receipts in the [cross-round input audit](../results/evaluation/pr99-r24-20261123/crossround-input-verification.json).

A trained network is a shared function across images. Correcting a few targets changes the fit; it does not install an independent lookup-table patch for each corrected corner. A correction can be physically justified while the resulting fitted model improves some predictions, leaves others unchanged and moves others away. That is a general explanation of why improvement need not be monotonic, not a demonstrated diagnosis of one internal network mechanism here.

The latest edit also addresses a different problem from several remaining failures. Adjusting visible corner locations does not directly provide new box-extent supervision for an almost hidden front, or teach that a bright rack support is background. The R24 misses include just such cases. There is no measured evidence that the latest result happened because the learning rate was too large, too small, or because a different architecture was required.

## 2. What actually improved, and what did not

The comparison below uses **the same saved accepted R24 reference for both models**, the same fresh CPU runtime and the same inference and scoring settings. It avoids confusing an annotation change with model learning. [Source: completed PR99 comparison](../results/evaluation/pr99-r24-20261123/README.md).

| Same R24 exploratory reference | R23 candidate | R24 candidate |
| --- | ---: | ---: |
| Matched fronts / 662 | 645 | 646 |
| Missed fronts | 17 | 16 |
| Unmatched outputs | 95 | 90 |
| Both corners within 5 px / 584 | 557 | 557 |
| Both corners within 10 px / 584 | 583 | 583 |
| Nonempty images without a miss / 17 | 9 | 10 |

This is a small box-matching improvement with **no aggregate improvement in the 5 px or 10 px pair totals**. The one gained match, `089:49`, already had a nearby R23 proposal; its extent changes enough to meet the fixed IoU requirement. It is not evidence that a previously absent physical detection appeared. Among 1,215 identically associated, locatable endpoints returned by both models, 594 errors fell and 621 rose; the average error change was +0.014118 px. The equal five-pixel pair totals contain two gains and two losses, mostly small threshold crossings. The [cross-round records](../results/evaluation/pr99-r24-20261123/crossround/summary.json) retain those identities.

The frozen 666-front primary reference gives a different ranking: PR35 has 18 misses and 533/584 five-pixel pairs, while R24 has 35 misses and 518/584 pairs. On the current R24 reference, PR35 instead has 37 misses and 544/584 pairs. This disagreement matters: accepted reference-box changes can alter whether an unchanged prediction satisfies IoU 0.50. **Sixteen of R24's frozen-reference misses become matches solely through the reference geometry change.** The [reference sensitivity trace](../results/evaluation/pr99-r24-20261123/reference-sensitivity.json) shows that directly.

Therefore neither “the latest annotations made the model worse” nor “the latest weights are definitely better everywhere” follows from one column. PR35 remains the selected working checkpoint under the established evaluation decision. R24's accepted physical annotations and its useful exploratory results remain available. A later R25 result must keep the reference versions explicit and score competing models on the same bytes.

### What the saved training curves tell us

The R24 candidate's recorded CSV contains all 30 epochs. These are its first-five and last-five epoch means, not selected favorable checkpoints:

| Recorded quantity | Epochs 1–5 mean | Epochs 26–30 mean | Reading |
| --- | ---: | ---: | --- |
| Training box loss | 1.315402 | 1.152016 | Training fit improves |
| Training classification loss | 0.330936 | 0.276100 | Training fit improves |
| Training pose loss | 0.015788 | 0.013082 | Training fit improves |
| Validation box loss | 1.707006 | 1.549902 | Improves over these windows |
| Validation classification loss | 0.363414 | 0.386916 | Worsens |
| Validation pose loss | 0.015456 | 0.015170 | Slight improvement |
| Validation box mAP50 | 0.926504 | 0.924228 | Slight decrease |

The classification gap is **consistent with** a training/generalization tension, but the validation trends are mixed. It does not prove that this model is simply overfitting or underfitting. The original trainer validation still has 664 fronts with its original labels, while the candidate fits accumulated corrected training labels; reference disagreement is an additional plausible contributor. R23 follows nearly the same trends, and all 30 logged learning-rate values in all three parameter groups match across R23, R24 and the R24 control. The logged final learning rate is 0.0000043. These observations do not isolate learning rate as the cause or show that more epochs would help.

The [original candidate CSV](../results/evaluation/pr99-r24-20261123/checkpoint-receipts/candidate/fit/results.csv) and [log](../results/evaluation/pr99-r24-20261123/checkpoint-receipts/candidate/train-log.txt) remain unchanged; all three roles' curve summaries and hashes are in the new inventory JSON. The log also contains the trainer's automatic best-checkpoint validation. The returned comparison uses the prespecified final-epoch checkpoint; this report does not retrospectively switch to the most favorable epoch. Trainer pose mAP is also a different metric from both corners landing within five original-image pixels.

## 3. Exactly where the current R24 model struggles

These are **model-versus-R24-reference outcomes**, not a list of labels that should be changed to agree with the model.

### All 16 missed targets

| Photo | Missed R24 rows | Visible difficulty |
| --- | --- | --- |
| `new_000008` | **33, 34, 36** | Right-border clips and a bottom-border fragment |
| `new_000009` | **55, 56, 57** | Tiny fragments in or beside the jaws |
| `new_000010` | **14, 33, 34** | Jaw-occluded strips; nearby boxes cover too much occluded area |
| `new_000011` | **55, 56, 57, 60** | Narrow jaw-side strips and a lower jaw-gap fragment |
| `new_000053` | **1** | Thin top-border fragment |
| `new_000086` | **57** | Approximately 8 px exposed beside the thumb |
| `new_000088` | **57** | Approximately 9 px exposed beside the thumb |

Ten have a retained overlapping rectangle below the fixed matching IoU; six have no retained overlap. No qualifying retained proposal was lost through assignment to another target. Seven missed targets have no locatable corner and nine have one. **All 584 targets with two locatable corners are matched.** The [sixteen-case pixel review](../results/evaluation/pr99-r24-20261123/partial-front-failure-queue.md) provides the complete source-image and PDF-page navigation.

The clearest box-versus-corner distinction is `011:55–57`: retained alternatives put their available left corners about **1.96, 3.36 and 1.43 px** from the accepted points, yet their boxes extend too far into occlusion and fail association. Correcting ordinary visible corners alone is unlikely to address that specific visible-extent problem. Those distances are diagnostic on unmatched alternatives, not official corner successes.

### All 27 two-corner pairs outside five pixels

| Photo | R24 rows with at least one endpoint outside 5 px |
| --- | --- |
| `new_000008` | **4, 5, 6, 10, 11, 18, 19, 23, 29** |
| `new_000010` | **26** |
| `new_000011` | **4, 49** |
| `new_000053` | **15** |
| `new_000055` | **21** |
| `new_000056` | **22** |
| `new_000058` | **1, 4, 7, 10** |
| `new_000059` | **1, 2, 7** |
| `new_000087` | **53** |
| `new_000088` | **52** |
| `new_000090` | **19, 44, 45** |

**All 27 are geometric errors, not endpoint-confidence failures.** Both endpoints are returned in every case. The complete ledger gives each left/right distance and confidence. The single pair outside **10 px** is `008:10`: its right endpoint is **12.502088 px** away despite confidence **0.908112**. Original crops of every listed pair show soft or rounded transitions, adjacent lips, rack edges or occlusion context; that makes consistent physical corner definitions useful, but it does not identify a single proven cause for all errors.

There are also **ten returned one-corner endpoints outside 5 px**, separately tabulated in the ledger. One further matched front, `008:1`, has a right-endpoint confidence of **0.382982** and a saved coordinate **12.262111 px** away. Lowering the endpoint threshold would expose that coordinate without making it accurate. Model confidence is not a calibrated probability that a point lies within five pixels.

### Duplicates and background confusion

R24 emits **33 scored duplicate outputs in 12 images**, plus **57 other unmatched outputs**. The ledger lists every duplicate by image, accepted target row and output index. The distinction is useful: some other extras are real partial-front proposals with mismatched extents, so all 57 should not be described as background hallucinations.

There are concrete background examples: `089` output **51** lies on the bright top rack support, and `090` output **50** lies over the lower-right keyboard/bezel region. Both have zero overlap with accepted target boxes. These are zero-based output IDs because background has no target row. At `011:4`, by contrast, an accurate duplicate pair already exists but ranks slightly below the canonically chosen pair. Knowing the reference lets us diagnose that ranking; it does not give an inference-time rule for choosing the better alternative.

## 4. Useful improvements with no additional captures

### First: use the completed R25 label amendments and isolate their effect

The [completed R25 review](front-review-r25.md) amends eight left points: development rows `008:35`, `010:18/19/20` and training rows `012:1/2/3/28`. The ambiguous right point at `012:28` is retained. Its matched notebook is pinned to these supported changes. The supplied `008:18–21` and `008:28–32` examples are useful **visual standards for annotations**. They are not promises that the model predicts those points correctly: the model's left errors at standards `008:18`, `008:19` and `008:29` are 6.599, 8.190 and 7.101 px.

The named `008:35`, `010:18/19` and `012:3/28` requests were handled through this physical amendment workflow, with the three adjacent rows recorded separately. Photo012 is training data; photos008 and010 remain development data. In the completed R24 predictions, `010:18/19` already satisfy five pixels against the accepted R24 labels. That score did not settle their physical corner placement. Conversely, an inaccurate model prediction is not evidence that a correct label needs amendment. No training prediction was evaluated for `012:3/28` in PR99.

For this first R25 comparison, retain the established recipe so the experiment answers the annotation question. Compare the returned candidate and matched control on unchanged frozen references and a separately identified current review. Use an identical reference for R24-to-R25 comparisons. Preserve the detection, corner, duplicate and empty-scene checks. The new notebook is **ready for the newly authorized amendment experiment**; the earlier completed-R24 decision remains historical.

### Second: review analogous existing training fragments for visible box extent

The strongest case-driven next supervision question is whether equivalent exposed fragments have consistent **visible** box bounds in eligible training originals. Start with the pattern of `011:55–57`, then the narrow jaw-gap layers in `009:55/57` and `010:33/34`, plus clipped border fronts like `008:33/34/36`. Inspect the whole scene and the relevant rows together. A box should follow the agreed observable front extent; it should not silently span a jaw merely because the cassette would continue behind it. Do not invent hidden corners or add a layer from spacing alone.

This is a hypothesis-directed review of existing training data. It is useful only if it finds **new, supported inconsistencies**. PR40's earlier five-image partial-front extension already failed to recover its prespecified targets; simply repeating those same edits or the same run is not a new intervention. Validation examples above are diagnostic examples, not permission to move them into training.

### Third: consider one bounded training-distribution experiment only if the audited failure pattern remains

If the eligible training set already contains correctly labeled, genuinely comparable jaw-side and border fragments, a **limited image-sampling rebalance** is a possible next hypothesis: ensure those existing images contribute sufficiently while retaining normal visible fronts and empty/background images. Keep the model, initialization, label bytes and optimizer-update budget fixed between an ordinary-sampling control and the declared alternative. Monitor the specific partial-front cohort and clean coverage, rather than optimizing only its small subset score.

This has not been tested and has no promised benefit. Oversampling a few familiar views can worsen generalization or background rejection. It should be a separate experiment after the label-only R25 question, not bundled with changed learning rate, larger resolution, longer training and new augmentations. The current logs do not establish which of those changes would help.

### Fourth: diagnose a specific unresolved proposal before changing filtering

For current targets with no retained overlap, such as `008:36` or `009:55–57`, the saved arrays cannot reveal whether a raw proposal was below confidence, removed by NMS, or geometrically unsuitable. If a later implementation choice depends on that distinction, inspect the current candidate's raw proposals for the named cases once. The earlier PR35 trace already answered this question for its own weights and led to the accepted confidence 0.13; it must not be assumed to diagnose R24. Rows `008:33/34` instead have small nonzero overlaps, 0.074066 and 0.065351, with neighboring-front outputs; neither reaches the matching requirement.

There is no recommendation for another generic threshold/NMS sweep. Prior stronger NMS removed valid partial-front matches, the accepted lower confidence is already in use, the 1920-resolution trial worsened misses and corners, and PR62's synthetic occlusion recipe failed clean preservation despite improving its synthetic probe. The [previous NMS result](postprocess-experiment-2026-11-07.md), [proposal review](proposal-filter-review-2026-11-11.md), [resolution/partial-front work](partial-front-improvement-2026-11-11.md) and [occlusion result](occlusion-results-2026-11-14.md) are completed evidence, not untried suggestions.

## 5. What to capture later, in practical priority order

These priorities follow observed failure patterns. They are a collection plan, not a statistical guarantee or a request to collect more data before finishing R25.

| Priority | Capture deliberately | Exact motivating cases | What the new evidence should clarify |
| --- | --- | --- | --- |
| 1 | Small front strips between jaws, beside either jaw, and several distinct visible-width levels; vary jaw opening and camera angle | `009:55–57`, `010:14/33/34`, `011:55–57/60` | Separate visible front from tool; learn fragment extent and distinct layers without assuming hidden corners |
| 2 | Natural thumb/hand occlusions leaving thin strips, with different grip placements | `086:57`, `088:57` | Recover identifiable narrow material while keeping unavailable corners unavailable |
| 3 | Fronts progressively clipped at right, top and bottom borders; include ordinary fully visible companions | `008:33/34/36`, `053:1` | Cover truncated boxes and one-/zero-corner states without discarding edge fragments |
| 4 | Bright rack supports, tool surfaces, walls, keyboard edges and reflections, both with and without real cassette fronts | Current `089` output51, `090` output50; earlier PR62 empty `147` wall/pipe output | Distinguish similar bright shapes from fronts without suppressing genuine low-confidence partial objects |
| 5 | Clear close views of rounded lower-front corners, lips and adjacent rack rails across angle, distance, shadow and exposure | `008:4–6/10/11/18/19/23/29`, `058:1/4/7/10`, `090:44/45` | Reduce confusion about the desired physical junction and assess localization beyond a single familiar appearance |

For difficult partial views, a separately captured unobstructed/context view can help a human establish object identity. It does not automatically reveal the exact hidden coordinate in the occluded image, particularly if the camera or object moved. Keep target-image visibility labels honest. Shadow and sudden brightness can help locate a junction, but illumination, reflections, rounding and exposure can shift those cues; neither is an infallible physical definition of a corner.

When new captures become possible, reserve an independently labeled capture session for evaluation and keep related views from one session on the same side of the split. Near-duplicate video frames should not be counted as independent proof of broader reliability. Vary the real failure conditions and retain ordinary successful scenes as controls. No particular frame count can guarantee success in advance.

## 6. What remains established, and what remains a hypothesis

The established facts are the staged 22-point training delta, the fixed recipe, the saved training curves, the same-reference scores, and the exact failure identities above. The plausible next improvements are **coherent pixel-supported R25 labels**, then **specific existing-training supervision or sampling changes justified by remaining cases**. Their effects must be measured after training. Learning-rate blame, automatic architecture replacement, guaranteed benefit from more epochs, and broad accuracy claims on future camera sessions remain unsupported.

The new inventory exactly reproduces every saved matching trace and the complete aggregate detection and corner-quality records using the existing scorer. All 16 misses, 27 failed pairs, 33 duplicates and endpoint accounting reconcile with the completed PR99 evidence. Existing R24 annotations and evaluation files remain unchanged.

## 7. Current baseline after the completed R25 reference repair

The R25 labels are now complete. Its manifest is pinned to SHA-256 `de80c21ed4b148efe6446e66b8a15f75c3b6363daf9c03d0b88a29e187483517`. To make “currently poor” precise, the **same saved predictions** from all four models were scored against that new reference. This uses no new inference and measures no new learning. The R25 notebook is ready; the table below contains existing checkpoints, not an R25-trained candidate.

| Unchanged saved model, R25 reference | Matched / 662 | Misses | Unmatched outputs | Both corners within 5 px / 584 | Within 10 px / 584 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PR35 | 625 | 37 | 103 | 543 | 573 |
| R24 control | 624 | 38 | 105 | 539 | 573 |
| R23 candidate | 645 | 17 | 95 | 557 | 583 |
| R24 candidate | 646 | 16 | 90 | 557 | 583 |

Only four left-point coordinates changed in the development reference. Boxes, classes, visibility and row identities stayed fixed, and all 80 model/image matching traces remain identical. PR35 and the control each lose a five-pixel pair at `010:20` compared with their R24-reference totals; both recent candidates keep the same pair totals. These are **reference-coordinate effects**, not changes to any model.

For the unchanged R24 candidate, the four amended development rows now read:

| Target | Left error on R24, px | Left error on R25, px | Right error, px | R25 status |
| --- | ---: | ---: | ---: | --- |
| `008:35` | 5.896019 | **1.871916** | Unlocatable | Available left now passes 5 px; right remains unlocatable |
| `010:18` | 2.751796 | **2.473627** | 0.597099 | Both remain within 5 px |
| `010:19` | 2.507165 | **0.665292** | 0.605535 | Both remain within 5 px |
| `010:20` | 2.851306 | **2.257313** | 1.622320 | Both remain within 5 px |

**The current R25-reference failure list for this R24 model therefore retains all 16 missed rows, all 27 failed pairs, and all 33 duplicates listed above.** Their identities and the failed-pair distances are unchanged. The separate returned one-corner error list falls from ten to **nine**: `008:8`; `009:49/50/51/52`; `055:23`; `056:12`; `058:18`; `089:50`. The confidence-withheld `008:1` endpoint remains separate. `008:35` is removed from the five-pixel error list because its physical reference point was corrected; the prediction itself did not move.

The [inventory's R25 section](../results/review/front-review-20261123-r25/model-failure-inventory.md#8-current-baseline-on-the-completed-r25-reference) preserves the endpoint confidences and before/after records for all four models. The original R24 tables remain intact for historical comparison, while this subsection is the explicit current baseline for assessing the next returned model.
