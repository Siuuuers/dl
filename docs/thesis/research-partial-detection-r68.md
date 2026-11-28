# Tiny partial fronts: diagnose the missing stage before another model

28 November 2026. **Completed bounded CPU experiment:** the [R68 replay](evidence/partial-detection-r68/results.md) diagnoses the three misses added by the [four-front inclusion review](partial-inclusion-r67.md), then measures one fixed union of saved original and unpaired reflected outputs. The independent verification passed. This note separates a small visible box, a low prediction score and suppression of a usable box. These observations can coexist, but they do not imply the same remedy. No new loss, detector architecture or operating threshold is selected here.

## 1. Correction to the earlier research sequence

The earlier [duplicate research note](research-duplicates-r66.md) described Soft-NMS as a future option without recovering its completed project history. **Linear Soft-NMS was already tested in PR143/R49**, with a byte-identical repeat in PR144/R50. At confidence strictly above 0.13, overlap 0.70, cap 300 and the unchanged postclip filter, R50 misses remained 15, unmatched outputs fell from 84 to 83 and formal duplicates from 13 to 12. Five-pixel pair success remained 566/583. Raw Soft-NMS also recovered no target before postclip filtering. This is a result for that checkpoint and fixed variant, not a proof that every Soft-NMS configuration fails. The [completed diagnosis](../front-diagnosis-r51.md) records the full stage comparison and two suppression competitions.

Larger whole-image inference is also completed. In [PR150/R53](r53-results.md), input 1280 to 1600 changed R50 misses from 15 to 16, extras from 84 to 77, duplicates from 13 to 17 and five-pixel pairs from 566 to 557. The 1280 control already preserves the original 1280-pixel photograph width. Its larger version interpolates those pixels. Earlier [1920 inference](../partial-front-improvement-2026-11-11.md) also failed its fixed screen. A generic request to try a larger image would repeat a known question.

The [trained one-to-one head](../front-review-r37.md) and [fixed cached box unions](evidence/cached-fusion-results.md) are likewise historical experiments. On R50, pooling original plus CLAHE outputs reduced misses from 15 to 12 but raised extras from 84 to 102. Original plus 1600 reduced misses to 10 but raised extras to 124. Both lowered five-pixel pair success. These results justify measuring any new aggregation rule's losses and duplicates alongside recoveries. They do not rule out a distinct, well-motivated fixed rule.

## 2. What the restored fronts actually require

The three newly restored misses have one known corner each. Their boxes describe observed fragments, not completed hidden cassette shapes. Their historical coordinates were restored after raw-image review. Changing the evaluator to accept a larger overlap error would not teach the model to recognize them.

The R68 [stage evidence](evidence/partial-detection-r68/summary.json) distinguishes the historical adapted model's full decoded original-view pool from newer archives that retain only post-NMS outputs. In the historical pool, input shape is 1 × 3 × 736 × 1280, with unit resize and eight pixels of padding above and below the photograph. The output pipeline keeps confidence strictly above 0.25. Evaluation matching requires box IoU at least 0.50.

| R67 row | Nominal visible size | Historical original-view evidence | Supported interpretation |
| --- | ---: | --- | --- |
| `new_000009:R055`, right-jaw fragment | 18 × 31 px | One decoded box reaches IoU 0.60849, with score 0.002152 | Usable box geometry exists below the confidence gate |
| `new_000009:R056`, lowest left band | 15 × 14 px | No decoded box reaches IoU 0.50. Best IoU is 0.40907, with score 0.33002, and that box survives | Lowering the score gate cannot recover this target from the same boxes |
| `new_000053:R022`, top wedge | 47 × 10 px | Two decoded boxes reach IoU 0.50. Best IoU is 0.69087 and the greatest eligible score is 0.00009662 | Usable geometry exists at an extremely low score |

These are pipeline locations, not established causes of learning failure. A low score could reflect the learned appearance, target construction or training exposure. The best-IoU box need not be the highest-scoring eligible box.

The newer PR157 QFL reflected view contains a box for the lowest left band at IoU 0.50475 and score 0.29001. That observation does not establish a ready-made union recovery. It already has a mutual original-view partner with overlap 0.73382. Its association status is `reflected_pair_unavailable`, which means that its corner pair cannot be averaged, not that its box has no partner. The original partner scores 0.40807, so pooling both and retaining the existing score-ranked suppression above 0.70 removes the reflected box. A rule that adds only unpaired reflected records excludes it before that step.

PR157 low-floor caches are post-NMS at confidence above 0.001. Some reach the 300-output cap. PR158 caches are post-NMS above 0.25. Absence from these saved pools does not establish absence from the decoded network output, and threshold replay cannot recreate discarded candidates. These limits must accompany any apparent coverage bound.

### Training representation is a separate clue

The verified PR157 training snapshot has 3,999 fronts across 142 photographs, including 465 with one known corner. Counting original, nominal one-known boxes whose width AND height are no greater than each restored shape gives zero at 15 × 14 px, ten at 18 × 31 px across eight images and zero at 47 × 10 px. The same counts hold with a 0.001-pixel serialization tolerance. These are overlapping descriptive subsets, not three disjoint or formal tiny-object classes. The thresholds were chosen after inspecting these evaluation targets and are diagnostic comparisons.

This suggests sparse nominal representation of these particular visible shapes. It does not establish that the loss is wrong or that the photographs contain no other useful visual examples. Random cropping and scaling change box dimensions, so these counts do not measure effective training exposure. The existing [augmentation audit](augmentation-audit.md) addresses survival under transforms, and the [frozen assignment audit](score-assignment-audit.md) found assignment for all 3,999 fronts in its replay. These measurements do not establish the learning cause or justify oversampling the same repeatedly inspected examples. Adding the three development targets to training would also compromise the intended comparison.

## 3. Why a few pixels matter for a thin box

The following is a **synthetic calculation**, not a measurement of any model output. Translate a box of width w and height h by dx and dy without changing its size. Its intersection area is

\[
I=\max(0,w-|dx|)\max(0,h-|dy|),\qquad
\operatorname{IoU}=\frac{I}{2wh-I}.
\]

| Synthetic box and displacement | Resulting IoU |
| --- | ---: |
| 15 × 14 px, shifted 3 px horizontally and 3 px vertically | 0.4583 |
| 18 × 31 px, shifted 5 px in both directions | 0.4344 |
| 47 × 10 px, shifted 4 px vertically | 0.4286 |

The last example shows why width alone is an inadequate description of size. A wide but shallow visible strip can be sensitive to vertical displacement. Keep the established IoU criterion and report continuous edge errors and available-corner errors as supplementary evidence. An annotation's nominal decimal precision does not establish physical accuracy at that precision.

## 4. Three relevant primary methods, with different intervention points

### NWD: a conditional change to training geometry

Wang et al. [1] encode a box as a Gaussian with its center and covariance diag(w²/4, h²/4). The resulting Wasserstein distance depends on center displacement and width/height differences. They normalize it as exp(−W2/C). Their paper separately changes assignment, regression and suppression. Its largest single-module gains are in assignment, so the paper does not establish a benefit from replacing only a pose detector's regression loss.

The [official loss implementation](https://github.com/jwwangchn/NWD/blob/9775ac2d24354895fbdf5b8db1f1600c01c54e33/mmdet/models/losses/iou_loss.py) makes C an explicit scale parameter. Its default cannot be transferred as a camera-independent constant. The [existing conditional design](research-next-models.md#4-one-conditional-localization-experiment) preserves normalized L1 and mixes only the CIoU term with NWD. It remains unexecuted. The current [native loss contract](../../scripts/quality_focal_r61.py) already contains CIoU and normalized L1, so absence of a geometry gradient outside overlap is not a valid motivation. A box-loss change also does not directly repair the two extremely low-score cases.

**Decision:** retain NWD as a hypothesis if wider evidence establishes persistent extent errors despite useful assignment and representative training targets. Keeping the assignment formula unchanged during retraining would not freeze actual assignments or soft targets, because network outputs still change.

### SAHI: sliced inference is different from enlarging the whole image

Akyon, Altinuc and Temizel [2] run a detector on overlapping resized crops, map results to the photograph and merge outputs. They also study sliced fine-tuning as a separate training change. Crops can make a small object occupy more model pixels, but may truncate neighbouring objects or remove useful context. Their paper reports that overlap can increase fragment-related false positives. Cropping and interpolation do not create new optical detail or reveal a hidden surface.

This is therefore distinct from the completed whole-image 1600 trial, but it adds several choices about crop geometry, overlap, full-frame inclusion and merging. Moreover, the inspected [official Ultralytics adapter](https://github.com/obss/sahi/blob/4e05479d93d36377301e508e36146ccf4d669f97/sahi/models/ultralytics.py) returns `result.boxes.data` in its ordinary detection branch. It does not preserve `result.keypoints`. That specific revision is not a ready-made cassette-pose pipeline. A trial would need explicit corner restoration and record provenance.

**Decision:** do not prepare tiling merely because these three reference boxes are small. First establish useful geometric headroom, preserve partial-front context and define a pose-preserving merge. No SAHI cassette result exists here.

### Box aggregation: correlated alternatives need a complete evaluation

Solovyev, Wang and Gabruseva [3] study ensembling with confidence-weighted box-coordinate averaging. Their TTA experiment combines original and horizontally reflected predictions and compares several aggregation rules. This supports distinguishing the views from the rule that merges them. Their published gains use another dataset and tuned aggregation parameters.

The completed R68 alternative retains original records with their existing averaged corners and adds reflected records without a mutual original-view partner. All mutual box partners count as used, including those without an available corner pair. Applying the existing score-ranked NMS above 0.70 to this pool is a project-specific union. Each native view already had a 300-output cap, and the pooled postclip pass introduces no additional cap. It does **not** implement Weighted Boxes Fusion, because it does not average box coordinates or use WBF's score rule. It also differs from the completed original-anchor and reflected-anchor corner averaging, which each preserve their anchor view's box population.

Reflection supplies another response to the same observed pixels. It is not an independent camera observation. Agreement can preserve a shared error, and disagreement can concern a real partial front. A union can add extras or cross-view duplicates, suppress an original detection, or change reference assignment. The already-paired QFL band above supplies no recovery for this rule. The broader saved-pool audit found unpaired reflected support for other missed fronts, motivating this one fixed CPU probe. The rule applied to every qualifying reflected record without consulting reference labels. An unmatched added output still lacks reference support rather than automatically proving background.

## 5. What the completed union establishes

On PR158's reviewed-label model, the fixed union changes r67 misses from **18 to 10**, extras from **50 to 60** and the duplicate subset from **13 to 18**. Its five-pixel complete-pair count stays **572/583**, and its ten-pixel count stays **583/583**. The [full matrix](evidence/partial-detection-r68/matrix.csv) retains all five methods under four references. **All three newly restored focus targets remain missed.** This experiment recovers other partial fronts, not the three cases that initiated the investigation.

The eight recovered r67 fronts comprise six with one known corner and two with none. Among those six known corners, four returned points are within five pixels and five are within ten. The remaining error is **11.3027 px** at `new_000008:R007`. Its box recovery is therefore not a ten-pixel corner success. The two zero-known targets provide no reference coordinates for evaluating their emitted corners. Neither detection gains nor unchanged complete-pair counts establish eight accurate new corner pairs. The [target-change ledger](evidence/partial-detection-r68/union-target-changes.json.gz) retains these confidence-qualified point errors and individual identities.

The result also depends on the evaluation reference. The same reviewed-label model and unchanged union outputs give:

| Reference | Misses, unchanged → union | Extras, unchanged → union | Duplicate subset, unchanged → union |
| --- | ---: | ---: | ---: |
| r50 | 15 → 7 | 51 → 61 | 13 → 18 |
| r67 | 18 → 10 | 50 → 60 | 13 → 18 |
| Untouched supplied664 | 50 → 48 | 76 → 92 | 12 → 16 |
| Earlier frozen666 | 38 → 36 | 62 → 78 | 14 → 18 |

Thus the eight-match benefit under r50 and r67 becomes two matches under either older reference, accompanied by sixteen extra outputs. These references reuse the same photographs and are not independent tests. The difference shows sensitivity to target definitions and reference support, not four independent confirmations of improvement. The union is **not promoted**.

## 6. Reproduce the result and choose the next question

The [four-cell CPU notebook](../../notebooks/baomai_partial_detection_r68.ipynb) pins source [`0e1f545a2cf33022b86bf316866b08cacfa55477`](https://github.com/Siuuuers/dl/tree/0e1f545a2cf33022b86bf316866b08cacfa55477). The analysis preserves exact restored-target stages, saved-pool limitations, geometry, the fixed threshold-only support ceiling, nominal training-shape counts and the union's per-target changes. It uses saved outputs and requires no new inference or training. Its purpose is to establish what the existing candidate pools can support before asking a new loss or transform to create missing geometry.

For any later trial, retain all reference versions, fresh one-to-one evaluation matching, misses, all extras, the duplicate subset, empty-scene outputs and five-/ten-pixel complete-pair counts. Identify recovered and newly lost partial fronts explicitly. A better complete-pair count cannot demonstrate improvement on targets with only one known corner. All these photographs have repeatedly informed development, so a favorable exploratory result still needs independent captures before a generalization claim.

## Primary references and inspected implementations

1. J. Wang, C. Xu, W. Yang and L. Yu, “A normalized Gaussian Wasserstein distance for tiny object detection,” arXiv:2110.13389, 2021, version 2 revised 14 June 2022. [Versioned paper](https://arxiv.org/abs/2110.13389v2) and [PDF](https://arxiv.org/pdf/2110.13389v2). Official repository revision `9775ac2d24354895fbdf5b8db1f1600c01c54e33`, inspected loss-file Git blob `4e063ada7453dbfb5c8764c91468568a681a1bad`. This entry identifies the original preprint, not the expanded journal study.

2. F. C. Akyon, S. O. Altinuc and A. Temizel, “Slicing aided hyper inference and fine-tuning for small object detection,” in *Proc. IEEE Int. Conf. Image Processing*, 2022, pp. 966–970, doi: [10.1109/ICIP46576.2022.9897990](https://doi.org/10.1109/ICIP46576.2022.9897990). [Versioned author paper](https://arxiv.org/abs/2202.06934v5). Official SAHI revision `4e05479d93d36377301e508e36146ccf4d669f97`, inspected `sahi/models/ultralytics.py` Git blob `49bcb054147cd1abd2c1c52ef60401dae69dc7b3`.

3. R. Solovyev, W. Wang and T. Gabruseva, “Weighted boxes fusion: Ensembling boxes from different object detection models,” *Image and Vision Computing*, vol. 107, Art. no. 104117, 2021, doi: [10.1016/j.imavis.2021.104117](https://doi.org/10.1016/j.imavis.2021.104117). [Versioned author paper](https://arxiv.org/abs/1910.13302v3) and [official implementation](https://github.com/ZFTurbo/Weighted-Boxes-Fusion). Section 6.3 concerns original/reflected prediction ensembles. It does not establish the cassette project's proposed record-selection rule.
