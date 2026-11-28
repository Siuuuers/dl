# Detection errors and the next experiment

28 November 2026, Asia/Hong_Kong.

**The remaining detection weakness is concentrated in fronts whose lower corners are not fully locatable.** All 18 misses of the retained two-view method belong to that group. Only five of its 51 extra predictions are duplicates. A completed saved-output experiment reduces misses and total extras by retaining the reflected view's boxes, but doubles the duplicate subset and slightly worsens corner agreement with older references. It remains an exploratory alternative.

The [CPU notebook](../../notebooks/baomai_detection_audit_r62.ipynb) reproduces this audit and that single experiment. It uses the already completed common-runtime predictions; no GPU, image download, training or new inference is required. The prepared [classification-loss experiment](quality-focal-experiment.md) remains the next model run, followed by the [controlled training-label comparison](training-label-comparison.md).

## 1. What unmatched and duplicate mean here

The unchanged scorer orders predictions by confidence and matches each to at most one same-class reference at box intersection-over-union (IoU) of at least 0.50. An unassigned prediction is an **extra**; an unassigned reference is a **miss**. An extra is classified as a **duplicate** when it overlaps an already matched reference at IoU of at least 0.50. Duplicates are included within extras.

These are evaluation outcomes. An extra may depict background, the wrong visible extent of a real front, another output for the same front, or a real front omitted or defined differently in the reference. A displaced box can contribute both an extra and a miss. Adding these counts therefore does not count distinct physical failures.

TIDE separates detection errors to make such distinctions useful [1]. This audit borrows its diagnostic motivation but keeps the project's existing scorer. It is not a TIDE run or an estimate of TIDE's change in average precision. The two added overlap groups are review cues; an overlap below 0.10 is not proof of background.

## 2. Completed audit

The input is the verified annotation-reference archive, containing the original supplied detector and the retained two-view method scored on the same 20 photographs. The audit reconstructs suppression and corner averaging from the saved native outputs and reproduces all 240 existing per-image score tables and all six original/proposed reference-comparison rows exactly. The [evidence folder](evidence/detection-errors/README.md) preserves the results and 530 method/reference-specific review records. Those records reuse the photographs and are not 530 independent test cases.

### Extra predictions

| Reviewed reference | Supplied model | Retained two-view method |
| --- | ---: | ---: |
| Matched fronts | 620 | 636 |
| Missed fronts | 34 | 18 |
| Extra predictions | 61 | 51 |
| Duplicate subset | 5 | 5 |
| Other extras with maximum reference IoU from 0.10 to below 0.50 | 49 | 36 |
| Other extras with maximum reference IoU below 0.10 | 7 | 10 |

The current 51 extras comprise exactly 5 + 36 + 10 records. Removing just the five duplicates while preserving every other prediction would raise reference precision from 636/687 (92.58%) to 636/682 (93.26%). This arithmetic is not an implemented filter or an achievable performance guarantee. It shows why duplicate removal alone cannot solve the whole problem.

Four photographs contain 39 of the 51 extras: `new_000009` has 15, `new_000089` has 9, `new_000090` has 8, and `new_000011` has 7. These are efficient review priorities, not a new training set. All three reference-empty photographs still have zero predictions; three familiar scenes cannot establish a general false-alarm rate.

### Detection according to reference corner availability

| Locatable corners | Reference fronts | Supplied model detected | Retained method detected |
| --- | ---: | ---: | ---: |
| Two | 583 | 580 | 583 |
| One | 54 | 31 | 41 |
| Zero | 17 | 9 | 12 |
| One or zero combined | 71 | 40 | 53 |

The retained method detects 53/71 (74.65%) fronts with incomplete corner annotations. Its 583/583 complete pairs within ten pixels does not include those 71 fronts. Unavailable coordinates may reflect occlusion, clipping or ambiguity; the table does not infer a physical cause from the visibility flag.

Ten of the 18 missed fronts have a retained prediction with reference IoU from 0.30 to below 0.50. Seven have maximum overlap below 0.10; the remaining case has overlap approximately 0.126. Review the former ten for front extent, association and reference interpretation before assuming the network never recognized them.

### What the cache can establish

The fresh native cache was already produced with object confidence greater than 0.25, native NMS at 0.70 and a 300-output cap. The boxes had also been clipped to the image. It contains no lower-confidence or previously suppressed proposals.

None of the 18 current misses has an eligible IoU-at-least-0.50 box in that trained native cache. The extra suppression after clipping is therefore not the stage at which those 18 lose an eligible box. The cache cannot distinguish native suppression, insufficient score, poor box geometry and absence of a proposal. Calling them all confidence failures or absent network responses would overstate the evidence.

## 3. One fixed experiment with reflected boxes

The retained method associates original and reflected predictions by mutual best predicted-box overlap, then averages the two corresponding corner coordinates equally while keeping the original-view boxes and scores. The new experiment changes only the anchoring view: retain the reflected boxes, scores, order and identities, with the same averaging rule and fallback.

The rule was declared before computing its scores, after inspecting the earlier results. No reference coordinates select associations or averaging weights; no thresholds are fitted. This is exploratory reuse of development photographs, not a preregistered independent evaluation.

| Reviewed reference | Supplied model | Retained two-view method | Reflected boxes with the same averaging |
| --- | ---: | ---: | ---: |
| Missed fronts | 34 | 18 | 15 |
| Extra predictions | 61 | 51 | 43 |
| Duplicate subset | 5 | 5 | 10 |
| Both corners within 5 pixels | 555/583 | 569/583 | 569/583 |
| Both corners within 10 pixels | 577/583 | 583/583 | 583/583 |

Relative to the retained method, eight fronts become matched and five become missed. Every changed match belongs to the incomplete-corner group. The complete reviewed pairs keep identical averaged coordinates and the same mean maximum-corner error of approximately 2.163 pixels. This is not a new learned corner improvement.

The older references retain adverse findings:

| Evaluation reference | Retained method within 5 / 10 pixels | Reflected-box method within 5 / 10 pixels |
| --- | ---: | ---: |
| Untouched supplied labels, 584 pairs | 514 / 565 | 512 / 563 |
| Earlier fixed labels, 584 pairs | 525 / 574 | 524 / 573 |
| Reviewed labels, 583 pairs | 569 / 583 | 569 / 583 |

The supplied original model remains better on the older annotation references. The new variant improves reviewed detection precision and recall but doubles duplicates and loses older-reference corner successes. Keep it as an explicit experimental alternative; the default method and the pending training protocols remain unchanged.

The [probe evidence](evidence/reflected-anchor/README.md) includes all twelve method/reference rows, paired changes, predictions and view associations. An independent implementation reproduces every averaged coordinate and association. All 648 averaged links reverse symmetrically when the anchoring view changes. No link is assigned to different reference targets in these photographs; this observed agreement is not a guarantee on unseen scenes.

## 4. Correct pairs among returned outputs

The complete-pair success denominator answers how many eligible reference fronts receive accurate pairs. Automatic software also needs to know what is contained in all returned pairs. Reusing the existing accepted-pair diagnostic at corner confidence at least 0.50 gives:

| Reviewed reference | Retained method | Reflected-box method |
| --- | ---: | ---: |
| Returned pairs passing the point-score screen | 678 | 669 |
| Matched complete pairs with both errors at most 5 pixels | 569 | 569 |
| Matched complete pairs exceeding 5 pixels | 14 | 14 |
| Matched fronts with an incomplete reference pair | 48 | 50 |
| Unmatched returned pairs | 47 | 36 |

The last four rows partition the first row. The reference supports 569/678 (83.92%) and 569/669 (85.05%) of returned pairs at five pixels. Outputs without complete reference support are unverified, not all proven physically wrong. This diagnostic is neither calibrated confidence nor a measured robot-grasp success rate. The five- and ten-pixel criteria remain image-space evaluation choices, not established physical positioning tolerances.

## 5. The smallest useful next steps

**Finish the prepared classification-loss comparison.** Quality Focal Loss supplies a small, research-based change in how training emphasizes score errors [2]. The implementation retains the native continuous targets, assignment, box and corner objectives. It applies squared probability-to-target disagreement to the existing classification BCE. It does not introduce localization awareness into a previously hard-label-only detector or reproduce the full GFL architecture. Its aggregate loss scale also changes, so the first result measures the whole objective intervention.

Evaluate that return with fixed-threshold misses/extras, the already planned AP50 ranking measure, corner success, and the availability groups above. A confidence-scale change alone should not be described as better recognition. The returned original/reflected caches can later reproduce this fixed anchoring comparison without another GPU training run.

**Inspect the physical meaning of the difficult regions before another label change.** Use the [annotation definition](../annotation-definition.md), unmarked photographs and stable row identities. Review the ten near-overlap misses and the four high-extra photographs first, followed by a complete-image sweep. An unmatched region must not automatically become a negative example.

**Choose any later model change from the observed error.** Verified training-only rack/gripper confusions would support a controlled difficult-example sampling experiment, motivated by hard-example mining [3]. Preserve all real fronts in every training image or crop. This is distinct from simply increasing the weight of every partial front, which already increased extras in the [earlier experiment](../partial-classification-r51-results.md). A second verifier would add a model and rejection threshold while leaving absent detections unresolved. It needs a demonstrated benefit before being added.

**Change suppression only when its cost is measured.** Soft-NMS was developed to reduce premature removal of overlapping objects [4]. It cannot recover proposals that have already been discarded, and reducing a hard-NMS overlap cutoff suppresses more neighboring candidates. The project already contains suppression experiments; five current duplicates alone do not justify repeating a broad sweep or replacing the detector family.

## 6. Remaining evidence gaps

The annotation definition is documented, but an independent repeatability study has not measured human agreement. Five pixels is not a proven annotation-accuracy bound. The reference sensitivity is already large enough to justify keeping all three comparisons and adjudicating physical disagreements without selecting whichever label favors a model [5].

The 20 photographs have guided repeated review and method selection. They remain development data. A new split of the original training photographs cannot by itself create an unseen evaluation if initialization still uses the supplied checkpoint trained on those photographs. Future group-held-out or out-of-fold experiments must exclude each held-out group from the entire checkpoint training history. This follows from leakage control [6]; it does not invalidate the prepared conditional fine-tuning comparison, whose starting checkpoint is intentionally identical in both arms.

The remaining output-interface choice is whether automatic downstream software can explicitly receive a detected front without a trusted coordinate pair. That choice affects a later acceptance stage. It does not require changing the current detection target or guessing missing annotation coordinates.

## References

[1] D. Bolya, S. Foley, J. Hays, and J. Hoffman, “TIDE: A general toolbox for identifying object detection errors,” ECCV, 2020. [Paper](https://arxiv.org/abs/2008.08115).

[2] X. Li et al., “Generalized Focal Loss: Learning qualified and distributed bounding boxes for dense object detection,” NeurIPS, 2020. [Paper](https://proceedings.neurips.cc/paper/2020/file/f0bda020d2470f2e74990a07a607ebd9-Paper.pdf).

[3] A. Shrivastava, A. Gupta, and R. Girshick, “Training region-based object detectors with online hard example mining,” CVPR, 2016. [Paper](https://arxiv.org/abs/1604.03540).

[4] N. Bodla, B. Singh, R. Chellappa, and L. S. Davis, “Soft-NMS — Improving object detection with one line of code,” ICCV, 2017. [Paper](https://arxiv.org/abs/1704.04503).

[5] J. Murrugarra-Llerena and C. R. Jung, “Noise-aware evaluation of object detectors,” WACV, 2025. [Paper](https://openaccess.thecvf.com/content/WACV2025/html/Llerena_Noise-Aware_Evaluation_of_Object_Detectors_WACV_2025_paper.html).

[6] S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” Patterns, vol. 4, no. 9, 2023. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/).
