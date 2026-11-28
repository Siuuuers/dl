# Annotation references and training effects

Updated 28 November 2026, Asia/Hong_Kong.

**Both methods have now been scored against the untouched supplied annotations as well as the earlier fixed and reviewed annotations. New model inference is unnecessary for this question: the photographs and predictions stay unchanged, and the reference used to measure them changes.**

## Two different questions

| Question | Required comparison | What it establishes |
| --- | --- | --- |
| How do the original and proposed methods score using previous annotations? | Score the same saved predictions against each reference, applying the same rule to both methods | Sensitivity of the measured comparison to the evaluation annotations |
| Did changing the training annotations improve the model? | Train two independent copies from the same initializer on the same photographs and recipe, changing only the training labels; score both against one fixed reference | The effect of that specified training-label intervention under the recorded conditions |

Annotations are not an input to the detector during ordinary inference. Passing the same photograph through an unchanged model again does not expose it to a different annotation version. The reference enters the scoring procedure afterward. A change in the measured score after rescoring is therefore not additional learning.

The proposed method combines an adapted detector, reviewed training labels, five additional training photographs and two-view coordinate averaging. Its comparison with the supplied checkpoint measures that combined method. It cannot assign the full gain to the annotation edits alone.

## The completed reference matrix

The comparison uses the same fresh inference archive from [the original-model study](original-comparison.md). It retains confidence strictly above 0.25, endpoint confidence at least 0.50, matching overlap at least 0.50, and identical additional suppression at overlap above 0.70. The proposed method filters both views before association and averages 648 of 687 adapted original-view records. No weight, threshold, model or annotation is selected by this matrix calculation.

| Evaluation annotations | Method | Missed fronts | Extra predictions | Duplicates | Both corners within 5 pixels | Both within 10 pixels |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Supplied: 664 fronts | Original supplied model | 39 | 56 | 11 | 521/584 | 568/584 |
| Supplied: 664 fronts | Proposed method | 46 | 69 | 8 | 514/584 | 565/584 |
| Earlier fixed: 666 fronts | Original supplied model | 26 | 41 | 11 | 534/584 | 580/584 |
| Earlier fixed: 666 fronts | Proposed method | 34 | 55 | 9 | 525/584 | 574/584 |
| Reviewed: 654 fronts | Original supplied model | 34 | 61 | 5 | 555/583 | 577/583 |
| Reviewed: 654 fronts | Proposed method | 18 | 51 | 5 | 569/583 | 583/583 |

Duplicates are included in extra predictions. Every complete reference pair remains in the denominator, including missed detections. The original model returns the same 681 retained records for every reference; the proposed method returns the same 687 records. Only their reference assignments and measured errors change.

Under the supplied annotations, the proposed method gains 16 and loses 23 individual five-pixel successes, a net loss of seven. At ten pixels it gains four and loses seven. Under the earlier fixed annotations, five-pixel success falls by nine. Under the reviewed annotations, it rises by 14. These references contain different entries and coordinates, so their counts must not be pooled into a larger test set or interpreted as identical row-by-row objects across versions.

The result supports a precise conclusion: the proposed method agrees better with the reviewed annotation standard on these development photographs, while the supplied model agrees better with the two earlier standards on several main measures. It does not establish that an annotation version is correct merely because one model scores higher.

## Supplied-reference provenance

The supplied 664-front reference is distinct from the earlier fixed 666-front reference. The previous four-reference comparison included the latter but not the untouched supplied labels.

The original manifest was fetched directly from `an94-doll/embedding_vision` at immutable historical baomai revision `351eb3cf960ccc34829e2bc72e1fa4645acc6295`, path `data/manifest.json`. Its 127,216 bytes are identical to this repository's original manifest, with SHA-256 `72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926`. All 20 original validation-label files match its recorded hashes, and all 20 photograph hashes match the completed inference archive. The [provenance receipt](evidence/annotation-reference/upstream-provenance.json) records these checks.

The source prediction ZIP has SHA-256 `aae05d14298afad0263197fe4ab7e0e753f63445d9d3503dc1b006030e16c516`. Its complete inventory, original/proposed coordinates and suppression are verified before rescoring. The earlier fixed and reviewed scores reproduce exactly.

## Existing controlled training evidence

An earlier [matched annotation-training experiment](../annotation-confirmation-2026-11-11.md) already isolates a small training-label update. Both arms start independently from the original epoch-173 checkpoint and use the same 137 photographs, seed 46, 30-epoch recipe and A100 runtime. The candidate changes five additions, six deletions and three retained rows across eight training images. The original-label arm has 3,772 fronts and the reviewed-label arm 3,771; the photographs are identical. Its fixed evaluation reference contains 661 fronts and 584 complete pairs.

| Earlier controlled experiment | Original training labels | Reviewed training labels |
| --- | ---: | ---: |
| Missed fronts | 24 | 24 |
| Extra predictions | 58 | 52 |
| Both corners within 5 pixels | 527/584 | 529/584 |
| Both corners within 10 pixels | 578/584 | 577/584 |

This supports fewer extra predictions for that label intervention at its declared operating point, with mixed corner results. It does not establish the effect of every subsequent annotation revision or the current 142-photograph training set. Its historical scoring procedure also predates the additional common postclip suppression in the current matrix; the two tables must not be merged as one experiment.

If a later thesis claim needs to isolate the latest training-label effect, a clean new pair could use the shared 137 photographs, original versus latest accepted training labels, the same initializer, fixed training settings and one fixed evaluation reference. That would answer a new causal question. The present report can already describe the combined method and its measured annotation sensitivity without repeating model inference or claiming that training-label causality has been established.

## Research basis

Murrugarra Llerena and Jung [1] examine how missing, incorrect and imprecise reference annotations affect object-detection evaluation. Their work shows why agreement scores can change with annotation quality and box-boundary uncertainty. It motivates retaining alternative-reference measurements; this project continues to use its declared matching rule and does not implement their alternative metric.

ObjectLab [2] uses predictions to prioritize possible overlooked objects, misplaced boxes and class-label errors for review. This supports model-assisted annotation auditing, while the physical image evidence and a consistent annotation policy must decide an accepted correction. Prediction disagreement alone does not prove a label is wrong.

The present annotation review and method selection have reused the same 20 development photographs. The three reference versions do not remove that dependence. Independently reviewed new capture groups remain needed to assess transfer.

## Notebook and evidence

Run [the CPU annotation-reference notebook](../../notebooks/baomai_annotation_reference_r60.ipynb) to reproduce this analysis. It verifies the published source and inputs, recomputes all six displayed cells, checks the earlier scores, and exports one evidence ZIP. It downloads no model weights or photographs and starts no training.

All four exact notebook code cells have completed from a fresh checkout of published source `8ee1d751817638d72eb7b385a7f4164478a2b7e8`. The driver SHA-256 is `2939b2789abf2566d7680de60a0a87efae2a41e45c2dc01c9d15d1221cb8bbf1`. Source and input verification, full score reconstruction, saved-result replay and packaging passed. Execution used an in-process IPython shell; the Colab browser-download interface was not exercised. Repeating the notebook is optional reproduction, not a prerequisite for using the completed results.

- [Driver](../../scripts/run_reference_matrix_r60.py)
- [Notebook builder](../../scripts/build_reference_matrix_r60_notebook.py)
- [Matrix summary](evidence/annotation-reference/summary.json)
- [Provenance and validation](evidence/annotation-reference/README.md)

## References

[1] J. Murrugarra Llerena and C. R. Jung, “Noise-Aware Evaluation of Object Detectors,” in *Proc. IEEE/CVF Winter Conf. Applications of Computer Vision*, 2025, doi: 10.1109/WACV61041.2025.00903. [Primary publication](https://openaccess.thecvf.com/content/WACV2025/html/Llerena_Noise-Aware_Evaluation_of_Object_Detectors_WACV_2025_paper.html).

[2] U. Tkachenko, A. Thyagarajan and J. Mueller, “ObjectLab: Automated Diagnosis of Mislabeled Images in Object Detection Data,” in *ICML Workshop on Data-centric Machine Learning Research*, 2023. [Author manuscript](https://arxiv.org/abs/2309.00832).
