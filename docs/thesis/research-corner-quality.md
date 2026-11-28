# What the saved predictions reveal about corner accuracy

28 November 2026, Asia/Hong_Kong.

## Finding and recommended next experiment

The most direct next accuracy experiment is a small second-stage model that refines the two lower corners inside each detected cassette front. A separate confidence study can examine whether changes between image views help identify unreliable coordinates. These address different outcomes: refinement changes a coordinate, whereas a reliability score only describes or screens an existing coordinate.

On the reviewed annotations, the unchanged control already detects every one of the 583 fronts for which both lower corners are annotated. Seventeen of their returned corner pairs have an error greater than five pixels at one or both ends. All seventeen retain a minimum corner score of at least 0.8. Raising that score requirement to 0.8 therefore would not remove these particular geometric errors.

A fixed crop feasibility check supports a refinement experiment: enlarging each predicted bounding box by a factor of 1.5 contains both annotated corners in all 583 matched complete fronts, including all seventeen inaccurate pairs. This establishes that the proposed crop includes the annotated locations. It does not establish how much a new model would improve them, whether the appearance around each annotation is unambiguous, or how unseen images will behave.

These are new CPU analyses of saved outputs, not new model predictions or training results. All images are the same twenty development photographs used previously. References, predictions, and scoring rules were unchanged.

## 1. Separate front detection from corner placement

A cassette can be detected correctly while one of its proposed lower corners is misplaced. Conversely, a recognisable partial cassette may legitimately lack one or both measurable lower corners. Combining these cases into a single success rate hides which part needs improvement.

The following table partitions every reference front that has two annotated lower corners. A complete pair is successful only if both returned coordinates lie within five pixels of their respective reference locations. All pixel distances are measured in the original 1280 by 720 image, after any model-input resizing has been reversed.

| Reference and saved model | All fronts | Fronts with two annotated corners | Missed complete fronts | Matched but pair unavailable | Matched pairs over 5 px | Both corners within 5 px |
|---|---:|---:|---:|---:|---:|---:|
| Fixed reference, unchanged control | 666 | 584 | 7 | 0 | 66 | 511 |
| Fixed reference, weighted candidate | 666 | 584 | 7 | 0 | 66 | 511 |
| Reviewed reference, unchanged control | 654 | 583 | 0 | 0 | 17 | 566 |
| Reviewed reference, weighted candidate | 654 | 583 | 0 | 0 | 16 | 567 |

The final four columns sum to the complete-reference count. The weighted candidate changes the training emphasis for partial fronts; it is not the model used for the comparison-view confidence analysis below. These saved outputs come from different experiment records. The table is a descriptive comparison, not a new controlled training trial. The fixed 666-front reference is distinct from the upstream validation reference.

The complete-reference partition does not capture the full automatic-output problem. At the same native box-score cutoff, the unchanged control has 34 missed fronts and 91 unmatched outputs under the fixed reference, of which 17 outputs are duplicates. Under the reviewed reference, it has 15 missed fronts and 84 unmatched outputs, including 13 duplicates. All fifteen reviewed-reference misses are among the 71 fronts with fewer than two annotated corners. The weighted candidate reduces reviewed-reference misses from 15 to 11 but increases unmatched outputs from 84 to 112 and duplicates from 13 to 23. Thus a corner refiner should preserve the detector's output set during its initial experiment; it cannot be credited with recovering missed fronts or removing extra detections merely by improving coordinates.

The two reference versions produce very different geometric-error counts. They must remain separate in the report. A change in reference annotations is not a change in model performance on new images.

## 2. Do confidence scores rank geometric error?

### 2.1 Fixed diagnostic cohort

The diagnostic uses the unchanged control from the saved enlarged-input experiment, with native box cutoff greater than 0.13, reference matching at box intersection-over-union of at least 0.50, and a returned corner requiring a score of at least 0.50 plus finite, in-image coordinates. A prediction is matched to one reference front according to the existing confidence-ordered one-to-one scorer. The coordinate error is the larger Euclidean error of its two corners.

Ranking comparisons use only matched fronts that have two annotated corners and both returned coordinates. This gives 577 pairs under the fixed reference and 583 under the reviewed reference. The seven missed complete fronts under the fixed reference remain visible in the table above. They do not receive invented corner errors. Incomplete-reference and unmatched outputs are not certified by these conditional ranking results.

Three initial features were checked:

1. The detection's box score.
2. The smaller of its two corner scores.
3. The overlap between its predicted box and the annotated box.

Lower values were specified as the putative risk direction for all three. The third feature is available only during evaluation because it uses the annotation. It cannot be calculated by the final application without a reference box.

### 2.2 Comparison-view disagreement uses predictions only

Two existing alternative views were used: local luminance contrast enhancement and an enlarged model input of 1600 pixels instead of 1280 pixels. Both sets of saved coordinates already refer to the original image. The contrast operation changes appearance without changing image geometry. Enlargement changes model sampling; interpolation does not add captured detail.

Each original prediction was linked to an alternative prediction using **only predicted boxes and their classes**. The pair had to be the best overlap match in both directions and have box overlap of at least 0.50. The threshold was fixed. Equal overlaps use the original output index as a deterministic tie-break. Reference identities and corner coordinates were not used to select these cross-view links.

For a linked pair, corresponding lower-corner positions were compared directly. The disagreement feature is the greatest Euclidean displacement across both corner slots and both alternative views. Larger displacement was specified as higher risk. No thresholds were fitted to classify or discard outputs.

Every original pair in the diagnostic cohort had usable linked outputs in both views: 577 of 577 and 583 of 583. There were no missing view associations, unavailable associated pairs, or exact maximum-overlap ties in these cohorts. However, ten contrast links and thirteen enlarged-view links had more than one eligible overlap candidate in at least one direction. These cases were retained, not silently removed. A mutual best match is an algorithmic correspondence, not proof of physical identity.

After these prediction-only links were fixed, their reference assignments were audited. Under the reviewed reference, all 583 links per view refer to the same evaluation target as the original output. Under the fixed reference, 576 links agree and one alternative output is unassigned in each view; none links to a different assigned target. This audit does not alter the association or the ranking cohort.

### 2.3 Ranking results

The area under the receiver operating characteristic curve, abbreviated AUROC, is used here as a ranking measure. It equals the fraction of a failing-pair versus successful-pair comparisons for which the proposed risk score places the failing pair above the successful pair; ties count as one-half. A value of 0.5 corresponds to uninformative ordering and 1.0 to perfect ordering in this recorded sample. It is not a probability that an individual coordinate is correct.

| Fixed feature and risk direction | Fixed reference, 577 pairs, 66 failures | Reviewed reference, 583 pairs, 17 failures |
|---|---:|---:|
| Lower box score | 0.583 | 0.532 |
| Lower minimum corner score | 0.392 | 0.398 |
| Lower reference-box overlap, evaluation only | 0.560 | 0.721 |
| Greater comparison-view corner displacement | 0.745 | 0.839 |

All 66 fixed-reference failures and all 17 reviewed-reference failures have minimum corner score at least 0.8. Their median minimum corner scores are 0.943 and 0.942, compared with 0.929 and 0.930 among successful pairs. The conventional interpretation that a smaller corner score reliably signals a larger coordinate error does not hold for this selected cohort. Reversing the score direction after seeing these results would be a newly fitted rule and is not recommended by this analysis.

The disagreement feature ranks the pooled errors more effectively, but the photographs are not independent collections of isolated objects. Twenty-seven of the 66 fixed-reference failures and eight of the 17 reviewed-reference failures occur in a single photograph. A score may appear effective partly because it distinguishes an easier photograph from a harder one.

To check this, the same ranking comparison was repeated using only failing and successful pairs from the same photograph. All such within-image comparisons were pooled, weighted by their number of possible failing-successful pairs; no image or failure was selected according to a feature score.

| Fixed feature | Fixed reference, within-image AUROC | Reviewed reference, within-image AUROC |
|---|---:|---:|
| Lower box score | 0.542 | 0.468 |
| Lower minimum corner score | 0.447 | 0.484 |
| Lower reference-box overlap, evaluation only | 0.689 | 0.565 |
| Greater comparison-view corner displacement | 0.587 | 0.659 |

Thirteen photographs have both successful and failing complete pairs under the fixed reference; seven do under the reviewed reference. The fall from pooled to within-image discrimination substantially weakens any automatic-screening claim. Some incorrect pairs also show small view differences: one reviewed-reference pair has a worst corner error of 6.61 px while its maximum view displacement is 1.07 px. Agreement alone is not a geometric guarantee.

**Interpretation:** retain disagreement as a candidate diagnostic feature. A new experiment must determine whether it helps on genuinely separate acquisition scenes before using it to suppress automatic outputs. The present evidence does not select a rejection cutoff, establish calibration, or establish application-level reliability.

## 3. Fixed crop feasibility

The crop was defined before its containment calculation: expand each matched predicted box by a factor of 1.5 about its centre, round minimum coordinates down and maximum coordinates up, and clip the crop to the image. The same rule was used for both saved models and all four reference versions. No margin was optimized against failures. This is an exploratory follow-up, not an experiment registered before the historical model runs.

| Reference and saved model | Both annotated corners inside crop / matched complete targets | Inaccurate pairs contained / pairs over 5 px | Crops clipped by image boundary | Inaccurate-pair crops clipped by boundary |
|---|---:|---:|---:|---:|
| Fixed reference, unchanged control | 577/577 | 66/66 | 34 | 11 |
| Fixed reference, weighted candidate | 577/577 | 66/66 | 34 | 11 |
| Reviewed reference, unchanged control | 583/583 | 17/17 | 34 | 5 |
| Reviewed reference, weighted candidate | 583/583 | 16/16 | 34 | 5 |

No matched complete target had either annotated lower corner outside its crop. No examined reference corner lay outside the original image. The seven missed complete fronts under the fixed reference do not have matched detector crops and cannot be counted as refinement successes.

This supports a concrete second-stage experiment: hold the original detections and reference assignments fixed, take the same predetermined crop around each front, jointly refine the two lower corners, and measure whether complete-pair error improves. Train on training images with the validation images excluded. Generate predicted-box training crops without using validation outputs to define labels or tune the crop. Keep image groups separate when related photographs show the same physical arrangement. Report both all-eligible pair success and the fixed matched-cohort coordinate changes, alongside the unchanged front-detection counts. A stronger score alone must not count as corrected geometry.

A crop can contain neighbouring fronts, especially in dense arrangements. Supplying the original corner proposals and predicting the two corners jointly is one hypothesis for retaining front identity; it still requires testing. Visibility or availability labels remain necessary. An unobserved corner must not be supplied with an invented training coordinate.

## 4. Two relevant primary papers

### 4.1 Quality scores must match the quantity being evaluated

VarifocalNet learns a score combining object presence with bounding-box overlap quality [1]. Its relevance is methodological: ordinary category confidence and accurate localization are distinct targets. Its box-oriented score and loss do not provide a five-pixel guarantee for a cassette landmark. A future landmark-quality output would need supervision and evaluation that explicitly concern corner error. Adding Varifocal Loss to the existing native model should not be described as a tested or compatible solution without implementation work.

### 4.2 Regression uncertainty is different from a visibility score

Residual Log-Likelihood Estimation learns a flexible residual distribution for coordinate regression using a likelihood objective [2]. It provides relevant background for modeling coordinate error. The recorded project already preserves a native RLE loss in its pose model, so proposing to “add RLE” would repeat an existing component. The saved third corner value is not thereby established as a calibrated probability of being within five pixels. Testing a learned error measure would require its actual outputs, a clearly specified interpretation, and separate validation evidence.

These papers motivate the distinction between presence, availability, and geometric quality. Neither supplies measured evidence that the current cassette system will improve.

## 5. Evidence and reproduction

The analysis replayed all eight original reference/candidate combinations in the input snapshot, reproducing every saved per-image score and aggregate exactly. The two saved controls were also checked for identical box and corner coordinates. Their confidence values may differ by small floating-point amounts, so the enlarged-input run's unchanged control is used consistently for the ranking study.

The reported AUROC values use direct pairwise comparisons. A separate script reproduced all 28 feature/cohort/reference AUROC values through the independent Mann–Whitney rank-sum identity to absolute tolerance 1e-12. No significance tests, fitted thresholds, or calibrated probabilities are reported. Outputs within photographs are correlated.

The three analysis scripts are in `scripts/`. The evidence files below are in `docs/thesis/evidence/corner-quality/`; see the [portable reproduction instructions](evidence/corner-quality/README.md).

- `analyze_corner_quality.py`: fixed-cohort ranking, prediction-only view association, sensitivity across four references, and image-conditioned comparisons.
- `corner-quality-summary.json`: aggregate results, denominators, association diagnostics, and source hashes.
- `corner-quality-rows.json.gz`: per-target errors and comparison-view details.
- `check_corner_crop_containment.py`: fixed factor 1.5 crop calculation for unchanged control and weighted candidate.
- `crop-containment-summary.json` and `crop-containment-rows.json.gz`: all-eligible accounting, detection counts, containment counts, and per-target crop coordinates.
- `verify_corner_quality_ranking.py` and `ranking-verification.json`: independent ranking verification.
- `protocol.md`: original diagnostic choices and explicit exploratory amendments.

The JSON retains the internal reference keys `frozen666`, `historical663`, `r38`, and `r50` for reproducibility. The prose uses descriptive names. The older reference and fixed reference have identical complete-pair counts, while the intermediate reviewed and latest reviewed references have identical complete-pair counts; their box-overlap ranking can differ.

The cached source paths, checked by SHA-256, are supplied as command-line arguments rather than silently discovered. The analysis consumes the R54 input snapshot for reference data and weighted-candidate outputs, the PR149 saved contrast outputs, and PR150 saved enlarged-input outputs. These identifiers belong in a technical evidence ledger, not in the explanatory narrative of the undergraduate report.

## References

[1] H. Zhang, Y. Wang, F. Dayoub, and N. Sünderhauf, “VarifocalNet: An IoU-aware dense object detector,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 8514–8523. [Primary publication](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.html).

[2] J. Li, S. Bian, A. Zeng, C. Wang, B. Pang, W. Liu, and C. Lu, “Human pose regression with residual log-likelihood estimation,” in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2021, pp. 11025–11034. [Primary publication](https://openaccess.thecvf.com/content/ICCV2021/html/Li_Human_Pose_Regression_With_Residual_Log-Likelihood_Estimation_ICCV_2021_paper.html); [author manuscript](https://arxiv.org/abs/2107.11291).
