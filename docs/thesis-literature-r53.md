# R53: primary literature and falsifiable cassette hypotheses

27 November 2026. **Status: research plan; no new model or preprocessing result.**

This evidence ledger extends the [R52 thesis plan](thesis-plan-r52.md). The
recommended thesis remains an empirical study of annotation quality, review
prioritization, and their effect on cassette-front detection and corner
localization. The papers motivate comparisons; they do not establish that a
particular filter, architecture, or annotation revision will improve this
application, and they do not establish novelty for the proposed thesis.

**Immediate decision:** retain the existing pose model and use two separate,
fixed-checkpoint inference probes: one photometric change, and one input-size
change. Keep label-only retraining conditional. Avoid a combined grid of
filters, resolutions, models, and loss weights whose effects cannot be
separated.

## 1. What the primary evidence supports

The final column contains project-specific inferences, not conclusions from
the cited papers.

| Primary work | Finding relevant to this project | Limit and project inference |
| --- | --- | --- |
| [Ma et al., 2022][ma2022] | Professional reannotation with explicit category guidelines produced mixed detector outcomes across categories and datasets. | Cleaner supervision is not a guaranteed score increase. Evaluate review quality separately from downstream model performance; retain negative results. |
| [ObjectLab, 2023][objectlab] | Model predictions can prioritize overlooked, mislocated, and wrongly classified annotations. Its method obtains out-of-sample predictions using five-fold cross-validation. | Prediction disagreement is a review signal. Do not silently replace labels with predictions, or call in-sample self-agreement an unbiased audit. Our single-class task makes mislocation and missing-instance review more relevant than class swaps. |
| [TIDE, 2020][tide] | Separates six detection error categories and isolates their effects on AP. | Its taxonomy motivates a diagnosis, but our fixed-threshold match counts are not TIDE's oracle AP computation. Keep corner quality as an additional outcome. |
| [Murrugarra-Llerena and Jung, 2025][noise2025] | Annotation boundary discrepancies can change AP, particularly for small objects and strict IoU thresholds. | Add reference sensitivity beside the unchanged primary scoring rule. A score increase caused only by a reference revision does not show that the model learned more. |
| [Zuiderveld, 1994][clahe] | CLAHE limits local histogram enhancement to reduce the noise amplification associated with ordinary adaptive equalization. | This is an image-processing method, not evidence of better cassette detection. Try one declared setting and measure detector outcomes; a more attractive image can still be a worse model input. |
| [Liu et al., 2022: IA-YOLO][iayolo] | A small network predicts differentiable filter parameters, and the filtering module and YOLOv3 are trained jointly for normal and adverse-weather images. | Demonstrates a mechanism for learned filtering. Its fog/low-light experiments do not establish benefits for laboratory cassette images or our existing pose checkpoint. A fixed CLAHE probe is not an IA-YOLO reproduction. |
| [Lin et al., 2017: FPN][fpn] | Top-down and lateral feature connections combine semantic information at multiple feature resolutions. | Feature scale is relevant, but changing image size is not an FPN architecture change. Do not infer that our current network lacks multiscale features. |
| [Akyon et al., 2022: SAHI][sahi] | Overlapping image slices and coordinate merging improve small-object detection on the studied aerial datasets; sliced fine-tuning is a separate intervention. | Tiling changes context, edge truncation, and duplicate handling. Their box-detection results do not validate two-corner pose merging. Test the simpler size probe first. |
| [He et al., 2019][he2019] | Random initialization can catch pretrained detectors with sufficient data and longer training, but it loses in the paper's very-small-data regimes. | The headline does not justify discarding our trained initializer. Dataset size, independence, task, and optimization budget differ substantially. |

## 2. Two immediate, controlled probes

Both notebooks must pin the **same PR137 checkpoint bytes**, source revision,
runtime, original images, reference labels, matcher, confidence threshold,
suppression settings, and postclip filter. The exact checkpoint identity comes
from the run manifest, not a mutable filename. First replay the ordinary
1280-pixel baseline and compare saved predictions and metrics with the verified
cache. Stop if that guard fails; otherwise a runtime change could be mistaken
for the proposed treatment.

The [PR146 result audit](partial-classification-r51-results.md) documents that
its reference prediction cache reproduces the earlier PR137 candidate. This
permits a baseline identity check; it does not make two caches independent
observations. All current reference versions reuse the repeatedly reviewed
development photographs.

| Hypothesis | Exact comparison | What to measure | Evidence against the bounded claim |
| --- | --- | --- | --- |
| **H-P: a fixed contrast transform improves this checkpoint's operating point.** | Original decoded pixels versus OpenCV BGR-to-Lab, CLAHE on **L only**, `clipLimit=2.0`, `tileGridSize=(8,8)`, then Lab-to-BGR; both at `imgsz=1280`. Preserve a copy of the original and leave labels untouched. | Recovered and newly missed stable IDs; all unmatched outputs and formal duplicates; all-eligible corner success; empty-scene outputs; added latency. Inspect original and transformed crops for affected cases. | No net miss reduction, or an increase in unmatched outputs, duplicates, empty-scene outputs, or corner failures, contradicts the proposed improvement under this conservative operating rule. Failure only rejects this checkpoint/setting/condition claim. |
| **H-S: a larger network input improves the same checkpoint's scale-sensitive errors.** | Unchanged original pixels at `imgsz=1280` versus `imgsz=1600`, filter off in both. Record actual padded tensor dimensions and return boxes and corners to original-image coordinates. | The same outcomes as H-P, plus errors by predeclared front size, peak device memory, and latency. Record effective object size in the input tensor. | No net miss reduction, or degradation in the protected outcomes, does not support this setting. If improvements are absent in the small-front group, the proposed small-object mechanism is unsupported even if another group improves. |

The numerical filter settings are a **prespecified engineering trial**, not
parameters claimed optimal by the literature. The [OpenCV API][opencv-clahe]
specifies a single-channel input and a grid counted in tiles; `(8,8)` means
eight tiles in each direction, not tiles eight pixels wide. Applying the
transform to L retains a three-channel color input after conversion. It still
changes the pixel distribution and may introduce quantization or clipping
effects; preserving the other Lab channels does not promise identical RGB
colors after the round trip.

These are paired deployment probes with frozen weights. A useful result may
motivate a later train-and-infer preprocessing study. An unfavorable result
does not prove that jointly trained preprocessing can never work. Never
overwrite original training photos with filtered images, and never move a
physical boundary because a contrast transform made a different edge look
stronger.

At 1600, a 1280-wide source image is interpolated. It contains no newly captured
detail. The test asks whether representing the same evidence at a different
network scale helps. It is neither a higher-resolution camera experiment nor
SAHI. Keep preprocessing, confidence, NMS, and label definitions unchanged so
the observed effect can be attributed to this size setting.

The [R53 size and visibility audit](size-visibility-r53.md) supplies a
descriptive reason to investigate scale: five of fourteen fronts in its
8–less-than-16-pixel height group are missed in both PR146 arms. However, all
583 fronts with two locatable corners match, while the remaining misses occur
among fronts with fewer available corners. Size, visibility, crowding, and
scene difficulty may therefore be confounded. These bins were selected after
seeing the data and remain exploratory even if reused unchanged in the new
probe. The 654 rows are not 654 independent capture conditions.

### Common reporting and decision rule

Use **frozen666 as the primary reference**, with the established matching
definition; publish R50, R38 and historical663 as diagnostic and
reference-sensitivity checks. A favorable secondary view does not replace the
primary screen. Keep the original
operating threshold. Report the full paired change vector, not only its most
favorable component. A conservative development-screen success requires fewer
misses with no increase in unmatched predictions, formal duplicates, or
empty-scene outputs, no new two-corner misses, and no reduction in all-eligible
corner success. This is
a declared project decision rule, not a universal optimum or an automatic
replacement of the existing model-selection gate.

Include continuous error values and stable case identities beside threshold
counts. A tiny movement across a five-pixel cutoff can change a count without
being a practically large gain. The corner denominator must include eligible
missed targets as failures; also retain the historical matched-only metric for
comparability. Show zero-, one-, and two-available-corner groups and original
front-size distributions. Availability is a diagnostic proxy, not a pure
measurement of occlusion.

Measure latency after equivalent warm-up on the same hardware and precision,
using device synchronization where needed. Report preprocessing and complete
pipeline latency separately. Label one model-timing sample per image as a
measurement, not a reliable tail-latency estimate. Keep CPU/GPU memory and
out-of-memory failures visible rather than silently lowering the candidate
resolution or batch size.

The two probes share a control and development data; they are not independent
confirmatory experiments. Resample images or supported capture groups for any
uncertainty estimate, not thousands of correlated cassette rows as independent
samples. Do not advertise AP from a prediction cache already truncated at the
operating threshold. A proper AP/TIDE extension needs a declared suitable
prediction collection and evaluation protocol.

## 3. Annotation hypotheses remain the thesis core

The [R52 amendment](front-review-r52.md) changes ten existing rows and eleven
box sides on `frame_0160`, with shifts approximately two original-image pixels.
Only that already included photograph changes in the effective 142-image
training membership. The nominal visual uncertainty was also approximately
two pixels. This is a small and potentially low-information intervention, not
an evidence-based reason to expect a large training gain.

| Hypothesis | Controlled comparison and outcomes | What would weaken or refute it |
| --- | --- | --- |
| **H-Q: geometry flags improve human review yield.** | On a predefined, independently adjudicated pool, compare the frozen geometry ranking with seeded random order and sequential review at equal unique-case budgets. Measure confirmed issues found, precision among reviewed flags, unresolved cases, and prospectively recorded active review time. | Equal or lower issue yield, or no measured time saving, does not support the corresponding advantage. Without reviewing unflagged cases, no claim about issue recall is justified. |
| **H-A: accepted corrections improve downstream performance.** | If a meaningful eligible correction set is ready, train baseline-label and corrected-label arms with identical image membership, initialization, native loss, seed, and fixed schedule. Score both on the frozen and separately adjudicated reference versions. | No model advantage under unchanged references does not support improved training. A favorable rescore of unchanged predictions establishes reference sensitivity only. One paired seed cannot establish stability. |
| **H-U: annotation uncertainty is large enough to affect a claimed model difference.** | Hold predictions fixed; compare historical and adjudicated references or a separately declared plausible-boundary sensitivity analysis. Record which target assignments and model rankings change. | Stable rankings and negligible metric changes across the justified range weaken the uncertainty explanation. Arbitrary favorable shifts are not a valid uncertainty model. |

The four-border checker is not a missing-instance detector: it has nothing to
flag when an entire front has no annotation. Include full-frame inspection and
a sample of unflagged cases. A neighbor corner lying inside an axis-aligned
box can result from legitimate projection and occlusion. It is a review cue,
not a geometric impossibility or permission to shrink a label automatically.

For an ObjectLab-style extension, generate predictions from models that did
not train on the scored image. Group related captures before splitting where
provenance allows it. Existing model overlays remain useful human-review aids,
but their use must be distinguished from an out-of-sample label-quality
benchmark. This requirement adds training cost, so it is a conditional
extension rather than a hidden prerequisite for the current geometry study.

## 4. Model type and initialization decisions

The current source asserts a pose task with one `cassette_front` class and
`kpt_shape=[2,3]`; the R51 criterion additionally checks the `Pose26`,
`E2ELoss`, normalized-L1, and RLE contract. See
[pose_checkpoint.py](../scripts/pose_checkpoint.py) and
[partial_classification_r51.py](../scripts/partial_classification_r51.py).
An off-the-shelf box detector would omit the required corners. A segmentation
model would need masks and a defined procedure to recover these particular
physical landmarks. Architecture migration is therefore a new experiment
with an output-contract and annotation-cost decision.

Starting a fresh optimizer run from `cassette_lower_epoch173.pt` is
**fine-tuning from a trained initializer**. It is not random initialization.
Restarting a run and discarding learned weights are different operations.

He et al. found that longer training from random initialization could match
pretraining with 10,000 COCO images, but not with their 1,000- or 3,500-image
subsets; their VOC comparison also failed to catch up. This is dataset-specific
evidence, not a universal minimum-image rule. Our 142 related photographs and
3,999 fronts do not supply 3,999 independent scenes. Retaining the verified
initializer is the practical default; there is no demonstrated need to reset
the weights to solve the current annotation and matching problems.

If initialization later becomes a research question, distinguish two studies:
equal-budget performance and performance after adequate convergence. A
30-epoch scratch-versus-fine-tune pair answers the former only. Preserve
architecture and data, define the checkpoint rule, repeat seeds when needed,
and account for inherited pretraining data/compute. Do not attribute a
difference to initialization if architecture, supervision, or schedule also
changes without being explicitly part of the question.

Consider tiled inference only if the scale probe and case analysis indicate a
remaining scale limitation that justifies the additional complexity. Validate
image transforms, tile-edge truncation, ownership of partial fronts, corner
visibility, mapping into original pixels, and cross-tile duplicate merging.
SAHI's box pipeline does not automatically preserve a two-landmark pose
contract. Keep a full-image control; tiles can remove useful neighboring
context and create fragments. A larger input or SAHI is not proof that an FPN
architecture replacement is needed.

## 5. Primary bibliography

Accessed 27 November 2026. Page ranges below follow the linked open-access
proceedings where available. The OpenCV item is implementation documentation;
the other items are original research or the original algorithm chapter.

1. **Jiaxin Ma, Yoshitaka Ushiku, and Miori Sagara.** 2022.
   *The Effect of Improving Annotation Quality on Object Detection Datasets:
   A Preliminary Study.* Proceedings of the IEEE/CVF Conference on Computer
   Vision and Pattern Recognition Workshops, pp. 4850–4859.
   DOI: [10.1109/CVPRW56347.2022.00532](https://doi.org/10.1109/CVPRW56347.2022.00532).
   [Primary paper][ma2022].

2. **Ulyana Tkachenko, Aditya Thyagarajan, and Jonas Mueller.** 2023.
   *ObjectLab: Automated Diagnosis of Mislabeled Images in Object Detection
   Data.* ICML Workshop on Data-centric Machine Learning Research.
   arXiv:2309.00832v1; DOI: [10.48550/arXiv.2309.00832](https://doi.org/10.48550/arXiv.2309.00832).
   [Author full text][objectlab].
   [Workshop PDF](https://dmlr.ai/assets/accepted-papers/112/CameraReady/objectlab_icml2023_lowerres.pdf).

3. **Daniel Bolya, Sean Foley, James Hays, and Judy Hoffman.** 2020.
   *TIDE: A General Toolbox for Identifying Object Detection Errors.*
   Computer Vision – ECCV 2020, Part III, Lecture Notes in Computer Science
   12348, pp. 558–573. Springer.
   DOI: [10.1007/978-3-030-58580-8_33](https://doi.org/10.1007/978-3-030-58580-8_33).
   [Author preprint][tide]. [Author implementation](https://dbolya.github.io/tide/).

4. **Jeffri Murrugarra-Llerena and Claudio R. Jung.** 2025.
   *Noise-Aware Evaluation of Object Detectors.* Proceedings of the IEEE/CVF
   Winter Conference on Applications of Computer Vision, pp. 9304–9313.
   [Primary publication and citation][noise2025].
   [IEEE record](https://ieeexplore.ieee.org/abstract/document/10943665).
   [Authors' implementation](https://github.com/Artcs1/Error-Aware).

5. **Karel Zuiderveld.** 1994.
   *Contrast Limited Adaptive Histogram Equalization.* In Paul S. Heckbert
   (ed.), Graphics Gems IV, Academic Press, pp. 474–485.
   DOI: [10.1016/B978-0-12-336156-1.50061-6](https://doi.org/10.1016/B978-0-12-336156-1.50061-6).
   [Original chapter, university-hosted scan][clahe].

6. **Wenyu Liu, Gaofeng Ren, Runsheng Yu, Shi Guo, Jianke Zhu, and Lei Zhang.**
   2022. *Image-Adaptive YOLO for Object Detection in Adverse Weather
   Conditions.* Proceedings of the AAAI Conference on Artificial Intelligence
   36(2), pp. 1792–1800.
   DOI: [10.1609/aaai.v36i2.20072](https://doi.org/10.1609/aaai.v36i2.20072).
   [Author full text with appendix][iayolo].
   [AAAI proceedings PDF](https://cdn.aaai.org/ojs/20072/20072-13-24085-1-2-20220628.pdf).

7. **Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan,
   and Serge Belongie.** 2017. *Feature Pyramid Networks for Object Detection.*
   Proceedings of the IEEE Conference on Computer Vision and Pattern
   Recognition, pp. 2117–2125. [Primary publication and citation][fpn].
   [Author preprint](https://arxiv.org/abs/1612.03144).

8. **Fatih Cagatay Akyon, Sinan Onur Altinuc, and Alptekin Temizel.** 2022.
   *Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection.*
   IEEE International Conference on Image Processing, pp. 966–970.
   DOI: [10.1109/ICIP46576.2022.9897990](https://doi.org/10.1109/ICIP46576.2022.9897990).
   [Author full text, arXiv:2202.06934v5][sahi].
   [Authors' implementation](https://github.com/obss/sahi).

9. **Kaiming He, Ross Girshick, and Piotr Dollár.** 2019.
   *Rethinking ImageNet Pre-Training.* Proceedings of the IEEE/CVF International
   Conference on Computer Vision, pp. 4918–4927.
   DOI: [10.1109/ICCV.2019.00502](https://doi.org/10.1109/ICCV.2019.00502).
   [Author full paper][he2019].
   [CVF publication and citation](https://openaccess.thecvf.com/content_ICCV_2019/html/He_Rethinking_ImageNet_Pre-Training_ICCV_2019_paper.html).

10. **OpenCV contributors.** *cv::CLAHE Class Reference.* OpenCV 4.x
    documentation (served as 4.13.0 when accessed).
    [Official API][opencv-clahe]. Record the actual installed version in each
    experiment; a documentation URL is not a runtime version pin.

[ma2022]: https://openaccess.thecvf.com/content/CVPR2022W/VDU/papers/Ma_The_Effect_of_Improving_Annotation_Quality_on_Object_Detection_Datasets_CVPRW_2022_paper.pdf
[objectlab]: https://arxiv.org/html/2309.00832v1
[tide]: https://arxiv.org/abs/2008.08115
[noise2025]: https://openaccess.thecvf.com/content/WACV2025/html/Llerena_Noise-Aware_Evaluation_of_Object_Detectors_WACV_2025_paper.html
[clahe]: https://www.cse.unr.edu/~bebis/CS474/StudentPaperPresentations/Constrast-Limited%20Adaptive%20Histogram%20Enhancement.pdf
[iayolo]: https://arxiv.org/html/2112.08088v3
[fpn]: https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html
[sahi]: https://arxiv.org/html/2202.06934v5
[he2019]: https://arxiv.org/pdf/1811.08883
[opencv-clahe]: https://docs.opencv.org/4.x/d6/db6/classcv_1_1CLAHE.html
