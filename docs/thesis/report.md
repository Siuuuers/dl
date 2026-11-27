# Cassette Front Detection and Corner Localization

Final Year Project I Report

Department of Electronic Engineering

## 1 Abstract

This project improves the detection of tissue cassette fronts and the location of their two lower corners in camera images. Closely arranged objects and partial occlusion can cause missed fronts, repeated detections, and inaccurate coordinates. The work combines consistent annotation review, adaptation of a supplied pose-estimation model, and equal averaging of corner estimates from original and horizontally reflected images.

The comparison uses the untouched supplied model as its baseline. Both methods are evaluated on the same 20 photographs with common confidence and matching settings. Against the reviewed annotations, the proposed method reduces missed fronts from 34 to 18 and extra predictions from 61 to 51. Duplicate predictions remain at five. The number of fronts with both corners within five pixels increases from 555 to 569 out of 583; at ten pixels it increases from 577 to 583. Two additional local corner models are tested, but neither exceeds the proposed method's five-pixel result.

The improvement depends on the annotation reference. An earlier reference favours the supplied model, showing why annotation consistency must accompany numerical comparisons. The contribution is a cassette-specific annotation procedure and an evaluated recognition method with traceable detection and coordinate measurements. Further work will test learned coordinate corrections and assess the method on independently captured images.

## 2 Table of Contents

<!-- DOCX_TOC_START -->
- [1 Abstract](#1-abstract)
- [2 Table of Contents](#2-table-of-contents)
- [3 Introduction](#3-introduction)
  - [3.1 Problem and aim](#31-problem-and-aim)
  - [3.2 Related research](#32-related-research)
- [4 Background](#4-background)
  - [4.1 Annotations and detection measurements](#41-annotations-and-detection-measurements)
  - [4.2 Corner accuracy and confidence](#42-corner-accuracy-and-confidence)
- [5 Methodology and Implementation](#5-methodology-and-implementation)
  - [5.1 Data and annotation review](#51-data-and-annotation-review)
  - [5.2 Proposed recognition method](#52-proposed-recognition-method)
  - [5.3 Local corner models](#53-local-corner-models)
  - [5.4 Common evaluation procedure](#54-common-evaluation-procedure)
- [6 Simulation Results](#6-simulation-results)
- [7 Experimental Results](#7-experimental-results)
  - [7.1 Comparison with the supplied model](#71-comparison-with-the-supplied-model)
  - [7.2 Results of the local corner models](#72-results-of-the-local-corner-models)
- [8 Discussions](#8-discussions)
  - [8.1 Findings and interpretation](#81-findings-and-interpretation)
  - [8.2 Next investigation](#82-next-investigation)
- [9 Conclusions](#9-conclusions)
- [10 References](#10-references)
- [11 Appendices](#11-appendices)
  - [11.1 Sensitivity to the earlier annotations](#111-sensitivity-to-the-earlier-annotations)
  - [11.2 Training and coordinate settings](#112-training-and-coordinate-settings)
<!-- DOCX_TOC_END -->

## 3 Introduction

### 3.1 Problem and aim

The system must identify each tissue cassette front in a colour image and return its lower-left and lower-right corner coordinates. These points describe the position and orientation of the lower edge. The application consumes predictions automatically, so a missed front, an extra response, and an inaccurate corner can each affect its operation.

Cassettes are closely arranged in a rack. A front may be partly covered by another cassette or a gripper, or cut by the image boundary. The required rectangle follows the cassette's own visible front material; rack and gripper boundaries must not define the object. Figure 1 illustrates the target.

![A cassette front and its two lower corners](figures-r57/figure-1-target.png)

Figure 1. Camera image and enlarged cassette front. The rectangle marks the annotated front and the two points mark its actual lower corners. Neighbouring cassettes and the rack provide context. A corner is marked unavailable when its position cannot be identified reliably.

The aim is to improve recognition quality while keeping the processing method clear. The project starts from a supplied image collection and trained cassette model. Its main comparison is between that original model and the proposed method, evaluated with the same measurement rules. The target output is every identifiable front, including a partial front, together with whichever corner locations can be supported by the image.

### 3.2 Related research

You Only Look Once, or YOLO, predicts objects through a single neural-network pipeline [1]. The supplied model uses the pose-estimation form of YOLO26, which adds landmark coordinates to the object predictions [2].

Annotation quality and evaluation both matter. ObjectLab uses model predictions to help identify possible annotation errors [3]. Noise-aware evaluation studies show that uncertain object boundaries can change measured detector performance [4]. TIDE separates missed objects, duplicates, background responses, and localization errors [5]. These findings motivate reviewing the actual images and reporting several error types together.

For corner refinement, Simple Baselines predicts landmark heatmaps from a residual image backbone and learned upsampling layers [6]. PoseFix combines an image with an initial pose estimate to predict a refined pose [7]. These methods motivate the two local corner models tested here. Their published results concern human landmarks; their suitability for cassettes is assessed by the present measurements.

## 4 Background

### 4.1 Annotations and detection measurements

Annotations specify rectangles and known corner positions. Image coordinate x increases to the right and y downwards. Normalized x and y values are multiplied by image width and height, respectively, to obtain pixel coordinates. The images are 1280 by 720 pixels.

A predicted box B is compared with a reference box G using intersection over union, abbreviated IoU [5]:

$$
\operatorname{IoU}(B,G)=\frac{|B\cap G|}{|B\cup G|}. \tag{1}
$$

Predictions are processed from highest to lowest object confidence. Each is assigned to the unassigned reference with the greatest IoU, provided that IoU is at least 0.50. A reference receives at most one prediction.

A missed front has no assigned prediction. An extra prediction has no assigned reference. A duplicate is an extra prediction that overlaps an already assigned front by at least 0.50. Duplicates are therefore included in the extra count. A displaced box can create both a miss and an extra response.

With TP matched fronts, FN missed fronts, and FP extra predictions, precision P and recall R are:

$$
P=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},\qquad R=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}. \tag{2}
$$

Precision describes the fraction of predictions that match a reference. Recall describes the fraction of reference fronts that are found.

### 4.2 Corner accuracy and confidence

The error of a predicted corner p relative to its annotated position g is its Euclidean distance. A pair succeeds when both corners meet the tolerance epsilon:

$$
e_k=\sqrt{(p_{x,k}-g_{x,k})^2+(p_{y,k}-g_{y,k})^2},\qquad \max(e_L,e_R)\leq\epsilon. \tag{3}
$$

A successful pair also requires a matched box, two finite coordinates inside the image, and both corner scores at least 0.50. The reported tolerances are five and ten original-image pixels. Every reference front with two known corners remains in the denominator, including missed detections and unavailable predicted pairs.

Confidence ranks or accepts predictions; it does not measure coordinate error. Calibration relates scores to observed correctness [8]. A pair with an unavailable reference corner cannot be confirmed as accurate and is reported separately. Pixel distances measure image accuracy; physical positioning requires calibrated camera geometry.

## 5 Methodology and Implementation

### 5.1 Data and annotation review

Table 1 describes the data. Entire photographs retain their training or evaluation membership. The expanded training set contains 13 images without annotated fronts; the evaluation set contains three. Empty images are retained to reveal responses on background.

Table 1. Image and annotation counts. The reviewed evaluation annotations are the primary reference for the final comparison.

| Set | Photographs | Annotated fronts | Known corners |
| --- | ---: | ---: | ---: |
| Supplied training data | 137 | 3772 | 6839 |
| Reviewed and expanded training data | 142 | 3999 | 7259 |
| Reviewed evaluation reference | 20 | 654 | 1220 |

The reviewed training data contain 3397 fronts with two known corners, 465 with one, and 137 with neither. The evaluation reference contains 583, 54, and 17 fronts in these respective groups.

Annotations follow the cassette's own front, including a continuous frontal rim or tab. Rack, gripper, and neighbouring cassette material are excluded when choosing the box boundaries. A shadow does not remove cassette material from the target. Lower-corner points are placed on the actual cassette corners. When a corner cannot be located reliably, it is marked unavailable rather than assigned an estimated training coordinate.

Review corrected omitted fronts, redundant entries, and inaccurate boundaries or landmarks. Each accepted edit was checked against the original image. Model predictions assisted the review, and the same evaluation photographs were inspected during development. An earlier fixed annotation set is retained in Appendix 11.1 to show how the conclusions change with the reference.

### 5.2 Proposed recognition method

The baseline is the supplied YOLO26s-Pose checkpoint saved after epoch 173. It predicts one object category and two corners per cassette. Its learned parameters are used unchanged for the baseline comparison.

The proposed detector starts from the same supplied checkpoint and is adapted using the reviewed and expanded training data. Training uses seed 46, 30 epochs, input size 1280, batches of four images, 32-bit arithmetic, and AdamW [9]. The initial learning rate is 0.0001 and weight decay is 0.0005. The final epoch is used. The existing box, landmark, availability, and residual log-likelihood losses are retained; the latter models coordinate-error distributions during training [10]. Further settings appear in Appendix 11.2.

The adapted detector processes the original image and its horizontal reflection. Different views can provide alternative estimates, and their aggregation affects the final result [11]. The present method uses a fixed equal-coordinate average.

For image width W, continuous horizontal coordinates from the reflected view are returned to the original image frame by:

$$
x=W-x',\qquad y=y'. \tag{4}
$$

Box boundaries are reversed accordingly, and the lower-left and lower-right corner slots are exchanged. This mapping is checked before comparison.

The original-view boxes define the output detections. A reflected prediction is associated with an original prediction when they have the same class, each selects the other as its greatest-overlap partner, and their box IoU is at least 0.50. Equal overlap ties select the earlier partner in output order. If both predictions contain two valid corners with scores at least 0.50, corresponding points are averaged:

$$
\mathbf p_k^{\mathrm{final}}=\frac{\mathbf p_k^{\mathrm{original}}+\mathbf p_k^{\mathrm{reflected}}}{2},\qquad k\in\{L,R\}. \tag{5}
$$

Otherwise, the original coordinates are retained. The procedure preserves every original-view box, score, class, and output identity. Reference annotations do not choose associations or weights. The retained scores define the comparison threshold; their calibration for the averaged coordinates has not been established.

### 5.3 Local corner models

Two additional models examine a crop around each adapted-detector prediction. The crop expands the box to 1.5 times its width and height, rounds boundaries outwards, clips them to the image, and preserves aspect ratio when resized into a 256 by 256 canvas.

Both models use an ImageNet-initialized ResNet50 backbone, three learned upsampling stages, and two 64 by 64 heatmaps, following the general heatmap design in [6]. The first model receives the colour crop. The second also receives two Gaussian maps centred on the detector's initial corner estimates, motivated by [7]. Both have the same five-channel architecture; the extra two channels contain zeros in the image-only model. Both start from the same weights.

Training uses only matched detector proposals from the 142 training photographs. A reference corner contributes to loss only when it is known and inside the predicted crop. Padded heatmap cells are excluded. Both runs use seed 46, 30 epochs, batch size 32, 32-bit AdamW, and the final checkpoint. They each use 3815 supervised crops and complete 3600 parameter updates. Boxes and original scores remain fixed during evaluation. A decoded unavailable coordinate remains an unavailable output; no reference-based replacement is applied.

### 5.4 Common evaluation procedure

The primary comparison evaluates the supplied model and proposed method at input size 1280, object confidence strictly above 0.25, corner confidence at least 0.50, and matching IoU at least 0.50. The model's ordinary suppression uses IoU 0.70 and at most 300 outputs. Both methods additionally suppress redundant boxes at IoU 0.70 after coordinates have been clipped to the image. This common processing makes the comparison about the supplied checkpoint and proposed method under declared settings.

Both methods were run again on all 20 photographs in the same CPU environment using 32-bit arithmetic. The proposed method filters both views at the common object threshold before association and averaging. It averages 648 of 687 retained original-view records. Saved predictions are scored against both annotation references, and all headline counts reproduce the archived results.

The local-model predictions are evaluated at the same object threshold and against the same references. Their detector outputs, initial model states, training examples, sample orders, and runtime settings are identical between runs. Both completed training on an NVIDIA Tesla T4. Archive inventories, model bytes, crop transforms, and scores were independently checked. Two selected adverse cases were also reproduced using the saved models on a CPU.

## 6 Simulation Results

A numerical check tests whether the local models' heatmap grid alone prevents five-pixel accuracy. For every matched complete target, synthetic Gaussian heatmaps are centred exactly at the reference corners and passed through the same crop and decoding procedure as the learned outputs. The decoder selects a valid-image maximum, applies the fixed quarter-cell neighbour adjustment, and maps the result back using the actual rounded resize scales.

All 583 complete targets under the reviewed reference satisfy the five-pixel criterion in this check. The largest corner error is approximately 1.50 pixels. Thus, the grid and decoder can represent these target positions accurately enough for the chosen criterion. The synthetic maps use known positions and do not measure model recognition. Their purpose is to separate numerical resolution from errors in the learned heatmap locations.

## 7 Experimental Results

### 7.1 Comparison with the supplied model

Table 2 compares the supplied checkpoint with the proposed method on the reviewed reference. Both use the common settings in Section 5.4. Duplicates are included in extra predictions.

Table 2. Recognition on the same 20 photographs and reviewed annotations. Pair counts use all 583 fronts with two known corners.

| Measure | Supplied model | Proposed method |
| --- | ---: | ---: |
| Matched fronts out of 654 | 620 | 636 |
| Missed fronts | 34 | 18 |
| Extra predictions | 61 | 51 |
| Duplicate predictions | 5 | 5 |
| Both corners within 5 pixels | 555/583 | 569/583 |
| Both corners within 10 pixels | 577/583 | 583/583 |

The proposed method has 16 fewer misses and ten fewer extra outputs. Detection precision increases from 91.04% to 92.58%, and recall increases from 94.80% to 97.25%. Five-pixel pair success increases from 95.20% to 97.60%, a gain of 14 targets or 2.40 percentage points. The ten-pixel criterion gains six targets. Duplicate detections remain unchanged.

These measurements support improved recognition against the reviewed annotations. They do not show that every individual prediction improves. The earlier fixed annotations favour the supplied model, as shown in Appendix 11.1; annotation choice materially affects the conclusion.

### 7.2 Results of the local corner models

Table 3 uses the supplied model as the same baseline for the additional local models. Both local methods retain the adapted detector's 636 matched fronts, 18 misses, 51 extra outputs, and five duplicates at the common threshold.

Table 3. Complete-pair results for the local models and the supplied baseline. All rows use the reviewed reference and 583 eligible targets.

| Method | Both corners within 5 pixels | Both within 10 pixels |
| --- | ---: | ---: |
| Supplied model | 555/583 | 577/583 |
| Local image model | 547/583 | 582/583 |
| Local image and initial-corner model | 558/583 | 581/583 |

The image-only model loses eight five-pixel successes relative to the supplied baseline. Including initial-corner maps gives three more than the supplied baseline, but neither local method reaches the proposed method's 569 successes. Their ten-pixel results also remain below the proposed method.

The learned heatmap locations explain observed failures. In one examined case, the right-corner response moves towards a neighbouring cassette junction, producing an error above 20 pixels. In another, the response falls below the annotated corner. Exact saved-checkpoint replay reproduces these coordinates on a CPU. Together with the numerical check in Section 6, this supports investigating how the model learns corrections rather than attributing the result solely to coordinate conversion.

## 8 Discussions

### 8.1 Findings and interpretation

The proposed method combines reviewed training data with a two-view coordinate rule. Against the reviewed annotations, it finds more fronts, produces fewer extra outputs, and returns more accurate corner pairs than the supplied model. Duplicate counts remain unchanged at the chosen operating point.

The contribution is a cassette-front annotation procedure and an evaluated recognition method, compared with the supplied model. Development experiments guide method selection. The final measurements describe the combined system; they do not assign its full gain to annotation review or averaging alone.

The annotation sensitivity is substantial. The earlier reference contains different object entries and boundaries, and gives the supplied model a higher five-pixel count. The reviewed reference represents the accepted visible-front annotations, but predictions informed its review. Both annotations and methods have been examined on the same 20 photographs. New capture groups with independently reviewed labels are needed to assess transfer and reduce this dependence on development decisions.

Corner confidence also needs further evaluation. An automatically offered pair may match an incompletely annotated front or have no reference assignment. Neither case establishes correct positioning. A physical positioning requirement should specify an acceptable distance after camera calibration, then relate that requirement to the image-coordinate criterion.

### 8.2 Next investigation

The local-model results suggest testing a correction to an existing estimate. Iterative Error Feedback uses image information and the current landmark estimate to predict updates [12]. A single-step cassette adaptation can use:

$$
\mathbf p_k^{\mathrm{new}}=\mathbf p_k^{\mathrm{initial}}+\Delta\mathbf p_k. \tag{6}
$$

Initializing the correction output to zero preserves the original coordinates before learning. The model can then learn offsets from known training corners, while unavailable corners remain excluded from coordinate supervision. This is a hypothesis for the next experiment, not a measured improvement. The comparison should retain the same photographs, detector boxes, and final-epoch rule.

Training proposals currently come from a detector trained on the same photographs. Proposals from models that excluded each training group may provide more representative errors. Promising methods should be repeated with additional fixed seeds and assessed on new capture groups.

## 9 Conclusions

A method combining reviewed training annotations, detector adaptation, and equal averaging of associated two-view corner estimates was developed for cassette recognition. On the reviewed 20-image reference, it reduces missed fronts from 34 to 18 and extra predictions from 61 to 51 compared with the supplied model. Five-pixel pair success increases from 555 to 569 out of 583, and ten-pixel success increases from 577 to 583. Duplicate predictions remain at five.

Two learned local corner models were evaluated, but neither exceeds the proposed method's five-pixel result. Numerical and checkpoint checks identify learned localization as a useful focus for further investigation. The earlier annotation reference favours the supplied model, so the demonstrated improvement is tied to the reviewed target definition and development images. The next phase will test learned residual corrections and independent capture evaluation.

## 10 References

[1] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2016, pp. 779–788, doi: 10.1109/CVPR.2016.91.

[2] Ultralytics, “Ultralytics YOLO26,” Ultralytics Docs. Accessed: Nov. 28, 2026. [Online]. Available: https://docs.ultralytics.com/models/yolo26/

[3] U. Tkachenko, A. Thyagarajan, and J. Mueller, “ObjectLab: Automated diagnosis of mislabeled images in object detection data,” in ICML Workshop on Data-centric Machine Learning Research, 2023. [Online]. Available: https://arxiv.org/abs/2309.00832

[4] J. Murrugarra-Llerena and C. R. Jung, “Noise-aware evaluation of object detectors,” in Proc. IEEE/CVF Winter Conf. Applications of Computer Vision (WACV), 2025, doi: 10.1109/WACV61041.2025.00903.

[5] D. Bolya, S. Foley, J. Hays, and J. Hoffman, “TIDE: A general toolbox for identifying object detection errors,” in Computer Vision – ECCV 2020, Lecture Notes in Computer Science, vol. 12348. Springer, 2020, pp. 558–573, doi: 10.1007/978-3-030-58580-8_33.

[6] B. Xiao, H. Wu, and Y. Wei, “Simple baselines for human pose estimation and tracking,” in Proc. European Conf. Computer Vision (ECCV), 2018, pp. 466–481. [Online]. Available: https://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html

[7] G. Moon, J. Y. Chang, and K. M. Lee, “PoseFix: Model-agnostic general human pose refinement network,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2019, pp. 7773–7781.

[8] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. 34th Int. Conf. Machine Learning (ICML), Proceedings of Machine Learning Research, vol. 70, 2017, pp. 1321–1330. [Online]. Available: https://proceedings.mlr.press/v70/guo17a.html

[9] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. Int. Conf. Learning Representations (ICLR), 2019. [Online]. Available: https://arxiv.org/abs/1711.05101

[10] J. Li et al., “Human pose regression with residual log-likelihood estimation,” in Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV), 2021, pp. 11025–11034.

[11] D. Shanmugam, D. Blalock, G. Balakrishnan, and J. Guttag, “Better aggregation in test-time augmentation,” in Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV), 2021, pp. 1214–1223. [Online]. Available: https://openaccess.thecvf.com/content/ICCV2021/html/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.html

[12] J. Carreira, P. Agrawal, K. Fragkiadaki, and J. Malik, “Human pose estimation with iterative error feedback,” in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2016, pp. 4733–4742, doi: 10.1109/CVPR.2016.512.

## 11 Appendices

### 11.1 Sensitivity to the earlier annotations

The earlier fixed reference contains 666 fronts, including 584 with two known corners. It describes the same photographs. Table 4 applies the same predictions and common operating settings used in the main comparison.

Table 4. Supplied-model comparison under the earlier fixed annotations. These are alternative labels for the same images, rather than an independent test set.

| Measure | Supplied model | Proposed method |
| --- | ---: | ---: |
| Missed fronts | 26 | 34 |
| Extra predictions | 41 | 55 |
| Duplicate predictions | 11 | 9 |
| Both corners within 5 pixels | 534/584 | 525/584 |
| Both corners within 10 pixels | 580/584 | 574/584 |

The proposed method loses nine five-pixel successes under this reference. This adverse comparison is retained because annotation changes affect both box assignment and corner distance. It prevents the reviewed-reference improvement from being described as an annotation-independent result.

### 11.2 Training and coordinate settings

The common inference rerun used Python 3.12.14, PyTorch 2.9.0+cpu, and Ultralytics 8.4.182.

The adapted detector uses AdamW with an initial learning rate of 0.0001, final learning-rate factor 0.01, weight decay 0.0005, and three warmup epochs. Training includes rotation up to five degrees, translation fraction 0.05, scale fraction 0.15, and horizontal reflection probability 0.50. Mosaic, mixup, vertical reflection, shear, and perspective transforms are disabled. The final checkpoint follows 1080 batches and 80 parameter updates; gradient accumulation combines several batches into each update. The supplied 664-front validation labels monitor training, and the final comparison uses the separately reviewed annotations.

The local models use batch size 32, AdamW learning rate 0.0001, weight decay 0.0001, and cosine decay to 0.000001 over 30 epochs. Both expose 3815 crops per epoch, giving 114450 crop presentations. Colour augmentation is disabled. Initial-corner hints have Gaussian standard deviation eight canvas pixels, with four-pixel standard-deviation position noise during training only. Target heatmaps have standard deviation two heatmap cells. The output grid represents canvas positions at four-pixel intervals; padding is masked in both loss and decoding.

Every comparison retains the image identity, reference version, model bytes, inference settings, predicted boxes and points, and individual gained or lost targets. Counts are recomputed from these records using the declared one-to-one matching rule. These records make the measurements traceable without changing the target definition after a result is observed.
