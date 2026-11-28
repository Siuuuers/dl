# Cassette Front Detection and Corner Localization

Final Year Project I Report

Department of Electronic Engineering

## 1 Abstract

This project investigates the detection of tissue cassette fronts and the location of their two lower corners in camera images. Closely arranged objects and partial occlusion can cause missed fronts, repeated detections, and inaccurate coordinates. The work combines consistent annotation review, adaptation of a supplied pose-estimation model, and equal averaging of corner estimates from original and horizontally reflected images.

The comparison uses the untouched supplied model as its baseline. Both methods are evaluated on the same 20 photographs with common confidence and matching settings. Against the reviewed annotations, the proposed method reduces missed fronts from 34 to 18 and extra predictions from 61 to 51. Duplicate predictions remain at five. The number of fronts with both corners within five pixels increases from 555 to 569 out of 583; at ten pixels it increases from 577 to 583. Three local corner models are also evaluated, including a learned coordinate correction, but none exceeds the proposed method.

Performance depends on the annotation reference. The original supplied annotations and an earlier fixed reference favour the supplied model. Rescoring unchanged predictions separates this evaluation effect from changes caused by training. The contribution is a cassette-specific annotation procedure and an evaluated recognition method with traceable detection and coordinate measurements. Further work will isolate the effect of the latest training-label revisions and assess transfer to independently captured images.

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
  - [8.2 Further work](#82-further-work)
- [9 Conclusions](#9-conclusions)
- [10 References](#10-references)
- [11 Appendices](#11-appendices)
  - [11.1 Sensitivity to the annotation reference](#111-sensitivity-to-the-annotation-reference)
  - [11.2 Training and coordinate settings](#112-training-and-coordinate-settings)
<!-- DOCX_TOC_END -->

## 3 Introduction

### 3.1 Problem and aim

The system detects cassette fronts and locates their lower-left and lower-right corners for robotic-arm localization. These points describe the lower edge's position and orientation. Evaluation measures misses, extra predictions and corner error. Arm control and integration are outside this study. The output message format and physical tolerance remain unspecified.

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

With C matched fronts, M missed fronts, and E extra predictions, precision P and recall R are:

$$
P=\frac{C}{C+E},\qquad R=\frac{C}{C+M}. \tag{2}
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

Review corrected omitted fronts, redundant entries, and inaccurate boundaries or landmarks. Each accepted edit was checked against the original image. Model predictions assisted the review, and the same evaluation photographs were inspected during development. The original supplied validation annotations contain 664 fronts; an earlier fixed reference contains 666. Appendix 11.1 compares both with the reviewed reference using unchanged predictions.

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

A third model learns a correction to an available initial point, motivated by Iterative Error Feedback [12]:

$$
\mathbf p_k^{\mathrm{new}}=\mathbf p_k^{\mathrm{initial}}+\Delta\mathbf p_k. \tag{6}
$$

It receives the colour crop and two initial-corner maps. A ResNet50 backbone feeds global average pooling and four coordinate outputs. The output weights and biases start at zero, preserving the initial points exactly before learning. Predicted crop-coordinate offsets are divided by the actual horizontal and vertical resize scales before addition in Equation (6).

Training requires both a known reference corner and an available initial estimate. Unavailable initial points remain unchanged. The loss is quadratic for small coordinate errors and linear for larger errors [13], with its transition at one resized-crop pixel. Hint noise is disabled. The model uses 3809 supervised crops, seed 46, 30 epochs and the final checkpoint. It preserves detector boxes and scores.

### 5.4 Common evaluation procedure

The primary comparison evaluates the supplied model and proposed method at input size 1280, object confidence strictly above 0.25, corner confidence at least 0.50, and matching IoU at least 0.50. The model's ordinary suppression uses IoU 0.70 and at most 300 outputs. Both methods additionally suppress redundant boxes at IoU 0.70 after coordinates have been clipped to the image. This common processing makes the comparison about the supplied checkpoint and proposed method under declared settings.

Both methods were run again on all 20 photographs in the same CPU environment using 32-bit arithmetic. The proposed method filters both views at the common object threshold before association and averaging. It averages 648 of 687 retained original-view records. Saved predictions are scored against the supplied, earlier fixed and reviewed annotation references. The earlier fixed and reviewed counts reproduce the archived results exactly. Changing only the scoring annotations requires no further inference or training.

The local models are evaluated at the same object threshold and reference definitions. The two heatmap runs share identical detector outputs, initialization, training examples and sample orders, and use an NVIDIA Tesla T4. The correction model uses an NVIDIA L4 and a different output layer, loss and target mask. It is an alternative design, rather than a one-variable comparison with the heatmap models. Saved model files, training records, coordinate transforms and scores were checked. Selected heatmap cases and all 71 predictions in two correction-model images were reproduced on a CPU without changing their scored outcomes.

## 6 Simulation Results

A numerical check tests whether the heatmap models' output grid alone prevents five-pixel accuracy. For every matched complete target, synthetic Gaussian heatmaps are centred exactly at the reference corners and passed through the same crop and decoding procedure as the learned outputs. The decoder selects a valid-image maximum, applies the fixed quarter-cell neighbour adjustment, and maps the result back using the actual rounded resize scales.

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

Detection precision increases from 91.04% to 92.58%, and recall from 94.80% to 97.25%. Five-pixel pair success rises from 95.20% to 97.60%, a gain of 2.40 percentage points. The ten-pixel criterion gains six fronts; duplicates remain unchanged.

The proposed method detects 583/583 complete fronts, versus 580/583 for the supplied model. For the 71 fronts with unavailable corners, detection rises from 40/71 (56.34%) to 53/71 (74.65%). Its 18 misses have one known corner (13 fronts) or neither (five). The 583/583 ten-pixel result therefore excludes these incomplete fronts.

Archived original-view proposals from the same checkpoint locate the 18 misses: 12 have no eligible box passing confidence filtering, three are lost in native non-maximum suppression, and three have no decoded box reaching IoU 0.50. This partition does not establish the cause during learning.

The 51 extras comprise five duplicates, 36 predictions whose greatest reference IoU is at least 0.10 but below 0.50, and ten below 0.10. These geometric groups indicate possible localization and background errors for review [5]; they are not physical verdicts. Displacement, omitted annotations and disputed boundaries require checking the original photograph.

Both older references favour the supplied model (Appendix 11.1), so the recognition improvement is tied to the reviewed annotations. Individual targets can improve or regress.

Keeping reflected-view detections with the same corner averaging gives 15 misses, 43 extras and unchanged five- and ten-pixel pair counts under the reviewed reference. It recovers eight fronts and loses five, but duplicates double from five to ten and both older references lose corner successes. This exploratory trade-off does not replace the original-view detections used in the main comparison.

### 7.2 Results of the local corner models

Table 3 uses the supplied model as the baseline for the three local models. All local methods retain the adapted detector's 636 matched fronts, 18 misses, 51 extra outputs and five duplicates at the common threshold.

Table 3. Complete-pair results for the local models and supplied baseline. All rows use the reviewed reference and 583 eligible targets.

| Method | Both corners within 5 pixels | Both within 10 pixels |
| --- | ---: | ---: |
| Supplied model | 555/583 | 577/583 |
| Local image model | 547/583 | 582/583 |
| Local image and initial-corner model | 558/583 | 581/583 |
| Local coordinate-correction model | 564/583 | 582/583 |

The correction model has nine more five-pixel successes than the supplied baseline, but five fewer than the proposed two-view method. Relative to the proposed method, three targets improve and eight regress. Its mean larger-of-two corner error is 2.27 pixels, compared with 2.16 pixels for the proposed method on the same 583 complete fronts. Thus, neither the success count nor the mean error supports replacing the proposed method.

The heatmap models sometimes select nearby cassette features. In one examined case, the right-corner response moves towards a neighbouring junction, producing an error above 20 pixels. The correction model can also move an acceptable point beyond the five-pixel tolerance. Saved-checkpoint replay reproduces these outcomes. Together with Section 6, this supports examining the training-error examples used by local models.

## 8 Discussions

### 8.1 Findings and interpretation

The proposed method improves reviewed-reference detection and corner results at the declared operating point, with unchanged duplicate counts.

The contribution is a cassette-front annotation procedure and an evaluated recognition method, compared with the supplied model. Development experiments guide method selection. The final measurements describe the combined system; they do not assign its full gain to annotation review or averaging alone.

The annotation sensitivity is substantial. The supplied and earlier fixed references contain different object entries, boundaries and corner positions, and both give the supplied model a higher five-pixel count. Rescoring changes agreement with a reference; it does not change the images, model parameters or predictions. The reviewed reference represents the accepted visible-front annotations, but predictions informed its review. Both annotations and methods have been examined on the same 20 photographs. New capture groups with independently reviewed labels are needed to assess transfer and reduce this dependence on development decisions.

At the current thresholds, the proposed method offers 678 pairs: 569 satisfy the five-pixel criterion, 14 matched pairs exceed it, 48 match incomplete reference pairs, and 47 are unmatched. The last two groups lack reference support; this does not establish physical error. The 569/583 score therefore describes eligible reference fronts, not all emitted outputs. Physical positioning also requires a distance tolerance and calibrated camera geometry.

### 8.2 Further work

The prepared training-label comparison holds photographs, starting checkpoint, settings, budget and scoring annotations fixed while changing training labels. The main comparison also changes training exposure and includes five extra photographs, so its full gain cannot be attributed to label review.

A separate paired experiment changes classification-loss emphasis on incorrectly scored proposals while retaining target scores and other losses. Both training comparisons await results; any replacement must be judged on detection and corner outcomes together.

Local models currently learn from proposals produced by a detector trained on the same photographs. To obtain errors on unseen photographs, both the detector's training and its initial checkpoint must exclude those photographs. Holding them out only during fine-tuning leaves earlier exposure intact. The prepared paired runs remain valid tests of adaptation from shared supplied weights. PoseFix suggests realistic synthetic errors as another training source [7]. Promising results need additional fixed seeds and independent capture groups.

## 9 Conclusions

A method combining reviewed training annotations, detector adaptation, and equal averaging of associated two-view corner estimates was developed for cassette recognition. On the reviewed 20-image reference, it reduces missed fronts from 34 to 18 and extra predictions from 61 to 51 compared with the supplied model. Five-pixel pair success increases from 555 to 569 out of 583, and ten-pixel success increases from 577 to 583. Duplicate predictions remain at five.

Three learned local corner models were evaluated, including a coordinate-correction model, but none exceeds the proposed method. Scoring unchanged predictions against the original supplied and earlier fixed annotations reverses the main comparison. The demonstrated improvement is therefore tied to the reviewed target definition and development photographs. The next phase will isolate training-label effects, improve the local models' training examples and evaluate independent captures.

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

[13] PyTorch Contributors, “SmoothL1Loss,” PyTorch Documentation. Accessed: Nov. 28, 2026. [Online]. Available: https://docs.pytorch.org/docs/2.15/generated/torch.nn.SmoothL1Loss.html

## 11 Appendices

### 11.1 Sensitivity to the annotation reference

All three annotation sets describe the same 20 photographs. The supplied labels contain 664 fronts, the earlier fixed labels contain 666, and the reviewed labels contain 654. Their numbers of complete corner pairs appear in Table 4. Original supplied labels were verified against the supplied data manifest before scoring.

Table 4. Effect of changing the scoring reference while keeping predictions fixed. Each arrow shows the supplied model followed by the proposed method. Corner-success denominators are the complete-pair counts in the first column.

| Annotation reference | Misses | Extra predictions | Within 5 pixels | Within 10 pixels |
| --- | ---: | ---: | ---: | ---: |
| Supplied, 584 pairs | 39 → 46 | 56 → 69 | 521 → 514 | 568 → 565 |
| Earlier fixed, 584 pairs | 26 → 34 | 41 → 55 | 534 → 525 | 580 → 574 |
| Reviewed, 583 pairs | 34 → 18 | 61 → 51 | 555 → 569 | 577 → 583 |

The proposed method loses seven five-pixel successes under the supplied labels and nine under the earlier fixed labels, while gaining 14 under the reviewed labels. Changes to object entries, boundaries and corner positions can change both box assignment and coordinate errors. This comparison measures evaluation-reference sensitivity; it does not isolate the effect of changing training labels.

### 11.2 Training and coordinate settings

The common inference rerun used Python 3.12.14, PyTorch 2.9.0+cpu, and Ultralytics 8.4.182.

The adapted detector uses AdamW with an initial learning rate of 0.0001, final learning-rate factor 0.01, weight decay 0.0005, and three warmup epochs. Training includes rotation up to five degrees, translation fraction 0.05, scale fraction 0.15, and horizontal reflection probability 0.50. Mosaic, mixup, vertical reflection, shear, and perspective transforms are disabled. The final checkpoint follows 1080 batches and 80 parameter updates; gradient accumulation combines several batches into each update. The supplied 664-front validation labels monitor training, and the final comparison uses the separately reviewed annotations.

The heatmap models use batch size 32, AdamW learning rate 0.0001, weight decay 0.0001, and cosine decay to 0.000001 over 30 epochs. Both expose 3815 crops per epoch, giving 114450 crop presentations. Colour augmentation is disabled. Initial-corner hints have Gaussian standard deviation eight canvas pixels, with four-pixel standard-deviation position noise during training only. Target heatmaps have standard deviation two heatmap cells. The output grid represents canvas positions at four-pixel intervals; padding is masked in both loss and decoding.

The correction model uses the same optimizer, learning-rate schedule, batch size and epoch budget, with no hint noise. Its 7204 eligible endpoint targets produce 114270 crop presentations and 3600 parameter updates. Eight known reference corners lack a usable initial estimate and are excluded. A nonfinite correction or a corrected point outside the image becomes unavailable. The recorded training runtime uses PyTorch 2.14.1+cu130; the selected CPU replay uses PyTorch 2.9.0+cpu.

Every comparison retains the image identity, reference version, model bytes, inference settings, predicted boxes and points, and individual gained or lost targets. These records allow the reported counts and individual outcomes to be recomputed using the declared one-to-one matching rule.
