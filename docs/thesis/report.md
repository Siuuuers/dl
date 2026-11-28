# Cassette Front Detection and Corner Localization

Final Year Project I Report

Department of Electronic Engineering

## 1 Abstract

This project investigates tissue cassette front detection and the location of two lower corners in camera images. Closely arranged objects and partial occlusion can cause missed fronts, repeated detections and inaccurate coordinates. The work combines annotation review, adaptation of a supplied pose-estimation model and equal averaging of corner estimates from original and horizontally reflected images.

Against reviewed annotations on 20 development photographs, the proposed method reduces missed fronts from 34 to 18 and extra predictions from 61 to 51 relative to the untouched supplied model. Duplicate predictions remain at five. Both corners lie within five pixels for 569 of 583 complete reference pairs, compared with 555 for the supplied model. Three local corner models do not exceed this result.

Two controlled training comparisons examine what produces useful recognition. Changing the classification loss recovers six fronts but adds 34 extra predictions, with almost unchanged detection average precision. Separately, changing only the training labels on the same 137 photographs reduces misses from 47 to 15 and extras from 85 to 51 under the reviewed reference. Five-pixel successes increase from 541 to 572, while duplicates increase from six to 13. The older evaluation references favour the model trained with supplied labels.

The contribution is an explicit cassette annotation procedure and controlled evidence separating training-label effects, evaluation-reference effects and model-design trade-offs. These single-seed comparisons support conclusions about the declared development setting. Independent annotation repeatability, repeated training seeds and separately captured scenes remain necessary to assess wider reliability.

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
  - [5.5 Controlled training comparisons](#55-controlled-training-comparisons)
- [6 Simulation Results](#6-simulation-results)
- [7 Experimental Results](#7-experimental-results)
  - [7.1 Comparison with the supplied model](#71-comparison-with-the-supplied-model)
  - [7.2 Results of the local corner models](#72-results-of-the-local-corner-models)
  - [7.3 Effect of classification loss](#73-effect-of-classification-loss)
  - [7.4 Effect of training annotations](#74-effect-of-training-annotations)
  - [7.5 Duplicate and unmatched output audit](#75-duplicate-and-unmatched-output-audit)
- [8 Discussions](#8-discussions)
  - [8.1 Findings and interpretation](#81-findings-and-interpretation)
  - [8.2 Further work](#82-further-work)
- [9 Conclusions](#9-conclusions)
- [10 References](#10-references)
- [11 Appendices](#11-appendices)
  - [11.1 Sensitivity to the annotation reference](#111-sensitivity-to-the-annotation-reference)
  - [11.2 Training and coordinate settings](#112-training-and-coordinate-settings)
  - [11.3 Training-label results under all references](#113-training-label-results-under-all-references)
<!-- DOCX_TOC_END -->

## 3 Introduction

### 3.1 Problem and aim

The system detects tissue cassette fronts and locates their lower-left and lower-right corners in camera images. The two points describe the lower edge's position and orientation in the image plane. The intended application sends vision outputs to a robotic arm for cassette localization. This study evaluates the image measurements; arm control and integration are outside its scope. A tolerance in physical units and the downstream message format have not been specified.

Cassettes are closely arranged in a rack. A front may be partly covered by another cassette or a gripper, or cut by the image boundary. The required rectangle follows the cassette's own visible front material; rack and gripper boundaries must not define the object. Figure 1 illustrates the target.

![A cassette front and its two lower corners](figures-r57/figure-1-target.png)

Figure 1. Camera image and enlarged cassette front. The rectangle marks the annotated front and the two points mark its actual lower corners. Neighbouring cassettes and the rack provide context. A corner is marked unavailable when its position cannot be identified reliably.

The aim is to improve recognition quality while keeping the processing method clear. The project starts from a supplied image collection and trained cassette model. Its main comparison is between that original model and the proposed method, evaluated with the same measurement rules. The target output is every identifiable front, including a partial front, together with whichever corner locations can be supported by the image.

The proposed processing has two main stages. First, a detector identifies candidate fronts and estimates their corner coordinates. Second, the same detector processes a horizontally reflected copy of the photograph; corresponding corner estimates are returned to the original image coordinates and averaged where the two views can be associated. Original-view boxes determine which detections are output. Separate experiments test local corner models, a different classification loss and the effect of replacing training annotations while holding the photographs and training procedure fixed.

### 3.2 Related research

Redmon et al. introduced You Only Look Once, or YOLO, which predicts object rectangles and class scores from a complete image through one neural network [1]. The supplied model belongs to the later YOLO26 family, whose pose-estimation variant also predicts landmark positions [2]. In this project, a pose consists of the two lower cassette corners. It is a two-dimensional image description; a full three-dimensional pose of the cassette is not estimated.

Annotation research helps define how model outputs should be checked. ObjectLab compares detector predictions with existing labels and ranks images that may contain missing, misplaced or wrongly classified boxes for review [3]. This supports using predictions to direct human attention. It does not make a disagreement sufficient evidence to change a label. Murrugarra-Llerena and Jung show that uncertainty in box boundaries can raise or lower measured detector performance, particularly for small objects and strict overlap requirements [4]. Their study motivates checking sensitivity to the annotation reference. It does not determine which cassette boundary is physically correct.

TIDE separates detection errors into categories and estimates how correcting each category would affect average precision, which summarizes a precision-recall curve [5]. Its central relevance here is that one aggregate score can hide different failure modes. This project therefore reports missed fronts, extra predictions and duplicates alongside corner accuracy. The fixed-threshold counts defined in Section 4 are project measurements; the full TIDE analysis is not reproduced.

For landmark localization, Xiao et al. use an image feature extractor followed by learned upsampling layers to predict a heatmap for each landmark [6]. A heatmap is a spatial grid of scores: a high response indicates a candidate landmark position. This design motivates the local image model. PoseFix adds an initial pose to the image input and trains a separate refinement network using synthetic pose errors derived from observed human-pose error statistics [7]. This motivates providing the initial cassette corners as additional input, while also suggesting that the errors seen during refinement training deserve examination. Both papers evaluate human landmarks. Their results provide design precedents; the cassette experiments establish how the adaptations perform on this task.

## 4 Background

### 4.1 Annotations and detection measurements

Annotations specify rectangles and known corner positions. A rectangle describes the extent of a front; the corner coordinates describe two particular points on that front. The landmarks need not coincide with the lower corners of an axis-aligned rectangle, particularly when the cassette is tilted or partly covered. These are separate measurements, so an adequate rectangle can contain inaccurate corner estimates. Image coordinate x increases to the right and y downwards. Normalized x and y values are multiplied by image width and height, respectively, to obtain pixel coordinates. The images are 1280 by 720 pixels.

A predicted box B is compared with a reference box G using intersection over union, abbreviated IoU:

$$
\operatorname{IoU}(B,G)=\frac{|B\cap G|}{|B\cup G|}. \tag{1}
$$

The numerator is the area shared by the two rectangles, and the denominator is their combined area counted once. IoU is zero for nonoverlapping boxes and one for identical boxes. It measures rectangle agreement and does not directly measure corner accuracy.

During evaluation, predictions are processed from highest to lowest object confidence. Each is assigned to the unassigned reference with the greatest IoU, provided that IoU is at least 0.50. A reference receives at most one prediction. This rule prevents several predictions of one front from being counted as several successful detections.

A missed front has no assigned prediction. An extra prediction has no assigned reference. A duplicate is an extra prediction that overlaps an already assigned front by at least 0.50. Duplicates are therefore included in the extra count. For example, if a photograph has one reference front and one prediction with IoU 0.49, the rule records both a miss and an extra. The counts describe agreement with the reference under a declared threshold; they do not, by themselves, establish whether a physical object is absent.

Before evaluation, non-maximum suppression (NMS) removes some overlapping predictions. It keeps a high-scoring prediction and discards lower-scoring predictions whose overlap with it exceeds the suppression threshold. NMS compares predictions with other predictions and requires no annotations. Evaluation matching compares predictions with reference annotations. The proposed two-view method also uses overlap to associate predictions across the two views. These three operations have different purposes even when they use the same IoU formula.

A scored duplicate can remain after correctly implemented NMS. Two predictions can both overlap the same reference by at least 0.50 without overlapping each other by more than the suppression threshold of 0.70. Their evaluation category and their suppression decision therefore need not coincide.

With C matched fronts, M missed fronts, and E extra predictions, precision P and recall R are:

$$
P=\frac{C}{C+E},\qquad R=\frac{C}{C+M}. \tag{2}
$$

Precision describes the fraction of predictions that match a reference. Recall describes the fraction of reference fronts that are found.

### 4.2 Corner accuracy and confidence

The error of a predicted corner p relative to its annotated position g is its Euclidean distance, measured in original-image pixels. Subscripts L and R denote the lower-left and lower-right corners. A pair succeeds when the larger of its two errors is no greater than the tolerance epsilon:

$$
e_k=\sqrt{(p_{x,k}-g_{x,k})^2+(p_{y,k}-g_{y,k})^2},\qquad \max(e_L,e_R)\leq\epsilon. \tag{3}
$$

A successful pair also requires a matched box, two finite coordinates inside the image, and both corner scores at least 0.50. The reported tolerances are five and ten original-image pixels. Every reference front with two known corners remains in the denominator, including missed detections and unavailable predicted pairs.

For example, errors of three and six pixels fail the five-pixel pair criterion but satisfy the ten-pixel criterion. Using the larger error prevents a good corner from concealing a poor second corner. Averaging errors across many fronts answers a different question and can improve even when several individual pairs cross a failure threshold.

Reference availability and predicted confidence also have different meanings. A reference corner is available when its location can be established from the photograph under the annotation rules. A predicted corner score is a model output used to accept or reject its coordinate. Guo et al. show that classification confidence can disagree with observed correctness and study ways to calibrate it [8]. This supports checking score reliability, but their classification results do not establish that a cassette corner score is the probability of being within five pixels. That relationship has not been calibrated here.

A front with an unavailable reference corner still contributes to detection evaluation, but its complete corner pair cannot be scored. Such outputs are reported separately. Pixel distances describe image accuracy. Converting them into physical positioning error would additionally require camera calibration and the relevant scene geometry; two image points alone do not provide that conversion.

## 5 Methodology and Implementation

The supplied model is the unchanged benchmark. The adapted detector continues training from its weights, and the proposed method combines that adapted detector's original and reflected corner estimates. The three local corner models are alternatives applied to the adapted detector's original-view outputs; they are not additional stages after two-view averaging. This separation makes it possible to compare the complete system with the supplied model and to measure what each added corner-processing step contributes.

### 5.1 Data and annotation review

Table 1 describes the data. Entire photographs retain their training or evaluation membership. The expanded training set contains 13 images without annotated fronts; the evaluation set contains three. Empty images are retained to reveal responses on background.

Table 1. Image and annotation counts. The 654-front reviewed reference is retained for the controlled comparisons; the extended reference is assessed separately in Section 7.6.

| Set | Photographs | Annotated fronts | Known corners |
| --- | ---: | ---: | ---: |
| Supplied training data | 137 | 3772 | 6839 |
| Reviewed and expanded training data | 142 | 3999 | 7259 |
| Reviewed evaluation reference | 20 | 654 | 1220 |
| Extended reference after inclusion review | 20 | 658 | 1223 |

The reviewed training data contain 3397 fronts with two known corners, 465 with one, and 137 with neither. The evaluation reference contains 583, 54, and 17 fronts in these respective groups.

Annotations follow the cassette's own front, including a continuous frontal rim or tab. Rack, gripper, and neighbouring cassette material are excluded when choosing the box boundaries. A shadow does not remove cassette material from the target. Lower-corner points are placed on the actual cassette corners. When a corner cannot be located reliably, it is marked unavailable rather than assigned an estimated training coordinate.

This definition allows an identifiable partial front to receive a box even when one or both lower corners are hidden. An unavailable point contributes no coordinate target. It does not change the front into a background example. The definition therefore separates evidence that a cassette front exists from evidence for a particular landmark location.

Review corrected omitted fronts, redundant entries, and inaccurate boundaries or landmarks. Each accepted edit was checked against the original image. Model predictions assisted the review, and the same evaluation photographs were inspected during development. The original supplied validation annotations contain 664 fronts; an earlier fixed reference contains 666. Appendix 11.1 compares both with the reviewed reference using unchanged predictions.

The frozen 654-front reference also inherits documented case-specific exclusions from earlier review, including visible partial fronts. It is the benchmark used in the controlled comparisons, without independent certification that it exhaustively applies the visible-front rule. An unmatched prediction may correspond to excluded material, disputed boundaries or a model error; the score alone cannot decide which interpretation applies.

A subsequent focused review inspected seven historical exclusions in five original photographs, first using unmarked images and then historical and current label overlays. Four distinct visible fronts across three photographs were restored in a separate reference. Their previous nominal boxes and three available corners remained supported by the images, so the intervention changes inclusion only. Three other regions remain unresolved because the front surface or tool ownership is uncertain. The extended reference contains 658 fronts: 583 with two known corners, 57 with one and 18 with neither. Existing rows, historical references and training labels retain their original bytes. This agent-assisted review is not an independently blinded human study or an exhaustive recertification of the photographs.

### 5.2 Proposed recognition method

The baseline is the supplied YOLO26s-Pose checkpoint saved after epoch 173. A checkpoint stores the model's learned parameters at a particular training stage. This model predicts one object category and two corners per cassette. Its parameters are used unchanged for the baseline comparison.

The proposed detector starts from the same checkpoint and continues training on the reviewed and expanded training data. This adaptation is called fine-tuning. Training uses seed 46, 30 epochs, input size 1280, batches of four images and 32-bit training tensors. An epoch presents the training set once; a random seed fixes the initial random-number sequence for a run. AdamW updates the parameters using adaptive gradient steps while applying weight decay separately [9]. The initial learning rate is 0.0001 and weight decay is 0.0005. The final epoch is used, avoiding selection of an intermediate checkpoint by the reported corner result.

The existing box, landmark, availability and residual log-likelihood losses are retained. A loss measures a training discrepancy and determines how parameters are updated. Residual log-likelihood estimation learns an adjustment to a reference distribution of coordinate errors, allowing the training objective to model errors more flexibly than a fixed distribution alone [10]. Retaining this objective does not establish calibrated probabilities for the final corner outputs. Further training settings appear in Appendix 11.2.

The adapted detector processes the original image and its horizontal reflection. Simple Baselines provides a pose-estimation precedent for combining original and reflected predictions, although it averages heatmaps [6]. Shanmugam et al. show in image classification that averaging transformed predictions can improve total accuracy while also turning some correct cases into errors [11]. This motivates examining gained and lost cases. The present method instead averages corresponding cassette coordinates with fixed equal weights; its benefit must be measured on the cassette images.

Both views come from the same photograph. Reflection supplies another network response to the existing image information; it does not reveal a hidden surface or provide a second camera observation. The two estimates may share errors, so averaging carries no general guarantee of improved accuracy.

For image width W, continuous horizontal coordinates from the reflected view are returned to the original image frame by:

$$
x=W-x',\qquad y=y'. \tag{4}
$$

The mapping uses continuous image coordinates, consistent with the detector's box and landmark representation. Box boundaries are reversed accordingly, and the lower-left and lower-right corner slots are exchanged: a cassette's original left corner appears on the right in the reflected image. This mapping is checked before comparison.

The original-view boxes define the output detections. A reflected prediction is associated with an original prediction when they have the same class, each selects the other as its greatest-overlap partner, and their box IoU is at least 0.50. Requiring this mutual selection avoids sending several reflected predictions to the same original detection. Equal overlap ties select the earlier partner in output order. If both predictions contain two valid corners with scores at least 0.50, corresponding points are averaged:

$$
\mathbf p_k^{\mathrm{final}}=\frac{\mathbf p_k^{\mathrm{original}}+\mathbf p_k^{\mathrm{reflected}}}{2},\qquad k\in\{L,R\}. \tag{5}
$$

Otherwise, the original coordinates are retained. The procedure preserves every original-view box, score, class and output identity. Consequently, averaging can change corner accuracy but cannot itself recover a missing detection or remove an extra box. Reference annotations do not choose associations or weights. The retained scores define the comparison threshold; their calibration for the averaged coordinates has not been established.

### 5.3 Local corner models

Two additional models examine a crop around each adapted-detector prediction. The crop expands the box to 1.5 times its width and height so that some surrounding image context is included. Its boundaries are rounded outwards and clipped to the photograph. The crop preserves its aspect ratio when resized into a 256 by 256 canvas; unused canvas space is padding.

Both models use a ResNet50 backbone initialized with weights learned on the ImageNet image-classification dataset. The backbone extracts image features. ResNet uses shortcut connections that add a block's input to its learned transformation, helping deep networks to be optimized [12]. These internal feature additions are distinct from the final coordinate correction tested below. Three learned upsampling stages enlarge the feature maps and produce two 64 by 64 heatmaps, following the general design of Simple Baselines [6]. There is one output map for each lower corner.

The first model receives the colour crop. The second also receives two Gaussian maps centred on the detector's initial corner estimates. Each map is a smooth peak that tells the model where the initial estimate lies; it is an input hint, not an additional observation from the camera. This use of image and pose input is motivated by PoseFix [7]. Both local models have the same five-channel architecture, but the additional two channels contain zeros in the image-only model. They start from the same weights, so this comparison examines the effect of supplying the initial-corner information.

The output heatmaps are trained against Gaussian peaks at the known reference corners using mean squared error over the valid cells. They are spatial scores, not calibrated probabilities. Decoding selects the highest valid response, makes a fixed small adjustment using neighbouring responses, and converts the resulting canvas position back to the original photograph. The actual horizontal and vertical resize scales are used after integer rounding. Padding has no corresponding image location and is excluded from loss and decoding.

Training uses only matched detector proposals from the 142 training photographs. A reference corner contributes to loss only when it is known and inside the predicted crop. Both runs use seed 46, 30 epochs, batch size 32, 32-bit AdamW and the final checkpoint. They each use 3815 supervised crops and complete 3600 parameter updates. Boxes and original scores remain fixed during evaluation. A decoded unavailable coordinate remains an unavailable output; no reference-based replacement is applied. These models borrow the image-and-hint idea from PoseFix but do not reproduce its synthetic-error generation or complete training objective.

A third model predicts a coordinate correction directly. Carreira et al.'s Iterative Error Feedback repeatedly combines an image with a rendered current pose, predicts a bounded correction, and feeds the corrected pose into the next iteration [13]. The cassette experiment uses one learned correction step, without reproducing that iterative procedure. For each available initial point:

$$
\mathbf p_k^{\mathrm{new}}=\mathbf p_k^{\mathrm{initial}}+\Delta\mathbf p_k. \tag{6}
$$

The model receives the colour crop and two initial-corner maps. A ResNet50 backbone feeds global average pooling, which summarizes each feature map by its mean, followed by four outputs: a horizontal and vertical offset for each corner. The output weights and biases start at zero, preserving the initial points exactly before learning. Predicted crop-coordinate offsets are divided by the actual horizontal and vertical resize scales before addition in Equation (6). Thus, a displacement learned in the resized crop is converted to the correct displacement in original-image pixels.

Training requires both a known reference corner and an available initial estimate. Unavailable initial points remain unchanged. SmoothL1 loss applies a quadratic penalty to a small coordinate-component error and a linear penalty to a larger one [14]. The transition is one resized-crop pixel, applied separately to the horizontal and vertical components. This training penalty differs from the Euclidean pair-success criterion in Equation (3). Hint noise is disabled. The model uses 3809 supervised crops, seed 46, 30 epochs and the final checkpoint. It preserves detector boxes and scores.

### 5.4 Common evaluation procedure

The primary comparison evaluates the supplied model and proposed method at input size 1280, object confidence strictly above 0.25, corner confidence at least 0.50, and matching IoU at least 0.50. The model's ordinary suppression uses IoU 0.70 and at most 300 outputs. Both methods additionally suppress redundant boxes at IoU 0.70 after coordinates have been clipped to the image. This common processing makes the comparison about the supplied checkpoint and proposed method under declared settings.

These thresholds serve different decisions. The object threshold decides whether a candidate is retained, the corner threshold decides whether a coordinate is available for the pair measurement, and the suppression threshold decides whether overlapping predictions compete. The matching threshold determines which retained predictions agree with reference fronts during evaluation. A lower object threshold can admit useful low-scoring detections and additional erroneous responses, so a recovered miss is assessed together with changes in extra predictions.

Both methods were run again on all 20 photographs in the same computing environment using 32-bit tensors. The proposed method filters both views at the common object threshold before association and averaging. It averages 648 of 687 retained original-view records. Saved predictions are scored against the supplied, earlier fixed and reviewed annotation references. The earlier fixed and reviewed counts reproduce the archived results exactly. Changing only the scoring annotations requires no further inference or training.

The local models are evaluated at the same object threshold and reference definitions. The two heatmap runs share identical detector outputs, initialization, training examples and sample orders. The correction model uses a different output layer, loss and target mask. It is an alternative design, rather than a one-variable comparison with the heatmap models. Saved model files, training records, coordinate transforms and scores were checked. Selected heatmap cases and all 71 predictions in two correction-model images were reproduced on a CPU without changing their scored outcomes. Appendix 11.2 records the computing environments.

Clipping a predicted box to the image can leave zero width or height. Such a record has no area and cannot match a reference. It is retained in the output count as an extra prediction, so evaluation does not improve merely by omitting invalid outputs. The fixed matching and suppression thresholds otherwise remain unchanged.

### 5.5 Controlled training comparisons

Two experiments separate changes that the complete proposed method combines. Each experiment trains a control and a candidate independently from the same supplied checkpoint, using seed 46, 30 epochs, batches of four, the same AdamW settings and the final epoch. Initial parameter hashes, image order, sample exposure and optimizer-update indices agree within each pair. These checks establish the intended comparison; one paired seed does not measure variation across repeated training runs.

The first experiment changes the classification loss while keeping the same 142 photographs and 3999 training fronts. Li et al.'s Generalized Focal Loss includes Quality Focal Loss, which supports continuous quality targets and gives greater relative emphasis to predictions farther from their targets [15]. The native cassette model already constructs a continuous classification target using box and available-corner agreement. The candidate multiplies each native binary cross-entropy term by the squared difference between its predicted score and that target. An already well-scored example therefore receives less relative weight than one with a large discrepancy. The assignment and target-construction rules, other loss terms and normalization remain native. Their realized target values can change as the models learn.

This is a classification-loss adaptation, without the paper's complete detector or distributed box representation. It also changes the total classification-loss scale relative to the other objectives. Consequently, its outcome cannot isolate error-dependent weighting from a general reduction in classification strength. Each arm receives 4260 image presentations and 80 optimizer updates.

The second experiment changes only the training annotations on the same 137 supplied photographs. One arm uses the supplied labels; the other uses the frozen reviewed labels already used by the adapted detector. These contain 3772 and 3784 fronts, respectively, and 92 label files differ numerically. Both arms exclude the five supplementary photographs, retain the same 13 empty training images, and use the same supplied validation labels during training. Each receives 4110 image presentations and 78 optimizer updates. The comparison estimates the effect of this label revision during adaptation from shared pretrained weights. The earlier training history of those weights remains common to both arms.

Each returned model is evaluated using its original view and the same two-view corner procedure. All three annotation references and the fixed operating thresholds are retained. The classification comparison additionally measures original-view AP50: average precision at reference IoU 0.50 over the ranked detection list. As lower-scoring predictions are admitted, it averages interpolated precision at 101 recall levels. It uses a prespecified score floor of 0.001 and at most 300 predictions per photograph. This helps distinguish a change at the 0.25 operating point from a change in ranking across scores. It is a single-overlap detection measure, not the multiple-overlap COCO mean average precision.

Both arms within each experiment use the same GPU and software environment. PyTorch documents that equal seeds do not guarantee identical results across platforms [16]. Hardware and software are therefore recorded briefly for reproducibility in Appendix 11.2. Recognition is assessed through the paired comparisons; the study does not compare processor speeds.

Two subsequent diagnostics hold the models fixed. The first observes the returned classification models on the unchanged, unaugmented training photographs in evaluation mode, with gradients disabled. It records candidate eligibility, assignments, predicted scores and normalized supervision targets separately for fronts with zero, one and two known corners. A separate box-only assignment pass removes the corner-agreement term while allowing candidate selection and normalization to change. This is a descriptive counterfactual at fixed predictions, not retraining with a different objective.

The second diagnostic uses saved final detections on the 20 development photographs. It separately applies an existing reflected-view support requirement and additional hard suppression at IoU 0.60 or 0.50. Each surviving output retains its box, scores and already averaged corner coordinates; evaluation matching is performed again after deletion. These exploratory restrictions act after the existing inference pipeline. They do not replace native NMS, recover previously suppressed proposals or select a deployment threshold. A maximum-cardinality matching calculation also measures how many reference matches are possible using the unchanged final boxes, without replacing the declared evaluator.

## 6 Simulation Results

A numerical check tests whether the heatmap models' output grid alone prevents five-pixel accuracy. A 64 by 64 map on a 256 by 256 canvas places neighbouring grid locations four canvas pixels apart, before conversion back to image pixels. For every matched complete target, synthetic Gaussian heatmaps are centred exactly at the reference corners and passed through the same crop and decoding procedure as the learned outputs. The decoder selects a valid-image maximum, shifts it by a quarter of a heatmap cell in each coordinate direction indicated by the neighbouring responses, and maps the result back using the actual rounded resize scales.

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

Archived original-view proposals from the same checkpoint locate where the 18 missed fronts lose support. A decoded proposal is a candidate box before confidence filtering and suppression; it is eligible for this diagnostic when its clipped box reaches reference IoU 0.50. Twelve missed fronts have an eligible proposal but none passes the object-confidence threshold. Three retain an eligible proposal after filtering but lose it in native NMS. The remaining three have no eligible decoded proposal. No additional reference coverage is lost in the extra suppression after clipping.

In all three NMS cases, the higher-scoring box that suppresses an eligible alternative survives as an unmatched extra. This suggests examining the relationship between score ordering and box accuracy. It does not establish why the scores were learned or prove that weakening NMS would improve the total result. Indeed, the common extra suppression already removes six duplicates without losing a matched front. Removing suppression indiscriminately would discard that observed benefit.

The 51 extras comprise five duplicates, 36 predictions whose greatest reference IoU is at least 0.10 but below 0.50, and ten below 0.10. These geometric groups indicate possible localization and background errors for review [5]; they are not physical verdicts. Displacement, omitted annotations and disputed boundaries require checking the original photograph.

Both older references favour the supplied model (Appendix 11.1), so the recognition improvement is tied to the reviewed annotations. Individual targets can improve or regress.

Keeping reflected-view detections with the same corner averaging gives 15 misses, 43 extras and unchanged five- and ten-pixel pair counts under the reviewed reference. It recovers eight fronts and loses five, but duplicates double from five to ten and both older references lose corner successes. This exploratory trade-off does not replace the original-view detections used in the main comparison.

### 7.2 Results of the local corner models

Table 3 retains the supplied model as the main benchmark and includes the adapted detector before either averaging or local refinement. This single-view control is the input to the local models and makes their added effect visible. All adapted methods retain 636 matched fronts, 18 misses, 51 extra outputs and five duplicates at the common threshold.

Table 3. Complete-pair results for the supplied benchmark, adapted controls and local models. All rows use the reviewed reference and 583 eligible targets.

| Method | Both corners within 5 pixels | Both within 10 pixels |
| --- | ---: | ---: |
| Supplied model | 555/583 | 577/583 |
| Adapted detector without refinement | 566/583 | 582/583 |
| Proposed two-view method | 569/583 | 583/583 |
| Local image model | 547/583 | 582/583 |
| Local image and initial-corner model | 558/583 | 581/583 |
| Local coordinate-correction model | 564/583 | 582/583 |

The correction model has nine more five-pixel successes than the supplied baseline, but two fewer than its unrefined adapted control and five fewer than the proposed two-view method. The improvement over the supplied model therefore cannot be credited to adding the correction stage. Relative to the proposed method, three targets improve and eight regress.

The mean larger-of-two corner error is 2.32 pixels for the adapted single-view control, 2.27 pixels after coordinate correction and 2.16 pixels for the proposed method, measured on the same 583 complete fronts. The correction slightly improves the mean over its input while reducing the five-pixel success count from 566 to 564. Small improvements on many targets can coexist with a few targets crossing the five-pixel boundary. Neither the success count nor the mean error supports replacing the proposed method.

The heatmap models sometimes select nearby cassette features. In one examined case, the right-corner response moves towards a neighbouring junction, producing an error above 20 pixels. The correction model can also move an acceptable point beyond the five-pixel tolerance. Saved-checkpoint replay reproduces these outcomes. Together with Section 6, this supports examining the training-error examples used by local models.

### 7.3 Effect of classification loss

Table 4 compares the new native-loss control with the Quality Focal Loss candidate under the reviewed reference. The control is trained alongside the candidate; the earlier proposed detector is retained as a separate comparison. Repeating a nominal recipe does not make a newly trained checkpoint identical to an older one.

Table 4. Paired classification-loss comparison. Detection and corner counts use the two-view method at object confidence above 0.25; AP50 uses original-view predictions above the prespecified 0.001 floor.

| Measure | Native loss | Quality Focal Loss |
| --- | ---: | ---: |
| Missed fronts | 18 | 12 |
| Extra predictions | 49 | 83 |
| Duplicates included in extras | 4 | 17 |
| Both corners within 5 pixels | 568/583 | 570/583 |
| Both corners within 10 pixels | 583/583 | 582/583 |
| Original-view AP50 | 0.983865 | 0.983953 |

All six recovered fronts have incomplete reference corners, and no previously matched front is lost. However, the candidate produces 34 additional extras: 13 more duplicates and 21 more other unmatched predictions. Its two new five-pixel successes have no corresponding five-pixel losses, while one previously successful pair fails the ten-pixel criterion. Both original-view models have 567 five-pixel successes before corner averaging. The final two-view gain therefore must be distinguished from an improvement in the original-view corner count.

AP50 changes by approximately 0.000088, or 0.0088 percentage point. At the lower score floor both models match 651 of 654 fronts, but the candidate emits 5183 predictions compared with 2768 for the control. The result demonstrates a recall-versus-extras trade-off at the fixed operating point, with little change in this ranking measure. It does not establish an overall improvement or justify choosing a new threshold from these same development photographs.

Training diagnostics also narrow the next question. The native training procedure has two prediction branches: one can assign several candidates to a reference target, while the other aims for one assigned candidate per target. Across 18,626 presentations of targets with incomplete corners, the one-to-many branch leaves four unassigned in the control and three in the candidate; the one-to-one branch leaves one unassigned in each. These are repeated augmented target presentations, not distinct fronts. Almost all therefore receive training assignments.

The completed frozen diagnostic examines all 3999 training fronts: 137 with no known corner, 465 with one and 3397 with two. Every front receives an assignment under both checkpoints, both heads and both overlap variants. No surviving assignment has a zero supervision target. This provides no evidence of missing assignment in the observed unaugmented pass; it cannot reconstruct transient events during historical training.

For the native control's one-to-many head, the mean predicted score and supervision target are respectively 0.167 and 0.283 for zero-known fronts, 0.560 and 0.626 for one-known fronts, and 0.791 and 0.803 for two-known fronts. These averages cover assigned anchors, not each front's highest-scoring detection, so they cannot directly predict which front passes the inference threshold. Among one-known fronts, the mean known-corner agreement is about 0.985, above the mean box component of about 0.68. In the box-only counterfactual, their mean target falls from 0.626 to 0.466 for the control and from 0.644 to 0.477 for the focal-loss model. Assignments and normalization also change. The observation therefore weakens the proposed explanation that removing corner agreement is an obvious remedy, without predicting the result of such retraining.

### 7.4 Effect of training annotations

Table 5 holds the reviewed evaluation reference fixed while changing only the training labels. This asks a different question from rescoring unchanged predictions against different references in Appendix 11.1.

Table 5. Paired training-label comparison on the same 137 training photographs. Both models use two-view corner averaging and the reviewed 654-front evaluation reference.

| Measure | Supplied training labels | Reviewed training labels |
| --- | ---: | ---: |
| Missed fronts | 47 | 15 |
| Extra predictions | 85 | 51 |
| Duplicates included in extras | 6 | 13 |
| Detected fronts with two known corners | 569/583 | 583/583 |
| Detected fronts with fewer than two known corners | 38/71 | 56/71 |
| Both corners within 5 pixels | 541/583 | 572/583 |
| Both corners within 10 pixels | 566/583 | 583/583 |

The reviewed-label model recovers 33 fronts and loses one, giving 32 fewer misses. Fourteen recoveries are complete fronts, 15 have one known corner and four have neither. It gains 37 five-pixel successes and loses six, giving a net increase of 31. Thirteen gains come from recovered complete fronts; 24 improve fronts already detected by the control. The remaining recovered complete front has a maximum-corner error of 5.607 pixels and passes only the ten-pixel criterion.

Here, recovery means that a front gains an eligible reference match. Focused inspection finds cases where the control already places a prediction over the cassette, but its box is too broad to reach IoU 0.50. Improved agreement can therefore arise from better object extent, rather than a newly appearing response to previously overlooked image material. These examples do not explain every recovered match.

The decrease in total extras coexists with an increase in duplicates. Twelve of the reviewed-label model's 13 duplicates correspond to incomplete reference fronts. Each duplicate overlaps the already matched prediction by no more than the suppression threshold, so their survival is consistent with the implemented NMS rule. This points to remaining instance-separation and localization questions; it does not show that every overlapping output should be removed.

The direction reverses under the older annotation references. With two-view averaging, supplied-reference five-pixel successes fall from 531 to 513 when reviewed training labels replace supplied labels. Under the earlier fixed reference they fall from 544 to 523. Appendix 11.3 reports the accompanying detection and ten-pixel results. The controlled experiment therefore shows that training-label choice changes recognition and that its measured benefit depends on the scoring definition. It does not independently establish which label version follows the physical surfaces more accurately.

The reviewed-label model also has three fewer misses and three more five-pixel successes than the earlier proposed method, with the same 51 extras but eight more duplicates. This is an encouraging additional candidate. The comparison cannot isolate the effect of excluding the five supplementary images because it spans separate runs and computing environments. The paired label intervention in Table 5 is the supported causal comparison for the declared training procedure.

### 7.5 Duplicate and unmatched output audit

The final-output audit tests whether simple restrictions resolve the reviewed-label candidate's remaining duplicates. The reflected-view rule retains an original output only when the existing mutual association supplies a reflected partner at IoU at least 0.50. Two additional rules suppress lower-scoring final boxes at IoU above 0.60 or 0.50. Table 6 reports all three restrictions separately; they are not combined.

Table 6. Exploratory restrictions on the same saved reviewed-label outputs. The reviewed reference contains 654 fronts, including 583 complete corner pairs. Duplicates are included in extras.

| Final-output rule | Misses | Extras | Duplicates | Within 5 pixels | Within 10 pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged outputs | 15 | 51 | 13 | 572/583 | 583/583 |
| Require reflected partner | 19 | 27 | 6 | 572/583 | 583/583 |
| Additional suppression at 0.60 | 16 | 32 | 2 | 572/583 | 583/583 |
| Additional suppression at 0.50 | 19 | 23 | 1 | 572/583 | 583/583 |

The 0.60 restriction removes 19 extras, including 11 duplicates, but also loses one previously matched incomplete front. The other two restrictions each lose four incomplete fronts, with different target identities. Consequently, unchanged complete-pair counts conceal a decline in partial-front detection, from 56 of 71 to 55 or 52 of 71. Requiring reflected agreement does not obtain an independent view of the physical scene: both network inputs derive from the same photograph. Agreement can support an output without establishing its correctness, and disagreement is not sufficient grounds to discard a real front.

For all five audited two-view output sets, the maximum number of matches permitted by the unchanged boxes equals the ordinary confidence-ordered matching count under the reviewed reference. The counts are 636 for the earlier proposed method, 636 for the paired native control, 642 for the focal-loss model, 607 for the supplied-label model and 639 for the reviewed-label model. Changing only the matching algorithm, while retaining the same IoU eligibility rule, cannot recover another matched front from those final boxes. It could change which prediction supplies a corner estimate, which is why the matching rule remains fixed.

These results support reporting partial-front coverage alongside complete-pair accuracy and separating output-selection failures from reference disagreements. They do not establish a better suppression threshold. All three annotation references and all three restrictions are retained in the diagnostic record; none is used to select a replacement method.

### 7.6 Effect of restoring identifiable partial fronts

Table 7 scores six unchanged output sets against the 654-front reference and its 658-front extension. The four restored targets comprise a lower band beside the gripper, a fragment behind the opposite jaw, a top-clipped wedge and a bottom-clipped band. The latter has no available lower corner; each of the other three has one. No model is rerun or retrained, and no operating threshold changes.

Table 7. Reference-inclusion effect with predictions fixed. Arrows show the 654-front reference followed by the 658-front extension. Duplicate and corner counts are unchanged; each corner count has denominator 583.

| Method | Misses | Extras | Duplicates | Within 5 pixels | Within 10 pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supplied model | 34 → 37 | 61 → 60 | 5 | 555 | 577 |
| Proposed method | 18 → 21 | 51 → 50 | 5 | 569 | 583 |
| Paired native-loss control | 18 → 21 | 49 → 48 | 4 | 568 | 583 |
| Focal-loss candidate | 12 → 15 | 83 → 82 | 17 | 570 | 582 |
| Supplied-label training | 47 → 50 | 85 → 84 | 6 | 541 | 566 |
| Reviewed-label training | 15 → 18 | 51 → 50 | 13 | 572 | 583 |

Every method matches the restored bottom-clipped band and misses the same other three additions. One existing extra therefore becomes a reference match, while three additional misses enter the evaluation population. No retained target changes its matched prediction. Duplicate counts and complete-pair outcomes stay unchanged, and the relative differences between the training arms are preserved. For the reviewed-label model, incomplete-front coverage becomes 57 of 75, compared with 56 of 71 before the additions.

The three previously tested output restrictions also retain that newly supported output. Under the extended reference, additional suppression at 0.60 gives 19 misses, 31 extras and two duplicates; at 0.50 it gives 22, 22 and one. Requiring a reflected partner gives 22, 26 and six. Their five- and ten-pixel complete-pair counts remain 572 and 583. The inclusion correction therefore preserves the earlier conclusion: fewer extras can accompany lost incomplete-front matches, without a change in the complete-pair score.

The added misses do not show deterioration of the detector, and the removed extra does not show new learning. Both follow from correcting which visible fronts are represented in the reference. The original pixels and target definition motivated the additions before the new scores were inspected. These results illustrate why an annotation amendment must retain its own version and why a complete-pair score cannot describe coverage of all identifiable fronts.

## 8 Discussions

### 8.1 Findings and interpretation

The proposed method improves reviewed-reference detection and corner results over the supplied model at the declared operating point. The subsequent controlled comparisons show why a single success count is insufficient for choosing a replacement: focal weighting recovers fronts while increasing extras, and reviewed-label training improves several measures while increasing duplicates.

The contribution is a cassette-front annotation procedure and an evaluated recognition method, compared with the supplied model. Development experiments guide method selection. The final measurements describe the combined system. The single-view control isolates the observed effect of adding averaging to the adapted detector at the declared settings: five-pixel successes increase from 566 to 569, and ten-pixel successes from 582 to 583. It does not separate the effects of annotation changes, additional photographs and further training that produced the adapted detector.

The annotation sensitivity is substantial. The supplied and earlier fixed references contain different object entries, boundaries and corner positions, and both give the supplied model a higher five-pixel count. Rescoring changes agreement with a reference; it does not change the images, model parameters or predictions. The new training-label experiment changes learning while holding each scoring reference fixed, and its ranking also reverses across reference versions. Predictions informed review, and the frozen reviewed reference retains documented target exclusions as well as geometric amendments. These findings are consistent with the motivation for noise-aware evaluation [4], without establishing that any one annotation version is independently correct.

Both annotations and methods have been examined on the same 20 photographs. These images consequently serve as development evidence rather than an untouched test of generalization. Many cassettes share a photograph's lighting, viewpoint and rack arrangement, so 583 complete fronts do not constitute 583 independently captured scenes. New capture groups with independently reviewed labels are needed to assess transfer. A repeat annotation exercise should also measure how consistently reviewers can place the specified corners before interpreting differences of a few pixels as model limitations alone.

Two denominators answer two different operational questions. The score 569/583 asks how many reference fronts with two known corners are successfully returned. Examining emitted pairs instead asks how much support the accepted outputs have in the reference. After pair acceptance and fresh matching, the proposed method offers 678 pairs: 569 satisfy the five-pixel criterion, 14 matched pairs exceed it, 48 match incomplete reference pairs, and 47 are unmatched. The last two groups lack complete reference support; they cannot automatically be counted as physically wrong. The 97.60% reference-pair result is therefore not the accuracy of all 678 emitted pairs.

The proposed method produces no detections on the three evaluation photographs without annotated fronts. This is a useful check on those images, but it provides limited evidence about other empty scenes. The study does not measure a robot's positioning success or validate an automatic decision to move an arm.

### 8.2 Further work

The controlled label result makes annotation repeatability a priority. Review should begin with unmarked originals and the visible-front definition, then compare uncertain instance identities, box boundaries and corner locations. Disagreement between models or label versions identifies a review region; it does not determine the correct edit. A second independent annotation pass should measure instance and availability disagreements separately from coordinate differences. A delayed repeat by the same reviewer would measure within-reviewer repeatability instead.

The completed score-target diagnostic provides no observed assignment gap and shows that removing known-corner agreement can lower targets for one-corner fronts. The next intervention should therefore address a measured instance-separation, geometry or score-ordering failure, rather than assume that partial fronts lack supervision. The output audit also shows that stricter selection trades previously matched incomplete-front detections for fewer extras. Neither observation supports an automatic change of loss or suppression settings.

The focused inclusion review resolves four historical exclusions while retaining three ownership uncertainties. Those remaining regions should be adjudicated explicitly; a plausible white fragment does not by itself establish a separate front. Deleting an annotation does not create an ignore region in the implemented evaluator. TIDE likewise shows that some apparent detection errors arise from ambiguous or missing annotations [5]. Model-unmatched outputs should therefore not be used indiscriminately as negative training examples. The restored cases belong to evaluation photographs, so their inclusion changes measurement rather than the completed fine-tuning supervision.

Two published methods offer bounded later tests. Soft-NMS reduces the scores of overlapping proposals instead of immediately deleting them [17]. A valid cassette comparison would start from decoded proposals before suppression and state which suppression stage it replaces; applying it only to the already retained outputs cannot recover discarded proposals. IoU-Net predicts localization quality to decide which competing box should survive [18]. Its relevance is the observed cases in which a higher-scoring but poorly placed box suppresses an eligible alternative. A learned ranker adds a component and would need to improve detection while preserving the two-corner task. Neither published result establishes that the adaptation improves cassette recognition.

The reviewed-label candidate warrants repeated paired seeds under the same declared protocol. A separate matched experiment would be required to measure the effect of the five supplementary photographs. Neither comparison can erase the supplied checkpoint's earlier exposure to its original training labels. Independent capture groups and separately reviewed evaluation labels remain necessary before claiming transfer beyond the reused development photographs.

Local models currently learn from proposals produced by a detector trained on the same photographs. Their training inputs may therefore contain a different distribution of errors from new photographs; this remains a hypothesis to measure. One option is to generate proposals with detectors whose complete training histories exclude the photographs used to train the refiner. Holding them out only during fine-tuning leaves earlier checkpoint exposure intact. PoseFix suggests synthetic errors based on measured error patterns as another source of training inputs [7]. Arbitrary coordinate noise is not evidence that those patterns have been reproduced. Promising results require repeated paired seeds and evaluation on independent capture groups.

## 9 Conclusions

A method combining reviewed training annotations, detector adaptation, and equal averaging of associated two-view corner estimates was developed for cassette recognition. On the 654-front reviewed reference from 20 photographs, it reduces missed fronts from 34 to 18 and extra predictions from 61 to 51 compared with the supplied model. Five-pixel pair success increases from 555 to 569 out of 583, and ten-pixel success increases from 577 to 583. Duplicate predictions remain at five.

Three learned local corner models were evaluated, including a coordinate-correction model, but none exceeds the proposed method. The coordinate correction reduces five-pixel success relative to its input detector despite a small improvement in mean error. Quality Focal Loss recovers six fronts relative to its matched native control but adds 34 extras, with almost unchanged AP50.

Changing only the training labels on 137 photographs gives 32 fewer misses, 34 fewer extras and 31 more five-pixel successes under the reviewed reference. Duplicates increase, and older references favour the supplied-label arm. Restoring four identifiable partial fronts in a separate reference adds three misses and removes one extra for every compared method, with predictions and relative model differences unchanged. The frozen assignment audit finds no unassigned training front, while simple output restrictions reduce duplicates at the cost of previously matched incomplete-front detections. These findings favour targeted annotation and error review before another component is introduced. The next phase should measure annotation repeatability and verify promising designs across repeated paired seeds and independent captures.

## 10 References

[1] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 779–788, doi: 10.1109/CVPR.2016.91.

[2] Ultralytics, “Ultralytics YOLO26,” Ultralytics Docs. Accessed: Nov. 28, 2026. [Online]. Available: https://docs.ultralytics.com/models/yolo26/

[3] U. Tkachenko, A. Thyagarajan, and J. Mueller, “ObjectLab: Automated diagnosis of mislabeled images in object detection data,” in ICML Workshop on Data-centric Machine Learning Research, 2023. [Online]. Available: https://arxiv.org/abs/2309.00832

[4] J. Murrugarra-Llerena and C. R. Jung, “Noise-aware evaluation of object detectors,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2025, pp. 9322–9331, doi: 10.1109/WACV61041.2025.00903.

[5] D. Bolya, S. Foley, J. Hays, and J. Hoffman, “TIDE: A general toolbox for identifying object detection errors,” in Computer Vision – ECCV 2020, Lecture Notes in Computer Science, vol. 12348. Springer, 2020, pp. 558–573, doi: 10.1007/978-3-030-58580-8_33.

[6] B. Xiao, H. Wu, and Y. Wei, “Simple baselines for human pose estimation and tracking,” in Computer Vision – ECCV 2018, Lecture Notes in Computer Science, vol. 11210. Springer, 2018, pp. 472–487, doi: 10.1007/978-3-030-01231-1_29.

[7] G. Moon, J. Y. Chang, and K. M. Lee, “PoseFix: Model-agnostic general human pose refinement network,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 7765–7773, doi: 10.1109/CVPR.2019.00796.

[8] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. 34th Int. Conf. Mach. Learn. (ICML), Proceedings of Machine Learning Research, vol. 70, 2017, pp. 1321–1330. [Online]. Available: https://proceedings.mlr.press/v70/guo17a.html

[9] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proc. Int. Conf. Learning Representations (ICLR), 2019. [Online]. Available: https://arxiv.org/abs/1711.05101

[10] J. Li et al., “Human pose regression with residual log-likelihood estimation,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 11005–11014, doi: 10.1109/ICCV48922.2021.01084.

[11] D. Shanmugam, D. Blalock, G. Balakrishnan, and J. Guttag, “Better aggregation in test-time augmentation,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 1194–1203, doi: 10.1109/ICCV48922.2021.00125.

[12] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.

[13] J. Carreira, P. Agrawal, K. Fragkiadaki, and J. Malik, “Human pose estimation with iterative error feedback,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 4733–4742, doi: 10.1109/CVPR.2016.512.

[14] PyTorch Contributors, “SmoothL1Loss,” PyTorch Documentation, ver. 2.14. Accessed: Nov. 28, 2026. [Online]. Available: https://docs.pytorch.org/docs/2.14/generated/torch.nn.SmoothL1Loss.html

[15] X. Li et al., “Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 21002–21012. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2020/file/f0bda020d2470f2e74990a07a607ebd9-Paper.pdf

[16] PyTorch Contributors, “Reproducibility,” PyTorch Documentation, ver. 2.14. Accessed: Nov. 28, 2026. [Online]. Available: https://docs.pytorch.org/docs/2.14/notes/randomness.html

[17] N. Bodla, B. Singh, R. Chellappa, and L. S. Davis, “Soft-NMS — Improving object detection with one line of code,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 5562–5570, doi: 10.1109/ICCV.2017.593.

[18] B. Jiang, R. Luo, J. Mao, T. Xiao, and Y. Jiang, “Acquisition of localization confidence for accurate object detection,” in Computer Vision – ECCV 2018, Lecture Notes in Computer Science, vol. 11218. Springer, 2018, pp. 816–832, doi: 10.1007/978-3-030-01264-9_48.

## 11 Appendices

### 11.1 Sensitivity to the annotation reference

All three annotation sets describe the same 20 photographs. The supplied labels contain 664 fronts, the earlier fixed labels contain 666, and the reviewed labels contain 654. Their numbers of complete corner pairs appear in Table 8. Original supplied labels were verified against the supplied data manifest before scoring.

Table 8. Effect of changing the scoring reference while keeping predictions fixed. Each arrow shows the supplied model followed by the proposed method. Corner-success denominators are the complete-pair counts in the first column.

| Annotation reference | Misses | Extra predictions | Within 5 pixels | Within 10 pixels |
| --- | ---: | ---: | ---: | ---: |
| Supplied, 584 pairs | 39 → 46 | 56 → 69 | 521 → 514 | 568 → 565 |
| Earlier fixed, 584 pairs | 26 → 34 | 41 → 55 | 534 → 525 | 580 → 574 |
| Reviewed, 583 pairs | 34 → 18 | 61 → 51 | 555 → 569 | 577 → 583 |

The proposed method loses seven five-pixel successes under the supplied labels and nine under the earlier fixed labels, while gaining 14 under the reviewed labels. Changes to object entries, boundaries and corner positions can change both box assignment and coordinate errors. This comparison measures evaluation-reference sensitivity; it does not isolate the effect of changing training labels.

### 11.2 Training and coordinate settings

The common inference rerun used Python 3.12.14, PyTorch 2.9.0+cpu, and Ultralytics 8.4.182.

The paired classification-loss experiment used one NVIDIA A100-SXM4-40GB; the paired training-label experiment used one NVIDIA L4. Both used Python 3.13.16, PyTorch 2.14.1+cu130 and Ultralytics 8.4.182. Within each experiment, both arms shared the same recorded runtime. Mixed-precision training was disabled. This setting specifies the training tensor precision; the saved training receipts do not separately establish every backend arithmetic flag. The classification experiment's inference explicitly disabled TensorFloat32. Exact cross-platform training identity was not measured [16]. The GPU models are recorded as experimental conditions; latency is not an outcome of this study.

The adapted detector uses AdamW with an initial learning rate of 0.0001, final learning-rate factor 0.01, weight decay 0.0005, and three warmup epochs. Training includes rotation up to five degrees, translation fraction 0.05, scale fraction 0.15, and horizontal reflection probability 0.50. Mosaic, mixup, vertical reflection, shear, and perspective transforms are disabled. The final checkpoint follows 1080 batches and 80 parameter updates; gradient accumulation combines several batches into each update. The supplied 664-front validation labels monitor training, and the final comparison uses the separately reviewed annotations.

The heatmap models share an NVIDIA Tesla T4 and use batch size 32, AdamW learning rate 0.0001, weight decay 0.0001, and cosine decay to 0.000001 over 30 epochs. Both expose 3815 crops per epoch, giving 114450 crop presentations. Colour augmentation is disabled. Initial-corner hints have Gaussian standard deviation eight canvas pixels, with four-pixel standard-deviation position noise during training only. Target heatmaps have standard deviation two heatmap cells. The output grid represents canvas positions at four-pixel intervals; padding is masked in both loss and decoding.

The correction model uses an NVIDIA L4 with the same optimizer, learning-rate schedule, batch size and epoch budget, with no hint noise. Its 7204 eligible endpoint targets produce 114270 crop presentations and 3600 parameter updates. Eight known reference corners lack a usable initial estimate and are excluded. A nonfinite correction or a corrected point outside the image becomes unavailable. The recorded training runtime uses PyTorch 2.14.1+cu130; the selected CPU replay uses PyTorch 2.9.0+cpu.

Every comparison retains the image identity, reference version, model bytes, inference settings, predicted boxes and points, and individual gained or lost targets. These records allow the reported counts and individual outcomes to be recomputed using the declared one-to-one matching rule.

### 11.3 Training-label results under all references

Table 9 keeps each evaluation reference fixed while comparing the two training-label arms. Each arrow runs from the model trained with supplied labels to the model trained with reviewed labels. Both use two-view corner averaging. The three rows concern the same 20 photographs and are sensitivity analyses, not three independent evaluation datasets.

Table 9. Training-label intervention under all three evaluation references. The first column gives the complete-pair denominator for both corner columns.

| Evaluation reference | Misses | Extra predictions | Within 5 pixels | Within 10 pixels |
| --- | ---: | ---: | ---: | ---: |
| Supplied, 584 pairs | 34 → 50 | 62 → 76 | 531 → 513 | 572 → 566 |
| Earlier fixed, 584 pairs | 23 → 38 | 49 → 62 | 544 → 523 | 584 → 575 |
| Reviewed, 583 pairs | 47 → 15 | 85 → 51 | 541 → 572 | 566 → 583 |

These results measure the effect of the frozen training-label revision from the shared supplied checkpoint. They do not compare training from random initialization or establish a unique physical reference from model agreement. The five supplementary photographs and any future annotation amendments require separate controlled comparisons.
