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

The system detects tissue cassette fronts and locates their lower-left and lower-right corners in camera images. The two points describe the lower edge's position and orientation in the image plane. The intended application sends vision outputs to a robotic arm for cassette localization. This study evaluates the image measurements; arm control and integration are outside its scope. A tolerance in physical units and the downstream message format have not been specified.

Cassettes are closely arranged in a rack. A front may be partly covered by another cassette or a gripper, or cut by the image boundary. The required rectangle follows the cassette's own visible front material; rack and gripper boundaries must not define the object. Figure 1 illustrates the target.

![A cassette front and its two lower corners](figures-r57/figure-1-target.png)

Figure 1. Camera image and enlarged cassette front. The rectangle marks the annotated front and the two points mark its actual lower corners. Neighbouring cassettes and the rack provide context. A corner is marked unavailable when its position cannot be identified reliably.

The aim is to improve recognition quality while keeping the processing method clear. The project starts from a supplied image collection and trained cassette model. Its main comparison is between that original model and the proposed method, evaluated with the same measurement rules. The target output is every identifiable front, including a partial front, together with whichever corner locations can be supported by the image.

The proposed processing has two main stages. First, a detector identifies candidate fronts and estimates their corner coordinates. Second, the same detector processes a horizontally reflected copy of the photograph; corresponding corner estimates are returned to the original image coordinates and averaged where the two views can be associated. Original-view boxes determine which detections are output. Separate experiments test whether a small model examining an enlarged cassette crop can further improve its corners.

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

Table 1. Image and annotation counts. The reviewed evaluation annotations are the primary reference for the final comparison.

| Set | Photographs | Annotated fronts | Known corners |
| --- | ---: | ---: | ---: |
| Supplied training data | 137 | 3772 | 6839 |
| Reviewed and expanded training data | 142 | 3999 | 7259 |
| Reviewed evaluation reference | 20 | 654 | 1220 |

The reviewed training data contain 3397 fronts with two known corners, 465 with one, and 137 with neither. The evaluation reference contains 583, 54, and 17 fronts in these respective groups.

Annotations follow the cassette's own front, including a continuous frontal rim or tab. Rack, gripper, and neighbouring cassette material are excluded when choosing the box boundaries. A shadow does not remove cassette material from the target. Lower-corner points are placed on the actual cassette corners. When a corner cannot be located reliably, it is marked unavailable rather than assigned an estimated training coordinate.

This definition allows an identifiable partial front to receive a box even when one or both lower corners are hidden. An unavailable point contributes no coordinate target. It does not change the front into a background example. The definition therefore separates evidence that a cassette front exists from evidence for a particular landmark location.

Review corrected omitted fronts, redundant entries, and inaccurate boundaries or landmarks. Each accepted edit was checked against the original image. Model predictions assisted the review, and the same evaluation photographs were inspected during development. The original supplied validation annotations contain 664 fronts; an earlier fixed reference contains 666. Appendix 11.1 compares both with the reviewed reference using unchanged predictions.

### 5.2 Proposed recognition method

The baseline is the supplied YOLO26s-Pose checkpoint saved after epoch 173. A checkpoint stores the model's learned parameters at a particular training stage. This model predicts one object category and two corners per cassette. Its parameters are used unchanged for the baseline comparison.

The proposed detector starts from the same checkpoint and continues training on the reviewed and expanded training data. This adaptation is called fine-tuning. Training uses seed 46, 30 epochs, input size 1280, batches of four images and 32-bit arithmetic. An epoch presents the training set once; a random seed fixes the initial random-number sequence for a run. AdamW updates the parameters using adaptive gradient steps while applying weight decay separately [9]. The initial learning rate is 0.0001 and weight decay is 0.0005. The final epoch is used, avoiding selection of an intermediate checkpoint by the reported corner result.

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

Both methods were run again on all 20 photographs in the same CPU environment using 32-bit arithmetic. The proposed method filters both views at the common object threshold before association and averaging. It averages 648 of 687 retained original-view records. Saved predictions are scored against the supplied, earlier fixed and reviewed annotation references. The earlier fixed and reviewed counts reproduce the archived results exactly. Changing only the scoring annotations requires no further inference or training.

The local models are evaluated at the same object threshold and reference definitions. The two heatmap runs share identical detector outputs, initialization, training examples and sample orders, and use an NVIDIA Tesla T4. The correction model uses an NVIDIA L4 and a different output layer, loss and target mask. It is an alternative design, rather than a one-variable comparison with the heatmap models. Saved model files, training records, coordinate transforms and scores were checked. Selected heatmap cases and all 71 predictions in two correction-model images were reproduced on a CPU without changing their scored outcomes.

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

## 8 Discussions

### 8.1 Findings and interpretation

The proposed method improves reviewed-reference detection and corner results at the declared operating point, with unchanged duplicate counts.

The contribution is a cassette-front annotation procedure and an evaluated recognition method, compared with the supplied model. Development experiments guide method selection. The final measurements describe the combined system. The single-view control isolates the observed effect of adding averaging to the adapted detector at the declared settings: five-pixel successes increase from 566 to 569, and ten-pixel successes from 582 to 583. It does not separate the effects of annotation changes, additional photographs and further training that produced the adapted detector.

The annotation sensitivity is substantial. The supplied and earlier fixed references contain different object entries, boundaries and corner positions, and both give the supplied model a higher five-pixel count. Rescoring changes agreement with a reference; it does not change the images, model parameters or predictions. The reviewed reference represents the accepted visible-front annotations, but predictions informed its review. The sensitivity finding is consistent with the motivation for noise-aware evaluation [4], without establishing that any one annotation version is independently correct.

Both annotations and methods have been examined on the same 20 photographs. These images consequently serve as development evidence rather than an untouched test of generalization. Many cassettes share a photograph's lighting, viewpoint and rack arrangement, so 583 complete fronts do not constitute 583 independently captured scenes. New capture groups with independently reviewed labels are needed to assess transfer. A repeat annotation exercise should also measure how consistently reviewers can place the specified corners before interpreting differences of a few pixels as model limitations alone.

Two denominators answer two different operational questions. The score 569/583 asks how many reference fronts with two known corners are successfully returned. Examining emitted pairs instead asks how much support the accepted outputs have in the reference. After pair acceptance and fresh matching, the proposed method offers 678 pairs: 569 satisfy the five-pixel criterion, 14 matched pairs exceed it, 48 match incomplete reference pairs, and 47 are unmatched. The last two groups lack complete reference support; they cannot automatically be counted as physically wrong. The 97.60% reference-pair result is therefore not the accuracy of all 678 emitted pairs.

The three evaluation photographs without annotated fronts produce no detections at the declared settings. This is a useful check on those images, but it provides limited evidence about other empty scenes. The study does not measure a robot's positioning success or validate an automatic decision to move an arm.

### 8.2 Further work

The prepared training-label comparison holds photographs, starting checkpoint, settings, budget and scoring annotations fixed while changing training labels. Both runs use the same 137 supplied training photographs, so the five additional photographs are excluded from this controlled comparison. The experiment tests adaptation from shared supplied weights. It cannot measure what would have happened if the entire training history, including training before the supplied checkpoint, had used a different annotation policy.

A separate paired experiment examines proposal scores. Li et al.'s Generalized Focal Loss includes Quality Focal Loss, which supports continuous quality targets and gives greater relative emphasis to predictions farther from their targets [15]. The native cassette training procedure already constructs soft classification targets influenced by box and landmark agreement. The prepared experiment retains that target-construction procedure and changes the weighting of its classification loss according to the score-target discrepancy. Following the same construction rule does not force identical target values as the two models learn. The experiment does not reproduce the paper's complete detector or distributed box representation.

Both training comparisons await results. The focal weighting also changes the overall scale of the classification term relative to other losses, so any gain cannot automatically be attributed only to better emphasis on difficult proposals. The analysis should inspect missed partial fronts, extra predictions, duplicates and corner outcomes together. The proposal-stage diagnostic makes score quality a relevant hypothesis, but it has not established that the current training targets are appropriate for every partial front.

Local models currently learn from proposals produced by a detector trained on the same photographs. Their training inputs may therefore contain a different distribution of errors from new photographs; this remains a hypothesis to measure. One option is to generate proposals with detectors whose complete training histories exclude the photographs used to train the refiner. Holding them out only during fine-tuning leaves earlier checkpoint exposure intact. PoseFix suggests synthetic errors based on measured error patterns as another source of training inputs [7]. Arbitrary coordinate noise is not evidence that those patterns have been reproduced. Promising results require repeated paired seeds and evaluation on independent capture groups.

## 9 Conclusions

A method combining reviewed training annotations, detector adaptation, and equal averaging of associated two-view corner estimates was developed for cassette recognition. On the reviewed 20-image reference, it reduces missed fronts from 34 to 18 and extra predictions from 61 to 51 compared with the supplied model. Five-pixel pair success increases from 555 to 569 out of 583, and ten-pixel success increases from 577 to 583. Duplicate predictions remain at five.

Three learned local corner models were evaluated, including a coordinate-correction model, but none exceeds the proposed method. The coordinate correction also reduces five-pixel success relative to the adapted detector that supplies its initial points, despite a small improvement in mean error. Scoring unchanged predictions against the original supplied and earlier fixed annotations reverses the main comparison. The demonstrated improvement is therefore tied to the reviewed target definition and development photographs. The next phase will isolate training-label effects, examine proposal scoring and refinement inputs, and evaluate independent captures.

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
