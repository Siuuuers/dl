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
  - [7.6 Effect of restoring identifiable partial fronts](#76-effect-of-restoring-identifiable-partial-fronts)
  - [7.7 Further tests of partial-front detection](#77-further-tests-of-partial-front-detection)
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

The system detects tissue cassette fronts and locates their lower-left and lower-right corners. These points describe the lower edge's position and orientation in the image. The intended application passes the vision outputs to a robotic arm for cassette localization. This project evaluates boxes and corner coordinates in image pixels. Arm control is a later integration task, whose physical tolerance and message format remain to be specified.

Cassettes are closely arranged in a rack. Another cassette or a gripper may cover part of a front, and the image boundary may cut it off. The target rectangle follows the cassette's own visible front material, excluding the rack and gripper. Figure 1 illustrates this definition.

![A cassette front and its two lower corners](figures-r57/figure-1-target.png)

Figure 1. Camera image and enlarged cassette front. The rectangle marks the annotated front and the two points mark its actual lower corners. Neighbouring cassettes and the rack provide context. A corner is marked unavailable when its position cannot be identified reliably.

The aim is to detect every identifiable front, including a partial front, and locate each corner supported by the image, using a simple processing method. The supplied image collection and trained model provide the starting point. The main comparison evaluates that unchanged model and the proposed method using the same measurement rules.

The proposed method first detects fronts and estimates their corners in the original photograph. It then processes a horizontal reflection, or left–right mirror image, of the same photograph. Corresponding corner estimates are mapped back and averaged. The original-view boxes define the output detections. Separate experiments examine local corner models, a different classification loss and revised training annotations.

### 3.2 Related research

Redmon et al. introduced You Only Look Once, or YOLO, which predicts object rectangles and class scores from a complete image through one neural network [1]. The supplied model belongs to the later YOLO26 family. Its pose-estimation variant also predicts landmarks, meaning specified points on an object [2]. Here, the pose consists of the two lower cassette corners in the two-dimensional image.

Annotation research guides the review process. ObjectLab compares detector predictions with labels and ranks images that may contain missing, misplaced or wrongly classified boxes [3]. This helps direct a reviewer to possible errors, which must then be checked in the photograph. Murrugarra-Llerena and Jung show that uncertain box boundaries can raise or lower measured performance, especially for small objects and strict overlap requirements [4]. Their findings motivate comparing annotation references, while the original photographs establish the cassette boundaries.

TIDE groups detection errors and estimates how correcting each group would affect average precision, a summary of performance across confidence thresholds [5]. This shows why an overall score can hide distinct failures. The cassette evaluation follows that principle by reporting misses, extras, duplicates and corner accuracy separately. It uses the fixed-threshold counts in Section 4 rather than TIDE's full average-precision analysis.

For landmark localization, Xiao et al. extract image features and enlarge them through learned upsampling layers to produce a heatmap for each landmark [6]. A heatmap is a grid of spatial scores, with a high score indicating a likely landmark position. This design guides the local image model. PoseFix also supplies an initial pose to a refinement network and trains it with synthetic errors based on measured human-pose mistakes [7]. It motivates using initial cassette corners as input and examining whether the training errors resemble those seen in use. Both papers study human landmarks. Their adaptations to cassettes are evaluated here.

## 4 Background

### 4.1 Annotations and detection measurements

Annotations specify a front's rectangle and its known lower corners. The rectangle's edges follow the image axes, so its lower vertices need not coincide with the physical corners of a tilted or partly covered cassette. A good box can therefore contain poor corner estimates. Image coordinate x increases to the right and y downwards. Multiplying normalized x and y values by image width and height gives pixel coordinates. The images are 1280 by 720 pixels.

A predicted box B is compared with a reference box G using intersection over union, abbreviated IoU:

$$
\operatorname{IoU}(B,G)=\frac{|B\cap G|}{|B\cup G|}. \tag{1}
$$

The numerator is the shared area, and the denominator is the combined area counted once. IoU is zero for nonoverlapping boxes and one for identical boxes. It measures box agreement. Corner accuracy is measured separately.

Evaluation processes predictions from highest to lowest object confidence. Each prediction is matched to the remaining reference with greatest IoU, provided the overlap is at least 0.50. Each reference receives at most one prediction, preventing repeated predictions from counting as several detected fronts.

A missed front has no matched prediction. An extra prediction has no matched reference. A duplicate is an extra that overlaps an already matched front by at least 0.50, so duplicates form part of the extra count. For example, one reference front and one prediction with IoU 0.49 produce both a miss and an extra. These categories describe agreement with the annotations. Deciding what physically occupies the region requires the photograph.

Before evaluation, non-maximum suppression (NMS) reduces repeated detections. It keeps a high-scoring prediction and removes lower-scoring predictions whose overlap with it exceeds a threshold. NMS compares predictions with predictions. Evaluation matching compares predictions with annotations. The proposed method also compares predictions across original and reflected inputs to associate their corners. These three operations use IoU for different decisions.

A duplicate can survive correctly implemented NMS. Both predictions may overlap the same reference by at least 0.50 while overlapping each other by no more than the suppression threshold of 0.70.

With C matched fronts, M missed fronts, and E extra predictions, precision P and recall R are:

$$
P=\frac{C}{C+E},\qquad R=\frac{C}{C+M}. \tag{2}
$$

Precision describes the fraction of predictions that match a reference. Recall describes the fraction of reference fronts that are found.

### 4.2 Corner accuracy and confidence

A corner's error is the straight-line, or Euclidean, distance between its predicted position p and annotated position g in original-image pixels. Subscripts L and R denote the lower-left and lower-right corners. Both corners meet tolerance epsilon when their larger error satisfies:

$$
e_k=\sqrt{(p_{x,k}-g_{x,k})^2+(p_{y,k}-g_{y,k})^2},\qquad \max(e_L,e_R)\leq\epsilon. \tag{3}
$$

A successful pair also requires a matched box, two finite coordinates inside the image and both corner scores at least 0.50. Results use tolerances of five and ten original-image pixels. The denominator includes every reference front with two known corners, even if detection fails or a predicted corner is unavailable.

Errors of three and six pixels fail the five-pixel criterion but pass the ten-pixel criterion. Testing the larger error prevents one good corner from hiding a poor second corner. Mean error answers a different question. It can fall even while a few pairs cross a failure threshold.

Availability records whether a reviewer can locate a corner from the photograph. Confidence is a model score used to accept or reject a predicted coordinate. Guo et al. show that classification confidence can disagree with observed correctness and study calibration, which measures and adjusts that relationship [8]. For cassette corners, the relationship between score and five-pixel accuracy still needs measurement.

A front with an unavailable reference corner still contributes to detection evaluation, but its complete pair cannot be scored. Such outputs are reported separately. Converting pixel error into physical positioning error requires camera calibration and scene geometry beyond these two image points.

## 5 Methodology and Implementation

The unchanged supplied model is the benchmark. The adapted detector continues training from its weights. The proposed method then averages that detector's original and reflected corner estimates. Three local corner models offer alternative refinements of the adapted detector's original-view outputs. Comparing each with its unrefined input reveals the contribution of the added step.

### 5.1 Data and annotation review

Table 1 describes the data. Whole photographs retain their training or evaluation membership. The expanded training set includes 13 images without annotated fronts, and the evaluation set includes three. These empty images help check responses to background.

Table 1. Image and annotation counts. The controlled comparisons retain the 654-front reviewed reference. Section 7.6 assesses the extension separately.

| Set | Photographs | Annotated fronts | Known corners |
| --- | ---: | ---: | ---: |
| Supplied training data | 137 | 3772 | 6839 |
| Reviewed and expanded training data | 142 | 3999 | 7259 |
| Reviewed evaluation reference | 20 | 654 | 1220 |
| Extended reference after inclusion review | 20 | 658 | 1223 |

The reviewed training data contain 3397 fronts with two known corners, 465 with one, and 137 with neither. The evaluation reference contains 583, 54, and 17 fronts in these respective groups.

Annotations follow the cassette's own front, including a continuous frontal rim or tab. The box excludes rack, gripper and neighbouring cassette material. Shadowed cassette material remains part of the target. Points mark the actual lower corners. A corner whose position is uncertain is marked unavailable.

An identifiable partial front can receive a box even when both lower corners are hidden. An unavailable corner supplies no coordinate target during training. The front still counts as an object.

Review corrected omitted fronts, repeated entries and inaccurate boxes or points using the original images. Model predictions assisted this work, and the evaluation photographs were inspected during development. The original supplied validation labels contain 664 fronts, and an earlier fixed reference contains 666. Appendix 11.1 compares these references with the reviewed labels using unchanged predictions.

The frozen 654-front reference retains earlier case-specific exclusions, including some visible partial fronts. It provides a stable benchmark for the controlled comparisons, with completeness still subject to review. An unmatched prediction may represent an excluded front, a disputed boundary or a model error.

A later review examined seven historical exclusions in five photographs, using unmarked images before historical and current label overlays. It restored four distinct visible fronts across three photographs in a separate reference. Their historical boxes and three available corners remained supported by the images, so only inclusion changed. Three regions remain unresolved because front or tool ownership is uncertain. The extended reference contains 658 fronts: 583 with two known corners, 57 with one and 18 with neither. Existing rows, historical references and training labels remain byte-for-byte unchanged. This agent-assisted review knew the prior cases and covered those seven regions. Independent human review of the full photographs remains outstanding.

### 5.2 Proposed recognition method

The baseline is the supplied YOLO26s-Pose checkpoint saved after epoch 173. A checkpoint stores learned model parameters at a particular training stage. This model predicts one object category and two corners per cassette. Its parameters remain unchanged for the baseline comparison.

The proposed detector continues training from the same checkpoint on the reviewed and expanded data, a process called fine-tuning. It uses seed 46, 30 epochs, input size 1280, batches of four images and 32-bit training tensors. An epoch presents the training set once. A seed fixes a run's initial random-number sequence. AdamW uses the loss gradient to guide parameter updates and applies weight decay separately [9], with initial learning rate 0.0001 and weight decay 0.0005. The final epoch is used, keeping checkpoint selection independent of the reported corner result.

A loss measures the discrepancy that training seeks to reduce. The existing losses for boxes, landmarks, availability and residual log-likelihood are retained. Residual log-likelihood estimation learns an adjustment to a reference distribution of coordinate errors, making the error model more flexible [10]. Confidence calibration is a separate question, as described in Section 4.2. Appendix 11.2 gives further training settings.

The adapted detector processes the original photograph and its horizontal reflection. Simple Baselines combines original and reflected pose estimates by averaging heatmaps [6]. Shanmugam et al. show that averaging transformed inputs' predictions can improve overall classification accuracy while worsening individual cases [11]. Here, corresponding cassette coordinates receive equal weights, and evaluation records both gains and losses.

The two inputs contain the same photograph's information. Reflection gives another network response, with no new observation of a hidden surface. Shared errors can persist after averaging.

For image width W, continuous horizontal coordinates from the reflected view are returned to the original image frame by:

$$
x=W-x',\qquad y=y'. \tag{4}
$$

This mapping follows the detector's continuous coordinate representation. Box boundaries reverse, and the lower-left and lower-right point slots exchange because the original left corner appears on the right in the reflection. The mapping is checked before comparison.

The original-view boxes define the output detections. An original and reflected prediction are associated when they share a class, each is the other's greatest-overlap partner and their IoU is at least 0.50. This mutual selection gives each prediction at most one partner. Equal overlaps select the earlier partner in output order. If both predictions have two valid corners with scores at least 0.50, corresponding coordinates are averaged:

$$
\mathbf p_k^{\mathrm{final}}=\frac{\mathbf p_k^{\mathrm{original}}+\mathbf p_k^{\mathrm{reflected}}}{2},\qquad k\in\{L,R\}. \tag{5}
$$

Otherwise, the original coordinates remain. Every original-view box, score, class and output identity is preserved, so averaging affects only corner accuracy. Associations and weights use predictions alone. Original corner scores still set the acceptance threshold, and their relationship to the averaged coordinates' accuracy remains unmeasured.

### 5.3 Local corner models

Two local models examine image crops around the adapted detector's predictions. Each crop expands the box to 1.5 times its width and height to include surroundings. Its boundaries round outwards and clip to the photograph. Resizing preserves its aspect ratio within a 256 by 256 canvas, with padding filling unused space.

Both models use a ResNet50 feature extractor, or backbone, initialized with ImageNet classification weights. ResNet shortcut connections add a block's input to its learned transformation, helping train deep networks [12]. These internal additions differ from the final coordinate correction described below. Three learned upsampling stages enlarge the feature maps into two 64 by 64 heatmaps, one per corner, following Simple Baselines [6].

The first model receives the colour crop. The second also receives two Gaussian maps, smooth peaks centred on the detector's initial corners. These maps show the starting estimates, following PoseFix's use of an image and initial pose [7]. Both models use the same starting weights and five input channels: three colour channels and two hint maps. The image-only model fills the two additional channels with zeros, isolating the effect of the corner hints.

Training compares output heatmaps with Gaussian peaks at known reference corners using mean squared error over valid cells. The maps encode spatial scores. Decoding takes the highest valid response, adjusts it using neighbouring responses and maps it back to the photograph. The conversion uses the actual horizontal and vertical resize scales after rounding. Padding is excluded from training loss and decoding because it represents no image location.

Training uses matched detector proposals from the 142 training photographs. A corner contributes only if it is known and inside the predicted crop. Both runs use seed 46, 30 epochs, batch size 32, 32-bit AdamW and the final checkpoint, with 3815 supervised crops and 3600 parameter updates each. Evaluation preserves detector boxes and scores. Unavailable decoded coordinates remain unavailable. These models adapt PoseFix's image-and-hint input, while using the cassette training examples and heatmap loss described here in place of its synthetic-error procedure.

A third model directly predicts coordinate corrections. Carreira et al.'s Iterative Error Feedback repeatedly combines an image with a rendered pose, predicts a bounded correction and feeds the result into the next iteration [13]. The cassette adaptation uses one correction step for each available initial point:

$$
\mathbf p_k^{\mathrm{new}}=\mathbf p_k^{\mathrm{initial}}+\Delta\mathbf p_k. \tag{6}
$$

The model receives the colour crop and two initial-corner maps. A ResNet50 backbone feeds global average pooling, which replaces each feature map with its mean, followed by four outputs: horizontal and vertical offsets for both corners. Output weights and biases start at zero, so the model initially preserves the input points. Dividing the predicted offsets by the actual horizontal and vertical resize scales converts them to original-image pixels before Equation (6) is applied.

Training requires a known reference corner and an available initial estimate. Unavailable initial points remain unchanged. SmoothL1 loss is quadratic for small errors and linear for larger ones [14]. Its transition is one resized-crop pixel, applied separately to horizontal and vertical errors. Equation (3) instead measures the combined distance in original-image pixels. Hint noise is disabled. The model uses 3809 supervised crops, seed 46, 30 epochs and the final checkpoint, preserving detector boxes and scores.

### 5.4 Common evaluation procedure

The primary comparison uses input size 1280, object confidence strictly above 0.25, corner confidence at least 0.50 and matching IoU at least 0.50. Ordinary model suppression uses IoU 0.70 and at most 300 outputs. Both methods also suppress overlapping boxes at IoU 0.70 after clipping coordinates to the image.

Object confidence determines which detections are retained. Corner confidence determines whether a point can enter the pair measurement. Suppression determines which predictions compete, and matching determines their agreement with annotations. Lower object confidence can admit both useful detections and erroneous responses, so gains in detected fronts must be read alongside changes in extras.

Both methods were rerun on all 20 photographs in the same computing environment using 32-bit tensors. The proposed method filters both views before association and averages 648 of 687 retained original-view records. The saved outputs are scored against the supplied, earlier fixed and reviewed references. The earlier fixed and reviewed counts exactly reproduce the archived results. This rescoring changes only the annotations used for measurement.

The local models use the same object threshold and references. The heatmap runs share detector outputs, initial weights, training examples and sample order. The correction model changes three things together: its output layer, its loss and the points eligible for training. It therefore tests an alternative design. Checks covered saved models, training records, coordinate transforms and scores. CPU replay reproduced selected heatmap cases and all 71 predictions in two correction-model images with unchanged scored outcomes. Appendix 11.2 records the environments.

Clipping can leave a box with zero width or height. This invalid output remains an extra because its zero area prevents a reference match. Counting it preserves the record of the detector's outputs.

### 5.5 Controlled training comparisons

Two paired experiments examine changes within the proposed method. Each trains a control and candidate from the same supplied checkpoint using seed 46, 30 epochs, batches of four, the same AdamW settings and the final epoch. Within each pair, initial parameter hashes, image order, exposure and update indices agree. Variation between repeated training runs remains unmeasured with one paired seed.

The first experiment changes classification loss on the same 142 photographs and 3999 fronts. Li et al.'s Quality Focal Loss, part of Generalized Focal Loss, supports continuous quality targets and gives more relative weight to predictions farther from their targets [15]. The existing cassette implementation already builds a target score from box and known-corner agreement. The candidate multiplies each existing binary cross-entropy term by the squared score–target difference. It therefore reduces the relative weight of examples whose scores already agree with their targets. Rules for assigning training candidates, constructing targets and normalizing loss remain unchanged, as do other losses. Target values can still change as the models learn.

The experiment adapts the paper's classification loss while retaining the cassette detector and box representation. The weighting also reduces total classification-loss scale relative to other objectives. The result therefore combines error-dependent weighting with a change in classification strength. Each arm receives 4260 image presentations and 80 optimizer updates.

The second experiment changes training annotations on the same 137 supplied photographs. The supplied and frozen reviewed labels contain 3772 and 3784 fronts, with numerical differences in 92 files. Both arms exclude the five supplementary photographs, retain the same 13 empty training images and use the same supplied validation labels. Each receives 4110 image presentations and 78 optimizer updates. This isolates the label revision during fine-tuning from shared weights, whose earlier training history is common to both arms.

Each trained model is evaluated with its original view and the same two-view procedure under all three references and the fixed thresholds. The classification comparison also measures original-view AP50, average precision at reference IoU 0.50 across the ranked detection list. It averages interpolated precision at 101 recall levels using a preset score floor of 0.001 and at most 300 predictions per photograph. This measures ranking across scores alongside the counts at the 0.25 operating point. AP50 uses one overlap threshold. COCO mean average precision averages over several.

Both arms of each experiment share a GPU and software environment. Equal seeds alone do not guarantee identical results across platforms [16]. Appendix 11.2 records the hardware, software and precision settings needed to interpret and reproduce these comparisons.

Later diagnostics hold the models fixed. The first replays the unchanged training photographs in evaluation mode, with augmentation and gradients disabled. It records eligible candidate locations, their assignments to reference fronts, predicted scores and supervision targets, meaning the scores training asks the model to predict. Results are grouped by zero, one or two known corners. A box-only recalculation removes corner agreement and permits assignments and target normalization to change. It measures the immediate effect on these fixed predictions. The effect on learning would require retraining.

The second diagnostic uses saved final detections on the 20 development photographs. It separately requires an existing reflected partner or adds hard suppression at IoU 0.60 or 0.50. Surviving outputs retain their boxes, scores and averaged corners, then undergo fresh evaluation matching. These rules remove outputs after the existing pipeline, where previously suppressed proposals are already unavailable. They are exploratory tests, with operating thresholds held fixed for the main comparisons. A maximum-cardinality calculation also finds the largest possible number of reference matches among the saved boxes, to check whether the ordinary matching order misses available matches.

A further test adds reflected detections that have no existing mutual box association. It retains the original outputs and their averaged corners, appends these unused reflected records and applies the same score-ranked suppression above IoU 0.70. A box association still counts when corner averaging was unavailable. Added records keep their own scores and corners, and may suppress an original output. Confidence thresholds stay unchanged. Fresh matching measures the result under all four references. The addition rule uses predictions alone, although development-reference misses motivated the experiment.

## 6 Simulation Results

A numerical check asks whether the heatmap grid can represent corners accurately enough for the five-pixel criterion. A 64 by 64 map on a 256 by 256 canvas spaces grid locations four canvas pixels apart. For every matched complete target, ideal Gaussian heatmaps centred on the reference corners pass through the same crop and decoder as learned outputs. The decoder selects the highest valid response, shifts it by a quarter cell towards larger neighbouring responses in each direction and converts it back using the actual rounded resize scales.

All 583 complete targets pass the five-pixel criterion, with a largest corner error of approximately 1.50 pixels. The grid and decoder can therefore represent these known positions accurately enough. The remaining question is whether the learned model places its heatmap peaks at those positions.

## 7 Experimental Results

### 7.1 Comparison with the supplied model

Table 2 compares the supplied checkpoint and proposed method using the reviewed reference and common settings in Section 5.4. Duplicates are included in extras.

Table 2. Recognition on the same 20 photographs and reviewed annotations. Pair counts use all 583 fronts with two known corners.

| Measure | Supplied model | Proposed method |
| --- | ---: | ---: |
| Matched fronts out of 654 | 620 | 636 |
| Missed fronts | 34 | 18 |
| Extra predictions | 61 | 51 |
| Duplicate predictions | 5 | 5 |
| Both corners within 5 pixels | 555/583 | 569/583 |
| Both corners within 10 pixels | 577/583 | 583/583 |

Detection precision increases from 91.04% to 92.58%, and recall from 94.80% to 97.25%. Five-pixel pair success rises from 95.20% to 97.60%, a gain of 2.40 percentage points. Six more fronts meet the ten-pixel criterion, while duplicate counts stay the same.

The proposed method detects 583/583 complete fronts, compared with 580/583 for the supplied model. Among the 71 fronts with fewer than two known corners, detection rises from 40/71 (56.34%) to 53/71 (74.65%). The 18 remaining misses have one known corner (13 fronts) or neither (five). Their coverage is separate from the 583/583 complete-pair result.

Saved original-view proposals show where the 18 misses arise. A proposal is a candidate before confidence filtering and suppression. For this check, it is eligible if its clipped box reaches reference IoU 0.50. Twelve missed fronts have eligible proposals, all below the confidence threshold. Three have eligible proposals after confidence filtering but lose them during native NMS. The remaining three have no eligible proposal. The additional suppression after clipping loses no further reference coverage.

In all three NMS cases, a higher-scoring box suppresses an eligible alternative, then survives as an unmatched extra. This points to score ordering as a possible cause worth investigating. Simply weakening suppression carries a cost, since the existing additional suppression removes six duplicates without losing a matched front.

The 51 extras comprise five duplicates, 36 predictions with greatest reference IoU from 0.10 to below 0.50, and ten below 0.10. These groups identify regions to inspect for misplaced boxes, background responses or annotation disagreements [5]. The photographs must resolve those possibilities.

Both older references favour the supplied model (Appendix 11.1), so the recognition improvement is tied to the reviewed annotations. Individual targets can improve or regress.

Using reflected-view detections with the same corner averaging gives 15 misses, 43 extras and unchanged five- and ten-pixel pair counts. It recovers eight fronts and loses five. Duplicates rise from five to ten, and both older references lose corner successes. The main comparison therefore retains original-view detections.

### 7.2 Results of the local corner models

Table 3 includes the supplied benchmark and the adapted detector before refinement. The latter is the input to the local models, so it reveals their added effect. All adapted methods retain 636 matched fronts, 18 misses, 51 extras and five duplicates.

Table 3. Complete-pair results for the supplied benchmark, adapted controls and local models. All rows use the reviewed reference and 583 eligible targets.

| Method | Both corners within 5 pixels | Both within 10 pixels |
| --- | ---: | ---: |
| Supplied model | 555/583 | 577/583 |
| Adapted detector without refinement | 566/583 | 582/583 |
| Proposed two-view method | 569/583 | 583/583 |
| Local image model | 547/583 | 582/583 |
| Local image and initial-corner model | 558/583 | 581/583 |
| Local coordinate-correction model | 564/583 | 582/583 |

The correction model has nine more five-pixel successes than the supplied baseline, but two fewer than its own unrefined input and five fewer than the proposed method. The added correction stage therefore provides no gain in this count. Compared with the proposed method, three targets improve and eight regress.

Across the same 583 complete fronts, the mean larger-of-two error is 2.32 pixels before correction, 2.27 after correction and 2.16 for the proposed method. Correction improves its input's mean while reducing five-pixel successes from 566 to 564. Small gains across many fronts can coexist with a few pairs crossing the failure boundary. The proposed method performs better on both measures.

The heatmap models sometimes select nearby cassette features. In one examined case, the right-corner peak moves towards a neighbouring junction and produces an error above 20 pixels. The correction model can also move an acceptable point beyond five pixels. Saved-checkpoint replay reproduces these failures. Alongside the grid check in Section 6, they suggest examining the error examples used to train the local models.

### 7.3 Effect of classification loss

Table 4 compares the newly trained native-loss control with the Quality Focal Loss candidate. This paired control provides the direct comparison for the loss change. The earlier proposed detector remains a separate trained model.

Table 4. Paired classification-loss comparison. Detection and corner counts use two-view predictions at object confidence above 0.25. AP50 uses original-view predictions above the preset 0.001 floor.

| Measure | Native loss | Quality Focal Loss |
| --- | ---: | ---: |
| Missed fronts | 18 | 12 |
| Extra predictions | 49 | 83 |
| Duplicates included in extras | 4 | 17 |
| Both corners within 5 pixels | 568/583 | 570/583 |
| Both corners within 10 pixels | 583/583 | 582/583 |
| Original-view AP50 | 0.983865 | 0.983953 |

All six recovered fronts have incomplete reference corners, and no previously matched front is lost. The candidate adds 34 extras: 13 duplicates and 21 other unmatched predictions. It gains two five-pixel successes without losing any, but loses one ten-pixel success. Both original-view models have 567 five-pixel successes, so the final gain appears after two-view averaging.

AP50 changes by approximately 0.000088, or 0.0088 percentage point. At the lower score floor, both models match 651 of 654 fronts. The candidate emits 5183 predictions, compared with 2768 for the control. The main result is therefore more detected fronts and more extras at the fixed operating point, with almost unchanged AP50.

Training records help explain where to investigate next. One prediction branch can assign several candidates to a reference target, while the other aims for one. Across 18,626 augmented presentations of fronts with incomplete corners, the one-to-many branch leaves four unassigned in the control and three in the candidate. The one-to-one branch leaves one unassigned in each. These counts include repeated presentations of the same fronts. Almost all receive assignments.

The fixed-model replay examines all 3999 training fronts: 137 with no known corner, 465 with one and 3397 with two. Every front receives an assignment under both models, both prediction branches and both overlap calculations. Every surviving assignment has a positive supervision target. This pass therefore finds no assignment gap on the unchanged images. Historical training, with augmentation and changing parameters, remains represented by the separate training records above.

For the native control's one-to-many branch, mean predicted scores and targets are respectively 0.167 and 0.283 for zero-known fronts, 0.560 and 0.626 for one-known fronts, and 0.791 and 0.803 for two-known fronts. These averages cover all assigned candidate locations. A front's highest-scoring detection may differ, so the averages cannot directly determine whether it passes the inference threshold. Among one-known fronts, mean known-corner agreement is about 0.985, above the mean box component of about 0.68. Removing corner agreement lowers their mean target from 0.626 to 0.466 for the control and from 0.644 to 0.477 for the focal-loss model. Assignments and normalization change too. Thus, the observed corner term often helps these targets, which weakens the case for removing it as an immediate fix.

### 7.4 Effect of training annotations

Table 5 changes the training labels while keeping the reviewed evaluation reference fixed. Appendix 11.1 instead changes the evaluation reference while keeping predictions fixed.

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

The reviewed-label model recovers 33 fronts and loses one, reducing misses by 32. Recoveries include 14 complete fronts, 15 with one known corner and four with neither. It gains 37 five-pixel successes and loses six, a net gain of 31. Thirteen gains come from recovered complete fronts and 24 from already detected fronts. The remaining recovered complete front has maximum corner error 5.607 pixels and passes only the ten-pixel criterion.

Recovery means gaining an eligible reference match. In some inspected cases, the control already predicts a box over the cassette, but the box is too broad to reach IoU 0.50. These gains reflect improved box extent. Other recovered fronts require their own case review.

Total extras fall while duplicates rise. Twelve of the reviewed-label model's 13 duplicates concern incomplete reference fronts. Each overlaps the already matched prediction by no more than the NMS threshold, explaining why it survives. Improving the separation and extent of neighbouring predictions is therefore a more specific question than simply removing all overlapping outputs.

The direction reverses under the older references. Five-pixel successes fall from 531 to 513 under the supplied reference and from 544 to 523 under the earlier fixed reference. Appendix 11.3 gives detection and ten-pixel results. Training-label choice changes the outputs, while the evaluation definition determines which version scores better. Image review is needed to judge their physical accuracy.

The reviewed-label model has three fewer misses and three more five-pixel successes than the earlier proposed method, with the same 51 extras and eight more duplicates. These separate runs also differ in computing environment and use of the five supplementary photographs, so their individual effects are unresolved. Table 5 isolates the training-label change within the paired experiment.

### 7.5 Duplicate and unmatched output audit

The output audit asks whether simple restrictions reduce the reviewed-label model's duplicates. One rule requires an existing mutual reflected partner at IoU at least 0.50. The others suppress lower-scoring final boxes at IoU above 0.60 or 0.50. Table 6 tests these rules separately.

Table 6. Exploratory restrictions on the same saved reviewed-label outputs. The reviewed reference contains 654 fronts, including 583 complete corner pairs. Duplicates are included in extras.

| Final-output rule | Misses | Extras | Duplicates | Within 5 pixels | Within 10 pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged outputs | 15 | 51 | 13 | 572/583 | 583/583 |
| Require reflected partner | 19 | 27 | 6 | 572/583 | 583/583 |
| Additional suppression at 0.60 | 16 | 32 | 2 | 572/583 | 583/583 |
| Additional suppression at 0.50 | 19 | 23 | 1 | 572/583 | 583/583 |

Suppression at 0.60 removes 19 extras, including 11 duplicates, but loses one previously matched incomplete front. The other rules each lose four incomplete fronts, with different identities. Unchanged complete-pair scores therefore hide a drop in incomplete-front detection from 56 of 71 to 55 or 52 of 71. Reflected agreement removes some extras, but also excludes real fronts represented in the reference.

For all five audited output sets, the largest possible match count equals the ordinary confidence-ordered count: 636 for the earlier proposed method, 636 for the paired native control, 642 for focal loss, 607 for supplied-label training and 639 for reviewed-label training. With these final boxes and the same IoU requirement, another matching algorithm cannot recover more fronts. It could assign a different prediction to a front and change the corner score, so the evaluator remains fixed.

The results show why incomplete-front coverage must accompany complete-pair accuracy. All three references and all three restrictions remain in the diagnostic record. Their trade-offs provide no basis here for replacing the main suppression threshold.

### 7.6 Effect of restoring identifiable partial fronts

Table 7 scores six unchanged output sets against the 654-front reference and its 658-front extension. The restored fronts are a lower band beside the gripper, a fragment behind the opposite jaw, a top-clipped wedge and a bottom-clipped band. The last has no available lower corner, and each other front has one. Predictions and operating thresholds stay fixed.

Table 7. Reference-inclusion effect with predictions fixed. Arrows show the 654-front reference followed by the 658-front extension. Duplicate and corner counts are unchanged. Each corner count has denominator 583.

| Method | Misses | Extras | Duplicates | Within 5 pixels | Within 10 pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supplied model | 34 → 37 | 61 → 60 | 5 | 555 | 577 |
| Proposed method | 18 → 21 | 51 → 50 | 5 | 569 | 583 |
| Paired native-loss control | 18 → 21 | 49 → 48 | 4 | 568 | 583 |
| Focal-loss candidate | 12 → 15 | 83 → 82 | 17 | 570 | 582 |
| Supplied-label training | 47 → 50 | 85 → 84 | 6 | 541 | 566 |
| Reviewed-label training | 15 → 18 | 51 → 50 | 13 | 572 | 583 |

Every method matches the restored bottom-clipped band and misses the other three additions. One existing extra becomes a match and three new misses enter the reference population. Previously included fronts retain their matched predictions. Duplicates, complete-pair outcomes and differences between training arms stay unchanged. For the reviewed-label model, incomplete-front coverage becomes 57 of 75, compared with 56 of 71.

All three output restrictions retain the newly supported prediction. Under the extended reference, additional suppression at 0.60 gives 19 misses, 31 extras and two duplicates. At 0.50, these counts are 22, 22 and one. Requiring a reflected partner gives 22, 26 and six. Five- and ten-pixel counts remain 572 and 583. Fewer extras still come with lost incomplete-front matches, hidden by unchanged complete-pair scores.

These changes come entirely from the corrected reference population. The image evidence and target definition supported the additions before rescoring. Keeping both reference versions makes that effect visible and preserves the earlier model comparisons.

### 7.7 Further tests of partial-front detection

Archived proposals for the earlier adapted detector place the three restored misses at different stages. Two fronts have boxes meeting reference IoU 0.50, but their highest eligible scores are approximately 0.00215 and 0.0000966, below the 0.25 cutoff. For the third front, no decoded box reaches IoU 0.50, with a best overlap of 0.409. Lowering confidence can only admit existing boxes. This trace covers the earlier detector's original view. The newer models' saved outputs begin after native suppression, so equivalent stage attribution requires their full candidate records.

Table 8 tests the additional reflected detections described in Section 5.5. It compares five completed methods on the extended 658-front reference. Each method finds more fronts while also producing more extras and duplicates. The three restored fronts examined above remain missed.

Table 8. Adding reflected detections without an existing box partner. Each arrow shows the unchanged method followed by the combined output. Duplicates are included in extras.

| Method | Misses | Extras | Duplicates |
| --- | ---: | ---: | ---: |
| Proposed method | 21 → 16 | 50 → 62 | 5 → 9 |
| Paired native-loss control | 21 → 15 | 48 → 60 | 4 → 9 |
| Focal-loss candidate | 15 → 11 | 82 → 115 | 17 → 33 |
| Supplied-label training | 50 → 48 | 84 → 93 | 6 → 8 |
| Reviewed-label training | 18 → 10 | 50 → 60 | 13 → 18 |

Five-pixel complete-pair counts remain unchanged in all five comparisons. Supplied-label training gains one ten-pixel success, from 566 to 567 out of 583. The other ten-pixel counts remain unchanged.

The reviewed-label model recovers eight fronts and loses no previously matched front. Six recoveries have one known corner and two have neither. All six known corners are returned, with four within five pixels and five within ten. Detection of these partial fronts therefore improves coverage without establishing a complete accurate corner pair. Incomplete-front coverage rises from 57 to 65 out of 75. Under each older reference, the same union recovers two fronts and adds 16 extras, showing that its measured benefit also depends on the reference.

These are saved-output comparisons on the repeatedly inspected development photographs. The original-view output rule is retained because the added coverage comes with more extras and duplicates. The per-target records identify which partial fronts a later method would need to recover with fewer unwanted outputs.

## 8 Discussions

### 8.1 Findings and interpretation

At the declared operating point, the proposed method improves detection and corner accuracy over the supplied model under the reviewed reference. Later comparisons reveal different trade-offs. Focal weighting finds more fronts but adds extras. Reviewed-label training improves several measures while adding duplicates.

The contribution is a cassette-front annotation procedure and a recognition method evaluated against the supplied model. The single-view control isolates averaging's contribution: five-pixel successes rise from 566 to 569, and ten-pixel successes from 582 to 583. The earlier adaptation combines label revision, additional photographs and further training, whose separate effects remain unresolved.

Model rankings reverse with the annotation reference. The older references differ in included fronts, boxes and corners, and favour the supplied model. The paired training-label comparison also reverses direction across references. Rescoring changes measured agreement while leaving images and predictions fixed. Training-label changes instead affect learning. Model-assisted review, retained exclusions and geometric amendments all shape the current reference. These results support the need for annotation sensitivity analysis [4] and independent image review.

The same 20 photographs have informed annotation and method development. Many fronts share lighting, viewpoint and rack arrangement, so 583 complete fronts are not 583 independent captures. New capture groups with independently reviewed labels are needed to test transfer. Repeated annotation should also measure how consistently reviewers place corners before differences of a few pixels are attributed entirely to model error.

The score 569/583 measures success among reference fronts with two known corners. A different question concerns the pairs the system emits. After pair acceptance and fresh matching, the proposed method emits 678 pairs: 569 pass five pixels, 14 matched pairs fail, 48 match incomplete references and 47 are unmatched. The last two groups lack complete reference support, so their physical accuracy remains unresolved. Thus, 97.60% describes reference-pair success, not accuracy across all 678 emitted pairs.

The proposed method produces no detections on the three evaluation photographs without annotated fronts. Wider background reliability and robotic positioning require their own evaluation data.

### 8.2 Further work

Annotation repeatability is a priority. Review should start with unmarked originals and the visible-front definition, then examine uncertain identities, boxes and corners. Model or label disagreement helps locate a region for review. An independent second pass should measure disagreements in instance identity and corner availability separately from coordinate differences. A delayed repeat by the same reviewer instead measures within-reviewer consistency.

The fixed-model diagnostic finds assignments for all training fronts and shows that corner agreement can raise targets for one-corner fronts. The output audit shows that stricter selection loses some detected incomplete fronts. The next change should therefore address an observed separation, box-geometry or score-ordering problem and measure its cost across all these outcomes.

The reflected-output experiment confirms a useful coverage opportunity and its cost. The next diagnostic should record full proposals from the reviewed-label model, including boxes and scores before filtering. This would distinguish missing box geometry from low scores and suppression in the current candidate. The earlier detector's stage trace supports that question, while the newer filtered caches leave it open. Training-label size counts also warrant checking whether the training images contain comparable narrow partial fronts before choosing a new loss or augmentation.

The focused review resolves four historical exclusions and leaves three uncertain regions for adjudication. A plausible fragment requires evidence that it belongs to a separate front. Removing its annotation makes it absent from the reference, without creating an ignore region in the evaluator. TIDE also identifies ambiguous or missing annotations as sources of apparent detection errors [5]. Unmatched predictions therefore need review before use as negative training examples. The restored fronts are in evaluation photographs, so the amendment changes scoring while leaving completed training unchanged.

Soft-NMS reduces overlapping proposals' scores instead of deleting them immediately [17]. An earlier cassette test of linear Soft-NMS, at confidence above 0.13 and suppression IoU 0.70, recovered no missed fronts and removed one final unmatched output. Its outputs before the common post-clipping filter also recovered no fronts, so that filter did not hide a recall gain. Another suppression experiment needs a specific failure to address and must identify which stage it changes.

IoU-Net predicts localization quality to guide which competing box survives [18]. This is relevant when a higher-scoring but poorly placed box suppresses an eligible alternative. A cassette adaptation would add a learned ranking component. It would need to improve detection while preserving corner accuracy to earn that complexity.

The reviewed-label candidate warrants repeated paired seeds under the same protocol. The five supplementary photographs require a separate matched comparison. Both investigations begin from weights already exposed to the supplied checkpoint's earlier training data. Independent capture groups and separately reviewed labels are still needed to assess transfer.

Local models learn from proposals made on photographs used to train their detector. Comparing these errors with errors on new photographs would test whether the refiner learns representative mistakes. One option is to generate proposals with detectors whose full training histories exclude the refiner's photographs, including any earlier checkpoint training. PoseFix offers another approach: synthetic errors derived from measured error patterns [7]. For cassettes, those patterns should first be measured before choosing synthetic perturbations.

## 9 Conclusions

The developed method combines reviewed training annotations, detector fine-tuning and equal averaging of associated original and reflected corner estimates. On the 654-front reviewed reference from 20 photographs, misses fall from 34 to 18 and extras from 61 to 51 compared with the supplied model. Five-pixel successes rise from 555 to 569 out of 583, and ten-pixel successes from 577 to 583. Duplicates remain at five.

None of the three learned local corner models exceeds the proposed method. Coordinate correction slightly lowers mean error but reduces five-pixel success relative to its input. Quality Focal Loss recovers six fronts over its paired native control while adding 34 extras, with almost unchanged AP50.

Changing training labels on 137 photographs gives 32 fewer misses, 34 fewer extras and 31 more five-pixel successes under the reviewed reference. Duplicates increase, and older references favour supplied-label training. Restoring four identifiable partial fronts in a separate reference adds three misses and removes one extra for every method, preserving their relative differences. The fixed-model replay assigns every training front. Output restrictions reduce duplicates at the cost of incomplete-front coverage, while additional reflected detections reverse that trade-off. These findings favour focused annotation and error review before adding complexity. The next phase should measure annotation repeatability, inspect full proposals from the current candidate, repeat promising paired experiments and evaluate independent captures.

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

All three references describe the same 20 photographs. Supplied labels contain 664 fronts, earlier fixed labels 666, and reviewed labels 654. Table 9 gives complete-pair counts. The supplied labels were checked against the original data manifest before scoring.

Table 9. Effect of changing the scoring reference while keeping predictions fixed. Each arrow shows the supplied model followed by the proposed method. Corner-success denominators are the complete-pair counts in the first column.

| Annotation reference | Misses | Extra predictions | Within 5 pixels | Within 10 pixels |
| --- | ---: | ---: | ---: | ---: |
| Supplied, 584 pairs | 39 → 46 | 56 → 69 | 521 → 514 | 568 → 565 |
| Earlier fixed, 584 pairs | 26 → 34 | 41 → 55 | 534 → 525 | 580 → 574 |
| Reviewed, 583 pairs | 34 → 18 | 61 → 51 | 555 → 569 | 577 → 583 |

The proposed method loses seven five-pixel successes under supplied labels and nine under earlier fixed labels, but gains 14 under reviewed labels. Changes to included fronts, boxes and corners affect both matching and coordinate error. This table measures sensitivity to the evaluation reference. Section 7.4 separately examines changing training labels.

### 11.2 Training and coordinate settings

The common inference rerun used Python 3.12.14, PyTorch 2.9.0+cpu, and Ultralytics 8.4.182.

The paired classification-loss experiment used one NVIDIA A100-SXM4-40GB. The paired training-label experiment used one NVIDIA L4. Both used Python 3.13.16, PyTorch 2.14.1+cu130 and Ultralytics 8.4.182, with the same recorded runtime for both arms of each pair. Mixed-precision training was disabled. The saved training records establish tensor precision but leave some backend arithmetic flags unspecified. Classification inference explicitly disabled TensorFloat32. Exact agreement between training runs on different platforms remains unmeasured [16].

The adapted detector uses AdamW with initial learning rate 0.0001, final learning-rate factor 0.01, weight decay 0.0005 and three warmup epochs. Training applies rotations up to five degrees, translation fraction 0.05, scale fraction 0.15 and horizontal reflection probability 0.50. Mosaic, mixup, vertical reflection, shear and perspective transforms are disabled. Gradient accumulation combines several batches per parameter update, giving 1080 batches and 80 updates. The supplied 664-front validation labels monitor training. Final evaluation uses the separately reviewed references.

The heatmap models share an NVIDIA Tesla T4 and use batch size 32, AdamW learning rate 0.0001, weight decay 0.0001 and cosine decay to 0.000001 over 30 epochs. Each sees 3815 crops per epoch, or 114450 presentations. Colour augmentation is disabled. Gaussian hints have standard deviation eight canvas pixels, with position noise of standard deviation four pixels during training only. Target peaks have standard deviation two heatmap cells. Grid locations are four canvas pixels apart, and padding is excluded from loss and decoding.

The correction model uses an NVIDIA L4 with the same optimizer, learning-rate schedule, batch size and epoch budget, and no hint noise. Its 7204 eligible endpoint targets produce 114270 crop presentations and 3600 updates. Eight known reference corners lack a usable initial estimate and are excluded. Nonfinite corrections and points outside the image become unavailable. Training uses PyTorch 2.14.1+cu130, and selected CPU replay uses PyTorch 2.9.0+cpu.

Saved records identify the images, reference versions, models, settings, predictions and individual gained or lost targets. They allow each count and outcome to be recomputed under the declared matching rule.

### 11.3 Training-label results under all references

Table 10 holds each evaluation reference fixed and compares supplied-label training with reviewed-label training. Both models use two-view averaging. All three rows rescore the same 20 photographs, providing three reference-sensitive comparisons.

Table 10. Training-label intervention under all three evaluation references. The first column gives the complete-pair denominator for both corner columns.

| Evaluation reference | Misses | Extra predictions | Within 5 pixels | Within 10 pixels |
| --- | ---: | ---: | ---: | ---: |
| Supplied, 584 pairs | 34 → 50 | 62 → 76 | 531 → 513 | 572 → 566 |
| Earlier fixed, 584 pairs | 23 → 38 | 49 → 62 | 544 → 523 | 584 → 575 |
| Reviewed, 583 pairs | 47 → 15 | 85 → 51 | 541 → 572 | 566 → 583 |

These results measure label revision during adaptation from a shared supplied checkpoint, whose earlier training exposure remains common. Physical label accuracy requires image review. The five supplementary photographs and future annotation amendments need separate controlled comparisons.
