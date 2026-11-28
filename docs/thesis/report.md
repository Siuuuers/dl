# Cassette Front Detection and Corner Localization

Final Year Project I Report

Department of Electronic Engineering

## 1 Abstract

This project investigates tissue cassette front detection and the location of two lower corners in camera images. Closely arranged objects and partial occlusion can cause missed fronts, repeated detections and inaccurate coordinates. The work combines annotation review, adaptation of a supplied pose-estimation model and equal averaging of corner estimates from original and horizontally reflected images.

Against reviewed annotations on 20 development photographs, the proposed method reduces missed fronts from 34 to 18 and extra predictions from 61 to 51 relative to the untouched supplied model. Duplicate predictions remain at five. Both corners lie within five pixels for 569 of 583 complete reference pairs, compared with 555 for the supplied model. Three local corner models do not exceed this result.

Two controlled training comparisons examine what produces useful recognition. Changing the classification loss recovers six fronts but adds 34 extra predictions, with almost unchanged detection average precision. Separately, changing only the training labels on the same 137 photographs reduces misses from 47 to 15 and extras from 85 to 51 under the reviewed reference. Five-pixel successes increase from 541 to 572, while duplicates increase from six to 13. The older evaluation references favour the model trained with supplied labels.

A final suppression rule uses agreement between predicted corners to remove six extras, including two duplicates, while preserving matches for the reviewed-label model. The work contributes a documented annotation procedure and controlled comparisons connecting training labels, output selection and corner accuracy.

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
  - [7.8 Using corner agreement in suppression](#78-using-corner-agreement-in-suppression)
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

The system detects tissue cassette fronts and locates their lower-left and lower-right corners. These points describe the lower edge's position and orientation in the image. The implemented vision stage returns boxes and corner coordinates in image pixels for subsequent robotic-arm integration.

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

The first measurement asks whether a predicted rectangle locates the annotated cassette front. Equation (1) compares the predicted box B with the reference box G through intersection over union, or IoU:

$$
\operatorname{IoU}(B,G)=\frac{|B\cap G|}{|B\cup G|}. \tag{1}
$$

IoU divides the area shared by the boxes by their combined area counted once, judging overlap relative to the front's size. The evaluator uses this value to decide which predictions count as detected fronts. Applying the same overlap requirement to every method gives a consistent basis for counting misses and extras.

Evaluation processes predictions from highest to lowest object confidence. Each prediction is matched to the remaining reference with greatest IoU, provided the overlap is at least 0.50. Each reference receives at most one prediction, preventing repeated predictions from counting as several detected fronts.

A missed front has no matched prediction. An extra prediction has no matched reference. A duplicate is an extra that overlaps an already matched front by at least 0.50, so duplicates form part of the extra count. For example, one reference front and one prediction with IoU 0.49 produce both a miss and an extra. These categories identify the type of disagreement to examine in the photograph.

Before evaluation, non-maximum suppression (NMS) reduces repeated detections. It keeps a high-scoring prediction and removes lower-scoring predictions whose overlap with it exceeds a threshold. NMS compares predictions with predictions. Evaluation matching compares predictions with annotations. The proposed method also compares predictions across original and reflected inputs to associate their corners. These three operations use IoU for different decisions.

A duplicate can survive correctly implemented NMS. Both predictions may overlap the same reference by at least 0.50 while overlapping each other by no more than the suppression threshold of 0.70.

The project reports precision P and recall R together because recovering more fronts can also introduce more unwanted outputs. Equation (2) converts matched fronts C, misses M and extras E into these two rates:

$$
P=\frac{C}{C+E},\qquad R=\frac{C}{C+M}. \tag{2}
$$

Precision measures the fraction of output boxes that match a reference front. Recall measures the fraction of reference fronts detected. Reading both rates together shows the cost of a recovery method that finds more fronts while adding extras.

### 4.2 Corner accuracy and confidence

For the lower edge, both corner positions matter. Equation (3) calculates each predicted point p's straight-line distance from its annotated position g in original-image pixels, then tests the larger error against tolerance epsilon. Subscripts L and R identify the two lower corners:

$$
e_k=\sqrt{(p_{x,k}-g_{x,k})^2+(p_{y,k}-g_{y,k})^2},\qquad \max(e_L,e_R)\leq\epsilon. \tag{3}
$$

Using the larger error requires both ends of the lower edge to meet the chosen tolerance. Results use five and ten original-image pixels. A successful pair also requires a matched box, two finite coordinates inside the image and both corner scores at least 0.50. Every reference front with two known corners stays in the denominator, so the pair count includes the effects of missed boxes and unavailable predictions.

Availability records whether a reviewer can locate a corner in the photograph. Confidence is the model score used to accept its predicted coordinate. Guo et al. explain that confidence and observed correctness can differ and study calibration of their relationship [8]. Accordingly, this project uses confidence to select returned points and pixel error to evaluate their accuracy.

A front with an unavailable reference corner still contributes to detection counts and to measurements of any known corner. Complete-pair results use fronts with both points annotated. All reported corner distances are measured in the original image, keeping the evaluation aligned with the vision stage's coordinate outputs.

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

The completed review corrected omitted fronts, repeated entries and inaccurate boxes or points using the original images, with model predictions assisting inspection. The original supplied validation labels contain 664 fronts, and an earlier fixed reference contains 666. Appendix 11.1 compares these versions with the reviewed labels using unchanged predictions.

The frozen 654-front reference preserves the earlier recorded inclusion decisions and provides the benchmark for the controlled comparisons. The separate extension records the accepted additions from the focused inclusion review, keeping the earlier results reproducible.

The focused inclusion review examined seven historical exclusions in five photographs, using unmarked images followed by historical and current label overlays. It restored four distinct visible fronts across three photographs. Their historical boxes and three available corners were supported by the images, so the accepted edits changed inclusion only. Three other regions were kept unannotated because the visible material could not be confidently assigned to a separate cassette front. The extended reference contains 658 fronts: 583 with two known corners, 57 with one and 18 with neither. Existing rows, historical references and training labels remain byte-for-byte unchanged.

### 5.2 Proposed recognition method

The baseline is the supplied YOLO26s-Pose checkpoint saved after epoch 173. A checkpoint stores learned model parameters at a particular training stage. This model predicts one object category and two corners per cassette. Its parameters remain unchanged for the baseline comparison.

The proposed detector continues training from the same checkpoint on the reviewed and expanded data, a process called fine-tuning. It uses seed 46, 30 epochs, input size 1280, batches of four images and 32-bit training tensors. An epoch presents the training set once. A seed fixes a run's initial random-number sequence. AdamW uses the loss gradient to guide parameter updates and applies weight decay separately [9], with initial learning rate 0.0001 and weight decay 0.0005. The final epoch is used, keeping checkpoint selection independent of the reported corner result.

A loss measures the discrepancy that training seeks to reduce. The existing losses for boxes, landmarks, availability and residual log-likelihood are retained. Residual log-likelihood estimation learns an adjustment to a reference distribution of coordinate errors, making the error model more flexible [10]. Appendix 11.2 gives further training settings.

The adapted detector processes the original photograph and its horizontal reflection, giving two estimates of the same visible corners. Simple Baselines combines original and reflected pose estimates by averaging heatmaps [6]. Shanmugam et al. show that averaging transformed inputs' predictions can improve overall classification accuracy while worsening individual cases [11]. The cassette method uses fixed equal weights for corresponding coordinates and measures both gains and losses.

Before combining these estimates, the reflected prediction must be expressed in the original photograph's coordinates. Equation (4) maps the reflected horizontal position x' back across image width W while leaving the vertical position unchanged:

$$
x=W-x',\qquad y=y'. \tag{4}
$$

The mapping follows the detector's continuous coordinate representation. It reverses the box boundaries and exchanges the lower-left and lower-right point slots. This aligns both predictions with the same cassette corners before averaging.

The original-view boxes define the output detections. An original and reflected prediction are associated when they share a class, each is the other's greatest-overlap partner and their IoU is at least 0.50. This mutual selection gives each prediction at most one partner. Equal overlaps select the earlier partner in output order. If both predictions have two valid corners with scores at least 0.50, corresponding coordinates are averaged:

$$
\mathbf p_k^{\mathrm{final}}=\frac{\mathbf p_k^{\mathrm{original}}+\mathbf p_k^{\mathrm{reflected}}}{2},\qquad k\in\{L,R\}. \tag{5}
$$

Equation (5) takes the midpoint of the two estimates, providing a fixed combination with no additional learned weights. If association or corner-availability checks fail, the original coordinates are retained. Every original-view box, score, class and output identity is preserved, so this step changes only corner accuracy. Original corner scores continue to determine point acceptance.

### 5.3 Local corner models

Two local models examine image crops around the adapted detector's predictions. Each crop expands the box to 1.5 times its width and height to include surroundings. Its boundaries round outwards and clip to the photograph. Resizing preserves its aspect ratio within a 256 by 256 canvas, with padding filling unused space.

Both models use a ResNet50 feature extractor, or backbone, initialized with ImageNet classification weights. ResNet shortcut connections add a block's input to its learned transformation, helping train deep networks [12]. These internal additions differ from the final coordinate correction described below. Three learned upsampling stages enlarge the feature maps into two 64 by 64 heatmaps, one per corner, following Simple Baselines [6].

The first model receives the colour crop. The second also receives two Gaussian maps, smooth peaks centred on the detector's initial corners. These maps show the starting estimates, following PoseFix's use of an image and initial pose [7]. Both models use the same starting weights and five input channels: three colour channels and two hint maps. The image-only model fills the two additional channels with zeros, isolating the effect of the corner hints.

Training compares output heatmaps with Gaussian peaks at known reference corners using mean squared error over valid cells. The maps encode spatial scores. Decoding takes the highest valid response, adjusts it using neighbouring responses and maps it back to the photograph. The conversion uses the actual horizontal and vertical resize scales after rounding. Padding is excluded from training loss and decoding because it represents no image location.

Training uses matched detector proposals from the 142 training photographs. A corner contributes only if it is known and inside the predicted crop. Both runs use seed 46, 30 epochs, batch size 32, 32-bit AdamW and the final checkpoint, with 3815 supervised crops and 3600 parameter updates each. Evaluation preserves detector boxes and scores. Unavailable decoded coordinates remain unavailable. These models adapt PoseFix's image-and-hint input, while using the cassette training examples and heatmap loss described here in place of its synthetic-error procedure.

A third model learns a correction to the detector's initial coordinates. Carreira et al.'s Iterative Error Feedback repeatedly combines an image with a rendered pose, predicts a bounded correction and feeds the result into the next iteration [13]. The cassette adaptation uses one step. Equation (6) adds the predicted displacement to each available initial point:

$$
\mathbf p_k^{\mathrm{new}}=\mathbf p_k^{\mathrm{initial}}+\Delta\mathbf p_k. \tag{6}
$$

This correction form uses the detector's estimate as a starting point and trains the local model to reduce its remaining position error. The model receives the colour crop and two initial-corner maps. A ResNet50 backbone feeds global average pooling, which replaces each feature map with its mean, followed by horizontal and vertical offsets for both corners. Zero output weights and biases preserve the starting coordinates before training. Dividing the offsets by the actual horizontal and vertical resize scales converts them to original-image pixels for Equation (6).

Training requires a known reference corner and an available initial estimate. Unavailable initial points remain unchanged. SmoothL1 loss is quadratic for small errors and linear for larger ones [14]. Its transition is one resized-crop pixel, applied separately to horizontal and vertical errors. Equation (3) instead measures the combined distance in original-image pixels. Hint noise is disabled. The model uses 3809 supervised crops, seed 46, 30 epochs and the final checkpoint, preserving detector boxes and scores.

### 5.4 Common evaluation procedure

The primary comparison uses input size 1280, object confidence strictly above 0.25, corner confidence at least 0.50 and matching IoU at least 0.50. Ordinary model suppression uses IoU 0.70 and at most 300 outputs. Both methods also suppress overlapping boxes at IoU 0.70 after clipping coordinates to the image.

Object confidence determines which detections are retained. Corner confidence determines whether a point can enter the pair measurement. Suppression determines which predictions compete, and matching determines their agreement with annotations. Lower object confidence can admit both useful detections and erroneous responses, so gains in detected fronts must be read alongside changes in extras.

Both methods were rerun on all 20 photographs in the same computing environment using 32-bit tensors. The proposed method filters both views before association and averages 648 of 687 retained original-view records. The saved outputs are scored against the supplied, earlier fixed and reviewed references. The earlier fixed and reviewed counts exactly reproduce the archived results. This rescoring changes only the annotations used for measurement.

The local models use the same object threshold and references. The heatmap runs share detector outputs, initial weights, training examples and sample order. The correction model changes three things together: its output layer, its loss and the points eligible for training. It therefore tests an alternative design. Checks covered saved models, training records, coordinate transforms and scores. CPU replay reproduced selected heatmap cases and all 71 predictions in two correction-model images with unchanged scored outcomes. Appendix 11.2 records the environments.

Clipping can leave a box with zero width or height. This invalid output remains an extra because its zero area prevents a reference match. Counting it preserves the record of the detector's outputs.

### 5.5 Controlled training comparisons

Two paired experiments examine changes within the proposed method. Each trains a control and candidate from the same supplied checkpoint using seed 46, 30 epochs, batches of four, the same AdamW settings and the final epoch. Within each pair, initial parameter hashes, image order, exposure and update indices agree. These controls make the changed training component the basis of the comparison.

The first experiment changes classification loss on the same 142 photographs and 3999 fronts. Li et al.'s Quality Focal Loss, part of Generalized Focal Loss, supports continuous quality targets and gives more relative weight to predictions farther from their targets [15]. The existing cassette implementation already builds a target score from box and known-corner agreement. The candidate multiplies each existing binary cross-entropy term by the squared score–target difference. It therefore reduces the relative weight of examples whose scores already agree with their targets. Rules for assigning training candidates, constructing targets and normalizing loss remain unchanged, as do other losses. Target values can still change as the models learn.

The experiment adapts the paper's classification loss while retaining the cassette detector and box representation. The weighting also reduces total classification-loss scale relative to other objectives. The result therefore combines error-dependent weighting with a change in classification strength. Each arm receives 4260 image presentations and 80 optimizer updates.

The second experiment changes training annotations on the same 137 supplied photographs. The supplied and frozen reviewed labels contain 3772 and 3784 fronts, with numerical differences in 92 files. Both arms exclude the five supplementary photographs, retain the same 13 empty training images and use the same supplied validation labels. Each receives 4110 image presentations and 78 optimizer updates. This isolates the label revision during fine-tuning from shared weights, whose earlier training history is common to both arms.

Each trained model is evaluated with its original view and the same two-view procedure under all three references and the fixed thresholds. The classification comparison also measures original-view AP50, average precision at reference IoU 0.50 across the ranked detection list. It averages interpolated precision at 101 recall levels using a preset score floor of 0.001 and at most 300 predictions per photograph. This measures ranking across scores alongside the counts at the 0.25 operating point. AP50 uses one overlap threshold. COCO mean average precision averages over several.

Both arms of each experiment share a GPU and software environment. Following PyTorch's reproducibility guidance, the seed, hardware, software and precision settings are recorded for the paired comparisons [16]. Appendix 11.2 gives these settings.

The completed diagnostics then hold the models fixed. The first replays the training photographs in evaluation mode, with augmentation and gradients disabled. It records eligible candidate locations, their assignments to reference fronts, predicted scores and supervision targets, meaning the scores training asks the model to predict. Results are grouped by zero, one or two known corners. A box-only recalculation removes corner agreement and measures the resulting assignments and normalized targets at the same model predictions.

A separate inference trace captures every decoded box, score and corner from the reviewed-label checkpoint before filtering, in both original and reflected views. It follows eligible boxes through the confidence cutoff, native suppression and suppression after clipping. Reconstructing the normal predictor outputs from these records checks the trace. Comparison with the archived outputs records computing-environment differences alongside the measured outcomes.

The second output diagnostic uses saved final detections on the 20 photographs. It separately requires an existing reflected partner or adds hard suppression at IoU 0.60 or 0.50. Surviving outputs retain their boxes, scores and averaged corners, then undergo fresh evaluation matching. These rules test output removal after the existing inference pipeline. A maximum-cardinality calculation also finds the largest possible number of reference matches among the saved boxes, checking whether the ordinary matching order misses available matches.

A further test adds reflected detections that have no existing mutual box association. It keeps the original outputs and averaged corners, appends the unused reflected records and applies the same score-ranked suppression above IoU 0.70. A box association still counts when corner averaging was unavailable. Added records keep their own scores and corners and can suppress an original output. Confidence thresholds stay unchanged. Fresh matching measures the result under all four references. The addition rule uses predictions alone.

A separate suppression test uses agreement between predicted corners. RMPE compares estimated human poses to remove repeated detections [17]. This motivates a simpler cassette rule using the two available landmarks. A higher-scoring output suppresses another only when their boxes overlap above IoU 0.60, both have two accepted corners, and the corresponding left corners and right corners are each at most five pixels apart. Equation (1) supplies the box comparison. The distance calculation in Equation (3), applied between predictions, measures agreement about the lower edge. The rule reuses the earlier 0.60 suppression setting and five-pixel tolerance, fixed before this comparison. It adds no learned parameters. Surviving records retain their coordinates and scores, and undergo fresh matching under all four references.

## 6 Simulation Results

A numerical check asks whether the heatmap grid can represent corners accurately enough for the five-pixel criterion. A 64 by 64 map on a 256 by 256 canvas spaces grid locations four canvas pixels apart. For every matched complete target, ideal Gaussian heatmaps centred on the reference corners pass through the same crop and decoder as learned outputs. The decoder selects the highest valid response, shifts it by a quarter cell towards larger neighbouring responses in each direction and converts it back using the actual rounded resize scales.

All 583 complete targets pass the five-pixel criterion, with a largest corner error of approximately 1.50 pixels. The grid and decoder therefore represent these known positions accurately enough. This check separates the resolution of the output representation from the learned localization results in Section 7.2.

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

The 51 extras comprise five duplicates, 36 predictions with greatest reference IoU from 0.10 to below 0.50, and ten below 0.10. These groups organize the output audit around box extent, background responses and annotation agreement [5].

Both older references favour the supplied model, as quantified in Appendix 11.1.

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

The fixed-model replay examines all 3999 training fronts: 137 with no known corner, 465 with one and 3397 with two. Every front receives an assignment under both models, both prediction branches and both overlap calculations. Every surviving assignment has a positive supervision target. This complete pass checks how the trained models assign and score each corner-availability group.

For the native control's one-to-many branch, mean predicted scores and targets are respectively 0.167 and 0.283 for zero-known fronts, 0.560 and 0.626 for one-known fronts, and 0.791 and 0.803 for two-known fronts. These means summarize assigned candidate locations, whereas inference accepts individual detections by confidence. Among one-known fronts, mean known-corner agreement is about 0.985, above the mean box component of about 0.68. Removing corner agreement lowers their mean target from 0.626 to 0.466 for the control and from 0.644 to 0.477 for the focal-loss model. Assignments and normalization change too. In this replay, the corner term supports stronger targets for these fronts, favouring its retention in the next comparison.

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

Recovery means gaining an eligible reference match. In some inspected cases, the control already predicts a box over the cassette, but the box is too broad to reach IoU 0.50. These examples show how improving box extent can recover a front.

Total extras fall while duplicates rise. Twelve of the reviewed-label model's 13 duplicates concern incomplete reference fronts. Each overlaps the already matched prediction by no more than the NMS threshold, explaining why it survives. Improving the separation and extent of neighbouring predictions is therefore a more specific question than simply removing all overlapping outputs.

The direction reverses under the older references. Five-pixel successes fall from 531 to 513 under the supplied reference and from 544 to 523 under the earlier fixed reference. Appendix 11.3 gives detection and ten-pixel results. Training-label choice changes the outputs, while the evaluation definition determines which version scores better.

The reviewed-label model has three fewer misses and three more five-pixel successes than the earlier proposed method, with the same 51 extras and eight more duplicates. This comparison spans separate runs, computing environments and use of the five supplementary photographs. Table 5 provides the controlled comparison of the training-label change.

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

The results show why incomplete-front coverage must accompany complete-pair accuracy. All three references and all three restrictions are retained in the diagnostic record. The main suppression threshold is kept because each tested restriction loses previously matched fronts.

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

The full proposal trace for the reviewed-label model places its 18 original-view misses at three stages. Twelve fronts have eligible boxes, all below the confidence cutoff. Five have no decoded box reaching reference IoU 0.50. One retains an eligible box after confidence filtering but loses it during native suppression. Suppression after clipping loses no further coverage. The reflected view has ten confidence-gate misses, one geometry miss and one native-suppression miss.

Among the three restored misses, two have eligible original-view boxes with highest scores approximately 0.0145 and 0.000162, below the 0.25 cutoff. The remaining front has a best original-view overlap of 0.447. Reflection produces a box with overlap 0.526 for that front, but its score of 0.0845 also fails the cutoff. These records distinguish a missing usable box from an existing box rejected by confidence.

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

The reviewed-label model recovers eight fronts and loses no previously matched front. Six recoveries have one known corner and two have neither. All six known corners are returned, with four within five pixels and five within ten. The added coverage is therefore concentrated in fronts with incomplete annotated corner pairs, rising from 57 to 65 out of 75. Under each older reference, the same union recovers two fronts and adds 16 extras.

The original-view output rule is retained because the added coverage comes with more extras and duplicates. The per-target records identify the partial fronts recovered by the additional view and the unwanted outputs introduced alongside them.

### 7.8 Using corner agreement in suppression

Table 9 tests the corner condition on the reviewed-label model. All rows use its same saved outputs and the extended 658-front reference. The comparison asks whether agreement about the lower edge helps distinguish repeated predictions from nearby fronts.

Table 9. Additional suppression with and without a corner-agreement condition. Duplicates are included in extras.

| Output rule | Misses | Extras | Duplicates |
| --- | ---: | ---: | ---: |
| Unchanged outputs | 18 | 50 | 13 |
| Suppression at 0.60 | 19 | 31 | 2 |
| Suppression at 0.60 with corner agreement | 18 | 44 | 11 |

Requiring corner agreement removes six extras, including two duplicates, while preserving every matched front and its returned corners. Incomplete-front coverage remains 57 of 75, and complete-pair success remains 572 of 583 at five pixels and 583 of 583 at ten pixels. Target-level comparisons confirm that the same rule preserves all matches and corner outcomes under the other three references. It also retains the partial front lost by suppression at 0.60 alone.

The benefit depends on the predictions being filtered. Applied to the supplied-label model, the same rule removes a useful one-corner-front prediction and increases misses from 50 to 51 under the extended reference. Its previously returned known corner was within two pixels. Similar predicted corners can therefore accompany boxes with different reference support. For the reviewed-label model, the measured reduction provides a small improvement using outputs the system already computes.

## 8 Discussions

### 8.1 Findings and interpretation

At the declared operating point, the proposed method improves detection and corner accuracy over the supplied model under the reviewed reference. Later comparisons reveal different trade-offs. Focal weighting finds more fronts but adds extras. Reviewed-label training improves several measures while adding duplicates.

The contribution is a cassette-front annotation procedure and a recognition method evaluated against the supplied model. The single-view control isolates averaging's contribution: five-pixel successes rise from 566 to 569, and ten-pixel successes from 582 to 583. The adaptation combines reviewed labels, additional photographs and further training. The paired experiments then test classification loss and training-label revision separately.

The annotation comparisons distinguish two effects. Rescoring measures agreement with a reference while leaving predictions fixed. Revising training labels changes what the model learns. Reporting both follows the motivation for annotation sensitivity analysis [4] and connects the reviewed target definition to the measured model behaviour.

The comparisons use 20 development photographs that also informed annotation and method development, with one paired seed in each controlled training experiment. Fronts within a photograph share lighting and viewpoint, so the 583 complete fronts are grouped within 20 images. Results retain their named annotation versions because reference choice changes model rankings. These conditions define the collection and training comparisons on which the reported design trade-offs were established.

The score 569/583 measures success among reference fronts with two known corners. The emitted-output audit asks how those results relate to the pairs returned by the system. After pair acceptance and fresh matching, the proposed method emits 678 pairs: 569 pass five pixels, 14 matched pairs fail, 48 match incomplete references and 47 are unmatched. The latter two categories describe the annotation support available for the outputs. The 97.60% result uses the 583 eligible reference pairs, keeping reference-front success and emitted-output coverage distinct.

The proposed method produces no detections on the three evaluation photographs without annotated fronts.

### 8.2 Further work

The fixed-model diagnostic finds assignments for all training fronts and shows that corner agreement can raise targets for one-corner fronts. The output audit shows that stricter selection loses some detected incomplete fronts. The next change should therefore address an observed separation, box-geometry or score-ordering problem and measure its cost across all these outcomes.

The reviewed-label proposal trace now identifies confidence and box geometry as the main sources of remaining misses. Its original-view candidates above the existing cutoff support at most 641 reference matches, compared with 640 after suppression. Changing only suppression within that fixed pool can therefore recover at most one further front. A focused comparison of representative partial-front training examples would address the larger group of low-scoring or poorly bounded fronts. The training-label size counts help identify which small, narrow examples to examine.

Soft-NMS reduces overlapping proposals' scores instead of deleting them immediately [18]. An earlier cassette test of linear Soft-NMS, at confidence above 0.13 and suppression IoU 0.70, recovered no missed fronts and removed one final unmatched output. Its outputs before the common post-clipping filter also recovered no fronts. At those settings, the remaining suppression conflicts point towards the ordering of competing boxes.

IoU-Net predicts localization quality to guide which competing box survives [19]. This is relevant when a higher-scoring but poorly placed box suppresses an eligible alternative. A cassette adaptation would add a learned ranking component. It would need to improve detection while preserving corner accuracy to earn that complexity.

The five supplementary photographs form a separate data question. A matched training comparison using the same checkpoint, label policy and optimizer would measure their contribution to recognition.

For local refinement, comparing detector errors on training photographs with errors on new captures would help choose representative correction examples. Proposals from detectors whose full training histories exclude the refiner's photographs would supply errors on unseen images, including exclusion from the starting checkpoint's data. PoseFix offers another approach through synthetic errors derived from measured patterns [7]. Both approaches focus the training inputs on the mistakes the refiner is intended to correct.

## 9 Conclusions

The developed method combines reviewed training annotations, detector fine-tuning and equal averaging of associated original and reflected corner estimates. On the 654-front reviewed reference from 20 photographs, misses fall from 34 to 18 and extras from 61 to 51 compared with the supplied model. Five-pixel successes rise from 555 to 569 out of 583, and ten-pixel successes from 577 to 583. Duplicates remain at five.

None of the three learned local corner models exceeds the proposed method. Coordinate correction slightly lowers mean error but reduces five-pixel success relative to its input. Quality Focal Loss recovers six fronts over its paired native control while adding 34 extras, with almost unchanged AP50.

Changing training labels on 137 photographs gives 32 fewer misses, 34 fewer extras and 31 more five-pixel successes under the reviewed reference. Duplicates increase, and older references favour supplied-label training. Restoring four identifiable partial fronts in a separate reference adds three misses and removes one extra for every method, preserving their relative differences. The fixed-model replay assigns every training front. Stricter box suppression loses incomplete fronts, while adding reflected detections increases coverage and extras. Requiring corner agreement before stricter suppression removes six extras, including two duplicates, from the reviewed-label model without changing its matches or corner results. This model-specific improvement uses already available predictions. The comparisons establish a practical selection rule: judge each change by misses, extras, duplicates and corner accuracy together.

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

[17] H.-S. Fang, S. Xie, Y.-W. Tai, and C. Lu, “RMPE: Regional multi-person pose estimation,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 2334–2343. [Online]. Available: https://openaccess.thecvf.com/content_ICCV_2017/papers/Fang_RMPE_Regional_Multi-Person_ICCV_2017_paper.pdf

[18] N. Bodla, B. Singh, R. Chellappa, and L. S. Davis, “Soft-NMS — Improving object detection with one line of code,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 5562–5570, doi: 10.1109/ICCV.2017.593.

[19] B. Jiang, R. Luo, J. Mao, T. Xiao, and Y. Jiang, “Acquisition of localization confidence for accurate object detection,” in Computer Vision – ECCV 2018, Lecture Notes in Computer Science, vol. 11218. Springer, 2018, pp. 816–832, doi: 10.1007/978-3-030-01264-9_48.

## 11 Appendices

### 11.1 Sensitivity to the annotation reference

All three references describe the same 20 photographs. Supplied labels contain 664 fronts, earlier fixed labels 666, and reviewed labels 654. Table 10 gives complete-pair counts. The supplied labels were checked against the original data manifest before scoring.

Table 10. Effect of changing the scoring reference while keeping predictions fixed. Each arrow shows the supplied model followed by the proposed method. Corner-success denominators are the complete-pair counts in the first column.

| Annotation reference | Misses | Extra predictions | Within 5 pixels | Within 10 pixels |
| --- | ---: | ---: | ---: | ---: |
| Supplied, 584 pairs | 39 → 46 | 56 → 69 | 521 → 514 | 568 → 565 |
| Earlier fixed, 584 pairs | 26 → 34 | 41 → 55 | 534 → 525 | 580 → 574 |
| Reviewed, 583 pairs | 34 → 18 | 61 → 51 | 555 → 569 | 577 → 583 |

The proposed method loses seven five-pixel successes under supplied labels and nine under earlier fixed labels, but gains 14 under reviewed labels. Changes to included fronts, boxes and corners affect both matching and coordinate error. This table measures sensitivity to the evaluation reference. Section 7.4 separately examines changing training labels.

### 11.2 Training and coordinate settings

The common inference rerun used Python 3.12.14, PyTorch 2.9.0+cpu, and Ultralytics 8.4.182.

The later reviewed-label proposal trace used PyTorch 2.10.0+cpu with the same Ultralytics version and 32-bit inference. Compared with its archived L4 outputs, matched-front status, corner availability and five- and ten-pixel outcomes agree under all four references. One extra changes category to a duplicate under the reviewed references, giving 14 instead of 13 duplicates. The fixed corner condition removes six outputs in each runtime, leaving 44 extras and respectively 12 or 11 duplicates. The recorded environment therefore accompanies each result.

The paired classification-loss experiment used one NVIDIA A100-SXM4-40GB. The paired training-label experiment used one NVIDIA L4. Both used Python 3.13.16, PyTorch 2.14.1+cu130 and Ultralytics 8.4.182, with the same recorded runtime for both arms of each pair. Mixed-precision training was disabled. Classification inference explicitly disabled TensorFloat32. These settings document the matched computing conditions used for the comparisons [16].

The adapted detector uses AdamW with initial learning rate 0.0001, final learning-rate factor 0.01, weight decay 0.0005 and three warmup epochs. Training applies rotations up to five degrees, translation fraction 0.05, scale fraction 0.15 and horizontal reflection probability 0.50. Mosaic, mixup, vertical reflection, shear and perspective transforms are disabled. Gradient accumulation combines several batches per parameter update, giving 1080 batches and 80 updates. The supplied 664-front validation labels monitor training. Final evaluation uses the separately reviewed references.

The heatmap models share an NVIDIA Tesla T4 and use batch size 32, AdamW learning rate 0.0001, weight decay 0.0001 and cosine decay to 0.000001 over 30 epochs. Each sees 3815 crops per epoch, or 114450 presentations. Colour augmentation is disabled. Gaussian hints have standard deviation eight canvas pixels, with position noise of standard deviation four pixels during training only. Target peaks have standard deviation two heatmap cells. Grid locations are four canvas pixels apart, and padding is excluded from loss and decoding.

The correction model uses an NVIDIA L4 with the same optimizer, learning-rate schedule, batch size and epoch budget, and no hint noise. Its 7204 eligible endpoint targets produce 114270 crop presentations and 3600 updates. Eight known reference corners lack a usable initial estimate and are excluded. Nonfinite corrections and points outside the image become unavailable. Training uses PyTorch 2.14.1+cu130, and selected CPU replay uses PyTorch 2.9.0+cpu.

Saved records identify the images, reference versions, models, settings, predictions and individual gained or lost targets. They allow each count and outcome to be recomputed under the declared matching rule.

### 11.3 Training-label results under all references

Table 11 holds each evaluation reference fixed and compares supplied-label training with reviewed-label training. Both models use two-view averaging. All three rows rescore the same 20 photographs, providing three reference-sensitive comparisons.

Table 11. Training-label intervention under all three evaluation references. The first column gives the complete-pair denominator for both corner columns.

| Evaluation reference | Misses | Extra predictions | Within 5 pixels | Within 10 pixels |
| --- | ---: | ---: | ---: | ---: |
| Supplied, 584 pairs | 34 → 50 | 62 → 76 | 531 → 513 | 572 → 566 |
| Earlier fixed, 584 pairs | 23 → 38 | 49 → 62 | 544 → 523 | 584 → 575 |
| Reviewed, 583 pairs | 47 → 15 | 85 → 51 | 541 → 572 | 566 → 583 |

These results measure the frozen training-label revision during adaptation from a shared supplied checkpoint. Both arms use the same photographs and retain the checkpoint's common earlier training exposure.
