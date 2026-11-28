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
  - [5.5 Controlled comparisons](#55-controlled-comparisons)
- [6 Simulation Results](#6-simulation-results)
- [7 Experimental Results](#7-experimental-results)
  - [7.1 Comparison with the supplied model](#71-comparison-with-the-supplied-model)
  - [7.2 Results of the local corner models](#72-results-of-the-local-corner-models)
  - [7.3 Effect of classification loss](#73-effect-of-classification-loss)
  - [7.4 Effect of training annotations](#74-effect-of-training-annotations)
  - [7.5 Removing extra predictions](#75-removing-extra-predictions)
  - [7.6 Including identifiable partial fronts](#76-including-identifiable-partial-fronts)
  - [7.7 Remaining misses and reflected detections](#77-remaining-misses-and-reflected-detections)
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

Annotations mark each cassette front with a rectangle and its known lower corners. They form the evaluation reference used to score predictions. Rectangle edges follow the image axes, while the points follow the cassette's actual corners. A well-placed box can therefore contain inaccurate corner estimates. Evaluation measures positions and errors in the original 1280 by 720 image, with x increasing to the right and y downwards.

Intersection over union (IoU) measures how well a predicted box B overlaps a reference box G. Equation (1) divides their shared area by their combined area, counted once:

$$
\operatorname{IoU}(B,G)=\frac{|B\cap G|}{|B\cup G|}. \tag{1}
$$

The evaluator processes predictions from highest to lowest object confidence. Each is matched to the remaining reference with greatest IoU, provided IoU is at least 0.50. Each reference receives at most one prediction.

A missed front has no matched prediction. An extra prediction has no matched reference. A duplicate is an extra that overlaps an already matched reference by at least 0.50. Duplicates are therefore included in extras. A poorly placed prediction can produce both a miss and an extra.

Non-maximum suppression (NMS) removes lower-scoring predictions that overlap a retained prediction too strongly. It compares predictions with each other. Evaluation compares predictions with annotations, while the two-view method associates predictions across original and reflected images. These different comparisons explain why a duplicate can survive NMS.

Equation (2) uses matched fronts C, misses M and extras E to calculate precision P and recall R:

$$
P=\frac{C}{C+E},\qquad R=\frac{C}{C+M}. \tag{2}
$$

Precision is the fraction of returned boxes that match a reference front. Recall is the fraction of reference fronts detected. Together, they show whether recovering more fronts also introduces more extras.

### 4.2 Corner accuracy and confidence

Equation (3) measures each predicted corner p's distance from its annotated position g in pixels. Subscripts L and R identify the lower-left and lower-right corners. Testing the larger error against tolerance epsilon requires both ends of the lower edge to be accurate:

$$
e_k=\sqrt{(p_{x,k}-g_{x,k})^2+(p_{y,k}-g_{y,k})^2},\qquad \max(e_L,e_R)\leq\epsilon. \tag{3}
$$

Results use five- and ten-pixel tolerances. A predicted corner is accepted when its coordinate is finite, lies inside the image and has a score of at least 0.50. A successful pair requires a matched box and two accepted corners within the tolerance. Every reference front with two known corners stays in the denominator, including fronts missed by the detector.

Annotation availability records whether a corner can be located in the photograph. Predicted confidence determines whether the model's point is accepted. Guo et al. show that confidence scores can differ from observed correctness [8]. This project therefore uses confidence for selection and pixel error for accuracy. Fronts with unavailable corners still contribute to detection results and to measurements of any known corner.

## 5 Methodology and Implementation

The supplied model is the benchmark. The proposed method continues its training with reviewed annotations, then averages corner estimates from original and reflected images. Three local models test alternative ways to refine the adapted detector's corners. Comparing each with its unrefined input identifies the added step's contribution.

### 5.1 Data and annotation review

Table 1 summarises the data. Whole photographs retain their training or evaluation membership. The expanded training set includes 13 images without annotated fronts, and the evaluation set includes three, allowing responses to background to be checked.

Table 1. Image and annotation counts. Controlled comparisons use the 654-front reviewed reference. Section 7.6 assesses the 658-front extension separately.

| Set | Photographs | Annotated fronts | Known corners |
| --- | ---: | ---: | ---: |
| Supplied training data | 137 | 3772 | 6839 |
| Reviewed and expanded training data | 142 | 3999 | 7259 |
| Reviewed evaluation reference | 20 | 654 | 1220 |
| Extended reference after inclusion review | 20 | 658 | 1223 |

Boxes follow the cassette's visible front material, including a continuous rim or tab, and exclude the rack, gripper and neighbouring cassettes. Points mark the actual lower corners. An identifiable partial front receives a box even when both corners are hidden. A corner that cannot be located reliably is marked unavailable and supplies no coordinate target during training.

The completed review corrected omitted fronts, repeated entries and inaccurate boxes or points using the original photographs, assisted by prediction overlays. The reviewed training set contains 3397 fronts with two known corners, 465 with one and 137 with neither. The 654-front evaluation reference contains 583, 54 and 17 in these groups.

Two earlier annotation versions cover the same 20 evaluation photographs: the supplied labels contain 664 fronts and an earlier revision contains 666. Appendix 11.1 measures how these versions affect the scores of the same predictions.

A focused review of seven previously excluded regions in five photographs restored four identifiable fronts across three photographs. Three regions remained excluded because their visible material could not be assigned confidently to a separate front. The extended reference contains 583 fronts with two known corners, 57 with one and 18 with neither.

### 5.2 Proposed recognition method

The supplied YOLO26s-Pose model predicts one object category and two corners per cassette. Its checkpoint, the learned parameters saved after epoch 173, provides both the unchanged benchmark and the starting point for fine-tuning.

Fine-tuning makes 30 passes through the reviewed and expanded training data, called epochs, with seed 46, input size 1280 and batches of four. AdamW updates the parameters using loss gradients and applies weight decay separately [9]. The existing training objectives are retained, including residual log-likelihood estimation, which learns a flexible distribution of coordinate errors [10]. The final epoch supplies the evaluated model. Appendix 11.2 records the remaining settings.

The adapted detector processes the original photograph and its horizontal reflection. Simple Baselines averages original and reflected heatmaps [6], while research on test-time augmentation shows that aggregation can improve overall classification results while worsening individual cases [11]. Here, equal averaging combines two coordinate estimates of the same visible corners.

Equation (4) maps a reflected position x', y' back to the original image of width W:

$$
x=W-x',\qquad y=y'. \tag{4}
$$

This follows the detector's continuous coordinate convention. The box boundaries are reversed and the lower-left and lower-right point slots exchanged, aligning corresponding corners before averaging.

Original and reflected detections are associated when each is the other's greatest-overlap partner and their IoU is at least 0.50. All detections belong to the cassette class. If both have two accepted corners, Equation (5) takes the midpoint of each corresponding pair:

$$
\mathbf p_k^{\mathrm{final}}=\frac{\mathbf p_k^{\mathrm{original}}+\mathbf p_k^{\mathrm{reflected}}}{2},\qquad k\in\{L,R\}. \tag{5}
$$

The original-view boxes define the detections. Their scores are retained, including the corner scores used for acceptance. If association or corner checks fail, the original coordinates are used. This step therefore refines corner positions without changing which fronts are returned.

### 5.3 Local corner models

The local models receive crops around the adapted detector's predictions. Each crop expands the box to 1.5 times its width and height, then fits within a padded 256 by 256 canvas while preserving its shape.

Two models use a ResNet50 feature extractor initialized with ImageNet weights. ResNet's shortcut connections help train deep networks [12]. Three upsampling stages produce a 64 by 64 heatmap for each corner, following Simple Baselines [6]. The first model receives the colour crop. The second also receives smooth Gaussian peaks marking the detector's initial corners, adapting PoseFix's image-and-pose input [7]. Both have five input channels, with the first model's two hint channels set to zero.

Training compares the heatmaps with Gaussian peaks at known corners inside the crop, using mean squared error. Padding is excluded from training and decoding. The decoder converts each heatmap into a point by selecting its strongest valid response, adjusting it using neighbouring responses and mapping it back to the photograph with the actual resize scales. Both models use the same 3815 training crops, initial weights, sample order and 30-epoch budget.

The third model predicts a displacement from each initial corner. Iterative Error Feedback repeatedly predicts and applies pose corrections [13]. The cassette model uses one correction step:

$$
\mathbf p_k^{\mathrm{new}}=\mathbf p_k^{\mathrm{initial}}+\Delta\mathbf p_k. \tag{6}
$$

The model receives the colour crop and initial-corner maps. A ResNet50 feature extractor and pooling layer predict horizontal and vertical offsets for both points. Dividing these offsets by the crop's resize scales converts them to original-image pixels before Equation (6) adds them to the starting coordinates.

Correction training uses 3809 crops with known reference corners and usable initial estimates. SmoothL1 loss gives quadratic penalties to small errors and linear penalties to larger errors [14]. This design changes the output layer, loss and eligible training points together. All three local models retain the detector's boxes and scores. Appendix 11.2 gives their training and decoding settings.

### 5.4 Common evaluation procedure

Evaluation uses input size 1280, object confidence strictly above 0.25, corner confidence at least 0.50 and matching IoU at least 0.50. NMS removes overlaps above 0.70 and retains at most 300 outputs per image. A second suppression pass uses the same threshold after clipping boxes to the image. A clipped box with zero area counts as an extra.

The supplied and proposed methods were run on all 20 photographs in the same environment using 32-bit tensors. Both views are filtered before association. The proposed method averages corners for 648 of 687 original-view detections. The local models use the same detector outputs and evaluation thresholds.

The main comparisons use the 654-front reviewed annotations. Sections 7.6–7.8 use the 658-front extension. The appendices report results under the earlier annotation versions and give the computing settings.

### 5.5 Controlled comparisons

Two paired training experiments use the same starting checkpoint, image order, seed, 30-epoch budget, AdamW settings and final-epoch selection within each pair. Each pair also shares its GPU, software and numerical precision settings.

The first changes classification loss on the same 142 photographs and 3999 fronts. Quality Focal Loss gives more relative weight to predictions farther from their quality targets [15]. The cassette implementation already derives targets from box and known-corner agreement. The candidate multiplies each binary cross-entropy term by the squared score–target difference. Assignment rules and other losses stay the same. This also reduces the total classification-loss scale, so the experiment tests both the weighting and its strength relative to the other objectives.

The second changes only the training annotations on the same 137 supplied photographs. Supplied and reviewed labels contain 3772 and 3784 fronts, with differences in 92 files. Both runs use the same 13 empty images and exclude the five supplementary photographs. This comparison measures the effect of the label revision during fine-tuning.

Both experiments report detection and corner outcomes at the common operating thresholds. The classification comparison also reports original-view AP50, which summarises precision across recall levels at matching IoU 0.50. It uses 101 recall levels, a preset score floor of 0.001 and at most 300 predictions per image. Appendix 11.2 records the computing settings according to PyTorch's reproducibility guidance [16].

Further diagnostics hold the trained models fixed. A replay without augmentation or parameter updates records assignments, predicted scores and supervision targets, the scores training asks the model to predict. Recalculating assignments and targets with box agreement alone tests the corner term's contribution. A separate inference trace follows all candidate boxes through confidence filtering and suppression to locate the remaining misses.

Output comparisons test three ways to remove extras: require a reflected partner, add suppression above IoU 0.60, or add suppression above 0.50. A separate test adds reflected detections without a mutual box partner, then applies the usual suppression above 0.70. Added detections retain their own corners and scores and can suppress an original detection.

The final rule uses corner agreement to guide removal. RMPE compares estimated human poses to suppress repeated detections [17]. The cassette rule removes a lower-scoring output only when box IoU exceeds 0.60, both outputs have two accepted corners, and each pair of corresponding corners is at most five pixels apart. It uses Equation (1) for box overlap and the distance calculation in Equation (3) between predictions. The thresholds were fixed before the comparison. Every output rule uses predictions alone and is evaluated by fresh matching to the annotations.

## 6 Simulation Results

A numerical check asks whether the heatmap grid can represent corners accurately enough for the five-pixel criterion. A 64 by 64 map on a 256 by 256 canvas spaces grid locations four canvas pixels apart. For every matched complete target, ideal Gaussian heatmaps centred on the reference corners pass through the same crop and decoder as learned outputs. The decoder selects the highest valid response, shifts it by a quarter cell towards larger neighbouring responses in each direction and converts it back using the actual rounded resize scales.

All 583 complete targets pass the five-pixel criterion, with a largest corner error of approximately 1.50 pixels. The grid and decoder therefore represent these known positions accurately enough. This check separates the resolution of the output representation from the learned localization results in Section 7.2.

## 7 Experimental Results

### 7.1 Comparison with the supplied model

Table 2 compares the supplied checkpoint and proposed method under the common settings. Duplicates are included in extras throughout the report.

Table 2. Recognition on the same 20 photographs and reviewed annotations. Pair counts use all 583 fronts with two known corners.

| Measure | Supplied model | Proposed method |
| --- | ---: | ---: |
| Matched fronts out of 654 | 620 | 636 |
| Missed fronts | 34 | 18 |
| Extra predictions | 61 | 51 |
| Duplicate predictions | 5 | 5 |
| Both corners within 5 pixels | 555/583 | 569/583 |
| Both corners within 10 pixels | 577/583 | 583/583 |

Precision increases from 91.04% to 92.58%, and recall from 94.80% to 97.25%. Five-pixel pair success rises from 95.20% to 97.60%.

The proposed method detects all 583 fronts with two known corners. Detection among the 71 fronts with fewer known corners rises from 40 to 53. All 18 remaining misses belong to this latter group, which complete-pair accuracy alone does not describe. The two earlier annotation versions favour the supplied model, as shown in Appendix 11.1.

### 7.2 Results of the local corner models

Table 3 includes the adapted detector before refinement, allowing each added stage to be judged against its input. All adapted methods retain 636 matched fronts, 18 misses, 51 extras and five duplicates.

Table 3. Complete-pair results for the supplied benchmark, adapted controls and local models. All rows use the reviewed reference and 583 eligible targets.

| Method | Both corners within 5 pixels | Both within 10 pixels |
| --- | ---: | ---: |
| Supplied model | 555/583 | 577/583 |
| Adapted detector without refinement | 566/583 | 582/583 |
| Proposed two-view method | 569/583 | 583/583 |
| Local image model | 547/583 | 582/583 |
| Local image and initial-corner model | 558/583 | 581/583 |
| Local coordinate-correction model | 564/583 | 582/583 |

None of the local models exceeds the two-view method. Coordinate correction reduces five-pixel successes from its input's 566 to 564, although the mean larger-of-two corner error improves from 2.32 to 2.27 pixels. Small improvements across many fronts can coexist with a few pairs crossing the failure threshold. The two-view method achieves 569 successes and a lower mean error of 2.16 pixels.

Some heatmap outputs select a neighbouring cassette feature, producing an error above 20 pixels in one examined case. Together with the successful grid check in Section 6, this points towards the learned localization and its training examples as the more useful refinement question.

### 7.3 Effect of classification loss

Table 4 compares Quality Focal Loss with its paired native-loss control.

Table 4. Paired classification-loss comparison. Detection and corner counts use two-view predictions at object confidence above 0.25. AP50 uses original-view predictions above the preset 0.001 floor.

| Measure | Native loss | Quality Focal Loss |
| --- | ---: | ---: |
| Missed fronts | 18 | 12 |
| Extra predictions | 49 | 83 |
| Duplicates included in extras | 4 | 17 |
| Both corners within 5 pixels | 568/583 | 570/583 |
| Both corners within 10 pixels | 583/583 | 582/583 |
| Original-view AP50 | 0.983865 | 0.983953 |

The candidate recovers six fronts with incomplete corner annotations and loses no previously matched front. It adds 34 extras, including 13 duplicates, and loses one ten-pixel success. AP50 changes by only 0.000088. The main effect is more detected fronts and more extras at the chosen confidence threshold.

In the fixed-model replay, all 3999 training fronts receive assignments with positive target scores in both models and both prediction branches. Assignment is therefore available to every corner-availability group in this replay.

The one-to-many branch can assign several candidates to a front. Recalculating its assignments and targets with box agreement alone lowers the mean target for fronts with one known corner from 0.626 to 0.466 in the control and from 0.644 to 0.477 in the focal-loss model. These averages cover assigned candidates in each recalculation. The corner term supports stronger targets for these fronts, favouring its retention while low inference scores are investigated.

### 7.4 Effect of training annotations

Table 5 holds the evaluation annotations fixed and changes the training labels.

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

Reviewed-label training recovers 33 fronts and loses one. It gains 37 five-pixel successes and loses six, giving a net increase of 31. Some recovered fronts already had a prediction from the control, but its box was too broad to reach matching IoU 0.50. Better box extent can therefore improve detection coverage.

Extras fall while duplicates rise. Twelve of the 13 duplicates concern incomplete reference fronts. Their overlap with the retained prediction does not exceed the suppression threshold, so removing them requires care around neighbouring partial fronts.

The earlier annotation versions favour supplied-label training. Appendix 11.3 gives those results. Training labels change what the model predicts, while the evaluation annotations determine how those predictions are scored.

### 7.5 Removing extra predictions

Table 6 tests the three output restrictions described in Section 5.5.

Table 6. Output restrictions on the same reviewed-label predictions. The reviewed reference contains 654 fronts, including 583 complete corner pairs.

| Final-output rule | Misses | Extras | Duplicates | Within 5 pixels | Within 10 pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged outputs | 15 | 51 | 13 | 572/583 | 583/583 |
| Require reflected partner | 19 | 27 | 6 | 572/583 | 583/583 |
| Additional suppression at 0.60 | 16 | 32 | 2 | 572/583 | 583/583 |
| Additional suppression at 0.50 | 19 | 23 | 1 | 572/583 | 583/583 |

Suppression at 0.60 removes 19 extras, including 11 duplicates, but loses one detected incomplete front. The other rules each lose four. Incomplete-front coverage falls from 56 of 71 to 55 or 52, although every complete-pair score stays the same.

The largest possible one-to-one match count equals the reported count for all five model output sets. Changing evaluation matching cannot recover more fronts from these saved boxes at the same IoU requirement. The boxes themselves limit the number of detectable reference fronts at this threshold.

### 7.6 Including identifiable partial fronts

Table 7 measures the effect of adding the four reviewed partial fronts to the evaluation annotations. Predictions and operating thresholds stay fixed.

Table 7. Reference-inclusion effect. Arrows show the 654-front reference followed by the 658-front extension. Each corner count has denominator 583.

| Method | Misses | Extras | Duplicates | Within 5 pixels | Within 10 pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supplied model | 34 → 37 | 61 → 60 | 5 | 555 | 577 |
| Proposed method | 18 → 21 | 51 → 50 | 5 | 569 | 583 |
| Paired native-loss control | 18 → 21 | 49 → 48 | 4 | 568 | 583 |
| Focal-loss candidate | 12 → 15 | 83 → 82 | 17 | 570 | 582 |
| Supplied-label training | 47 → 50 | 85 → 84 | 6 | 541 | 566 |
| Reviewed-label training | 15 → 18 | 51 → 50 | 13 | 572 | 583 |

Every method detects one added front and misses the other three. One former extra becomes a match, while three misses enter the count. Duplicate counts, complete-pair results and differences between methods stay the same. For the reviewed-label model, incomplete-front coverage becomes 57 of 75. These changes measure the expanded annotation set's effect on scoring.

### 7.7 Remaining misses and reflected detections

Candidate boxes are the detector's predictions before confidence filtering and suppression. Their reference overlap is measured after mapping and clipping them to the image. The reviewed-label model's 18 original-view misses arise at three stages. Twelve fronts have candidate boxes that meet the matching requirement but fall below the confidence cutoff. Five have no candidate box reaching IoU 0.50. One loses its eligible box during suppression. This makes confidence and box placement the main sources of remaining misses.

Table 8 tests adding reflected detections without an existing box partner. Each method finds more fronts while also producing more extras and duplicates. The three newly included missed fronts remain undetected.

Table 8. Adding reflected detections on the 658-front reference. Each arrow shows the unchanged method followed by the combined output. Duplicates are included in extras.

| Method | Misses | Extras | Duplicates |
| --- | ---: | ---: | ---: |
| Proposed method | 21 → 16 | 50 → 62 | 5 → 9 |
| Paired native-loss control | 21 → 15 | 48 → 60 | 4 → 9 |
| Focal-loss candidate | 15 → 11 | 82 → 115 | 17 → 33 |
| Supplied-label training | 50 → 48 | 84 → 93 | 6 → 8 |
| Reviewed-label training | 18 → 10 | 50 → 60 | 13 → 18 |

The reviewed-label model recovers eight fronts and loses none, increasing incomplete-front coverage from 57 to 65 of 75. Six recoveries have one known corner and two have neither. Of the six known corners returned, four are within five pixels and five within ten.

Five-pixel complete-pair counts stay unchanged for all models. Supplied-label training gains one ten-pixel success, while the other ten-pixel counts stay unchanged. The main method retains original-view detections because the added coverage comes with more extras and duplicates.

### 7.8 Using corner agreement in suppression

Table 9 tests whether corner agreement reduces repeated outputs while retaining nearby fronts.

Table 9. Additional suppression on the same reviewed-label outputs and 658-front reference. Duplicates are included in extras.

| Output rule | Misses | Extras | Duplicates |
| --- | ---: | ---: | ---: |
| Unchanged outputs | 18 | 50 | 13 |
| Suppression at 0.60 | 19 | 31 | 2 |
| Suppression at 0.60 with corner agreement | 18 | 44 | 11 |

The corner condition removes six extras, including two duplicates, and preserves every matched front and returned corner. Incomplete-front coverage stays at 57 of 75, with 572 of 583 complete pairs within five pixels and all 583 within ten. It also retains the partial front lost by box suppression alone.

The same rule increases misses from 50 to 51 for the supplied-label model, removing a useful prediction whose known corner was within two pixels. The outcome therefore depends on the model's predictions. For the reviewed-label model, it provides a small improvement using corners the system already predicts.

## 8 Discussions

### 8.1 Findings and interpretation

At the declared operating point, the proposed method improves detection and corner accuracy over the supplied model under the reviewed reference. Later comparisons reveal different trade-offs. Focal weighting finds more fronts but adds extras. Reviewed-label training improves several measures while adding duplicates.

The contribution is a cassette-front annotation procedure and a recognition method evaluated against the supplied model. The single-view control isolates averaging's contribution: five-pixel successes rise from 566 to 569, and ten-pixel successes from 582 to 583. The adaptation combines reviewed labels, additional photographs and further training. The paired experiments then test classification loss and training-label revision separately.

The annotation comparisons distinguish two effects. Rescoring measures agreement with a reference while leaving predictions fixed. Revising training labels changes what the model learns. Reporting both follows the motivation for annotation sensitivity analysis [4] and connects the reviewed target definition to the measured model behaviour.

The comparisons use 20 photographs that informed annotation and method development, with one paired seed in each controlled training experiment. Fronts within a photograph share lighting and viewpoint, so the 583 complete fronts are grouped within 20 images. The annotation version matters because it changes which model scores better.

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

All three references describe the same 20 photographs. Supplied labels contain 664 fronts, earlier fixed labels 666, and reviewed labels 654. Table 10 gives complete-pair counts.

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

If a heatmap yields no valid point inside the crop, that corner is recorded as unavailable. The correction model instead updates only usable initial points.

The correction model uses an NVIDIA L4 with the same optimizer, learning-rate schedule, batch size and epoch budget, and no hint noise. Its 7204 eligible endpoint targets produce 114270 crop presentations and 3600 updates. Eight known reference corners lack a usable initial estimate and are excluded. Nonfinite corrections and points outside the image become unavailable. Training uses PyTorch 2.14.1+cu130, and selected CPU replay uses PyTorch 2.9.0+cpu.

### 11.3 Training-label results under all references

Table 11 holds each evaluation reference fixed and compares supplied-label training with reviewed-label training. Both models use two-view averaging. All three rows rescore the same 20 photographs, providing three reference-sensitive comparisons.

Table 11. Training-label intervention under all three evaluation references. The first column gives the complete-pair denominator for both corner columns.

| Evaluation reference | Misses | Extra predictions | Within 5 pixels | Within 10 pixels |
| --- | ---: | ---: | ---: | ---: |
| Supplied, 584 pairs | 34 → 50 | 62 → 76 | 531 → 513 | 572 → 566 |
| Earlier fixed, 584 pairs | 23 → 38 | 49 → 62 | 544 → 523 | 584 → 575 |
| Reviewed, 583 pairs | 47 → 15 | 85 → 51 | 541 → 572 | 566 → 583 |

These results measure the frozen training-label revision during adaptation from a shared supplied checkpoint. Both arms use the same photographs and retain the checkpoint's common earlier training exposure.
