# Research directions after R53

27 November 2026. Research note updated after the completed PR149/PR150 audit and cached fusion replay. The new R56 reflection notebook is prepared; no R56 model result is claimed.

## Recommendation

Prepare one explicit horizontal-flip inference notebook, with the current 1280 original-image control. The existing R53 prediction caches have now been replayed: the fixed two-view combinations recover some targets but increase unmatched and duplicate outputs and do not improve complete-pair success. See [the measured fusion results](evidence/cached-fusion-results.md). The two new R53 treatments fail the declared conservative screen: neither supplies evidence for adopting contrast enhancement or increasing whole-image scale. The history also contains a negative 1920 probe, partial-image sampling/loss trials, synthetic-occlusion training with clean-image regressions, and context geometry that did not improve direct corner predictions.

The directions below separate one prepared model experiment, one completed negative cache replay, and later research hypotheses. Paper results are not demonstrated cassette improvements.

## 1 Horizontal flip inference

**Hypothesis.** The same checkpoint may produce complementary detections or landmark placements when the input is horizontally reflected and predictions are correctly returned to the original coordinate system.

**Primary motivation.** Xiao et al. use original/flipped heatmap aggregation in a human-pose estimator [1]. This establishes a pose-specific precedent for flip testing, not a guarantee for our two-landmark regression head. Huang et al. show that coordinate-processing biases can misalign flipped pose predictions [2]. Shanmugam et al. show that aggregate improvement from test-time augmentation can coexist with newly corrupted cases [3]; that evidence is for classification, so we use it to motivate paired case reporting, not to claim a pose result.

**Smallest experiment.** Keep the verified PR137/reference checkpoint, all 20 originals, four references, original 1280 inference size, box confidence 0.13, ordinary NMS 0.70, postclip rule, endpoint cutoff 0.50, and FP32 environment. Reproduce the original baseline first. Infer an explicit horizontally reflected image with ordinary prediction. Return its full box and two-point record to original coordinates and exchange the two point slots. Report the original and reflected views separately, with the full paired case ledger and complementarity. The prepared notebook performs no fusion. Any later combined rule needs its own fixed protocol and must retain the meaning of a complete pose record.

**Decision.** Require the existing primary screen and case-level 5/10-pixel gains/losses. After the returned archive passes verification, apply the same R55 offered-pair definitions as an additional CPU analysis. No new cutoff search or per-target ground-truth choice is allowed. If the reflected view has no useful complementary successes, stop rather than adding an ensemble grid.

**Implementation trap.** The pinned Ultralytics 8.4.182 `DetectionModel._predict_augment` falls back to ordinary single-scale prediction when `end2end` is true or the head is not exactly `Detect`. A pose call with `augment=True` is therefore not evidence that any augmentation occurred. Use explicit image processing and record distinct input hashes. The installed source was checked directly.

For our native normalized-by-width coordinate convention, `Instances.fliplr` maps xyxy boxes to `(W-x2, y1, W-x1, y2)` and keypoint x to `W-x`; `RandomFlip` then applies `flip_idx=[1,0]`. Match that convention and test the inverse twice, including edge coordinates. Do not replace it casually with `W-1-x` from discrete array-index reasoning. This is a convention-sensitive transformation, not a claim about subpixel physical truth. Preserve scores, invalid coordinates, and original bounds checks consistently.

## 2 Completed cached-view complementarity and fusion

**Hypothesis.** A small pair of existing views or models might combine useful detections that no single operating point retains.

**Primary motivation.** Solovyev et al. propose confidence-weighted averaging of corresponding boxes across detectors [4]. It is box evidence. The paper neither establishes the correspondence of neighboring cassette fronts nor defines how to fuse our two physical landmarks.

**Smallest experiment.** First replay existing 1280, CLAHE, and 1600 outputs and count exact reference-target sets: successes shared, gained, and lost for detection and corner pairs. A reference-assisted union is an oracle headroom diagnostic, not executable runtime performance; unmatched outputs and ambiguity remain visible. If the union adds little headroom, do not build fusion. If it adds meaningful headroom, test one declared two-view rule with fixed association and suppression, preferably selecting a complete pose record before considering coordinate averaging.

**Decision.** Report whether any useful gain survives a practical fixed rule while controlling duplicates, unmatched outputs, all-eligible corner success, and empty scenes. A gain under oracle per-case selection alone is insufficient. Scores from distinct checkpoints are not automatically comparable, and correlated predictions must not be multiplied as independent evidence.

**Pitfalls.** Thin neighboring boxes can overlap strongly; merging them can erase distinct cassettes. A mean corner can land on neither correct physical junction. One model may have learned a different target extent. Keep training/reference policy provenance explicit.

### Measured result from the available R53 caches

The fixed ordinary-NMS combinations are now complete and independently checked. On Frozen666, unchanged plus CLAHE changes misses/unmatched/duplicates from 34/91/17 to 33/111/25 and pair-five-pixel success from 511 to 510. Unchanged plus scale gives 33/135/34 and 506 pair successes. R50 recovers more fronts, but its unmatched and duplicate counts also increase, with pair success falling from 566 to 565 and 557. The [portable analysis](evidence/cached-fusion-results.md) preserves all four references and exact gained/lost identities.

This was complete-record selection through fixed suppression, not weighted boxes fusion. A reference-aware selection among all three separate views could count 534 primary pair successes, but its selector requires ground truth. That diagnostic does not establish an implementable improvement. We therefore proceed with a distinct single reflection comparison and defer more complex aggregation.

## 3 Consistency-guided training-side annotation review

**Hypothesis.** Disagreement between original and transformed predictions identifies useful training-side review cases more efficiently than the existing geometry queue or random review.

**Primary motivation.** Yu et al. use consistency of original and augmented object predictions to guide active sample selection [5]. Their method combines classification and localization information and tests Faster R-CNN on general box-detection datasets. ObjectLab supplies a complementary precedent for annotation-error review with out-of-sample predictions [6]. Neither paper authorizes automatically relabeling disagreement cases.

**Smallest experiment.** Freeze a fully defined pool of training-eligible photographs and a unique-case budget. Obtain predictions from a model that did not train on the reviewed image, grouping related captures when provenance permits. Rank the same pool by one declared box/landmark consistency score; compare with seeded random order and the frozen geometry queue. Adjudicate original images, include full-frame and unflagged checks, and record confirmed issues, unresolved cases, and actual review time. Only a meaningful accepted correction set justifies subsequent paired label-only training.

**Decision.** No superior issue yield means the ranking hypothesis fails. No recorded review time means no time-saving claim. Better review yield and better model performance remain separate claims. This adapts the papers' idea; it is not a reproduction of CALD or ObjectLab and needs no claim of a novel active-learning algorithm.

## 4 Pose-aware tiling as a conditional later probe

**Hypothesis.** A limited overlapping crop arrangement may help genuinely tiny fronts by changing effective scale and local context differently from whole-image interpolation.

**Primary motivation.** SAHI reports benefits of slicing and merging for aerial box detection [7]. It does not validate two-corner pose outputs, our cassette scenes, or arbitrary tile settings.

**Smallest experiment.** Only proceed after identifying residual cases whose geometry supports a scale/context mechanism. Fix one tile arrangement and overlap before running. Retain the full-image branch, map all boxes and keypoints back through each crop's transform, and specify ownership/truncation rules and cross-tile merging. Score on full original images, including all unmatched outputs. Separate inference tiling from any later tiled training.

**Decision.** Require more supported recoveries without worse protected outcomes. Newly truncated fronts, increased duplicates, neighboring-front merges, or lost corner accuracy weaken the claim. Given the unfavorable whole-image 1600 return and earlier 1920 study, tiling is lower priority than the explicit flip probe.

## What to defer

A larger compatible pose model is a possible later capacity comparison, but current evidence does not identify capacity as the cause. Retain learned initialization; a scratch run changes optimization and pretraining as well as cost. Segmentation needs new masks and a landmark extraction definition, so it is not a minimal response to the current results. Do not re-run generic Soft-NMS, partial oversampling, or pixel-occlusion augmentation without a distinct supported change to their previously tested hypotheses.

## Verified primary references in IEEE form

[1] B. Xiao, H. Wu, and Y. Wei, “Simple baselines for human pose estimation and tracking,” in Proc. European Conf. Computer Vision (ECCV), 2018, pp. 466–481. [Online]. Available: https://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html

The primary PDF's Testing paragraph states that joint locations come from averaged original/flipped heatmaps. Official implementation: https://github.com/microsoft/human-pose-estimation.pytorch

[2] J. Huang, Z. Zhu, F. Guo, and G. Huang, “The devil is in the details: Delving into unbiased data processing for human pose estimation,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2020, pp. 5700–5709. [Online]. Available: https://openaccess.thecvf.com/content_CVPR_2020/html/Huang_The_Devil_Is_in_the_Details_Delving_Into_Unbiased_Data_CVPR_2020_paper.html

[3] D. Shanmugam, D. Blalock, G. Balakrishnan, and J. Guttag, “Better aggregation in test-time augmentation,” in Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV), 2021, pp. 1214–1223. [Online]. Available: https://openaccess.thecvf.com/content/ICCV2021/html/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.html

[4] R. Solovyev, W. Wang, and T. Gabruseva, “Weighted boxes fusion: Ensembling boxes from different object detection models,” Image and Vision Computing, vol. 107, Art. no. 104117, 2021, doi: 10.1016/j.imavis.2021.104117. [Online]. Available: https://arxiv.org/abs/1910.13302

[5] W. Yu, S. Zhu, T. Yang, and C. Chen, “Consistency-based active learning for object detection,” in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW), 2022, pp. 3951–3960. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/html/Yu_Consistency-Based_Active_Learning_for_Object_Detection_CVPRW_2022_paper.html

Page range follows the linked CVF open-access record; other indexing services can paginate differently.

[6] U. Tkachenko, A. Thyagarajan, and J. Mueller, “ObjectLab: Automated diagnosis of mislabeled images in object detection data,” in ICML Workshop on Data-centric Machine Learning Research, 2023, arXiv:2309.00832. [Online]. Available: https://arxiv.org/abs/2309.00832

[7] F. C. Akyon, S. O. Altinuc, and A. Temizel, “Slicing aided hyper inference and fine-tuning for small object detection,” in Proc. IEEE Int. Conf. Image Processing (ICIP), 2022, pp. 966–970, doi: 10.1109/ICIP46576.2022.9897990. [Online]. Available: https://arxiv.org/abs/2202.06934

Software reference: Ultralytics, “Reference for ultralytics/nn/tasks.py,” accessed Nov. 27, 2026. https://docs.ultralytics.com/reference/nn/tasks/

Pinned 8.4.182 source identities from the available project runtime:

- nn/tasks.py SHA-256: 1048ab8db7504e7c57a77d05bfbd855e9de0ef477e15a5aae8cdb7e8a958246a
- utils/instance.py SHA-256: ebc8b6337f4636c1457473c079d007693b3d8766c4891e2b5dd521c22c2e5527
- data/augment.py SHA-256: 802333ea359def55e88f61d6726e35c29d28a6491110840a5e912ec1fafd1de3

All these are source checks, not new model predictions. The current official API documentation independently shows the augment fallback, but the notebook should bind its exact installed runtime rather than the moving documentation page.
