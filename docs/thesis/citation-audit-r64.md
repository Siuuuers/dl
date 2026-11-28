# Citation and explanation audit

28 November 2026.

## Requirement and scope

The report should explain its methods to Electronic Engineering readers who are not specialists in computer vision. IEEE citations such as `[6]` and `[7]` should accompany substantive explanations of the borrowed idea, its use in this project and its limits. A link alone is not an explanation. Published human-pose or general-detection results must not be presented as cassette results.

This audit covers [the report](report.md) and [its bibliography](bibliography-fyp-i.json). It preserves the completed empirical results, the distinction between development and independent evaluation, and the pending status of both GPU training comparisons. It does not validate document pagination or equation rendering.

## Citation style and source access

The requested [IJSSST-hosted IEEE Citation Style Guide](http://www.ijssst.info/info/IEEE-Citation-StyleGuide.pdf) could not be retrieved: the web fetch returned 502, a direct HTTP request returned 503, and an HTTPS variant was also inaccessible. The fallback was the current **IEEE Reference Guide, version 3.28.2025**, linked through the [IEEE Author Center](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-the-text-of-your-article/ieee-editorial-style-manual/). Its accessible [official public document](https://docs.google.com/document/d/e/2PACX-1vSBoV_hTIURPZKtQkxI6bAONJtjyDLxQIUvCHi-XPzD8NKLcxRfZrpvQLFrjWZ3M6FFR_Obau3R-t63/pub) was read directly.

The relevant rules are baseline square brackets inside punctuation, reuse of a source's number, one work per reference number, author initials before surnames, and verified publication details. The current guide writes separate citations such as `[3], [4]` rather than compressed reference ranges. Up to six authors are listed; longer author lists use the first author followed by *et al.* The thesis's required section numbering remains governed by its own requirements.

Publisher or author versions supplied all research material needed for this audit. Some publisher pages returned access errors, but author copies or indexed primary texts supplied the needed content. **No research-paper upload is needed.** The unavailable guide does not block the work because the current official IEEE guide is accessible.

## Source, application and limit

Numbers below follow the expanded report: ResNet [12], Iterative Error Feedback [13], SmoothL1 [14] and Generalized Focal Loss [15]. Links identify the primary sources used to check the claims.

| Reference | Supported idea and project application | Limit to preserve |
| --- | --- | --- |
| [1] [YOLO](https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html) | A single network predicts object boxes and class scores from a complete image; explains the detector's historical design family. | The 2016 architecture does not document the supplied YOLO26 checkpoint or its cassette accuracy. |
| [2] [YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/) | Documents detection and pose variants; explains why the project output can include landmarks. | Live defaults and recipes must not replace the recorded inference branch, suppression settings or training configuration. |
| [3] [ObjectLab](https://arxiv.org/abs/2309.00832) | Detector predictions can prioritize images containing possible missing, misplaced or wrongly classified annotations for review. | The project uses model-assisted human review; it does not claim to implement ObjectLab or to treat disagreement as proof of a bad label. |
| [4] [Noise-aware evaluation](https://openaccess.thecvf.com/content/WACV2025/papers/Llerena_Noise-Aware_Evaluation_of_Object_Detectors_WACV_2025_paper.pdf) | Boundary uncertainty can change detector evaluation, motivating reference-sensitivity reporting. | It does not determine which cassette annotation version is physically correct. |
| [5] [TIDE](https://arxiv.org/html/2008.08115v2) | Separates detection error types and measures their effects on average precision; motivates reporting more than one aggregate outcome. | The project's fixed-threshold counts are not TIDE's full error-oracle decomposition. Its geometric extra-prediction groups are review cues, not physical verdicts. |
| [6] [Simple Baselines](https://openaccess.thecvf.com/content_ECCV_2018/papers/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.pdf) | Uses a ResNet feature extractor, learned upsampling and landmark heatmaps. Its testing combines original and reflected heatmaps. | A heatmap is a spatial score grid, not automatically a calibrated probability map. The cassette architecture and coordinate averaging are adaptations. |
| [7] [PoseFix](https://arxiv.org/html/1812.03595v3) | Combines an image with Gaussian markers for the initial pose; trains with synthetic errors based on observed human-pose statistics. | The cassette model borrows input hints. It does not reproduce PoseFix's error synthesis, one-hot/integral objective or soft-argmax decoder. Arbitrary hint noise does not reproduce the published error distribution. |
| [8] [Confidence calibration](https://proceedings.mlr.press/v70/guo17a.html) | Explains that confidence and observed correctness can disagree. | No calibrator was fitted here; a corner score is not established as a probability of meeting the five-pixel rule. |
| [9] [AdamW](https://arxiv.org/abs/1711.05101) | Separates weight decay from adaptive gradient updates; explains the optimizer. | The source does not prove that the project's learning rate or optimizer is optimal. |
| [10] [Residual log-likelihood estimation](https://arxiv.org/html/2107.11291v3) | Learns an adjustment to an assumed coordinate-error distribution during training. | This retained native loss is distinct from a separate network's coordinate offset and does not establish calibrated output confidence. |
| [11] [Test-time aggregation](https://openaccess.thecvf.com/content/ICCV2021/html/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.html) | Classification experiments show that aggregation can improve totals while worsening individual cases; motivates paired outcome inspection. | The paper does not validate the cassette correspondence rule or fixed coordinate average. |
| [12] [ResNet](https://arxiv.org/html/1512.03385v1) | Shortcut additions help optimize deep feature networks; explains the local models' feature extractor. | Internal feature residuals and final coordinate residuals are different operations. |
| [13] [Iterative Error Feedback](https://arxiv.org/html/1507.06550v3) | Repeatedly predicts bounded corrections from an image and a rendering of the current pose. | The cassette model uses one learned correction step and does not reproduce the iterative procedure. |
| [14] [SmoothL1, PyTorch 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.SmoothL1Loss.html) | Uses a quadratic penalty below beta and a linear penalty above it. The implemented beta is one resized-crop pixel per x/y component. | This is a training penalty, not the Euclidean pair-success metric. Documentation defines the function; project code establishes the selected beta and masking. |
| [15] [Generalized Focal Loss](https://proceedings.neurips.cc/paper_files/paper/2020/file/f0bda020d2470f2e74990a07a607ebd9-Paper.pdf) | Its Quality Focal Loss component downweights score predictions already near their continuous targets. This motivates the pending loss-modulation comparison. | Native soft targets already exist and their construction rule is preserved; realized values and assignments can differ as the models learn. The experiment neither replaces them with pure IoU nor implements the full GFL detector. Modulation also changes total classification-loss scale; results remain pending. |

## Publisher pagination resolution

CVF and final publisher records use different pagination for several papers. The CVF ranges were valid for those copies. The report should consistently use the publisher DOI's pagination, while the bibliography JSON retains the CVF range and the reason for the difference. The publisher-deposited Crossref records below were retrieved directly; their titles, authors, venue, year, pages and DOI were checked. [Springer's Simple Baselines record](https://link.springer.com/chapter/10.1007/978-3-030-01231-1_29) also confirms its final pagination.

| Reference | Final publisher pages | CVF pages | Publisher DOI metadata |
| --- | --- | --- | --- |
| [4] Noise-aware evaluation | 9322–9331 | 9304–9313 | [10.1109/WACV61041.2025.00903](https://api.crossref.org/works/10.1109/WACV61041.2025.00903) |
| [6] Simple Baselines | 472–487 | 466–481 | [10.1007/978-3-030-01231-1_29](https://api.crossref.org/works/10.1007/978-3-030-01231-1_29) |
| [7] PoseFix | 7765–7773 | 7773–7781 | [10.1109/CVPR.2019.00796](https://api.crossref.org/works/10.1109/CVPR.2019.00796) |
| [10] Residual log-likelihood estimation | 11005–11014 | 11025–11034 | [10.1109/ICCV48922.2021.01084](https://api.crossref.org/works/10.1109/ICCV48922.2021.01084) |
| [11] Test-time aggregation | 1194–1203 | 1214–1223 | [10.1109/ICCV48922.2021.00125](https://api.crossref.org/works/10.1109/ICCV48922.2021.00125) |

ResNet's [publisher record](https://api.crossref.org/works/10.1109/CVPR.2016.90) confirms CVPR 2016, pp. 770–778 and DOI `10.1109/CVPR.2016.90`. GFL's [official NeurIPS BibTeX](https://proceedings.neurips.cc/paper_files/paper/11487-/bibtex) confirms volume 33, 2020, pp. 21002–21012 and eight authors, including Jinhui Tang. That record supplies no publisher DOI. ObjectLab remains a workshop paper with no invented proceedings page range. AdamW is cited as ICLR 2019, rather than using its 2017 preprint submission date as the conference year.

The SmoothL1 reference uses PyTorch 2.14 documentation to align with the recorded 2.14.1 training family; [2.9 replay documentation](https://docs.pytorch.org/docs/2.9/generated/torch.nn.SmoothL1Loss.html) gives the same elementwise definition. This alignment does not change the recorded runtime or implementation.

## COCO and expanded-report review

COCO is not needed as a general authority for the cassette annotation or pixel-success rules. The [2014 published dataset chapter](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48) supports natural-image dataset background, not this project's visible-front ontology, five-/ten-pixel criterion or later keypoint-visibility protocol. If used later, its publisher metadata lists eight authors and LNCS 8693, pp. 740–755; the updated arXiv author list differs and must not be merged silently with the publisher entry.

The expanded report now explains the source/application distinctions above, including image-plane pose, the three uses of IoU, reference versus emitted-output denominators, the one-step correction, and the pending controlled experiments. It describes QFL as giving **greater relative emphasis to predictions farther from their targets**: its multiplier need not increase an example's absolute contribution compared with native binary cross-entropy. A check after bibliography synchronization confirms first appearances in order 1–15 and exact agreement between all 15 report entries and the bibliography JSON. No additional paper or empirical result is required to support these explanations.

## Addendum after the paired training returns

The QFL and training-label experiments have now returned and are independently replayed in [their](quality-focal-results.md) [result analyses](training-label-results.md). The earlier pending-status descriptions above document the original citation audit. The report now adds [16], PyTorch Contributors, “Reproducibility,” version 2.14, accessed November 28, 2026, https://docs.pytorch.org/docs/2.14/notes/randomness.html. It supports the bounded statement that equal seeds do not guarantee identical results across computing platforms; it does not estimate a GPU effect for these models. All 16 report entries match bibliography metadata and first appear in order. Hardware names and software versions come from experiment receipts, and remain brief appendix conditions rather than performance findings.


## Completed diagnostic and duplicate research update

The current report has 18 external references. Its completed assignment diagnostic and saved-output suppression/consistency results replace the earlier future-work descriptions. References [17] and [18] explain conditional next tests, not implemented cassette gains. Soft-NMS decays proposal scores instead of immediate hard deletion; the current post-output hard-suppression audit is explicitly different. IoU-Net learns localization confidence for suppression and also introduces a separate refinement component. Its full published gain is not attributed solely to ranking, and box quality does not establish corner quality.

Soft-NMS metadata uses the DOI-linked final pages 5562–5570, corroborated by the [author institutional record](https://pure.johnshopkins.edu/en/publications/soft-nms-improving-object-detection-with-one-line-of-code/); the CVF accepted copy has 5561–5569. IoU-Net uses [Springer volume 11218, pages 816–832](https://link.springer.com/chapter/10.1007/978-3-030-01264-9_48); the CVF copy has 784–799. The bibliography retains these version distinctions. TIDE [5], Section 3.4, additionally supports inspecting missing or ambiguous references before equating unmatched detections with physical background.
