# Research note: local refinement of cassette-front corners

**28 November 2026 · Hong Kong**

**Current status:** both heatmap models and the residual-correction model have completed training and independent audit; see [heatmap results](local-refinement-results.md) and [residual results](residual-results.md). None exceeds the current two-view method on the reviewed five-pixel criterion. The [three-reference annotation experiment](annotation-comparison.md) is complete. The sections below retain the original literature and design rationale; prospective wording is historical, not a request to repeat completed training.

## 1. Recommendation

A useful next hypothesis is: **after the existing model finds a cassette front, can a second model use an enlarged view of that same front to locate its two lower corners more accurately?** Keep the detector, its rectangles, scores, suppression, and object identities fixed. Change only the proposed corner coordinates. This separates a localization question from the already measured tradeoff between recovering partial fronts and producing additional unmatched outputs.

The [completed training comparison](../partial-classification-r51-results.md) supports this focus. Both arms of the partial-target training comparison detected all 583 fronts with two locatable corners under the reviewed reference. Nevertheless, the unchanged control had 566 five-pixel pair successes and the weighted candidate had 567. In the separate automatic-output audit, increasing the candidate's endpoint cutoff retained all 16 known five-pixel localization failures. A coordinate refiner could address an error that simple confidence screening did not remove. It would not recover fronts for which the frozen detector supplies no box.

The crop-containment check is complete: all matched two-corner targets fit inside the fixed crop (Section 5). The [reflection comparison](reflection-results.md) and [fixed-box coordinate average](corner-average-experiment.md) are now complete. The next step is to run the prepared image-only and image-plus-hints refinement comparisons, retaining the remaining localization failures for paired analysis. The detector remains fixed, so unmatched and duplicate boxes are unchanged; the [automatic-output quality investigation](research-corner-quality.md) addresses the complementary question of which pairs to offer.

## 2. Three primary papers and their relevance

### 2.1 PoseFix: refine an imperfect initial estimate

G. Moon, J. Y. Chang, and K. M. Lee, “PoseFix: Model-Agnostic General Human Pose Refinement Network,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 7773–7781.

[CVF publication](https://openaccess.thecvf.com/content_CVPR_2019/html/Moon_PoseFix_Model-Agnostic_General_Human_Pose_Refinement_Network_CVPR_2019_paper.html) · [Published PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Moon_PoseFix_Model-Agnostic_General_Human_Pose_Refinement_Network_CVPR_2019_paper.pdf) · [Author manuscript](https://arxiv.org/abs/1812.03595)

The method takes an image and an imperfect pose, then predicts a refined pose. It represents the input points as broad Gaussian maps aligned with the image, and trains with synthetic pose errors based on human-pose error statistics. Its separate refinement stage motivates preserving the cassette detector and adding a focused corner stage. The proposed adaptation transfers the image-conditioned correction idea. Its human-joint output layer and error frequencies would need to be replaced for the two cassette corners.

### 2.2 HRNet: preserve spatial detail inside the localizer

K. Sun, B. Xiao, D. Liu, and J. Wang, “Deep High-Resolution Representation Learning for Human Pose Estimation,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 5693–5703.

[CVF publication](https://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html) · [Published PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.pdf) · [Author manuscript](https://arxiv.org/abs/1902.09212)

HRNet maintains a high-resolution feature stream and repeatedly exchanges information with lower-resolution streams. Its human-pose pipeline uses object crops and produces landmark heatmaps. This motivates a two-corner crop model with a fine spatial representation, rather than another whole-image size increase. A crop can give the target more representation capacity while keeping the detection result fixed. Resizing does not create detail absent from the camera photograph.

### 2.3 DARK: avoid unnecessary heatmap-coordinate rounding

F. Zhang, X. Zhu, H. Dai, M. Ye, and C. Zhu, “Distribution-Aware Coordinate Representation for Human Pose Estimation,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 7093–7102, doi: 10.1109/CVPR42600.2020.00712.

[CVF publication](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.html) · [Published PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.pdf) · [Author manuscript](https://arxiv.org/abs/1910.06278)

DARK studies errors introduced when coordinates are encoded as heatmaps and then decoded. It uses non-rounded target centres and distribution-aware subpixel decoding. The paper explicitly describes integration with heatmap-based pose models. **It is not a direct decoding patch for the present native Pose26 coordinate-regression head.** It becomes relevant if a new heatmap refiner is introduced. Use one fixed, verified decoder for the initial pilot; DARK can then be a separate decoder comparison.

## 3. Compatibility with the present model and annotation contract

The model configuration names `Pose26`; the native training comparison explicitly preserves the existing landmark, visibility, and residual log-likelihood estimation (RLE) losses. RLE is already present and is not a new proposed contribution; see the [native-loss invariants](../partial-classification-r51.md).

The target is one identifiable cassette front and its two actual lower-front corners, in image-X order. An unavailable corner is exactly `0 0 0`; that sentinel is not a coordinate target. The [annotation policy](../annotation-policy.md) permits an expanded input crop while preserving the original target geometry. Expanding a crop must not enlarge or redefine the annotated front box.

A joint two-corner crop is preferable to unrelated left and right patches for the first experiment. It retains the connecting lower edge and can help the network keep both points attached to the same front. Surrounding rack and neighbouring fronts can help distinguish a lower front edge from an upper lip or the next cassette, but a larger crop also introduces competing corners. The existing point proposals can serve as approximate location hints; they must not be treated as correct labels.

## 4. Smallest controlled notebook design

The first question should be whether a single refiner helps the existing system, not whether every component of a new architecture is optimal.

| Component | Fixed proposal for the first pilot |
| --- | --- |
| Detection | Use the same pinned unchanged detector and original-view prediction recipe as the completed reflection control. Preserve boxes, scores, suppression, proposal IDs, and maximum detections. |
| Crop | Expand each predicted box to 1.5 times its width and height about its centre; floor the left/top, ceil the right/bottom, then clip to the original image. Preserve aspect ratio when placing the resulting crop into a 256 × 256 input canvas. Record the actual integer crop and affine transform exactly. |
| Target identity | Process one detector record at a time. Give the refiner the RGB crop plus two image-aligned maps marking the initial left and right proposals. Do not select a neighbouring detection by comparing it with the reference. |
| Refiner | One pretrained image backbone with a new two-output landmark head. A small HRNet configuration is a defensible implementation option; load only compatible pretrained parameters and record what was reinitialized. Treat the result as a project-specific adaptation, not a PoseFix reproduction. |
| Supervision | Use only training photographs and their accepted labels. Coordinate/heatmap losses apply only to available corners, independently for each slot. Do not supervise unavailable corners at zero or invent their positions. |
| Initial estimates for training | Obtain crops and point proposals from the frozen detector on training photographs. Add a single prespecified, seeded point perturbation during training to reduce dependence on unrealistically accurate in-sample proposals. Record this as a distribution assumption; do not fit it to the 20 development photographs. |
| Training budget | One declared seed, final epoch, optimizer, initialization and schedule, all locked before scoring development predictions. Record actual optimizer updates and the number of usable crops; photo epochs and crop epochs do not imply equal training effort. |
| Evaluation | Run the refiner on every retained detector record, including unmatched outputs and partial fronts. Transform its coordinates back into the original photograph and score all eligible targets under the existing matcher and reference versions. |
| Main comparison | Unchanged detector coordinates versus refined coordinates for the same proposals. Do not change the score cutoff, box geometry, labels, or matching rule at the same time. |

The first pilot tests the complete add-on. If it helps, later matched comparisons can isolate crop resolution, initial-point hints, or the decoder.

The original endpoint scores concern the original head. For the controlled coordinate comparison, preserve the original score-selected candidate IDs and count any invalid refined result as a failure on that fixed cohort. Also report how many refined pairs would actually be offered after coordinate-validity checks. Do not present the old scores as calibrated confidence in the new coordinates. Save the old and new coordinates separately. Record invalid refined points explicitly rather than clipping them into a successful-looking location or silently falling back to whichever result scores better.

## 5. Completed crop-containment diagnostic

The [completed quality analysis](research-corner-quality.md) checked existing original-view predictions with one fixed 1.5× expansion, image clipping, and the existing box assignment. It counted corner containment for every matched two-corner target, keeping missed targets, unavailable pairs and image-edge cases separate. The [saved summary](evidence/corner-quality/crop-containment-summary.json) preserves the parameters, counts and per-image failures.

Containment establishes that the labelled location lies in the crop. It does not establish visual clarity, correct annotation or achievable refinement accuracy. Reference coordinates are used to measure containment, never to choose the crop or determine when the application refines a prediction.

The check uses box confidence strictly above 0.13, corner confidence at or above 0.50, and matching IoU at least 0.50. Crops use integer floor/ceil boundaries and image clipping; point containment is half-open, consistent with the image coordinate contract:

| Reference and detector arm | Matched two-corner targets | Both corners in fixed crop | Current five-pixel failures, all contained | Unmatched two-corner targets |
| --- | ---: | ---: | ---: | ---: |
| Fixed 666-front reference; unchanged control | 577 | 577 | 66 | 7 |
| Fixed 666-front reference; weighted candidate | 577 | 577 | 66 | 7 |
| Reviewed 654-front reference; unchanged control | 583 | 583 | 17 | 0 |
| Reviewed 654-front reference; weighted candidate | 583 | 583 | 16 | 0 |

Each arm/reference combination has 34 matched two-corner crops clipped at an image edge. These include 11 of the fixed-reference five-pixel failures and five of the reviewed-reference failures. Every matched target retains both labelled locations within its crop. The seven unmatched fixed-reference targets still have no crop and cannot enter a frozen-box refiner. These results were computed on the CPU from saved predictions.

## 6. Failure risks that change the design

- **Training/development leakage:** split by photograph or acquisition group, never by individual cassette crop. Several neighbouring crops share nearly all their pixels. Development errors must not determine augmentation noise, crop margin, epoch selection, or a case-specific correction. Predictions from a detector trained on the same training photographs may be too accurate; held-out-group predictions are a stronger later way to model realistic detector errors. Refiner-only cross-validation with a detector that has already seen those groups must not be described as a completely unseen pipeline evaluation.
- **Partial visibility:** mask each unavailable point's coordinate loss. A confident inferred hidden corner remains an estimate. Keep the original front proposal and any individually useful point even if a complete pair is withheld. Keep inferred hidden coordinates distinct from observed reference points.
- **Crop boundaries:** a predicted box can be misplaced; the expanded crop can still cut off a corner or necessary context. Padding introduces artificial edges. Mask padding during heatmap decoding, store exact resize conventions, and verify forward/inverse transforms, including clipped image edges. Do not use an approximate inverse after rounded crop slicing.
- **Instance confusion:** a wide crop may include several near-parallel lower edges. Joint point output and approximate input-point maps help define the intended target but do not guarantee it. Inspect paired visual failures for a left point from one cassette and a right point from another. Never reorder the zero sentinel into a real slot.
- **Reference uncertainty:** the unchanged control's five-pixel pair counts differ substantially between annotation versions. Review the original pixels when references disagree. A small threshold crossing may reflect annotation placement as much as improved physical localization; report continuous errors and paired gains/losses alongside threshold counts.
- **Automatic-output quality:** frozen boxes make detection counts invariant; they do not make false proposals useful. Report offered pairs, supported pairs, localization failures on complete references, partial-reference matches, and unmatched pairs. Do not report only a lower mean error on successfully matched corners.

## 7. What would count as useful evidence?

Require increased both-corner success under the declared fixed reference, a compatible result under the reviewed reference, and a paired account of gained and lost targets. Preserve the denominator of all eligible two-corner fronts, including misses and withheld pairs. Also report median and upper-tail original-pixel error on a clearly named common cohort, successful pair coverage, and the supported fraction of offered outputs.

Because the detector is frozen, any changes in full-record box misses, unmatched counts, duplicates, or proposal IDs would indicate a pipeline or scoring error. Pair-offering counts can still change when refined coordinates fail validity checks. The paired gain/loss counts must show whether corrected points come at the cost of newly wrong pairs.

The existing 20 photographs have informed development repeatedly. Their results can guide whether to retain the refiner as a candidate. A later capture session would test whether the improvement survives changed lighting, arrangement and camera conditions. Pixel tolerances remain image diagnostics until a downstream physical tolerance and calibration are supplied.

## 8. Plain-language wording suitable for the report

“A second-stage corner model is a possible extension of the system. The existing detector would first identify a cassette front. A larger local view of that front would then be supplied to a model trained to locate its two lower corners. The detector's rectangles would remain fixed, allowing changes in corner accuracy to be measured separately from changes in object detection. The local view must retain enough context to distinguish the intended lower edge from adjacent cassettes. The fixed crop check is complete; the effect on prediction accuracy still needs to be measured.”

## 9. Evidence and reproducibility

- [Project report](report.md): experiment definitions, model comparisons and original-image error measures.
- [Corner-quality investigation](research-corner-quality.md): confidence and prediction-agreement diagnostics, plus the completed crop check.
- [Crop-containment summary](evidence/corner-quality/crop-containment-summary.json): fixed parameters, model/reference counts, per-image failures and source hashes.
- [Annotation policy](../annotation-policy.md): target material, independent endpoint availability, coordinate conventions and expanded context crops.
- [Model card](../../metadata/model_card.json) and [architecture](../../models/cassette_lower_yolo26s_pose.yaml): model identity and two-corner configuration.
- [Partial-front training protocol](../partial-classification-r51.md) and [verified results](../partial-classification-r51-results.md): data membership, initialization and native-loss invariants.

The three papers study human pose. The cassette adaptation and its experimental controls are proposed design choices; the measured cassette result in this note is the completed crop-containment analysis.
