# Research basis for the cassette annotation definition

The [operating definition](../annotation-definition.md) consolidates the established visible-front policy. Its physical target comes from the supplied task and the photographs. Research helps make the annotation procedure explicit and testable; it does not determine which strip in a particular cassette image belongs to the rack.

## 1. Choose one observable target

CityPersons deliberately separates a visible-region box from a full-body, or *amodal*, box that represents hidden extent [1]. These are different annotation targets. Our cassette task retains the visible-front rectangle. Guessing the full hidden cassette outline would change its target and would require separately supported labels.

The practical rule is to box all visible material belonging to the front, including its frontal rim and shaded surface. The receding upper plane, tray and rack remain excluded. A rectangle can span an occluder between two visible pieces of one front; it is not a pixel mask. Occlusion and missing keypoints do not by themselves remove an identifiable object.

This remains `baomai-visible-front-v1`. The canonical [policy](../annotation-policy.md), prior accepted image decisions and frozen historical datasets are retained. This documentation pass performs no new image adjudication or label mutation.

## 2. Anchor box placement to the object's extrema

Extreme Clicking asks annotators to identify four extreme points of the visible object, then obtains its enclosing box [2]. This motivates reviewing the cassette's four visible extrema explicitly. It does not require a new editor or extra training targets: the existing rectangle can be checked one side at a time.

This separates two geometric tasks. The box encodes visible extent; the pose points encode particular lower-front junctions. The bottom of an axis-aligned envelope may be elsewhere than either physical corner. Neither a fixed inset nor snapping a landmark to a rectangle corner implements the task.

## 3. Separate object presence from point availability

COCO distinguishes an unlabeled point (`v=0`), a labeled but hidden point (`v=1`) and a labeled visible point (`v=2`) [3]. That convention shows why presence and visibility should be recorded explicitly, but it does not authorize guessing hidden cassette coordinates.

The supplied cassette format uses only observed `v=2` points and canonical missing `0 0 0` slots. An identifiable front with two missing corners therefore remains a detection target. The project parser accepts only flags 0 and 2. In the pinned Ultralytics 8.4.182 pose loss, both nonzero flags would enter the coordinate-loss mask; simply changing a flag to 1 would not give an inferred coordinate lower weight [4].

Accordingly, unknown corners stay missing. Context-derived estimates can be studied as explicitly marked proposals, as already described in the canonical policy, without turning them into observed reference points.

## 4. Measure ambiguity instead of selecting favorable labels

Noise-Aware Evaluation of Object Detectors studies how box-annotation discrepancies change evaluation, especially for small objects and strict overlap thresholds [5]. It supports retaining sensitivity analyses and separating physical annotation judgment from model agreement. Its alternative metric has not been substituted for this project's declared matching rule.

Our completed comparison illustrates why this matters: with unchanged predictions, the original model wins the five-pixel comparison under both older references, whereas the proposed method wins under the reviewed reference. Those score differences do not establish which reference is physically correct. The original photographs must resolve instance ownership and landmark identity.

The next annotation measurement is a small independent repeatability study. Select the calibration cases before examining new model errors; include clear and difficult cases. Hide existing labels and predictions during each initial annotation pass. Record instance and availability disagreements separately from side-coordinate and corner-coordinate differences. Independent human review has not yet been obtained for this protocol; a repeat pass by the same reviewer must be described as within-reviewer repeatability.

Do not adopt an arbitrary “correct within one pixel” rule. The five-pixel model criterion and the 0.0001-pixel serialization tolerance answer different questions; neither demonstrates human annotation precision. If reviewers disagree about the physical edge, averaging their coordinates does not resolve the disagreement.

## 5. Preserve unresolved cases and provenance

Datasheets for Datasets recommends recording target meaning, missing information, noise, splits and data processing [6]. For this project, the useful record is small: image identity/hash, policy, source-row identity, changed fields, reason, uncertainty category and reviewer status. Existing manifests and review ledgers already supply most of this information.

The current YOLO row and evaluator do not implement an object-level ignore region. For newly annotated data, unresolved photographs must be adjudicated or explicitly quarantined. They must not become empty negative scenes, and deleting only the uncertain object must not falsely imply exhaustive annotation. For a controlled incremental experiment, retaining the same inherited supervision in both arms on a held photograph preserves the experiment's declared scope; it does not certify that inherited supervision.

No new ignore implementation, segmentation target, hidden-corner target or automatic relabeling is introduced by this definition. These would each change the task or experiment and need explicit evidence.

## 6. What this contributes to the thesis

The usable contribution is an operational target definition with a reproducible correction record, followed by measured sensitivity and a controlled training-label comparison. The controlled experiment must hold photographs, initialization, architecture, training settings, training budget and evaluation references fixed while changing only eligible training labels. Report the number of changed images and rows, plus the held photographs.

Current evidence supports the annotation-reference sensitivity result. The new written definition, proposed repeatability study and future training-label run do not themselves establish an accuracy gain. Their status must remain separate from completed measurements.

## References

[1] S. Zhang, R. Benenson, and B. Schiele, “CityPersons: A diverse dataset for pedestrian detection,” in *Proc. IEEE Conf. Computer Vision and Pattern Recognition*, 2017, pp. 3213–3221. [Primary paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_CityPersons_A_Diverse_CVPR_2017_paper.pdf).

[2] D. P. Papadopoulos, J. R. R. Uijlings, F. Keller, and V. Ferrari, “Extreme clicking for efficient object annotation,” in *Proc. IEEE Int. Conf. Computer Vision*, 2017, pp. 4940–4949, doi: 10.1109/ICCV.2017.528. [Primary paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Papadopoulos_Extreme_Clicking_for_ICCV_2017_paper.pdf).

[3] COCO Consortium, “Data format: Keypoint detection,” accessed Nov. 28, 2026. [Official format specification](https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/format-data.htm).

[4] Ultralytics, “Pose loss implementation,” version 8.4.182, accessed Nov. 28, 2026. [Versioned official source](https://github.com/ultralytics/ultralytics/blob/v8.4.182/ultralytics/utils/loss.py).

[5] J. Murrugarra Llerena and C. R. Jung, “Noise-aware evaluation of object detectors,” in *Proc. IEEE/CVF Winter Conf. Applications of Computer Vision*, 2025, pp. 9304–9313. [Primary paper](https://openaccess.thecvf.com/content/WACV2025/html/Llerena_Noise-Aware_Evaluation_of_Object_Detectors_WACV_2025_paper.html).

[6] T. Gebru *et al.*, “Datasheets for datasets,” *Communications of the ACM*, vol. 64, no. 12, pp. 86–92, 2021. [Author manuscript](https://arxiv.org/html/1803.09010v8).

## Project evidence for maintainers

These links document implementation provenance and are not external references to paste into the submitted report:

- [Pinned upstream data definition](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/README.md): actual lower-front corners, independent missing slots and genuine empty scenes.
- [Canonical project policy](../annotation-policy.md) and [detailed labeling guide](../front-labeling-guide.md): visible-front extent, molded rim, shadow ownership, uncertainty and existing image decisions.
- [Completed annotation-reference comparison](annotation-comparison.md): three references applied to unchanged predictions.
- [Parser](../../scripts/evaluate_pose.py) and [geometry verifier](../../scripts/verify_pose_candidate.py): implemented label states and mechanical acceptance.
