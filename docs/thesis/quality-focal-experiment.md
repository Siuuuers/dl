# Training experiment: focus learning on mis-scored cassette proposals

28 November 2026, Asia/Hong_Kong.

**Completed:** PR157 returns both paired training arms and their evaluation. The [independent result analysis](quality-focal-results.md) reproduces all saved comparisons: reviewed-reference misses fall from 18 to 12, extras rise from 49 to 83, duplicates rise from 4 to 17, and five-pixel successes rise from 568 to 570 while one ten-pixel success is lost. Original-view AP50 changes by only about 0.000088. The trade-off does not establish an overall replacement. The [notebook](../../notebooks/baomai_quality_focal_r61.ipynb) and protocol below remain available for reproduction; the [preparation evidence](evidence/quality-focal/README.md) is historical. The [training-label result](training-label-results.md) is a separate controlled intervention.

## Question and choice

Can changing how the detector learns its object scores improve the balance between missed cassette fronts and extra predictions, while retaining useful corner coordinates?

The current two-view method has 18 misses, 51 extras and 569/583 complete pairs within five pixels under the reviewed reference at box confidence strictly above 0.25. The three completed corner refiners do not beat that corner result and cannot change detection counts when boxes remain fixed. Earlier partial-target classification weighting recovered some partial fronts but increased extra predictions. These observations motivate a new classification objective rather than another coordinate-only model. They do not predict whether it will succeed. See the [residual results](residual-results.md) and [partial-classification result](../partial-classification-r51-results.md).

The later [full-proposal replay](proposal-stages.md) locates 12 of the 18 current misses at confidence filtering, three at native NMS, and three without any decoded box reaching reference IoU 0.50. In the three NMS cases, a higher-score, lower-reference-overlap output suppresses an eligible alternative. These findings make score learning a relevant question, but do not establish a classification-loss cause or a guaranteed QFL benefit. The existing notebook, source pin, targets and training protocol remain fixed.

The source audit also prevents repeating existing ingredients. Horizontal flipping is already enabled during training with probability 0.5 and a left/right slot exchange. Native coordinate training already includes residual log-likelihood estimation. Soft suppression, a larger inference input, and synthetic covered-corner training have already been tested; none establishes an overall replacement for the retained method.

## Research basis

Li et al. introduce Quality Focal Loss (QFL) to train classification with continuous quality targets [1]. Its modulating factor reduces the relative contribution of predictions already close to their targets. The paper evaluates QFL separately from its distributional box-regression component and uses an exponent of two. That gives a small, identifiable change to test in the present detector.

The native Pose26 assigner already produces continuous target scores. We retain that target-construction rule and apply the QFL modulation to its classification binary cross-entropy. Actual target values and assignments may differ as the two models learn. We do not replace the targets with pure box overlap, add a quality head, or implement the complete Generalized Focal Loss detector from the paper.

For logit z, probability p = sigmoid(z), and detached native soft target y, the candidate classification term is

\[
L_{\mathrm{QFL}}(z,y)=|p-y|^2\operatorname{BCEWithLogits}(z,y).
\]

The control uses the original binary cross-entropy directly. Modulation remains differentiable with respect to z. Detaching it would produce a different gradient and a different experiment.

A background term with target zero and prediction 0.05 receives a factor of 0.0025; one with prediction 0.80 receives 0.64. Thus an already low-scored background example contributes much less relative to a confidently wrong one. Positive soft targets follow the same rule. This does not guarantee that every true-object score increases or every false-object score decreases after training.

**Interpretation:** retaining the existing classification gain and normalizer means the change also reduces the aggregate classification-loss magnitude. This two-arm experiment measures the complete objective change. It does not isolate error-dependent weighting from a uniform reduction in classification strength. Both loss numerators and their foreground/background partitions are recorded for that reason.

## Controlled comparison

| Setting | Native control | QFL candidate |
| --- | --- | --- |
| Training data | Same verified 142 photographs and 3,999 targets | Identical image and label bytes |
| Initial weights | Supplied epoch-173 checkpoint | Same checkpoint bytes |
| Epochs, seed, batch, input size | 30, 46, 4, 1280 | Identical |
| Optimizer and arithmetic | Existing AdamW recipe, FP32 | Identical |
| Augmentation and sample exposure | Existing recipe, each photo once per epoch | Identical |
| Classification | Native BCE; exponent zero bypasses modulation | Native BCE multiplied by squared probability–target error |
| Assignments, soft targets and normalization | Native | Native |
| Box, corner, availability and RLE losses | Native | Native |
| Selected checkpoint | Final epoch | Final epoch |

The same native one-to-many and one-to-one branches, including their changing training weights, remain active. Both arms have 1,080 batches, 4,260 image presentations and the established schedule of 80 optimizer updates. A separate resource trial must succeed before either full arm starts. Its weights cannot initialize a full arm.

These training annotations are the fixed, audited inputs used by the earlier research detector. Defining a clearer annotation policy does not silently alter this experiment's supervision. The controlled label comparison changes supervision separately.

## Evaluation fixed before training

Fresh common-runtime inference compares the supplied original model, the trained native control and the QFL candidate. Each receives original and reflected image views. The current report's two-view rule filters both pools at confidence strictly above 0.25 before suppression and coordinate averaging. Associations depend on predicted boxes, never on reference labels.

Keep the same 20 development photographs, matching overlap of at least 0.50, endpoint confidence of at least 0.50, and five-/ten-pixel corner criteria. Score all three annotation references: supplied, earlier fixed, and reviewed. Retain every eligible reference pair in the denominator, including missed detections.

Report missed fronts, extras, duplicate subsets, empty-scene outputs, and complete-pair success together. Preserve paired recovered/newly missed identities and corner gains/losses. A new matched control is essential; an old model trained in a different runtime is not a substitute.

QFL can change the score distribution. Therefore the fixed-confidence results are accompanied by AP50 from original-view outputs, using a prespecified confidence floor of 0.001 and maximum 300 detections per photograph. This is 101-point interpolated detection average precision at IoU 0.50 under the declared prediction limits, not COCO mAP. It is calculated separately for each annotation reference without selecting a new operating threshold from the development results.

Any apparent improvement must be assessed against both the native control and the supplied model, with the current two-view result retained as the practical corner comparator. Lower loss alone is not a recognition result. The reused photographs support development conclusions, not an independent capture-session claim.

## Implementation and completed checks

The [loss module](../../scripts/quality_focal_r61.py) wraps only the classification BCE in the two native loss branches. It does not copy or replace the native assignment body. Ordinary class weights and normalization execute in their original code paths. The training-only wrapper is installed after the native exponential-moving-average model is created, keeping saved inference checkpoints native.

The [CPU guard](../../scripts/check_quality_focal_r61.py) uses the actual supplied weights and two verified training photos, plus a synthetic blank image, at reduced resolution for differentiation checks. It makes no optimizer update. Its checks establish:

- Exponent zero is bitwise identical to native loss and all 498 parameter tensors receiving gradients in the full model.
- At fixed predictions, exponent two preserves assignments, soft targets, nonclassification losses, box/corner/sigma gradients and RLE parameter gradients in both branches.
- Classification gradients agree with an independently derived formula; double-precision numerical differentiation also passes.
- Optional class weights, empty targets and fronts with no available corners remain supported.
- Changing the coordinates of unavailable corner slots has no effect on supervision.
- The native 30-epoch branch-weight schedule is preserved.

The whole trained model may later change its boxes and corners because classification shares features with those tasks. The fixed-prediction checks prove the intended implementation boundary; they do not promise identical geometry after learning.

## Alternatives retained for later investigation

| Idea | Why it could help | Why it is not the first experiment |
| --- | --- | --- |
| Localization-quality prediction, inspired by IoU-Net [2] | Rank overlapping proposals using a learned measure of box quality. | Requires a new trained output, a defined quality target and reliable proposal supervision. |
| A second box-refinement stage, inspired by Cascade R-CNN [3] | Correct poorly delimited fronts and distinguish close false proposals. | Adds proposal sampling and another model; ambiguous unmatched regions must not automatically become negative training examples. |

Both are hypotheses for cassette images. Neither paper demonstrates an improvement on this dataset. The classification-only QFL comparison is smaller and makes direct use of the existing native soft targets.

## References

[1] X. Li et al., “Generalized Focal Loss: Learning qualified and distributed bounding boxes for dense object detection,” in *Advances in Neural Information Processing Systems*, vol. 33, 2020, pp. 21002–21012. [Official publication](https://proceedings.neurips.cc/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html) · [Paper, Eq. (2) and Tables 1/3](https://proceedings.neurips.cc/paper/2020/file/f0bda020d2470f2e74990a07a607ebd9-Paper.pdf).

[2] B. Jiang, R. Luo, J. Mao, T. Xiao, and Y. Jiang, “Acquisition of localization confidence for accurate object detection,” in *Proc. European Conf. Computer Vision (ECCV)*, 2018, pp. 784–799. [Official publication](https://openaccess.thecvf.com/content_ECCV_2018/html/Borui_Jiang_Acquisition_of_Localization_ECCV_2018_paper.html).

[3] Z. Cai and N. Vasconcelos, “Cascade R-CNN: Delving into high quality object detection,” in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 6154–6162. [Official publication](https://openaccess.thecvf.com/content_cvpr_2018/html/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.html).
