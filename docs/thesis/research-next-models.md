# Which model change is justified next?

**Recommendation:** use the completed [classification-loss](quality-focal-results.md) and [training-label](training-label-results.md) comparisons to focus the next investigation. QFL recovers six fronts but adds 34 extras; its AP50 barely changes. Reviewed-label training improves reviewed-reference counts while increasing duplicates and reversing rank under older references. The completed [frozen assignment audit](score-assignment-audit.md) finds no missing assignment among 3999 training fronts and no obvious benefit from removing the corner-quality term. The [duplicate-output audit](duplicate-audit-r66.md) quantifies how stricter filtering removes extras while losing partial-front matches. Review uncertain instance identity and extent before another training loss. These results do not select a new detector, suppression default or Wasserstein box loss.
One small localization-loss comparison is specified below as a conditional next
experiment, not an implemented or successful method.

The later [score and assignment addendum](research-score-assignment.md) checks
TOOD, IoU-Net and VarifocalNet against the exact pinned pose implementation. It
shows why assignment, normalized training targets and predicted scores must be
examined separately before another loss change. These additions refine the
interpretation of the completed experiments. The focused [score-target diagnostic](score-assignment-audit.md) requires no training; no new GPU training comparison is added.

This note supplements the [experiment index](experiments.md),
[proposal-stage analysis](proposal-stages.md), [detection-error analysis](detection-errors.md)
and [annotation definition](research-annotation-definition.md). The consumer is
intended to use the vision outputs for robotic arms; arm control, calibration
and motion planning are outside this research task. Recognition evidence must
therefore be stated without treating a high model score as verified physical
geometry.

## 1. What the current box geometry actually shows

The [reproducible geometry analysis](evidence/box-geometry/README.md) uses saved
predictions, without new inference, changes to labels or threshold selection.
It exactly reconstructs the canonical outputs and 120 per-image scoring tables
for the supplied model and retained method on all three references. For each
unmatched reference, the descriptive candidate is the retained same-class box
with greatest IoU. This association is separate from official one-to-one
matching; it does not prove that both boxes describe the same physical front.

Under the reviewed reference, all 18 misses of the retained method have fewer
than two locatable corners. Their median width is 35.5 px, compared with 81 px
for the 53 matched incomplete fronts. Their median square-root area is 28.85 px,
compared with 42.79 px for those matched incomplete fronts. These observations
associate missed status with smaller visible extents in this sample. They do
not establish that size, rather than occlusion, appearance, ownership or training
exposure, caused the misses. [Evidence](evidence/box-geometry/results.md).

Ten missed references have a retained candidate with maximum IoU in [0.30, 0.50).
Eight of those candidates are more than 1.5 times the reference width. One
different case has height 2.49 times the reference height. Several have one
horizontal edge close to the reference and the other displaced by 20–36 px.
These are appreciable extent disagreements, not merely one-pixel threshold
crossings. All ten candidates are themselves unmatched in the official score.
The remaining case, a 21 x 16 px reference at IoU 0.4906, has more modest
displacement and extent differences. The continuous edge errors and all ten
identities remain in [cases.json](evidence/box-geometry/cases.json).

The pattern depends on the annotation definition. On the untouched supplied
reference, 16 of 29 near-overlap misses have a candidate height above 1.5 times
the reference; only one has width above that ratio. On the reviewed reference,
the corresponding counts are one and eight out of ten. The predictions did not
change. A localization hypothesis must therefore say which visible boundaries
it intends to learn and preserve this sensitivity analysis.

## 2. What the Wasserstein papers support

Wang et al. [1] encode an axis-aligned box as a Gaussian and define normalized
Wasserstein distance (NWD). Their Faster R-CNN/AI-TOD experiments study assignment,
suppression and regression separately. Table 1 reports 12.1 AP with the NWD
regression loss versus 10.8 with IoU; the larger single-module gains in Table 2
come from assignment. Table 3 reports 17.8 AP when replacing three RPN components,
but 15.2 when replacing all six RPN/R-CNN components. The headline gain is not
evidence for a loss-only change in YOLO pose.

The expanded journal study by Xu et al. [2] combines NWD with ranking-based
assignment. Its subject is insufficient positive training assignment for tiny
aerial objects. AI-TOD-v2 has mean square-root box area 12.7 px, and about 86% of
objects are below 16 px by that size measure. Table 7 separates NWD and ranking
assignment: AP is 12.8 for the baseline, 20.2 with NWD, 17.9 with ranking and
21.4 with both. Those are assignment results for that benchmark; they are not
cassette pose results.

NWD is already cited in the [earlier thesis plan](history/plan-r48.md). This note
reassesses its fit to the current errors; it does not present the paper as a
newly discovered idea. No NWD model result exists in this evidence set.

### The useful geometric interpretation

For centers c, widths w and heights h, the Gaussian box encoding uses covariance
diag(w squared / 4, h squared / 4). Its squared 2-Wasserstein distance is [1]

\[
W_2^2=\|c_p-c_r\|_2^2+
\frac{(w_p-w_r)^2+(h_p-h_r)^2}{4}.
\]

NWD is exp(-W2/C), and the corresponding regression loss is 1-NWD. The
normalization C gives the distance its pixel scale [1, 2]. It cannot be copied
uncritically from a different image resolution or object-size distribution.

For these axis-aligned boxes, elementary expansion gives

\[
W_2^2=\tfrac12(\Delta x_{\mathrm{left}}^2+
\Delta y_{\mathrm{top}}^2+\Delta x_{\mathrm{right}}^2+
\Delta y_{\mathrm{bottom}}^2).
\]

The diagnostic independently checks both expressions. This identity explains
what information the proposed distance would add to the objective: a particular
penalty on the same four box-edge errors, with a chosen scale. It provides no
extra observation of an occluded edge, no physical-identity test and no calibrated
uncertainty. The Gaussian representation here is a geometric construction,
not evidence that a cassette's position is normally distributed.

The current native box objective already uses CIoU and normalized L1, as recorded
by the [pinned loss contract](../../scripts/quality_focal_r61.py). CIoU explicitly
accounts for overlap, center distance and aspect ratio [3]. Current official
YOLO26 documentation also explains that its `dfl` gain weights normalized L1
box distance in this DFL-free architecture [4]. Thus, adding NWD cannot be
justified by asserting that the existing model has no localization gradient
outside overlap. The frozen implementation and checkpoint, rather than the
latest web documentation, remain authoritative for a reproducible experiment.

## 3. The smallest useful investigation now

The [proposal-stage replay](proposal-stages.md) distinguishes geometry absent
from the decoded pool, adequate geometry below the confidence cutoff, suppression
and competition in the fixed matcher. This is a different question from the
geometry table above, which considers only retained outputs.

The completed replay separates the current 18 misses into **12 lost at the
confidence cutoff, three lost at native NMS, and three without any decoded box
reaching IoU 0.50**. Decoded coverage is 651 of 654 reference fronts, falls to
639 after confidence filtering, and to 636 after native NMS; postclip suppression
loses no further coverage. Maximum one-to-one matching has the same bounds here,
so candidate contention does not explain additional losses. These are measured
pipeline stages, not established learning causes. [Stage evidence](proposal-stages.md).

The NMS cases are `new_000089/R043`, `new_000089/R048` and `new_000090/R039`.
Their suppressors survive as unmatched outputs, with reference IoUs approximately
0.4689, 0.4951 and 0.4937. These are three of the near-overlap geometry cases
above. Consequently, a poorly delimited retained box does not prove that the
network never produced a better box. This makes ranking and suppression relevant
to these cases and weakens a blanket claim that regression loss is the main
bottleneck. The completed QFL comparison tests that score-learning hypothesis and finds
a substantial increase in extras. The earlier stage count did not predict a benefit.

The completed [augmentation audit](augmentation-audit.md) measures target
survival in a fixed diagnostic sample from the existing native pipeline. A small
original target can be clipped, filtered or left with very little visible area
after a valid random transform. Original label counts alone cannot establish
effective training exposure. The three fixed passes are new diagnostic draws,
not a reconstruction of historical training randomness.

It records original and transformed row identities, size and corner availability,
whether each survives the native filter, and reasons for removal. On the exact
142 training photographs, 11,997 source-object presentations become 11,880
retained targets; 117 are removed and 368 retained boxes are clipped. No source
unavailable corner becomes known. Of the removed presentations, 108 have zero
clipped box area and nine are thin edge fragments. The six selected panels show
cropping/clipping without a concrete supervision inconsistency in that bounded
sample. This supports retaining the current augmentation policy for now.

The audit uses the exact current geometric settings; mosaic and mixup are
already disabled. Numerical target survival remains separate from physical
annotation correctness. Affine movement of a rectangle does not establish that
its enclosing box follows the intended visible surface, and this diagnostic
does not prove that augmentation helps or harms recognition.

If the audit finds an implementation error, repair that error before testing
a different loss. If it finds valid but harmful filtering or augmentation, a
paired experiment changing only that identified operation is more interpretable
than simultaneously changing labels, sampling and regression. If no such issue
appears, do not infer that assignment and regression must be correct: their
training-time coverage remains a distinct measurement.

## 4. One conditional localization experiment

**Hypothesis:** after confirming visible extent and training survival, some
incomplete fronts receive adequate native foreground assignment but learn
overextended boxes. A fixed mixture of CIoU and a pixel-scaled NWD term might
reduce those extent errors while preserving complete-front corners.

This is an adaptation to test, not a result established by the papers. It becomes
worth preparing only if the proposal trace and training audit support the
hypothesis, and the completed classification comparison does not resolve the
relevant failures. The present cached predictions alone do not meet that gate.

| Component | Fixed comparison if the gate is met |
| --- | --- |
| Control | Current native CIoU term, normalized L1 and other native losses |
| Candidate | Replace only the CIoU term with 0.5 times CIoU loss plus 0.5 times (1-NWD); preserve normalized L1 |
| Foreground assignment and target weights | Preserve native task-aligned assignment, soft targets and weighting in both branches |
| NWD units | Decoded boxes converted to input-image pixels before the distance; do not mix feature-stride units |
| Normalization C | Freeze the mean square-root box area across unique training targets after nominal input resize, without random augmentation or development labels |
| Scope of the change | Apply the same term to every assigned positive; do not add partial-only weights |
| Data and initialization | The same verified 142-photo data and supplied epoch-173 weights used for the completed native/QFL comparison |
| Budget | Independent matched arms, existing 30-epoch recipe, seed 46, batch 4, input 1280, FP32, final-epoch selection |
| Classification, pose and branch schedule | Native and identical in both arms; no simultaneous QFL or head switch |
| Evaluation | Original supplied benchmark, new matched control and candidate at the established operating settings; all three reference versions and paired identities |

The blend and C rule are prospective choices for this hypothesis, not tuned
values. There is no grid search over mixing ratios, NMS, object confidence or
reference IoU. Before training, check native equivalence when the NWD coefficient
is zero and the intended box gradients when it is nonzero. Record objective
magnitudes and gradient contributions: retaining gains does not keep the loss
scale constant, so two arms would measure the whole objective change rather than
isolate a particular theoretical mechanism.

Report missed fronts, all extras, duplicates, empty-scene outputs, paired gains
and new failures, and 5/10 px complete-pair success. Retain AP50 at the existing
prespecified low score floor if scores change. A lower box distance is not itself
a recognition gain. A candidate that exchanges fewer misses for more false
outputs requires an explicit tradeoff report; it is not automatically selected.

Changing the regression loss cannot supply supervision to a target that receives
no foreground assignment. If assignment coverage is poor, diagnose that instead
of claiming this loss comparison tests it. A change to assignment would require
its own prospective protocol; importing the complete NWD-RKA strategy into this
dual-head pose model is a substantially larger intervention.

## 5. Earlier experiments that must not be repeated under a new name

| Idea | Existing evidence and implication |
| --- | --- |
| Use the trained one-to-one output head | The PR121 probe already tried this. Under its R36 reference, extras fell 94 to 67, but misses rose 15 to 16 and duplicates 31 to 37; ten-pixel pairs fell 582 to 581. One zero-height clipped output was explicitly counted. Under frozen666, misses rose 36 to 38. This is historical evidence for a tradeoff, not a current-checkpoint result or a duplicate cure. [Head audit](../front-review-r37.md). |
| Emphasize partial targets | Partial-image sampling was completed in PR107. The factor-two partial-box-loss trial changed frozen666 misses 36 to 35, extras 108 to 111 and strict pairs 572 to 570; its seed-47 screen also failed. The later partial-classification trial recovered four reviewed-reference misses with 28 more extras. These experiments use different historical checkpoints and protocols. [Prior-trial record](../project-direction-r50.md), [classification result](../partial-classification-r51-results.md). |
| Add synthetic covered-corner training | PR62 improved its fixed covered-side ten-pixel result from 54/60 to 58/60 but failed its declared clean-image preservation rule. Clean misses remained 20, extras rose 85 to 86, returned known endpoints fell 1,207 to 1,205, and an empty-scene output appeared. It does not justify simply repeating synthetic occlusion. [Completed analysis](../occlusion-results-2026-11-14.md). |

Official YOLO26 documentation describes one-to-many inference with NMS and
one-to-one inference selected by `nms=False` [5]. Switching that argument selects
a different trained output branch, not the same prediction set with suppression
removed. The project has already tested that distinction; it is not an untried
architectural shortcut.

## 6. Focused reading for the thesis

Use the two NWD papers for the specific distinction between positive assignment
and box regression, and the CIoU paper to explain the native objective fairly.
For broader study, Szeliski's *Computer Vision: Algorithms and Applications*,
second edition [6], has separate chapters on image formation, model fitting,
deep learning and recognition. Those chapters provide a useful route from camera
coordinates to learning and evaluation. The book is background reading, not
experimental evidence that any proposed cassette method will improve.

For the immediate augmentation question, read Section 7.4 of Goodfellow, Bengio
and Courville's *Deep Learning* [7]. It explains that transformations must
preserve the task's correct target and that algorithm comparisons should use
matched augmentation. Its examples concern classification; applying that
principle to this task requires checking transformed front extents and each
corner's availability. The new audit makes that project-specific inference
inspectable rather than assuming that a standard transform is automatically
appropriate.

## References

[1] J. Wang, C. Xu, W. Yang, and L. Yu, “A Normalized Gaussian Wasserstein Distance
for Tiny Object Detection,” arXiv:2110.13389, 2021, revised June 2022.
[Author preprint](https://arxiv.org/abs/2110.13389) ·
[Full paper, equations and Tables 1–3](https://arxiv.org/pdf/2110.13389).

[2] C. Xu, J. Wang, W. Yang, H. Yu, L. Yu, and G.-S. Xia, “Detecting tiny objects
in aerial images: A normalized Wasserstein distance and a new benchmark,”
*ISPRS Journal of Photogrammetry and Remote Sensing*, vol. 190, pp. 79–93, 2022,
doi:10.1016/j.isprsjprs.2022.06.002.
[Author preprint](https://arxiv.org/abs/2206.13996) ·
[Full paper, Table 7](https://arxiv.org/pdf/2206.13996) ·
[Author project](https://chasel-tsui.github.io/AI-TOD-v2/).

[3] Z. Zheng, P. Wang, W. Liu, J. Li, R. Ye, and D. Ren, “Distance-IoU Loss:
Faster and Better Learning for Bounding Box Regression,” in *Proceedings of the
AAAI Conference on Artificial Intelligence*, vol. 34, no. 7, pp. 12993–13000,
2020. [Author preprint](https://arxiv.org/abs/1911.08287) ·
[AAAI paper](https://cdn.aaai.org/ojs/6999/6999-13-10228-1-10-20200525.pdf).

[4] Ultralytics, “YOLO26 Training Recipe,” official documentation, accessed
28 November 2026. [Loss weights](https://docs.ultralytics.com/guides/yolo26-training-recipe/).

[5] Ultralytics, “Understanding End-to-End Detection in Ultralytics YOLO26,”
official documentation, accessed 28 November 2026.
[Inference paths](https://docs.ultralytics.com/guides/end2end-detection/).

[6] R. Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed.
Springer, 2022, doi:10.1007/978-3-030-34372-9.
[Author's book page](https://szeliski.org/Book/) ·
[Publisher and contents](https://link.springer.com/book/10.1007/978-3-030-34372-9).

[7] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. MIT Press, 2016,
sec. 7.4, “Dataset Augmentation.”
[Official online chapter](https://www.deeplearningbook.org/contents/regularization.html).
