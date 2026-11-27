# R48: a focused thesis and experiment plan

**R50 clarification:** the user has not settled a thesis topic. Professor LI
Hongsheng's uploaded document is an object-detection lecture; the stated current
assignment is to annotate data and train a model. This page is a historical
proposal for discussion. Use [the revised project direction](project-direction-r50.md)
for the current scope and work possible while new captures are unavailable.

27 November 2026, Asia/Hong_Kong.

**Proposed research plan.** This document explains the next scientific question;
it does not report a new training, filtering, tiling, or robot experiment. The
PR137 result audit and versioned R48 annotation review supply the measured
evidence. A correction to a development reference is an annotation revision,
not a model improvement. Preserve the originally published scores alongside
any explicitly named rescore.

## 1. Recommended scope

Suggested working title: **Error-Guided Detection and Corner Localization of
Partially Occluded Tissue Cassettes**.

Main question: **Which sources of unmatched predictions limit reliable
two-corner localization, and can one targeted change reduce them while
preserving detection recall and corner accuracy?**

A defensible applied thesis can contribute a clear target definition, an
auditable dataset, a reproducible error analysis, and a controlled experiment.
An improvement is a hypothesis to test. A negative result that explains why
more data or a filter failed can still answer the question. Whether that scope
satisfies the degree requirements should be confirmed with the supervisor.

The present task is 2D localization of the visible cassette front and its two
actual lower-front corners. A successful image-space result alone does not
establish a safe grasp, 3D pose, or millimetre positioning accuracy. A thesis
requiring those outcomes additionally needs camera/robot calibration, depth or
other geometric evidence, a declared physical tolerance, and robot trials.

## 2. Keep the physical target consistent

The [annotation policy](annotation-policy.md) defines one row per identifiable
visible front, its tight axis-aligned visible-front extent, and independently
available lower corners. The image origin is upper-left: `ymin` is the **top**
bound and `ymax` the **bottom** bound. User descriptions such as "ymin (bottom)"
must be resolved from the named original crop and stable row identity before
changing a coordinate.

Small size does not change which cassette owns a pixel. Include the target's
own shaded front or molded frontal rim. A cast shadow on that front does not
remove the underlying front material. Do not enlarge the outer bounds to a
shadow on the rack, another cassette, or a receding upper plane merely to make
the object bigger. An enclosing rectangle may contain some foreign pixels
between the target's visible extrema; that is different from using those
foreign pixels to define the extrema. Neighboring cassettes provide context
for interpretation without becoming part of the target label.

The detector receives contextual pixels from the input image even when the
target box is tight. Adding a neighboring shadow to the label is therefore
not required to make that context available. If a crop-based experiment needs
more context, enlarge its **input crop**, then keep its target annotation and
coordinate transform faithful to the same physical definition.

## 3. What the measured result says, and what size can explain

### The strongest current lead is partial-front supervision

The [audited PR137 return](additional-front-pilot-r47-results.md) gives the
following comparison under the **unchanged R38 reference**:

| Quantity | 138-photo reference | 142-photo candidate |
| --- | ---: | ---: |
| Missed targets | 13 | 15 |
| All unmatched outputs | 90 | 84 |
| Formal duplicate subset | 18 | 14 |
| Other unmatched subset | 72 | 70 |
| Two-corner fronts matched | 583 / 583 | 583 / 583 |
| Both corners within 5 px | 569 / 583 | 566 / 583 |
| Both corners within 10 px | 581 / 583 | 582 / 583 |

The candidate misses **12 of 54 one-corner targets** and **3 of 17 zero-corner
targets**. It misses no two-corner target. There are no recovered R38 targets;
`new_000008 / R001` and `new_000010 / R030` become newly missed, with their
best retained prediction IoUs of approximately 0.3906 and 0.3296. These are
assignment outcomes under the stated reference, not a claim that every
reference boundary has been independently certified. Availability groups are
a diagnostic proxy: a missing corner may be occluded, clipped, or otherwise
unlocatable.

The added four photographs contain **172 fronts: 164 with two locatable
corners, eight with one, and none with zero**. The eight one-corner additions
all come from `frame_0178`. Training counts in the two/one/zero-corner groups
change from **3,233 / 457 / 137** to **3,397 / 465 / 137**. Thus the expansion
adds mainly fully locatable examples, while every current miss occurs in the
partially locatable groups.

This makes consistent partial-front boundaries and useful supervision for
occluded/clipped cases the leading next research question. It does not prove
that the mixture caused the regression or that adding any partial examples
will solve it. The complete-photo eligibility rule can also favor easier
scenes when the hardest photographs remain unresolved; that is a potential
selection effect to examine, not a reason to admit incomplete labels.

Size alone does not separate the errors. Under this same reference, missed
partial targets have heights **10-45.56 px** (median **23 px**), matched
partial targets **11-60 px** (median **26.18 px**), and matched two-corner
targets **9.5-67 px** (median **29 px**). The ranges overlap substantially.
Use the [complete error index](../results/evaluation/front-errors-r48-20261127/error-index.json)
for the underlying identities and diagnostic definitions. A scale ablation
remains reasonable if the joint size/visibility analysis supports it; it is
not the demonstrated primary cure.

### Measured reference geometry

The frozen current-R38 development reference contains 654 fronts at original
1280 x 720 coordinates. Reading every development TXT after checking its
manifest hash gives a median box width of **92 px** and median height of
**29.0 px**. Height quartiles are approximately **24.0 px** and **35.0 px**;
the tenth percentile is **20.9 px**. These are annotation dimensions, not
physical cassette dimensions or new model measurements.

Source: [`data/reviews/fronts-2026-11-26-r38/manifest.json`](../data/reviews/fronts-2026-11-26-r38/manifest.json),
SHA-256 `0921081930f49340ff34290d5752be2d669f084f4d1518e2019060a5600c43b5`.
Percentiles use linear interpolation at `(N - 1) * p`; all 654 rows are included.
The measurements support investigating **thin fronts and short-axis error**.
They do not establish that every object is tiny or that scale causes the
observed unmatched predictions.

### A short derivation

For two identical boxes of width `w` and height `h`, aligned horizontally but
displaced vertically by `d`, where `0 <= d < h`:

\[
 A_{\rm intersection}=w(h-d),\qquad
 A_{\rm union}=w(h+d),\qquad
 \operatorname{IoU}=\frac{h-d}{h+d}.
\]

These are illustrative calculations, not measured errors:

| Box height | Vertical displacement | IoU |
| ---: | ---: | ---: |
| 12 px | 4 px | 0.500 |
| 24 px | 4 px | 0.714 |
| 60 px | 4 px | 0.875 |

Thus, the same absolute displacement penalizes a thin front more strongly.
If a box instead contains a correct height `h` plus an unnecessary margin `r`
above and below, with the correct width, its IoU is `h / (h + 2r)`. This also
explains why extra rack or neighboring-shadow extent can matter.

The NWD research documents IoU sensitivity for tiny aerial objects and studies
different training assignment/localization measures [2]. It motivates this
diagnosis; its published detector changes are not an established improvement
for the present pose model. Keep the existing IoU criterion for the primary
comparison. Any supplementary distance diagnostic must be named separately.

### Three different dimension questions

| Question | Check before changing the model |
| --- | --- |
| Are image/label transforms wrong? | Verify original width and height, normalized-to-pixel conversion, aspect ratio, letterbox padding, crop offsets, and inverse mapping of both corners. A transform bug can affect every method. |
| Does the target occupy too few useful input pixels? | Stratify errors by front height/short side and occlusion, then compare one supported inference scale using the same checkpoint. |
| Is the image dim or low contrast? | Inspect original unmarked crops and compare error rates in a declared illumination/contrast grouping. Low intensity alone is not evidence that enhancement will help. |

Upsampling the existing 1280 x 720 image interpolates existing samples; it does
not acquire new optical detail. It can still alter the scale at which the
network processes a front. Capture at genuinely higher resolution, changing
camera distance, larger inference inputs, and sliced inference are different
experiments and should not be combined under a single "higher resolution"
claim.

## 4. Analyze the failure before choosing its treatment

TIDE separates detection errors and evaluates their effects rather than relying
on one aggregate AP number [1]. The present single-class, two-corner task can
use that principle while retaining its existing matcher and additional
corner criteria. Do not call a custom count table an implementation of TIDE's
oracle AP analysis unless those oracles are actually run.

| Observed case | Plausible cause to verify | Smallest useful next test |
| --- | --- | --- |
| Unmatched output overlaps a missed front just below the matching gate | Boundary extent or position, including gripper/rack ownership | Inspect original pixels; report box error and corner error separately; rescore both models under a separately versioned corrected reference if justified. |
| Several outputs describe the same visible cassette | Ranking, localization, or suppression | Link them by original-image evidence; check which output is assigned and which distinct neighbor might be lost by stronger suppression. |
| Output lies on confirmed rack/gripper/background | Background discrimination | Review complete hard scenes; add verified negative supervision only when every co-visible cassette is accounted for. |
| A real front has no suitable retained proposal | Visibility, scale, confidence, or an earlier suppression stage | Trace its processing stage before changing a global threshold; then run one targeted scale or data experiment. |
| Matched front has a bad corner | Local junction ambiguity, visibility confidence, or coordinate error | Compare the actual physical corner and endpoint confidence; suppression alone does not fix that coordinate. |
| Competing material owners or instance identities remain plausible | Incomplete annotation evidence | Record uncertainty and apply the existing whole-photo gate. Do not silently label it as background. |

"Unmatched" is an assignment outcome, not a physical object class. One bad box
can generate both a missed reference target and an unmatched model output.
Formal duplicates are a subset of unmatched outputs. Low or zero IoU does not
by itself establish that an output describes physical background, especially
where annotation identity remains under review.

For each case, retain the image hash, reference version, checkpoint, inference
settings, stable target ID, output index, box and corner coordinates, and the
match reason. Stratify error counts by front height and availability of zero,
one or two reference corners. Use denominators for each group: ten errors in
twenty objects and ten in two hundred objects are different findings.

## 5. Can the model use a filter?

Yes. The useful design depends on what "filter" means.

### Image preprocessing

A deterministic filter can be placed before the detector:

\[
 x' = F(x),\qquad (\hat B,\hat K)=f_\theta(x').
\]

Examples include a fixed contrast or gamma transform. Compare a single
well-defined transform against the unchanged input first. A prettier image
is not sufficient evidence: measure missed targets, unmatched outputs,
corner success, latency, and artifacts. If the transform is adopted for a new
training experiment, specify its use at training and inference; otherwise an
uncontrolled change in input distribution may explain the result.

### Learned, image-dependent preprocessing

An auxiliary network can predict the filter parameters:

\[
 a=g_\phi(x),\qquad (\hat B,\hat K)=f_\theta(F(x;a)).
\]

IA-YOLO is a concrete primary example: it predicts differentiable image-filter
parameters and trains them with YOLOv3 using detection loss, with experiments
in fog and low light [5]. This establishes feasibility, not benefit on these
cassette images. If original-image analysis identifies a lighting bottleneck,
test a simple fixed transform before adding such a learned module.

### Filtering predicted objects

NMS acts after detection. Soft-NMS reduces overlapping detections' scores
instead of immediately deleting them at one hard overlap cutoff [4]. Learned
NMS shows that a network can learn rescoring from boxes and confidence [6].
Both directions address competition among proposals; neither guarantees
better corner coordinates or recovery of visually unsupported objects.

Stronger suppression can remove a second box for one cassette and also remove
a distinct adjacent cassette. Compare recall and endpoint success along with
the unmatched count. An output count reduced by suppressing correct targets
does not meet the project objective.

**Cache limitation:** the repository's `raw` prediction records are the model
outputs **after ordinary model NMS**, but before the additional postclip
suppression. Applying a new filter to that cache cannot recover hypotheses
already discarded by the ordinary NMS stage. A fair replacement-NMS study
requires a verified inference rerun or correctly decoded pre-NMS candidates,
with their corners and identities kept together. See the existing
[prediction definitions](front-error-review-r47.md).

## 6. Recommended experiment order

### A. Finish the evidence and label release

R48 has completed the audit of PR137's receipts, input inventories and saved
predictions. The [fixed-cache reference sensitivity replay](front-reference-sensitivity-r48.md)
also compares both unchanged arms under the original R38 and proposed R48
labels. Misses remain 13/15 and total unmatched remains 90/84 (reference/candidate)
under either version. Formal duplicates change 18/14 to 16/13 because several
already-unmatched outputs cross the reference-dependent IoU boundary. The
reference arm's strict corner successes change 569 to 570 after one annotation
correction; the candidate stays566. No target assignment changes.

This is a concrete thesis example of annotation sensitivity, not a model
improvement. Report the original R38 result first. Never compare an old-model
score under one label version with a new-model score under another and
attribute the difference to training.

Check whether the new amendments actually change eligible training inputs.
If they affect only development references or still-held photographs, CPU
rescoring is useful and repeating identical training is not a new treatment.

Use the existing experimental history before proposing another intervention:

| Existing project evidence | What can already be used in the thesis |
| --- | --- |
| [Fixed 1280 versus 1920 inference comparison](partial-front-improvement-2026-11-11.md) | The larger input worsened misses and corner outcomes for that checkpoint/reference. More pixels at the network input was already tested; a new scale experiment needs a new, explicit reason. |
| [Completed PR40 partial-front label extension](partial-front-results-2026-11-11.md) | The specified extension did not recover additional targets under its declared recipe. A physically justified label correction and a measured model benefit are separate conclusions. |
| [Decoded-proposal and confidence-filter investigation](proposal-filter-review-2026-11-11.md) | The project already distinguishes low confidence from absent qualifying geometry and validates original/input coordinate transforms. Its subsequent threshold comparison is a useful bounded diagnostic example. |

These older experiments use their own checkpoints, label versions and operating
points. Present each comparison under its original contract. Their numbers
cannot be pooled with R38 or a revised R48 reference as if they were one fixed
benchmark. Existing R29 sampling and R31 partial-box-loss implementations also
need their returned evidence reviewed before treating those mechanisms as
untried thesis ideas.

### B. Select one cause from the complete error analysis

The first additional model experiment should test the dominant supported
cause, rather than combine more data, a filter, a new loss and new suppression.

For the present PR137 evidence, prioritize a consistent, complete release of
training photographs that actually adds useful partial-front supervision.
Review the front/gripper boundaries and resolve the remaining physical
identity questions; obtain new evidence where the original image cannot do
so. Preserve genuinely unavailable corners. Do not fill missing endpoints
from predictions, label unresolved fronts as background, or bypass whole-photo
completeness to increase the example count.

If errors concentrate on thin fronts, a small first experiment is one frozen
checkpoint at the existing full-image scale versus one supported larger input
scale. Keep aspect-ratio handling, confidence, matching, suppression and
original-pixel corner criteria fixed. Report compute/memory and wall time.
Verify the complete forward/inverse box-and-corner transform before comparing
scores. This is an inference-scale experiment, not additional training. The
earlier negative 1920 result means this is a conditional branch, not the
default next run or a reason to start a resolution sweep.

If that comparison supports a scale bottleneck, overlapping input crops are a
separate candidate. SAHI provides primary evidence for sliced inference and
sliced fine-tuning on small aerial objects [3]. For this project, a pose-aware
implementation additionally needs correct corner mapping, treatment of
crop-truncated objects, global merging, and enough context to distinguish
neighboring fronts. The paper's box-detection results do not certify those
pose operations. Include full-image inference as the comparison and account
for all additional inference calls.

If the dominant problem is confirmed front/rack ownership, a changed,
complete training-label release is the more direct treatment. Use common
initialization and an explicit training budget. Equal epochs on unequal
dataset sizes give different exposure; the
[R47 protocol](additional-front-pilot-r47.md) correctly declares that extra
exposure as part of its treatment. A thesis claiming an isolated label-quality
effect needs to control the competing exposure change or limit its claim.

### C. Reserve enhancement or suppression replacement for its own evidence

Run an enhancement ablation only after documenting a contrast/illumination
bottleneck. Run a suppression replacement only after distinguishing duplicate
physical hypotheses from adjacent true fronts and obtaining suitable inputs
to that suppression stage. If a simple change fails the relevant endpoint,
report the result and stop expanding that branch without new evidence.

## 7. Minimal scientific evaluation contract

1. **Freeze the task and final reference.** Record the front extent, corner
   definition, unavailable-point policy, split, thresholds and source hashes.
   Where feasible, have a second reviewer inspect a small declared sample of
   clear and partial fronts using unmarked originals before seeing model
   overlays. Report boundary/corner disagreement and adjudication; a favorable
   prediction is a review hint, not independent ground truth.
2. **Use development data for decisions.** The repeatedly inspected twenty
   development photographs and their R38/frozen666/historical663 annotation
   versions are not three independent test sets. Repeated selection against a
   finite development sample can bias evaluation [8].
3. **Hold out capture groups.** Obtain independent capture sessions or scene
   arrangements, and keep each group entirely within one split. Check both
   exact duplicates and neighboring/near-duplicate frames. Applying blocked
   evaluation principles [7] to camera sessions is a methodological proposal
   for this project, not evidence that its current groups are already known.
4. **Declare primary and companion metrics.** A suitable primary task measure
   is matched-front success with both locatable corners within the fixed 5 px
   tolerance, divided by all eligible two-corner reference fronts. Missed
   fronts and rejected predicted endpoints remain failures. Retain 10 px,
   one-corner results, misses, unmatched/duplicate counts, precision/recall,
   and runtime as companion measures. A pixel tolerance still needs physical
   justification before any robotic acceptance claim.
5. **Compare paired cases.** Evaluate both treatments on the same frozen
   images. Show recovered and newly failed identities, including regressions.
   Where repeated capture groups exist, quantify variation at that group
   level. Do not treat hundreds of fronts within related frames as hundreds
   of independent capture experiments.
6. **Separate a pilot from a reliable effect estimate.** A single seed can
   identify a promising direction. When a training claim matters and resources
   permit, repeat the selected comparison with multiple seeds and report each
   result and its variation. Avoid selecting the best seed after seeing the
   test results.

If an independent capture set cannot be obtained, the project can still be
reported as a development case study. Its conclusion should then describe
performance on the reviewed scenes rather than generalization to unseen
capture sessions.

## 8. A practical thesis outline

| Chapter | Question it answers | Evidence to prepare |
| --- | --- | --- |
| Introduction | Why are visible-front detection and reliable corner localization useful? | The intended positioning task and the limits of a 2D study. |
| Related work | Which mechanisms address our observed failure types? | The selected papers below, with their datasets and limits. |
| Data and annotation | What exactly counts as one target and one corner? | Capture grouping, original examples, annotation policy, stable revisions and uncertainty handling. |
| Baseline and error analysis | Where does the existing system fail? | Reproduced settings, complete error inventory, size/occlusion strata and case-linked metrics. |
| Targeted experiment | Does one supported intervention improve the intended endpoint? | A declared hypothesis, controlled comparison, runtime and both gains and regressions. |
| Discussion and conclusion | What was learned, and what remains unproven? | Evidence-backed answer, failed hypotheses, annotation/test limitations and next capture requirement. |

The most consequential scope question for the supervisor is whether the
required outcome is **image-space detection/corner localization** or a
**demonstrated physical positioning/grasping system**. Resolve that before
adding depth processing, robot interfaces, or a substantially larger model.

## References and what each supports

The publications below motivate hypotheses. Their benchmark improvements are
not results for this project.

1. Daniel Bolya, Sean Foley, James Hays, and Judy Hoffman. **TIDE: A General
   Toolbox for Identifying Object Detection Errors.** ECCV, 2020.
   [Official paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480562.pdf).
   Supports error decomposition and isolated error-effect analysis; it does not
   replace physical adjudication or pose-specific endpoint evaluation.

2. Chang Xu, Jinwang Wang, Wen Yang, Huai Yu, Lei Yu, and Gui-Song Xia.
   **Detecting tiny objects in aerial images: A normalized Wasserstein distance
   and a new benchmark.** ISPRS Journal of Photogrammetry and Remote Sensing,
   2022. [DOI: 10.1016/j.isprsjprs.2022.06.002](https://doi.org/10.1016/j.isprsjprs.2022.06.002).
   [Author preprint of the earlier four-author NWD study](https://arxiv.org/abs/2110.13389).
   Supports the diagnosis of small-box IoU sensitivity. Its aerial, anchor-based
   detector experiments are a different setting from the current pose task.

3. Fatih Cagatay Akyon, Sinan Onur Altinuc, and Alptekin Temizel.
   **Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection.**
   ICIP, 2022, pp. 966-970.
   [DOI: 10.1109/ICIP46576.2022.9897990](https://doi.org/10.1109/ICIP46576.2022.9897990);
   [author paper](https://arxiv.org/abs/2202.06934).
   Supports overlapping crops, resized inference and merging. The reported
   experiments concern aerial box detection, not cassette corner accuracy.

4. Navaneeth Bodla, Bharat Singh, Rama Chellappa, and Larry S. Davis.
   **Soft-NMS -- Improving Object Detection With One Line of Code.** ICCV,
   2017, pp. 5561-5569.
   [Official paper page](https://openaccess.thecvf.com/content_iccv_2017/html/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.html).
   Supports continuous score decay as an alternative to hard suppression;
   it does not guarantee fewer unmatched outputs at the current operating point.

5. Wenyu Liu, Gaofeng Ren, Runsheng Yu, Shi Guo, Jianke Zhu, and Lei Zhang.
   **Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions.**
   AAAI, 2022, pp. 1792-1800. DOI: 10.1609/aaai.v36i2.20072.
   [Official paper](https://cdn.aaai.org/ojs/20072/20072-13-24085-1-2-20220628.pdf).
   Establishes that a detector can jointly learn image-dependent filter
   parameters. Its fog/low-light evidence does not establish a lighting
   bottleneck in the present images.

6. Jan Hosang, Rodrigo Benenson, and Bernt Schiele. **Learning Non-Maximum
   Suppression.** CVPR, 2017, pp. 4507-4515.
   [Official paper page](https://openaccess.thecvf.com/content_cvpr_2017/html/Hosang_Learning_Non-Maximum_Suppression_CVPR_2017_paper.html).
   Establishes learned rescoring/suppression as a feasible design; it is an
   additional trained component rather than a free repair of the current output.

7. David R. Roberts et al. **Cross-validation strategies for data with temporal,
   spatial, hierarchical, or phylogenetic structure.** Ecography 40, 2017,
   pp. 913-929. [DOI: 10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881);
   [author-hosted paper](https://www.biom.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf/at_download/file).
   Supports evaluation that respects dependent groups. The appropriate camera
   grouping must be established from this project's capture provenance.

8. Gavin C. Cawley and Nicola L. C. Talbot. **On Over-fitting in Model Selection
   and Subsequent Selection Bias in Performance Evaluation.** Journal of
   Machine Learning Research 11, 2010, pp. 2079-2107.
   [Official paper](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf).
   Supports separating model-selection data from final evaluation and reporting
   the limitations of repeated development-set optimization.
