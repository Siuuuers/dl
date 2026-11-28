# What should a cassette detection score represent?

28 November 2026. Research addendum; no new model training or notebook changes.

**Decision:** finish the existing paired classification-loss and training-label
experiments. Before preparing another training run, distinguish insufficient
assignment, a low native supervision target, and a prediction below a useful
target. These are different hypotheses. The most useful new finding is that the
pinned pose model already includes corner quality in its object-score targets.
Adding “quality-aware classification” or “task-aligned assignment” without
specifying a difference would repeat an existing component.

This note develops three bounded directions: inspect the existing assignment
and score target; consider localization-quality ranking if the suppression
failures persist; and retain asymmetric positive/negative loss weighting as a
conditional alternative. None is an established cassette improvement. See the
[current model priorities](research-next-models.md),
[frozen QFL experiment](quality-focal-experiment.md),
[proposal-stage evidence](proposal-stages.md), and [report](report.md).

## 1. What the existing evidence establishes

The original-view replay partitions the retained method's 18 reviewed-reference
misses into 12 lost at confidence filtering, three lost at native NMS, and three
without any decoded proposal at box IoU 0.50. The threshold is object confidence
strictly above 0.25; native and postclip NMS use IoU 0.70. Coverage falls from
651/654 in the decoded pool to 639 after confidence filtering and 636 after
native NMS. These are disappearance stages, not causes during learning.
[Reproducible evidence](proposal-stages.md).

The three NMS losses have higher-scoring suppressors that survive as unmatched
outputs. That makes the ranking of overlapping proposals a concrete issue.
However, a change confined to NMS ranking above the fixed confidence cutoff can
recover **at most three additional reference fronts** on this sample: it cannot
recover the 12 already excluded by confidence or the three absent from the
decoded pool. This bound follows from the measured 639-front input coverage;
it is not an achievable-model result. Changing ranking also need not remove
duplicates whose mutual overlap remains below the suppression threshold.

The current result remains 18 misses, 51 extras including five duplicates, and
569/583 complete pairs within five pixels. All misses have fewer than two known
reference corners. This association does not show that missing corners cause
low scores; size, occlusion, target definition and training exposure co-vary.
[Error analysis](detection-errors.md) ·
[Box geometry](evidence/box-geometry/results.md).

## 2. What is already native in the pinned model

The inspected Ultralytics 8.4.182 files exactly match the hashes frozen in
[`quality_focal_r61.py`](../../scripts/quality_focal_r61.py):

| File | SHA-256 |
| --- | --- |
| `utils/loss.py` | `f8db64a3093ef310b23cb83d587dd5c8aa3db20a27da94d95a62a734a6c055d2` |
| `utils/tal.py` | `7fc3127e4b094247ccbbcf8af9829972295f96fc53027313ef51b076d27274cd` |

Both native training branches use `PoseLoss26` and
`PoseTaskAlignedAssigner`. For a candidate/reference pair, define c as the
nonnegative clamped CIoU and k as the native object-keypoint similarity (OKS)
over reference corners marked known. The overlap used by this assigner is

\[
u=\begin{cases}(c+k)/2,&\text{at least one known corner},\\
c,&\text{no known corners}.\end{cases}
\]

Here c is **clamped CIoU, not plain IoU**. OKS uses the configured per-corner
sigmas and a box-area scale of 0.53 times width times height. It averages over
known corners only. An unavailable corner does not automatically halve the
OKS. A zero-corner target uses the box component alone. Conversely, the area
scale means that a fixed pixel error need not have the same OKS on two differently
sized fronts. This differs from the report's fixed five-pixel endpoint criterion.

The assignment metric is t = p^0.5 u^6, where p is the detached predicted class
score. The native code selects eligible top candidates, resolves conflicts, and
normalizes the surviving metric separately for each target. For an assigned
anchor a of target g, the soft class target has the form

\[
q_a=\frac{t_a}{\max_{j\in P_g}t_j+\epsilon}
       \max_{j\in P_g}u_j,
\]

where P_g is the surviving positive set. Thus q is neither a binary presence
label nor raw box IoU. Normalization matters: uniformly small alignment values
do not by themselves imply uniformly small targets. Ignoring the small epsilon,
the largest q for a target with positive alignment equals its largest surviving
u. The actual assigned targets, not the unnormalized t values, must be inspected.

The same mixed overlap affects assignment, conflict resolution and soft class
targets; the target scores also weight the native box objective. Predicted
geometry and scores are detached for assignment. Native classification applies
BCE to these targets and divides by their summed weight, clamped at one.
The existing QFL arm preserves all of this and changes only the BCE modulation.
It does not replace q with raw IoU.

The candidate-center eligibility rule also already enlarges each very short
reference-box side to a native stride-based minimum around its center. That
expands the assignment search region; it does not enlarge the regression target.
Therefore “tiny boxes have no anchor center” must be measured after this rule,
not inferred from the original rectangle alone. The fixed source files are
authoritative; the [current API documentation][4] is a readable but moving
reference.

## 3. Three directions supported by the reading

### A. Task alignment: inspect the existing mechanism first

TOOD [1] combines a task-aligned head with task-aligned learning. Its learning
rule uses classification/overlap alignment, instance-normalized soft targets,
and weighted localization. In Table 2, the ATSS baseline is 39.2 AP, ATSS with
QFL is 39.9, and TAL is 40.3 under the same base schedule. The full model also
changes the head; its overall improvement cannot be attributed to assignment
alone. These are COCO detection results.

**Project implication:** the core alignment idea and continuous targets are
already present, with a pose-specific overlap. Installing TOOD as an entire
detector would not isolate the question raised by the partial-front errors.
The useful adaptation is to inspect whether eligible incomplete targets receive
positive anchors, whether their mixed-quality targets are low, and whether
predicted scores fall below those targets. No such per-target finding is claimed
here.

### B. Localization-quality ranking: relevant to three measured NMS losses

IoU-Net [2] learns localization quality and uses it to choose NMS survivors.
Its Table 1 FPN comparison raises AP from 36.4 to 37.3 and AP90 from 9.8 to 13.3,
but **lowers AP50 from 58.0 to 56.0**. Training samples jittered reference boxes,
keeps IoU at least 0.50 and samples across IoU; the paper notes poorer quality
estimation near low IoUs. Its algorithm also transfers the maximum class score
within a suppressed cluster. The paper includes a separate box-refinement method.

**Conditional adaptation:** if the returned models retain the three suppression
failures, investigate a small, separate box-quality output that ranks the
unchanged above-0.25 proposals before native NMS. Freeze the detector and retain
its boxes, corners and original reported scores to isolate survivor selection.
This would be a ranking-only adaptation, not a reproduction of the paper's
clustering or refinement. It was already named as a later option in the QFL plan;
the new stage evidence gives that option a bounded purpose.

Do not multiply another box-quality estimate into the existing score without
specifying why: the current target already includes box and corner quality.
Training the added predictor requires training-image proposals spanning the
relevant quality range, including the boundary near IoU 0.50. An unmatched
development prediction is not automatically a background label. A ranker must
preserve corner outcomes as well as improve box selection, since its pure box
criterion could prefer worse corners. This is more work than a loss-only change
and is not added to the current GPU queue.

### C. Varifocal weighting: a distinct, conditional classification alternative

VarifocalNet [3] separates its loss ablation from a star-shaped feature sampler
and box refinement. Table 3 changes AP from 39.0 with focal loss to 40.1 with
varifocal loss; the additional modules raise it to 41.6. For positive targets,
its Eq. (2) is q times BCE(p,q); for zero targets it is alpha times p^gamma times
BCE(p,0). The paper uses raw box IoU for positive q and selects alpha 0.75,
gamma 2. The full detector's gain is not a loss-only result.

**Difference from the prepared experiment:** current QFL multiplies every native
BCE term by |p-q| squared. A varifocal-style adaptation would preserve positive
learning differently while suppressing easy negatives. However, multiplying a
positive by q reduces its contribution when q is small; it is not automatically
a remedy for low-quality partial-front targets.

Only consider a matched weighting comparison if the returned native/QFL traces
show adequate positive assignment and useful targets but a persistent learning
imbalance. Preserve the native target-construction, assignment and normalization
rules and all other loss definitions in that comparison. The realized q values
and assignments can differ as the models' predictions evolve; they are not
forced to remain identical across training arms. Changing the target definition
and weighting together would prevent attribution. With native pose targets it would be a
varifocal-style weighting test, not the paper's IoU-aware-score reproduction.
No parameter search or third training notebook is proposed now.

## 4. One diagnostic that distinguishes these hypotheses

**First consume the diagnostics already being produced.** The frozen QFL driver
logs assigned/unassigned partial and full target counts, foreground anchors,
positive class terms, target-score sums, and native/modulated foreground and
background BCE contributions in each branch. Read the native control alongside
the candidate over training. A change in total loss alone cannot identify the
source of the final errors. The present “partial” grouping combines zero and one
known corner, so it cannot by itself test the pose-target distinction above.

If those aggregates leave the question open, specify **one frozen-checkpoint,
no-optimizer assignment pass** on the verified training images, preserving the
source images/labels, input transforms and checkpoint hashes. This is a proposed
diagnostic, not a completed experiment. Record each target's source identity,
post-transform width/height and zero/one/two known-corner group. For each native
branch, retain:

1. Candidate-center eligibility, selected candidates before conflicts, and
   surviving foreground anchors after all assignment filters.
2. The box CIoU, known-corner OKS, mixed overlap, normalized q and predicted p
   for those anchors; retain distributions and per-target values, not only means.
3. Distinct counts for “no foreground anchor” and “foreground anchor but no
   positive class target.” These are not interchangeable.

Use the same frozen predictions for one descriptive counterfactual: recompute
the assignment with box-only overlap while retaining the candidate pool,
hyperparameters and target geometry. Report changed assignments and q values
alongside both original quality components. This isolates how the pose component
affects the target construction at that checkpoint; it is **not** a trained
candidate, an inference rule or evidence that removing pose alignment improves
recognition. Do not modify the frozen GPU notebooks to collect it mid-run.

| Observation in that pass | Next question it supports |
| --- | --- |
| No eligible centers after the native enlargement rule | Inspect geometry, stride and target encoding before changing a loss. |
| Eligible candidates but no surviving assignment | Inspect selection/conflict behavior and branch differences. |
| Low q alongside low box quality | Investigate visible-extent supervision and localization; classification modulation cannot create missing geometry. |
| Low q that changes substantially when only pose overlap is removed | Test whether the pose-based detection target matches the requirement to detect incomplete fronts. |
| Useful q but systematically lower p | The current classification-loss comparison addresses a plausible optimization question; asymmetric weighting remains a later option. |
| Adequate above-cutoff geometry loses to a poorer overlapping box | Consider the bounded quality-ranking experiment, retaining corner and extra-output checks. |

Assignment depends on the model's current predictions. A snapshot cannot
reconstruct historical training causes, and a training-image discrepancy cannot
prove the cause of a particular development miss. These restrictions also apply
if the descriptive box-only counterfactual looks preferable.

Any later trained comparison needs matched initialization, data, schedule and
selection, the supplied-model benchmark, and unchanged operating settings.
Report misses, extras, duplicates, empty images, complete-pair errors and paired
case identities across all three references. Keep the prespecified AP50 measure
when a method changes score distributions. Do not select a new confidence
threshold from the reused development set.

## Primary sources and access

[1] C. Feng, Y. Zhong, Y. Gao, M. R. Scott, and W. Huang, “TOOD: Task-Aligned
One-Stage Object Detection,” in *Proc. ICCV*, 2021, pp. 3510–3519.
[Full author paper, Section 3.2 and Table 2](https://arxiv.org/pdf/2108.07755) ·
[Author project](https://fcjian.github.io/tood/index.html).

[2] B. Jiang, R. Luo, J. Mao, T. Xiao, and Y. Jiang, “Acquisition of Localization
Confidence for Accurate Object Detection,” in *Proc. ECCV*, 2018, pp. 784–799.
[Full author paper, Sections 3.1–3.2 and Table 1](https://arxiv.org/pdf/1807.11590) ·
[Official publication](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Borui_Jiang_Acquisition_of_Localization_ECCV_2018_paper.php).

[3] H. Zhang, Y. Wang, F. Dayoub, and N. Sünderhauf, “VarifocalNet: An IoU-Aware
Dense Object Detector,” in *Proc. CVPR*, 2021, pp. 8514–8523.
[Full primary paper, Eq. (2) and Table 3](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.pdf) ·
[Author preprint record](https://arxiv.org/abs/2008.13367).

[4]: https://docs.ultralytics.com/reference/utils/tal/

The native-code findings above come from the hash-verified installed package and
the repository's frozen loss contract, not from a current-version assumption.
The current official [assigner reference][4] provides additional reading.
TOOD and IoU-Net full text was accessible through the authors' arXiv PDFs.
Direct CVF fetches initially failed; the indexed primary VarifocalNet PDF exposed
its method, equations and ablation tables. No necessary paper remained inaccessible
for this addendum, so no user upload is required for the claims made here.
