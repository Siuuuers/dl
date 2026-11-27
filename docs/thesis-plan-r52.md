# R52: an annotation-focused thesis plan

27 November 2026.

**Status: proposed research, not a professor-approved thesis specification or a
claim of improvement.** The user's stated assignment is annotation and model
training, and the user is choosing the thesis topic. This plan develops the
[R50 project direction](project-direction-r50.md) using the existing data.
The returned [PR146 experiment](https://github.com/Siuuuers/dl/pull/146) requires
its own result audit; this document reports no new training results.

## 1. Recommended question and scope

**Working title:** Geometry-guided annotation quality control for partially
occluded cassette-front detection and corner localization.

**Research question:** Can an explicit annotation policy and simple checks
between neighboring instances help a reviewer find genuine labeling mistakes
at a fixed review budget, and do accepted corrections improve an otherwise
unchanged detector?

The study has two separate outcomes: the usefulness of the review process and
the effect of its accepted corrections on detection and corner localization.
A useful checker may improve review consistency without improving every model
metric. Professional reannotation produced mixed detector outcomes across
categories and datasets in [Ma et al., 2022][ma2022]; higher benchmark scores
must be measured, not assumed.

The intended contribution is an auditable annotation procedure, a measured
review method, and a controlled experiment on this application. No claim of a
new detector architecture, established algorithmic novelty, safe grasping, or
physical positioning accuracy is needed. The supervisor still determines
whether this empirical scope meets the degree requirements.

## 2. Preserve a precise physical target

The existing [annotation policy](annotation-policy.md),
`baomai-visible-front-v1`, remains authoritative. The summary below clarifies
the proposed study; it does not introduce a new label schema.

| Decision | Rule |
| --- | --- |
| Instance | One row per identifiable, visible cassette front. Entirely hidden cassettes are not created from stack spacing or model predictions. |
| Box | The tight axis-aligned enclosure of the observed front surface, clipped to the image. Its own molded frontal rim or continuous frontal tab is included. A broad receding upper plane is a different surface. |
| Shadows | Decide ownership from the **surface receiving the shadow**. A shadow on the target's front does not remove that front, even when another cassette or the rack casts it. A shadow on rack/background does not enlarge the cassette. |
| Foreign material | Rack, gripper, hand, and neighboring cassette material do not determine the target's outer bounds. An enclosing rectangle can nevertheless contain foreign pixels between owned visible extrema. |
| Corners | The actual two lower-front corners, ordered by image X, judged independently of the rectangle's intersections. A visible, reliably locatable corner uses `v=2`; an unavailable corner retains canonical `0 0 0`. |
| Ambiguity | Record uncertain ownership or competing junctions. Do not fill hidden corners from symmetry, box edges, neighboring cassettes, or predictions. Preserve recorded holds and exclusions. |
| Coordinates | Original-image X increases rightward and Y downward. Up decreases Y; `ymin` is top and `ymax` is bottom. Translate informal directions explicitly before editing. |

Small size does not change material ownership. A bright turn with a darker
underside or lateral gray region is a useful lighting-dependent clue, not a
physical law. Trace contour continuity and inspect an unmarked context crop
before accepting a coordinate.

**Context and annotation extent are separate choices.** Local and global
context can help detection, as studied by [Mottaghi et al., 2014][mottaghi2014].
Our inference for this project is that a full input image can provide rack and
neighbor context while a tight annotation still describes only the target
front. If a future crop-based method needs more context, enlarge its input
crop and preserve target geometry through the coordinate transform. Do not
expand the ground-truth box merely to provide context.

Visible and amodal targets also differ. [CityPersons][citypersons] explicitly
provides tight visible boxes and separate full-body boxes. That illustrates
why the choice must be declared; it does not prescribe an amodal cassette
task. The current study retains visible fronts and observed corners. The
professor's general detection lecture is not evidence for a different
cassette annotation policy.

## 3. A bounded study of human review

### Freeze the pool and the review method

Choose a manageable pool of existing full images before inspecting the
checker results. Record how the pool was selected, including dense layouts,
partially visible fronts, rack adjacency, and genuine empty scenes where
available. User-nominated examples are useful case studies but are a selected
sample; they cannot estimate the dataset-wide error rate by themselves.

Reuse the existing [geometry ranking script](../scripts/rank_front_geometry_candidates.py)
and the [R52 four-border audit](../scripts/audit_front_geometry_r52.py) as
review aids. Pin their versions and thresholds in the study manifest. Keep
one review item per instance, with multiple flags attached, so repeated
neighbor contacts do not count as multiple discovered labeling mistakes.

For every cue, preserve target and neighbor IDs, the foreign corner and its
visibility, the relevant border, distance in original-image pixels, and an
unmarked context crop alongside the overlay. Stable IDs must map to the exact
label version; source rows and prediction indices are not interchangeable.

**A neighboring corner inside or touching a box is a review cue, not proof of
error.** Slanted fronts, axis-aligned envelopes, perspective, and occlusion
can produce legitimate projected overlap. The neighboring point itself may
be wrong. Do not automatically shrink a box, delete a row, copy a neighboring
coordinate, or impose one offset on a cassette column.

### Establish an adjudicated subset

Inspect original pixels and record accepted correction, valid overlap,
uncertain ownership, wrong neighbor point, or another explicit reason. Where
practical, judge the crop before showing the proposed correction or model
confidence. Retain before/after labels, reviewer identity, source hashes, and
the evidence for each accepted change.

Exhaustive review of a small, predefined set of full images gives the strongest
bounded reference for evaluating the queue. If that is infeasible, review all
flagged cases plus a preselected random or stratified sample of unflagged
cases, retaining selection probabilities. A label-only geometry check cannot
find an entirely missing instance that has no box or corner. Full-frame
inspection is therefore part of the adjudication design.

This process produces adjudicated labels under a declared policy, not
infallible physical truth. Keep unresolved cases visible in the ledger. Apply
the existing whole-photograph eligibility rules before releasing training
labels; an unresolved image is not an empty negative.

### Compare queues under the same budget

Predeclare review budgets before inspecting outcomes; for example, the first
25, 50, and 100 unique cases, capped by the pool size. These are proposed
budgets, not completed review counts. Compare the frozen geometry ranking
with sequential review and a random-order baseline using recorded random
seeds. On an exhaustively adjudicated subset, the same accepted-issue labels
can score each queue without repeatedly asking a person to correct the same
case.

| Review measure | Interpretation and limit |
| --- | --- |
| Precision among reviewed flags | Confirmed issue-bearing cases divided by reviewed flagged cases. Disclose uncertain decisions separately. |
| Confirmed issues at budget K | Unique accepted issues found in the first K reviewed cases. Use a consistent unit for all queues. |
| Review time | Log actual active time prospectively. Case count alone does not establish a time saving; dense and simple images differ. |
| Issue recall | Requires an exhaustive adjudicated subset or a justified sampling estimate. Flag-only review cannot measure missed issues. |
| Repeated boundary/corner placement | Report pixel differences and availability agreement. A delayed, masked repeat by one person measures intra-annotator repeatability. |
| Inter-annotator agreement | Requires independent judgments from a second person; repeated agent suggestions approved by one person are insufficient. |

An optional later comparison is a model-disagreement queue.
[ObjectLab][objectlab] prioritizes overlooked, poorly located, and misclassified
boxes for review, using out-of-sample predictions to avoid fitting bias. Its
study obtains these with five-fold cross-validation. If adopted here, hold
related captures together and ensure the scoring model did not train on the
scored image or its near duplicate. Existing self-predictions can still point
to useful questions, but do not constitute an unbiased model-assisted review
benchmark. Five new training runs are not a prerequisite for starting the
geometry-only study.

## 4. Isolate the effect of corrected training annotations

After adjudication, define two immutable training-label versions: the current
effective baseline and a candidate containing only accepted corrections.
Record exactly which corrections actually enter training. A correction to a
development reference alone does not change model supervision.

Keep the following equal between the two training arms:

- Original image bytes, image membership, and train/development split.
- Model architecture, initialization checkpoint, optimizer, seed, batch size,
  training duration, augmentation policy, and input resolution.
- Loss functions and weights, visibility handling, inference thresholds,
  suppression method, and evaluation code.

Use the same predetermined checkpoint rule, preferably the fixed final epoch
for this comparison, rather than selecting different favorable epochs after
viewing results. Preserve the repository's existing model-selection gate.
The PR146 loss-weighting experiment remains a separate question.

If an unresolved image cannot support the candidate labels, preserve the
declared baseline fallback in both arms or choose a common eligible pool
before training. Do not silently remove it from only one arm. Accepted
instance additions or removals are legitimate annotation changes, but report
them separately from boundary/corner refinements. If both kinds are bundled,
the measured effect belongs to that combined correction package, not solely
to the geometry checker.

A single paired seed is an exploratory experiment. If a consequential benefit
appears, repeat the same pair with a predeclared additional seed before making
a stronger stability claim. Do not add several unrelated changes to rescue
an unfavorable outcome.

### Score two models against two reference versions

| Training labels | Existing frozen reference | Separately adjudicated reference |
| --- | --- | --- |
| Baseline labels | A | B |
| Accepted corrected labels | C | D |

Both columns use the same evaluation images and the same cached predictions
for each model. This is two training conditions and four scoring combinations,
not four independent experiments.

- Compare **A with C** to measure the training change against the historical
  reference.
- Compare **B with D** to measure that change against the adjudicated
  reference.
- Compare **A with B**, or **C with D**, to measure reference-version
  sensitivity for an unchanged model.

Preserve the original scores and publish the versioned rescores alongside
them. Reference amendments must be supported by image evidence, not chosen
to favor one model. If adjudication was prompted by those models' errors,
disclose that dependence. A higher score after changing reference labels does
not demonstrate additional learning.

## 5. Metrics and analysis

Predeclare the primary comparison, matcher, confidence gate, and corner
tolerances. Keep their historical definitions for compatibility. Report
uncertainty and qualitative evidence alongside counts rather than selecting
only the favorable metric.

| Measure | Required interpretation |
| --- | --- |
| Missed targets and all unmatched outputs | Report recovered and newly missed stable identities. An unmatched output is an assignment outcome, not automatically physical background. |
| Duplicates, localization mismatches, background-like outputs | State classification rules and review ambiguous cases. One localization failure can produce both a miss and an unmatched prediction. |
| Corner localization | Report the existing corner metric with its exact denominator, plus an all-eligible-target success measure in which misses fail. A matched-only average can hide detection failures. |
| Zero/one/two locatable-corner groups | Show group denominators. Availability is a useful diagnostic proxy, not a pure measurement of occlusion. |
| Front size and crowding | Stratify by a declared short-side/height grouping and scene context; report distributions rather than assuming every miss is caused by size. |
| Empty-scene false positives | Retain verified negatives and report their outputs separately. |
| AP, if evaluated | Use suitable saved predictions across confidence levels. A cache already truncated at the operating threshold cannot reproduce an unrestricted AP evaluation. |

[TIDE][tide] separates detection error types and estimates their effects on
AP. Its taxonomy motivates a more useful diagnosis than one aggregate score.
Custom fixed-threshold counts and pipeline-stage traces are complementary;
do not call them TIDE's oracle AP analysis unless that implementation is
actually run. This project's corner measures remain additional outcomes.

[Noise-Aware Evaluation of Object Detectors][noise2025] studies annotation
boundary discrepancies and their influence on evaluation. Use repeated
annotation or plausible-boundary sensitivity as a separately named analysis.
Do not lower the primary overlap requirement simply to convert unmatched
cases into successes. The study can find that measurement uncertainty is
larger than a small apparent model difference.

## 6. Data independence and claims

The repeatedly reviewed evaluation scenes are development data. Multiple
label versions, checkpoints, or crops of the same scenes are not independent
test sets. Document how often these scenes informed annotation, thresholds,
model selection, and the proposed checker.

Before any new split, identify original capture sessions, adjacent frames,
and near duplicates where the provenance supports doing so. Group related
images together. Creating a new split from already inspected material does
not make it untouched test data retroactively. Do not invent session
independence when metadata are absent.

For uncertainty estimates, resample at the image or supported capture-group
level, not as though every cassette in repeated rack views were independent.
With few groups, describe the resulting estimates as limited. Additional
independent captures can later support a prospective generalization check;
their current absence does not block a clearly scoped annotation study.

This plan can establish review yield, policy consistency, and development-set
behavior on the available material. It cannot establish broad deployment
reliability, a millimetre-level tolerance, or superiority on unseen sessions.

## 7. Practical chapter and deliverable outline

| Chapter | Main question | Concrete deliverable |
| --- | --- | --- |
| 1. Problem and scope | What is the visible cassette-front task, and why does annotation consistency matter? | Task definition, research questions, declared scope and limits. |
| 2. Related work | How do annotation quality, review prioritization, context, and evaluation noise relate? | Focused synthesis of the six primary references below. |
| 3. Data and annotation protocol | Which physical features are labeled, and how are revisions traced? | Versioned policy, image/label manifests, visibility rules, illustrated accepted and uncertain cases. |
| 4. Review method | Which geometry cues are useful, and what does a human verify? | Reproducible queue, fixed-budget comparison, adjudication ledger and measured review effort. |
| 5. Controlled experiments | What changes when training labels change while other choices remain fixed? | Paired training record, two-by-two scoring table, error/corner analysis and reference sensitivity. |
| 6. Discussion and conclusions | What worked, what remained uncertain, and what needs new captures? | Evidence-supported conclusions, limitations, and a prospective follow-up protocol. |

**Immediate sequence:** finalize the policy examples; freeze the review pool
and budget; run and adjudicate the geometry queue; verify a genuinely changed,
eligible training candidate; then prepare the controlled training comparison.
An unchanged effective training set does not justify another identical run.

If the full scope is too large, two narrower titles remain viable proposals:
**Annotation uncertainty and evaluation stability for small cassette fronts**,
or **Human review prioritization for cassette annotation correction**. Both
can produce a substantive result even when a detector gain is absent. The
main open academic questions are whether an empirical contribution satisfies
the degree requirements and whether a second independent reviewer is
available; neither needs to block the current annotation audit.

## 8. Primary references

These sources support the concepts discussed above. They do not prove a
cassette-specific improvement or establish novelty for this proposal.

1. **Jiaxin Ma, Yoshitaka Ushiku, and Miori Sagara.** *The Effect of Improving
   Annotation Quality on Object Detection Datasets: A Preliminary Study.*
   IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops
   (CVPRW), 2022.
   [Primary paper][ma2022].
   [IEEE record](https://ieeexplore.ieee.org/abstract/document/9857348).

2. **Ulyana Tkachenko, Aditya Thyagarajan, and Jonas Mueller.** *ObjectLab:
   Automated Diagnosis of Mislabeled Images in Object Detection Data.*
   ICML Workshop on Data-centric Machine Learning Research, 2023.
   [Workshop paper][objectlab].
   [Author preprint](https://arxiv.org/abs/2309.00832).

3. **Daniel Bolya, Sean Foley, James Hays, and Judy Hoffman.** *TIDE: A General
   Toolbox for Identifying Object Detection Errors.* European Conference on
   Computer Vision (ECCV), 2020.
   [Author preprint][tide].
   [Author project](https://dbolya.github.io/tide/).

4. **Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee,
   Sanja Fidler, Raquel Urtasun, and Alan Yuille.** *The Role of Context for
   Object Detection and Semantic Segmentation in the Wild.* IEEE Conference
   on Computer Vision and Pattern Recognition (CVPR), 2014.
   [Author-hosted paper][mottaghi2014].

5. **Jeffri Murrugarra-Llerena and Claudio R. Jung.** *Noise-Aware Evaluation
   of Object Detectors.* IEEE/CVF Winter Conference on Applications of
   Computer Vision (WACV), 2025.
   [Primary publication page][noise2025].
   [IEEE record](https://ieeexplore.ieee.org/abstract/document/10943665).
   [Authors' code and citation](https://github.com/Artcs1/Error-Aware).

6. **Shanshan Zhang, Rodrigo Benenson, and Bernt Schiele.** *CityPersons: A
   Diverse Dataset for Pedestrian Detection.* IEEE Conference on Computer
   Vision and Pattern Recognition (CVPR), 2017.
   [Full author preprint][citypersons].
   [Preprint record](https://arxiv.org/abs/1702.05693).

[ma2022]: https://openaccess.thecvf.com/content/CVPR2022W/VDU/papers/Ma_The_Effect_of_Improving_Annotation_Quality_on_Object_Detection_Datasets_CVPRW_2022_paper.pdf
[objectlab]: https://dmlr.ai/assets/accepted-papers/112/CameraReady/objectlab_icml2023_lowerres.pdf
[tide]: https://arxiv.org/abs/2008.08115
[mottaghi2014]: https://cs.stanford.edu/~roozbeh/pascal-context/mottaghi_et_al_cvpr14.pdf
[noise2025]: https://openaccess.thecvf.com/content/WACV2025/html/Llerena_Noise-Aware_Evaluation_of_Object_Detectors_WACV_2025_paper.html
[citypersons]: https://arxiv.org/html/1702.05693v1
