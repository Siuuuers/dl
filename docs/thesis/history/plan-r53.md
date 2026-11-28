> Historical snapshot. The current report and completed PR149/PR150 interpretation are in [the thesis index](../README.md). Statements about pending work below describe the original revision.

# Thesis workspace: annotation quality for cassette-front perception

Updated 27 November 2026, Asia/Hong_Kong.

**Working title:** Geometry-guided annotation quality control for partially
occluded cassette-front detection and corner localization.

**Status:** a proposed empirical thesis, awaiting the supervisor's scope
requirements. The assignment currently known is to annotate images and train a
model. No new architecture, independent test-set improvement, or physical
grasping accuracy is claimed. This is the canonical place to update the topic,
evidence, hypotheses, experiments, references and decisions; earlier R48–R52
plans remain historical records.

## 1. The question we can answer with the data we have

Can an explicit visible-front annotation policy and a simple geometry-based
review queue help find genuine label errors at a fixed review budget? After
accepted corrections, does an otherwise unchanged pose model improve its
detection and corner localization?

These are two measurements. A useful review process may improve consistency
without improving every detector metric. Conversely, a model score can rise
because the evaluation labels changed, even when the model learned nothing
new. Keep the review result, training effect and reference-version effect
separate. [Ma et al.][ma] and [noise-aware evaluation][noise] motivate measuring
these distinctions instead of assuming better-looking labels guarantee a
better score.

The immediate supporting engineering question is narrower: **does changing
only image contrast or inference sampling scale help the existing model?**
Two separate R53 notebooks test that question. They support the annotation
study without turning it into a search over many filters, models and losses.

### A manageable contribution

1. A reproducible annotation policy, original-image evidence and versioned
   correction ledger.
2. A measured review queue compared with sequential and seeded random review.
3. An error analysis that separates missed targets, unmatched proposals,
   duplicate proposals and corner failures.
4. A controlled training comparison once an adjudicated, meaningful correction
   set is ready. Cheap inference probes can be reported as supporting evidence.

This is a defensible application study proposal, not evidence that the
geometry rule is novel. The supervisor should confirm whether an empirical
study is sufficient or a new methodological contribution is required.

## 2. What is established in this repository

| Item | Verified state | Consequence |
| --- | --- | --- |
| Target | One visible cassette front, plus two actual lower-front corner slots | A box-only detector is not a drop-in replacement. |
| Latest completed training | PR146/R51: 142 full photographs, 3,999 fronts; same photos in both arms | We have trained on frame photographs, not only isolated PDF crops. |
| Photo membership | 137 original training PNGs plus `frame_0146`, `frame_0160`, `frame_0178`, `frame_0371`, `frame_0383` | Supplementary images have already entered this experiment; other held photos have not automatically entered training. |
| Training initialization | Verified cassette checkpoint at epoch 173; fresh optimizer/training run, `resume=False` | Restarting these arms is fine-tuning, not random initialization. Public pretraining ancestry is not established by the available metadata alone. |
| Model contract | YOLO26s-Pose / Pose26, one class, two 3-value keypoints, native pose/RLE losses | Architecture comparisons must retain corner and unavailable-corner semantics. |
| Evaluation | 20 repeatedly examined development photographs, scored against four label versions | Four references are not four independent test sets. |
| Selected model | PR35 remains selected under the historical gate | PR137 and PR146 are studied checkpoints; neither is automatically promoted. |
| New captures | Unavailable for now | Existing-data results are developmental; future capture-session validation remains outstanding. |

Sources: [PR146 independent audit](https://github.com/Siuuuers/dl/blob/5001d83439eaf31c7ecae378c27a0de847e65686/docs/partial-classification-r51-results.md),
[selected-model record](https://github.com/Siuuuers/dl/blob/5001d83439eaf31c7ecae378c27a0de847e65686/docs/current-model.md), [model metadata](https://github.com/Siuuuers/dl/blob/5001d83439eaf31c7ecae378c27a0de847e65686/metadata/model_card.json),
[architecture](https://github.com/Siuuuers/dl/blob/5001d83439eaf31c7ecae378c27a0de847e65686/models/cassette_lower_yolo26s_pose.yaml), and
[training contract](https://github.com/Siuuuers/dl/blob/5001d83439eaf31c7ecae378c27a0de847e65686/scripts/pose_checkpoint.py).

### Latest measured tradeoff

The R51 reference uses classification weight 1; its candidate uses weight 2 for
anchors assigned to targets with unavailable corners. All other declared
training ingredients are equal. On R50 after the existing postclip step:

| Measurement | Reference | Candidate |
| --- | ---: | ---: |
| Matched targets / 654 | 639 | 643 |
| Missed targets | 15 | 11 |
| Unmatched outputs | 84 | 112 |
| Formal duplicate outputs | 13 | 23 |
| Other unmatched outputs | 71 | 89 |
| Both corners within 5 px / 583 eligible targets | 566 | 567 |
| Both corners within 10 px / 583 | 582 | 582 |

The four recovered rows accompany 28 additional unmatched outputs. The entire
net 5 px gain comes from one near-threshold case, approximately 5.094 to 4.984
px. This is a measured tradeoff, not a broad improvement. See the
[case ledger](https://github.com/Siuuuers/dl/blob/5001d83439eaf31c7ecae378c27a0de847e65686/results/evaluation/pr146-r51-20261127/case-inventory.json)
for exact identities. An unmatched output is a matching outcome; it is not
automatically a picture of background or a missing annotation.

The new [size and visibility analysis](https://github.com/Siuuuers/dl/blob/5001d83439eaf31c7ecae378c27a0de847e65686/docs/size-visibility-r53.md) replays these
assignments without inference. All 583 fronts with two locatable corners match
in both arms. The remaining misses have one or both corners unavailable.
Five misses occur in the 14 targets with visible-front box heights from 8 to
under 16 original pixels. Six candidate misses occur at heights of at least
16 pixels. Size may matter, but the present evidence does not isolate it from
partial visibility, clipping, appearance and target definition.

## 3. Annotation is a statement about the observed object

Keep [the current annotation policy](https://github.com/Siuuuers/dl/blob/5001d83439eaf31c7ecae378c27a0de847e65686/docs/annotation-policy.md) authoritative:

| Question | Decision |
| --- | --- |
| What should the box enclose? | The observed front and its own continuous frontal rim/tab/cap. Use a tight axis-aligned envelope. A broad receding top plane is not the front. |
| Should rack, gripper or another cassette determine an edge? | No. Foreign material does not establish the target's extrema. Its pixels can still occur inside an axis-aligned enclosing rectangle. |
| Should a shadow be included? | Follow the surface receiving it. A shadowed target surface still belongs to the target; shadow on the rack does not enlarge the cassette. |
| Is a bright junction always the corner? | No. Brightness and shading are useful, lighting-dependent cues. Trace the physical contour in an unmarked context crop. |
| May a small cassette include a neighbor's shadow line? | Small size does not change material ownership. Preserve uncertainty rather than adopting a different target for small instances. |
| What if a corner is unavailable? | Preserve canonical `0 0 0`; do not infer it from symmetry, box intersections or neighboring fronts. |
| What does a neighboring point inside this box mean? | Review it. Perspective, occlusion, the axis-aligned envelope or a wrong neighbor point can produce the contact. It is not proof of an impossible object. |
| How do coordinates move? | X increases right; Y increases down. `ymin` is top, `ymax` bottom. Moving up decreases Y. |

The model receives the full photograph and can use context outside a tight
target box. The contextual benefit studied by [Mottaghi et al.][context] does
not imply that the rack should become part of the cassette label. Visible and
amodal targets are different tasks; [CityPersons][citypersons] illustrates an
explicit separation. Do not silently change our visible-front task into
amodal reconstruction or physical grasp planning.

R52 reviewed 507 rows and proposed 11 boundary changes on 10 rows in one
already-trained photograph, `frame_0160`. The shifts are around 2 pixels with
recorded visual uncertainty of about ±2 pixels. This is useful annotation
maintenance but a weak basis for an expensive claim that another whole
training cycle should improve the model. Preserve held fields and full-photo
completeness requirements. No additional labels are changed by R53 merely to
make a prediction match.

## 4. What the papers motivate, and how to falsify our assumptions

The [literature matrix](https://github.com/Siuuuers/dl/blob/5001d83439eaf31c7ecae378c27a0de847e65686/docs/thesis-literature-r53.md) records the papers' findings,
their limits and our project-specific inferences. The hypotheses below are
not findings about our cassettes.

| Hypothesis | Single comparison | Evidence against it / limitation |
| --- | --- | --- |
| H1: the geometry queue helps reviewers discover genuine label errors | Frozen geometry ranking versus sequential and seeded random order at the same reviewed-case budgets | No better confirmed-issue yield; many legitimate overlaps; time savings cannot be claimed without actual time logs. |
| H2: mild contrast enhancement on Lab lightness helps the frozen checkpoint | Identity versus Lab-L CLAHE, clip limit 2.0, 8×8 tile grid, both at inference size 1280 | No net miss reduction, more unmatched or formal duplicate outputs, worse original-pixel corners, or new empty-scene outputs. Use the frozen666 screen below; a failed inference-only test does not disprove jointly trained enhancement. |
| H3: a higher inference scale helps thin/partial fronts | Identity at 1280 versus 1600, same checkpoint and all other settings | No net miss reduction, worsening duplicate/unmatched outputs, corner regressions or unacceptable runtime/memory. Use the frozen666 screen below. Upscaling adds no new captured detail. |
| H4: accepted training-label corrections improve the unchanged learner | Same image membership, initialization, recipe, seed and fixed checkpoint rule; only accepted labels differ | No useful effect or conflicting metrics. Small corrections near annotation uncertainty may be undetectable. |
| H5: a learned adaptive filter could help | Later, only after H2 evidence and a declared training design | IA-YOLO is jointly trained on different adverse-image tasks; its results do not establish a useful plug-in for this checkpoint. |

H1 follows the general label-review motivation of [ObjectLab][objectlab], but
our neighboring-point checker is a different method. We have not implemented
ObjectLab or measured its performance here. H2 uses the [CLAHE
algorithm][clahe] as one bounded intervention, not a paper-derived optimum.
[IA-YOLO][iayolo] establishes the feasibility of learned image-dependent
enhancement; it does not establish that enhancing every cassette image helps.
[FPN][fpn] and [SAHI][sahi] motivate scale sensitivity while describing
different mechanisms. Increasing the whole-image input size is neither a new
feature pyramid nor tiled inference.

## 5. The two immediate notebooks

Use the [R53 experiment guide](https://github.com/Siuuuers/dl/blob/5001d83439eaf31c7ecae378c27a0de847e65686/docs/inference-ablation-r53.md) for exact operation,
dependencies, runtime checks and returned bundle contents.

| Notebook | Control | Treatment | Training |
| --- | --- | --- | --- |
| [Contrast](https://github.com/Siuuuers/dl/blob/5001d83439eaf31c7ecae378c27a0de847e65686/notebooks/baomai_clahe_inference_r53.ipynb) | Original colour image, 1280 | Same image with fixed Lab-L CLAHE, 1280 | None |
| [Scale](https://github.com/Siuuuers/dl/blob/5001d83439eaf31c7ecae378c27a0de847e65686/notebooks/baomai_scale_inference_r53.ipynb) | Original colour image, 1280 | Same original image, 1600 | None |

Both use the immutable PR137 candidate checkpoint whose saved predictions
equal the PR146 reference. Its checkpoint hash is
`222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9`.
This choice provides a previously verified control; it does not change the
selected PR35 deployment model. The Lab L channel is perceptual lightness, not a new physical luminance
measurement. The notebooks fix image membership, original
image bytes, reference labels, box confidence 0.13, ordinary NMS IoU 0.70,
postclip policy, matching IoU 0.50 and endpoint confidence 0.50. Corner errors
remain in original 1280×720 coordinates.

Each notebook first reruns the unchanged control. It must agree with saved
outputs under explicit numeric tolerances and retain the same discrete match,
miss, duplicate and corner-threshold outcomes on all four references. A failed
control gate produces diagnostics and stops treatment interpretation. Changing
the environment and the image operation simultaneously would otherwise make
the result uninterpretable.

Report paired recovered and newly missed target IDs, unmatched and formal
duplicate counts, two/one/zero-corner groups, corner distances and 5/10 px
success, empty-scene outputs, preprocessing and inference latency, and memory.
Compare the contrast and scale treatments with their own controls; do not
combine them yet. The latency measurement is software timing on the recorded
device, not a camera-to-actuator deadline measurement.

**Decision rule for a promising low-complexity treatment:** on the primary
frozen666 reference, fewer misses with no increase in unmatched outputs or
formal duplicate outputs, no new two-corner misses, no loss of strict
all-eligible corner successes, and no new outputs in the three empty scenes is
a strong developmental result worth confirming. Otherwise report the actual
tradeoff. R50, R38 and historical663 provide diagnostic and reference-sensitivity
views; they do not replace the primary reference when another view looks better.
A Pareto-style rule avoids inventing an arbitrary cost for one miss versus one
unmatched proposal before the application requirements are known. Report all
four reference versions and all case changes even if this simple screen passes.
The existing selection gate still applies; R53 does not promote a model or
preprocessor automatically.

If a contrast treatment looks better to a human but fails detection metrics,
keep it out of deployment. If it helps only under inference size 1600, that is
a later interaction hypothesis, not evidence available from these two separate
tests. If a run fails the control gate, repair the runtime reproducibility
problem before drawing a filter conclusion.

## 6. Camera filters, model filters and retraining

The [camera pipeline audit](https://github.com/Siuuuers/dl/blob/5001d83439eaf31c7ecae378c27a0de847e65686/docs/camera-pipeline-r53.md) distinguishes four things:

1. The D405 produces colour through its onboard image signal processor.
   Repository “raw” colour means overlay-free SDK colour, not untouched sensor
   samples. Physical exposure/white-balance settings were not recorded in the
   inspected application, so we cannot reconstruct them from these PNGs.
2. The live viewer enables a BGR colour stream and passes it directly to
   `model.predict`. It does not currently call CLAHE, an RGB denoiser or the
   RealSense depth-filter chain.
3. RealSense spatial/temporal/hole-filling filters process depth data. They are
   not RGB contrast filters and are not active in this viewer's colour path.
4. Confidence thresholds, ordinary NMS and the optional folder postclip step
   filter predicted detections after image/model processing. They do not
   enhance the photograph.

Software can apply a fixed preprocessor immediately before inference. A
learned controller can also predict filter parameters, as in IA-YOLO, but it
needs its own trained and validated design; the present pose network does not
spontaneously call a filter. R53 applies the fixed filter only inside the
experiment runner. Identity remains the application default.

There is a concrete configuration discrepancy to resolve before future camera
comparisons: `start_live.sh` and `weights/base.json` select epoch 173, whereas
`docs/current-model.md` retains PR35. Explicit `--weights` alone does not
necessarily supply the intended confidence threshold. Record the effective
checkpoint hash and all thresholds at launch. Neither a GitHub PR nor a saved
review image proves which model is running on a physical camera.

### Should we change model type now?

Keep the current pose contract for these two experiments. The inspected model
already has multiscale feature heads. A detector without keypoints drops a
required output; instance segmentation would require masks or contours and a
new evaluation task. A different pose architecture needs a compatible label
conversion, unavailable-point handling, matching rules and equal training
budget before its result can be interpreted. These are legitimate later
comparisons if the simpler tests expose a persistent limitation.

### Should we start from scratch?

There are three distinct actions:

| Action | Meaning | Current decision |
| --- | --- | --- |
| Resume a run | Continue optimizer/training state from an interruption | Appropriate only for completing that same declared run. |
| Start a fresh paired fine-tuning run | Reset optimizer/seed state and load the same learned epoch 173 weights for both arms | Appropriate when a useful training hypothesis is ready. |
| Random initialization | Construct the architecture without loading learned weights | Not the immediate experiment on 142 related photographs. |

[He, Girshick and Dollár][scratch] showed that sufficient training can let
random initialization catch up in several COCO regimes, but their small-data
and VOC experiments also show breakdowns. That paper does not say that
pretraining never matters or that a brief random-start run on our data is
preferable. A future equal-30-epoch comparison would answer a budget-limited
practical question, not establish the best attainable quality of either
initialization. Available metadata establish our learned cassette initializer,
not its full public-pretraining genealogy.

## 7. Make the annotation study measurable

Freeze a predefined set of full images and the checker version before scoring
review-queue quality. User-nominated problem rows are useful examples but a
selected sample. Exhaustively adjudicate a small image subset where feasible;
otherwise review flagged cases plus a prespecified sampled set of unflagged
cases and record selection probabilities. A geometry rule cannot discover an
entirely missing instance with no existing label, so full-frame inspection is
essential.

Record one decision per stable instance ID: genuine label issue, valid
overlap, wrong neighbor point, uncertain ownership, or other reason. Preserve
original pixels, before/after coordinates, label hashes, reviewer identity and
actual active review time. An agent suggestion is not independent human
adjudication. Repeated judgments by one person measure repeatability; two
independent reviewers are needed to claim inter-annotator agreement.

Predeclare budgets such as 25/50/100 unique cases, capped by pool size. Report
confirmed issues at each budget and precision among reviewed flags. Issue
recall requires an exhaustive or justified sampled denominator. Flag count
reduction alone is not proof of better labels, and case count alone is not a
time-saving measurement. Out-of-sample model predictions are required for a
fair model-disagreement review benchmark; ObjectLab uses cross-validation.
Keep related captures together if this is added later.

When accepted corrections are sufficient to justify training, freeze baseline
and corrected training labels on identical image membership. Hold the model,
learned initializer, optimizer, schedule, augmentations, inference, matching
and checkpoint rule fixed. Distinguish instance additions/removals from small
coordinate refinements. Train both arms from the same starting checkpoint;
do not continue only the candidate for extra epochs.

Score both models against both the historical and independently adjudicated
reference. This produces two models and four scoring combinations. The
training effect is the difference between models within one reference; the
reference effect is the difference between label versions for an unchanged
model. A one-seed result is exploratory; repeat an informative pair with a
predeclared additional seed before a strong stability claim. Keep future
capture sessions separate from all development decisions.

## 8. Planned thesis structure and evidence register

| Chapter | Material | What remains |
| --- | --- | --- |
| Problem and target | Visible fronts, corner availability, application and data constraints | Supervisor scope and operational output requirements |
| Related work | Annotation quality, diagnostic evaluation, scale, enhancement, transfer learning | Keep citations tied to exact claims, not a long unrelated survey |
| Data and annotation method | Original image lineage, policy, full-photo holds, stable row mapping | Prospective adjudication and reviewer-time records |
| Experiments | Fixed baseline, queue comparison, R53 probes, later label-only training if justified | Execute the inference notebooks and record outcomes |
| Results and failure analysis | Paired identities, misses/unmatched/duplicates, corner strata, runtime | Negative findings and uncertainty belong in the same tables |
| Discussion | Correlated captures, repeated development use, boundary uncertainty, capture drift | Future independent-session validation |

### Decision log

| Date | Decision | Reason/evidence |
| --- | --- | --- |
| 2026-11-27 | Retain visible-front labels and physical ownership policy | Tight target definition is compatible with full-image context. |
| 2026-11-27 | Keep PR35 selected | PR146 recovers 4 R50 targets but adds 28 unmatched outputs; historical gate not passed. |
| 2026-11-27 | Prepare two separate inference probes | Fixed checkpoint and one change per notebook make the results attributable. |
| 2026-11-27 | Defer random initialization and architecture replacement | Existing pose output is required; small-data transfer evidence does not justify discarding learned weights first. |
| 2026-11-27 | Keep R52 corrections versioned; defer another tiny-label retrain | 10 rows on one image, roughly 2 px changes near recorded uncertainty. |
| 2026-11-27 | Audit live configuration before a future camera comparison | Launcher epoch 173 and selected PR35 differ; physical camera options unrecorded. |

### Questions that materially affect later architecture

- Must the application locate every identifiable visible front, or only fronts
  with two observable corners suitable for a later operation? The present
  dataset includes partial fronts; a downstream eligibility decision should
  not silently redefine training labels.
- What sustained throughput, latency, hardware and memory budget must the
  final application meet? Until known, report measurements instead of claiming
  that 1600 or an additional filter is affordable.
- Does the supervisor accept a controlled annotation-quality application
  study, or require a new method? That changes the thesis scope, not the need
  to validate the existing data and baseline.

## 9. References and reading order

Start with ObjectLab and TIDE for the review/diagnosis question, then the
annotation-quality and noise-aware papers for experimental controls. Read
CLAHE and IA-YOLO for the filter mechanism, FPN/SAHI for scale, and He et al.
for initialization. The [detailed literature matrix](https://github.com/Siuuuers/dl/blob/5001d83439eaf31c7ecae378c27a0de847e65686/docs/thesis-literature-r53.md)
contains bibliographic metadata, applicability limits and proposed tests.

- [ObjectLab: Automated Diagnosis of Mislabeled Images in Object Detection Data][objectlab].
- [TIDE: A General Toolbox for Identifying Object Detection Errors][tide].
- [The Effect of Improving Annotation Quality on Object Detection Datasets][ma].
- [Noise-Aware Evaluation of Object Detectors][noise].
- [Contrast Limited Adaptive Histogram Equalization][clahe].
- [Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions][iayolo].
- [Feature Pyramid Networks for Object Detection][fpn].
- [Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection][sahi].
- [Rethinking ImageNet Pre-training][scratch].
- [The Role of Context for Object Detection and Semantic Segmentation in the Wild][context].
- [CityPersons: A Diverse Dataset for Pedestrian Detection][citypersons].

[objectlab]: https://arxiv.org/abs/2309.00832
[tide]: https://arxiv.org/abs/2008.08115
[ma]: https://openaccess.thecvf.com/content/CVPR2022W/VDU/papers/Ma_The_Effect_of_Improving_Annotation_Quality_on_Object_Detection_Datasets_CVPRW_2022_paper.pdf
[noise]: https://openaccess.thecvf.com/content/WACV2025/html/Llerena_Noise-Aware_Evaluation_of_Object_Detectors_WACV_2025_paper.html
[clahe]: https://doi.org/10.1016/B978-0-12-336156-1.50061-6
[iayolo]: https://arxiv.org/abs/2112.08088
[fpn]: https://arxiv.org/abs/1612.03144
[sahi]: https://arxiv.org/abs/2202.06934
[scratch]: https://arxiv.org/abs/1811.08883
[context]: https://cs.stanford.edu/~roozbeh/pascal-context/mottaghi_et_al_cvpr14.pdf
[citypersons]: https://arxiv.org/abs/1702.05693
