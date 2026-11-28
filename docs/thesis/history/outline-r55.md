> Historical snapshot. The current report and completed PR149/PR150 interpretation are in [the thesis index](../README.md). Statements about pending work below describe the original revision.

# Cassette Front Detection and Corner Localization for Automatic Image Analysis

## R55 thesis blueprint

27 November 2026 · Asia/Hong_Kong

This is a writing and evaluation blueprint, with a provisional abstract and completed evidence. It is not a finished thesis. R53 inference results remain pending. The user has confirmed that the application consumes outputs automatically, that baomai should determine the target semantics, and that the research contribution may be chosen freely within the supplied report structure.

**Working research question:** How can annotation quality, controlled inference interventions, and explicit output selection improve cassette front detection and corner localization while making the evidence adequate for an automatic consumer?

The thesis will be an empirical computer-vision study. Its contribution can consist of an explicit task definition, reproducible comparisons, an error analysis, and a measured account of when the system provides supported outputs. A new neural architecture is an optional experiment, not an assumed requirement. The supplied headings do not establish any additional institutional rubric, word limit, citation style, or requirement for a novel network.

## Report requirements and chapter map

The starred items below are compulsory in the user's supplied requirements. All other headings are included because existing project material fits them. Page numbers belong in the final document after pagination.

| Order | Exact heading | Required? | Planned content |
| --- | --- | --- | --- |
| a | Abstract | Yes; 150–300 words | Problem, methods, measured findings, limits, contribution |
| b | Acknowledgements | Yes | Actual contributors and support, with accurate names and roles |
| c | Table of Contents | Yes | Generated from final headings and pagination |
| d | Introduction | Yes | Application, problem, questions, scope, contributions |
| e | Background | Where appropriate | Detection, two-point pose, annotation uncertainty, error decomposition, selective output |
| f | Methodology & Implementation | Where appropriate | Data provenance, target policy, model, controls, metrics, output contract, reproducibility |
| g | Simulation Results | Where appropriate | Existing controlled digital occlusion experiment, clearly identified as image perturbation |
| h | Experimental Results | Where appropriate | Completed training and inference analyses, including R54/R55; R53 when returned |
| i | Discussions | Yes | Interpretation, tradeoffs, uncertainty, reference sensitivity, generalization limits |
| j | Conclusions | Yes | Answers supported by completed evidence and bounded future work |
| k | References | Yes | Primary papers, upstream documentation, and reproducible project evidence |
| l | Appendices | Where appropriate | Protocols, manifests, full tables, case identities, label decisions, reproduction commands |

## Abstract

*Provisional draft; update the final findings after R53. The abstract below contains 222 words under whitespace counting.*

Automatic analysis of cassette images requires detection of identifiable fronts and localization of their actual lower corners, including scenes with partial visibility and tightly packed objects. This study investigates how annotation policy, controlled model comparisons, and output selection affect these tasks. A single-class, two-keypoint pose pipeline is evaluated using versioned annotations and reproducible prediction records. The latest paired training comparison uses 142 photographs containing 3,999 annotated fronts; evaluation uses four reference versions of the same 20 development photographs.

The analysis distinguishes missed targets, unmatched predictions, duplicate predictions, and localization errors. It also separates front detection from the decision to offer a corner pair for automatic consumption. A saved-output replay shows that confidence selection changes the detection tradeoff without retraining. A subsequent endpoint-screen analysis preserves 567 reference-supported pairs within five pixels while reducing offered pairs from 709 to 650 under one reviewed reference. The primary frozen reference gives 511 supported pairs for those same output counts, demonstrating sensitivity to annotation policy. Controlled digital occlusion further shows that high endpoint scores can accompany substantial localization errors.

The work provides a reproducible evaluation framework and evidence for targeted improvements without assuming that confidence establishes physical accuracy. Results remain exploratory because thresholds were examined on repeatedly used development images. Independent capture evaluation and application-specific tolerances are required before claiming reliable performance in an automatic downstream task.

## Acknowledgements

Use factual acknowledgements rather than inventing people, institutions, funding, or assistance. A suitable opening, subject to the author's final review, is:

> I thank the contributors to the embedding_vision project for making the cassette images, annotations, models, and supporting tools available for this study.

Add the supervisor and other contributors only with their actual names and roles. Follow the institution's applicable disclosure rules for software or AI assistance; no particular institutional policy has been supplied here.

## Table of Contents

Use the exact chapter headings in the requirements table. Suggested subsections are provided below. Generate page numbers from the final formatted document rather than adding provisional page numbers to this Markdown blueprint.

## Introduction

### Problem and motivation

The application analyzes camera images of cassette fronts. Its outputs are consumed automatically. Consequently, a low miss count alone does not describe the usefulness of the final result: a duplicate can produce an extra item, an unmatched output can be unsupported by the reference, and an inaccurate corner pair can survive a successful box match. Which consequence matters most depends on what the downstream operation consumes. The current scope supports both front detection and optional corner coordinates without assuming a robot, three-dimensional reconstruction, or a clinical decision.

### Questions

1. How do partial visibility, box extent, annotation uncertainty, and confidence selection contribute to missed and unmatched fronts?
2. Which controlled interventions improve the measured detection/localization tradeoff under fixed references?
3. How does screening proposed corner pairs change their reference-supported quality and coverage?

Annotation review efficiency is a further testable question if actual review budgets, controls, and time logs are collected. The thesis must not claim review time savings merely because a review queue was generated.

### Proposed contributions

- An explicit, baomai-aligned definition of the front and its two actual lower-corner slots, preserving unavailable annotations.
- Reproducible comparisons that separate data, inference settings, confidence thresholds, and reference revisions.
- A case-level account of misses, duplicates, other unmatched outputs, and corner regressions.
- An output evaluation that includes every offered pair, rather than reporting localization only for convenient matched, fully annotated cases.

These are proposed contribution categories. Their eventual strength depends on the completed experiments; they do not assert architectural novelty or independent generalization.

## Background

### Physical target and annotation semantics

Baomai specifies one `cassette_front` class and two actual lower-front corners, ordered by image X. The points are not the upper lip or tray corners. A visible, reliably locatable point uses `v=2`; an unavailable point remains `0 0 0`, with no coordinate guessed from an occlusion boundary. One available endpoint is retained even when the other is unavailable. [1]

The upstream text does not completely define the box extent or a minimum visible fraction. The adopted project policy therefore retains an identifiable partial front and its tight observed extent, including its own frontal rim/tab/cap. Foreign material does not determine the target boundary, although an enclosing rectangle may contain occluder pixels. A fully hidden front must not be invented from regular stack spacing alone. [3]

An unavailable annotation and a hidden-point prediction are different. The existing project permits hidden-point estimates as predictions; these do not become observed training coordinates or verified physical locations merely because the model gives a high score. [3, 4]

### Evaluation concepts

Explain one-to-one matching, IoU, detection precision/recall, and keypoint error in original-image pixels. Duplicates form a subset of unmatched outputs. A poorly aligned thin box can produce both a miss and an unmatched prediction. TIDE motivates cause-specific error analysis, but the project's fixed-threshold scorer is not an implementation of the full TIDE oracle-AP procedure. [5]

Noise-aware evaluation provides context for sensitivity to uncertain box boundaries. It does not justify moving a reference toward the prediction of a favored model. ObjectLab motivates model-assisted review and the use of predictions obtained without training on the image being reviewed. [6, 7]

### Confidence and selective output

Guo et al. study confidence calibration for classification models. Their findings motivate testing score meanings rather than treating scores as probabilities, but temperature scaling from that paper does not by itself calibrate cassette keypoint error. [8]

Geifman and El-Yaniv study the tradeoff between making fewer predictions and reducing classification risk. This motivates reporting quality together with coverage. Their statistical guarantees do not transfer automatically to our pose task, incomplete coordinates, and repeatedly used development images. [9]

## Methodology & Implementation

### Data and provenance

Document source photographs, split membership, annotation versions, training initialization, and every evaluation artifact hash. The latest R51 paired experiment uses the same 142 full photographs and 3,999 fronts in both arms. Both arms fine-tune from the learned epoch173 cassette checkpoint; this is not random initialization. [10]

All four evaluation references describe the same 20 development photographs. They are sensitivity views of the same images, not four independent datasets. R50 has 654 fronts, including 583 with two locatable corners, 54 with one, and 17 with none. The primary frozen666 reference has 584 two-corner fronts. Keep counts and row identities reference-specific. [10, 11]

These development photographs have been examined repeatedly. New independent captures are currently unavailable. Accordingly, report development findings transparently and reserve independent generalization claims for future capture-session evaluation. A single seed or many predictions from a few related images do not provide independent replication.

### Controlled interventions

| Study | Controlled comparison | Current status |
| --- | --- | --- |
| R49 / PR143 | Ordinary NMS versus the declared linear Soft-NMS on the same decoded outputs | Completed; no missed target recovered |
| R51 / PR146 | Classification weight 1 versus 2 for anchors assigned to partial targets; other declared training ingredients equal | Completed; detection tradeoff observed |
| R53 contrast | Original image versus fixed Lab-L CLAHE, both at 1280 | Awaiting notebook return |
| R53 scale | Whole-image 1280 versus 1600, with original-pixel evaluation | Awaiting notebook return; not tiled inference |
| R54 | Confidence operating points replayed from fixed postclip predictions | Completed, exploratory |
| R55 | Endpoint thresholds applied to candidate pairs from the same saved outputs | Completed, exploratory |

Keep each intervention separate before considering combinations. Higher input resolution introduces no new captured information. A negative CLAHE result would concern that transform and checkpoint, not all learned image enhancement. [10–12]

### Minimal output contract

Use one detection model with an explicit output record. Front presence and coordinate-pair availability are separate decisions.

| Part | Meaning | Required behavior |
| --- | --- | --- |
| Front proposal | Box, score, and class returned for an identifiable front candidate | Retain independently of corner-pair screening; evaluate presence/count errors separately |
| Raw landmarks | Two coordinate proposals and endpoint scores | Preserve for audit, including estimates of hidden points; do not call the scores physical visibility |
| Pair status | `pair_candidate` or `withheld` under a named, versioned rule | The consumer must branch on status; a withheld pair is an expected outcome |
| Pair payload | Two coordinates when the rule offers a candidate, otherwise an explicit absence | Never use `(0, 0)` as an automatic-action substitute for missing coordinates |
| Provenance | Original image dimensions, coordinate system, model identity, and rule version | Prevent display offsets, resize mappings, or changed settings from silently altering the consumer |

`pair_candidate` means that the pair passes the declared screen. It does not mean physically verified or sufficiently accurate for an unspecified operation. Individual point proposals remain accessible if a later consumer needs only one corner. The zero/one/two-corner annotation strata are evaluation information; the runtime must not use unavailable ground-truth visibility as an oracle.

The inspected viewer applies the endpoint threshold when drawing points. Its snapshot JSON retains the entire raw `detections` array and stores display information separately. An automatic consumer reading the JSON therefore cannot assume that undrawn coordinates were removed. The minimal implementation change, when integrating the consumer, is an explicit serialization rule with an abstention state. No additional neural network is needed to express that contract. [2]

This blueprint specifies the behavior but does not alter the running viewer, checkpoint, or thresholds. Application-specific numerical tolerances and the consequences of a wrong output remain to be defined before approving a production gate. Those details do not prevent the current thesis experiments.

### Detection metrics

Use the existing confidence-ordered, same-class matching at IoU at least 0.50. Report missed fronts, unmatched predictions, the duplicate subset, and matched localization errors. Include empty scenes and stratify by reference corner availability. Keep exact case identities for gains and regressions. Continue original-pixel errors and both-corner success at the established 5- and 10-pixel diagnostic tolerances. [11]

### R55 pair metrics

The simulated screen keeps boxes with confidence strictly above the stated box cutoff and offers a pair only when both points are finite, lie inside the original 1280×720 image, and each meets the inclusive endpoint cutoff. Recompute the original matcher on these offered pairs against **all** reference fronts. [13]

Let A be the number of offered pairs, G2 the number of reference fronts with two locatable corners, and Sε the number of offered pairs assigned to those fronts with both corner errors at most ε pixels. Then report:

\[
\text{reference-supported pair fraction}=S_\epsilon/A,
\qquad
\text{eligible-pair recall}=S_\epsilon/G_2.
\]

Also report the pair offering rate, A divided by all retained front proposals. The first fraction is undefined when A is zero.

For each tolerance, partition A into supported successes, measured localization failures on fully locatable references, matches to references without a complete pair, and unmatched outputs. The last two groups are unverified under this criterion; they are not all proven physical mistakes. A one-corner reference can still supply evidence about its available endpoint even though it cannot fully verify the pair. [13]

### Verification

R55 reproduces the eight original model/reference baseline tables before its 72-setting analysis. It verifies that the four outcome groups partition every image's offered outputs. An independent recalculation using a separately implemented greedy matcher verified the six highlighted candidate settings and their 120 image-level results. This checks the arithmetic and matching implementation, not external validity. [13]

## Simulation Results

### Controlled digital occlusion

Use the completed pixel-occlusion study as an image-level simulation. Thirty fixed targets in four familiar development photographs were tested intact and with masks over the left corner, right corner, or center face. Known coordinates from the intact image remain the evaluation reference after masking. The model receives only the altered image. This differs from naturally hidden corners, whose accurate coordinates are unknown. [4]

| Condition | Designated targets matched | Both corners within 5 px | Both corners within 10 px |
| --- | ---: | ---: | ---: |
| Intact | 30/30 | 27/30 | 30/30 |
| Left corner masked | 30/30 | 21/30 | 28/30 |
| Right corner masked | 30/30 | 22/30 | 26/30 |
| Center face masked | 30/30 | 22/30 | 30/30 |

All 60 covered-endpoint scores exceeded 0.85, while six covered-endpoint observations exceeded 10 pixels of error across five objects. For one example, a score of approximately 0.954 accompanied a 14.93-pixel error. These are direct counterexamples to equating a high endpoint score with a small localization error in this experiment. [4]

The center mask also harms strict pair accuracy, so side-mask effects cannot be assigned solely to removal of the local corner pixels. Masks sometimes alter neighboring targets. Report the full-image effects and paired target identities from the source study. Gray rectangles, few familiar source images, and known intact coordinates limit transfer to natural occlusion. This chapter does not claim a physical simulator or independent deployment test. [4]

## Experimental Results

### Completed training and detection evidence

On R50 at the original 0.13 box cutoff, the R51 reference/candidate have 15/11 misses, 84/112 unmatched outputs, and 13/23 formal duplicates. The corresponding both-corner 5-pixel successes are 566/583 and 567/583. The four recovered fronts therefore accompany 28 additional unmatched outputs; the single strict corner gain is a near-threshold crossing. [10]

R54 replays the candidate at the exploratory box cutoff 0.20: R50 gives 12 misses, 80 unmatched outputs, and 11 duplicates. The primary frozen666 result is 33 misses, 89 unmatched, and 15 duplicates. Fair comparisons also search the reference model's available confidence frontier. These are operating-point findings, not newly trained models or independent confirmation of an optimum. PR35 remains selected under the historical frozen-reference criteria. [12]

### Automatic pair-output diagnostic

The following rows use the R51 candidate at box confidence strictly above 0.20. Both endpoint scores must meet the listed inclusive cutoff. [13]

| Reference | Endpoint cutoff | Offered pairs | Supported pairs ≤5 px | Supported fraction ≤5 px | Eligible-pair recall ≤5 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen666, primary | 0.50 | 709 | 511 | 72.07% | 87.50% |
| Frozen666, primary | 0.80 | 650 | 511 | 78.62% | 87.50% |
| Frozen666, primary | 0.95 | 116 | 92 | 79.31% | 15.75% |
| R50 | 0.50 | 709 | 567 | 79.97% | 97.26% |
| R50 | 0.80 | 650 | 567 | 87.23% | 97.26% |
| R50 | 0.95 | 116 | 111 | 95.69% | 19.04% |

From 0.50 to 0.80, neither reference loses or gains any previously supported target identity at 5 or 10 pixels. Fifty-nine offered pairs disappear, but measured localization failures also remain unchanged: 66 on frozen666 and 16 on R50 at 5 pixels. The improvement in supported fraction therefore comes from removing reference-unverified outputs, not moving retained coordinates closer to their references. [13]

At R50 cutoff 0.50, the 709 offered pairs partition into 567 supported successes, 16 measured errors over 5 pixels, 53 matches to incomplete reference pairs, and 73 unmatched outputs. Do not call the latter two groups 126 proven physical errors. At cutoff 0.95, supported coverage collapses; at 0.99, no pairs are offered and the supported fraction is undefined. [13]

Pair screening leaves all 722 retained front proposals in the presence channel. It does not solve automatic counting errors or remove all duplicate boxes. These must retain separate detection evaluations. The score cutoffs are post-hoc development probes and are not approved runtime settings. [13]

### Pending R53 results

Insert each notebook return only after its unchanged control reproduces the expected baseline. Use separate contrast and scale tables, all four reference names, original-pixel corners, empty-scene outputs, and case identities. If an experiment has no useful effect, report that outcome with its scope. Do not fill this subsection with expected improvements.

## Discussions

### Priority and interpretation

The next improvement target is identifiable partial-front detection, subject to constraints on unsupported front outputs and corner quality. Under R50, both R51 arms already detect all 583 two-corner fronts; the remaining misses have zero or one locatable corner. Recovering such a front can improve presence/counting while adding no fully verifiable corner pair. That distinction explains why the thesis must report both tasks. [10, 11]

For automatic consumption, unmatched and duplicate outputs remain consequential. Review their causes before choosing remedies: extent/ownership problems, repeated detections, confirmed background responses, and policy-sensitive cases call for different changes. A stronger NMS cutoff is not a universal solution. R49's tested Soft-NMS recovered no missed target. [10]

### Threats to validity

- **Repeated development use:** threshold exploration and many prior experiments preclude treating these images as a fresh test set.
- **Reference sensitivity:** four annotation versions do not create independent replication; primary and reviewed-reference outcomes can differ substantially.
- **Incomplete coordinates:** naturally unavailable points do not support a full physical error measurement. Synthetic masking supplies known source coordinates but is not equivalent to natural occlusion.
- **Conditional reporting:** errors only among matched, fully annotated targets can conceal misses, withheld outputs, duplicates, and unsupported emitted pairs.
- **Aggregate cancellation:** an unchanged success count can hide a lost successful target offset by a new one. Retain paired identities.
- **Annotation uncertainty:** approximately two-pixel boundary changes may be comparable to visual uncertainty; tiny threshold crossings need cautious interpretation.
- **Downstream ambiguity:** pixel thresholds are diagnostic. The required tolerance and value of a front-only output depend on the actual automatic operation.

These limits constrain the claims; they do not erase the usefulness of reproducible negative and positive results. A thesis can explain why an apparently helpful change fails under a stronger evaluation, provided the question and evidence remain clear.

### Minimal next experiments

1. Finish the already-running R53 controls and comparisons.
2. Reuse the declared R55 pair-output metrics when comparing R53 outputs; keep the original thresholds fixed for the primary comparison.
3. Investigate the smallest supported failure cause. Use training-side annotation review when geometry is wrong, a bounded inference probe when scale is implicated, or a fair capacity comparison only when residual evidence supports it.
4. Train another paired comparison only for a meaningful accepted correction set or a distinct hypothesis. Preserve membership for a label-only study and labels for a sampling-only study.
5. Predeclare an additional seed for an informative result. Add independent capture-session evaluation when data become available.

Do not launch a new large training notebook solely to fill the report structure. The completed R55 analysis needs only Python's standard library and cached predictions.

## Conclusions

The final conclusion should answer the three research questions with completed results. Current evidence supports three bounded statements: confidence selection changes the detection tradeoff; complete pair-output evaluation reveals limitations concealed by target-only corner metrics; and high endpoint scores do not establish physical accuracy. [4, 12, 13]

No completed evidence in this blueprint establishes a universally superior replacement for PR35, an optimal deployment threshold, or independently verified automatic operation. Final conclusions must incorporate R53 only after its actual results arrive. Future work should follow observed residual errors and unavailable validation conditions.

## References

1. an94-doll. *embedding_vision*, baomai data README, commit `351eb3cf960ccc34829e2bc72e1fa4645acc6295`. [Target definitions and data provenance](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/README.md).
2. an94-doll. *embedding_vision*, same commit, [scripts/view_d405_yolo.py](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/scripts/view_d405_yolo.py). `render` and `save_snapshot` were inspected separately. Local `Siuuuers/dl` uses clearer display terminology but retains raw snapshot coordinates.
3. Siuuuers/dl. [Annotation policy](https://github.com/Siuuuers/dl/blob/5001d83439eaf31c7ecae378c27a0de847e65686/docs/annotation-policy.md) and [visible and hidden point decisions](https://github.com/Siuuuers/dl/blob/5001d83439eaf31c7ecae378c27a0de847e65686/docs/visible-and-hidden-2026-11-14.md). Inspected at local project commit `5001d83439eaf31c7ecae378c27a0de847e65686` on 27 November 2026.
4. Siuuuers/dl. [Pixel occlusion study](https://github.com/Siuuuers/dl/blob/5001d83439eaf31c7ecae378c27a0de847e65686/docs/pixel-occlusion-2026-11-13.md). Protocol/runner frozen at [commit 3647d0937ab036974cda4867815a8fe20aef07d7](https://github.com/Siuuuers/dl/commit/3647d0937ab036974cda4867815a8fe20aef07d7).
5. Bolya, D., Foley, S., Hays, J., and Hoffman, J. (2020). [TIDE: A General Toolbox for Identifying Object Detection Errors](https://arxiv.org/abs/2008.08115). ECCV.
6. Murrugarra-Llerena, J., and Jung, C. R. (2025). [Noise-Aware Evaluation of Object Detectors](https://openaccess.thecvf.com/content/WACV2025/html/Llerena_Noise-Aware_Evaluation_of_Object_Detectors_WACV_2025_paper.html). WACV, 9304–9313.
7. Tkachenko, U., Thyagarajan, A., and Mueller, J. (2023). [ObjectLab: Automated Diagnosis of Mislabeled Images in Object Detection Data](https://arxiv.org/abs/2309.00832). ICML Data-centric Machine Learning Research Workshop.
8. Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). ICML, PMLR 70, 1321–1330.
9. Geifman, Y., and El-Yaniv, R. (2017). [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500). NeurIPS.
10. Siuuuers/dl. [PR143](https://github.com/Siuuuers/dl/pull/143), [PR146](https://github.com/Siuuuers/dl/pull/146), and local `docs/partial-classification-r51-results.md`, `docs/front-diagnosis-r51.md`.
11. Siuuuers/dl. Local `scripts/pose_scoring.py`, `docs/current-model.md`, and `docs/size-visibility-r53.md`. Original scorer SHA-256: `04e5cdbef26df978707d0b6e4b82f05471b6012845f7986b9c05c71644ec0018`.
12. R54 cached confidence-frontier analysis. `thesis-next-steps-r54.md` and the immutable `r54-cached-frontier.zip`; verified saved-input snapshot SHA-256: `b6059a3810e29ba3afb245f1585704521c2633d8caca6fddd908be015e249125`.
13. R55 automatic pair-output diagnostic. `r55-automatic-output-audit.zip`: analysis script, original scorer, saved inputs, machine-readable results, and case identities. No new inference or training. The companion report documents all highlighted fractions and their denominators.

## Appendices

### A. Required reproducibility material

Include split membership, reference hashes, model hashes, exact recipes, initialization, seeds, checkpoint-selection rules, environment versions, coordinate transforms, and score inequalities. Preserve the scorer and cached inputs needed for every reported replay. Record control reproduction before treatment comparisons.

### B. Case and annotation ledgers

Keep missed/recovered target identities, duplicate assignments, unmatched review categories, corner success gains/losses, and unavailable-reference cases. Preserve uncertain annotations without treating model agreement as physical proof. Include full-image checks for missing labels, since checks on existing rows cannot discover every entirely missing object.

### C. R55 reproduction

Extract `r55-automatic-output-audit.zip`. From its `r55-analysis` folder, run:

```bash
python -B analyze_automatic_pairs.py --inputs ../r54-analysis/inputs.json --scorer ../r54-analysis/pose_scoring.py --output replay
```

The replay needs Python 3 and its standard library. It does not require images, model weights, a GPU, network access, or another training notebook. Hash checks establish input identity; baseline checks establish agreement with the original scorer. The original cache is censored below box confidence 0.13 and cannot establish full AP.

### D. Decisions resolved on 27 November 2026

| Question | Decision |
| --- | --- |
| Which source defines the task? | Baomai's actual lower-front corner contract, supplemented by the existing documented visible-front policy where upstream is silent |
| What about an identifiable partial front? | Retain its front proposal and independently represented point proposals; do not require two locatable annotation corners to include it |
| How is output consumed? | Automatically; therefore serialize explicit pair status and evaluate emitted outputs and coverage |
| Must the thesis introduce a new architecture? | No such requirement was supplied; choose a controlled empirical study |
| Must every listed report section be invented? | No; starred sections are compulsory, others apply where appropriate. Existing digital occlusion material can populate Simulation Results |
| Are latency and hardware optimization goals? | No; prioritize recognition quality and simple implementation |
