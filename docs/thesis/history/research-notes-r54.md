> Historical snapshot. The current report and completed PR149/PR150 interpretation are in [the thesis index](../README.md). Statements about pending work below describe the original revision.

# Cassette perception thesis: next steps after R53

27 November 2026 · Asia/Hong_Kong · updated with the user's automatic-use decision and R55 audit

## Recommendation

Prioritize the detection of identifiable partial fronts, while protecting the rate of unmatched outputs, duplicate outputs, and actual corner accuracy. The user has now confirmed automatic consumption, so evaluate the quality and coverage of emitted outputs as a first-class outcome. Keep the existing one-front/two-corner pose contract and distinguish front presence, raw corner proposals, and whether a coordinate pair passes the declared consumer screen. The present evidence supports a focused annotation and evaluation study before another architecture change.

The most useful new result is a CPU replay of the completed PR146 experiment. At an exploratory candidate confidence cutoff of 0.20, the reviewed R50 comparison improves from 15 misses and 84 unmatched outputs to 12 misses and 80 unmatched outputs. This same cutoff gives a smaller improvement on the primary frozen666 reference, from 34/91 to 33/89. No previously matched target becomes missed at this cutoff. However, a primary-reference corner regression remains, and the candidate does not supersede selected PR35 under the established frozen-reference criteria.

This is development evidence from saved predictions. It is not a new training result, a complete AP benchmark, or independent confirmation of an optimal threshold. The two already-running R53 notebooks still have their original purpose and controls. The new R55 audit, described below, examines complete pair outputs from the same verified cache. A companion thesis-outline-r55.md maps the confirmed report requirements to existing evidence and includes a 222-word provisional abstract.

## 1. Current project state

The local canonical thesis and model-status records establish the following sequence. PR #143 returns the R49 inference diagnosis. PR #146 returns completed R51 paired training. R52 contains subsequent annotation review and a CPU report workflow. R53 prepares separate contrast and scale inference probes. Thus the PR number, annotation revision, experiment revision, and selected checkpoint are distinct identifiers.

| Item | Verified state |
| --- | --- |
| Target | One identifiable visible cassette front, including its own frontal rim/tab/cap, with two actual lower-front corner slots ordered by image X |
| Missing annotation corner | Unavailable annotation slots remain 0 0 0; no guessed hidden coordinates in the labels |
| Predicted hidden corners | May remain identifiable model estimates; annotation unavailability does not prove their coordinates wrong |
| Output consumption | Automatic; front proposals and optional corner-pair output require explicit, separate semantics |
| Latest completed paired training | PR146/R51, 142 full photographs and 3,999 fronts in each arm |
| Added training photographs | frame_0146, frame_0160, frame_0178, frame_0371, frame_0383, in addition to the original 137 |
| Initialization | Both arms independently fine-tune from the learned epoch173 cassette checkpoint |
| Single R51 treatment | Double classification BCE weight only for foreground anchors assigned to partial targets; other declared training ingredients unchanged |
| Selected research model | PR35 remains selected under the historical frozen-reference gate |
| R53 contrast probe | Original pixels versus fixed Lab-L CLAHE, both at input size 1280 |
| R53 scale probe | Whole-image size 1280 versus 1600; this is not tiled inference |
| Evaluation data | Four annotation versions of the same 20 repeatedly examined development photographs |
| Optimization preference | Recognition quality is the goal; latency and hardware are not selection constraints |

R50 contains 654 targets: 583 with two locatable corners, 54 with one, and 17 with none. Both R51 arms detect all 583 two-corner targets under that reference. At the original 0.13 cutoff, the candidate's eleven misses consist of nine one-corner targets and two zero-corner targets. Corner availability is a useful stratum; it is not a measurement of occlusion percentage or proof that occlusion caused every failure. [P1, P2]

## 2. What duplicate, unmatched and missed mean

The repository scorer processes predictions in descending confidence order and assigns each to the best still-unmatched same-class reference at IoU at least 0.50. Each prediction and reference can participate in at most one match. [P3]

| Term | Exact meaning | What it does not establish |
| --- | --- | --- |
| Matched target | A reference assigned an eligible output under the scoring protocol | Accurate corners or correct extent beyond the IoU criterion |
| Missed target | A reference left without an assigned output | That the network never produced any candidate for it |
| Unmatched output | A retained prediction not assigned to a reference | That its image region is physically background |
| Formal duplicate | An unmatched prediction with IoU at least 0.50 to a same-class reference that already has a match | That ordinary NMS necessarily malfunctioned |
| Other unmatched | Unmatched outputs excluding formal duplicates | A visually adjudicated error cause |

Duplicates are already included in unmatched outputs. For example, 112 unmatched outputs including 23 duplicates means 23 duplicates plus 89 other unmatched, not 135 errors.

A localization error can create a miss and an unmatched output simultaneously. Consider equal-width boxes of height h shifted vertically by d, with 0 <= d < h. Their IoU is (h-d)/(h+d). If h=8 pixels and d=3 pixels, IoU=5/11, approximately 0.455, below the 0.50 matching threshold. The detector can respond to the correct physical front and still receive one miss plus one unmatched output. A confidence threshold cannot move that box boundary.

The duplicate and NMS overlap criteria also differ. Suppose a reference spans X=0 to 100, and two same-height predictions span -20 to 80 and 20 to 120. Each prediction has IoU 2/3 with the reference, but the predictions have IoU 3/7 with each other. Both can survive NMS at 0.70; after assignment, one can be a formal duplicate. Stronger suppression may also remove distinct adjacent fronts, so formal duplicate counts alone cannot determine the NMS cutoff.

TIDE provides a useful precedent for separating localization, duplicate, background and missed-instance effects. The current fixed-cutoff report is not TIDE's full oracle-AP computation. [L1]

## 3. New CPU analysis: compare operating points before rejecting a model

The replay first reproduces all eight original postclip score tables: two model arms times four references, including assignments, endpoint errors and corner counts. It verifies the original archive, prediction cache, source closure, and reference labels. It then examines all attainable confidence breakpoints at or above the saved 0.13 floor. Each image prefix is scored with the original matcher, and every selected point is directly rescored. No model, training run, or image transformation is involved. [A1]

### One readable cutoff applied to all four references

The 0.20 point below was examined after seeing the development frontier. It is an exploratory sensitivity result, not a threshold selected on an independent validation set. The same 0.20 cutoff is applied to every reference.

Each cell lists matched / missed / unmatched / duplicate counts.

| Reference | Reference model at 0.13 | Candidate at 0.13 | Candidate at 0.20 |
| --- | --- | --- | --- |
| Frozen666, primary | 632 / 34 / 91 / 17 | 635 / 31 / 120 / 23 | 633 / 33 / 89 / 15 |
| Historical663 | 633 / 30 / 90 / 17 | 634 / 29 / 121 / 23 | 633 / 30 / 89 / 15 |
| R38 | 639 / 15 / 84 / 14 | 643 / 11 / 112 / 24 | 642 / 12 / 80 / 13 |
| R50 | 639 / 15 / 84 / 13 | 643 / 11 / 112 / 23 | 642 / 12 / 80 / 11 |

At 0.20, no reference loses a previously matched target relative to the reference model at 0.13. The three empty scenes still have zero outputs. R50 recovers new_000010/R030, new_000010/R033, and new_000090/R039. These are two one-corner targets and one zero-corner target. Frozen666 recovers new_000010/R034. Row IDs belong to their named reference; identical numbers across references are not automatically identical objects.

This means the original fixed-0.13 tradeoff did not describe every useful operating point of the candidate. It does not prove probability calibration, a universal ranking advantage, or generalization to unseen photographs.

### Tune the reference too

A fair diagnostic must not search the candidate while keeping every reference setting artificially fixed. The reference also receives the full retained-score analysis.

On R50, the reference preserves all 639 matches with 80 unmatched outputs and 11 duplicates at a stricter cutoff around 0.14448. The candidate at 0.20 has 642 matches, the same 80 unmatched and the same 11 duplicates. Thus the three-target R50 gain persists against this tuned reference point.

On frozen666, the reference can preserve its 632 matches while reducing unmatched outputs to 48. The best candidate point under the original 91-output budget has 633 matches and 76 unmatched outputs. This is a tradeoff: one extra matched target accompanies more unmatched outputs than the reference's cleanest point retaining its original recall. The candidate does not dominate every reference operating point.

### Do not round an exact optimum without replaying it

| Candidate cutoff | Frozen666 misses / unmatched | R50 misses / unmatched |
| --- | --- | --- |
| 0.18 | 33 / 97 | 12 / 88 |
| 0.19 | 33 / 89 | 12 / 80 |
| 0.20 | 33 / 89 | 12 / 80 |
| 0.21 | 33 / 81 | 15 / 75 |
| 0.22 | 34 / 76 | 16 / 70 |

The 0.19 and 0.20 points retain exactly the same outputs; their constant-output interval is [0.18951362371444702, 0.2001393586397171). The best retained-score candidate point under frozen666’s budget of at most 91 unmatched outputs is on a much narrower interval, [0.21771249175071716, 0.21839691698551178). Rounding that cutoff to 0.22 loses its primary recovery. All interval endpoints use the strict keep rule confidence > cutoff.

### Aggregate corner counts conceal a regression

At candidate 0.20, frozen666 two-corner success within 5 pixels remains 511/584. However, new_000090/R045 changes from a worst-corner error of 4.955 pixels to 6.311 pixels and loses its success. Another two-corner target improves from 5.399 to 4.065 pixels, offsetting the aggregate count. Therefore the candidate fails a stronger requirement that every prior corner success be preserved.

R50 has no lost 5/10-pixel successes at 0.20. Its two-corner 5-pixel count changes 566/583 to 567/583, but that single gain is the already documented movement from 5.094 to 4.984 pixels on new_000010/R020. This approximately 0.11-pixel crossing is not evidence of a broad improvement in corner localization.

### PR35 still has the stronger frozen-reference result

| Frozen666 outcome | Retained PR35 | PR146 candidate at 0.20 |
| --- | --- | --- |
| Missed targets | 18 | 33 |
| Unmatched outputs | 80 | 89 |
| Both corners within 5 px | 533/584 | 511/584 |

The PR35 figures are the recorded retained-model result, not newly inferred in this CPU replay. The new candidate comparison does not satisfy the established promotion case. Historical compatibility and performance under the reviewed current labels must remain separate. [P2, A1]

## 4. New blind spot: corner confidence is not observed availability

The scorer measures location error only where a reference corner is locatable. This leaves another outcome to report: an in-bounds, high-confidence returned point for a matched target whose reference slot is marked unavailable.

Under R50, using endpoint confidence at least 0.50 and requiring finite coordinates inside the original image:

| Output setting | Reported points on matched unavailable slots | Unavailable slots on missed targets |
| --- | --- | --- |
| Reference, 0.13 | 64 / 70 | 18 |
| Candidate, 0.13 | 69 / 75 | 13 |
| Candidate, 0.20 | 68 / 74 | 14 |

There are 88 unavailable slots in this reference overall. The first denominator includes only slots on matched targets; the second column preserves the unavailable slots on missed targets rather than hiding them through selection.

These are unsupported corner outputs relative to the annotation. They are not proven physical localization errors: unavailable slots provide no reference coordinates with which to establish that claim. Unavailability can arise from occlusion, clipping, or inability to identify a unique point. Nevertheless, the result makes it unsafe to interpret a high endpoint score alone as proof that the physical corner was observable.

The user has now confirmed that outputs are used automatically. Retain an identifiable partial front and keep its presence separate from whether the application offers a complete coordinate pair. Preserve individual corner proposals, including hidden-point estimates, without turning them into observed labels. A runtime screen must operate on available image/model information; it cannot look up the reference's v=0 flag. This calls for an explicit output contract before it calls for another network.

### Baomai contract and the export blind spot

Baomai explicitly specifies actual lower-front corners, image-X ordering, and independent retention of a visible corner when the other is unavailable. It does not fully specify the box extent or a minimum visible fraction. The existing project policy supplies the identifiable-partial-front rule where upstream is silent. Its prohibition on guessing unavailable annotation coordinates does not prohibit a separately identified hidden-point prediction. [P8]

The inspected viewer uses endpoint scores and image bounds to decide which points to draw. Its save_snapshot function nevertheless writes the entire raw detections array to JSON; display information is separate. An automatic consumer reading those records must not assume that undrawn coordinates were removed. The minimal proposed contract is a front proposal plus raw landmarks and an explicit pair_candidate/withheld status under a named rule. Passing that rule does not itself establish physical accuracy. This document specifies the behavior; it does not change the running viewer. [P9]

### R55: evaluate every offered pair

R55 first reproduces all eight original model/reference baseline tables. It then examines 72 settings: four references, three model/box-cutoff combinations, and six endpoint cutoffs. A pair passes the simulated screen when both points are finite, in the original 1280×720 image, and meet the declared inclusive endpoint cutoff. The original matcher is recomputed on those pairs against all reference fronts. A supported success additionally needs two locatable reference corners and both errors within the stated original-pixel tolerance. [A2]

Every offered pair remains in the supported-fraction denominator, including unmatched outputs and matches to incomplete reference pairs. Recall uses all two-corner reference fronts, including missed and withheld ones. Thus high target recall does not imply that most offered outputs are verified. These fractions are reference-supported diagnostics, not calibrated probabilities or proven physical automatic-use accuracy.

All rows below use candidate box confidence strictly above 0.20.

| Reference | Endpoint cutoff | Offered pairs | Supported ≤5 px | Supported fraction ≤5 px | Eligible-pair recall ≤5 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen666, primary | 0.50 | 709 | 511 | 72.07% | 87.50% |
| Frozen666, primary | 0.80 | 650 | 511 | 78.62% | 87.50% |
| Frozen666, primary | 0.95 | 116 | 92 | 79.31% | 15.75% |
| R50 | 0.50 | 709 | 567 | 79.97% | 97.26% |
| R50 | 0.80 | 650 | 567 | 87.23% | 97.26% |
| R50 | 0.95 | 116 | 111 | 95.69% | 19.04% |

Raising 0.50 to 0.80 removes 59 offered pairs and loses zero supported target identities at either 5 or 10 pixels under both shown references. It also leaves the known >5-pixel localization errors unchanged: 66 frozen666 and 16 R50. It removes outputs without full reference support; it does not correct the retained coordinates.

For R50 at 0.50, 709 = 567 supported successes + 16 measured localization failures + 53 matches to incomplete pairs + 73 unmatched outputs. The latter 126 outputs are not all established physical errors. A one-corner reference can still measure one endpoint, although it cannot fully verify a pair. At 0.95, supported coverage collapses; at 0.99, no pair survives and the supported fraction is undefined.

All 722 front proposals remain in the separate presence channel at candidate box >0.20, regardless of this pair screen. Front counting therefore retains its own duplicate/unmatched problem. The six highlighted settings were independently recomputed, including all 120 image-level results and both 5/10-pixel partitions. The scores are post-hoc development probes and do not authorize a new runtime default. [A2]

Selective classification motivates examining quality together with coverage, but its published guarantees do not transfer automatically to this pose task and dataset. Confidence-calibration research likewise supports checking score meanings; classification temperature scaling alone does not establish a pixel-error guarantee. [L10, L11]

## 5. Which problem deserves priority?

The current evidence supports partial-front detection as the main improvement target. Protect already detected two-corner targets and report actual corner error continuously. With automatic consumption confirmed, keep unmatched and duplicate outputs as constraints and cause-specific review categories, and report the support and coverage of emitted coordinate pairs. Recovering a zero/one-corner front can improve presence/counting without producing a fully verifiable pair; both outcomes belong in the report.

The existing visual ledger classifies the 112 original-cutoff candidate unmatched outputs provisionally as 63 front-extent/fragments, 23 formal duplicates, eight rack/gripper responses, five keyboard/other background responses, six policy-sensitive exclusions, and seven ambiguous partial fragments. These are agent-assisted review categories, not independent human truth. The largest group involves cassette material or its extent, not established empty background. [P1]

| Supported cause | Smallest useful response |
| --- | --- |
| Real front, wrong visible extent | Review material ownership and consistent box supervision in eligible training photographs |
| Missing or policy-inconsistent reference | Adjudicate from original pixels, version the decision, and rescore all models without changing predictions |
| True repeated output on one front | Inspect ranking and the specific suppression pair; protect distinct adjacent fronts |
| Confirmed rack/gripper/background response | Check negative coverage within fully annotated training-eligible scenes |
| Identifiable partial front below confidence | Inspect the operating-point frontier and relevant partial-front supervision |
| No IoU-eligible decoded box | Investigate localization supervision, representation and scale; suppression cannot recover an absent eligible proposal |
| Unresolved physical ownership | Preserve uncertainty and request adjudication; do not manufacture a positive or negative label |

Do not make every unmatched output a hard negative. Do not expand target boxes to include the rack as context. The model already receives full photographs; context pixels can help recognition without becoming part of the target's geometric definition.

### Preserve completed negative results

PR143/R49 already tested linear Soft-NMS on the same decoded predictions. It recovered no target and removed one unmatched output. A lower-confidence 0.05 replay already recovered six R38/R50 targets but added 74 unmatched outputs, including outputs on previously empty scenes. The reference model's fifteen missed cases were traced to ten confidence-gate cases, three cases without eligible decoded geometry, and two ordinary-NMS cases. These causes belong to that checkpoint; they cannot be assigned to the newer candidate without its decoded traces. [P2, P4]

Partial-image sampling and partial-box-loss trials also exist in the history. A generic instruction to oversample difficult images or double a loss would repeat earlier work unless it introduces a distinct, supported hypothesis. R52's approximately two-pixel adjustments to eleven boundaries on ten rows are close to recorded visual uncertainty and do not by themselves justify expecting a large training gain. [P5]

## 6. A minimal next sequence

1. Finish and audit the two R53 returns. Each must first reproduce its unchanged control. Compare contrast and scale separately; keep original-image coordinates and all four named references. Reuse the declared R55 output metrics with fixed primary thresholds rather than introducing a new threshold search for each treatment.
2. Preserve the new retained-score analysis as a secondary diagnostic. A fresh inference confirmation of candidate 0.20 would be the next bounded runtime check if that operating point is pursued; do not change defaults from this replay.
3. Define a small, fixed training-side review pool containing identifiable partial fronts and their confusing surroundings. Use fully reviewed photographs and documented eligibility. Keep repeated development photographs out of training.
4. Compare the frozen geometry review queue with sequential review and seeded random review at equal unique-case budgets. Measure confirmed issues found and actual active review time. Include unflagged cases and full-image inspection: a geometry check on existing labels cannot find an entirely missing annotation.
5. Train a new paired comparison only when a meaningful accepted correction set or a distinct data hypothesis exists. Keep image membership identical for a label-only study. Keep labels and membership identical for a sampling-only study. Initialize both arms identically and use the same update budget and checkpoint rule.
6. Repeat an informative result with a predeclared additional seed before claiming stability. Independent capture-session evaluation remains a later requirement when such data become available.

If whole-image scaling helps identifiable thin fronts, a later pose-aware tiled inference probe is plausible. It must preserve the full-image control, map both landmarks correctly, handle tile-edge truncation, and merge repeated outputs without merging neighboring cassettes. SAHI's aerial box-detection results motivate that experiment; they do not establish the outcome for this pose task. [L6]

If CLAHE fails, record the bounded negative result. It rejects that fixed inference-only transform for this checkpoint and setting, not every jointly trained image enhancement method. If either intervention helps only certain cases, inspect the case identities before combining interventions. [L7]

## 7. Thesis contribution and research hypotheses

The user has confirmed freedom to choose the research contribution, subject to the supplied report structure. Adopt the broader working title Cassette Front Detection and Corner Localization for Automatic Image Analysis. Geometry-guided annotation review remains one component of this empirical study, alongside controlled inference and explicit output evaluation. No new neural architecture is required by the supplied instructions.

The report blueprint now includes all supplied headings. Abstract (150–300 words), Acknowledgements, Table of Contents, Introduction, Discussions, Conclusions, and References are compulsory. Background, Methodology & Implementation, Simulation Results, Experimental Results, and Appendices are included where applicable. The completed controlled digital-occlusion study can support Simulation Results, with its known-source-coordinate design and limits stated; no physical simulation is invented. R53 results remain pending. See thesis-outline-r55.md. [P10]

Three claims should be measured separately:

| Claim | Evidence required |
| --- | --- |
| The annotation policy is more consistent | Original-image adjudication, versioned decisions, uncertainty and repeatability or independent-review evidence |
| The review queue is more efficient | Confirmed issues per fixed review budget, random/sequential controls, actual time logs for any time-saving claim |
| Corrected supervision improves recognition | Paired training on identical images and a fixed evaluation reference, followed by seed and later session checks |

A label revision can change a score while the model is unchanged. A model can improve under one label policy and regress under another. Noise-aware evaluation research supports documenting those effects, especially for small boxes and strict overlap criteria. It does not license moving boundaries toward whichever model is preferred. [L3]

ObjectLab supplies a relevant label-review precedent. For a formal model-based queue comparison, its use of predictions from models that did not train on the scored image is important. Group related captures before splitting wherever provenance supports that grouping. A model agreeing with its own training labels is not an independent label audit. [L2]

One useful additional hypothesis comes from localization-aware active learning: images where predicted geometry changes under a controlled perturbation may deserve earlier review. R53 differences can provide illustrative cases, but CLAHE/scale disagreement is not an implementation of the paper's Gaussian-noise method. Instability is a review cue, not authority to relabel. [L4]

## 8. Models, initialization and the camera path

Retain the current one-class, two-landmark pose model while the active experiments finish. If a later capacity comparison is justified, a larger compatible pose model is simpler to evaluate than simultaneously adding segmentation, a verifier and a new label type. Preserve the task contract and compare converged recipes fairly. More parameters are a testable hypothesis, not a guaranteed solution to inconsistent targets or missing evidence.

Fresh fine-tuning and random initialization are different. A new optimizer run from the epoch173 weights still uses learned representations. The upstream baomai README and training_config.json explicitly document official COCO YOLO26s-Pose initialization. This is upstream provenance documentation; the public pretraining run has not been independently reproduced here. The earlier local wording that the inspected model card alone did not establish ancestry should be supplemented with this upstream statement. [P6]

Rethinking ImageNet Pre-training demonstrates that random initialization can catch up under some sufficiently trained detection regimes, while its very-small-data experiments show important limitations. It does not establish that discarding the cassette initializer is preferable for 142 related photographs. [L8]

The audited live application passes the SDK color image to the model without a separate CLAHE call. Camera image formation, RGB enhancement, detection filtering and endpoint display are different stages. The launcher defaults to epoch173 at confidence 0.25; the selected PR35 research operating point uses 0.13, and optional folder postclip suppression is not automatically the live-viewer pipeline. A future camera comparison must identify the actual checkpoint and settings. These configuration facts do not require a hardware experiment now. [P7]

## 9. Decisions resolved

| Question | User answer and working decision |
| --- | --- |
| An identifiable front has one or both corners unavailable | Follow baomai; use judgment where silent. Retain front presence and separately represented landmark estimates, preserving unavailable annotations and explicit pair status |
| What happens after detection? | Output is consumed automatically. Evaluate emitted outputs, duplicates, localization and coverage; do not rely on a person's rejection step |
| What contribution is required? | Free choice within the provided headings. Proceed with a controlled empirical thesis; a new architecture is optional |

The actual downstream operation, acceptable coordinate error, and consequences of a wrong extra output have not been specified. Do not invent a numerical exchange rate or deployment acceptance threshold. This is a later consumer-integration requirement, not a blocker for the current thesis or R53 analysis. Hardware/latency and the absence of new captures are already known constraints and do not need another interview.

## 10. Reproduction and evidence

The accompanying r54-cached-frontier.zip contains the complete replay script, original scorer, verified reference/prediction snapshot, machine-readable results, compact results report, provenance, this research note and the tradeoff figure. It needs no network, model weights, PyTorch or image files for the saved-output replay.

That R54 bundle remains an immutable snapshot of the earlier note. The updated note adds the user's decisions and R55 findings. The new r55-automatic-output-audit.zip contains the pair diagnostic, its results and ledgers, the same verified inputs/scorer, and the thesis blueprint. From its r55-analysis folder, run python -B analyze_automatic_pairs.py --inputs ../r54-analysis/inputs.json --scorer ../r54-analysis/pose_scoring.py --output replay.

Run python -B analyze_cached_frontier.py --inputs inputs.json --output replay from the extracted folder. Verification checks include all eight original baseline tables and direct rescoring of selected cutoffs. The saved snapshot permits only confidence thresholds at or above 0.13; it cannot establish full AP or recover proposals removed before the cache was saved.

Primary input archive SHA-256: 9538932145e90201d407c339b7da1f73c5c905935853191ab9683b221bd5a09e.

Prediction-cache SHA-256: 5fb49479ccbd93b0f9f538a0ad72e9d410c674f2a576107c1ec46f6507b1dbec.

### Project evidence

- A1: New replay files results.md and frontier.json in the accompanying bundle; baseline and selected-point checks are in the verification field.
- A2: R55 analyze_automatic_pairs.py, automatic-pair-results.json and automatic-pair-results.md in r55-automatic-output-audit.zip; verified original baseline checks, category partitions, and paired supported-target identities.
- P1: [PR146](https://github.com/Siuuuers/dl/pull/146), and the local docs/partial-classification-r51-results.md audit.
- P2: Local docs/current-model.md and docs/size-visibility-r53.md.
- P3: The unchanged scripts/pose_scoring.py, supplied verbatim in the bundle with its recorded hash.
- P4: [PR143](https://github.com/Siuuuers/dl/pull/143), and the local docs/front-diagnosis-r51.md audit.
- P5: Local docs/project-direction-r50.md, docs/front-review-r52.md and docs/thesis.md.
- P6: Upstream [training configuration](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/metadata/training_config.json) and [README](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/README.md), checked at baomai commit 351eb3cf960ccc34829e2bc72e1fa4645acc6295.
- P7: Local docs/camera-pipeline-r53.md. Source audit, not a physical camera run.
- P8: Upstream [data README](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/README.md), local docs/annotation-policy.md and docs/visible-and-hidden-2026-11-14.md.
- P9: Upstream [viewer source](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/scripts/view_d405_yolo.py) and local scripts/view_d405_yolo.py; render and save_snapshot read separately.
- P10: User-supplied thesis headings and local docs/pixel-occlusion-2026-11-13.md; companion thesis-outline-r55.md contains the complete report map.

### Primary literature

- L1: Bolya, Foley, Hays and Hoffman. [TIDE: A General Toolbox for Identifying Object Detection Errors](https://arxiv.org/abs/2008.08115). ECCV 2020. Error decomposition and application-dependent consequences.
- L2: Tkachenko, Thyagarajan and Mueller. [ObjectLab: Automated Diagnosis of Mislabeled Images in Object Detection Data](https://arxiv.org/html/2309.00832v1). ICML Data-centric Machine Learning Research Workshop, 2023. Model-assisted label review with out-of-sample predictions.
- L3: Murrugarra-Llerena and Jung. [Noise-Aware Evaluation of Object Detectors](https://openaccess.thecvf.com/content/WACV2025/html/Llerena_Noise-Aware_Evaluation_of_Object_Detectors_WACV_2025_paper.html). WACV 2025, pp. 9304–9313. Annotation boundary uncertainty and metric sensitivity.
- L4: Kao, Lee, Sen and Liu. [Localization-Aware Active Learning for Object Detection](https://arxiv.org/abs/1801.05124). ACCV 2018. Localization stability as a sample-selection signal.
- L5: Bodla, Singh, Chellappa and Davis. [Soft-NMS: Improving Object Detection With One Line of Code](https://arxiv.org/abs/1704.04503). ICCV 2017. Overlap-based score decay; already tested locally with no miss recovery.
- L6: Akyon, Altinuc and Temizel. [Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection](https://arxiv.org/abs/2202.06934). ICIP 2022. Conditional motivation for a later tiled probe.
- L7: Liu et al. [Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions](https://arxiv.org/abs/2112.08088). AAAI 2022. Jointly learned enhancement; not a plug-in validation of the current CLAHE transform.
- L8: He, Girshick and Dollár. [Rethinking ImageNet Pre-training](https://arxiv.org/pdf/1811.08883). ICCV 2019. Distinguishes sufficient optimization from an assumption that scratch initialization is always preferable.
- L9: Shrivastava, Gupta and Girshick. [Training Region-based Object Detectors with Online Hard Example Mining](https://arxiv.org/html/1604.03540v1). CVPR 2016. Supports studying difficult examples; its RoI-based algorithm is not a drop-in YOLO pose loss.
- L10: Geifman and El-Yaniv. [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500). NeurIPS 2017. Motivates studying quality together with the coverage of offered outputs; no statistical guarantee is claimed here.
- L11: Guo, Pleiss, Sun and Weinberger. [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html). ICML 2017, PMLR 70:1321–1330. Classification-score calibration; not a validated keypoint-error model.
