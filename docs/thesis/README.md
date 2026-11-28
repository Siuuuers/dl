# Cassette vision Final Year Project I

Updated 28 November 2026, Asia/Hong_Kong.

**[Current report](report.md)** explains cassette-front detection and lower-corner localization for undergraduate Electronic Engineering readers. It compares the proposed method with the untouched supplied baomai model, explains project results directly, and cites external academic or professional sources. Acknowledgements are omitted as requested.

The report explains what each cited method establishes and how the cassette implementation adapts it. Sections 4, 5 and 7 now focus on the measurements, implemented choices and experimental findings. Earlier annotation versions are named directly, and detailed replay checks stay in the supporting records. The [citation audit](citation-audit-r64.md) records source claims and publication details. Table 3 retains the adapted single-view control, comparing the correction model's 564/583 five-pixel result with its input's 566/583 and the proposed method's 569/583.

The Word report uses six centered display equations with separate right-aligned numbers and automatic row height. The [validation record](report-validation-fyp-i.json) contains the exact page count, abstract length, linked contents map, equation geometry and all-page visual checks. It is the authoritative source for presentation counts.

## Completed comparisons and diagnostics

The **[notebook guide](experiments.md)** records the completed GPU comparisons and focused CPU experiments. The [current-checkpoint trace and corner-condition comparison](research-detection-decisions-r69.md) adds verified full-proposal evidence and a small measured output improvement.

The latest **[crop context and scale comparison](research-crop-context-r71.md)** adds two overlapping local views at fixed sizes. Original-scale crops give one fewer miss but two fewer ten-pixel pair successes. Enlarged crops increase extras from 50 to 106 and duplicates from 14 to 44 while reducing five-pixel pair successes from 572 to 566. The [four-cell CPU notebook](../../notebooks/baomai_crop_context_r71.ipynb) is complete, with [independent verification and full results](evidence/crop-context-r71/README.md). These findings support retaining full-image inference and prioritizing representative real partial-front examples.

The **[R69 four-cell notebook](../../notebooks/baomai_current_proposals_r69.ipynb)** has completed from source `41d0e6937c695377373cc55071ffb780345870ee`. The reviewed-label model's 18 r67 original-view misses comprise 12 confidence failures, five geometry failures and one native-suppression loss. The [corner condition](evidence/corner-suppression-r69/README.md) reduces its archived extras from 50 to 44 and duplicates from 13 to 11 while preserving every match and known-corner outcome under all four references. The [fresh CPU trace](evidence/current-proposals-r69/README.md) confirms the same extra reduction, with the measured runtime duplicate-category difference recorded separately. The [citation update](citation-audit-r69.md) explains the applied RMPE connection and the purpose of each equation.

The **[r68 diagnosis and fixed union](research-partial-detection-r68.md)** uses saved outputs to separate confidence and geometry failures, then measures one union with unused reflected records. For the reviewed-label model under r67, misses change from 18 to 10, extras from 50 to 60 and duplicates from 13 to 18. Complete-pair counts stay at 572/583 within five pixels and 583/583 within ten. Six one-known and two zero-known fronts are recovered, but one returned known corner has an 11.3027-pixel error. The three newly restored focus misses remain, and older references give only two added matches with sixteen more extras. The [CPU notebook](../../notebooks/baomai_partial_detection_r68.ipynb) reproduces this bounded experiment without inference or training. No union, loss or model is promoted.

The **[partial-front review](partial-inclusion-r67.md)** restores four exact historical rows in a separate r67 evaluation reference after raw-image adjudication. The review excluded three other candidates whose visible material did not establish a separate front. All 654 r50 rows preserve their bytes and positions; r67 contains 658 fronts, with 18/57/583 fronts having zero/one/two known corners and 1,223 known corners in total. For all six unchanged methods, the reference change adds one match and three misses while removing one other-extra. Duplicates and complete-pair scores are unchanged. The [research note](research-partial-inclusion-r67.md), [reference manifest](../../data/reviews/fronts-2026-11-28-r67/manifest.json) and [evidence guide](evidence/partial-inclusion-r67/README.md) preserve the reasoning and exact changes.

The [classification-loss result](quality-focal-results.md) verifies six fewer misses but 34 more extras, including 13 more duplicates, and almost unchanged AP50. The [controlled training-label result](training-label-results.md) verifies 32 fewer misses, 34 fewer extras and 31 more five-pixel successes under the reviewed reference, with seven more duplicates. Both older references favour the supplied-label arm. The paired controls and reference dependence remain explicit in the report.

The [annotation definition](../annotation-definition.md) gives the practical rule: one identifiable visible front, its visible extent, and each actual lower corner when locatable. The [focused photo review](annotation-review-r65.md) records bounded image judgments; it is separate from an independent annotation-repeatability study. Any future amendment must preserve source identities and show changed details on only the affected photographs.

The completed [score and assignment diagnostic](score-assignment-audit.md) observes all 3999 training fronts under both checkpoints and both native heads without an optimizer. Every front receives an assignment. Removing corner agreement lowers the one-corner group's mean targets in this frozen counterfactual, which does not support an immediate box-only training switch.

The [duplicate and unmatched audit](duplicate-audit-r66.md) separately applies reflected support and additional output suppression to saved detections. Under frozen r50, requiring reflected support on the reviewed-label candidate gives 19 misses, 27 extras and six duplicates; additional suppression at IoU 0.50 gives 19, 23 and one, respectively, versus the unchanged 15, 51 and 13. Both restrictions lose four partial-front matches while retaining 572/583 five-pixel and 583/583 ten-pixel successes. The [r67 sensitivity check](partial-inclusion-r67.md#the-suppression-risk-remains) retains the newly supported bottom-band output under every restriction, but those other partial-front losses persist. These diagnostic trade-offs do not select a replacement. [Primary-source research](research-duplicates-r66.md) explains the next conditional choices.

The new **[detection audit and fixed reflected-box experiment](detection-errors.md)** are complete. The retained method detects all 583 fronts with complete corner annotations but only 53 of 71 with incomplete annotations. A reflected-box alternative reduces misses/extras to 15/43 and keeps 569/583 five-pixel successes, while duplicates rise to ten and older-reference corner scores decline. The **[CPU reproduction notebook](../../notebooks/baomai_detection_audit_r62.ipynb)** needs no model or photograph download. It adds an explicit tradeoff, not an automatic replacement.

The **[full-proposal replay](proposal-stages.md)** now explains where those 18 misses lose eligible boxes: 12 at the confidence gate, three during native NMS, and three without a decoded box reaching matching IoU. Matching outcomes agree with the current comparison under all three references. The accompanying box-geometry audit finds that eight of ten near misses have a greatest-overlap retained box more than 1.5 times the reference width. These are geometric diagnoses, not proof of physical box ownership or a particular learning cause. The **[augmentation audit](augmentation-audit.md)** retains 11,880 of 11,997 target presentations, with 117 removed and 368 retained boxes clipped. Most removals are wholly outside the canvas; the bounded visual review gives no concrete reason to change the policy. Both new CPU notebooks completed all four actual cells. The [research note](research-next-models.md) connects these findings to primary literature and records which ideas have already been tried.

The additional [score and assignment research](research-score-assignment.md) examines TOOD, IoU-Net and VarifocalNet against the pinned native implementation. That implementation already uses box and known-corner agreement in assignment and soft score targets. The completed diagnostic distinguishes missing assignment, low supervision targets and predictions below useful targets without training. Its anchor-level averages are descriptive, not each front's maximum inference confidence. The returned notebook source includes the PR156 repair that retains clipped zero-area boxes as unmatched extras.

## Frozen r50 comparison

The [original-model comparison](original-comparison.md) uses the same 20 photographs, software environment and operating settings for both methods. The proposed method adapts the supplied detector and averages associated original/reflected corner predictions. This table preserves the 654-front r50 benchmark; the [r67 guide](partial-inclusion-r67.md) reports the separately versioned 658-front comparison.

| Reviewed annotations | Supplied model | Proposed method |
| --- | ---: | ---: |
| Missed fronts | 34 | 18 |
| Extra predictions | 61 | 51 |
| Duplicates, included in extras | 5 | 5 |
| Both corners within 5 pixels | 555/583 | 569/583 |
| Both corners within 10 pixels | 577/583 | 583/583 |

The completed [annotation-reference experiment](annotation-comparison.md) scores those same predictions against three verified label sets. At five pixels the original/proposed comparison is 521/514 out of 584 under the untouched supplied labels, 534/525 out of 584 under the earlier fixed labels, and 555/569 out of 583 under the reviewed labels. No inference, training or labels changed during this comparison. The direction depends on the reference; it is not an isolated measurement of the training-label effect.

All three local-model training returns are complete. [RGB and hint heatmap models](local-refinement-results.md) achieve 547/583 and 558/583 reviewed five-pixel successes. The [residual correction model](residual-results.md) achieves 564/583, below the two-view method's 569/583. Its mean maximum-corner error improves slightly over its unrefined control, while its five-pixel success count declines. The report preserves that distinction.

## Read and reproduce

| File | Purpose |
| --- | --- |
| [Report](report.md) | Current explanatory manuscript |
| [Requirements](requirements.md) | Supplied report requirements and their implementation |
| [Bibliography](bibliography-fyp-i.json) | External references in IEEE first-citation order |
| [Citation audit](citation-audit-r64.md) | Source claims, cassette adaptations, scope limits and verified publisher metadata |
| [Word validation](report-validation-fyp-i.json) | Final document, contents, equation and visual checks |
| [Report source map](report-source-map.md) | Exact technical provenance |
| [Annotation definition](../annotation-definition.md) | Practical labeling decisions and unresolved-case handling |
| [Annotation research](research-annotation-definition.md) | Primary sources and a proposed repeatability measurement |
| [Notebook guide](experiments.md) | Two completed GPU comparisons and the focused CPU diagnostics, with execution status |
| [Crop context and scale research](research-crop-context-r71.md) | Completed two-scale crop comparison, applied literature and next data priority |
| [Crop notebook and evidence](evidence/crop-context-r71/README.md) | Four actual CPU cells completed, complete archive and standalone independent replay |
| [Current-model and suppression research](research-detection-decisions-r69.md) | Completed raw stages, fixed corner rule and four primary research directions |
| [Current-model notebook](../../notebooks/baomai_current_proposals_r69.ipynb) | Four actual CPU cells completed from the published R69 source |
| [Current-model evidence](evidence/current-proposals-r69/README.md) | Full decoded proposals, native lineage, runtime comparisons and independent verification |
| [Corner-condition evidence](evidence/corner-suppression-r69/README.md) | Five models, four references, partial-front preservation and the counterexample |
| [Citation and equation update](citation-audit-r69.md) | Applied RMPE reference and project purpose of the six equations |
| [Partial-detection research and results](research-partial-detection-r68.md) | Three restored miss stages, completed reflection union, partial-corner and reference sensitivity, and conditional methods |
| [Partial-detection notebook](../../notebooks/baomai_partial_detection_r68.ipynb) | Four-cell CPU reproduction pinned to source 0e1f545a2cf33022b86bf316866b08cacfa55477 |
| [Partial-detection results](evidence/partial-detection-r68/results.md) | Five methods, four references, saved-pool limits and the full union tradeoff |
| [Partial-front inclusion](partial-inclusion-r67.md) | Four restored rows, three held cases and the completed six-method reference-only replay |
| [Partial-inclusion research](research-partial-inclusion-r67.md) | Supplied Baomai guidance, identifiable partial fronts and unknown-corner semantics |
| [Partial-inclusion notebook](../../notebooks/baomai_partial_inclusion_r67.ipynb) | Four actual CPU cells completed from pinned source and preseeded archives |
| [Partial-inclusion evidence](evidence/partial-inclusion-r67/README.md) | Review decisions, exact row lineage, numeric replay, filter sensitivity and independent verification |
| [Proposal stages](proposal-stages.md) | Full decoded-proposal continuity, score/NMS losses and box geometry |
| [Proposal and geometry notebook](../../notebooks/baomai_proposal_audit_r63.ipynb) | Four-cell CPU reproduction from immutable source |
| [Augmentation audit](augmentation-audit.md) | Native transforms, target lineage and review panels |
| [Augmentation notebook](../../notebooks/baomai_augmentation_audit_r63.ipynb) | Four-cell CPU audit of the existing training recipe |
| [Research and model decisions](research-next-models.md) | Evidence-based priorities, prior attempts and conditional next tests |
| [Score and assignment research](research-score-assignment.md) | Native pose-aware targets, three additional primary papers and one conditional diagnostic |
| [Detection errors](detection-errors.md) | Current extra-output groups, partial-front recall and the next model decision |
| [Detection audit notebook](../../notebooks/baomai_detection_audit_r62.ipynb) | Completed CPU replay; all four unchanged code cells executed |
| [Detection audit evidence](evidence/detection-errors/README.md) | Exact baseline replay and method/reference-specific review cases |
| [Reflected-box evidence](evidence/reflected-anchor/README.md) | All references, independent averaging replay and adverse paired cases |
| [Classification-loss results](quality-focal-results.md) | PR157 paired return, exact score/AP replay and adverse trade-offs |
| [Training-label results](training-label-results.md) | PR158 controlled intervention and all-reference rank reversal |
| [Score and assignment diagnostic](score-assignment-audit.md) | Frozen-checkpoint, per-target investigation; exact execution status in the guide |
| [Duplicate and unmatched audit](duplicate-audit-r66.md) | Saved-output restrictions, partial-front losses and matching bounds |
| [Duplicate research](research-duplicates-r66.md) | TIDE, Soft-NMS, IoU-Net, existing one-to-one head and reference inclusion |
| [Focused annotation review](annotation-review-r65.md) | Original-image review and recorded decisions |
| [Residual results](residual-results.md) | PR155 audit, paired changes, mean-error nuance and CPU replay |
| [Residual evidence](evidence/residual-return/README.md) | Reproduction and complete audit identities |
| [Annotation comparison](annotation-comparison.md) | Evaluation-label versus training-label effects |
| [Annotation-reference notebook](../../notebooks/baomai_annotation_reference_r60.ipynb) | Completed CPU experiment; all four code cells executed |
| [Annotation-reference evidence](evidence/annotation-reference/README.md) | Untouched upstream-label provenance, six matrix cells and execution receipt |
| [Original-model comparison](original-comparison.md) | Completed common-runtime model inference |
| [Original-comparison notebook](../../notebooks/baomai_original_comparison_r59.ipynb) | Optional reproduction of the completed inference comparison |
| [Residual training protocol](residual-correction-experiment.md) | Exact completed pilot and original notebook |

The annotation-reference and partial-inclusion notebooks use saved predictions and the Python standard library. They need no GPU, model download or image download and have completed locally; rerunning them is optional. The r67 notebook uses immutable source `2c3886271c12e501d236a5eff54e3eb78007d1b0`; its four actual cells ran with preseeded source and archives, without a hosted Colab or live-download claim. Earlier GPU notebooks retain their original pins for reproduction and need not be repeated to answer an evaluation-reference question.

## Interpretation and next work

Vision outputs are intended for robotic arms to locate cassettes; arm control and integration are outside this study. Recognition quality is the priority. The report measures misses, extras, duplicates and corner accuracy together. The 20 photographs provide development evidence under the named annotation references. A downstream message format and physical acceptance tolerance have not been supplied.

The completed training-label comparison excludes the five additional photographs from both arms to isolate the label intervention. It tests the frozen effective labels, not a newly relabeled implementation of the written definition. The completed review establishes the annotation versions used in these comparisons. Local-model training also needs errors representative of photographs unseen by the proposal detector.

The selected deployment model and live defaults are unchanged. Technical experiment identifiers remain in this folder; the report uses descriptive names.

## Rebuild the Word report

The [builder](../../scripts/build_thesis_docx_r56.py) creates editable native Office Math display paragraphs inside symmetric borderless equation rows. The whole formula is centered, its number occupies a separate right-hand cell, and neither line nor row has a fixed height. Ordinary labels render upright and vector symbols retain their intended bold appearance.

    "$CODEX_PRIMARY_RUNTIME_PYTHON" scripts/build_thesis_docx_r56.py docs/thesis/report.md --output report-first.docx

Render using the canonical documents renderer, then rebuild with its PDF page map:

    "$CODEX_PRIMARY_RUNTIME_PYTHON" scripts/build_thesis_docx_r56.py docs/thesis/report.md --output cassette-vision-fyp-i-report.docx --page-map-from-pdf report-first.pdf --require-page-map

Render the final document, run [equation geometry verification](../../scripts/verify_thesis_equations_r60.py), inspect every page and confirm the contents page numbers. The final Word file is delivered separately.
