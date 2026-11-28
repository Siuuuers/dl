# Cassette vision Final Year Project I

Updated 28 November 2026, Asia/Hong_Kong.

**[Current report](report.md)** explains cassette-front detection and lower-corner localization for undergraduate Electronic Engineering readers. It compares the proposed method with the untouched supplied baomai model, explains project results directly, and cites external academic or professional sources. Acknowledgements are omitted as requested.

The expanded report now explains what each cited method establishes and how the cassette implementation adapts it. It distinguishes suppression, cross-view association and evaluation matching; reference availability and predicted confidence; and image accuracy and physical positioning. The [citation audit](citation-audit-r64.md) records the original 15 verified IEEE references and the new reproducibility reference, with publisher-pagination decisions. Table 3 includes the adapted single-view control, making the residual model's 564/583 result comparable with its unrefined input's 566/583, as well as the proposed method's 569/583.

The Word report uses six centered display equations with separate right-aligned numbers and automatic row height. The [validation record](report-validation-fyp-i.json) contains the exact page count, abstract length, linked contents map, equation geometry and all-page visual checks. It is the authoritative source for presentation counts.

## Completed comparisons and next diagnostic

The **[notebook guide](experiments.md)** records both completed GPU comparisons and the fixed-weight diagnostic now being prepared from their findings.

The [classification-loss result](quality-focal-results.md) verifies six fewer misses but 34 more extras, including 13 more duplicates, and almost unchanged AP50. The [controlled training-label result](training-label-results.md) verifies 32 fewer misses, 34 fewer extras and 31 more five-pixel successes under the reviewed reference, with seven more duplicates. Both older references favour the supplied-label arm. The paired controls and reference dependence remain explicit in the report.

The [annotation definition](../annotation-definition.md) gives the practical rule: one identifiable visible front, its visible extent, and each actual lower corner when locatable. The [focused photo review](annotation-review-r65.md) records bounded image judgments; it is separate from an independent annotation-repeatability study. Any future amendment must preserve source identities and show changed details on only the affected photographs.

The next [score and assignment diagnostic](score-assignment-audit.md) uses frozen returned checkpoints and no optimizer. Existing aggregate traces show almost complete partial-target assignment; the remaining question is how per-target scores relate to native supervision for zero, one and two known corners. A new trained loss or detector is not justified by these aggregates alone.

The new **[detection audit and fixed reflected-box experiment](detection-errors.md)** are complete. The retained method detects all 583 fronts with complete corner annotations but only 53 of 71 with incomplete annotations. A reflected-box alternative reduces misses/extras to 15/43 and keeps 569/583 five-pixel successes, while duplicates rise to ten and older-reference corner scores decline. The **[CPU reproduction notebook](../../notebooks/baomai_detection_audit_r62.ipynb)** needs no model or photograph download. It adds an explicit tradeoff, not an automatic replacement.

The **[full-proposal replay](proposal-stages.md)** now explains where those 18 misses lose eligible boxes: 12 at the confidence gate, three during native NMS, and three without a decoded box reaching matching IoU. Matching outcomes agree with the current comparison under all three references. The accompanying box-geometry audit finds that eight of ten near misses have a greatest-overlap retained box more than 1.5 times the reference width. These are geometric diagnoses, not proof of physical box ownership or a particular learning cause. The **[augmentation audit](augmentation-audit.md)** retains 11,880 of 11,997 target presentations, with 117 removed and 368 retained boxes clipped. Most removals are wholly outside the canvas; the bounded visual review gives no concrete reason to change the policy. Both new CPU notebooks completed all four actual cells. The [research note](research-next-models.md) connects these findings to primary literature and records which ideas have already been tried.

The additional [score and assignment research](research-score-assignment.md) examines TOOD, IoU-Net and VarifocalNet against the pinned native implementation. That implementation already uses box and known-corner agreement in assignment and soft score targets. The note specifies a conditional diagnostic to distinguish missing assignment, low supervision targets and predictions below useful targets. The detailed diagnostic develops that question without training. The returned notebook source includes the PR156 repair that retains clipped zero-area boxes as unmatched extras.

## Current results

The [original-model comparison](original-comparison.md) uses the same 20 photographs, software environment and operating settings for both methods. The proposed method adapts the supplied detector and averages associated original/reflected corner predictions.

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
| [Focused annotation review](annotation-review-r65.md) | Original-image review and recorded decisions |
| [Residual results](residual-results.md) | PR155 audit, paired changes, mean-error nuance and CPU replay |
| [Residual evidence](evidence/residual-return/README.md) | Reproduction and complete audit identities |
| [Annotation comparison](annotation-comparison.md) | Evaluation-label versus training-label effects |
| [Annotation-reference notebook](../../notebooks/baomai_annotation_reference_r60.ipynb) | Completed CPU experiment; all four code cells executed |
| [Annotation-reference evidence](evidence/annotation-reference/README.md) | Untouched upstream-label provenance, six matrix cells and execution receipt |
| [Original-model comparison](original-comparison.md) | Completed common-runtime model inference |
| [Original-comparison notebook](../../notebooks/baomai_original_comparison_r59.ipynb) | Optional reproduction of the completed inference comparison |
| [Residual training protocol](residual-correction-experiment.md) | Exact completed pilot and original notebook |

The new annotation-reference notebook uses saved predictions and the Python standard library. It needs no GPU, model download or image download. It already completed locally; rerunning it is optional. Earlier GPU notebooks retain their immutable source pins for reproduction and need not be repeated to answer the evaluation-reference question.

## Interpretation and next work

Vision outputs are intended for robotic arms to locate cassettes; arm control and integration are outside this study. Recognition quality is the priority. The report measures misses, extras, duplicates and corner accuracy together. The 20 photographs have guided development and annotation review; independent captures are still needed. A downstream message format and physical acceptance tolerance have not been supplied.

The completed training-label comparison excludes the five additional photographs from both arms to isolate the label intervention. It tests the frozen effective labels, not a newly relabeled implementation of the written definition. Independent annotation repeatability remains unmeasured. Local-model training also needs errors representative of photographs unseen by the proposal detector.

The selected deployment model and live defaults are unchanged. Technical experiment identifiers remain in this folder; the report uses descriptive names.

## Rebuild the Word report

The [builder](../../scripts/build_thesis_docx_r56.py) creates editable native Office Math display paragraphs inside symmetric borderless equation rows. The whole formula is centered, its number occupies a separate right-hand cell, and neither line nor row has a fixed height. Ordinary labels render upright and vector symbols retain their intended bold appearance.

    "$CODEX_PRIMARY_RUNTIME_PYTHON" scripts/build_thesis_docx_r56.py docs/thesis/report.md --output report-first.docx

Render using the canonical documents renderer, then rebuild with its PDF page map:

    "$CODEX_PRIMARY_RUNTIME_PYTHON" scripts/build_thesis_docx_r56.py docs/thesis/report.md --output cassette-vision-fyp-i-report.docx --page-map-from-pdf report-first.pdf --require-page-map

Render the final document, run [equation geometry verification](../../scripts/verify_thesis_equations_r60.py), inspect every page and confirm the contents page numbers. The final Word file is delivered separately.
