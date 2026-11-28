# Report source map

This technical companion maps the report's descriptive names to reproducible records. The report itself explains methods and results and cites external academic or professional sources. Updated 28 November 2026, Asia/Hong_Kong.

## References and model identities

| Report name | Technical source | Fronts | Complete pairs |
| --- | --- | ---: | ---: |
| Supplied annotation reference | Untouched data/labels/val, verified against upstream baomai commit 351eb3cf960ccc34829e2bc72e1fa4645acc6295 | 664 | 584 |
| Earlier fixed annotation reference | frozen666 | 666 | 584 |
| Reviewed annotation reference | r50 | 654 | 583 |

All three label sets describe the same 20 photographs. The [upstream provenance receipt](evidence/annotation-reference/upstream-provenance.json) verifies the byte-identical supplied manifest and every original validation-label hash. The supplied validation labels also monitored detector training; they are not the earlier fixed 666-front set.

The current upstream `baomai` branch was rechecked at commit `351eb3cf960ccc34829e2bc72e1fa4645acc6295`. Its [README](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/README.md) describes live display, recording and snapshots. The [data guidance](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/data/README.md) defines two corner slots with independent missing-point labels. The [viewer](https://github.com/an94-doll/embedding_vision/blob/351eb3cf960ccc34829e2bc72e1fa4645acc6295/scripts/view_d405_yolo.py) gates point rendering independently but preserves both returned keypoint slots in snapshots. These conventions do not specify a downstream consumer interface or physical acceptance tolerance. On 28 November 2026 the user clarified that vision outputs go to robotic arms for cassette localization, while arm control and integration are outside the study. Section 3.1 records that application boundary and the unspecified message format and physical tolerance.

- Supplied checkpoint SHA-256: e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e.
- Adapted detector SHA-256: 222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9.
- Residual correction checkpoint SHA-256: 19c7e95aca0019c3b1eb60de4c36c6ecd30d35fdcaaa103606cdd7d553f7006a.

## Measurements

| Report description | Evidence |
| --- | --- |
| Original versus proposed recognition | [Fresh common-runtime comparison](original-comparison.md); proposed method preserves adapted original-view boxes while averaging associated corner pairs |
| Remaining misses and extra predictions | [Detection-error analysis](detection-errors.md) and [audit evidence](evidence/detection-errors/README.md); all 583 complete fronts detected, incomplete-front detection 40/71 to 53/71, and proposed extras partitioned into 5 duplicates, 36 nearby unmatched and 10 low-overlap unmatched predictions |
| Where current misses lose eligible boxes | [Full-proposal stage audit](proposal-stages.md) and [evidence](evidence/proposal-stages/README.md); 12 confidence-gate losses, three native-NMS losses and three with no decoded box reaching reference IoU 0.50, linked to the current comparison by model/image hashes and scored-outcome continuity |
| Exploratory reflected-view output detections | [Reflected-anchor evidence](evidence/reflected-anchor/README.md); same saved inference, reversed output anchor, paired outcomes under all three references; trade-off retained without replacing the report's proposed method |
| Three annotation references | [CPU reference matrix](annotation-comparison.md), [six-cell summary](evidence/annotation-reference/summary.json), and [actual notebook execution](evidence/annotation-reference/notebook-execution.json) |
| Local image model | PR154 RGB-only heatmap return; [paired local-model results](local-refinement-results.md) |
| Local image and initial-corner model | PR153 Gaussian-hint heatmap return; same architecture/initialization/order as PR154 |
| Local coordinate-correction model | PR155, completed L4 training; [results](residual-results.md) and [independent audit](evidence/residual-return/audit.json) |
| Selected residual model replay | [CPU receipt](evidence/residual-return/selected-cpu-forward.json); all 71 raw proposals from two images, unchanged scored outcomes |
| Ideal-heatmap simulation | [Synthetic-map diagnostic](evidence/refinement/ideal-heatmap-diagnostic.json); reference positions are supplied, so this is not recognition accuracy |
| Controlled classification-loss result | [PR157 analysis](quality-focal-results.md) and [return evidence](evidence/quality-focal-return/README.md); matched native/QFL controls, all-reference replay, AP50 and training trace audit |
| Controlled training-label result | [PR158 analysis](training-label-results.md) and [return evidence](evidence/training-label-return/README.md); fixed 137-photo label intervention, reference-dependent rank reversal and duplicate trade-off |
| Focused annotation inspection | [Photo review](annotation-review-r65.md); frozen reference includes inherited case-specific exclusions, including a visible partial front explicitly recorded in [earlier review](../front-review-r38.md) |
| Historical annotation-only training | Earlier PR35 two-arm 137-photo trial, explained separately in [annotation comparison](annotation-comparison.md); its historical scores are not pasted into the current matrix |

Main report measurements use object confidence strictly above 0.25, endpoint confidence at least 0.50, matching IoU at least 0.50 and common postclip suppression. Equal averaging affects 648 of 687 adapted original-view records. The older research runs also retain their original confidence-above-0.13 scores in technical evidence; those counts are not mixed into the main table.

The residual/control/two-view mean maximum-corner errors use the same 583 complete pairs. The supplied-model conditional mean uses 580 available matched pairs, so it is not presented as a directly comparable 583-pair mean in the report.

The detection audit keeps all 654 reviewed fronts in detection evaluation. The supplied model misses three complete, 23 one-corner and eight zero-corner fronts; the proposed method misses zero, 13 and five respectively. For nonduplicate extras, the nearby geometric band is greatest reference IoU at least 0.10 and below 0.50; the low-overlap band is below 0.10. Neither band is a physical annotation verdict or an automatic negative training label.

The later stage audit reuses the PR143/R49 full decoded tensors for the same adapted checkpoint and the same photographs. All 20 historical outputs at confidence above 0.13 replay exactly. At confidence above 0.25, detection identities and per-front/per-corner threshold outcomes agree across 60 reference-image comparisons; maximum CPU/GPU coordinate and score differences are recorded rather than claiming byte-identical inference. Of 654 reviewed fronts, 651 have an eligible decoded box, 639 retain eligible support after confidence filtering, and 636 after native NMS. Maximum one-to-one matching has the same bounds, and postclip suppression loses no further reference coverage. These stages partition the 18 current misses; they do not establish which training mechanism or physical interpretation caused them.

The accompanying [box-geometry audit](evidence/box-geometry/README.md) reconstructs 120 per-image scores across all three references and both methods. Eight of ten near-overlap reviewed misses have a greatest-IoU retained box more than 1.5 times the reference width; another has height 2.49 times the reference. A greatest-overlap association is not proof of physical identity. The [research note](research-next-models.md) uses these findings to keep a Wasserstein regression change conditional and distinguish retained geometry from the earlier decoded pool.

The exploratory reflected-view anchor has 15 misses, 43 extras and ten duplicates under the reviewed reference. It retains 569/583 and 583/583 successes at five and ten pixels, recovering eight incomplete fronts while losing five other incomplete fronts. Supplied-reference successes change from 514/565 to 512/563, and earlier-fixed-reference successes from 525/574 to 524/573. These adverse results prevent presenting the variant as an unconditional replacement.

The same reflected-anchor evidence includes the refreshed automatic-pair diagnostic. For the retained proposed method, 678 accepted pairs comprise 569 supported within five pixels, 14 matched localization failures, 48 matches to incomplete reference pairs and 47 unmatched accepted predictions. The latter two groups lack reference support rather than establishing physical error. Acceptance screening is followed by fresh matching; these counts must not be inferred by subtracting the full detection table.

The report now records the completed [classification-loss pilot](quality-focal-results.md) and [controlled training-label comparison](training-label-results.md). Section 5.5 states the matched protocols, Sections 7.3–7.4 give their results, and Appendix 11.3 preserves all three training-label sensitivity comparisons. A future unseen-group or out-of-fold proposal claim must exclude each held group from the starting weights' training history as well as from subsequent fine-tuning. This does not invalidate the completed comparisons' stated intervention from shared supplied weights.

## Reproduction and presentation

The explanatory revision adds the adapted single-view control to Table 3 without new inference. In [the fresh original-baseline summary](evidence/original-baseline/fresh/summary.json), `measurement.references.r50.trained.pairs` records 566/583 successes at five pixels and 582/583 at ten pixels, with all 583 complete pairs returned. The corresponding mean maximum-corner error is 2.3167385644521508 pixels. This is the correct input control for the local models, distinct from the unchanged supplied benchmark. The corrected model's 564/583 and mean 2.266 pixels retain their previously reported opposite movements.

The [citation audit](citation-audit-r64.md) maps the original 15 external references and new reproducibility reference [16] to a supported source claim, the project adaptation and its limit. The bibliography uses publisher pagination with matching DOIs and retains differing CVF pagination as metadata. ResNet is cited for feature shortcuts; Generalized Focal Loss is cited for the completed classification weighting experiment. PyTorch 2.14 reproducibility guidance supports the limited cross-platform identity statement, not a measured GPU effect. The main text distinguishes both from separate coordinate correction and from a complete implementation of the cited detector. The [score-target note](research-score-assignment.md) records the frozen native code hashes supporting the new explanation of pose-aware soft targets. The returned model notebooks preserve their budgets and pin PR156 scoring source acaf5d4b4043583fdd81f5ea848b3a6a68ada728; finite collapsed boxes count as unmatched extras. Both returned runs are complete. The fixed-weight assignment diagnostic introduces no optimizer or new trained candidate.

- [Notebook guide](experiments.md): completed paired GPU experiments, the fixed-weight diagnostic and the proposal/geometry and augmentation CPU audits, with current execution evidence and return instructions.
- [Proposal and box-geometry notebook](../../notebooks/baomai_proposal_audit_r63.ipynb): all four actual cells completed unchanged from clean source commit `bb1d74cce4739a72ad0f7d9f227ca1e4cac37c7c`; [execution receipt](evidence/proposal-stages/notebook-execution.json), independent package verification and exact second analytical replay retained.
- [Native augmentation notebook](../../notebooks/baomai_augmentation_audit_r63.ipynb): all four actual cells completed unchanged from clean source commit `87b3f1dfd547bf46fef35b171aebe16d832bd1be`; [execution evidence](evidence/augmentation-audit/README.md) preserves the training-only input inventory, 426 image presentations, row lineage, source/runtime identities and six review panels. These additional diagnostics remain in the technical thesis notes; their counts are not inserted into the main manuscript.
- [Detection-audit notebook](../../notebooks/baomai_detection_audit_r62.ipynb): all four unchanged cells completed from clean published source, with [execution evidence](evidence/detection-errors/notebook-execution.json); both analyses and the exact result package verified without inference or training.
- [Annotation-reference notebook](../../notebooks/baomai_annotation_reference_r60.ipynb): all four exact cells completed from published source, no new inference or training.
- [Original-comparison notebook](../../notebooks/baomai_original_comparison_r59.ipynb): all four cells and full 60 image/model passes completed previously.
- [Residual notebook](../../notebooks/baomai_residual_corner_r59.ipynb): completed return PR155; retained for reproduction.
- [Report](report.md), [bibliography](bibliography-fyp-i.json), and [final validation](report-validation-fyp-i.json): manuscript, external sources, final presentation counts and artifact identities.
- [Word builder](../../scripts/build_thesis_docx_r56.py) and [equation validator](../../scripts/verify_thesis_equations_r60.py): genuine centered display OMML, automatic-height equation rows, and measured whole-formula positions.

Only Figure 1 is used from the retained figure set. Older charts and dated plans remain historical assets. The selected deployment model and live defaults are separate from the thesis comparisons.

## Paired training returns and fixed-source replay

PR157 at `f971e880e8aff2859ce4d8ae2923efea826eddb4` and PR158 at `6bde3b7a759604d605965d468e6102bf99bf7946` both execute source `acaf5d4b4043583fdd81f5ea848b3a6a68ada728`; notebook pin `587514e557db13be287a5483eaf952dfd5e14761` carries that source. Their compact archive SHA-256 values are `cf7b7b71349b9412bd5272001f5071afcf8d3b356423deb6c233cb1e6c326c4a` and `33dbf73b0d8e145fb0aa74184b16a306d148f0b177787349b13465e7b77c3974`, respectively. Independent audits verify archived code/input identities, checkpoint wrappers/member order/complete hashes, paired receipts and saved-prediction replay. The original result PRs remain unchanged.

QFL native/QFL models have SHA-256 `15d7c97d5c58f16e40462807231c7bb01a5ee9a719225cb756493e1b41714d67` / `8760602df814f47685dde25a2e389a554f623a7280473c90d5e94a0f30fe519f`. Supplied-label/reviewed-label models have SHA-256 `493f34d4af03e6f0e96b21b026da52437837e631bafe893a5df3cec0c92cd232` / `c749fcff2126467cbc770f75a71fd79da51ebd08308ba909a448c3dd8cb5b372`.

The QFL Table 4 native control is a newly paired run with 18 misses, 49 extras and 4 duplicates; it is not the historical Table 2 proposed checkpoint with 18/51/5. Both QFL arms use 142 training photographs on an A100; both training-label arms use 137 on an L4. These within-run controls support their respective interventions. The 137-photo reviewed-label candidate's difference from the historical 142-photo candidate does not isolate the five supplementary photographs.

The PR158 auxiliary original-label progress member begins with `arm\n` and cannot be parsed as JSON. Its remaining bytes exactly equal the valid final receipt. The inventory preserves those original bytes and the producer verifier uses the final receipt. This bounded anomaly is documented in the audit; no evidence is silently repaired.

The report describes the reviewed benchmark as a frozen accepted reference with documented inherited exclusions. The generic visible-front definition is not a claim that every frozen row and omission has been independently certified. No labels are changed by these result analyses.
