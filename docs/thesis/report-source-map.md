# Report source map

This technical companion maps the report's descriptive names to reproducible records. The report itself explains methods and results and cites external academic or professional sources. Updated 28 November 2026, Asia/Hong_Kong.

## References and model identities

| Report name | Technical source | Fronts | Complete pairs |
| --- | --- | ---: | ---: |
| Supplied annotation reference | Untouched data/labels/val, verified against upstream baomai commit 351eb3cf960ccc34829e2bc72e1fa4645acc6295 | 664 | 584 |
| Earlier fixed annotation reference | frozen666 | 666 | 584 |
| Reviewed annotation reference | r50 | 654 | 583 |

All three label sets describe the same 20 photographs. The [upstream provenance receipt](evidence/annotation-reference/upstream-provenance.json) verifies the byte-identical supplied manifest and every original validation-label hash. The supplied validation labels also monitored detector training; they are not the earlier fixed 666-front set.

- Supplied checkpoint SHA-256: e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e.
- Adapted detector SHA-256: 222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9.
- Residual correction checkpoint SHA-256: 19c7e95aca0019c3b1eb60de4c36c6ecd30d35fdcaaa103606cdd7d553f7006a.

## Measurements

| Report description | Evidence |
| --- | --- |
| Original versus proposed recognition | [Fresh common-runtime comparison](original-comparison.md); proposed method preserves adapted original-view boxes while averaging associated corner pairs |
| Three annotation references | [CPU reference matrix](annotation-comparison.md), [six-cell summary](evidence/annotation-reference/summary.json), and [actual notebook execution](evidence/annotation-reference/notebook-execution.json) |
| Local image model | PR154 RGB-only heatmap return; [paired local-model results](local-refinement-results.md) |
| Local image and initial-corner model | PR153 Gaussian-hint heatmap return; same architecture/initialization/order as PR154 |
| Local coordinate-correction model | PR155, completed L4 training; [results](residual-results.md) and [independent audit](evidence/residual-return/audit.json) |
| Selected residual model replay | [CPU receipt](evidence/residual-return/selected-cpu-forward.json); all 71 raw proposals from two images, unchanged scored outcomes |
| Ideal-heatmap simulation | [Synthetic-map diagnostic](evidence/refinement/ideal-heatmap-diagnostic.json); reference positions are supplied, so this is not recognition accuracy |
| Historical annotation-only training | Earlier PR35 two-arm 137-photo trial, explained separately in [annotation comparison](annotation-comparison.md); its historical scores are not pasted into the current matrix |

Main report measurements use object confidence strictly above 0.25, endpoint confidence at least 0.50, matching IoU at least 0.50 and common postclip suppression. Equal averaging affects 648 of 687 adapted original-view records. The older research runs also retain their original confidence-above-0.13 scores in technical evidence; those counts are not mixed into the main table.

The residual/control/two-view mean maximum-corner errors use the same 583 complete pairs. The supplied-model conditional mean uses 580 available matched pairs, so it is not presented as a directly comparable 583-pair mean in the report.

## Reproduction and presentation

- [Annotation-reference notebook](../../notebooks/baomai_annotation_reference_r60.ipynb): all four exact cells completed from published source, no new inference or training.
- [Original-comparison notebook](../../notebooks/baomai_original_comparison_r59.ipynb): all four cells and full 60 image/model passes completed previously.
- [Residual notebook](../../notebooks/baomai_residual_corner_r59.ipynb): completed return PR155; retained for reproduction.
- [Report](report.md), [bibliography](bibliography-fyp-i.json), and [final validation](report-validation-fyp-i.json): manuscript, external sources, final presentation counts and artifact identities.
- [Word builder](../../scripts/build_thesis_docx_r56.py) and [equation validator](../../scripts/verify_thesis_equations_r60.py): genuine centered display OMML, automatic-height equation rows, and measured whole-formula positions.

Only Figure 1 is used from the retained figure set. Older charts and dated plans remain historical assets. The selected deployment model and live defaults are separate from the thesis comparisons.
