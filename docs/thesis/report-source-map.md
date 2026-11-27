# Report source map

This technical companion maps the report's descriptive names to reproducible records. The manuscript itself explains its methods and results and cites external academic or professional sources. Updated 28 November 2026, Asia/Hong_Kong.

## Annotation references

| Report name | Internal source | Fronts | Complete pairs | Use |
| --- | --- | ---: | ---: | --- |
| Reviewed reference | `r50` | 654 | 583 | Main measurement reference on 20 development photographs |
| Earlier fixed reference | `frozen666` | 666 | 584 | Sensitivity comparison in Appendix 11.1 on the same photographs |
| Supplied validation labels | Original trainer validation | 664 | See training records | Training monitor; not silently equated with either comparison reference |

The earlier `historical663` and `r38` references remain in technical evidence but are not additional test sets or separate report comparisons. Reviewed/expanded training uses 142 photographs, 3999 fronts and 7259 known corners; supplied training uses 137 photographs, 3772 fronts and 6839 known corners.

## Models and measurements

| Report description | Exact evidence |
| --- | --- |
| Supplied model | Untouched epoch-173 checkpoint, SHA-256 `e0aab1b377491700318e0b14cd4c558ec3c1684087a13b7dde2298ebdbf2847e` |
| Adapted detector | PR137 checkpoint, SHA-256 `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9` |
| Proposed method | Adapted detector plus original/reflected equal-coordinate averaging, with original boxes and scores preserved |
| Main original-model comparison | [Fresh common-runtime comparison](original-comparison.md), [fresh summary](evidence/original-baseline/fresh/summary.json) and [execution receipt](evidence/original-baseline/fresh/notebook-execution.json) |
| Local image model | PR154 completed RGB-only arm; two hint input channels are zero |
| Local image and initial-corner model | PR153 completed Gaussian-hint arm; same architecture, data, initializer and sample orders |
| Local-model measurements | [Returned-run analysis](local-refinement-results.md), [paired audit](evidence/refinement/paired-analysis.json), [full evidence inventory](evidence/refinement/README.md) |
| Ideal-heatmap numerical check | [Synthetic reference-centered heatmaps](evidence/refinement/ideal-heatmap-diagnostic.json); not a measured recognition result |
| Proposed residual correction | [Next pilot](residual-correction-experiment.md); full training and accuracy remain pending |

The main comparison filters both model views at object confidence strictly above 0.25 before common postclip suppression and association. Corner confidence is at least 0.50. It averages 648 of 687 retained adapted original-view outputs. The earlier saved-output averaging study used confidence above 0.13 and averaged 666 of 723; these protocols must not be mixed. The fresh comparison reruns both complete checkpoints and all three image/model views in one FP32 CPU environment, and reproduces the historical per-object outcomes.

The thesis's local-model table is rescored at the same object threshold above 0.25. The technical audit also preserves original returned scores above 0.13 and the adapted-detector control for diagnosis; that control is not the untouched supplied model. PR35 remains the selected deployment candidate under the established selection gate, separately from the report's original-baseline comparison.

## Current notebooks and presentation

| File | Status |
| --- | --- |
| [Original comparison](../../notebooks/baomai_original_comparison_r59.ipynb) | All four actual code cells and full 60 image/model passes completed locally; browser download interface not exercised |
| [Residual correction](../../notebooks/baomai_residual_corner_r59.ipynb) | Next training pilot; full training and recognition results pending |
| [Image-only local model](../../notebooks/baomai_corner_refinement_rgb_r58.ipynb) | Completed return PR154, audited |
| [Image-plus-hints local model](../../notebooks/baomai_corner_refinement_hints_r58.ipynb) | Completed return PR153, audited |
| [Report](report.md) | 207-word abstract, 25 headings, 12 references, six equations, four tables, one figure; 14 Word pages |
| [Bibliography](bibliography-fyp-i.json) | Current external citation order and verification metadata |
| [Word validation](report-validation-fyp-i.json) | Artifact hashes, stable contents map, structural checks and all-page visual review |

Only Figure 1, the photographed cassette front and labelled corners, is used from [the retained figure set](figures-r57/captions.md). Earlier charts and reports remain historical assets. The Word builder's filename, `scripts/build_thesis_docx_r56.py`, is an implementation identifier, not report terminology.
