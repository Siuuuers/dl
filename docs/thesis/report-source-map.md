# Report source map

This technical companion maps the descriptive names in the Final Year Project I report to reproducible repository records. The manuscript explains its methods and conclusions directly and cites only external academic or professional sources. Readers of the report do not need these internal names. Updated 28 November 2026, Asia/Hong_Kong.

## Annotation references

| Report name | Internal key or source | Fronts | Two known corners | Interpretation |
| --- | --- | ---: | ---: | --- |
| Fixed reference | `frozen666` | 666 | 584 | Fixed main comparison reference; it is not the untouched supplied trainer validation set |
| Reviewed reference | `r50` | 654 | 583 | Later reviewed annotations for the same 20 development photographs |
| Earlier reference | `historical663` | 663 | 584 | Intermediate annotation version retained in Appendix 11.1 |
| Intermediate review | `r38` | 654 | 583 | Earlier reviewed version retained in Appendix 11.1 |
| Annotation-training experiment reference | [Seed-46 confirmation](../annotation-confirmation-2026-11-11.md) | 661 | 584 | Separate historical comparison, box confidence 0.25 |

The supplied trainer validation has 664 targets; it must not be silently equated with the fixed 666-front reference. See [training annotation records](../training-annotations-seed46.md). All four later scoring references describe the same 20 development photographs.

## Models and experiments

| Report description | Repository evidence |
| --- | --- |
| Supplied model | Upstream cassette checkpoint at epoch 173; verified by the existing checkpoint contract |
| Original-label and reviewed-label models | [Annotation-training confirmation](../annotation-confirmation-2026-11-11.md); the reviewed-label model is the PR35 candidate |
| Reviewed-label model in the digital-mask study | [Controlled image masking](../pixel-occlusion-2026-11-13.md); this is not the later contrast/scale baseline |
| Later baseline / expanded-data baseline | PR137 candidate, reproduced as the R51 reference; [training results](../partial-classification-r51-results.md) |
| Partial-front-weighted model | PR146/R51 candidate; weight 2 for assigned targets with unavailable corners |
| Contrast enhancement | [PR149](https://github.com/Siuuuers/dl/pull/149), unchanged recipe plus Lab-lightness CLAHE |
| Larger input | [PR150](https://github.com/Siuuuers/dl/pull/150), input size 1280 versus 1600 |
| Combined views | [Fixed cached-view replay](evidence/cached-fusion-results.md) |
| Coordinate-pair acceptance | [Automatic-pair analysis](evidence/automatic-pair-results.md), box score strictly above 0.20 |
| Local crop feasibility | [Corner diagnostic evidence](evidence/corner-quality/README.md), saved baseline/candidate predictions at box confidence above 0.13 |
| Horizontal reflection | [Completed PR152 results](reflection-results.md), [independent evidence](evidence/reflection/README.md), and [original protocol](flip-experiment-r56.md) |
| Averaging corners while preserving detections | [Completed fixed-rule CPU analysis](corner-average-experiment.md) and [saved summary](evidence/corner-average/summary.json); no new inference or training |
| Proposed local corner refinement | [Prepared training comparison](corner-refinement-experiment.md); GPU training results pending |

The later baseline checkpoint SHA-256 is `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9`. Reflection uses that same checkpoint, as does the original-view detection set preserved by the coordinate-average analysis.

## Current notebooks

| Notebook | Work and status |
| --- | --- |
| [Coordinate average](../../notebooks/baomai_corner_average_r58.ipynb) | CPU replay of the completed saved-output analysis; no model inference |
| [Image-only local refinement](../../notebooks/baomai_corner_refinement_rgb_r58.ipynb) | New GPU training with RGB input and zero-valued hint channels |
| [Image and initial-corner hints](../../notebooks/baomai_corner_refinement_hints_r58.ipynb) | Matched GPU training with the same architecture and Gaussian initial-corner hints |
| [Original reflection comparison](../../notebooks/baomai_flip_inference_r56.ipynb) | Completed inference experiment; original source pin retained for reproduction |

The two refinement arms have implementation and CPU smoke checks; their full training and accuracy results are not yet available. The [training guide](corner-refinement-experiment.md) specifies the actual model, data, initialization, and fixed comparison settings.

## Current presentation files

- [Report](report.md) is the manuscript source: 278-word abstract, 40 headings, 18 references, nine equations, 11 tables, and four figures. The delivered Word report has 29 pages.
- [Current bibliography](bibliography-fyp-i.json) records external first-citation numbering and verification metadata. Earlier numbering remains in `bibliography-r56.json` for historical traceability.
- [Current figures](figures-r57/captions.md) retain their measured data and reader-facing labels. Crop feasibility is Table 8, reflection is Table 9, and coordinate averaging is Table 10.
- [Current validation](report-validation-fyp-i.json) records the final Word artifact hash, stable contents page map, structural checks, and visual inspection of all 29 pages.
- The Word builder remains `scripts/build_thesis_docx_r56.py`; its filename is an implementation identifier, not report terminology.
- The completed reflection experiment keeps its original notebook and immutable source pin. The separate CPU and training notebooks do not change that returned evidence.
