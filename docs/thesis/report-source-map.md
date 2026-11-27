# Report source map

This technical companion maps the descriptive names in the Final Year Project I report to reproducible repository records. Readers of the report do not need these internal names. Updated 28 November 2026, Asia/Hong_Kong.

## Annotation references

| Report name | Internal key or source | Fronts | Two known corners | Interpretation |
| --- | --- | ---: | ---: | --- |
| Fixed reference | `frozen666` | 666 | 584 | Fixed main comparison reference; it is not the untouched supplied trainer validation set |
| Reviewed reference | `r50` | 654 | 583 | Later reviewed annotations for the same 20 development photographs |
| Earlier reference | `historical663` | 663 | 584 | Intermediate annotation version retained in Appendix 11.1 |
| Intermediate review | `r38` | 654 | 583 | Earlier reviewed version retained in Appendix 11.1 |
| Annotation-training experiment reference | [seed-46 confirmation](../annotation-confirmation-2026-11-11.md) | 661 | 584 | Separate historical comparison, box confidence 0.25 |

The supplied trainer validation has 664 targets; it must not be silently equated with the fixed 666-front reference. See [training annotation records](../training-annotations-seed46.md).

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
| Horizontal reflection | [Pinned notebook protocol](flip-experiment-r56.md); result pending |

The later baseline checkpoint SHA-256 is `222bf3c91a651183047a10eb1a50e16ae8fe999f15372afe005bfca2d0b157f9`.

## Current presentation files

- [Report](report.md) is the manuscript source.
- [Current bibliography](bibliography-fyp-i.json) records first-citation numbering and verification metadata. Earlier bibliography numbering remains in `bibliography-r56.json` for historical traceability.
- [Current figures](figures-r57/captions.md) retain the same measured data as the earlier figure set with reader-facing labels. The new feasibility result is Table 8.
- The Word builder remains `scripts/build_thesis_docx_r56.py`; its filename is an implementation identifier, not report terminology.
- The reflection notebook and immutable source pin are unchanged during this presentation revision.
