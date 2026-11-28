# Cassette vision Final Year Project I

Updated 28 November 2026, Asia/Hong_Kong.

**[Current report](report.md)** explains cassette-front detection and lower-corner localization for undergraduate Electronic Engineering readers. It compares the proposed method with the untouched supplied baomai model, explains project results directly, and cites external academic or professional sources. Acknowledgements are omitted as requested.

The Word report uses six centered display equations with separate right-aligned numbers and automatic row height. The [validation record](report-validation-fyp-i.json) contains the exact page count, abstract length, linked contents map, equation geometry and all-page visual checks. It is the authoritative source for presentation counts.

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
| [Report](report.md) | Current concise manuscript |
| [Requirements](requirements.md) | Supplied report requirements and their implementation |
| [Bibliography](bibliography-fyp-i.json) | External references in IEEE first-citation order |
| [Word validation](report-validation-fyp-i.json) | Final document, contents, equation and visual checks |
| [Report source map](report-source-map.md) | Exact technical provenance |
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

Outputs are used automatically, and recognition quality is the priority. The report measures misses, extras, duplicates and corner accuracy together. The 20 photographs have guided development and annotation review; independent captures are still needed.

A future annotation-only training comparison must keep photographs, starting weights, training settings and evaluation labels fixed, changing only training labels. The earlier 137-photo paired trial is documented in the annotation guide; it does not isolate every revision in the current 142-photo method. Local-model training also needs errors representative of photographs unseen by the proposal detector.

The selected deployment model and live defaults are unchanged. Technical experiment identifiers remain in this folder; the report uses descriptive names.

## Rebuild the Word report

The [builder](../../scripts/build_thesis_docx_r56.py) creates editable native Office Math display paragraphs inside symmetric borderless equation rows. The whole formula is centered, its number occupies a separate right-hand cell, and neither line nor row has a fixed height. Ordinary labels render upright and vector symbols retain their intended bold appearance.

    "$CODEX_PRIMARY_RUNTIME_PYTHON" scripts/build_thesis_docx_r56.py docs/thesis/report.md --output report-first.docx

Render using the canonical documents renderer, then rebuild with its PDF page map:

    "$CODEX_PRIMARY_RUNTIME_PYTHON" scripts/build_thesis_docx_r56.py docs/thesis/report.md --output cassette-vision-fyp-i-report.docx --page-map-from-pdf report-first.pdf --require-page-map

Render the final document, run [equation geometry verification](../../scripts/verify_thesis_equations_r60.py), inspect every page and confirm the contents page numbers. The final Word file is delivered separately.
