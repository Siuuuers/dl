# Cassette vision: Final Year Project I

Updated 28 November 2026, Asia/Hong_Kong.

**Current report:** [Cassette Front Detection and Corner Localization](report.md), an undergraduate Electronic Engineering report written for readers unfamiliar with computer vision. It explains the methods and results directly, with the untouched supplied baomai checkpoint as the comparison baseline and external academic or professional references.

The concise Word report has 14 pages, a 207-word abstract, 25 decimal-numbered headings, 12 IEEE references in first-citation order, six centered editable equations, four tables, and one figure. Acknowledgements are omitted as requested. It contains no GitHub links or unexplained internal experiment codes.

## Current findings

The [fresh original-model comparison](original-comparison.md) runs the supplied checkpoint and the proposed method on all 20 photographs in one CPU software environment, at the same object threshold above 0.25. The proposed method uses the adapted detector and equal averaging of associated original/reflected corner predictions.

| Reviewed annotations: 654 fronts, 583 complete pairs | Supplied model | Proposed method |
| --- | ---: | ---: |
| Missed fronts | 34 | 18 |
| Extra predictions | 61 | 51 |
| Duplicates, included in extras | 5 | 5 |
| Both corners within 5 pixels | 555/583 | 569/583 |
| Both corners within 10 pixels | 577/583 | 583/583 |

The earlier fixed annotations reverse the five-pixel comparison: 534/584 for the supplied model and 525/584 for the proposed method. The report retains this result in a short appendix and explains the dependence on reviewed annotations and repeatedly inspected development photographs. These measurements do not establish automatic-output reliability on new captures.

Both returned [local-refinement runs](local-refinement-results.md) are complete and audited. At the common object threshold, the RGB model gives 547/583 five-pixel successes and the image-plus-initial-corner model gives 558/583. Neither reaches the proposed method's 569/583. Their boxes remain fixed, so these models cannot recover missed fronts or remove duplicate detections. An ideal-heatmap numerical check shows that the crop and decoder can represent all 583 complete reference pairs within five pixels; learned peak locations remain a concrete source of error.

The next focused experiment is [zero-initialized residual correction](residual-correction-experiment.md): learn an offset from an available initial point, starting with exactly unchanged coordinates. Full training and recognition results for this new method remain pending.

## Read and run

| File | Purpose |
| --- | --- |
| [Report](report.md) | Concise manuscript, explanatory methods, results and limitations |
| [Requirements](requirements.md) | Supplied report requirements and their implementation |
| [Bibliography](bibliography-fyp-i.json) | External reference metadata and citation order |
| [Word validation](report-validation-fyp-i.json) | Final artifact hashes, contents map and all-page visual checks |
| [Report source map](report-source-map.md) | Exact technical provenance kept separate from the manuscript |
| [Original comparison](original-comparison.md) | Common-runtime results and comparison procedure |
| [Original-comparison notebook](../../notebooks/baomai_original_comparison_r59.ipynb) | Reproduce the completed full inference comparison; already run locally |
| [Original-comparison evidence](evidence/original-baseline/README.md) | Original model identity, fresh predictions, score replay and execution receipt |
| [Local-refinement results](local-refinement-results.md) | Both returned training runs, paired changes, audits and failure diagnosis |
| [Refinement evidence](evidence/refinement/README.md) | Reproducible audit and ideal-heatmap results |
| [Residual-correction experiment](residual-correction-experiment.md) | One new training hypothesis and fixed protocol |
| [Residual-correction notebook](../../notebooks/baomai_residual_corner_r59.ipynb) | Prepare, train, evaluate and export the next GPU pilot |

Earlier work remains available: [reflection](reflection-results.md), [coordinate averaging](corner-average-experiment.md), [contrast and input size](r53-results.md), [local-model protocol](corner-refinement-experiment.md), [research notes](research-local-refinement.md), and [history](history). Their original notebooks and immutable source pins retain their historical meaning. Completed runs do not need to be repeated before starting the residual pilot.

## Working decisions

The target is each identifiable cassette front and its actual lower-corner image coordinates. Outputs are used automatically, and recognition quality is the priority. New captures are unavailable for now. The report therefore measures missed fronts, extra and duplicate outputs, and corner accuracy together, and clearly identifies the present data as development photographs.

The selected deployment model and live defaults are unchanged. A completed thesis comparison is not automatic model promotion. Repository names support reproducibility; the report uses descriptive terms.

## Rebuild the Word report

The [DOCX builder](../../scripts/build_thesis_docx_r56.py) uses python-docx, Pillow, lxml, PyMuPDF, and the primary runtime's bundled KaTeX module. Runtime paths come from `CODEX_PRIMARY_RUNTIME_NODE` and `CODEX_PRIMARY_RUNTIME_NODE_MODULES`. It creates editable centered equations with right-aligned numbers, real multilevel heading numbering, linked contents, and a verified page map. The main body flows between sections while front matter, Introduction, References and Appendices start on fresh pages.

```bash
"$CODEX_PRIMARY_RUNTIME_PYTHON" scripts/build_thesis_docx_r56.py docs/thesis/report.md --output report-first.docx
```

Render the document with the canonical documents renderer, then rebuild using its PDF:

```bash
"$CODEX_PRIMARY_RUNTIME_PYTHON" scripts/build_thesis_docx_r56.py docs/thesis/report.md \
  --output cassette-vision-fyp-i-report.docx \
  --page-map-from-pdf report-first.pdf --require-page-map
```

Render and inspect every final page, including the contents page numbers. The Word file is delivered separately; the repository contains its source and validation record.
