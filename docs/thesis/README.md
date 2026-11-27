# Cassette vision: Final Year Project I

Updated 28 November 2026, Asia/Hong_Kong.

**Current report:** [Cassette Front Detection and Corner Localization](report.md), prepared as an undergraduate Final Year Project I report for the Department of Electronic Engineering. It introduces the recognition task and measurement methods for readers unfamiliar with computer vision, explains the project's methods and results directly, and uses external academic publications and professional documentation as references.

The final Word report has 29 pages, a 278-word abstract, 40 decimal-numbered headings and contents entries, 18 IEEE references in first-citation order, nine numbered equations, 11 tables, and four figures. Acknowledgements are omitted as requested. The report contains no GitHub links or unexplained internal experiment codes.

## Main findings

A matched annotation-training comparison reduced extra predictions from 58 to 52 while maintaining 24 missed fronts. Later experiments show why finding more fronts and returning better coordinates need separate measurements. Contrast enhancement and a larger input each reduced misses from 34 to 33 against the fixed reference, but five-pixel corner-pair success fell from 511 to 506 and 503. Combining processed views increased extra and duplicate predictions.

The [horizontal-reflection experiment](reflection-results.md) is complete and independently verified. Against the fixed reference, misses change from 34 to 32, extras from 91 to 84, and duplicates from 17 to 23. Five-pixel pair success changes from 511 to 518/584 under the fixed reference, but from 566 to 562/583 under the reviewed reference. This tradeoff does not support replacing the original view as an overall improvement.

A separate [coordinate-averaging analysis](corner-average-experiment.md) preserves every original box, score, and record identity. It averages 666 of 723 original prediction records using a fixed association rule based only on predicted boxes. Five-pixel pair success rises from 511 to 525/584 and from 566 to 569/583 under the fixed and reviewed references. Fixed-reference ten-pixel success falls from 575 to 574/584; reviewed ten-pixel success rises from 582 to 583/583. Detection outcomes remain unchanged. This is completed CPU analysis of saved outputs, with no new inference or training.

The automatic-output analysis shows that higher corner confidence can remove some unconfirmed pairs without correcting known localization errors. Fixed local crops contain both annotated corners of every matched complete target, including all 17 five-pixel baseline failures against the reviewed reference. Two [local-refinement notebooks](corner-refinement-experiment.md) now prepare a matched comparison of image-only input and image input with initial-corner hints. Their implementation checks include a pretrained CPU forward/backward and masked-gradient smoke run; full GPU training and recognition results remain pending.

## Read and run

| File | Purpose |
| --- | --- |
| [report.md](report.md) | Current FYP I manuscript and linked contents |
| [requirements.md](requirements.md) | Supplied report requirements and their implementation |
| [bibliography-fyp-i.json](bibliography-fyp-i.json) | External citation metadata and numbering |
| [report-validation-fyp-i.json](report-validation-fyp-i.json) | Final 29-page Word document checks and artifact hashes |
| [report-source-map.md](report-source-map.md) | Mapping from descriptive report names to exact experiment records |
| [r53-results.md](r53-results.md) | Completed contrast and input-size evidence |
| [reflection-results.md](reflection-results.md) | Completed reflection results, paired changes, and numerical checks |
| [corner-average-experiment.md](corner-average-experiment.md) | Completed coordinate-average results and CPU replay instructions |
| [CPU averaging notebook](../../notebooks/baomai_corner_average_r58.ipynb) | Reproduce the saved-output analysis without a GPU or new model inference |
| [corner-refinement-experiment.md](corner-refinement-experiment.md) | Matched local-corner training protocol and run instructions |
| [GPU image-only notebook](../../notebooks/baomai_corner_refinement_rgb_r58.ipynb) | Train the local model with RGB input and zero-valued hint channels |
| [GPU image-and-hints notebook](../../notebooks/baomai_corner_refinement_hints_r58.ipynb) | Train the same model with RGB input and initial-corner Gaussian hints |
| [research-local-refinement.md](research-local-refinement.md) | Primary papers, crop evidence, and the earlier design rationale |
| [research-corner-quality.md](research-corner-quality.md) | Saved-prediction diagnostics of confidence, disagreement, and crop feasibility |
| [evidence/corner-quality/README.md](evidence/corner-quality/README.md) | Diagnostic results, source hashes, and reproduction commands |
| [literature-r56.md](literature-r56.md) | Earlier literature review and experiment priorities |
| [flip-experiment-r56.md](flip-experiment-r56.md) | Original reflection protocol and immutable source pin |
| [Original reflection notebook](../../notebooks/baomai_flip_inference_r56.ipynb) | Reproduce the completed inference comparison if needed |
| [flip-bootstrap-r56.json](flip-bootstrap-r56.json) | Historical verification of the reflection notebook setup cells |
| [figures-r57/captions.md](figures-r57/captions.md) | Current figure captions and provenance |
| [evidence/README.md](evidence/README.md) | Earlier saved-output evidence and reproduction instructions |
| [history](history) | Earlier thesis plans and notes |

The reflection return has already been audited; rerunning that GPU notebook is unnecessary for the CPU averaging study. Its original source remains pinned to `dc7eec630bb0b613fc1fe784e4d9818a0893658e` for reproducibility. The three new notebooks have separate purposes: one replays completed saved-output evidence, and two request new GPU training. Follow their individual guides rather than combining their output archives.

## Working decisions

The target is each identifiable cassette front and its actual lower-corner image coordinates. The application consumes outputs automatically. Recognition quality is the priority. New captures are unavailable for now, so the existing photographs support development comparisons. The report keeps the fixed comparison annotations separate from their reviewed version and keeps annotation changes separate from model changes.

The selected deployment model and live defaults are unchanged by these investigations. Repository experiment codes remain useful for reproducing the work; the report uses descriptive names and explains its conclusions. Current report and figure sources supersede the earlier report presentation. Older bibliography, figure assets, validation records, and dated plans retain their historical meaning.

## Rebuild the Word report

The [DOCX builder](../../scripts/build_thesis_docx_r56.py) uses python-docx, Pillow, lxml, PyMuPDF, and the primary runtime's bundled KaTeX module. Node/runtime-module paths are supplied by `CODEX_PRIMARY_RUNTIME_NODE` and `CODEX_PRIMARY_RUNTIME_NODE_MODULES`. The document uses editable Word equations, actual multilevel heading numbering, linked contents, and a verified page map.

```bash
"$CODEX_PRIMARY_RUNTIME_PYTHON" scripts/build_thesis_docx_r56.py docs/thesis/report.md --output report-first.docx
```

Render the document to PDF with the canonical documents renderer, then rebuild using that PDF:

```bash
"$CODEX_PRIMARY_RUNTIME_PYTHON" scripts/build_thesis_docx_r56.py docs/thesis/report.md \
  --output cassette-vision-fyp-i-report.docx \
  --page-map-from-pdf report-first.pdf --require-page-map
```

Render and inspect every final page. Check that each contents page number matches the rendered heading. The final Word file is delivered separately; the repository contains its source and validation record.
