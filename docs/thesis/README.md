# Cassette vision: Final Year Project I

Updated 28 November 2026, Asia/Hong_Kong.

**Current report:** [Cassette Front Detection and Corner Localization](report.md), prepared as an undergraduate Final Year Project I report for the Department of Electronic Engineering. It introduces the recognition task and measurement methods for readers unfamiliar with computer vision. Completed experiments establish the findings; the final discussion sets out the investigations for Project II.

The manuscript includes a 231-word abstract, 38 decimal-numbered headings and contents entries, 29 IEEE references in first-citation order, eight numbered equations, nine tables, and four figures. Acknowledgements are omitted as requested.

## Main findings

A matched annotation-training comparison reduced extra predictions from 58 to 52 while maintaining 24 missed fronts. Later experiments show why finding more fronts and returning better coordinates need separate measurements. Contrast enhancement and a larger input each reduced misses from 34 to 33 against the fixed reference, but five-pixel corner-pair success fell from 511 to 506 and 503. Combining processed views increased extra and duplicate predictions.

The automatic-output analysis shows that higher corner confidence can remove some unconfirmed pairs without correcting known localization errors. A new fixed-crop feasibility check contains both annotated corners of every matched complete target, including all 17 five-pixel failures against the reviewed reference. This supports testing a separate local corner model; no refiner has been trained yet.

The horizontal reflection notebook is the current model experiment. Its returned results are still pending. New capture sessions will be needed to assess transfer to unfamiliar scenes.

## Read and run

| File | Purpose |
| --- | --- |
| [report.md](report.md) | Current FYP I manuscript and linked contents |
| [requirements.md](requirements.md) | Supplied report requirements and their implementation |
| [bibliography-fyp-i.json](bibliography-fyp-i.json) | Current citation metadata and numbering |
| [report-validation-fyp-i.json](report-validation-fyp-i.json) | Final 25-page Word document checks and artifact hashes |
| [report-source-map.md](report-source-map.md) | Mapping from plain-language report names to exact experiment records |
| [r53-results.md](r53-results.md) | Completed contrast and input-size evidence |
| [research-local-refinement.md](research-local-refinement.md) | Primary papers and a controlled crop-refinement experiment design |
| [research-corner-quality.md](research-corner-quality.md) | Saved-prediction diagnostics of confidence, disagreement, and crop feasibility |
| [evidence/corner-quality/README.md](evidence/corner-quality/README.md) | New diagnostic results, source hashes, and reproduction commands |
| [literature-r56.md](literature-r56.md) | Earlier literature review and experiment priorities |
| [flip-experiment-r56.md](flip-experiment-r56.md) | Reflection protocol and immutable source pin |
| [Reflection notebook](../../notebooks/baomai_flip_inference_r56.ipynb) | Four code cells for the controlled inference comparison |
| [flip-bootstrap-r56.json](flip-bootstrap-r56.json) | Verified execution of the published notebook setup cells |
| [figures-r57/captions.md](figures-r57/captions.md) | Current figure captions and provenance |
| [evidence/README.md](evidence/README.md) | Earlier saved-output evidence and reproduction instructions |
| [history](history) | Earlier thesis plans and notes |

The [reflection notebook in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/thesis-r56-report-and-flip-probe-20261127/notebooks/baomai_flip_inference_r56.ipynb) returns `baomai-flip-inference-r56-results.zip`. Its source stays pinned to `dc7eec630bb0b613fc1fe784e4d9818a0893658e` while the run is in progress. The first two setup cells passed in a fresh checkout. The report revision and CPU diagnostics do not change its input or implementation.

## Working decisions

The target is each identifiable cassette front and its actual lower-corner image coordinates. The application consumes outputs automatically. Recognition quality is the priority. New captures are unavailable for now, so the existing photographs support development comparisons. The report keeps the fixed comparison annotations separate from their reviewed version and keeps annotation changes separate from model changes.

Repository experiment codes remain useful for reproducing the work. The report uses descriptive names so it can be read without knowing that history. Current report and figure sources supersede the earlier report presentation; the older bibliography, figure assets, and validation record retain their historical meaning.

## Rebuild the Word report

The [DOCX builder](../../scripts/build_thesis_docx_r56.py) uses python-docx, Pillow, lxml, PyMuPDF, and the primary runtime's bundled KaTeX module. Node/runtime-module paths are supplied by `CODEX_PRIMARY_RUNTIME_NODE` and `CODEX_PRIMARY_RUNTIME_NODE_MODULES`. The document uses editable Word equations, actual multilevel heading numbering, linked contents, and a verified page map.

```bash
python scripts/build_thesis_docx_r56.py docs/thesis/report.md --output report-first.docx
```

Render the document to PDF with an Office-compatible engine, then rebuild using that PDF:

```bash
python scripts/build_thesis_docx_r56.py docs/thesis/report.md \
  --output cassette-vision-fyp-i-report.docx \
  --page-map-from-pdf report-first.pdf --require-page-map
```

Render and inspect the final document. Check that every contents page number matches the rendered heading. The final Word file is delivered separately; the repository contains its source and validation record.
