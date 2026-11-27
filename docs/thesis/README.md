# Cassette vision thesis

Updated 27 November 2026, Asia/Hong_Kong.

**Current report:** [Cassette Front Detection and Corner Localization for Automatic Image Analysis](report.md). This folder brings the report, research notes, measured results, figure sources, and next experiment together. The report includes a 230-word abstract, decimal section numbering, 26 IEEE references in first-citation order, eight numbered equations, eight tables, and four figures. Acknowledgements are omitted as requested.

## Main findings

An earlier matched annotation experiment reduced unmatched outputs from 58 to 52 while tying misses at 24, under its original 661-front reference and confidence 0.25. That favorable result is retained alongside the later tradeoffs.

The completed [PR149 contrast](https://github.com/Siuuuers/dl/pull/149) and [PR150 scale](https://github.com/Siuuuers/dl/pull/150) experiments both pass their reproduction checks. Neither passes the joint quality screen: on the primary Frozen666 reference, misses change 34 to 33 but both-corner five-pixel successes fall from 511 to 506 and 503. The fixed cached-view combinations also increase unmatched and duplicate outputs. See [the complete R53 interpretation](r53-results.md) and [fusion evidence](evidence/cached-fusion-results.md).

The [R55 pair-output analysis](evidence/automatic-pair-results.md) distinguishes finding eligible fronts from the quality of all offered coordinate pairs. An output lacking complete reference support is not automatically a physically incorrect prediction. All these analyses reuse the same development photographs.

**PR35 remains selected under the historical rule.** This work does not change deployed model, camera, label, or confidence defaults.

## Read and run

| File | Purpose |
| --- | --- |
| [report.md](report.md) | Complete current manuscript and linked contents |
| [requirements.md](requirements.md) | Supplied report requirements and their implementation |
| [r53-results.md](r53-results.md) | Completed CLAHE/scale audit and decisions |
| [literature-r56.md](literature-r56.md) | Papers, hypotheses, and ranked next investigations |
| [bibliography-r56.json](bibliography-r56.json) | Verified metadata and first-appearance reference numbering |
| [flip-experiment-r56.md](flip-experiment-r56.md) | Explicit reflection protocol, fixed source pin, and interpretation |
| [R56 notebook](../../notebooks/baomai_flip_inference_r56.ipynb) | Four code cells for the next controlled inference comparison |
| [flip-bootstrap-r56.json](flip-bootstrap-r56.json) | Actual clean-checkout execution of notebook setup cells 1–2 |
| [figures/captions-r56.md](figures/captions-r56.md) | Detailed captions, measured data provenance, and figure reproduction |
| [evidence/README.md](evidence/README.md) | Saved-output analysis and reproduction instructions |
| [history](history) | Earlier R48–R55 thesis plans and notes, preserved as historical snapshots |

For the next model experiment, open the [R56 notebook in Colab](https://colab.research.google.com/github/Siuuuers/dl/blob/codex/thesis-r56-report-and-flip-probe-20261127/notebooks/baomai_flip_inference_r56.ipynb), select a GPU runtime, and run all four code cells. Return `baomai-flip-inference-r56-results.zip`. The source is pinned to `dc7eec630bb0b613fc1fe784e4d9818a0893658e`. The first two cells have passed in a fresh checkout; no R56 model inference or training result exists yet.

## Scope and remaining evidence

The study concerns identifiable cassette fronts and actual lower-corner image coordinates. The outputs are automatically consumed. Recognition quality is the priority; latency and hardware are recorded but are not selection constraints. Robot calibration and physical grasp accuracy are outside the present project scope. New capture sessions are unavailable for now.

There is sufficient completed evidence for a substantive empirical report. A favorable reflection result is not required to make the existing report valid. Stronger claims still need a fixed, physically adjudicated reference, a downstream pixel-tolerance definition, and later independent capture-session evaluation. A budget-controlled training-side annotation review study is a useful next contribution if time permits; it is not already a completed result.

## Rebuild the Word report

The [DOCX builder](../../scripts/build_thesis_docx_r56.py) uses Python with python-docx, Pillow, lxml, PyMuPDF, and the primary runtime’s bundled KaTeX module. Its Node/runtime-module paths are provided through `CODEX_PRIMARY_RUNTIME_NODE` and `CODEX_PRIMARY_RUNTIME_NODE_MODULES`. The document uses editable Word equations, actual multilevel heading numbering, internal contents links, and a verified page map. It does not need model inference.

```bash
python scripts/build_thesis_docx_r56.py docs/thesis/report.md --output report-first.docx
```

Render the file to PDF with an Office-compatible engine, then rebuild using that rendered PDF:

```bash
python scripts/build_thesis_docx_r56.py docs/thesis/report.md \
  --output report.docx --page-map-from-pdf report-first.pdf --require-page-map
```

Render and visually inspect the final document. The final heading-page map must agree with the rendered pages. Do not treat an unrendered file or a changed page map as a completed formatting check.
