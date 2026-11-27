# Final Year Project I workspace

Start with the **[thesis folder](thesis/README.md)** and **[current report](thesis/report.md)**. The concise undergraduate Electronic Engineering report explains cassette-front detection and lower-corner localization directly, using the untouched supplied baomai model as its comparison baseline. It cites external academic and professional sources, with no GitHub links or unexplained project codes.

The Word report has 14 pages, a 207-word abstract, 25 numbered headings, 12 IEEE references, six centered editable equations, four tables and one figure. Acknowledgements are omitted as requested.

The [fresh comparison](thesis/original-comparison.md) runs both models under the same software and operating settings. On the reviewed annotations, the proposed method reduces misses 34→18 and extras 61→51; duplicates remain five. Complete pairs within five pixels increase 555→569/583, and within ten pixels 577→583/583. The earlier fixed annotations favour the supplied model (534 versus 525 five-pixel successes), which the report explains and retains in an appendix.

Both [local-refinement training returns](thesis/local-refinement-results.md) are complete and independently audited. The image-only and image-plus-hints models achieve 547 and 558 reviewed five-pixel successes. Neither exceeds the proposed method's 569. A numerical ideal-heatmap check and saved-model replay point to learned peak placement as a useful next focus.

| Notebook | Purpose |
| --- | --- |
| [Original-model comparison](../notebooks/baomai_original_comparison_r59.ipynb) | Optional reproduction of the completed fresh inference comparison; all four cells already ran locally |
| [Residual corner correction](../notebooks/baomai_residual_corner_r59.ipynb) | Next GPU training pilot; learns offsets from available initial points with a zero-initialized output head |

Outputs are used automatically, and recognition quality is the priority. The report measures misses, extras, duplicates and corner accuracy together. The current results use repeatedly inspected development photographs; independent captures remain needed. The selected deployment model and live defaults are unchanged.

Exact technical provenance is in the [source map](thesis/report-source-map.md). Earlier protocols and results remain linked from the thesis folder; dated plans in [history](thesis/history) retain their historical meaning.
