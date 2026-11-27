# Final Year Project I workspace

Start with the **[thesis folder](thesis/README.md)** and the **[current report](thesis/report.md)**, prepared for undergraduate Electronic Engineering readers. The manuscript explains cassette-front detection and lower-corner localization without requiring knowledge of repository revision codes or project links. It explains the methods and conclusions directly and cites external academic and professional sources. The final Word report has 29 pages, a 278-word abstract, 40 numbered headings, 18 IEEE references, nine equations, 11 tables, four figures, and a Project II plan. Acknowledgements are omitted as requested.

The [contrast and input-size results](thesis/r53-results.md) and [horizontal-reflection results](thesis/reflection-results.md) are complete. Reflection reduces misses and total extras but increases duplicates, and its five-pixel corner change has opposite signs under the fixed and reviewed annotations. A separate [coordinate-average analysis](thesis/corner-average-experiment.md) keeps the original detections unchanged and improves five-pixel pair success under both references, while losing one fixed-reference ten-pixel success. This is completed analysis of saved outputs.

The [corner-quality diagnostic](thesis/research-corner-quality.md) shows that fixed local crops contain both reference corners in every matched complete target. The [local-refinement experiment guide](thesis/corner-refinement-experiment.md) prepares the next matched GPU training comparison: RGB input with zero hints versus RGB input with initial-corner Gaussian hints. Full training and recognition results remain pending.

| Next notebook | Runtime and purpose |
| --- | --- |
| [Coordinate-average replay](../notebooks/baomai_corner_average_r58.ipynb) | CPU; reproduce completed results without new inference |
| [Image-only local refinement](../notebooks/baomai_corner_refinement_rgb_r58.ipynb) | GPU; train the zero-hint control |
| [Image and corner hints](../notebooks/baomai_corner_refinement_hints_r58.ipynb) | GPU; train the matched initial-corner-hint treatment |

The application uses outputs automatically, and recognition quality is the priority. The report evaluates missed fronts, extra predictions, duplicates, and corner accuracy together. The selected deployment model and live defaults remain unchanged. All present comparisons use existing development photographs; later capture sessions will test transfer.

Technical names and exact evidence paths are in the [report source map](thesis/report-source-map.md). Earlier plans remain in [history](thesis/history); their pending-work statements describe their original dates.
