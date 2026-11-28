# Completed crop context and scale comparison

The frozen reviewed-label model was tested with two overlapping full-height crops at two input sizes. Neither candidate preserves the full-image control's corner accuracy. The enlarged-crop candidate also adds many extra predictions. Full-image inference remains the simpler supported choice at these settings.

| Method on r67 | Misses | Extras | Duplicate subset | Both corners within 5 px | Both within 10 px |
|---|---:|---:|---:|---:|---:|
| Fresh full-image two-view control | 18 | 50 | 14 | 572/583 | 583/583 |
| Add crops at original scale | 17 | 51 | 11 | 572/583 | 581/583 |
| Add enlarged crops | 17 | 106 | 44 | 566/583 | 581/583 |

The primary reference has 658 fronts, including 75 incomplete fronts and 583 complete corner pairs. All arms produce zero detections on the three empty photographs. The [full matrix](results.md) retains all four annotation references. The [research note](../../research-crop-context-r71.md) explains the motivation, applied papers and decision.

## Fixed source and completed execution

The [protocol](protocol.json) and implementation were published before crop inference at commit `5da7aa4f903dceb1e8300234ff98aebbdff06c17`. The [four-cell notebook](../../../../notebooks/baomai_crop_context_r71.ipynb) was published at `2ad9361ef4d468efdf6580418ee30a4d891cb0eb` and pins that implementation.

All four actual cells passed unchanged in one local IPython session on CPU. The [execution receipt](notebook-execution.json) binds the notebook, clean source checkout, input hashes, package and cell logs. The [executed copy](executed-notebook.ipynb) preserves captured outputs. Source and input acquisition used preseeded, reverified copies, so the receipt distinguishes this execution from hosted Colab and live downloading. No user GPU run is needed for the completed result.

The fresh baseline exactly reproduces the earlier R69 CPU records. Its duplicate count is 14. The separate archived L4 result has 13 duplicates and is not used as the treatment control.

## What the independent replay checks

The [independent verification](independent-verification.json) uses a [standalone verifier](verify_results.py) that imports no repository functions. It reconstructs the baseline, crop mappings, filters, pooled suppression and evaluation from the stored records:

- 2,294 retained native crop records and 382 artificial-boundary rejections.
- Every pre-clipping point decision and mapped coordinate.
- Twelve method/reference rows and 240 per-image score objects.
- 5,284 paired target records and 8,516 output classifications.
- Known-corner outcomes, final ZIP member hashes and the four executed cells.

The [interpretation audit](independent-interpretation.json) joins recovered targets to their earlier proposal-loss stages. Original-scale crops recover one of the twelve confidence failures, while enlarged crops recover two. Neither recovers a geometry or native-NMS failure. The three restored focus misses remain. All four ten-pixel pair-loss records have valid crop points and directly replace a better baseline record through score-ranked suppression.

The [training-size audit](training-size-audit.json) separately verifies the 137-photo reviewed label set and its five supplementary photographs. It distinguishes nominal box-size counts from effective augmented training exposure.

## Complete result archive

[baomai-crop-context-r71-results.zip](baomai-crop-context-r71-results.zip) contains every retained native crop record, unclipped corner, mapping decision, source identity, suppression decision and scored outcome, together with the frozen protocol and source identity. Its [inventory](archive-inventory.json) includes `source.json` and matches the final ZIP.

| File | Bytes | SHA-256 |
|---|---:|---|
| Complete result ZIP | 3,322,927 | `99e18b001b0107d0728802042616548f9045cc8b2d8ec041ea22e3dc67e47c9d` |
| Fixed protocol | 9,689 | `ae0dbd0e183aacb3f27c14b4691655c1e9f57823698188baa844c857291c42b7` |
| Summary | 34,648 | `1ff056381446648de939c9942af40286e1b70818a9eacf0b8c1bf399df6e361f` |

To verify without running the model, extract the ZIP into an empty directory and run:

```bash
python verify_results.py \
  --results /absolute/path/to/extracted/results \
  --receipt /absolute/path/to/new-verification.json \
  --archive /absolute/path/to/baomai-crop-context-r71-results.zip \
  --execution-receipt /absolute/path/to/notebook-execution.json
```

Keep the four cell logs beside the execution receipt. The archive and execution arguments are optional. The verifier was also tested against a fresh archive extraction.

This archive covers outputs retained by native crop inference. It does not inventory all raw low-confidence crop candidates. The comparison uses the existing development photographs, with no optimizer updates or annotation changes.
