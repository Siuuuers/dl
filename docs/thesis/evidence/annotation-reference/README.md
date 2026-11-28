# Annotation-reference comparison evidence

This analysis keeps the original and proposed model predictions fixed and scores them against supplied, earlier fixed and reviewed annotations of the same 20 photographs. It performs no model inference, training or annotation edits. See [the interpretation and protocol](../../annotation-comparison.md).

## Inputs

- Existing fresh comparison ZIP: `../original-baseline/fresh/baomai-original-comparison-r59-results.zip`, 1,133,014 bytes, SHA-256 `aae05d14298afad0263197fe4ab7e0e753f63445d9d3503dc1b006030e16c516`.
- Original annotation manifest: `data/manifest.json`, SHA-256 `72ae8365d5abf0b45b75d400a2e6d22119c0e31b54ee33035feb8edc1f967926`.
- The original manifest was directly fetched at upstream baomai commit `351eb3cf960ccc34829e2bc72e1fa4645acc6295` and compared byte for byte. All 20 original validation labels match its recorded hashes.
- Later label/manifest identities and the shared inference settings are bound to the fresh archive contract. The existing fixed and reviewed results must reproduce exactly.

## Files

| File | Meaning |
| --- | --- |
| [summary.json](summary.json) | Complete six-cell matrix, input identities and per-reference paired changes |
| [upstream-provenance.json](upstream-provenance.json) | Direct immutable-upstream manifest equality and all original-label hashes |
| [prepublication-check.json](prepublication-check.json) | Completed source-level calculation and notebook compilation; not a published-source execution claim |
| [notebook-execution.json](notebook-execution.json) | All four actual code cells passed from a fresh fetch of the published implementation |
| [verification.json](verification.json) | Complete saved-result replay passed |
| [preflight.json](preflight.json) | Immutable source and exact input identities |
| [executed-notebook.ipynb](executed-notebook.ipynb) | Exact delivered cell sources with captured outputs |
| [baomai-annotation-reference-r60-results.zip](baomai-annotation-reference-r60-results.zip) | Full labels, predictions, score tables, associations and paired outcomes |
| [manifest.json](manifest.json) | Evidence-file sizes and SHA-256 hashes; cell logs use `.txt` files |

The original model gives 521/584 five-pixel successes under the supplied annotations and the proposed method 514/584. The earlier fixed result is 534 versus 525/584, and the reviewed result is 555 versus 569/583. Different annotations change measured agreement, not the stored predictions. The references are not independent test sets.

The four-cell notebook is pinned to `8ee1d751817638d72eb7b385a7f4164478a2b7e8`. All four unchanged code cells completed in an in-process IPython shell, including a fresh source fetch, all six matrix cells, exact reproduction of the existing reference results, archive creation and full saved-result replay. The source checkout remained clean. This does not claim that the Colab browser-download interface was exercised.

The notebook SHA-256 is `de1142323805e6f855277c4e36496031c7cdffaba844c96126e5277416205d69`; the driver SHA-256 is `2939b2789abf2566d7680de60a0a87efae2a41e45c2dc01c9d15d1221cb8bbf1`. The complete result archive SHA-256 is `ce8df4bcf223b6bee49fa67e829a443acb572e4a6aad75fa0f92657a16cc2601`. No GPU or third-party scientific package is required for the calculation.
