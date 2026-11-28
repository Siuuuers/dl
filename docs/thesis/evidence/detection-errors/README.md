# Detection error audit evidence

This completed CPU analysis reuses the frozen annotation-reference archive. It changes no model, prediction, annotation, threshold or matching rule.

The [interpretation](../../detection-errors.md) connects the results to the next model experiment. The [analysis source](../../../../scripts/analyze_detection_errors_r62.py) and [focused checks](../../../../tests/test_detection_errors_r62.py) are reproducible with Python's standard library and the existing repository helpers.

| File | Purpose |
| --- | --- |
| `summary.json` | Source identities, six model/reference rows, availability groups, per-image counts and cache limits |
| `cases.json.gz` | 530 extra-prediction or missed-reference records across the six comparisons |
| `results.md` | Compact generated tables and interpretation limits |
| `notebook-execution.json` | Completed four-cell execution, source identities, environment and independent package verification |
| `executed-notebook.ipynb` | Original notebook source with all four executed cells and captured outputs |
| `notebook-cell-1.txt` through `notebook-cell-4.txt` | Captured setup, analysis and verification output |
| `baomai-detection-audit-r62-results.zip` | Verified notebook result containing both analyses and its member manifest |
| `manifest.json` | Byte counts and hashes of the saved evidence |

The archive SHA-256 is `ce8df4bcf223b6bee49fa67e829a443acb572e4a6aad75fa0f92657a16cc2601` (972,658 bytes). Its inventory and all member hashes are verified. Reconstruction reproduces the saved suppression decisions, associations, predictions, reference cases and 240 per-image score tables exactly. The six original/proposed aggregate rows also agree exactly.

For the retained method on reviewed labels, the result is 654 references, 636 matches, 18 misses and 51 extras. Extras partition into five duplicates, 36 other outputs with maximum same-class reference IoU from 0.10 to below 0.50, and ten below 0.10. The groups do not identify physical background or prove annotation error.

Every miss has an incomplete reference corner pair: 13 of 54 one-corner fronts and five of 17 zero-corner fronts. All 583 complete-corner fronts are detected. Each review record preserves exact method/reference/image identity, zero-based indices, one-based display positions, and the relevant prediction/reference geometry.

The native input has already passed confidence 0.25 and native suppression. A missing eligible cached box is not proof that the network produced no response. The archive cannot recover candidates discarded earlier.

Run from a source checkout:

```sh
python -B scripts/analyze_detection_errors_r62.py --output /tmp/cassette-detection-errors
python -B -m unittest discover -s tests -p test_detection_errors_r62.py
```

All six focused checks pass. They cover overlap boundaries, duplicate accounting, empty images, unavailable corners, distinct cache stages and competing reference assignments.

## Completed notebook execution

All four actual code cells of the [CPU notebook](../../../../notebooks/baomai_detection_audit_r62.ipynb) completed unchanged and sequentially in one IPython namespace. Setup started with an absent source directory and fetched published commit `f41ce86d49aec0df7acba0389fd0624e3325fd35`. The source checkout still had exactly that commit and a clean working tree after the final cell.

The executed notebook's input SHA-256 is `0c228df62e29594363bcf769b321b412e25f4b0a507555900bfee8b7e4ff5d16`. The execution used Python 3.12.14 and IPython 9.18.1 on Linux. It reproduced this audit, the fixed [reflected-box probe](../reflected-anchor/README.md), all saved outputs and the final result package without inference or training.

The result ZIP contains 11 members and is 701,582 bytes, with SHA-256 `67ebe6cae8cfa51e738d8b671bd4549b9794cfa163a80cb24abab98145800dac`. A separate verification after notebook completion checked the exact member set, CRCs, every member's byte count and SHA-256, and the source commit and driver hashes recorded in its manifest. The [execution receipt](notebook-execution.json) records every cell outcome, source hash, elapsed time and log identity.

This was local IPython execution, not a hosted Google Colab session. The non-Colab file-link fallback completed; the Colab browser-download interface was not exercised. Rerunning the notebook is optional because the CPU analysis is already complete.
