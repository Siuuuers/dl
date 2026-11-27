# Corner-average evidence

These files record the completed equal-coordinate-average analysis described in [the experiment guide](../../corner-average-experiment.md) and Section 7.9 of the report. Detection boxes and output identities are preserved. No model was trained or run for this calculation.

| File | Contents |
| --- | --- |
| `summary.json` | Fixed procedure, input identity, association counts, all four reference comparisons, pair gains/losses, and automatic-output groups |
| `averaged-predictions.json.gz` | All 723 original detection records with the eligible point coordinates averaged |
| `associations.json.gz` | Every original record's association or fallback reason, overlap, and candidate multiplicity |
| `case-changes.json.gz` | Every target's original, reflected, and averaged detection/coordinate state |
| `independent-check.json` | Separate implementation's exact-record comparison, threshold changes, and reference-identity audit |
| `independent-verification.md` | Independent method, findings, and reproduction command |
| `input-provenance.json` | Immutable returned archive and analysis source hashes |
| `manifest.json` | Byte lengths and SHA-256 values of the evidence members |

The source is [`run_corner_average_r58.py`](../../../../scripts/run_corner_average_r58.py). The original predictions are taken from the exact returned archive at commit `dfe5392109433fcee51f41716710001bb123a3d5`, path `results/a100/baomai-flip-inference-r56-a100-001/baomai-flip-inference-r56-results.zip`. Its size is 2,931,102 bytes and SHA-256 is `88689ce9f3d03178683ea98ce98d50f5ddc4e07ba1fb24262e7820e9488bd114`.

The [four-cell CPU notebook](../../../../notebooks/baomai_corner_average_r58.ipynb) reproduces the experiment from a published immutable source commit and generates a separately verified result ZIP. Its source receipt lists the script hashes. Verification checks the actual compressed-member hashes, then compares decompressed data with the recomputed outputs, so harmless gzip-header differences between Python versions do not invalidate the experiment.

The development set has been inspected repeatedly. Improved counts here establish the behavior of this fixed rule on these photographs; a later capture group is needed to assess transfer. Original scores preserve the accepted-output comparison groups and do not establish confidence calibration for the averaged points.
