# Detection error audit evidence

This completed CPU analysis reuses the frozen annotation-reference archive. It changes no model, prediction, annotation, threshold or matching rule.

The [interpretation](../../detection-errors.md) connects the results to the next model experiment. The [analysis source](../../../../scripts/analyze_detection_errors_r62.py) and [focused checks](../../../../tests/test_detection_errors_r62.py) are reproducible with Python's standard library and the existing repository helpers.

| File | Purpose |
| --- | --- |
| `summary.json` | Source identities, six model/reference rows, availability groups, per-image counts and cache limits |
| `cases.json.gz` | 530 extra-prediction or missed-reference records across the six comparisons |
| `results.md` | Compact generated tables and interpretation limits |
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

All six focused checks pass. They cover overlap boundaries, duplicate accounting, empty images, unavailable corners, distinct cache stages and competing reference assignments. The [CPU notebook](../../../../notebooks/baomai_detection_audit_r62.ipynb) also runs the one fixed [reflected-box probe](../reflected-anchor/README.md) and packages both analyses. Its published execution receipt is stored alongside this evidence after the four unchanged code cells run.
